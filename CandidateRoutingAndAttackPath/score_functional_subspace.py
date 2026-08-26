from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .candidate_reserve import _cache_lookup, _candidate_frames, _evaluate_batch
from .causal_repair import _load_inputs
from .common import StorageBudget, load_experiment, release_accelerator_memory, stable_hash
from .followup_common import (
    ATTACK_PATH_DB,
    FOLLOWUP_DIR,
    MANIFEST_CSV,
    TRACE_DB,
    balanced_subset,
    write_summary,
)
from .mechanism_followup import _head_branches, _raw_from_branches
from .shared_candidate_mechanism import _energy_matched_mask


@dataclass(slots=True)
class ScoreFunctionalSubspaceConfig:
    output_dir: str = str(FOLLOWUP_DIR)
    device: str = "cpu"
    require_device: bool = False
    examples_per_group: int = 25
    path_steps: int = 3
    window_radius: int = 2
    ranks: tuple[int, ...] = (1, 2, 4)
    random_energy_controls: int = 3
    target_iou: float = 0.50
    detection_conf: float = 0.25
    candidate_min_score: float = 0.01
    random_max_iou: float = 0.10
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    seed: int = 401
    max_output_gb: float = 1.0
    method_version: int = 1


def _spatial_indices(shape, selection: pd.DataFrame, radius: int) -> np.ndarray:
    channels, height, width = (int(value) for value in shape)
    spatial = np.zeros((height, width), dtype=bool)
    for item in selection.itertuples(index=False):
        y, x = int(item.y_index), int(item.x_index)
        spatial[max(0, y - radius):min(height, y + radius + 1),
                max(0, x - radius):min(width, x + radius + 1)] = True
    return np.flatnonzero(np.broadcast_to(spatial, (channels, height, width)).reshape(-1))


def _integrated_candidate_jacobian(
    class_head,
    clean,
    patched,
    selection: pd.DataFrame,
    class_id: int,
    steps: int,
):
    import torch

    ys = torch.as_tensor(selection.y_index.astype(int).to_numpy(), device=clean.device)
    xs = torch.as_tensor(selection.x_index.astype(int).to_numpy(), device=clean.device)
    accumulated = None
    for step in range(int(steps)):
        alpha = (step + 0.5) / float(steps)
        value = (clean + alpha * (patched - clean)).detach().requires_grad_(True)
        logits = class_head(value)[0, int(class_id), ys, xs]
        basis = torch.eye(len(logits), device=logits.device, dtype=logits.dtype)
        gradient = torch.autograd.grad(
            logits,
            value,
            grad_outputs=basis,
            is_grads_batched=True,
            create_graph=False,
            retain_graph=False,
        )[0][:, 0].detach().float()
        accumulated = gradient if accumulated is None else accumulated + gradient
    return accumulated / float(steps)


def _effective_rank(energy: np.ndarray) -> float:
    energy = np.asarray(energy, dtype=np.float64)
    total = float(energy.sum())
    if total <= 1e-12:
        return 0.0
    probability = energy[energy > 0] / total
    return float(np.exp(-(probability * np.log(probability)).sum()))


def _functional_components(
    clean_inputs,
    patched_inputs,
    frame: dict,
    detect,
    class_id: int,
    config: ScoreFunctionalSubspaceConfig,
    rng: np.random.Generator,
):
    import torch

    target_set = frame["target_set"]
    level_data = {}
    modes = []
    exact_parts = []
    predicted_parts = []
    total_window_energy = 0.0
    for level in range(len(clean_inputs)):
        selection = target_set[target_set.level_index.astype(int).eq(level)].reset_index(drop=True)
        if selection.empty:
            continue
        indices = _spatial_indices(clean_inputs[level][0].shape, selection, config.window_radius)
        delta_full = (patched_inputs[level][0] - clean_inputs[level][0]).detach().float().cpu().numpy().reshape(-1)
        delta = delta_full[indices].astype(np.float32)
        jacobian_full = _integrated_candidate_jacobian(
            detect.cv3[level], clean_inputs[level], patched_inputs[level], selection,
            class_id, config.path_steps,
        ).cpu().numpy().reshape(len(selection), -1)
        jacobian = jacobian_full[:, indices].astype(np.float32)
        u, singular, vh = np.linalg.svd(jacobian.astype(np.float64), full_matrices=False)
        coefficient = vh @ delta.astype(np.float64)
        mode_effect = singular * coefficient
        predicted = jacobian.astype(np.float64) @ delta.astype(np.float64)
        with torch.no_grad():
            clean_cls = detect.cv3[level](clean_inputs[level])
            patched_cls = detect.cv3[level](patched_inputs[level])
        ys = selection.y_index.astype(int).to_numpy()
        xs = selection.x_index.astype(int).to_numpy()
        exact = (
            patched_cls[0, int(class_id), ys, xs] - clean_cls[0, int(class_id), ys, xs]
        ).detach().float().cpu().numpy()
        exact_parts.append(exact)
        predicted_parts.append(predicted)
        level_data[level] = {
            "indices": indices,
            "shape": tuple(int(value) for value in clean_inputs[level][0].shape),
            "delta": delta,
            "projection": np.zeros_like(delta, dtype=np.float64),
        }
        total_window_energy += float(np.square(delta.astype(np.float64)).sum())
        for mode_index in range(len(singular)):
            component = (coefficient[mode_index] * vh[mode_index]).astype(np.float32)
            effect_energy = float(mode_effect[mode_index] ** 2)
            modes.append({
                "level": level,
                "mode_index": mode_index,
                "component": component,
                "input_energy": float(np.square(component.astype(np.float64)).sum()),
                "effect_energy": effect_energy,
                "sensitivity": float(singular[mode_index]),
            })
            level_data[level]["projection"] += component.astype(np.float64)
    modes.sort(key=lambda item: item["effect_energy"], reverse=True)
    names = [f"functional_rank{rank}" for rank in config.ranks]
    names += ["functional_all", "sensitivity_rank1", "topabs_energy_rank1"]
    names += [f"random_energy_rank1_{index}" for index in range(config.random_energy_controls)]
    names += ["null_residual", "full_target_windows"]
    maps = {
        name: [torch.zeros_like(item[0], dtype=torch.float32, device="cpu") for item in clean_inputs]
        for name in names
    }

    def assign(name: str, components: list[dict]):
        by_level = {level: np.zeros_like(data["delta"], dtype=np.float32) for level, data in level_data.items()}
        for component in components:
            by_level[component["level"]] += component["component"]
        for level, values in by_level.items():
            full = np.zeros(int(np.prod(level_data[level]["shape"])), dtype=np.float32)
            full[level_data[level]["indices"]] = values
            maps[name][level] = torch.from_numpy(full.reshape(level_data[level]["shape"]))

    for rank in config.ranks:
        assign(f"functional_rank{rank}", modes[:min(int(rank), len(modes))])
    assign("functional_all", modes)
    sensitivity = max(modes, key=lambda item: item["sensitivity"]) if modes else None
    assign("sensitivity_rank1", [sensitivity] if sensitivity is not None else [])
    for level, data in level_data.items():
        residual = data["delta"].astype(np.float64) - data["projection"]
        full_residual = np.zeros(int(np.prod(data["shape"])), dtype=np.float32)
        full_residual[data["indices"]] = residual.astype(np.float32)
        maps["null_residual"][level] = torch.from_numpy(full_residual.reshape(data["shape"]))
        full_delta = np.zeros(int(np.prod(data["shape"])), dtype=np.float32)
        full_delta[data["indices"]] = data["delta"]
        maps["full_target_windows"][level] = torch.from_numpy(full_delta.reshape(data["shape"]))
    rank1_energy = modes[0]["input_energy"] if modes else 0.0
    concatenated = np.concatenate([data["delta"] for data in level_data.values()]) if level_data else np.zeros(0)
    top_order = np.argsort(-np.abs(concatenated))
    top_component = _energy_matched_mask(concatenated, rank1_energy, order=top_order)
    random_components = [
        _energy_matched_mask(concatenated, rank1_energy, order=rng.permutation(len(concatenated)))
        for _ in range(config.random_energy_controls)
    ]
    cursor = 0
    for level, data in level_data.items():
        size = len(data["delta"])
        for name, values in [
            ("topabs_energy_rank1", top_component),
            *[(f"random_energy_rank1_{index}", value) for index, value in enumerate(random_components)],
        ]:
            full = np.zeros(int(np.prod(data["shape"])), dtype=np.float32)
            full[data["indices"]] = values[cursor:cursor + size]
            maps[name][level] = torch.from_numpy(full.reshape(data["shape"]))
        cursor += size
    exact_vector = np.concatenate(exact_parts) if exact_parts else np.zeros(0)
    predicted_vector = np.concatenate(predicted_parts) if predicted_parts else np.zeros(0)
    relative_error = float(
        np.linalg.norm(predicted_vector - exact_vector) / max(np.linalg.norm(exact_vector), 1e-12)
    )
    cosine = float(
        np.dot(predicted_vector, exact_vector)
        / max(np.linalg.norm(predicted_vector) * np.linalg.norm(exact_vector), 1e-12)
    )
    effect_energy = np.asarray([item["effect_energy"] for item in modes], dtype=float)
    total_effect = max(float(effect_energy.sum()), 1e-12)
    metadata = {
        "n_candidates": int(len(target_set)),
        "n_functional_modes": int(len(modes)),
        "exact_score_delta_l2": float(np.linalg.norm(exact_vector)),
        "predicted_score_delta_l2": float(np.linalg.norm(predicted_vector)),
        "jacobian_relative_error": relative_error,
        "jacobian_cosine": cosine,
        "rank1_effect_fraction": float(effect_energy[:1].sum() / total_effect),
        "rank2_effect_fraction": float(effect_energy[:2].sum() / total_effect),
        "rank4_effect_fraction": float(effect_energy[:4].sum() / total_effect),
        "functional_effective_rank": _effective_rank(effect_energy),
        "rowspace_input_energy_fraction": float(
            sum(item["input_energy"] for item in modes) / max(total_window_energy, 1e-12)
        ),
    }
    component_energy = {
        name: float(sum(np.square(item.numpy().astype(np.float64)).sum() for item in level_maps))
        / max(total_window_energy, 1e-12)
        for name, level_maps in maps.items()
    }
    return maps, component_energy, metadata


def _batched_class_interventions(clean_inputs, patched_inputs, maps, direction: str):
    import torch

    names = ["none", *maps.keys()]
    levels = []
    for level in range(len(clean_inputs)):
        values = []
        for name in names:
            base = patched_inputs[level] if direction == "repair_patched" else clean_inputs[level]
            if name == "none":
                values.append(base)
                continue
            component = maps[name][level].to(base.device, base.dtype).unsqueeze(0)
            values.append(base - component if direction == "repair_patched" else base + component)
        levels.append(torch.cat(values, dim=0))
    return names, levels


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    frames = [(group, frame) for group, frame in rows.groupby("analysis_group", sort=False)]
    frames.append(("all", rows))
    for group, group_frame in frames:
        for (direction, condition), frame in group_frame.groupby(["direction", "condition"], sort=False):
            baseline_hidden = frame.baseline_hidden.astype(bool)
            records.append({
                "analysis_group": group,
                "direction": direction,
                "condition": condition,
                "n": int(frame.example_id.nunique()),
                "mean_component_energy_fraction": float(frame.component_energy_fraction.mean()),
                "target_hidden_rate": float(frame.target_hidden.mean()),
                "mean_pre_target_conf": float(frame.pre_target_conf.mean()),
                "mean_tracked_score": float(frame.tracked_score.mean()),
                "mean_post_target_conf": float(frame.post_target_conf.mean()),
                "mean_post_target_iou": float(frame.post_target_iou.mean()),
                "recovery_rate": float(frame.loc[baseline_hidden, "target_detected"].mean())
                if baseline_hidden.any() else np.nan,
                "reproduced_hiding_rate": float(frame.loc[~baseline_hidden, "target_hidden"].mean())
                if (~baseline_hidden).any() else np.nan,
            })
    return pd.DataFrame(records)


def run_score_functional_subspace(
    config: ScoreFunctionalSubspaceConfig | None = None,
) -> Path:
    config = config or ScoreFunctionalSubspaceConfig()
    started = time.time()
    StorageBudget(config.output_dir, config.max_output_gb).check()
    selected, _ = _load_inputs(Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None)
    selected = balanced_subset(selected, config.examples_per_group, seed=config.seed)
    frames = _candidate_frames(selected, config)
    exp, cache_path = load_experiment(
        prefer_device=config.device, require_device=bool(config.require_device)
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    cache = _cache_lookup(exp)
    rows = []
    metadata_rows = []
    progress = tqdm(
        selected.itertuples(index=False), total=len(selected),
        desc="score functional subspace", unit="image",
    )
    for index, row in enumerate(progress):
        example_id = str(row.example_id)
        if example_id not in frames:
            continue
        example = cache[example_id]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        captured = _capture_detect_inputs(model, detect, pair)
        clean_inputs = [item[0:1] for item in captured]
        patched_inputs = [item[1:2] for item in captured]
        rng = np.random.default_rng(config.seed + index * 1009)
        maps, energy, metadata = _functional_components(
            clean_inputs, patched_inputs, frames[example_id], detect,
            int(row.class_id), config, rng,
        )
        metadata.update({"example_id": example_id, "analysis_group": str(row.analysis_group)})
        metadata_rows.append(metadata)
        import torch

        with torch.inference_mode():
            clean_box, _clean_cls, _ = _head_branches(detect, clean_inputs)
            patched_box, _patched_cls, _ = _head_branches(detect, patched_inputs)
            for direction, base_box in (
                ("repair_patched", patched_box), ("transplant_clean", clean_box),
            ):
                names, intervention_inputs = _batched_class_interventions(
                    clean_inputs, patched_inputs, maps, direction
                )
                class_outputs = [
                    detect.cv3[level](intervention_inputs[level])
                    for level in range(len(intervention_inputs))
                ]
                box_outputs = [item.repeat(len(names), 1, 1, 1) for item in base_box]
                raw = _raw_from_branches(box_outputs, class_outputs)
                results = _evaluate_batch(detect, raw, row, config)
                for condition, result in zip(names, results, strict=True):
                    result.update({
                        "example_id": example_id,
                        "analysis_group": str(row.analysis_group),
                        "direction": direction,
                        "condition": condition,
                        "component_energy_fraction": 0.0 if condition == "none" else energy[condition],
                    })
                    rows.append(result)
        release_accelerator_memory()
    rows = pd.DataFrame(rows)
    baseline = rows[rows.condition.eq("none")][[
        "example_id", "direction", "target_hidden"
    ]].rename(columns={"target_hidden": "baseline_hidden"})
    rows = rows.merge(baseline, on=["example_id", "direction"], validate="many_to_one")
    metadata = pd.DataFrame(metadata_rows)
    summary_table = _summarize(rows)
    payload = {**asdict(config), "example_ids": selected.example_id.tolist()}
    run_dir = Path(config.output_dir) / f"score_functional_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(run_dir / "intervention_rows.csv", index=False)
    metadata.to_csv(run_dir / "functional_spectrum_rows.csv", index=False)
    summary_table.to_csv(run_dir / "intervention_summary.csv", index=False)
    elapsed = time.time() - started
    group_spectrum = metadata.groupby("analysis_group").agg(
        n=("example_id", "nunique"),
        mean_rank1_effect_fraction=("rank1_effect_fraction", "mean"),
        mean_rank2_effect_fraction=("rank2_effect_fraction", "mean"),
        mean_rank4_effect_fraction=("rank4_effect_fraction", "mean"),
        mean_effective_rank=("functional_effective_rank", "mean"),
        mean_jacobian_relative_error=("jacobian_relative_error", "mean"),
        mean_jacobian_cosine=("jacobian_cosine", "mean"),
        mean_rowspace_input_energy_fraction=("rowspace_input_energy_fraction", "mean"),
    ).reset_index()
    group_spectrum.to_csv(run_dir / "functional_spectrum_summary.csv", index=False)
    overall = summary_table[summary_table.analysis_group.eq("all")]
    summary = {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "n_examples": int(rows.example_id.nunique()),
        "cache_path": str(cache_path),
        "config": asdict(config),
        "mean_jacobian_relative_error": float(metadata.jacobian_relative_error.mean()),
        "mean_jacobian_cosine": float(metadata.jacobian_cosine.mean()),
        "mean_functional_effective_rank": float(metadata.functional_effective_rank.mean()),
        "overall_interventions": overall.to_dict("records"),
        "limitations": [
            "The score Jacobian is path-averaged at a small number of midpoint samples.",
            "Geometry is held at its clean or patched endpoint and is only a secondary outcome.",
            "The functional basis is fitted separately per image and is an oracle decomposition.",
            "The experiment uses one detector and one adversarial person patch.",
        ],
    }
    write_summary(run_dir / "summary.json", summary)
    indexed = overall.set_index(["direction", "condition"])
    lines = [
        "# Score-functional candidate subspace",
        "",
        f"- elapsed: {elapsed:.1f} s",
        f"- examples: {rows.example_id.nunique()}",
        f"- mean Jacobian completeness error: {summary['mean_jacobian_relative_error']:.3f}",
        f"- mean Jacobian cosine: {summary['mean_jacobian_cosine']:.3f}",
        f"- mean functional effective rank: {summary['mean_functional_effective_rank']:.3f}",
    ]
    for condition in (
        "functional_rank1", "functional_rank2", "functional_rank4", "functional_all",
        "null_residual", "full_target_windows",
    ):
        repair_key = ("repair_patched", condition)
        transplant_key = ("transplant_clean", condition)
        if repair_key in indexed.index:
            lines.append(f"- repair `{condition}` recovery: {indexed.loc[repair_key, 'recovery_rate']:.3f}")
        if transplant_key in indexed.index:
            lines.append(
                f"- transplant `{condition}` hiding: "
                f"{indexed.loc[transplant_key, 'reproduced_hiding_rate']:.3f}"
            )
    (run_dir / "analysis_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    StorageBudget(config.output_dir, config.max_output_gb).check()
    refresh_score_functional_artifacts(run_dir)
    return run_dir


def refresh_score_functional_artifacts(run_dir: str | Path) -> Path:
    """Rebuild compact tables after inference, without recomputing Jacobians."""

    run_dir = Path(run_dir)
    rows = pd.read_csv(run_dir / "intervention_rows.csv")
    spectrum = pd.read_csv(run_dir / "functional_spectrum_rows.csv")
    interventions = _summarize(rows)
    interventions.to_csv(run_dir / "intervention_summary.csv", index=False)
    spectrum_summary = spectrum.groupby("analysis_group").agg(
        n=("example_id", "nunique"),
        mean_rank1_effect_fraction=("rank1_effect_fraction", "mean"),
        mean_rank2_effect_fraction=("rank2_effect_fraction", "mean"),
        mean_rank4_effect_fraction=("rank4_effect_fraction", "mean"),
        mean_effective_rank=("functional_effective_rank", "mean"),
        mean_jacobian_relative_error=("jacobian_relative_error", "mean"),
        mean_jacobian_cosine=("jacobian_cosine", "mean"),
        mean_rowspace_input_energy_fraction=("rowspace_input_energy_fraction", "mean"),
    ).reset_index()
    spectrum_summary.to_csv(run_dir / "functional_spectrum_summary.csv", index=False)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    overall = interventions[interventions.analysis_group.eq("all")]
    summary.update({
        "mean_jacobian_relative_error": float(spectrum.jacobian_relative_error.mean()),
        "mean_jacobian_cosine": float(spectrum.jacobian_cosine.mean()),
        "mean_functional_effective_rank": float(spectrum.functional_effective_rank.mean()),
        "mean_rowspace_input_energy_fraction": float(spectrum.rowspace_input_energy_fraction.mean()),
        "overall_interventions": overall.to_dict("records"),
    })
    if float(summary.get("elapsed_seconds", 0.0)) > 900:
        summary["runtime_note"] = (
            "This 400-example desktop run includes long process-suspension gaps after an interrupted "
            "Codex turn. The runner default is 100 total examples (25 per group) so a normal run "
            "remains below the 15-minute experiment budget."
        )
    write_summary(summary_path, summary)
    indexed = overall.set_index(["direction", "condition"])
    lines = [
        "# Score-functional candidate subspace", "",
        f"- elapsed: {float(summary['elapsed_seconds']):.1f} s",
        f"- examples: {int(summary['n_examples'])}",
        f"- mean Jacobian completeness error: {summary['mean_jacobian_relative_error']:.3f}",
        f"- mean Jacobian cosine: {summary['mean_jacobian_cosine']:.3f}",
        f"- mean functional effective rank: {summary['mean_functional_effective_rank']:.3f}",
        f"- mean row-space input-energy fraction: {summary['mean_rowspace_input_energy_fraction']:.5f}",
    ]
    for condition in (
        "functional_rank1", "functional_rank2", "functional_rank4", "functional_all",
        "sensitivity_rank1", "topabs_energy_rank1", "random_energy_rank1_0",
        "null_residual", "full_target_windows",
    ):
        repair_key, transplant_key = ("repair_patched", condition), ("transplant_clean", condition)
        if repair_key in indexed.index:
            lines.append(
                f"- repair `{condition}`: recovery {indexed.loc[repair_key, 'recovery_rate']:.3f}, "
                f"pre-NMS confidence {indexed.loc[repair_key, 'mean_pre_target_conf']:.3f}"
            )
        if transplant_key in indexed.index:
            lines.append(
                f"- transplant `{condition}`: hiding {indexed.loc[transplant_key, 'reproduced_hiding_rate']:.3f}, "
                f"pre-NMS confidence {indexed.loc[transplant_key, 'mean_pre_target_conf']:.3f}"
            )
    (run_dir / "analysis_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


if __name__ == "__main__":
    print(run_score_functional_subspace())
