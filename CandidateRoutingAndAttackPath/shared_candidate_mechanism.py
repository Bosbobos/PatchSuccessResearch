from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .candidate_reserve import (
    _cache_lookup,
    _candidate_frames,
    _evaluate_batch,
    _matched_random,
)
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
from .mechanism_followup import _head_branches


@dataclass(slots=True)
class SharedCandidateMechanismConfig:
    output_dir: str = str(FOLLOWUP_DIR)
    device: str = "cpu"
    require_device: bool = False
    examples_per_group: int = 100
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
    seed: int = 307
    max_output_gb: float = 1.0
    method_version: int = 1


def _pairwise_cosine(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=np.float64)
    if len(matrix) < 2:
        return np.nan
    norms = np.linalg.norm(matrix, axis=1)
    valid = norms > 1e-12
    matrix = matrix[valid]
    norms = norms[valid]
    if len(matrix) < 2:
        return np.nan
    normalized = matrix / norms[:, None]
    cosine = normalized @ normalized.T
    upper = cosine[np.triu_indices(len(matrix), k=1)]
    return float(upper.mean()) if len(upper) else np.nan


def _svd_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if not matrix.size:
        return (
            np.zeros((matrix.shape[0], 0), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, matrix.shape[1]), dtype=np.float32),
        )
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    return u.astype(np.float32), singular.astype(np.float32), vh.astype(np.float32)


def _rank_reconstruction(
    u: np.ndarray,
    singular: np.ndarray,
    vh: np.ndarray,
    rank: int,
) -> np.ndarray:
    use = min(max(int(rank), 0), len(singular))
    if use == 0:
        return np.zeros((u.shape[0], vh.shape[1]), dtype=np.float32)
    return ((u[:, :use] * singular[:use]) @ vh[:use]).astype(np.float32)


def _effective_rank(singular: np.ndarray) -> float:
    energy = np.square(np.asarray(singular, dtype=np.float64))
    total = float(energy.sum())
    if total <= 1e-12:
        return 0.0
    probability = energy / total
    probability = probability[probability > 0]
    return float(np.exp(-(probability * np.log(probability)).sum()))


def _energy_matched_mask(
    matrix: np.ndarray,
    target_energy: float,
    *,
    order: np.ndarray,
) -> np.ndarray:
    """Restore actual clean->patch deltas until the requested squared energy is reached."""

    flat = np.asarray(matrix, dtype=np.float32).reshape(-1)
    output = np.zeros_like(flat)
    remaining = max(float(target_energy), 0.0)
    for index in np.asarray(order, dtype=int):
        value = float(flat[index])
        energy = value * value
        if energy <= 0:
            continue
        if energy <= remaining + 1e-12:
            output[index] = value
            remaining -= energy
        else:
            output[index] = value * float(np.sqrt(max(remaining, 0.0) / energy))
            remaining = 0.0
        if remaining <= 1e-10:
            break
    return output.reshape(matrix.shape)


def _extract_windows(delta, selection: pd.DataFrame, radius: int) -> np.ndarray:
    import torch.nn.functional as functional

    width = 2 * int(radius) + 1
    padded = functional.pad(delta.unsqueeze(0), (radius, radius, radius, radius))[0]
    rows = []
    for item in selection.itertuples(index=False):
        y, x = int(item.y_index), int(item.x_index)
        rows.append(
            padded[:, y:y + width, x:x + width].detach().float().cpu().numpy().reshape(-1)
        )
    channels = int(delta.shape[0])
    return np.asarray(rows, dtype=np.float32).reshape(len(rows), channels * width * width)


def _translated_layout(
    selection: pd.DataFrame,
    height: int,
    width: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Translate a candidate layout without changing its level, size, or relative geometry."""

    if selection.empty:
        return selection.copy()
    ys = selection.y_index.astype(int).to_numpy()
    xs = selection.x_index.astype(int).to_numpy()
    dy_values = np.arange(-int(ys.min()), int(height - 1 - ys.max()) + 1)
    dx_values = np.arange(-int(xs.min()), int(width - 1 - xs.max()) + 1)
    shifts = [(int(dy), int(dx)) for dy in dy_values for dx in dx_values if dy or dx]
    if not shifts:
        return selection.copy()
    # Prefer a non-overlapping placement, while retaining a deterministic fallback.
    target = {(int(y), int(x)) for y, x in zip(ys, xs, strict=True)}
    order = rng.permutation(len(shifts))
    chosen = shifts[int(order[0])]
    for index in order:
        candidate = shifts[int(index)]
        shifted = {(int(y + candidate[0]), int(x + candidate[1])) for y, x in zip(ys, xs, strict=True)}
        if target.isdisjoint(shifted):
            chosen = candidate
            break
    output = selection.copy()
    output["y_index"] = ys + chosen[0]
    output["x_index"] = xs + chosen[1]
    return output


def _scatter_windows(component: np.ndarray, selection: pd.DataFrame, shape, radius: int):
    import torch

    channels, height, width = (int(value) for value in shape)
    window = 2 * int(radius) + 1
    values = torch.as_tensor(component, dtype=torch.float32).reshape(-1, channels, window, window)
    accumulated = torch.zeros((channels, height, width), dtype=torch.float32)
    counts = torch.zeros((1, height, width), dtype=torch.float32)
    for value, item in zip(values, selection.itertuples(index=False), strict=True):
        y, x = int(item.y_index), int(item.x_index)
        y1, y2 = max(0, y - radius), min(height, y + radius + 1)
        x1, x2 = max(0, x - radius), min(width, x + radius + 1)
        wy1, wx1 = y1 - (y - radius), x1 - (x - radius)
        wy2, wx2 = wy1 + (y2 - y1), wx1 + (x2 - x1)
        accumulated[:, y1:y2, x1:x2] += value[:, wy1:wy2, wx1:wx2]
        counts[:, y1:y2, x1:x2] += 1
    return accumulated / counts.clamp_min(1.0)


def _spectral_row(
    example_id: str,
    analysis_group: str,
    set_kind: str,
    level: int,
    matrix: np.ndarray,
) -> dict:
    _u, singular, _vh = _svd_decomposition(matrix)
    energy = np.square(singular.astype(np.float64))
    total = max(float(energy.sum()), 1e-12)
    return {
        "example_id": example_id,
        "analysis_group": analysis_group,
        "set_kind": set_kind,
        "level_index": int(level),
        "n_candidates": int(len(matrix)),
        "feature_dimension": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "total_delta_energy": float(total),
        "rank1_energy_fraction": float(energy[:1].sum() / total),
        "rank2_energy_fraction": float(energy[:2].sum() / total),
        "rank4_energy_fraction": float(energy[:4].sum() / total),
        "effective_rank": _effective_rank(singular),
        "mean_pairwise_cosine": _pairwise_cosine(matrix),
    }


def _build_components(
    clean_inputs,
    patched_inputs,
    frame: dict,
    config: SharedCandidateMechanismConfig,
    rng: np.random.Generator,
):
    import torch

    target_set = frame["target_set"]
    random_set = _matched_random(target_set, frame["random_pool"])
    condition_names = [f"rank{rank}_common" for rank in config.ranks]
    condition_names += ["topabs_energy_rank1"]
    condition_names += [
        f"random_energy_rank1_{index}" for index in range(int(config.random_energy_controls))
    ]
    condition_names += ["full_target_windows"]
    maps = {
        name: [torch.zeros_like(item[0], dtype=torch.float32, device="cpu") for item in clean_inputs]
        for name in condition_names
    }
    spectral_rows = []
    energy = {name: 0.0 for name in condition_names}
    total_energy = 0.0
    for level in range(len(clean_inputs)):
        target_level = target_set[target_set.level_index.astype(int) == int(level)].reset_index(drop=True)
        random_level = random_set[random_set.level_index.astype(int) == int(level)].reset_index(drop=True)
        delta = (patched_inputs[level][0] - clean_inputs[level][0]).detach()
        if len(target_level):
            matrix = _extract_windows(delta, target_level, config.window_radius)
            u, singular, vh = _svd_decomposition(matrix)
            singular_energy = np.square(singular.astype(np.float64))
            rank1_energy = float(singular_energy[:1].sum())
            matrix_energy = float(np.square(matrix.astype(np.float64)).sum())
            total_energy += matrix_energy
            spectral_rows.append(("target", level, matrix))
            translated = _translated_layout(
                target_level, int(delta.shape[1]), int(delta.shape[2]), rng
            )
            spectral_rows.append(("translated_layout", level, _extract_windows(
                delta, translated, config.window_radius
            )))
            components = {
                f"rank{rank}_common": _rank_reconstruction(u, singular, vh, rank)
                for rank in config.ranks
            }
            flat_order = np.argsort(-np.abs(matrix.reshape(-1)))
            components["topabs_energy_rank1"] = _energy_matched_mask(
                matrix, rank1_energy, order=flat_order
            )
            for control in range(int(config.random_energy_controls)):
                components[f"random_energy_rank1_{control}"] = _energy_matched_mask(
                    matrix, rank1_energy, order=rng.permutation(matrix.size)
                )
            components["full_target_windows"] = matrix
            for name, component in components.items():
                maps[name][level] = _scatter_windows(
                    component, target_level, delta.shape, config.window_radius
                )
                energy[name] += float(np.square(component.astype(np.float64)).sum())
        if len(random_level):
            spectral_rows.append(("matched_random", level, _extract_windows(
                delta, random_level, config.window_radius
            )))
    return maps, spectral_rows, {
        name: float(value / max(total_energy, 1e-12)) for name, value in energy.items()
    }


def _batched_interventions(clean_inputs, patched_inputs, maps: dict, direction: str):
    import torch

    names = ["none", *maps.keys()]
    levels = []
    for level in range(len(clean_inputs)):
        variants = []
        for name in names:
            if name == "none":
                value = patched_inputs[level] if direction == "repair_patched" else clean_inputs[level]
            else:
                component = maps[name][level].to(
                    device=clean_inputs[level].device, dtype=clean_inputs[level].dtype
                ).unsqueeze(0)
                value = (
                    patched_inputs[level] - component
                    if direction == "repair_patched"
                    else clean_inputs[level] + component
                )
            variants.append(value)
        levels.append(torch.cat(variants, dim=0))
    return names, levels


def _summarize_interventions(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    grouped = rows.groupby(["analysis_group", "direction", "condition"], sort=False)
    for (group, direction, condition), frame in grouped:
        baseline_hidden = frame.baseline_hidden.astype(bool)
        records.append({
            "analysis_group": group,
            "direction": direction,
            "condition": condition,
            "n": int(frame.example_id.nunique()),
            "mean_component_energy_fraction": float(frame.component_energy_fraction.mean()),
            "target_hidden_rate": float(frame.target_hidden.mean()),
            "mean_post_target_conf": float(frame.post_target_conf.mean()),
            "mean_post_target_iou": float(frame.post_target_iou.mean()),
            "recovery_rate": float(frame.loc[baseline_hidden, "target_detected"].mean())
            if baseline_hidden.any() else np.nan,
            "reproduced_hiding_rate": float(frame.loc[~baseline_hidden, "target_hidden"].mean())
            if (~baseline_hidden).any() else np.nan,
        })
    overall = []
    for (direction, condition), frame in rows.groupby(["direction", "condition"], sort=False):
        baseline_hidden = frame.baseline_hidden.astype(bool)
        overall.append({
            "analysis_group": "all",
            "direction": direction,
            "condition": condition,
            "n": int(frame.example_id.nunique()),
            "mean_component_energy_fraction": float(frame.component_energy_fraction.mean()),
            "target_hidden_rate": float(frame.target_hidden.mean()),
            "mean_post_target_conf": float(frame.post_target_conf.mean()),
            "mean_post_target_iou": float(frame.post_target_iou.mean()),
            "recovery_rate": float(frame.loc[baseline_hidden, "target_detected"].mean())
            if baseline_hidden.any() else np.nan,
            "reproduced_hiding_rate": float(frame.loc[~baseline_hidden, "target_hidden"].mean())
            if (~baseline_hidden).any() else np.nan,
        })
    return pd.concat([pd.DataFrame(records), pd.DataFrame(overall)], ignore_index=True)


def _summarize_spectrum(rows: pd.DataFrame) -> pd.DataFrame:
    # Rank concentration is tautologically 1 for a one-row matrix, so those
    # levels are preserved in the raw table but excluded from spectral claims.
    rows = rows[rows.n_candidates >= 2].copy()
    metrics = [
        "rank1_energy_fraction", "rank2_energy_fraction", "rank4_energy_fraction",
        "effective_rank", "mean_pairwise_cosine",
    ]
    records = []
    for (group, kind), frame in rows.groupby(["analysis_group", "set_kind"], sort=False):
        record = {
            "analysis_group": group,
            "set_kind": kind,
            "n_examples": int(frame.example_id.nunique()),
            "n_level_rows": int(len(frame)),
            "mean_candidates": float(frame.n_candidates.mean()),
        }
        for metric in metrics:
            record[f"mean_{metric}"] = float(frame[metric].mean())
            record[f"median_{metric}"] = float(frame[metric].median())
        records.append(record)
    for kind, frame in rows.groupby("set_kind", sort=False):
        record = {
            "analysis_group": "all",
            "set_kind": kind,
            "n_examples": int(frame.example_id.nunique()),
            "n_level_rows": int(len(frame)),
            "mean_candidates": float(frame.n_candidates.mean()),
        }
        for metric in metrics:
            record[f"mean_{metric}"] = float(frame[metric].mean())
            record[f"median_{metric}"] = float(frame[metric].median())
        records.append(record)
    return pd.DataFrame(records)


def run_shared_candidate_mechanism(
    config: SharedCandidateMechanismConfig | None = None,
) -> Path:
    config = config or SharedCandidateMechanismConfig()
    started = time.time()
    budget = StorageBudget(config.output_dir, config.max_output_gb)
    budget.check()
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
    intervention_rows: list[dict] = []
    spectral_rows: list[dict] = []
    progress = tqdm(
        selected.itertuples(index=False), total=len(selected),
        desc="shared candidate mechanism", unit="image",
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
        rng = np.random.default_rng(int(config.seed) + index * 1009)
        maps, spectrum, energy_fraction = _build_components(
            clean_inputs, patched_inputs, frames[example_id], config, rng
        )
        for set_kind, level, matrix in spectrum:
            spectral_rows.append(_spectral_row(
                example_id, str(row.analysis_group), set_kind, level, matrix
            ))
        import torch

        with torch.inference_mode():
            for direction in ("repair_patched", "transplant_clean"):
                names, intervention_inputs = _batched_interventions(
                    clean_inputs, patched_inputs, maps, direction
                )
                _box, _cls, raw = _head_branches(detect, intervention_inputs)
                results = _evaluate_batch(detect, raw, row, config)
                for condition, result in zip(names, results, strict=True):
                    result.update({
                        "example_id": example_id,
                        "analysis_group": str(row.analysis_group),
                        "direction": direction,
                        "condition": condition,
                        "component_energy_fraction": 0.0
                        if condition == "none" else energy_fraction[condition],
                        "clean_set_size": int(len(frames[example_id]["target_set"])),
                    })
                    intervention_rows.append(result)
        release_accelerator_memory()
    rows = pd.DataFrame(intervention_rows)
    baseline = rows[rows.condition == "none"][[
        "example_id", "direction", "target_hidden"
    ]].rename(columns={"target_hidden": "baseline_hidden"})
    rows = rows.merge(baseline, on=["example_id", "direction"], validate="many_to_one")
    spectrum_rows = pd.DataFrame(spectral_rows)
    intervention_summary = _summarize_interventions(rows)
    spectrum_summary = _summarize_spectrum(spectrum_rows)
    payload = {**asdict(config), "example_ids": selected.example_id.tolist()}
    run_dir = Path(config.output_dir) / f"shared_candidate_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(run_dir / "intervention_rows.csv", index=False)
    intervention_summary.to_csv(run_dir / "intervention_summary.csv", index=False)
    spectrum_rows.to_csv(run_dir / "spectrum_rows.csv", index=False)
    spectrum_summary.to_csv(run_dir / "spectrum_summary.csv", index=False)
    budget.check()
    elapsed = time.time() - started
    overall_spectrum = spectrum_summary[spectrum_summary.analysis_group == "all"]
    overall_intervention = intervention_summary[
        intervention_summary.analysis_group == "all"
    ]
    summary = {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "n_examples": int(rows.example_id.nunique()),
        "n_intervention_conditions": int(rows.condition.nunique()),
        "cache_path": str(cache_path),
        "overall_spectrum": overall_spectrum.to_dict("records"),
        "overall_interventions": overall_intervention.to_dict("records"),
        "config": asdict(config),
        "limitations": [
            "The SVD is fitted separately per image and is an oracle decomposition, not a universal direction.",
            "Five-by-five Detect-input windows overlap for nearby candidates; the experiment tests shared local support as well as channel alignment.",
            "The clean target set is limited to saved top-50 person candidates.",
            "The experiment uses one YOLO detector, one person patch, and balanced endpoint groups.",
        ],
    }
    write_summary(run_dir / "summary.json", summary)
    target_spectrum = overall_spectrum[overall_spectrum.set_kind == "target"]
    random_spectrum = overall_spectrum[overall_spectrum.set_kind == "translated_layout"]
    target_r1 = float(target_spectrum.mean_rank1_energy_fraction.iloc[0]) if len(target_spectrum) else np.nan
    random_r1 = float(random_spectrum.mean_rank1_energy_fraction.iloc[0]) if len(random_spectrum) else np.nan
    repair = overall_intervention[overall_intervention.direction == "repair_patched"].set_index("condition")
    transplant = overall_intervention[overall_intervention.direction == "transplant_clean"].set_index("condition")
    lines = [
        "# Shared candidate mechanism",
        "",
        f"- elapsed: {elapsed:.1f} s",
        f"- examples: {rows.example_id.nunique()}",
        f"- target rank-1 energy fraction: {target_r1:.3f}",
        f"- translated-layout rank-1 energy fraction: {random_r1:.3f}",
    ]
    for name in ("rank1_common", "topabs_energy_rank1", "full_target_windows"):
        if name in repair.index:
            lines.append(f"- repair `{name}` recovery: {repair.loc[name, 'recovery_rate']:.3f}")
        if name in transplant.index:
            lines.append(
                f"- transplant `{name}` reproduced hiding: "
                f"{transplant.loc[name, 'reproduced_hiding_rate']:.3f}"
            )
    lines += [
        "",
        "See `spectrum_summary.csv` and `intervention_summary.csv` for the compact result tables.",
    ]
    (run_dir / "analysis_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    refresh_shared_candidate_artifacts(run_dir)
    return run_dir


def refresh_shared_candidate_artifacts(run_dir: str | Path) -> Path:
    """Rebuild compact summaries from raw rows without rerunning model inference."""

    run_dir = Path(run_dir)
    spectrum_rows = pd.read_csv(run_dir / "spectrum_rows.csv")
    intervention_rows = pd.read_csv(run_dir / "intervention_rows.csv")
    spectrum_summary = _summarize_spectrum(spectrum_rows)
    intervention_summary = _summarize_interventions(intervention_rows)
    spectrum_summary.to_csv(run_dir / "spectrum_summary.csv", index=False)
    intervention_summary.to_csv(run_dir / "intervention_summary.csv", index=False)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    overall_spectrum = spectrum_summary[spectrum_summary.analysis_group == "all"]
    overall_intervention = intervention_summary[
        intervention_summary.analysis_group == "all"
    ]
    target = overall_spectrum[overall_spectrum.set_kind == "target"]
    translated = overall_spectrum[overall_spectrum.set_kind == "translated_layout"]
    target_r1 = float(target.mean_rank1_energy_fraction.iloc[0]) if len(target) else np.nan
    translated_r1 = (
        float(translated.mean_rank1_energy_fraction.iloc[0]) if len(translated) else np.nan
    )
    target_rows = spectrum_rows[
        spectrum_rows.set_kind.eq("target") & spectrum_rows.n_candidates.ge(2)
    ].copy()
    hidden = target_rows.analysis_group.str.startswith("hidden")
    visible = target_rows.analysis_group.str.startswith("visible")
    hidden_r1 = float(target_rows.loc[hidden, "rank1_energy_fraction"].mean())
    visible_r1 = float(target_rows.loc[visible, "rank1_energy_fraction"].mean())
    summary["spectrum_min_candidates"] = 2
    summary["overall_spectrum"] = overall_spectrum.to_dict("records")
    summary["overall_interventions"] = overall_intervention.to_dict("records")
    summary["key_results"] = {
        "target_rank1_energy_fraction": target_r1,
        "translated_rank1_energy_fraction": translated_r1,
        "hidden_target_rank1_energy_fraction": hidden_r1,
        "visible_target_rank1_energy_fraction": visible_r1,
    }
    write_summary(summary_path, summary)
    repair = overall_intervention[
        overall_intervention.direction == "repair_patched"
    ].set_index("condition")
    transplant = overall_intervention[
        overall_intervention.direction == "transplant_clean"
    ].set_index("condition")
    lines = [
        "# Shared candidate mechanism",
        "",
        f"- elapsed: {float(summary['elapsed_seconds']):.1f} s",
        f"- examples: {int(summary['n_examples'])}",
        "- spectral summaries exclude one-candidate levels, where rank-1 is trivially 1",
        f"- target rank-1 energy fraction: {target_r1:.3f}",
        f"- translated-layout rank-1 energy fraction: {translated_r1:.3f}",
        f"- hidden target rank-1 energy fraction: {hidden_r1:.3f}",
        f"- visible target rank-1 energy fraction: {visible_r1:.3f}",
    ]
    for name in (
        "rank1_common", "rank2_common", "rank4_common",
        "topabs_energy_rank1", "random_energy_rank1_0", "full_target_windows",
    ):
        if name in repair.index:
            lines.append(f"- repair `{name}` recovery: {repair.loc[name, 'recovery_rate']:.3f}")
        if name in transplant.index:
            lines.append(
                f"- transplant `{name}` reproduced hiding: "
                f"{transplant.loc[name, 'reproduced_hiding_rate']:.3f}"
            )
    lines += [
        "",
        "See `spectrum_summary.csv` and `intervention_summary.csv` for compact tables;",
        "raw per-example values remain in `spectrum_rows.csv` and `intervention_rows.csv`.",
    ]
    (run_dir / "analysis_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


if __name__ == "__main__":
    print(run_shared_candidate_mechanism())
