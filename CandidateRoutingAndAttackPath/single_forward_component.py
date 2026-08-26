from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .candidate_reserve import _cache_lookup, _evaluate_batch
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
from .full_success_closure import (
    _candidate_closure,
    _local_indices,
    _rowspace_projection,
)
from .mechanism_followup import _head_branches
from .score_functional_subspace import _integrated_candidate_jacobian


@dataclass(slots=True)
class SingleForwardComponentConfig:
    output_dir: str = str(FOLLOWUP_DIR)
    device: str = "mps"
    require_device: bool = True
    examples_per_group: int = 50
    reference_examples_per_group: int = 25
    window_radius: int = 2
    top_negative_k: tuple[int, ...] = (50, 100, 250, 500)
    consensus_fraction: float = 0.75
    target_iou: float = 0.50
    detection_conf: float = 0.25
    candidate_min_score: float = 0.01
    max_candidate_routes: int = 20
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    nms_max_time_img: float = 1.0
    seed: int = 733
    max_output_gb: float = 1.0
    method_version: int = 1


def _split_reference_evaluation(selected: pd.DataFrame, config):
    reference = []
    evaluation = []
    for _group, frame in selected.groupby("analysis_group", sort=False):
        shuffled = frame.sample(frac=1.0, random_state=int(config.seed) + 19)
        count = min(int(config.reference_examples_per_group), len(shuffled) - 1)
        reference.append(shuffled.iloc[:count])
        evaluation.append(shuffled.iloc[count:])
    return (
        pd.concat(reference, ignore_index=True),
        pd.concat(evaluation, ignore_index=True),
    )


def _collect_clean_channel_moments(exp, model, detect, cache, rows):
    import torch

    sums = None
    squares = None
    counts = None
    for row in tqdm(
        rows.itertuples(index=False),
        total=len(rows),
        desc="clean activation reference",
        unit="image",
    ):
        example = cache[str(row.example_id)]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        captured = _capture_detect_inputs(model, detect, pair[0:1])
        if sums is None:
            sums = [
                torch.zeros(item.shape[1], dtype=torch.float64) for item in captured
            ]
            squares = [torch.zeros_like(item) for item in sums]
            counts = [0 for _ in captured]
        for level, item in enumerate(captured):
            value = item[0].detach().cpu().double()
            sums[level] += value.sum(dim=(1, 2))
            squares[level] += value.square().sum(dim=(1, 2))
            counts[level] += int(value.shape[1] * value.shape[2])
        del captured
        release_accelerator_memory()
    means = [(total / count).float() for total, count in zip(sums, counts, strict=True)]
    stds = [
        (square / count - mean.double().square()).clamp_min(1e-10).sqrt().float()
        for square, mean, count in zip(squares, means, counts, strict=True)
    ]
    return means, stds


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else np.nan


def _single_forward_maps(
    detect,
    endpoint_inputs,
    selection: pd.DataFrame,
    class_id: int,
    means,
    config: SingleForwardComponentConfig,
    oracle_delta_inputs=None,
):
    import torch

    names = [
        "negative_sum_modes",
        "negative_consensus_modes",
        "full_score_rowspace",
        *[f"top_negative_{int(k)}" for k in config.top_negative_k],
    ]
    maps = {
        name: [
            torch.zeros_like(item[0], dtype=torch.float32, device="cpu")
            for item in endpoint_inputs
        ]
        for name in names
    }
    metadata = {
        "candidate_routes": int(len(selection)),
        "negative_sum_modes": 0,
        "negative_consensus_modes": 0,
        "estimated_energy": {name: 0.0 for name in names},
        "oracle_energy": 0.0,
        "cosine": {name: [] for name in names},
    }
    top_records = {name: [] for name in names if name.startswith("top_negative_")}
    predicted_gains = {name: 0.0 for name in top_records}
    candidate_gains = {name: [] for name in top_records}
    total_available_negative_gain = 0.0
    total_residual_energy = 0.0
    for level, endpoint in enumerate(endpoint_inputs):
        subset = selection[
            selection.level_index.astype(int).eq(level)
        ].head(int(config.max_candidate_routes)).reset_index(drop=True)
        if subset.empty:
            continue
        indices = _local_indices(
            endpoint[0].shape, subset, radius=int(config.window_radius)
        )
        mean = means[level].to(endpoint.device, endpoint.dtype).reshape(1, -1, 1, 1)
        residual_full = (endpoint - mean).detach().float().cpu().numpy()[0].reshape(-1)
        residual = residual_full[indices].astype(np.float32)
        total_residual_energy += float(np.square(residual.astype(np.float64)).sum())
        jacobian_full = _integrated_candidate_jacobian(
            detect.cv3[level], endpoint, endpoint, subset, int(class_id), 1
        ).cpu().numpy().reshape(len(subset), -1)
        jacobian = jacobian_full[:, indices].astype(np.float32)
        _u, singular, vh = np.linalg.svd(
            jacobian.astype(np.float64), full_matrices=False
        )
        tolerance = (
            max(jacobian.shape)
            * np.finfo(np.float64).eps
            * max(float(singular.max()) if len(singular) else 0.0, 1.0)
        )
        basis = vh[singular > tolerance]
        coefficients = basis @ residual.astype(np.float64)
        components = coefficients[:, None] * basis
        effects = components @ jacobian.astype(np.float64).T
        sum_mask = effects.mean(axis=1) < 0
        consensus_mask = (
            (effects < 0).mean(axis=1) >= float(config.consensus_fraction)
        ) & sum_mask
        metadata["negative_sum_modes"] += int(sum_mask.sum())
        metadata["negative_consensus_modes"] += int(consensus_mask.sum())
        estimates = {
            "negative_sum_modes": components[sum_mask].sum(axis=0)
            if bool(sum_mask.any()) else np.zeros_like(residual),
            "negative_consensus_modes": components[consensus_mask].sum(axis=0)
            if bool(consensus_mask.any()) else np.zeros_like(residual),
            "full_score_rowspace": components.sum(axis=0)
            if len(components) else np.zeros_like(residual),
        }
        aggregate_contribution = residual.astype(np.float64) * jacobian.sum(axis=0)
        negative_order = np.flatnonzero(aggregate_contribution < 0)
        negative_order = negative_order[
            np.argsort(aggregate_contribution[negative_order])
        ]
        total_available_negative_gain += float(
            -aggregate_contribution[negative_order].sum()
        )
        for k in config.top_negative_k:
            name = f"top_negative_{int(k)}"
            chosen = negative_order[: min(int(k), len(negative_order))]
            values = np.zeros_like(residual)
            values[chosen] = residual[chosen]
            estimates[name] = values
            top_records[name].append(int(len(chosen)))
            predicted_gains[name] += float(
                -aggregate_contribution[chosen].sum()
            )
            candidate_gains[name].extend(
                (-(jacobian.astype(np.float64) @ values.astype(np.float64))).tolist()
            )
        oracle = None
        if oracle_delta_inputs is not None:
            oracle_full = oracle_delta_inputs[level][0].detach().float().cpu().numpy().reshape(-1)
            oracle, _rank = _rowspace_projection(
                jacobian, oracle_full[indices].astype(np.float32)
            )
            metadata["oracle_energy"] += float(
                np.square(oracle.astype(np.float64)).sum()
            )
        for name, values in estimates.items():
            full = np.zeros_like(residual_full, dtype=np.float32)
            full[indices] = values.astype(np.float32)
            maps[name][level] = torch.from_numpy(full.reshape(endpoint[0].shape))
            metadata["estimated_energy"][name] += float(
                np.square(values.astype(np.float64)).sum()
            )
            if oracle is not None:
                metadata["cosine"][name].append(_cosine(values, oracle))
    flat = {
        "candidate_routes": metadata["candidate_routes"],
        "negative_sum_mode_count": metadata["negative_sum_modes"],
        "negative_consensus_mode_count": metadata["negative_consensus_modes"],
        "oracle_score_component_energy": metadata["oracle_energy"],
    }
    for name in names:
        flat[f"{name}_residual_energy_fraction"] = (
            metadata["estimated_energy"][name] / max(total_residual_energy, 1e-12)
        )
        values = [value for value in metadata["cosine"][name] if np.isfinite(value)]
        flat[f"{name}_oracle_cosine"] = (
            float(np.mean(values)) if values else np.nan
        )
    for name, counts in top_records.items():
        flat[f"{name}_actual_k"] = int(sum(counts))
        flat[f"{name}_predicted_gain"] = float(predicted_gains[name])
        energy = float(metadata["estimated_energy"][name])
        gains = np.asarray(candidate_gains[name], dtype=np.float64)
        flat[f"{name}_gain_concentration"] = float(
            predicted_gains[name] / max(total_available_negative_gain, 1e-12)
        )
        flat[f"{name}_gain_per_l2"] = float(
            predicted_gains[name] / max(np.sqrt(energy), 1e-12)
        )
        flat[f"{name}_candidate_positive_fraction"] = (
            float((gains > 0).mean()) if len(gains) else np.nan
        )
        flat[f"{name}_candidate_gain_mean"] = (
            float(gains.mean()) if len(gains) else np.nan
        )
        flat[f"{name}_candidate_gain_std"] = (
            float(gains.std()) if len(gains) else np.nan
        )
        flat[f"{name}_candidate_gain_cv"] = (
            float(gains.std() / max(abs(float(gains.mean())), 1e-12))
            if len(gains) else np.nan
        )
        flat[f"{name}_candidate_gain_min"] = (
            float(gains.min()) if len(gains) else np.nan
        )
    flat["total_available_negative_gain"] = float(total_available_negative_gain)
    return maps, flat


def _fast_aggregate_top_negative_map(
    detect,
    endpoint_inputs,
    selection: pd.DataFrame,
    class_id: int,
    means,
    config: SingleForwardComponentConfig,
    k: int = 1000,
):
    """Exact top-negative map without per-candidate Jacobians or diagnostic SVD.

    The production intervention only uses the sum of candidate Jacobians:
    ``residual * jacobian.sum(axis=0)``. By linearity, this is exactly the
    gradient of the sum of candidate logits, so one scalar backward per
    populated FPN level replaces a batched VJP with one row per candidate.
    """

    import torch

    output = [
        torch.zeros_like(item[0], dtype=torch.float32, device="cpu")
        for item in endpoint_inputs
    ]
    total_available_negative_gain = 0.0
    predicted_gain = 0.0
    estimated_energy = 0.0
    actual_k = 0
    for level, endpoint in enumerate(endpoint_inputs):
        subset = selection[
            selection.level_index.astype(int).eq(level)
        ].head(int(config.max_candidate_routes)).reset_index(drop=True)
        if subset.empty:
            continue
        indices = _local_indices(
            endpoint[0].shape, subset, radius=int(config.window_radius)
        )
        mean = means[level].to(endpoint.device, endpoint.dtype).reshape(
            1, -1, 1, 1
        )
        residual_full = (
            endpoint - mean
        ).detach().float().cpu().numpy()[0].reshape(-1)
        residual = residual_full[indices].astype(np.float32)
        value = endpoint.detach().requires_grad_(True)
        ys = torch.as_tensor(
            subset.y_index.astype(int).to_numpy(), device=value.device
        )
        xs = torch.as_tensor(
            subset.x_index.astype(int).to_numpy(), device=value.device
        )
        logits = detect.cv3[level](value)[0, int(class_id), ys, xs]
        gradient_full = torch.autograd.grad(
            logits.sum(),
            value,
            create_graph=False,
            retain_graph=False,
        )[0][0].detach().float().cpu().numpy().reshape(-1)
        gradient = gradient_full[indices].astype(np.float32)
        contribution = (
            residual.astype(np.float64) * gradient.astype(np.float64)
        )
        negative_order = np.flatnonzero(contribution < 0)
        negative_order = negative_order[
            np.argsort(contribution[negative_order])
        ]
        total_available_negative_gain += float(
            -contribution[negative_order].sum()
        )
        chosen = negative_order[: min(int(k), len(negative_order))]
        values = np.zeros_like(residual)
        values[chosen] = residual[chosen]
        predicted_gain += float(-contribution[chosen].sum())
        estimated_energy += float(
            np.square(values.astype(np.float64)).sum()
        )
        actual_k += int(len(chosen))
        full = np.zeros_like(residual_full, dtype=np.float32)
        full[indices] = values
        output[level] = torch.from_numpy(full.reshape(endpoint[0].shape))
    metadata = {
        "actual_k": int(actual_k),
        "predicted_gain": float(predicted_gain),
        "gain_concentration": float(
            predicted_gain / max(total_available_negative_gain, 1e-12)
        ),
        "gain_per_l2": float(
            predicted_gain / max(np.sqrt(estimated_energy), 1e-12)
        ),
        "total_available_negative_gain": float(
            total_available_negative_gain
        ),
    }
    metadata["diffuse_negative_leverage"] = float(
        metadata["total_available_negative_gain"]
        * (1.0 - metadata["gain_concentration"])
        * metadata["gain_per_l2"]
    )
    return output, metadata


def _fast_aggregate_top_negative_maps(
    detect,
    endpoint_inputs,
    selection: pd.DataFrame,
    class_id: int,
    means,
    config: SingleForwardComponentConfig,
    fixed_ks: tuple[int, ...] = (1000,),
    coverage_fractions: tuple[float, ...] = (0.75, 0.90),
):
    """Compute fixed-budget and global gain-coverage maps with one backward.

    Fixed budgets preserve the historical behavior (up to ``k`` coordinates
    independently on every populated FPN level). Coverage budgets instead
    pool all negative coordinates across levels and retain the smallest set
    whose predicted gain reaches the requested fraction of the available
    negative mass.
    """

    import torch

    names = [
        *[f"fixed_k{int(k)}" for k in fixed_ks],
        *[
            f"coverage_{int(round(float(fraction) * 100))}"
            for fraction in coverage_fractions
        ],
    ]
    level_data = {}
    total_available_negative_gain = 0.0
    global_records = []
    for level, endpoint in enumerate(endpoint_inputs):
        subset = selection[
            selection.level_index.astype(int).eq(level)
        ].head(int(config.max_candidate_routes)).reset_index(drop=True)
        if subset.empty:
            continue
        indices = _local_indices(
            endpoint[0].shape, subset, radius=int(config.window_radius)
        )
        mean = means[level].to(endpoint.device, endpoint.dtype).reshape(
            1, -1, 1, 1
        )
        residual_full = (
            endpoint - mean
        ).detach().float().cpu().numpy()[0].reshape(-1)
        residual = residual_full[indices].astype(np.float32)
        value = endpoint.detach().requires_grad_(True)
        ys = torch.as_tensor(
            subset.y_index.astype(int).to_numpy(), device=value.device
        )
        xs = torch.as_tensor(
            subset.x_index.astype(int).to_numpy(), device=value.device
        )
        logits = detect.cv3[level](value)[0, int(class_id), ys, xs]
        gradient_full = torch.autograd.grad(
            logits.sum(),
            value,
            create_graph=False,
            retain_graph=False,
        )[0][0].detach().float().cpu().numpy().reshape(-1)
        contribution = (
            residual.astype(np.float64)
            * gradient_full[indices].astype(np.float64)
        )
        negative_order = np.flatnonzero(contribution < 0)
        negative_order = negative_order[
            np.argsort(contribution[negative_order])
        ]
        gains = -contribution[negative_order]
        total_available_negative_gain += float(gains.sum())
        global_records.extend(
            (float(gain), int(level), int(local_index))
            for gain, local_index in zip(gains, negative_order, strict=True)
        )
        level_data[level] = {
            "shape": tuple(int(value) for value in endpoint[0].shape),
            "indices": indices,
            "residual_full": residual_full,
            "residual": residual,
            "contribution": contribution,
            "negative_order": negative_order,
        }

    chosen_by_name: dict[str, dict[int, np.ndarray]] = {
        name: {} for name in names
    }
    for k in fixed_ks:
        name = f"fixed_k{int(k)}"
        for level, data in level_data.items():
            chosen_by_name[name][level] = data["negative_order"][
                : min(int(k), len(data["negative_order"]))
            ]
    ordered_global = sorted(global_records, key=lambda item: item[0], reverse=True)
    global_gains = np.asarray(
        [item[0] for item in ordered_global], dtype=np.float64
    )
    cumulative = np.cumsum(global_gains)
    for fraction in coverage_fractions:
        name = f"coverage_{int(round(float(fraction) * 100))}"
        if not len(ordered_global):
            continue
        target = float(fraction) * total_available_negative_gain
        count = int(np.searchsorted(cumulative, target, side="left")) + 1
        selected = ordered_global[: min(count, len(ordered_global))]
        for level in level_data:
            chosen_by_name[name][level] = np.asarray(
                [local for _gain, item_level, local in selected
                 if item_level == level],
                dtype=int,
            )

    maps = {}
    metadata = {}
    for name in names:
        output = [
            torch.zeros_like(item[0], dtype=torch.float32, device="cpu")
            for item in endpoint_inputs
        ]
        predicted_gain = 0.0
        estimated_energy = 0.0
        actual_k = 0
        for level, data in level_data.items():
            chosen = chosen_by_name[name].get(
                level, np.asarray([], dtype=int)
            )
            values = np.zeros_like(data["residual"])
            values[chosen] = data["residual"][chosen]
            predicted_gain += float(-data["contribution"][chosen].sum())
            estimated_energy += float(
                np.square(values.astype(np.float64)).sum()
            )
            actual_k += int(len(chosen))
            full = np.zeros_like(data["residual_full"], dtype=np.float32)
            full[data["indices"]] = values
            output[level] = torch.from_numpy(full.reshape(data["shape"]))
        concentration = float(
            predicted_gain / max(total_available_negative_gain, 1e-12)
        )
        gain_per_l2 = float(
            predicted_gain / max(np.sqrt(estimated_energy), 1e-12)
        )
        maps[name] = output
        metadata[name] = {
            "actual_k": int(actual_k),
            "predicted_gain": float(predicted_gain),
            "gain_concentration": concentration,
            "gain_per_l2": gain_per_l2,
            "total_available_negative_gain": float(
                total_available_negative_gain
            ),
            "diffuse_negative_leverage": float(
                total_available_negative_gain
                * (1.0 - concentration)
                * gain_per_l2
            ),
        }
    return maps, metadata


def _intervention_raw(detect, endpoint_inputs, maps):
    import torch

    names = ["observed", *maps]
    levels = []
    for level, endpoint in enumerate(endpoint_inputs):
        levels.append(torch.cat([
            endpoint,
            *[
                endpoint
                - maps[name][level].to(endpoint.device, endpoint.dtype).unsqueeze(0)
                for name in maps
            ],
        ], dim=0))
    with torch.inference_mode():
        _box, _cls, raw = _head_branches(detect, levels)
    return names, raw


def run_single_forward_component(
    config: SingleForwardComponentConfig | None = None,
) -> Path:
    config = config or SingleForwardComponentConfig()
    started = time.time()
    StorageBudget(config.output_dir, config.max_output_gb).check()
    selected, _ = _load_inputs(
        Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None
    )
    selected = balanced_subset(
        selected, int(config.examples_per_group), seed=int(config.seed)
    )
    reference, evaluation = _split_reference_evaluation(selected, config)
    exp, cache_path = load_experiment(
        prefer_device=config.device, require_device=bool(config.require_device)
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    cache = _cache_lookup(exp)
    means, stds = _collect_clean_channel_moments(
        exp, model, detect, cache, reference
    )
    records = []
    for row in tqdm(
        evaluation.itertuples(index=False),
        total=len(evaluation),
        desc="single-forward component",
        unit="image",
    ):
        example_id = str(row.example_id)
        example = cache[example_id]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        captured = _capture_detect_inputs(model, detect, pair)
        clean_inputs = [item[0:1] for item in captured]
        patched_inputs = [item[1:2] for item in captured]
        import torch

        with torch.inference_mode():
            clean_box, clean_cls, clean_raw = _head_branches(detect, clean_inputs)
            patched_box, patched_cls, patched_raw = _head_branches(detect, patched_inputs)
            selection = _candidate_closure(
                detect, clean_raw, patched_raw, row, config
            )
        oracle_delta = [
            patched - clean
            for clean, patched in zip(clean_inputs, patched_inputs, strict=True)
        ]
        for input_kind, endpoint in (
            ("clean", clean_inputs),
            ("patched", patched_inputs),
        ):
            maps, metadata = _single_forward_maps(
                detect,
                endpoint,
                selection,
                int(row.class_id),
                means,
                config,
                oracle_delta_inputs=oracle_delta if input_kind == "patched" else None,
            )
            names, raw = _intervention_raw(detect, endpoint, maps)
            with torch.inference_mode():
                evaluated = _evaluate_batch(detect, raw, row, config)
            baseline_hidden = int(evaluated[0]["target_hidden"])
            for condition, result in zip(names, evaluated, strict=True):
                result.update({
                    "example_id": example_id,
                    "analysis_group": str(row.analysis_group),
                    "input_kind": input_kind,
                    "condition": condition,
                    "source_hidden": baseline_hidden,
                    **metadata,
                })
                records.append(result)
        del captured
        release_accelerator_memory()
    rows = pd.DataFrame(records)
    summary_rows = rows.groupby(
        ["input_kind", "condition"], as_index=False
    ).agg(
        n=("example_id", "nunique"),
        target_detection_rate=("target_detected", "mean"),
        target_hidden_rate=("target_hidden", "mean"),
        mean_post_target_conf=("post_target_conf", "mean"),
        mean_oracle_cosine=("negative_consensus_modes_oracle_cosine", "mean"),
    )
    patched = rows[rows.input_kind.eq("patched")]
    source_hidden = patched.source_hidden.astype(bool)
    recovery = (
        patched[source_hidden]
        .groupby("condition", as_index=False)
        .agg(
            hidden_n=("example_id", "nunique"),
            recovery_rate=("target_detected", "mean"),
            mean_post_target_conf=("post_target_conf", "mean"),
        )
    )
    summary_rows = summary_rows.merge(
        recovery, on="condition", how="left", suffixes=("", "_hidden")
    )
    payload = {
        **asdict(config),
        "reference_ids": reference.example_id.astype(str).tolist(),
        "evaluation_ids": evaluation.example_id.astype(str).tolist(),
    }
    run_dir = Path(config.output_dir) / f"single_forward_component_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(run_dir / "single_forward_rows.csv", index=False)
    summary_rows.to_csv(run_dir / "single_forward_summary.csv", index=False)
    moments = pd.DataFrame([
        {
            "level_index": level,
            "channel": channel,
            "mean": float(means[level][channel]),
            "std": float(stds[level][channel]),
        }
        for level in range(len(means))
        for channel in range(len(means[level]))
    ])
    moments.to_csv(run_dir / "clean_channel_moments.csv", index=False)
    elapsed = time.time() - started
    summary = {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "reference_examples": int(reference.example_id.nunique()),
        "evaluation_examples": int(evaluation.example_id.nunique()),
        "cache_path": str(cache_path),
        "config": asdict(config),
        "limitations": [
            "No image masking or patched-clean delta is used by the estimators.",
            "This signature diagnostic still uses the oracle target-candidate reserve; paired clean is evaluation-only except for that route selection.",
            "Clean-population channel means are learned on a disjoint reference split.",
            "The experiment is score-first; geometry is intentionally excluded.",
        ],
    }
    write_summary(run_dir / "summary.json", summary)
    (run_dir / "analysis_digest.md").write_text(
        "\n".join([
            "# Single-forward component signature",
            "",
            f"- elapsed: {elapsed:.1f} s",
            f"- clean reference: {summary['reference_examples']}",
            f"- evaluation: {summary['evaluation_examples']}",
            "- Read single_forward_summary.csv first.",
        ]) + "\n",
        encoding="utf-8",
    )
    StorageBudget(config.output_dir, config.max_output_gb).check()
    return run_dir


if __name__ == "__main__":
    print(run_single_forward_component())
