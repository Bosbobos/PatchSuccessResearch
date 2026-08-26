from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .autonomous_negative_repair import _clusters_from_frame
from .candidate_reserve import _cache_lookup, _evaluate_batch
from .causal_repair import _load_inputs
from .common import REPO_ROOT, load_experiment, release_accelerator_memory, stable_hash
from .component_student import (
    FEATURE_SETS,
    _capture_with_grad,
    _coordinate_arrays,
    _feature_matrix,
    _reset_detect_inference_cache,
    _sample_indices,
)
from .component_targeted_patch import (
    ComponentTargetedPatchConfig,
    _record_lookup,
    build_teacher_cache,
)
from .followup_common import ATTACK_PATH_DB, GROUP_ORDER, MANIFEST_CSV, TRACE_DB
from .improved_component_defense import _proposal_frame
from .mechanism_aware_patch import dynamic_score_geometry_loss
from .mechanism_followup import _head_branches
from .self_counterfactual_defense import _all_class_nms, _detection_set_metrics


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "CandidateRoutingAndAttackPath" / "large_component_student_outputs"
)
MODES = ("known_support", "blind_support")


@dataclass(slots=True)
class LargeComponentStudentConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "mps"
    require_device: bool = False
    train_examples: int = 200
    test_examples: int = 150
    teacher_path_steps: int = 3
    feature_sets: tuple[str, ...] = FEATURE_SETS
    modes: tuple[str, ...] = MODES
    known_max_coordinates_per_image: int = 6000
    blind_candidates_per_level: int = 4000
    blind_apply_top_k: int = 6000
    blind_selector: str = "coordinate"
    blind_top_clusters: int = 3
    blind_person_top_k: int = 1000
    blind_class_agnostic_top_k: int = 1000
    blind_class_agnostic_per_level_k: int = 300
    blind_cluster_candidate_limit: int = 2200
    blind_cluster_min_score: float = 1e-8
    blind_cluster_iou: float = 0.50
    blind_max_cluster_members: int = 100
    max_iter: int = 120
    max_leaf_nodes: int = 31
    learning_rate: float = 0.08
    l2_regularization: float = 1.0
    positive_weight: float = 12.0
    correction_scales: tuple[float, ...] = (1.0,)
    diagnostic_ablations: bool = False
    target_iou: float = 0.50
    detection_conf: float = 0.25
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    nms_max_time_img: float = 1.0
    init_patch: str = str(REPO_ROOT / "data" / "patch.png")
    seed: int = 1201
    method_version: int = 2

    @property
    def match_iou(self) -> float:
        return float(self.target_iou)


def _allocate_balanced(total: int, groups: list[str]) -> dict[str, int]:
    base, remainder = divmod(int(total), len(groups))
    return {
        group: base + int(index < remainder)
        for index, group in enumerate(groups)
    }


def exact_disjoint_split(
    selected: pd.DataFrame,
    *,
    train_examples: int,
    test_examples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create an exact, group-balanced split with no shared underlying scene."""

    groups = [group for group in GROUP_ORDER if selected.analysis_group.eq(group).any()]
    train_counts = _allocate_balanced(train_examples, groups)
    test_counts = _allocate_balanced(test_examples, groups)
    train_parts = []
    test_parts = []
    for index, group in enumerate(groups):
        frame = (
            selected[selected.analysis_group.eq(group)]
            .drop_duplicates("example_id")
            .sample(frac=1.0, random_state=int(seed) + index)
            .reset_index(drop=True)
        )
        required = train_counts[group] + test_counts[group]
        if len(frame) < required:
            raise RuntimeError(
                f"Group {group!r} has {len(frame)} unique examples; "
                f"{required} are required for the requested split."
            )
        train_n = train_counts[group]
        train_parts.append(frame.iloc[:train_n])
        test_parts.append(frame.iloc[train_n:required])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    if len(train) != int(train_examples) or len(test) != int(test_examples):
        raise AssertionError("Exact split size invariant failed.")
    overlap = set(train.example_id.astype(str)) & set(test.example_id.astype(str))
    if overlap:
        raise AssertionError(f"Train/test scene leakage: {sorted(overlap)[:3]}")
    return train, test


def _attacked_reference_statistics(exp, model, detect, records):
    """Population normalization fitted only on attacked training endpoints."""

    sums = sums2 = counts = None
    examples = _cache_lookup(exp)
    for record in tqdm(records, desc="attacked normalization", unit="image"):
        example = examples[record["example_id"]]
        _clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, patched_image, patched_image)
        levels = _capture_detect_inputs(model, detect, pair[:1])
        if sums is None:
            sums = [np.zeros(level.shape[1], dtype=np.float64) for level in levels]
            sums2 = [np.zeros(level.shape[1], dtype=np.float64) for level in levels]
            counts = [0 for _ in levels]
        for level_index, level in enumerate(levels):
            array = level[0].float().cpu().numpy()
            sums[level_index] += array.sum(axis=(1, 2))
            sums2[level_index] += np.square(array, dtype=np.float64).sum(axis=(1, 2))
            counts[level_index] += int(array.shape[1] * array.shape[2])
    means = [value / count for value, count in zip(sums, counts, strict=True)]
    stds = [
        np.sqrt(np.maximum(second / count - mean**2, 1e-6))
        for second, mean, count in zip(sums2, means, counts, strict=True)
    ]
    return means, stds


def _blind_cluster_config(config: LargeComponentStudentConfig) -> SimpleNamespace:
    return SimpleNamespace(
        person_top_k=int(config.blind_person_top_k),
        class_agnostic_top_k=int(config.blind_class_agnostic_top_k),
        class_agnostic_per_level_k=int(config.blind_class_agnostic_per_level_k),
        cluster_min_score=float(config.blind_cluster_min_score),
        cluster_candidate_limit=int(config.blind_cluster_candidate_limit),
        cluster_iou=float(config.blind_cluster_iou),
        max_cluster_members=int(config.blind_max_cluster_members),
    )


def _blind_targets(detect, levels, config: LargeComponentStudentConfig):
    """Generate functional targets from the current endpoint only."""

    import torch

    with torch.no_grad():
        _box, _cls, raw = _head_branches(detect, levels)
        frame = _proposal_frame(
            detect, raw, class_id=0, policy="hybrid",
            config=_blind_cluster_config(config),
        )
    clusters = _clusters_from_frame(frame, _blind_cluster_config(config))
    clusters.sort(
        key=lambda item: (
            float(item["object_suppression_tension"]),
            float(item["reserve_tension"]),
        ),
        reverse=True,
    )
    targets = []
    for cluster in clusters[: int(config.blind_top_clusters)]:
        selection = cluster["selection"]
        score_column = (
            "proposal_score" if "proposal_score" in selection else "score"
        )
        representative = selection.iloc[int(selection[score_column].argmax())]
        class_id = int(
            representative.predicted_class
            if "predicted_class" in selection
            else 0
        )
        targets.append({
            "box": (
                float(representative.x1),
                float(representative.y1),
                float(representative.x2),
                float(representative.y2),
            ),
            "class_id": class_id,
            "flat_index": int(representative.flat_index),
        })
    if not targets and len(frame):
        representative = frame.iloc[0]
        targets.append({
            "box": tuple(
                float(getattr(representative, key))
                for key in ("x1", "y1", "x2", "y2")
            ),
            "class_id": int(representative.predicted_class),
            "flat_index": int(representative.flat_index),
        })
    return targets, len(clusters)


def _blind_endpoint(model, detect, image, config: LargeComponentStudentConfig):
    import torch

    _reset_detect_inference_cache(detect)
    image = image.detach().requires_grad_(True)
    decoded, levels = _capture_with_grad(model, detect, image)
    targets, cluster_count = _blind_targets(detect, levels, config)
    losses = []
    for target in targets:
        box = torch.as_tensor(
            [[target["box"]]], device=image.device, dtype=torch.float32
        ).reshape(1, 4)
        class_id = torch.as_tensor(
            [target["class_id"]], device=image.device, dtype=torch.long
        )
        loss, _ = dynamic_score_geometry_loss(
            decoded,
            box,
            class_id,
            match_iou=0.50,
            iou_temperature=0.07,
            iou_weight=4.0,
            smoothmax_temperature=0.35,
        )
        losses.append(loss)
    if losses:
        objective = torch.stack(losses).mean()
    else:
        objective = decoded[:, 4:, :].sigmoid().amax()
    gradients = torch.autograd.grad(objective, levels)
    return (
        [item.detach() for item in levels],
        [item.detach() for item in gradients],
        {"blind_target_count": len(targets), "blind_cluster_count": cluster_count},
    )


def _teacher_arrays(record, levels):
    indices = []
    components = []
    with np.load(record["component_path"], allow_pickle=False) as data:
        for level_index in range(len(levels)):
            indices.append(data[f"indices_{level_index}"].astype(np.int64))
            components.append(data[f"component_{level_index}"].astype(np.float32))
    return indices, components


def _parts_for_indices(
    levels,
    gradients,
    indices_by_level,
    reference_mean,
    reference_std,
):
    parts = []
    for level_index, (level, gradient, indices) in enumerate(
        zip(levels, gradients, indices_by_level, strict=True)
    ):
        arrays = _coordinate_arrays(
            level[0],
            indices,
            reference_mean[level_index],
            reference_std[level_index],
            gradient[0],
        )
        arrays["level"].fill(level_index / max(len(levels) - 1, 1))
        parts.append(arrays)
    return parts


def _coordinate_candidate_indices(
    levels,
    gradients,
    reference_mean,
    reference_std,
    per_level: int,
):
    """Select endpoint-internal coordinates without teacher support."""

    import torch

    output = []
    for level_index, (level, gradient) in enumerate(
        zip(levels, gradients, strict=True)
    ):
        values = level[0].float()
        channels, height, width = values.shape
        mean = torch.as_tensor(
            reference_mean[level_index], device=values.device, dtype=values.dtype
        ).reshape(-1, 1, 1)
        std = torch.as_tensor(
            reference_std[level_index], device=values.device, dtype=values.dtype
        ).reshape(-1, 1, 1)
        z = ((values - mean) / std).flatten()
        grad = gradient[0].float().flatten()
        grad = grad / torch.sqrt(grad.square().mean()).clamp_min(1e-8)
        leverage = grad.abs() * (1.0 + z.abs().clamp_max(5.0))
        count = min(int(per_level), int(leverage.numel()))
        half = max(1, count // 2)
        by_gradient = grad.abs().topk(half).indices
        by_leverage = leverage.topk(count).indices
        indices = torch.unique(torch.cat((by_gradient, by_leverage)))
        if len(indices) > count:
            chosen = leverage[indices].topk(count).indices
            indices = indices[chosen]
        output.append(
            indices.detach().cpu().numpy().astype(np.int64)
        )
    return output


def _spatial_candidate_indices(
    levels,
    gradients,
    reference_mean,
    reference_std,
    per_level: int,
):
    """Select compact spatial regions, then retain all channels in each region."""

    import torch
    import torch.nn.functional as functional

    output = []
    for level_index, (level, gradient) in enumerate(
        zip(levels, gradients, strict=True)
    ):
        values = level[0].float()
        channels, height, width = values.shape
        mean = torch.as_tensor(
            reference_mean[level_index], device=values.device, dtype=values.dtype
        ).reshape(-1, 1, 1)
        std = torch.as_tensor(
            reference_std[level_index], device=values.device, dtype=values.dtype
        ).reshape(-1, 1, 1)
        z = (values - mean) / std
        grad = gradient[0].float()
        grad = grad / torch.sqrt(grad.square().mean()).clamp_min(1e-8)
        coordinate_leverage = grad.abs() * (1.0 + z.abs().clamp_max(5.0))
        spatial = torch.sqrt(
            coordinate_leverage.square().mean(dim=0).clamp_min(1e-12)
        )
        cell_budget = max(
            1, min(height * width, int(per_level) // max(channels, 1))
        )
        # Select fewer seeds, expand them by one cell, and keep the strongest
        # expanded positions. This matches the local spatial support used by
        # the offline teacher without consulting its coordinates.
        seed_budget = max(1, int(np.ceil(cell_budget / 9)))
        seeds = spatial.flatten().topk(min(seed_budget, spatial.numel())).indices
        seed_mask = torch.zeros(
            (1, 1, height, width), device=values.device, dtype=values.dtype
        )
        seed_mask.flatten()[seeds] = 1.0
        expanded = functional.max_pool2d(seed_mask, 3, 1, 1)[0, 0].bool()
        cells = torch.nonzero(expanded.flatten(), as_tuple=False).reshape(-1)
        if len(cells) < cell_budget:
            extra = spatial.flatten().topk(cell_budget).indices
            cells = torch.unique(torch.cat((cells, extra)))
        if len(cells) > cell_budget:
            cells = cells[spatial.flatten()[cells].topk(cell_budget).indices]
        channels_index = torch.arange(
            channels, device=values.device, dtype=torch.long
        ).reshape(-1, 1)
        indices = channels_index * (height * width) + cells.reshape(1, -1)
        output.append(
            indices.flatten().detach().cpu().numpy().astype(np.int64)
        )
    return output


def _blind_candidate_indices(
    levels,
    gradients,
    reference_mean,
    reference_std,
    per_level: int,
    selector: str = "coordinate",
):
    if selector == "coordinate":
        return _coordinate_candidate_indices(
            levels, gradients, reference_mean, reference_std, per_level
        )
    if selector == "spatial":
        return _spatial_candidate_indices(
            levels, gradients, reference_mean, reference_std, per_level
        )
    if selector == "hybrid":
        coordinate_budget = max(1, int(round(0.70 * int(per_level))))
        spatial_budget = max(1, int(per_level) - coordinate_budget)
        coordinate = _coordinate_candidate_indices(
            levels,
            gradients,
            reference_mean,
            reference_std,
            coordinate_budget,
        )
        spatial = _spatial_candidate_indices(
            levels,
            gradients,
            reference_mean,
            reference_std,
            spatial_budget,
        )
        fallback = _coordinate_candidate_indices(
            levels, gradients, reference_mean, reference_std, per_level
        )
        output = []
        for coordinate_level, spatial_level, fallback_level in zip(
            coordinate, spatial, fallback, strict=True
        ):
            ordered = []
            seen = set()
            for index in np.concatenate(
                (coordinate_level, spatial_level, fallback_level)
            ):
                value = int(index)
                if value not in seen:
                    ordered.append(value)
                    seen.add(value)
                if len(ordered) >= int(per_level):
                    break
            output.append(np.asarray(ordered, dtype=np.int64))
        return output
    raise ValueError(f"Unknown blind selector: {selector}")


def _labels_on_candidates(candidate_indices, teacher_indices, teacher_components):
    labels = []
    for candidates, support, component in zip(
        candidate_indices, teacher_indices, teacher_components, strict=True
    ):
        current = np.zeros(len(candidates), dtype=np.float32)
        if len(candidates) and len(support):
            order = np.argsort(support)
            sorted_support = support[order]
            positions = np.searchsorted(sorted_support, candidates)
            valid = positions < len(sorted_support)
            matched = np.zeros(len(candidates), dtype=bool)
            matched[valid] = (
                sorted_support[positions[valid]] == candidates[valid]
            )
            current[matched] = component[order[positions[matched]]]
        labels.append(current)
    return labels


def _support_metrics(candidate_indices, teacher_indices, teacher_components):
    matched_n = total_n = 0
    matched_energy = total_energy = 0.0
    for candidates, support, component in zip(
        candidate_indices, teacher_indices, teacher_components, strict=True
    ):
        total_n += len(support)
        total_energy += float(np.square(component, dtype=np.float64).sum())
        if len(support):
            matched = np.isin(support, candidates, assume_unique=False)
            matched_n += int(matched.sum())
            matched_energy += float(
                np.square(component[matched], dtype=np.float64).sum()
            )
    return {
        "support_recall": matched_n / max(total_n, 1),
        "support_energy_recall": matched_energy / max(total_energy, 1e-12),
    }


def _subset_flat_parts(parts, labels, maximum: int, seed: int):
    flat_labels = np.concatenate(labels)
    keep = _sample_indices(flat_labels, int(maximum), int(seed))
    sizes = [len(value) for value in labels]
    offsets = np.cumsum([0, *sizes])
    subset_parts = []
    subset_labels = []
    for level_index, part in enumerate(parts):
        local = keep[
            (keep >= offsets[level_index]) & (keep < offsets[level_index + 1])
        ] - offsets[level_index]
        subset_parts.append({key: value[local] for key, value in part.items()})
        subset_labels.append(labels[level_index][local])
    return subset_parts, np.concatenate(subset_labels)


def _build_training_rows(
    exp,
    model,
    detect,
    records,
    reference_mean,
    reference_std,
    config,
):
    examples = _cache_lookup(exp)
    rows = {mode: [] for mode in config.modes}
    localization = []
    for record_index, record in enumerate(
        tqdm(records, desc="student training features", unit="image")
    ):
        example = examples[record["example_id"]]
        _clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, patched_image, patched_image)
        levels, gradients, blind_metadata = _blind_endpoint(
            model, detect, pair[:1], config
        )
        teacher_indices, teacher_components = _teacher_arrays(record, levels)
        if "known_support" in config.modes:
            parts = _parts_for_indices(
                levels, gradients, teacher_indices, reference_mean, reference_std
            )
            parts, labels = _subset_flat_parts(
                parts,
                teacher_components,
                config.known_max_coordinates_per_image,
                config.seed + 17 * record_index,
            )
            rows["known_support"].append({"parts": parts, "labels": labels})
        if "blind_support" in config.modes:
            candidates = _blind_candidate_indices(
                levels,
                gradients,
                reference_mean,
                reference_std,
                config.blind_candidates_per_level,
                config.blind_selector,
            )
            labels_by_level = _labels_on_candidates(
                candidates, teacher_indices, teacher_components
            )
            parts = _parts_for_indices(
                levels, gradients, candidates, reference_mean, reference_std
            )
            rows["blind_support"].append({
                "parts": parts,
                "labels": np.concatenate(labels_by_level),
            })
            localization.append({
                "example_id": record["example_id"],
                "split": "train",
                "blind_selector": config.blind_selector,
                **blind_metadata,
                **_support_metrics(
                    candidates, teacher_indices, teacher_components
                ),
            })
        del levels, gradients
        release_accelerator_memory()
    return rows, localization


def _fit_students(rows_by_mode, config):
    from sklearn.ensemble import HistGradientBoostingRegressor

    students = {}
    metadata = {}
    for mode, rows in rows_by_mode.items():
        if not rows:
            continue
        target = np.concatenate([row["labels"] for row in rows])
        nonzero = target[np.abs(target) > 1e-8]
        target_scale = max(
            float(np.std(nonzero if len(nonzero) else target)), 1e-8
        )
        y = target / target_scale
        weights = (
            1.0
            + float(config.positive_weight) * (np.abs(target) > 1e-8)
            + 2.0 * np.minimum(np.abs(y), 5.0)
        )
        for feature_set in config.feature_sets:
            x = np.concatenate(
                [_feature_matrix(row["parts"], feature_set)[0] for row in rows],
                axis=0,
            )
            student = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=float(config.learning_rate),
                max_iter=int(config.max_iter),
                max_leaf_nodes=int(config.max_leaf_nodes),
                l2_regularization=float(config.l2_regularization),
                random_state=int(config.seed),
            )
            student.fit(x, y, sample_weight=weights)
            students[(mode, feature_set)] = student
            metadata[mode] = {
                "target_scale": target_scale,
                "training_coordinates": int(len(target)),
                "positive_coordinates": int((np.abs(target) > 1e-8).sum()),
            }
    return students, metadata


def _prediction_maps(
    student,
    feature_set,
    parts,
    indices_by_level,
    levels,
    target_scale,
    *,
    apply_top_k: int | None,
):
    import torch

    matrix, _ = _feature_matrix(parts, feature_set)
    prediction = student.predict(matrix).astype(np.float32) * float(target_scale)
    if apply_top_k is not None and len(prediction) > int(apply_top_k):
        keep = np.argpartition(
            np.abs(prediction), -int(apply_top_k)
        )[-int(apply_top_k):]
        mask = np.zeros(len(prediction), dtype=bool)
        mask[keep] = True
        prediction = np.where(mask, prediction, 0.0)
    maps = []
    cursor = 0
    for level, indices_np in zip(levels, indices_by_level, strict=True):
        count = len(indices_np)
        values = prediction[cursor:cursor + count]
        cursor += count
        correction = torch.zeros_like(level[0]).reshape(-1)
        if count:
            indices = torch.as_tensor(
                indices_np, device=level.device, dtype=torch.long
            )
            correction[indices] = torch.as_tensor(
                values, device=level.device, dtype=level.dtype
            )
        maps.append(correction.reshape_as(level[0]).unsqueeze(0))
    return maps, {
        "prediction_l2": float(np.linalg.norm(prediction)),
        "prediction_nonzero": int(np.count_nonzero(prediction)),
    }


def _direct_maps(values_by_level, indices_by_level, levels):
    import torch

    maps = []
    flat_values = []
    for values, indices_np, level in zip(
        values_by_level, indices_by_level, levels, strict=True
    ):
        correction = torch.zeros_like(level[0]).reshape(-1)
        if len(indices_np):
            indices = torch.as_tensor(
                indices_np, device=level.device, dtype=torch.long
            )
            correction[indices] = torch.as_tensor(
                values, device=level.device, dtype=level.dtype
            )
            flat_values.append(np.asarray(values, dtype=np.float32))
        maps.append(correction.reshape_as(level[0]).unsqueeze(0))
    concatenated = (
        np.concatenate(flat_values)
        if flat_values else np.empty(0, dtype=np.float32)
    )
    return maps, {
        "prediction_l2": float(np.linalg.norm(concatenated)),
        "prediction_nonzero": int(np.count_nonzero(concatenated)),
    }


def _evaluate(
    exp,
    model,
    detect,
    records,
    students,
    fit_metadata,
    reference_mean,
    reference_std,
    config,
):
    examples = _cache_lookup(exp)
    output_rows = []
    localization_rows = []
    for record in tqdm(records, desc="student evaluation", unit="scene"):
        example = examples[record["example_id"]]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        for input_kind, image in (("patched", pair[1:2]), ("clean", pair[0:1])):
            mode_inputs: dict[str, dict[str, Any]] = {}
            levels, gradients, blind_metadata = _blind_endpoint(
                model, detect, image, config
            )
            teacher_indices, teacher_components = _teacher_arrays(record, levels)
            if "known_support" in config.modes:
                parts = _parts_for_indices(
                    levels, gradients, teacher_indices, reference_mean, reference_std
                )
                mode_inputs["known_support"] = {
                    "levels": levels,
                    "parts": parts,
                    "indices": teacher_indices,
                    "support_recall": 1.0,
                    "support_energy_recall": 1.0,
                }
            if "blind_support" in config.modes:
                candidates = _blind_candidate_indices(
                    levels,
                    gradients,
                    reference_mean,
                    reference_std,
                    config.blind_candidates_per_level,
                    config.blind_selector,
                )
                metrics = _support_metrics(
                    candidates, teacher_indices, teacher_components
                )
                parts = _parts_for_indices(
                    levels, gradients, candidates, reference_mean, reference_std
                )
                mode_inputs["blind_support"] = {
                    "levels": levels,
                    "parts": parts,
                    "indices": candidates,
                    **metrics,
                    **blind_metadata,
                }
                localization_rows.append({
                    "example_id": record["example_id"],
                    "split": "test",
                    "input_kind": input_kind,
                    "blind_selector": config.blind_selector,
                    **metrics,
                    **blind_metadata,
                })
            base_levels = next(iter(mode_inputs.values()))["levels"]
            conditions = {"observed": base_levels}
            condition_metadata = {}
            for (mode, feature_set), student in students.items():
                current = mode_inputs[mode]
                correction, prediction_metadata = _prediction_maps(
                    student,
                    feature_set,
                    current["parts"],
                    current["indices"],
                    current["levels"],
                    fit_metadata[mode]["target_scale"],
                    apply_top_k=(
                        config.blind_apply_top_k
                        if mode == "blind_support" else None
                    ),
                )
                for scale in config.correction_scales:
                    name = f"{mode}__{feature_set}__s{float(scale):g}"
                    conditions[name] = [
                        level - float(scale) * delta
                        for level, delta in zip(
                            current["levels"], correction, strict=True
                        )
                    ]
                    condition_metadata[name] = {
                        "mode": mode,
                        "feature_set": feature_set,
                        "correction_scale": float(scale),
                        "blind_selector": (
                            config.blind_selector
                            if mode == "blind_support" else "teacher"
                        ),
                        **prediction_metadata,
                        "support_recall": current["support_recall"],
                        "support_energy_recall": current[
                            "support_energy_recall"
                        ],
                    }
            if config.diagnostic_ablations and input_kind == "patched":
                oracle_maps, oracle_metadata = _direct_maps(
                    teacher_components, teacher_indices, levels
                )
                name = "oracle_component__teacher__s1"
                conditions[name] = [
                    level - delta
                    for level, delta in zip(levels, oracle_maps, strict=True)
                ]
                condition_metadata[name] = {
                    "mode": "oracle_component",
                    "feature_set": "teacher",
                    "correction_scale": 1.0,
                    "blind_selector": "teacher",
                    **oracle_metadata,
                    "support_recall": 1.0,
                    "support_energy_recall": 1.0,
                }
                if "blind_support" in mode_inputs:
                    blind = mode_inputs["blind_support"]
                    blind_values = _labels_on_candidates(
                        blind["indices"], teacher_indices, teacher_components
                    )
                    blind_oracle_maps, blind_oracle_metadata = _direct_maps(
                        blind_values, blind["indices"], levels
                    )
                    name = "blind_oracle_values__teacher__s1"
                    conditions[name] = [
                        level - delta
                        for level, delta in zip(
                            levels, blind_oracle_maps, strict=True
                        )
                    ]
                    condition_metadata[name] = {
                        "mode": "blind_oracle_values",
                        "feature_set": "teacher",
                        "correction_scale": 1.0,
                        "blind_selector": config.blind_selector,
                        **blind_oracle_metadata,
                        "support_recall": blind["support_recall"],
                        "support_energy_recall": blind[
                            "support_energy_recall"
                        ],
                    }
            import torch

            with torch.no_grad():
                names = list(conditions)
                batched = [
                    torch.cat(
                        [conditions[name][level] for name in names], dim=0
                    )
                    for level in range(len(base_levels))
                ]
                _box, _cls, raw = _head_branches(detect, batched)
                target_results = _evaluate_batch(
                    detect, raw, record["row"], config
                )
                nms_results = _all_class_nms(detect, raw, config)
            target_by_name = dict(zip(names, target_results, strict=True))
            nms_by_name = dict(zip(names, nms_results, strict=True))
            for name in names[1:]:
                item_metadata = condition_metadata[name]
                full_set = _detection_set_metrics(
                    nms_by_name["observed"],
                    nms_by_name[name],
                    config.target_iou,
                )
                output_rows.append({
                    "example_id": record["example_id"],
                    "analysis_group": record["analysis_group"],
                    "input_kind": input_kind,
                    "baseline_target_detected": target_by_name["observed"][
                        "target_detected"
                    ],
                    "corrected_target_detected": target_by_name[name][
                        "target_detected"
                    ],
                    "baseline_target_conf": target_by_name["observed"][
                        "post_target_conf"
                    ],
                    "corrected_target_conf": target_by_name[name][
                        "post_target_conf"
                    ],
                    "full_detection_f1": full_set["detection_f1"],
                    **item_metadata,
                })
            _reset_detect_inference_cache(detect)
            release_accelerator_memory()
    return pd.DataFrame(output_rows), pd.DataFrame(localization_rows)


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    grouping = ["mode", "feature_set", "correction_scale", "blind_selector"]
    for keys, group in rows.groupby(grouping):
        mode, feature_set, correction_scale, blind_selector = keys
        patched = group[group.input_kind.eq("patched")]
        clean = group[group.input_kind.eq("clean")]
        hidden = patched.baseline_target_detected.eq(0)
        summaries.append({
            "mode": mode,
            "feature_set": feature_set,
            "correction_scale": correction_scale,
            "blind_selector": blind_selector,
            "patched_n": int(len(patched)),
            "clean_n": int(len(clean)),
            "baseline_target_rate": float(
                patched.baseline_target_detected.mean()
            ),
            "corrected_target_rate": float(
                patched.corrected_target_detected.mean()
            ),
            "hidden_n": int(hidden.sum()),
            "hidden_recovered_n": int(
                (
                    hidden
                    & patched.corrected_target_detected.eq(1)
                ).sum()
            ),
            "baseline_lost_n": int(
                (
                    patched.baseline_target_detected.eq(1)
                    & patched.corrected_target_detected.eq(0)
                ).sum()
            ),
            "patched_conf_gain": float(
                (
                    patched.corrected_target_conf
                    - patched.baseline_target_conf
                ).mean()
            ),
            "clean_full_detection_f1": float(
                clean.full_detection_f1.mean() if len(clean) else np.nan
            ),
            "clean_target_change_rate": float(
                (
                    clean.corrected_target_detected
                    != clean.baseline_target_detected
                ).mean() if len(clean) else np.nan
            ),
            "support_recall": float(patched.support_recall.mean()),
            "support_energy_recall": float(
                patched.support_energy_recall.mean()
            ),
        })
    return pd.DataFrame(summaries)


def run_large_component_student(
    config: LargeComponentStudentConfig | None = None,
) -> Path:
    config = config or LargeComponentStudentConfig()
    started = time.time()
    invalid_features = set(config.feature_sets) - set(FEATURE_SETS)
    invalid_modes = set(config.modes) - set(MODES)
    if invalid_features or invalid_modes:
        raise ValueError(
            f"Invalid feature sets {invalid_features} or modes {invalid_modes}."
        )
    if config.blind_selector not in {"coordinate", "spatial", "hybrid"}:
        raise ValueError(f"Invalid blind selector: {config.blind_selector}")
    if not config.correction_scales or any(
        float(scale) <= 0 for scale in config.correction_scales
    ):
        raise ValueError("Correction scales must be non-empty and positive.")
    selected, _ = _load_inputs(
        Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None
    )
    train_rows, test_rows = exact_disjoint_split(
        selected,
        train_examples=config.train_examples,
        test_examples=config.test_examples,
        seed=config.seed,
    )
    exp, attack_cache_path = load_experiment(
        prefer_device=config.device, require_device=config.require_device
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, detector = exp.load_model()
    detector.eval()
    detect = get_detect_module(detector, exp.config.detect_layer)
    detect.eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    teacher_config = ComponentTargetedPatchConfig(
        output_dir=str(
            REPO_ROOT
            / "CandidateRoutingAndAttackPath"
            / "component_patch_outputs"
        ),
        device=config.device,
        require_device=config.require_device,
        train_examples_per_group=1,
        eval_examples_per_group=1,
        teacher_path_steps=config.teacher_path_steps,
        init_patch=config.init_patch,
        seed=config.seed,
        method_version=3,
    )
    teacher_rows = pd.concat([train_rows, test_rows], ignore_index=True)
    teacher_cache_dir, manifest = build_teacher_cache(
        teacher_config,
        exp=exp,
        model=detector,
        detect=detect,
        rows=teacher_rows,
        cache_path=attack_cache_path,
    )
    train = _record_lookup(exp, train_rows, manifest)
    test = _record_lookup(exp, test_rows, manifest)
    row_lookup = {
        str(row.example_id): row
        for row in teacher_rows.itertuples(index=False)
    }
    for record in [*train, *test]:
        record["row"] = row_lookup[record["example_id"]]
    reference_mean, reference_std = _attacked_reference_statistics(
        exp, detector, detect, train
    )
    training_rows, train_localization = _build_training_rows(
        exp,
        detector,
        detect,
        train,
        reference_mean,
        reference_std,
        config,
    )
    students, fit_metadata = _fit_students(training_rows, config)
    evaluation, test_localization = _evaluate(
        exp,
        detector,
        detect,
        test,
        students,
        fit_metadata,
        reference_mean,
        reference_std,
        config,
    )
    payload = {
        **asdict(config),
        "train_ids": [item["example_id"] for item in train],
        "test_ids": [item["example_id"] for item in test],
    }
    run_dir = Path(config.output_dir) / f"large_student_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_rows.to_csv(run_dir / "train_split.csv", index=False)
    test_rows.to_csv(run_dir / "test_split.csv", index=False)
    evaluation.to_csv(run_dir / "evaluation_rows.csv", index=False)
    localization = pd.concat(
        [pd.DataFrame(train_localization), test_localization],
        ignore_index=True,
    )
    localization.to_csv(run_dir / "blind_localization_rows.csv", index=False)
    summary = _summary(evaluation)
    summary.to_csv(run_dir / "summary.csv", index=False)
    np.savez_compressed(
        run_dir / "attacked_reference_statistics.npz",
        **{
            **{
                f"mean_{index}": value
                for index, value in enumerate(reference_mean)
            },
            **{
                f"std_{index}": value
                for index, value in enumerate(reference_std)
            },
        },
    )
    for (mode, feature_set), student in students.items():
        joblib.dump(
            student, run_dir / f"student_{mode}_{feature_set}.joblib"
        )
    metadata = {
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "teacher_cache_dir": str(teacher_cache_dir),
        "fit_metadata": fit_metadata,
        "config": asdict(config),
        "train_endpoint_counts": {"patched": len(train)},
        "test_endpoint_counts": {
            "patched": len(test),
            "clean": len(test),
        },
        "train_test_scene_overlap": 0,
    }
    (run_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "README.md").write_text(
        "# Large component student run\n\n"
        f"- train: {len(train)} patched endpoints only\n"
        f"- test: {len(test)} patched + {len(test)} clean endpoints\n"
        "- train/test underlying scenes are disjoint\n"
        "- blind_support receives no teacher coordinates, target box, class, "
        "or clean endpoint at inference\n\n"
        + summary.to_csv(index=False),
        encoding="utf-8",
    )
    return run_dir


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train attacked-only component students and evaluate known and "
            "blind support on disjoint patched/clean endpoints."
        )
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--train-examples", type=int, default=200)
    parser.add_argument("--test-examples", type=int, default=150)
    parser.add_argument(
        "--feature-sets",
        type=_csv_tuple,
        default=FEATURE_SETS,
        help="Comma-separated subset of activation,local,functional,combined.",
    )
    parser.add_argument(
        "--modes",
        type=_csv_tuple,
        default=MODES,
        help="Comma-separated subset of known_support,blind_support.",
    )
    parser.add_argument(
        "--known-max-coordinates-per-image", type=int, default=6000
    )
    parser.add_argument(
        "--blind-candidates-per-level", type=int, default=4000
    )
    parser.add_argument("--blind-apply-top-k", type=int, default=6000)
    parser.add_argument(
        "--blind-selector",
        choices=("coordinate", "spatial", "hybrid"),
        default="coordinate",
    )
    parser.add_argument("--blind-top-clusters", type=int, default=3)
    parser.add_argument(
        "--correction-scales",
        type=_csv_float_tuple,
        default=(1.0,),
        help="Comma-separated positive correction scales.",
    )
    parser.add_argument("--diagnostic-ablations", action="store_true")
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--teacher-path-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.smoke:
        args.train_examples = 4
        args.test_examples = 2
        args.known_max_coordinates_per_image = 500
        args.blind_candidates_per_level = 300
        args.blind_apply_top_k = 300
        args.blind_top_clusters = 1
        args.max_iter = 5
        args.teacher_path_steps = 1
    config = LargeComponentStudentConfig(
        output_dir=args.output_dir,
        device=args.device,
        require_device=args.require_device,
        train_examples=args.train_examples,
        test_examples=args.test_examples,
        teacher_path_steps=args.teacher_path_steps,
        feature_sets=args.feature_sets,
        modes=args.modes,
        known_max_coordinates_per_image=args.known_max_coordinates_per_image,
        blind_candidates_per_level=args.blind_candidates_per_level,
        blind_apply_top_k=args.blind_apply_top_k,
        blind_selector=args.blind_selector,
        blind_top_clusters=args.blind_top_clusters,
        correction_scales=args.correction_scales,
        diagnostic_ablations=args.diagnostic_ablations,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    print(run_large_component_student(config))


if __name__ == "__main__":
    main()
