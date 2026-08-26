from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _preprocess_pair
from .candidate_reserve import _cache_lookup
from .candidate_routing import _box_iou, _xywh_to_xyxy
from .common import REPO_ROOT, load_experiment, release_accelerator_memory, stable_hash
from .component_student import (
    _capture_with_grad,
    _feature_matrix,
    _reset_detect_inference_cache,
)
from .component_targeted_patch import _record_lookup
from .large_component_student import (
    _blind_candidate_indices,
    _direct_maps,
    _labels_on_candidates,
    _parts_for_indices,
    _support_metrics,
    _teacher_arrays,
)
from .learned_cluster_ranker import (
    FEATURE_COLUMNS as ENDPOINT_FEATURE_COLUMNS,
    _average_gradient_for_clusters,
    _cluster_energy_fraction,
    _cluster_features,
    _cluster_mask,
    _clusters_for_levels,
    _teacher_spatial_energy,
)
from .localization_mechanism_sweep import _evaluate_conditions, _load_base_config
from .mechanism_followup import _decode, _head_branches


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "CandidateRoutingAndAttackPath" / "component_ranker_outputs"
)
COMPONENT_FEATURE_COLUMNS = (
    "student_energy_fraction",
    "student_abs_fraction",
    "student_energy_density",
    "student_abs_mean",
    "student_abs_max",
    "student_active_fraction",
    "activation_energy_fraction",
    "gradient_energy_fraction",
    "leverage_energy_fraction",
    "fused_energy_fraction",
    "activation_energy_density",
    "gradient_energy_density",
    "leverage_energy_density",
)
FEATURE_COLUMNS = (*ENDPOINT_FEATURE_COLUMNS, *COMPONENT_FEATURE_COLUMNS)


@dataclass(slots=True)
class ComponentAwareRankerConfig:
    base_run: str
    ranker_run: str | None = None
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "mps"
    require_device: bool = False
    student_feature_set: str = "functional"
    top_clusters: int = 5
    candidate_budget: int = 4000
    student_apply_top_k: int = 6000
    pair_samples_per_scene: int = 600
    pairwise_max_iter: int = 180
    pairwise_max_leaf_nodes: int = 31
    set_rank_weight: float = 0.65
    expansion_factor: int = 2
    condition_batch_size: int = 10
    max_train_scenes: int | None = None
    max_test_scenes: int | None = None
    seed: int = 1801
    method_version: int = 2


def _heuristic_indices(clusters, top_k: int) -> list[int]:
    return sorted(
        range(len(clusters)),
        key=lambda index: (
            clusters[index]["object_suppression_tension"],
            clusters[index]["reserve_tension"],
        ),
        reverse=True,
    )[: int(top_k)]


def _student_spatial_signal(
    student,
    feature_set,
    target_scale,
    levels,
    gradients,
    reference_mean,
    reference_std,
    candidate_budget,
    apply_top_k,
):
    candidates = _blind_candidate_indices(
        levels,
        gradients,
        reference_mean,
        reference_std,
        int(candidate_budget),
        "hybrid",
    )
    parts = _parts_for_indices(
        levels,
        gradients,
        candidates,
        reference_mean,
        reference_std,
    )
    matrix, _ = _feature_matrix(parts, feature_set)
    prediction = (
        student.predict(matrix).astype(np.float32) * float(target_scale)
    )
    if apply_top_k is not None and len(prediction) > int(apply_top_k):
        keep = np.argpartition(
            np.abs(prediction), -int(apply_top_k)
        )[-int(apply_top_k):]
        retained = np.zeros(len(prediction), dtype=bool)
        retained[keep] = True
        prediction = np.where(retained, prediction, 0.0)
    energy_maps = []
    abs_maps = []
    count_maps = []
    activation_maps = []
    gradient_maps = []
    leverage_maps = []
    cursor = 0
    for level_index, (level, gradient, indices) in enumerate(
        zip(levels, gradients, candidates, strict=True)
    ):
        channels, height, width = map(int, level.shape[1:])
        count = len(indices)
        values = prediction[cursor:cursor + count]
        cursor += count
        spatial = indices % (height * width)
        energy = np.zeros(height * width, dtype=np.float64)
        absolute = np.zeros(height * width, dtype=np.float64)
        counts = np.zeros(height * width, dtype=np.float64)
        np.add.at(energy, spatial, np.square(values, dtype=np.float64))
        np.add.at(absolute, spatial, np.abs(values).astype(np.float64))
        np.add.at(counts, spatial, (np.abs(values) > 0).astype(np.float64))
        energy_maps.append(energy.reshape(height, width))
        abs_maps.append(absolute.reshape(height, width))
        count_maps.append(counts.reshape(height, width))
        values_t = level[0].detach().float()
        gradient_t = gradient[0].detach().float()
        import torch

        mean_t = torch.as_tensor(
            reference_mean[level_index],
            device=values_t.device,
            dtype=values_t.dtype,
        ).reshape(-1, 1, 1)
        std_t = torch.as_tensor(
            reference_std[level_index],
            device=values_t.device,
            dtype=values_t.dtype,
        ).reshape(-1, 1, 1)
        z_t = (values_t - mean_t) / std_t
        gradient_t = gradient_t / torch.sqrt(
            gradient_t.square().mean()
        ).clamp_min(1e-8)
        leverage_t = gradient_t.abs() * (1.0 + z_t.abs().clamp_max(5.0))
        activation_maps.append(
            z_t.square().mean(dim=0).cpu().numpy().astype(np.float64)
        )
        gradient_maps.append(
            gradient_t.square().mean(dim=0).cpu().numpy().astype(np.float64)
        )
        leverage_maps.append(
            leverage_t.square().mean(dim=0).cpu().numpy().astype(np.float64)
        )
    families = (energy_maps, activation_maps, gradient_maps, leverage_maps)
    normalized = []
    for family in families:
        total = max(sum(float(value.sum()) for value in family), 1e-12)
        normalized.append([value / total for value in family])
    fused_maps = [
        (
            0.45 * normalized[0][level]
            + 0.15 * normalized[1][level]
            + 0.15 * normalized[2][level]
            + 0.25 * normalized[3][level]
        )
        for level in range(len(levels))
    ]
    return {
        "energy_maps": energy_maps,
        "abs_maps": abs_maps,
        "count_maps": count_maps,
        "activation_maps": activation_maps,
        "gradient_maps": gradient_maps,
        "leverage_maps": leverage_maps,
        "fused_maps": fused_maps,
        "prediction_l2": float(np.linalg.norm(prediction)),
        "prediction_nonzero": int(np.count_nonzero(prediction)),
    }


def _component_cluster_features(clusters, levels, signal) -> pd.DataFrame:
    total_energy = max(
        sum(float(value.sum()) for value in signal["energy_maps"]), 1e-12
    )
    total_abs = max(
        sum(float(value.sum()) for value in signal["abs_maps"]), 1e-12
    )
    records = []
    family_names = ("activation", "gradient", "leverage", "fused")
    family_maps = {
        name: signal[f"{name}_maps"]
        for name in family_names
    }
    family_totals = {
        name: max(sum(float(value.sum()) for value in maps), 1e-12)
        for name, maps in family_maps.items()
    }
    for cluster_index, cluster in enumerate(clusters):
        masks = _cluster_mask(cluster, levels)
        energy = sum(
            float(value[mask].sum())
            for value, mask in zip(signal["energy_maps"], masks, strict=True)
        )
        absolute = sum(
            float(value[mask].sum())
            for value, mask in zip(signal["abs_maps"], masks, strict=True)
        )
        active = sum(
            float(value[mask].sum())
            for value, mask in zip(signal["count_maps"], masks, strict=True)
        )
        mask_cells = max(sum(int(mask.sum()) for mask in masks), 1)
        cell_energy = np.concatenate(
            [
                value[mask]
                for value, mask in zip(
                    signal["energy_maps"], masks, strict=True
                )
            ]
        )
        cell_abs = np.concatenate(
            [
                value[mask]
                for value, mask in zip(
                    signal["abs_maps"], masks, strict=True
                )
            ]
        )
        family_values = {
            name: sum(
                float(value[mask].sum())
                for value, mask in zip(maps, masks, strict=True)
            )
            for name, maps in family_maps.items()
        }
        records.append({
            "cluster_index": cluster_index,
            "student_energy_fraction": energy / total_energy,
            "student_abs_fraction": absolute / total_abs,
            "student_energy_density": energy / mask_cells,
            "student_abs_mean": float(cell_abs.mean()),
            "student_abs_max": float(cell_abs.max(initial=0.0)),
            "student_active_fraction": active / mask_cells,
            **{
                f"{name}_energy_fraction": (
                    family_values[name] / family_totals[name]
                )
                for name in family_names
            },
            **{
                f"{name}_energy_density": (
                    family_values[name] / mask_cells
                )
                for name in ("activation", "gradient", "leverage")
            },
        })
    return pd.DataFrame(records)


def _scene_features(
    decoded,
    levels,
    clusters,
    image_size,
    student,
    target_scale,
    reference_mean,
    reference_std,
    config,
):
    heuristic = _heuristic_indices(clusters, config.top_clusters)
    gradients = _average_gradient_for_clusters(
        decoded, levels, [clusters[index] for index in heuristic]
    )
    signal = _student_spatial_signal(
        student,
        config.student_feature_set,
        target_scale,
        levels,
        gradients,
        reference_mean,
        reference_std,
        config.candidate_budget,
        config.student_apply_top_k,
    )
    endpoint = _cluster_features(clusters, int(image_size))
    component = _component_cluster_features(clusters, levels, signal)
    return endpoint.merge(component, on="cluster_index"), signal, gradients


def _pairwise_training_data(frame, config):
    rng = np.random.default_rng(int(config.seed))
    x_parts = []
    y_parts = []
    for _example_id, group in frame.groupby("example_id", sort=False):
        matrix = group[list(FEATURE_COLUMNS)].to_numpy(np.float32)
        target = group.teacher_energy_fraction.to_numpy(np.float32)
        if len(group) < 2:
            continue
        requested = int(config.pair_samples_per_scene)
        ordered = np.argsort(-target, kind="stable")
        leaders = ordered[: min(12, len(ordered))]
        pairs: set[tuple[int, int]] = set()
        for leader in leaders:
            lower = np.flatnonzero(target < target[leader] - 1e-8)
            if len(lower):
                chosen = rng.choice(
                    lower, size=min(24, len(lower)), replace=False
                )
                pairs.update((int(leader), int(other)) for other in chosen)
        attempts = 0
        while len(pairs) < requested and attempts < 4 * requested:
            left = int(rng.integers(0, len(group)))
            right = int(rng.integers(0, len(group)))
            attempts += 1
            if left != right and abs(float(target[left] - target[right])) > 1e-8:
                pairs.add((left, right))
        if not pairs:
            continue
        pairs_list = list(pairs)
        if len(pairs_list) > requested:
            selected = rng.choice(
                len(pairs_list), size=requested, replace=False
            )
            pairs_list = [pairs_list[int(index)] for index in selected]
        left = np.asarray([item[0] for item in pairs_list], dtype=np.int64)
        right = np.asarray([item[1] for item in pairs_list], dtype=np.int64)
        difference = matrix[left] - matrix[right]
        label = (target[left] > target[right]).astype(np.int8)
        x_parts.append(difference)
        y_parts.append(label)
    if not x_parts:
        raise RuntimeError("No non-tied cluster pairs were available.")
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    return np.concatenate((x, -x)), np.concatenate((y, 1 - y))


def _fit_pairwise_ranker(training_frame, config):
    from sklearn.ensemble import HistGradientBoostingClassifier

    x, y = _pairwise_training_data(training_frame, config)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.07,
        max_iter=int(config.pairwise_max_iter),
        max_leaf_nodes=int(config.pairwise_max_leaf_nodes),
        l2_regularization=2.0,
        random_state=int(config.seed),
    )
    model.fit(x, y)
    return model, len(y)


def _pairwise_scores(model, features):
    matrix = features[list(FEATURE_COLUMNS)].to_numpy(np.float32)
    count = len(matrix)
    if count == 1:
        return np.ones(1, dtype=np.float64)
    left = np.repeat(np.arange(count), count)
    right = np.tile(np.arange(count), count)
    valid = left != right
    probabilities = model.predict_proba(
        matrix[left[valid]] - matrix[right[valid]]
    )[:, 1]
    scores = np.zeros(count, dtype=np.float64)
    counts = np.zeros(count, dtype=np.float64)
    np.add.at(scores, left[valid], probabilities)
    np.add.at(counts, left[valid], 1.0)
    return scores / np.maximum(counts, 1.0)


def _greedy_marginal_indices(
    clusters,
    levels,
    energy_maps,
    top_k,
    *,
    base_scores=None,
    base_weight=0.0,
):
    covered = [
        np.zeros_like(value, dtype=bool)
        for value in energy_maps
    ]
    remaining = set(range(len(clusters)))
    chosen = []
    base = (
        np.asarray(base_scores, dtype=np.float64)
        if base_scores is not None
        else np.zeros(len(clusters), dtype=np.float64)
    )
    if len(base) and float(np.ptp(base)) > 1e-12:
        base = (base - base.min()) / np.ptp(base)
    total = max(sum(float(value.sum()) for value in energy_maps), 1e-12)
    for _ in range(min(int(top_k), len(clusters))):
        candidates = []
        marginal_values = []
        for index in remaining:
            masks = _cluster_mask(clusters[index], levels)
            marginal = sum(
                float(energy[mask & ~seen].sum())
                for energy, mask, seen in zip(
                    energy_maps, masks, covered, strict=True
                )
            ) / total
            candidates.append(index)
            marginal_values.append(marginal)
        marginal_values = np.asarray(marginal_values, dtype=np.float64)
        if len(marginal_values) and marginal_values.max() > 1e-12:
            marginal_values = marginal_values / marginal_values.max()
        utilities = (
            (1.0 - float(base_weight)) * marginal_values
            + float(base_weight) * base[np.asarray(candidates)]
        )
        winner = candidates[int(np.argmax(utilities))]
        chosen.append(winner)
        remaining.remove(winner)
        for seen, mask in zip(
            covered, _cluster_mask(clusters[winner], levels), strict=True
        ):
            seen |= mask
    return chosen


def _selected_sets(
    clusters,
    levels,
    features,
    pairwise_model,
    student_signal,
    teacher_energy_maps,
    teacher_energy,
    config,
):
    pairwise = _pairwise_scores(pairwise_model, features)
    student_energy = features.student_energy_fraction.to_numpy(float)
    return {
        "heuristic_top5": _heuristic_indices(clusters, config.top_clusters),
        "student_energy_top5": np.argsort(-student_energy)[
            : config.top_clusters
        ].astype(int).tolist(),
        "fused_energy_top5": features.nlargest(
            config.top_clusters, "fused_energy_fraction"
        ).cluster_index.astype(int).tolist(),
        "component_pairwise_top5": np.argsort(-pairwise)[
            : config.top_clusters
        ].astype(int).tolist(),
        "component_set_top5": _greedy_marginal_indices(
            clusters,
            levels,
            student_signal["fused_maps"],
            config.top_clusters,
            base_scores=pairwise,
            base_weight=config.set_rank_weight,
        ),
        "oracle_energy_top5": np.argsort(-np.asarray(teacher_energy))[
            : config.top_clusters
        ].astype(int).tolist(),
        "oracle_union_top5": _greedy_marginal_indices(
            clusters,
            levels,
            teacher_energy_maps,
            config.top_clusters,
        ),
    }, pairwise


def _summary(evaluation):
    summary_rows = []
    group_rows = []
    for condition, current in evaluation.groupby("condition"):
        hidden = current.baseline_target_detected.eq(0)
        summary_rows.append({
            "condition": condition,
            "n": int(len(current)),
            "baseline_target_rate": float(
                current.baseline_target_detected.mean()
            ),
            "corrected_target_rate": float(
                current.corrected_target_detected.mean()
            ),
            "hidden_n": int(hidden.sum()),
            "hidden_recovered_n": int(
                (hidden & current.corrected_target_detected.eq(1)).sum()
            ),
            "baseline_lost_n": int(
                (
                    current.baseline_target_detected.eq(1)
                    & current.corrected_target_detected.eq(0)
                ).sum()
            ),
            "confidence_gain": float(
                (
                    current.corrected_target_conf
                    - current.baseline_target_conf
                ).mean()
            ),
            "support_energy_recall": float(
                current.support_energy_recall.mean()
            ),
            "selected_teacher_union_energy": float(
                current.selected_teacher_union_energy.mean()
            ),
        })
        for analysis_group, group in current.groupby("analysis_group"):
            group_hidden = group.baseline_target_detected.eq(0)
            group_rows.append({
                "condition": condition,
                "analysis_group": analysis_group,
                "n": int(len(group)),
                "corrected_target_rate": float(
                    group.corrected_target_detected.mean()
                ),
                "hidden_recovered_n": int(
                    (
                        group_hidden
                        & group.corrected_target_detected.eq(1)
                    ).sum()
                ),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(group_rows)


def _union_energy(indices, clusters, levels, energy_maps, total_energy):
    union = [np.zeros_like(value, dtype=bool) for value in energy_maps]
    for index in indices:
        for current, mask in zip(
            union, _cluster_mask(clusters[index], levels), strict=True
        ):
            current |= mask
    return sum(
        float(energy[mask].sum())
        for energy, mask in zip(energy_maps, union, strict=True)
    ) / max(float(total_energy), 1e-12)


def _ordered_candidate_union(*candidate_sets):
    output = []
    for level_sets in zip(*candidate_sets, strict=True):
        seen = set()
        ordered = []
        for indices in level_sets:
            for index in indices:
                value = int(index)
                if value not in seen:
                    seen.add(value)
                    ordered.append(value)
        output.append(np.asarray(ordered, dtype=np.int64))
    return output


def _condition_from_candidates(
    levels,
    candidates,
    teacher_indices,
    teacher_components,
):
    values = _labels_on_candidates(
        candidates, teacher_indices, teacher_components
    )
    correction, _ = _direct_maps(values, candidates, levels)
    return [
        level - delta
        for level, delta in zip(levels, correction, strict=True)
    ]


def _evaluate_detection_merge(detect, branches, row, config):
    """Merge corrected detector predictions without using target coordinates."""

    import torch
    from ultralytics.utils.nms import non_max_suppression

    decoded_parts = []
    with torch.no_grad():
        for levels in branches:
            _box, _cls, raw = _head_branches(detect, levels)
            decoded_parts.append(_decode(detect, raw))
        decoded = torch.cat(decoded_parts, dim=2)
        detections = non_max_suppression(
            decoded.detach().clone(),
            conf_thres=config.nms_conf,
            iou_thres=config.nms_iou,
            classes=[int(row.class_id)],
            max_det=config.nms_max_det,
            nc=int(detect.nc),
            max_time_img=float(getattr(config, "nms_max_time_img", 1.0)),
        )[0]
    target_box = torch.as_tensor(
        [[
            row.clean_target_x1,
            row.clean_target_y1,
            row.clean_target_x2,
            row.clean_target_y2,
        ]],
        device=decoded.device,
        dtype=torch.float32,
    )
    post_conf = 0.0
    post_iou = 0.0
    if len(detections):
        ious = _box_iou(detections[:, :4], target_box).reshape(-1)
        post_iou = float(ious.max().cpu())
        valid = torch.nonzero(
            ious >= config.target_iou, as_tuple=False
        ).reshape(-1)
        if len(valid):
            post_conf = float(detections[valid, 4].max().cpu())
    detected = int(
        post_conf >= config.detection_conf
        and post_iou >= config.target_iou
    )
    return {
        "post_target_conf": post_conf,
        "post_target_iou": post_iou,
        "target_detected": detected,
    }


def run_component_aware_ranker(config: ComponentAwareRankerConfig) -> Path:
    started = time.time()
    base_run = Path(config.base_run).resolve()
    metadata = json.loads((base_run / "run.json").read_text(encoding="utf-8"))
    base_config = _load_base_config(metadata, config)
    train_rows = pd.read_csv(base_run / "train_split.csv")
    test_rows = pd.read_csv(base_run / "test_split.csv")
    if config.max_train_scenes is not None:
        train_rows = train_rows.head(int(config.max_train_scenes)).copy()
    if config.max_test_scenes is not None:
        hidden = test_rows[
            test_rows.analysis_group.astype(str).str.startswith("hidden")
        ]
        test_rows = hidden.head(int(config.max_test_scenes)).copy()
    student_path = (
        base_run
        / f"student_blind_support_{config.student_feature_set}.joblib"
    )
    if not student_path.exists():
        raise FileNotFoundError(
            f"Blind student not found for {config.student_feature_set!r}: "
            f"{student_path}"
        )
    student = joblib.load(student_path)
    target_scale = float(
        metadata["fit_metadata"]["blind_support"]["target_scale"]
    )
    stats = np.load(base_run / "attacked_reference_statistics.npz")
    level_count = len([key for key in stats.files if key.startswith("mean_")])
    reference_mean = [stats[f"mean_{index}"] for index in range(level_count)]
    reference_std = [stats[f"std_{index}"] for index in range(level_count)]
    manifest = json.loads(
        (
            Path(metadata["teacher_cache_dir"]) / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    exp, _cache_path = load_experiment(
        prefer_device=config.device, require_device=config.require_device
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_records = _record_lookup(exp, train_rows, manifest)
    test_records = _record_lookup(exp, test_rows, manifest)
    all_rows = pd.concat([train_rows, test_rows], ignore_index=True)
    row_lookup = {
        str(row.example_id): row
        for row in all_rows.itertuples(index=False)
    }
    for record in [*train_records, *test_records]:
        record["row"] = row_lookup[record["example_id"]]
    examples = _cache_lookup(exp)

    if config.ranker_run is not None:
        ranker_run = Path(config.ranker_run).resolve()
        pairwise_model = joblib.load(
            ranker_run / "component_pairwise_ranker.joblib"
        )
        training_frame = pd.read_csv(
            ranker_run / "component_ranker_training_rows.csv"
        )
        ranker_metadata = json.loads(
            (ranker_run / "run.json").read_text(encoding="utf-8")
        )
        if tuple(ranker_metadata["feature_columns"]) != FEATURE_COLUMNS:
            raise RuntimeError(
                "The pretrained ranker feature schema does not match."
            )
        training_pair_count = int(ranker_metadata["training_pairs"])
    else:
        training_rows = []
        for record in tqdm(
            train_records, desc="component ranker features", unit="scene"
        ):
            example = examples[record["example_id"]]
            _clean, patched, _ = exp._images_for_example(example)
            pair = _preprocess_pair(exp, patched, patched)
            _reset_detect_inference_cache(detect)
            image = pair[:1].detach().requires_grad_(True)
            decoded, levels = _capture_with_grad(model, detect, image)
            clusters = _clusters_for_levels(detect, levels, base_config)
            features, _student_signal, _ = _scene_features(
                decoded,
                levels,
                clusters,
                int(pair.shape[-1]),
                student,
                target_scale,
                reference_mean,
                reference_std,
                config,
            )
            teacher_indices, teacher_components = _teacher_arrays(
                record, levels
            )
            energy_maps, total_energy = _teacher_spatial_energy(
                teacher_indices, teacher_components, levels
            )
            features["teacher_energy_fraction"] = [
                _cluster_energy_fraction(
                    cluster, energy_maps, total_energy, levels
                )
                for cluster in clusters
            ]
            features["example_id"] = record["example_id"]
            features["analysis_group"] = record["analysis_group"]
            training_rows.append(features)
            _reset_detect_inference_cache(detect)
            release_accelerator_memory()
        training_frame = pd.concat(training_rows, ignore_index=True)
        pairwise_model, training_pair_count = _fit_pairwise_ranker(
            training_frame, config
        )

    evaluation_rows = []
    selected_rows = []
    for record in tqdm(
        test_records, desc="component ranker evaluation", unit="scene"
    ):
        example = examples[record["example_id"]]
        _clean, patched, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, patched, patched)
        _reset_detect_inference_cache(detect)
        image = pair[:1].detach().requires_grad_(True)
        decoded, levels = _capture_with_grad(model, detect, image)
        clusters = _clusters_for_levels(detect, levels, base_config)
        features, student_signal, _ = _scene_features(
            decoded,
            levels,
            clusters,
            int(pair.shape[-1]),
            student,
            target_scale,
            reference_mean,
            reference_std,
            config,
        )
        teacher_indices, teacher_components = _teacher_arrays(record, levels)
        energy_maps, total_energy = _teacher_spatial_energy(
            teacher_indices, teacher_components, levels
        )
        teacher_energy = [
            _cluster_energy_fraction(
                cluster, energy_maps, total_energy, levels
            )
            for cluster in clusters
        ]
        selected, pairwise_scores = _selected_sets(
            clusters,
            levels,
            features,
            pairwise_model,
            student_signal,
            energy_maps,
            teacher_energy,
            config,
        )
        conditions = {}
        condition_metadata = {}
        candidate_sets = {}
        gradient_sets = {}
        for condition, indices in selected.items():
            gradients = _average_gradient_for_clusters(
                decoded, levels, [clusters[index] for index in indices]
            )
            candidates = _blind_candidate_indices(
                levels,
                gradients,
                reference_mean,
                reference_std,
                int(config.candidate_budget),
                "hybrid",
            )
            gradient_sets[condition] = gradients
            candidate_sets[condition] = candidates
            conditions[condition] = _condition_from_candidates(
                levels,
                candidates,
                teacher_indices,
                teacher_components,
            )
            condition_metadata[condition] = {
                **_support_metrics(
                    candidates, teacher_indices, teacher_components
                ),
                "selected_teacher_union_energy": _union_energy(
                    indices,
                    clusters,
                    levels,
                    energy_maps,
                    total_energy,
                ),
            }
            for rank, index in enumerate(indices, start=1):
                selected_rows.append({
                    "example_id": record["example_id"],
                    "analysis_group": record["analysis_group"],
                    "condition": condition,
                    "rank": rank,
                    "cluster_index": index,
                    "pairwise_score": float(pairwise_scores[index]),
                    "student_energy_fraction": float(
                        features.iloc[index].student_energy_fraction
                    ),
                    "teacher_energy_fraction": float(teacher_energy[index]),
                })

        deployable_branches = (
            "component_pairwise_top5",
            "heuristic_top5",
            "component_set_top5",
        )
        union_specs = {
            "dual_support_union_8k": deployable_branches[:2],
            "triple_support_union_12k": deployable_branches,
        }
        for condition, branches in union_specs.items():
            candidates = _ordered_candidate_union(
                *(candidate_sets[name] for name in branches)
            )
            candidate_sets[condition] = candidates
            conditions[condition] = _condition_from_candidates(
                levels,
                candidates,
                teacher_indices,
                teacher_components,
            )
            cluster_indices = sorted({
                index
                for name in branches
                for index in selected[name]
            })
            condition_metadata[condition] = {
                **_support_metrics(
                    candidates, teacher_indices, teacher_components
                ),
                "selected_teacher_union_energy": _union_energy(
                    cluster_indices,
                    clusters,
                    levels,
                    energy_maps,
                    total_energy,
                ),
            }

        expanded_name = (
            f"component_pairwise_{config.expansion_factor}x_support"
        )
        expanded_candidates = _blind_candidate_indices(
            levels,
            gradient_sets["component_pairwise_top5"],
            reference_mean,
            reference_std,
            int(config.candidate_budget) * int(config.expansion_factor),
            "hybrid",
        )
        candidate_sets[expanded_name] = expanded_candidates
        conditions[expanded_name] = _condition_from_candidates(
            levels,
            expanded_candidates,
            teacher_indices,
            teacher_components,
        )
        condition_metadata[expanded_name] = {
            **_support_metrics(
                expanded_candidates,
                teacher_indices,
                teacher_components,
            ),
            "selected_teacher_union_energy": _union_energy(
                selected["component_pairwise_top5"],
                clusters,
                levels,
                energy_maps,
                total_energy,
            ),
        }

        # Closed-loop localization: remove the first-stage support, rebuild
        # proposals on the residual endpoint, then union the newly found
        # coordinates with the original pairwise support.
        intermediate = [
            value.detach().requires_grad_(True)
            for value in conditions["component_pairwise_top5"]
        ]
        _box, _cls, intermediate_raw = _head_branches(detect, intermediate)
        intermediate_decoded = _decode(detect, intermediate_raw)
        intermediate_clusters = _clusters_for_levels(
            detect, intermediate, base_config
        )
        if intermediate_clusters:
            intermediate_features, _signal, _ = _scene_features(
                intermediate_decoded,
                intermediate,
                intermediate_clusters,
                int(pair.shape[-1]),
                student,
                target_scale,
                reference_mean,
                reference_std,
                config,
            )
            intermediate_pairwise = _pairwise_scores(
                pairwise_model, intermediate_features
            )
            iterative_selections = {
                "iterative_pairwise_then_heuristic": _heuristic_indices(
                    intermediate_clusters, config.top_clusters
                ),
                "iterative_pairwise_then_pairwise": np.argsort(
                    -intermediate_pairwise
                )[: config.top_clusters].astype(int).tolist(),
            }
            for condition, indices in iterative_selections.items():
                residual_gradients = _average_gradient_for_clusters(
                    intermediate_decoded,
                    intermediate,
                    [intermediate_clusters[index] for index in indices],
                )
                residual_candidates = _blind_candidate_indices(
                    intermediate,
                    residual_gradients,
                    reference_mean,
                    reference_std,
                    int(config.candidate_budget),
                    "hybrid",
                )
                candidates = _ordered_candidate_union(
                    candidate_sets["component_pairwise_top5"],
                    residual_candidates,
                )
                candidate_sets[condition] = candidates
                conditions[condition] = _condition_from_candidates(
                    levels,
                    candidates,
                    teacher_indices,
                    teacher_components,
                )
                condition_metadata[condition] = {
                    **_support_metrics(
                        candidates, teacher_indices, teacher_components
                    ),
                    "selected_teacher_union_energy": np.nan,
                }
        exact_maps, _ = _direct_maps(
            teacher_components, teacher_indices, levels
        )
        conditions["exact_teacher"] = [
            level - delta
            for level, delta in zip(levels, exact_maps, strict=True)
        ]
        condition_metadata["exact_teacher"] = {
            "support_recall": 1.0,
            "support_energy_recall": 1.0,
            "selected_teacher_union_energy": 1.0,
        }
        evaluated = _evaluate_conditions(
            detect,
            levels,
            conditions,
            record["row"],
            base_config,
            config.condition_batch_size,
        )
        baseline = evaluated["observed"]
        for condition, item in condition_metadata.items():
            corrected = evaluated[condition]
            evaluation_rows.append({
                "example_id": record["example_id"],
                "analysis_group": record["analysis_group"],
                "condition": condition,
                "baseline_target_detected": baseline["target_detected"],
                "corrected_target_detected": corrected["target_detected"],
                "baseline_target_conf": baseline["post_target_conf"],
                "corrected_target_conf": corrected["post_target_conf"],
                **item,
            })

        merge_specs = {
            "merge_pairwise_heuristic": deployable_branches[:2],
            "merge_pairwise_heuristic_set": deployable_branches,
        }
        iterative_available = [
            name
            for name in (
                "iterative_pairwise_then_heuristic",
                "iterative_pairwise_then_pairwise",
            )
            if name in conditions
        ]
        if iterative_available:
            merge_specs["merge_pairwise_heuristic_iterative"] = (
                *deployable_branches[:2],
                *iterative_available,
            )
        for condition, branches in merge_specs.items():
            corrected = _evaluate_detection_merge(
                detect,
                [conditions[name] for name in branches],
                record["row"],
                base_config,
            )
            candidates = _ordered_candidate_union(
                *(candidate_sets[name] for name in branches)
            )
            evaluation_rows.append({
                "example_id": record["example_id"],
                "analysis_group": record["analysis_group"],
                "condition": condition,
                "baseline_target_detected": baseline["target_detected"],
                "corrected_target_detected": corrected["target_detected"],
                "baseline_target_conf": baseline["post_target_conf"],
                "corrected_target_conf": corrected["post_target_conf"],
                **_support_metrics(
                    candidates, teacher_indices, teacher_components
                ),
                "selected_teacher_union_energy": np.nan,
            })
        _reset_detect_inference_cache(detect)
        release_accelerator_memory()

    evaluation = pd.DataFrame(evaluation_rows)
    summary, group_summary = _summary(evaluation)
    payload = {
        **asdict(config),
        "base_run": str(base_run),
        "train_ids": train_rows.example_id.astype(str).tolist(),
        "test_ids": test_rows.example_id.astype(str).tolist(),
    }
    run_dir = (
        Path(config.output_dir)
        / f"component_ranker_{stable_hash(payload)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    training_frame.to_csv(
        run_dir / "component_ranker_training_rows.csv", index=False
    )
    evaluation.to_csv(run_dir / "component_ranker_evaluation_rows.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(
        run_dir / "component_ranker_selected_rows.csv", index=False
    )
    summary.to_csv(run_dir / "component_ranker_summary.csv", index=False)
    group_summary.to_csv(
        run_dir / "component_ranker_group_summary.csv", index=False
    )
    joblib.dump(pairwise_model, run_dir / "component_pairwise_ranker.joblib")
    deployable = summary[
        summary.condition.isin({
            "heuristic_top5",
            "student_energy_top5",
            "fused_energy_top5",
            "component_pairwise_top5",
            "component_set_top5",
            "dual_support_union_8k",
            "triple_support_union_12k",
            expanded_name,
            "iterative_pairwise_then_heuristic",
            "iterative_pairwise_then_pairwise",
            "merge_pairwise_heuristic",
            "merge_pairwise_heuristic_set",
            "merge_pairwise_heuristic_iterative",
        })
    ].sort_values(
        ["hidden_recovered_n", "confidence_gain"], ascending=False
    )
    best = deployable.iloc[0]
    baseline = summary[summary.condition.eq("heuristic_top5")].iloc[0]
    (run_dir / "recommendation.md").write_text(
        "# Component-aware cluster ranker\n\n"
        f"- heuristic top-5: {int(baseline.hidden_recovered_n)}/"
        f"{int(baseline.hidden_n)} recovered\n"
        f"- best deployable: {best.condition}, "
        f"{int(best.hidden_recovered_n)}/{int(best.hidden_n)} recovered\n"
        f"- deployable gain: "
        f"{int(best.hidden_recovered_n - baseline.hidden_recovered_n)}\n",
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "elapsed_seconds": time.time() - started,
                "base_run": str(base_run),
                "student_path": str(student_path),
                "train_scenes": len(train_records),
                "test_scenes": len(test_records),
                "training_pairs": training_pair_count,
                "feature_columns": FEATURE_COLUMNS,
                "config": asdict(config),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_dir


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a component-aware within-scene cluster ranker and compare "
            "independent and coverage-aware localization on patched inputs."
        )
    )
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--ranker-run")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--student-feature-set", default="functional")
    parser.add_argument("--top-clusters", type=int, default=5)
    parser.add_argument("--candidate-budget", type=int, default=4000)
    parser.add_argument("--student-apply-top-k", type=int, default=6000)
    parser.add_argument("--expansion-factor", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = ComponentAwareRankerConfig(
        base_run=args.base_run,
        ranker_run=args.ranker_run,
        output_dir=args.output_dir,
        device=args.device,
        require_device=args.require_device,
        student_feature_set=args.student_feature_set,
        top_clusters=args.top_clusters,
        candidate_budget=args.candidate_budget,
        student_apply_top_k=args.student_apply_top_k,
        expansion_factor=args.expansion_factor,
    )
    if args.smoke:
        config.max_train_scenes = 4
        config.max_test_scenes = 2
        config.pair_samples_per_scene = 40
        config.pairwise_max_iter = 12
    print(run_component_aware_ranker(config))


if __name__ == "__main__":
    main()
