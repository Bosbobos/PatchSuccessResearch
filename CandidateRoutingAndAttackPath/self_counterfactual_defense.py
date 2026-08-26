from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .candidate_reserve import _cache_lookup, _evaluate_batch
from .candidate_routing import _box_iou, _flat_location, _level_slices, _xywh_to_xyxy
from .common import StorageBudget, load_experiment, release_accelerator_memory, stable_hash
from .followup_common import (
    ATTACK_PATH_DB,
    FOLLOWUP_DIR,
    MANIFEST_CSV,
    TRACE_DB,
    balanced_subset,
    write_summary,
)
from .full_success_closure import _functional_components
from .causal_repair import _load_inputs
from .mechanism_followup import _decode, _head_branches


@dataclass(slots=True)
class SelfCounterfactualDefenseConfig:
    output_dir: str = str(FOLLOWUP_DIR)
    device: str = "mps"
    require_device: bool = True
    examples_per_group: int = 25
    path_steps: int = 5
    proxy_kinds: tuple[str, ...] = ("gray", "context_mean", "blur")
    proxy_strength: float = 1.0
    repair_scales: tuple[float, ...] = (1.0,)
    blur_kernel: int = 41
    context_width: int = 24
    target_iou: float = 0.50
    detection_conf: float = 0.25
    candidate_min_score: float = 0.01
    candidate_top_k: int = 200
    max_candidate_routes: int = 20
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    nms_max_time_img: float = 1.0
    clean_calibration_fraction: float = 0.50
    clean_gate_quantile: float = 0.99
    min_consensus: int = 2
    blind_search: bool = False
    blind_coarse_size: int = 192
    blind_coarse_stride: int = 160
    blind_top_coarse: int = 2
    blind_top_refined: int = 5
    blind_refine_sizes: tuple[int, ...] = (128, 160, 192)
    blind_scan_batch_size: int = 24
    seed: int = 617
    max_output_gb: float = 1.0
    method_version: int = 1


def _clip_bbox(bbox, height: int, width: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Empty patch bbox after clipping: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def _counterfactual_views(image, bbox, config: SelfCounterfactualDefenseConfig):
    """Create patch-neutralized views from the observed tensor alone."""

    import torch
    import torch.nn.functional as F

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected one BCHW image, got {tuple(image.shape)}")
    _, channels, height, width = image.shape
    x1, y1, x2, y2 = _clip_bbox(bbox, int(height), int(width))
    views = {}
    for kind in config.proxy_kinds:
        value = image.clone()
        if kind == "gray":
            value[:, :, y1:y2, x1:x2] = 114.0 / 255.0
        elif kind == "context_mean":
            margin = int(config.context_width)
            rx1, ry1 = max(0, x1 - margin), max(0, y1 - margin)
            rx2, ry2 = min(int(width), x2 + margin), min(int(height), y2 + margin)
            ring = image[:, :, ry1:ry2, rx1:rx2].clone()
            mask = torch.ones(
                (ry2 - ry1, rx2 - rx1), dtype=torch.bool, device=image.device
            )
            mask[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
            pixels = ring[0, :, mask]
            fill = (
                pixels.mean(dim=1)
                if pixels.numel()
                else torch.full((channels,), 114.0 / 255.0, device=image.device)
            )
            value[:, :, y1:y2, x1:x2] = fill.reshape(1, channels, 1, 1)
        elif kind == "blur":
            kernel = int(config.blur_kernel)
            if kernel % 2 == 0:
                kernel += 1
            blurred = F.avg_pool2d(image, kernel_size=kernel, stride=1, padding=kernel // 2)
            value[:, :, y1:y2, x1:x2] = blurred[:, :, y1:y2, x1:x2]
        else:
            raise ValueError(f"Unknown proxy kind: {kind}")
        strength = float(config.proxy_strength)
        value = image + strength * (value - image)
        views[str(kind)] = value
    return views


def _decoded_frame(detect, raw, class_id: int) -> pd.DataFrame:
    decoded = _decode(detect, raw)
    boxes = _xywh_to_xyxy(decoded[0, :4].transpose(0, 1))
    scores = decoded[0, 4 + int(class_id)]
    slices = _level_slices(raw)
    top_k = min(int(scores.numel()), 1000)
    indices = scores.topk(top_k).indices.detach().cpu().numpy()
    box_values = boxes[indices].detach().float().cpu().numpy()
    score_values = scores[indices].detach().float().cpu().numpy()
    records = []
    for flat, score, box in zip(indices, score_values, box_values, strict=True):
        level, y, x = _flat_location(int(flat), slices)
        records.append({
            "flat_index": int(flat),
            "level_index": int(level),
            "y_index": int(y),
            "x_index": int(x),
            "score": float(score),
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
        })
    return pd.DataFrame(records)


def _max_matching_score(frame: pd.DataFrame, box: np.ndarray, iou_threshold: float) -> float:
    if frame.empty:
        return 0.0
    boxes = frame[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    x1 = np.maximum(boxes[:, 0], box[0])
    y1 = np.maximum(boxes[:, 1], box[1])
    x2 = np.minimum(boxes[:, 2], box[2])
    y2 = np.minimum(boxes[:, 3], box[3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    area_b = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    iou = intersection / np.maximum(area_a + area_b - intersection, 1e-12)
    matched = frame.loc[iou >= float(iou_threshold), "score"]
    return float(matched.max()) if len(matched) else 0.0


def _discover_proxy_target(
    observed: pd.DataFrame,
    proxies: dict[str, pd.DataFrame],
    config: SelfCounterfactualDefenseConfig,
) -> dict:
    """Select the person candidate with the largest counterfactual score gain."""

    best = None
    for proxy_kind, frame in proxies.items():
        for item in frame.head(int(config.candidate_top_k)).itertuples(index=False):
            if float(item.score) < float(config.candidate_min_score):
                continue
            box = np.asarray([item.x1, item.y1, item.x2, item.y2], dtype=np.float32)
            observed_score = _max_matching_score(observed, box, config.target_iou)
            gain = float(item.score) - observed_score
            candidate = {
                "proxy_kind": str(proxy_kind),
                "flat_index": int(item.flat_index),
                "level_index": int(item.level_index),
                "y_index": int(item.y_index),
                "x_index": int(item.x_index),
                "proxy_score": float(item.score),
                "observed_match_score": float(observed_score),
                "proxy_gain": float(gain),
                "box": box,
            }
            if best is None or (
                candidate["proxy_gain"], candidate["proxy_score"]
            ) > (best["proxy_gain"], best["proxy_score"]):
                best = candidate
    if best is None:
        raise RuntimeError("No proxy person candidate passed candidate_min_score")
    scores = {
        kind: _max_matching_score(frame, best["box"], config.target_iou)
        for kind, frame in proxies.items()
    }
    best["proxy_scores_json"] = json.dumps(scores, sort_keys=True)
    best["consensus_count"] = int(
        sum(score >= float(config.detection_conf) for score in scores.values())
    )
    best["proxy_score_std"] = float(np.std(list(scores.values())))
    return best


def _axis_positions(length: int, size: int, stride: int) -> list[int]:
    size = min(int(size), int(length))
    last = max(0, int(length) - size)
    positions = list(range(0, last + 1, max(1, int(stride))))
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def _coarse_windows(
    height: int,
    width: int,
    config: SelfCounterfactualDefenseConfig,
) -> list[tuple[int, int, int, int]]:
    size = min(int(config.blind_coarse_size), int(height), int(width))
    xs = _axis_positions(width, size, int(config.blind_coarse_stride))
    ys = _axis_positions(height, size, int(config.blind_coarse_stride))
    return [(x, y, x + size, y + size) for y in ys for x in xs]


def _refined_windows(
    coarse: tuple[int, int, int, int],
    height: int,
    width: int,
    config: SelfCounterfactualDefenseConfig,
) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = coarse
    windows = set()
    for requested_size in config.blind_refine_sizes:
        size = min(int(requested_size), int(height), int(width))
        xs = {
            max(0, min(width - size, x1)),
            max(0, min(width - size, int(round((x1 + x2 - size) / 2)))),
            max(0, min(width - size, x2 - size)),
        }
        ys = {
            max(0, min(height - size, y1)),
            max(0, min(height - size, int(round((y1 + y2 - size) / 2)))),
            max(0, min(height - size, y2 - size)),
        }
        windows.update((x, y, x + size, y + size) for y in ys for x in xs)
    return sorted(windows)


def _window_gain_batch(
    detect,
    observed_raw,
    view_raw,
    class_id: int,
    config: SelfCounterfactualDefenseConfig,
) -> list[tuple[float, float]]:
    """Maximum matched person-score gain for each counterfactual window."""

    import torch

    with torch.inference_mode():
        observed_decoded = _decode(detect, observed_raw)
        view_decoded = _decode(detect, view_raw)
        observed_scores = observed_decoded[0, 4 + int(class_id)]
        observed_boxes = _xywh_to_xyxy(observed_decoded[0, :4].transpose(0, 1))
        observed_k = min(1000, int(observed_scores.numel()))
        observed_indices = observed_scores.topk(observed_k).indices
        observed_scores = observed_scores[observed_indices]
        observed_boxes = observed_boxes[observed_indices]
        results = []
        for index in range(int(view_decoded.shape[0])):
            scores = view_decoded[index, 4 + int(class_id)]
            boxes = _xywh_to_xyxy(view_decoded[index, :4].transpose(0, 1))
            top_k = min(int(config.candidate_top_k), int(scores.numel()))
            indices = scores.topk(top_k).indices
            scores = scores[indices]
            boxes = boxes[indices]
            iou = _box_iou(boxes, observed_boxes)
            matched = torch.where(
                iou >= float(config.target_iou),
                observed_scores.unsqueeze(0),
                torch.zeros((), dtype=observed_scores.dtype, device=observed_scores.device),
            ).amax(dim=1)
            gains = scores - matched
            valid = scores >= float(config.candidate_min_score)
            if bool(valid.any()):
                valid_gains = torch.where(
                    valid, gains, torch.full_like(gains, float("-inf"))
                )
                winner = int(valid_gains.argmax())
                results.append((float(gains[winner].cpu()), float(scores[winner].cpu())))
            else:
                results.append((float("-inf"), 0.0))
    return results


def _blind_window_search(
    model,
    detect,
    image,
    class_id: int,
    config: SelfCounterfactualDefenseConfig,
) -> dict:
    """Choose a suspicious region using only the single observed image."""

    import torch

    _, _, height, width = image.shape
    height, width = int(height), int(width)
    scan_config = replace(config, proxy_kinds=("gray",), proxy_strength=1.0)
    observed_inputs = _capture_detect_inputs(model, detect, image)
    with torch.inference_mode():
        _box, _cls, observed_raw = _head_branches(detect, observed_inputs)

    def score_windows(windows):
        scored = []
        batch_size = max(1, int(config.blind_scan_batch_size))
        for start in range(0, len(windows), batch_size):
            current = windows[start:start + batch_size]
            views = [
                _counterfactual_views(image, bbox, scan_config)["gray"]
                for bbox in current
            ]
            captured = _capture_detect_inputs(model, detect, torch.cat(views, dim=0))
            with torch.inference_mode():
                _view_box, _view_cls, view_raw = _head_branches(detect, captured)
            scores = _window_gain_batch(
                detect, observed_raw, view_raw, int(class_id), config
            )
            scored.extend(
                {
                    "bbox": tuple(int(value) for value in bbox),
                    "search_gain": float(gain),
                    "search_proxy_score": float(proxy_score),
                }
                for bbox, (gain, proxy_score) in zip(current, scores, strict=True)
            )
            del captured, view_raw, views
            release_accelerator_memory()
        return scored

    coarse_scores = score_windows(_coarse_windows(height, width, config))
    top_coarse = sorted(
        coarse_scores,
        key=lambda item: (item["search_gain"], item["search_proxy_score"]),
        reverse=True,
    )[: max(1, int(config.blind_top_coarse))]
    refined = set()
    for item in top_coarse:
        refined.update(_refined_windows(item["bbox"], height, width, config))
    refine_scores = score_windows(sorted(refined))
    all_scores = coarse_scores + refine_scores
    ordered = sorted(
        all_scores,
        key=lambda item: (item["search_gain"], item["search_proxy_score"]),
        reverse=True,
    )
    finalists = []
    seen_boxes = set()
    for item in ordered:
        if item["bbox"] in seen_boxes:
            continue
        finalists.append(item)
        seen_boxes.add(item["bbox"])
        if len(finalists) >= max(1, int(config.blind_top_refined)):
            break
    finalist_images = []
    finalist_names = []
    for item in finalists:
        views = _counterfactual_views(image, item["bbox"], config)
        finalist_images.extend(views.values())
        finalist_names.append(list(views))
    captured = _capture_detect_inputs(
        model, detect, torch.cat(finalist_images, dim=0)
    )
    with torch.inference_mode():
        _final_box, _final_cls, final_raw = _head_branches(detect, captured)
    observed_frame = _decoded_frame(detect, observed_raw, int(class_id))
    rescored = []
    offset = 0
    for item, names in zip(finalists, finalist_names, strict=True):
        frames = {}
        for name in names:
            current_raw = [level[offset:offset + 1] for level in final_raw]
            frames[name] = _decoded_frame(detect, current_raw, int(class_id))
            offset += 1
        target = _discover_proxy_target(observed_frame, frames, config)
        rescored.append({
            **item,
            "search_gain": float(target["proxy_gain"]),
            "search_proxy_score": float(target["proxy_score"]),
            "search_consensus_count": int(target["consensus_count"]),
            "search_proxy_score_std": float(target["proxy_score_std"]),
        })
    rescored.sort(
        key=lambda item: (
            item["search_consensus_count"],
            item["search_gain"],
            item["search_proxy_score"],
            -item["search_proxy_score_std"],
        ),
        reverse=True,
    )
    best = dict(rescored[0])
    second_gain = (
        float(rescored[1]["search_gain"]) if len(rescored) > 1 else float("-inf")
    )
    best["search_second_gain"] = second_gain
    best["search_margin"] = float(best["search_gain"] - second_gain)
    best["search_windows"] = int(len(all_scores))
    best["search_unique_windows"] = int(len({item["bbox"] for item in all_scores}))
    best["search_finalists"] = int(len(finalists))
    del observed_inputs, observed_raw, captured, final_raw, finalist_images
    release_accelerator_memory()
    return best


def _bbox_evaluation(
    selected_bbox,
    reference_bbox,
    height: int,
    width: int,
) -> dict:
    selected = _clip_bbox(selected_bbox, height, width)
    reference = _clip_bbox(reference_bbox, height, width)
    ax1, ay1, ax2, ay2 = selected
    bx1, by1, bx2, by2 = reference
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - intersection
    center_x, center_y = (bx1 + bx2) / 2, (by1 + by2) / 2
    return {
        "search_bbox_x1": ax1,
        "search_bbox_y1": ay1,
        "search_bbox_x2": ax2,
        "search_bbox_y2": ay2,
        "search_patch_iou": float(intersection / max(union, 1)),
        "search_patch_coverage": float(intersection / area_b),
        "search_center_hit": int(ax1 <= center_x <= ax2 and ay1 <= center_y <= ay2),
    }


def _proxy_candidate_routes(
    observed: pd.DataFrame,
    proxy: pd.DataFrame,
    target: dict,
    config: SelfCounterfactualDefenseConfig,
) -> pd.DataFrame:
    combined = pd.concat(
        [
            observed.assign(endpoint="observed"),
            proxy.assign(endpoint="proxy"),
        ],
        ignore_index=True,
    )
    box = target["box"]
    boxes = combined[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    x1 = np.maximum(boxes[:, 0], box[0])
    y1 = np.maximum(boxes[:, 1], box[1])
    x2 = np.minimum(boxes[:, 2], box[2])
    y2 = np.minimum(boxes[:, 3], box[3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    area_b = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    iou = intersection / np.maximum(area_a + area_b - intersection, 1e-12)
    relevant = combined[
        (iou >= float(config.target_iou))
        & (combined.score >= float(config.candidate_min_score))
    ].copy()
    relevant = relevant.sort_values("score", ascending=False).drop_duplicates("flat_index")
    relevant = relevant.head(int(config.max_candidate_routes))
    if int(target["flat_index"]) not in set(relevant.flat_index.astype(int)):
        chosen = proxy[proxy.flat_index.astype(int).eq(int(target["flat_index"]))].head(1)
        relevant = pd.concat([chosen, relevant], ignore_index=True).drop_duplicates("flat_index")
    return relevant[
        ["flat_index", "level_index", "y_index", "x_index"]
    ].astype(int).reset_index(drop=True)


def _pseudo_row(target: dict, class_id: int):
    box = target["box"]
    return SimpleNamespace(
        class_id=int(class_id),
        clean_target_x1=float(box[0]),
        clean_target_y1=float(box[1]),
        clean_target_x2=float(box[2]),
        clean_target_y2=float(box[3]),
        clean_target_flat=int(target["flat_index"]),
    )


def _intervention_raw(detect, observed_inputs, proxy_inputs, maps):
    import torch

    conditions = {
        "observed": observed_inputs,
        "proxy_view": proxy_inputs,
        "full_window_repair": [
            observed
            - maps["full_candidate_windows"][level].to(observed.device, observed.dtype).unsqueeze(0)
            for level, observed in enumerate(observed_inputs)
        ],
    }
    for scale in maps["_repair_scales"]:
        label = f"joint_repair_x{float(scale):g}"
        conditions[label] = [
            observed
            - float(scale)
            * maps["joint_rowspace"][level].to(observed.device, observed.dtype).unsqueeze(0)
            for level, observed in enumerate(observed_inputs)
        ]
    names = list(conditions)
    batched = [
        torch.cat([conditions[name][level] for name in names], dim=0)
        for level in range(len(observed_inputs))
    ]
    with torch.inference_mode():
        _box, _cls, raw = _head_branches(detect, batched)
    return names, raw


def _all_class_nms(detect, raw, config: SelfCounterfactualDefenseConfig):
    from ultralytics.utils.nms import non_max_suppression

    decoded = _decode(detect, raw)
    return non_max_suppression(
        decoded.detach().clone(),
        conf_thres=float(config.detection_conf),
        iou_thres=float(config.nms_iou),
        max_det=int(config.nms_max_det),
        nc=int(detect.nc),
        max_time_img=float(config.nms_max_time_img),
    )


def _detection_set_metrics(reference, candidate, iou_threshold: float = 0.50) -> dict:
    """Match all post-NMS detections by class and IoU to a clean reference."""

    import torch

    reference = reference.detach()
    candidate = candidate.detach()
    matched_reference = set()
    matched_candidate = set()
    pairs = []
    for class_id in sorted(
        set(reference[:, 5].int().tolist()).union(candidate[:, 5].int().tolist())
    ):
        ref_indices = torch.nonzero(
            reference[:, 5].int().eq(int(class_id)), as_tuple=False
        ).reshape(-1)
        cand_indices = torch.nonzero(
            candidate[:, 5].int().eq(int(class_id)), as_tuple=False
        ).reshape(-1)
        if not len(ref_indices) or not len(cand_indices):
            continue
        iou = _box_iou(reference[ref_indices, :4], candidate[cand_indices, :4])
        while iou.numel():
            flat = int(torch.argmax(iou))
            value = float(iou.reshape(-1)[flat].cpu())
            if value < float(iou_threshold):
                break
            row = flat // int(iou.shape[1])
            col = flat % int(iou.shape[1])
            ref_index = int(ref_indices[row])
            cand_index = int(cand_indices[col])
            matched_reference.add(ref_index)
            matched_candidate.add(cand_index)
            pairs.append((ref_index, cand_index))
            iou[row, :] = -1
            iou[:, col] = -1
    recall = len(matched_reference) / max(len(reference), 1)
    precision = len(matched_candidate) / max(len(candidate), 1)
    f1 = 2.0 * recall * precision / max(recall + precision, 1e-12)
    confidence_mae = (
        float(np.mean([
            abs(float(reference[i, 4].cpu()) - float(candidate[j, 4].cpu()))
            for i, j in pairs
        ]))
        if pairs else np.nan
    )
    return {
        "reference_count": int(len(reference)),
        "candidate_count": int(len(candidate)),
        "matched_count": int(len(pairs)),
        "detection_recall": float(recall),
        "detection_precision": float(precision),
        "detection_f1": float(f1),
        "matched_confidence_mae": confidence_mae,
    }


def _build_gate_tables(rows: pd.DataFrame, config: SelfCounterfactualDefenseConfig):
    clean = rows[rows.input_kind.eq("clean")].copy()
    rng = np.random.default_rng(int(config.seed) + 901)
    ids = clean.example_id.drop_duplicates().to_numpy()
    rng.shuffle(ids)
    split = max(1, min(len(ids) - 1, int(round(len(ids) * config.clean_calibration_fraction))))
    calibration_ids = set(ids[:split])
    rows = rows.copy()
    rows["clean_split"] = np.where(
        rows.input_kind.ne("clean"),
        "not_clean",
        np.where(rows.example_id.isin(calibration_ids), "calibration", "test"),
    )
    calibration = rows[
        rows.input_kind.eq("clean") & rows.clean_split.eq("calibration")
    ]
    metrics = ["proxy_gain", "repair_gain", "conservative_gain"]
    if bool(config.blind_search):
        oriented = {
            "low_proxy_std": -np.log10(rows.proxy_score_std.clip(lower=1e-5)),
            "low_search_margin": -np.log10(rows.search_margin.clip(lower=0) + 1e-4),
            "high_proxy_score": rows.proxy_score,
            "high_repair_gain": rows.repair_gain,
        }
        calibration_mask = (
            rows.input_kind.eq("clean") & rows.clean_split.eq("calibration")
        )
        composite = pd.Series(0.0, index=rows.index)
        for values in oriented.values():
            clean_values = values[calibration_mask]
            median = float(clean_values.median())
            iqr = max(
                float(clean_values.quantile(0.75) - clean_values.quantile(0.25)),
                1e-6,
            )
            composite += (values - median) / iqr
        rows["self_consistency_anomaly"] = composite
        calibration = rows[calibration_mask]
        metrics.append("self_consistency_anomaly")
    thresholds = []
    decisions = []
    for metric in metrics:
        threshold = float(calibration[metric].quantile(float(config.clean_gate_quantile)))
        thresholds.append({
            "gate_metric": metric,
            "threshold": threshold,
            "clean_quantile": float(config.clean_gate_quantile),
            "min_consensus": int(config.min_consensus),
            "calibration_examples": int(calibration.example_id.nunique()),
            "calibration_scope": "clean_only",
        })
        gate = (
            (rows[metric] > threshold)
            & (rows.consensus_count >= int(config.min_consensus))
            & (rows.pseudo_repair_detected.astype(bool))
        )
        current = rows.copy()
        current["gate_metric"] = metric
        current["gate_threshold"] = threshold
        current["gate_applied"] = gate.astype(int)
        current["defended_target_detected"] = np.where(
            gate,
            current.joint_repair_target_detected,
            current.observed_target_detected,
        ).astype(int)
        current["proxy_defended_target_detected"] = np.where(
            gate,
            current.proxy_view_target_detected,
            current.observed_target_detected,
        ).astype(int)
        decisions.append(current)
    return rows, pd.DataFrame(thresholds), pd.concat(decisions, ignore_index=True)


def _summarize_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = ["gate_metric", "input_kind", "clean_split", "analysis_group"]
    groups = list(decisions.groupby(keys, sort=False, dropna=False))
    groups += [
        ((metric, kind, split, "all"), frame)
        for (metric, kind, split), frame in decisions.groupby(
            ["gate_metric", "input_kind", "clean_split"], sort=False, dropna=False
        )
    ]
    for key, frame in groups:
        source_hidden = frame.source_hidden.astype(bool)
        source_visible = ~source_hidden
        records.append({
            **dict(zip(keys, key, strict=True)),
            "n": int(frame.example_id.nunique()),
            "gate_rate": float(frame.gate_applied.mean()),
            "source_hidden_n": int(source_hidden.sum()),
            "repair_rate": (
                float(frame.loc[source_hidden, "defended_target_detected"].mean())
                if source_hidden.any() else np.nan
            ),
            "proxy_repair_rate": (
                float(frame.loc[source_hidden, "proxy_defended_target_detected"].mean())
                if source_hidden.any() else np.nan
            ),
            "source_visible_n": int(source_visible.sum()),
            "visible_preservation_rate": (
                float(frame.loc[source_visible, "defended_target_detected"].mean())
                if source_visible.any() else np.nan
            ),
            "clean_harm_rate": (
                float(1.0 - frame.defended_target_detected.mean())
                if str(key[1]) == "clean" else np.nan
            ),
        })
    return pd.DataFrame(records)


def run_self_counterfactual_defense(
    config: SelfCounterfactualDefenseConfig | None = None,
) -> Path:
    config = config or SelfCounterfactualDefenseConfig()
    started = time.time()
    StorageBudget(config.output_dir, config.max_output_gb).check()
    selected, _ = _load_inputs(Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None)
    selected = balanced_subset(selected, config.examples_per_group, seed=config.seed)
    exp, cache_path = load_experiment(
        prefer_device=config.device, require_device=bool(config.require_device)
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    cache = _cache_lookup(exp)
    records = []
    progress = tqdm(
        selected.itertuples(index=False),
        total=len(selected),
        desc="self-counterfactual defense",
        unit="example",
    )
    for row in progress:
        example_id = str(row.example_id)
        example = cache[example_id]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        clean_reference = None
        for input_kind, observed in (("clean", pair[0:1]), ("patched", pair[1:2])):
            import torch

            if config.blind_search:
                search = _blind_window_search(
                    model, detect, observed, int(row.class_id), config
                )
                proxy_bbox = search["bbox"]
            else:
                proxy_bbox = example.patch_bbox_lb
                search = {
                    "bbox": tuple(int(round(float(value))) for value in proxy_bbox),
                    "search_gain": np.nan,
                    "search_proxy_score": np.nan,
                    "search_second_gain": np.nan,
                    "search_margin": np.nan,
                    "search_windows": 0,
                    "search_unique_windows": 0,
                    "search_finalists": 0,
                    "search_consensus_count": np.nan,
                    "search_proxy_score_std": np.nan,
                }
            views = _counterfactual_views(observed, proxy_bbox, config)
            image_names = ["observed", *views]
            image_batch = torch.cat([observed, *views.values()], dim=0)
            captured = _capture_detect_inputs(model, detect, image_batch)
            with torch.inference_mode():
                all_box, all_cls, all_raw = _head_branches(detect, captured)
            raw_by_name = {
                name: [level[index:index + 1] for level in all_raw]
                for index, name in enumerate(image_names)
            }
            frames = {
                name: _decoded_frame(detect, raw, int(row.class_id))
                for name, raw in raw_by_name.items()
            }
            target = _discover_proxy_target(
                frames["observed"],
                {name: frames[name] for name in views},
                config,
            )
            proxy_kind = target["proxy_kind"]
            proxy_index = image_names.index(proxy_kind)
            observed_inputs = [level[0:1] for level in captured]
            proxy_inputs = [level[proxy_index:proxy_index + 1] for level in captured]
            observed_box = [level[0:1] for level in all_box]
            observed_cls = [level[0:1] for level in all_cls]
            proxy_box = [level[proxy_index:proxy_index + 1] for level in all_box]
            proxy_cls = [level[proxy_index:proxy_index + 1] for level in all_cls]
            selection = _proxy_candidate_routes(
                frames["observed"], frames[proxy_kind], target, config
            )
            pseudo = _pseudo_row(target, int(row.class_id))
            maps, energy, metadata = _functional_components(
                detect,
                proxy_inputs,
                observed_inputs,
                proxy_box,
                proxy_cls,
                observed_box,
                observed_cls,
                selection,
                pseudo,
                config,
            )
            maps["_repair_scales"] = tuple(sorted(float(value) for value in config.repair_scales))
            names, intervention_raw = _intervention_raw(detect, observed_inputs, proxy_inputs, maps)
            with torch.inference_mode():
                actual = {
                    name: result
                    for name, result in zip(
                        names,
                        _evaluate_batch(detect, intervention_raw, row, config),
                        strict=True,
                    )
                }
                pseudo_results = {
                    name: result
                    for name, result in zip(
                        names,
                        _evaluate_batch(detect, intervention_raw, pseudo, config),
                        strict=True,
                    )
                }
                all_detections = {
                    name: detections
                    for name, detections in zip(
                        names, _all_class_nms(detect, intervention_raw, config), strict=True
                    )
                }
            if input_kind == "clean":
                clean_reference = all_detections["observed"]
            if clean_reference is None:
                raise RuntimeError("Clean reference must be evaluated before patched input")
            repair_conditions = [
                f"joint_repair_x{float(scale):g}" for scale in maps["_repair_scales"]
            ]
            detected_repairs = [
                name for name in repair_conditions
                if pseudo_results[name]["target_detected"]
            ]
            if detected_repairs:
                chosen_repair = detected_repairs[0]
            else:
                chosen_repair = max(
                    repair_conditions,
                    key=lambda name: pseudo_results[name]["post_target_conf"],
                )
            actual["joint_repair"] = actual[chosen_repair]
            pseudo_results["joint_repair"] = pseudo_results[chosen_repair]
            record = {
                "example_id": example_id,
                "analysis_group": str(row.analysis_group),
                "input_kind": input_kind,
                "source_hidden": int(
                    actual["observed"]["target_hidden"] if input_kind == "patched" else 0
                ),
                "proxy_kind": proxy_kind,
                "proxy_gain": float(target["proxy_gain"]),
                "proxy_score": float(target["proxy_score"]),
                "observed_match_score": float(target["observed_match_score"]),
                "consensus_count": int(target["consensus_count"]),
                "proxy_score_std": float(target["proxy_score_std"]),
                "proxy_scores_json": target["proxy_scores_json"],
                "blind_search": int(bool(config.blind_search)),
                "search_gain": float(search["search_gain"]),
                "search_proxy_score": float(search["search_proxy_score"]),
                "search_second_gain": float(search["search_second_gain"]),
                "search_margin": float(search["search_margin"]),
                "search_windows": int(search["search_windows"]),
                "search_unique_windows": int(search["search_unique_windows"]),
                "search_finalists": int(search["search_finalists"]),
                "search_consensus_count": float(search["search_consensus_count"]),
                "search_proxy_score_std": float(search["search_proxy_score_std"]),
                **_bbox_evaluation(
                    proxy_bbox,
                    example.patch_bbox_lb,
                    int(observed.shape[-2]),
                    int(observed.shape[-1]),
                ),
                "candidate_routes": int(len(selection)),
                "chosen_repair_condition": chosen_repair,
                "chosen_repair_scale": float(chosen_repair.rsplit("x", 1)[1]),
                "repair_gain": float(
                    pseudo_results["joint_repair"]["post_target_conf"]
                    - pseudo_results["observed"]["post_target_conf"]
                ),
                "conservative_gain": float(
                    min(
                        target["proxy_gain"],
                        pseudo_results["joint_repair"]["post_target_conf"]
                        - pseudo_results["observed"]["post_target_conf"],
                    )
                ),
                "pseudo_observed_conf": float(pseudo_results["observed"]["post_target_conf"]),
                "pseudo_proxy_conf": float(pseudo_results["proxy_view"]["post_target_conf"]),
                "pseudo_repair_conf": float(pseudo_results["joint_repair"]["post_target_conf"]),
                "pseudo_repair_detected": int(pseudo_results["joint_repair"]["target_detected"]),
                "joint_energy_fraction": float(energy["joint_rowspace"]),
                **metadata,
            }
            for condition in names:
                for key in (
                    "target_detected", "target_hidden", "post_target_conf",
                    "post_target_iou", "pre_target_conf", "nms_only_hidden",
                ):
                    record[f"{condition}_{key}"] = actual[condition][key]
                fidelity = _detection_set_metrics(
                    clean_reference, all_detections[condition], config.target_iou
                )
                for key, value in fidelity.items():
                    record[f"{condition}_{key}"] = value
            for key in (
                "target_detected", "target_hidden", "post_target_conf",
                "post_target_iou", "pre_target_conf", "nms_only_hidden",
            ):
                record[f"joint_repair_{key}"] = actual["joint_repair"][key]
            joint_fidelity = _detection_set_metrics(
                clean_reference, all_detections[chosen_repair], config.target_iou
            )
            for key, value in joint_fidelity.items():
                record[f"joint_repair_{key}"] = value
            records.append(record)
            release_accelerator_memory()

    rows = pd.DataFrame(records)
    rows, thresholds, decisions = _build_gate_tables(rows, config)
    decision_summary = _summarize_decisions(decisions)
    payload = {**asdict(config), "example_ids": selected.example_id.astype(str).tolist()}
    run_dir = Path(config.output_dir) / f"self_counterfactual_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(run_dir / "self_counterfactual_rows.csv", index=False)
    thresholds.to_csv(run_dir / "gate_thresholds.csv", index=False)
    decisions.to_csv(run_dir / "gate_decisions.csv", index=False)
    decision_summary.to_csv(run_dir / "gate_summary.csv", index=False)
    elapsed = time.time() - started
    summary = {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "examples": int(selected.example_id.nunique()),
        "evaluated_inputs": int(len(rows)),
        "patched_hidden_examples": int(
            rows.loc[rows.input_kind.eq("patched"), "source_hidden"].sum()
        ),
        "cache_path": str(cache_path),
        "config": asdict(config),
        "limitations": [
            (
                "The blind estimator uses neither the paired clean image nor the patch bbox; "
                "the saved true-box overlap is evaluation-only."
                if config.blind_search
                else "The estimator never uses the paired clean image, but this first stage uses the known patch bbox."
            ),
            "Joint repair requires Detect-input access and gradients; it is no-clean gray-box, not strict output-only black-box.",
            "The clean gate is calibrated only on held-out clean inputs; attack labels are not used for thresholds.",
            "Full-output clean fidelity is evaluated, but a larger clean calibration/test set is required.",
            "Current cached attacks place the patch in one fixed location, so localization generalization requires a new varied-location dataset.",
        ],
    }
    write_summary(run_dir / "summary.json", summary)
    digest = [
        "# Self-counterfactual no-clean defense pilot",
        "",
        f"- elapsed: {elapsed:.1f} s",
        f"- examples: {summary['examples']}",
        f"- evaluated inputs: {summary['evaluated_inputs']}",
        f"- patched endpoint-hidden: {summary['patched_hidden_examples']}",
    ]
    for item in decision_summary[
        decision_summary.analysis_group.eq("all")
        & decision_summary.gate_metric.eq("conservative_gain")
    ].itertuples(index=False):
        digest.append(
            f"- conservative gate / {item.input_kind} / {item.clean_split}: "
            f"gate={item.gate_rate:.3f}, repair={item.repair_rate}, "
            f"preserve={item.visible_preservation_rate}, clean_harm={item.clean_harm_rate}"
        )
    (run_dir / "analysis_digest.md").write_text("\n".join(digest) + "\n", encoding="utf-8")
    StorageBudget(config.output_dir, config.max_output_gb).check()
    return run_dir


if __name__ == "__main__":
    print(run_self_counterfactual_defense())
