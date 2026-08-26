from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_MAX_OUTPUT_GB,
    DEFAULT_OUTPUT_DIR,
    StorageBudget,
    connect_db,
    release_accelerator_memory,
    stable_hash,
    upsert_metadata,
    write_json,
    write_markdown,
)


@dataclass(slots=True)
class CandidateTraceConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    max_output_gb: float = DEFAULT_MAX_OUTPUT_GB
    max_examples: int | None = None
    batch_size: int = 8
    top_k: int = 50
    lineage_top_k: int = 20
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    nms_survival_iou: float = 0.99
    same_object_iou: float = 0.50
    replacement_iou: float = 0.10
    progress: bool = True
    method_version: int = 4


@dataclass(slots=True)
class CandidateTraceResult:
    run_dir: Path
    db_path: Path
    summary_path: Path
    digest_path: Path
    config: CandidateTraceConfig


EXAMPLE_COLUMNS = (
    "example_id", "path", "success", "confidence_drop", "conf_clean_cached", "conf_patch_cached",
    "conf_clean_recomputed", "conf_patch_recomputed", "clean_conf_abs_error", "patch_conf_abs_error",
    "class_id", "patch_x1", "patch_y1", "patch_x2", "patch_y2",
    "clean_target_flat", "clean_target_level", "clean_target_y", "clean_target_x",
    "clean_target_score", "clean_target_logit", "clean_target_x1", "clean_target_y1",
    "clean_target_x2", "clean_target_y2", "clean_target_nms_survived",
    "patched_tracked_score", "patched_tracked_logit", "patched_tracked_x1", "patched_tracked_y1",
    "patched_tracked_x2", "patched_tracked_y2", "patched_tracked_nms_survived",
    "tracked_score_delta", "tracked_logit_delta", "tracked_box_iou_clean",
    "patched_winner_flat", "patched_winner_level", "patched_winner_y", "patched_winner_x",
    "patched_winner_score", "patched_winner_logit", "patched_winner_x1", "patched_winner_y1",
    "patched_winner_x2", "patched_winner_y2", "winner_iou_clean", "winner_same_flat",
    "winner_level_changed", "patched_winner_minus_tracked", "mechanism_mode", "error",
)

CANDIDATE_COLUMNS = (
    "example_id", "variant", "rank", "flat_index", "level_index", "level_name", "y_index", "x_index",
    "stride", "decoded_score", "class_logit", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "bbox_area", "center_x", "center_y", "dfl_left", "dfl_top", "dfl_right", "dfl_bottom",
    "nms_max_iou", "nms_survived", "nms_rank", "nms_conf",
)

LINEAGE_COLUMNS = (
    "example_id", "clean_rank", "patched_rank", "clean_flat", "patched_flat", "clean_level",
    "patched_level", "same_flat", "level_changed", "bbox_iou", "clean_score", "patched_score",
    "score_delta", "center_distance", "match_cost",
)


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS examples (
            example_id TEXT PRIMARY KEY, path TEXT, success INTEGER, confidence_drop REAL,
            conf_clean_cached REAL, conf_patch_cached REAL, conf_clean_recomputed REAL,
            conf_patch_recomputed REAL, clean_conf_abs_error REAL, patch_conf_abs_error REAL,
            class_id INTEGER, patch_x1 REAL, patch_y1 REAL, patch_x2 REAL, patch_y2 REAL,
            clean_target_flat INTEGER, clean_target_level INTEGER, clean_target_y INTEGER, clean_target_x INTEGER,
            clean_target_score REAL, clean_target_logit REAL, clean_target_x1 REAL, clean_target_y1 REAL,
            clean_target_x2 REAL, clean_target_y2 REAL, clean_target_nms_survived INTEGER,
            patched_tracked_score REAL, patched_tracked_logit REAL, patched_tracked_x1 REAL,
            patched_tracked_y1 REAL, patched_tracked_x2 REAL, patched_tracked_y2 REAL,
            patched_tracked_nms_survived INTEGER, tracked_score_delta REAL, tracked_logit_delta REAL,
            tracked_box_iou_clean REAL, patched_winner_flat INTEGER, patched_winner_level INTEGER,
            patched_winner_y INTEGER, patched_winner_x INTEGER, patched_winner_score REAL,
            patched_winner_logit REAL, patched_winner_x1 REAL, patched_winner_y1 REAL,
            patched_winner_x2 REAL, patched_winner_y2 REAL, winner_iou_clean REAL,
            winner_same_flat INTEGER, winner_level_changed INTEGER, patched_winner_minus_tracked REAL,
            mechanism_mode TEXT, error TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            example_id TEXT, variant TEXT, rank INTEGER, flat_index INTEGER, level_index INTEGER,
            level_name TEXT, y_index INTEGER, x_index INTEGER, stride REAL, decoded_score REAL,
            class_logit REAL, bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
            bbox_area REAL, center_x REAL, center_y REAL, dfl_left REAL, dfl_top REAL,
            dfl_right REAL, dfl_bottom REAL, nms_max_iou REAL, nms_survived INTEGER,
            nms_rank INTEGER, nms_conf REAL,
            PRIMARY KEY(example_id, variant, rank)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lineage (
            example_id TEXT, clean_rank INTEGER, patched_rank INTEGER, clean_flat INTEGER,
            patched_flat INTEGER, clean_level INTEGER, patched_level INTEGER, same_flat INTEGER,
            level_changed INTEGER, bbox_iou REAL, clean_score REAL, patched_score REAL,
            score_delta REAL, center_distance REAL, match_cost REAL,
            PRIMARY KEY(example_id, clean_rank)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_examples_mode ON examples(mechanism_mode, success)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_candidates_flat ON candidates(example_id, variant, flat_index)")
    conn.commit()


def _insert_rows(conn: sqlite3.Connection, table: str, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    placeholders = ",".join("?" for _ in columns)
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    conn.executemany(sql, [[row.get(column) for column in columns] for row in rows])


def _parse_forward_output(out: Any):
    import torch

    if isinstance(out, (list, tuple)) and len(out) >= 2 and isinstance(out[0], torch.Tensor):
        decoded = out[0]
        raw = out[1]
    elif isinstance(out, torch.Tensor):
        raise RuntimeError("YOLO forward did not expose raw Detect levels; expected (decoded, levels).")
    else:
        raise RuntimeError(f"Unexpected YOLO forward output: {type(out)}")
    levels = [item for item in raw if isinstance(item, torch.Tensor) and item.ndim == 4]
    if not levels:
        raise RuntimeError("No 4D Detect levels found in YOLO output.")
    return decoded, levels


def _level_slices(levels: list[Any]) -> list[dict[str, int]]:
    slices = []
    offset = 0
    for level_index, level in enumerate(levels):
        h, w = int(level.shape[-2]), int(level.shape[-1])
        slices.append({"level": level_index, "h": h, "w": w, "start": offset, "end": offset + h * w})
        offset += h * w
    return slices


def _flat_location(flat_index: int, slices: list[dict[str, int]]) -> tuple[int, int, int]:
    for item in slices:
        if item["start"] <= int(flat_index) < item["end"]:
            local = int(flat_index) - item["start"]
            return int(item["level"]), int(local // item["w"]), int(local % item["w"])
    raise IndexError(flat_index)


def _xywh_to_xyxy(xywh):
    import torch

    x, y, w, h = xywh.unbind(dim=-1)
    return torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)


def _box_iou(a, b):
    import torch

    if int(a.numel()) == 0 or int(b.numel()) == 0:
        return torch.zeros((int(a.shape[0]), int(b.shape[0])), device=a.device, dtype=torch.float32)
    a = a.float()
    b = b.to(device=a.device, dtype=torch.float32)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))[None, :]
    return inter / (area_a + area_b - inter).clamp(min=1e-9)


def _candidate_from_flat(
    *,
    batch_index: int,
    flat_index: int,
    decoded,
    levels,
    slices,
    class_id: int,
    reg_max: int,
    imgsz: int,
    nms_boxes,
    nms_survival_iou: float,
) -> dict[str, Any]:
    import torch

    level_index, y_index, x_index = _flat_location(flat_index, slices)
    level = levels[level_index]
    h, w = int(level.shape[-2]), int(level.shape[-1])
    stride = float(imgsz) / float(h)
    class_channel = 4 * int(reg_max) + int(class_id)
    class_logit = float(level[batch_index, class_channel, y_index, x_index].detach().float().cpu())
    score = float(decoded[batch_index, 4 + int(class_id), flat_index].detach().float().cpu())
    box = _xywh_to_xyxy(decoded[batch_index, :4, flat_index].reshape(1, 4))[0].detach().float()
    dist = level[batch_index, : 4 * reg_max, y_index, x_index].reshape(4, reg_max).float()
    bins = torch.arange(reg_max, device=dist.device, dtype=dist.dtype)
    expected = torch.softmax(dist, dim=1).matmul(bins).detach().cpu().tolist()

    nms_rank = None
    nms_max_iou = 0.0
    nms_conf = None
    survived = False
    if nms_boxes is not None and int(nms_boxes.shape[0]) > 0:
        ious = _box_iou(box.reshape(1, 4), nms_boxes[:, :4]).reshape(-1)
        best = int(torch.argmax(ious).item())
        nms_max_iou = float(ious[best].detach().cpu())
        survived = bool(nms_max_iou >= float(nms_survival_iou))
        if survived:
            nms_rank = best
            nms_conf = float(nms_boxes[best, 4].detach().float().cpu())

    values = [float(v) for v in box.cpu().tolist()]
    return {
        "flat_index": int(flat_index),
        "level_index": int(level_index),
        "level_name": f"P{level_index + 3}",
        "y_index": int(y_index),
        "x_index": int(x_index),
        "stride": stride,
        "decoded_score": score,
        "class_logit": class_logit,
        "bbox_x1": values[0], "bbox_y1": values[1], "bbox_x2": values[2], "bbox_y2": values[3],
        "bbox_area": max(0.0, values[2] - values[0]) * max(0.0, values[3] - values[1]),
        "center_x": (values[0] + values[2]) / 2.0,
        "center_y": (values[1] + values[3]) / 2.0,
        "dfl_left": float(expected[0]), "dfl_top": float(expected[1]),
        "dfl_right": float(expected[2]), "dfl_bottom": float(expected[3]),
        "nms_max_iou": nms_max_iou,
        "nms_survived": int(survived),
        "nms_rank": nms_rank,
        "nms_conf": nms_conf,
    }


def _match_post_nms_to_flat(batch_index: int, post_box, decoded, class_id: int) -> int:
    import torch

    all_boxes = _xywh_to_xyxy(decoded[batch_index, :4, :].transpose(0, 1))
    scores = decoded[batch_index, 4 + int(class_id), :]
    ious = _box_iou(all_boxes, post_box[:4].reshape(1, 4)).reshape(-1)
    rank = ious + 0.05 * scores.clamp(0, 1)
    return int(torch.argmax(rank).item())


def _match_clean_detection(batch_index: int, target_box, decoded, class_id: int) -> int:
    import torch

    all_boxes = _xywh_to_xyxy(decoded[batch_index, :4, :].transpose(0, 1))
    scores = decoded[batch_index, 4 + int(class_id), :]
    box = torch.as_tensor(target_box, device=all_boxes.device, dtype=all_boxes.dtype).reshape(1, 4)
    ious = _box_iou(all_boxes, box).reshape(-1)
    return int(torch.argmax(ious + 0.05 * scores.clamp(0, 1)).item())


def _top_candidates(
    *, example_id: str, variant: str, batch_index: int, decoded, levels, slices,
    class_id: int, reg_max: int, imgsz: int, nms_boxes, config: CandidateTraceConfig,
) -> list[dict[str, Any]]:
    import torch

    scores = decoded[batch_index, 4 + int(class_id), :]
    k = min(int(config.top_k), int(scores.numel()))
    _values, indices = torch.topk(scores, k=k)
    rows = []
    for rank, flat_index in enumerate(indices.detach().cpu().tolist(), start=1):
        item = _candidate_from_flat(
            batch_index=batch_index, flat_index=int(flat_index), decoded=decoded, levels=levels,
            slices=slices, class_id=class_id, reg_max=reg_max, imgsz=imgsz, nms_boxes=nms_boxes,
            nms_survival_iou=config.nms_survival_iou,
        )
        item.update({"example_id": example_id, "variant": variant, "rank": rank})
        rows.append(item)
    return rows


def _candidate_box(item: dict[str, Any]) -> np.ndarray:
    return np.asarray([item["bbox_x1"], item["bbox_y1"], item["bbox_x2"], item["bbox_y2"]], dtype=np.float32)


def _numpy_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    return inter / np.maximum(1e-9, area_a[:, None] + area_b[None, :] - inter)


def _lineage_rows(example_id: str, clean: list[dict[str, Any]], patched: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    from scipy.optimize import linear_sum_assignment

    clean = clean[: int(top_k)]
    patched = patched[: int(top_k)]
    if not clean or not patched:
        return []
    clean_boxes = np.stack([_candidate_box(item) for item in clean])
    patch_boxes = np.stack([_candidate_box(item) for item in patched])
    ious = _numpy_iou_matrix(clean_boxes, patch_boxes)
    clean_scores = np.asarray([item["decoded_score"] for item in clean], dtype=np.float32)
    patch_scores = np.asarray([item["decoded_score"] for item in patched], dtype=np.float32)
    score_cost = np.abs(clean_scores[:, None] - patch_scores[None, :])
    cost = (1.0 - ious) + 0.05 * score_cost
    row_indices, col_indices = linear_sum_assignment(cost)
    out = []
    for clean_idx, patch_idx in zip(row_indices.tolist(), col_indices.tolist(), strict=True):
        c, p = clean[clean_idx], patched[patch_idx]
        center_distance = math.hypot(float(c["center_x"] - p["center_x"]), float(c["center_y"] - p["center_y"]))
        out.append({
            "example_id": example_id,
            "clean_rank": int(c["rank"]), "patched_rank": int(p["rank"]),
            "clean_flat": int(c["flat_index"]), "patched_flat": int(p["flat_index"]),
            "clean_level": int(c["level_index"]), "patched_level": int(p["level_index"]),
            "same_flat": int(c["flat_index"] == p["flat_index"]),
            "level_changed": int(c["level_index"] != p["level_index"]),
            "bbox_iou": float(ious[clean_idx, patch_idx]),
            "clean_score": float(c["decoded_score"]), "patched_score": float(p["decoded_score"]),
            "score_delta": float(p["decoded_score"] - c["decoded_score"]),
            "center_distance": center_distance, "match_cost": float(cost[clean_idx, patch_idx]),
        })
    return out


def classify_mechanism(
    *, patched_winner: dict[str, Any] | None, clean_target: dict[str, Any], patched_tracked: dict[str, Any],
    same_object_iou: float = 0.5, replacement_iou: float = 0.1,
) -> tuple[str, float, float]:
    clean_box = _candidate_box(clean_target).reshape(1, 4)
    tracked_iou = float(_numpy_iou_matrix(clean_box, _candidate_box(patched_tracked).reshape(1, 4))[0, 0])
    if patched_winner is None:
        return "no_patched_detection", float("nan"), tracked_iou
    winner_iou = float(_numpy_iou_matrix(clean_box, _candidate_box(patched_winner).reshape(1, 4))[0, 0])
    if int(patched_winner["flat_index"]) == int(clean_target["flat_index"]):
        return "same_pre_nms_candidate", winner_iou, tracked_iou
    if winner_iou >= float(same_object_iou):
        if int(patched_winner["level_index"]) != int(clean_target["level_index"]):
            return "same_object_cross_scale_reroute", winner_iou, tracked_iou
        return "same_object_same_scale_reroute", winner_iou, tracked_iou
    if bool(patched_tracked["nms_survived"]):
        return "competing_candidate_wins", winner_iou, tracked_iou
    if tracked_iou < float(same_object_iou) and winner_iou >= float(replacement_iou):
        return "geometry_or_object_shift", winner_iou, tracked_iou
    return "replacement_or_fabrication", winner_iou, tracked_iou


def _preprocess_batch(predictor, images, *, model):
    from new_experiments.patch_success_analysis.yolo import pil_to_np_bgr

    batch = predictor.preprocess([pil_to_np_bgr(image) for image in images])
    parameter = next(model.parameters())
    return batch.to(device=parameter.device, dtype=parameter.dtype)


def _forward_batch(model, inputs, *, class_id: int, config: CandidateTraceConfig, nc: int):
    import torch
    from ultralytics.utils.nms import non_max_suppression
    from segmentig_detector.yolo_utils import safe_model_forward

    with torch.inference_mode():
        decoded, levels = _parse_forward_output(safe_model_forward(model, inputs))
        # Ultralytics NMS may rewrite the box channels of its input while
        # converting xywh to xyxy.  Keep the decoded tensor pristine because
        # candidate tracing below still interprets it as xywh.
        nms = non_max_suppression(
            decoded.detach().clone(),
            conf_thres=float(config.nms_conf),
            iou_thres=float(config.nms_iou),
            classes=[int(class_id)],
            max_det=int(config.nms_max_det),
            nc=int(nc),
        )
    return decoded, levels, nms


def _example_id(example: Any) -> str:
    return stable_hash({"path": str(example.path), "drop": float(example.drop), "success": bool(example.success)})


def _pad_bbox(value) -> tuple[float | None, float | None, float | None, float | None]:
    if value is None:
        return None, None, None, None
    values = list(value)
    return tuple(float(values[idx]) if idx < len(values) else None for idx in range(4))


def _process_pair(
    *, example: Any, batch_index: int, clean_decoded, clean_levels, clean_nms,
    patched_decoded, patched_levels, patched_nms, slices, class_id: int, reg_max: int,
    imgsz: int, config: CandidateTraceConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    example_id = _example_id(example)
    clean_post = clean_nms[batch_index]
    patch_post = patched_nms[batch_index]
    if int(clean_post.shape[0]) == 0:
        raise RuntimeError("Clean image has no post-NMS target detection at the configured threshold.")
    # Both tensors are in the letterboxed model-input coordinate system.
    # The cached bbox is in original-image coordinates and must not be mixed
    # with these decoded candidates.
    clean_target_flat = _match_post_nms_to_flat(batch_index, clean_post[0], clean_decoded, class_id)
    patched_winner_flat = None
    if int(patch_post.shape[0]) > 0:
        patched_winner_flat = _match_post_nms_to_flat(batch_index, patch_post[0], patched_decoded, class_id)

    clean_candidates = _top_candidates(
        example_id=example_id, variant="clean", batch_index=batch_index, decoded=clean_decoded,
        levels=clean_levels, slices=slices, class_id=class_id, reg_max=reg_max, imgsz=imgsz,
        nms_boxes=clean_post, config=config,
    )
    patched_candidates = _top_candidates(
        example_id=example_id, variant="patched", batch_index=batch_index, decoded=patched_decoded,
        levels=patched_levels, slices=slices, class_id=class_id, reg_max=reg_max, imgsz=imgsz,
        nms_boxes=patch_post, config=config,
    )
    clean_target = _candidate_from_flat(
        batch_index=batch_index, flat_index=clean_target_flat, decoded=clean_decoded, levels=clean_levels,
        slices=slices, class_id=class_id, reg_max=reg_max, imgsz=imgsz, nms_boxes=clean_post,
        nms_survival_iou=config.nms_survival_iou,
    )
    patched_tracked = _candidate_from_flat(
        batch_index=batch_index, flat_index=clean_target_flat, decoded=patched_decoded, levels=patched_levels,
        slices=slices, class_id=class_id, reg_max=reg_max, imgsz=imgsz, nms_boxes=patch_post,
        nms_survival_iou=config.nms_survival_iou,
    )
    patched_winner = None
    if patched_winner_flat is not None:
        patched_winner = _candidate_from_flat(
            batch_index=batch_index, flat_index=patched_winner_flat, decoded=patched_decoded,
            levels=patched_levels, slices=slices, class_id=class_id, reg_max=reg_max, imgsz=imgsz,
            nms_boxes=patch_post, nms_survival_iou=config.nms_survival_iou,
        )

    mode, winner_iou, tracked_iou = classify_mechanism(
        patched_winner=patched_winner, clean_target=clean_target, patched_tracked=patched_tracked,
        same_object_iou=config.same_object_iou, replacement_iou=config.replacement_iou,
    )
    clean_conf_recomputed = float(clean_post[0, 4].detach().cpu()) if int(clean_post.shape[0]) else 0.0
    patch_conf_recomputed = float(patch_post[0, 4].detach().cpu()) if int(patch_post.shape[0]) else 0.0
    patch_bbox = _pad_bbox(getattr(example, "patch_bbox_lb", None))
    row = {column: None for column in EXAMPLE_COLUMNS}
    row.update({
        "example_id": example_id, "path": str(example.path), "success": int(bool(example.success)),
        "confidence_drop": float(example.drop), "conf_clean_cached": float(example.conf_clean),
        "conf_patch_cached": float(example.conf_patch), "conf_clean_recomputed": clean_conf_recomputed,
        "conf_patch_recomputed": patch_conf_recomputed,
        "clean_conf_abs_error": abs(clean_conf_recomputed - float(example.conf_clean)),
        "patch_conf_abs_error": abs(patch_conf_recomputed - float(example.conf_patch)),
        "class_id": class_id, "patch_x1": patch_bbox[0], "patch_y1": patch_bbox[1],
        "patch_x2": patch_bbox[2], "patch_y2": patch_bbox[3],
        "clean_target_flat": clean_target["flat_index"], "clean_target_level": clean_target["level_index"],
        "clean_target_y": clean_target["y_index"], "clean_target_x": clean_target["x_index"],
        "clean_target_score": clean_target["decoded_score"], "clean_target_logit": clean_target["class_logit"],
        "clean_target_x1": clean_target["bbox_x1"], "clean_target_y1": clean_target["bbox_y1"],
        "clean_target_x2": clean_target["bbox_x2"], "clean_target_y2": clean_target["bbox_y2"],
        "clean_target_nms_survived": clean_target["nms_survived"],
        "patched_tracked_score": patched_tracked["decoded_score"],
        "patched_tracked_logit": patched_tracked["class_logit"],
        "patched_tracked_x1": patched_tracked["bbox_x1"], "patched_tracked_y1": patched_tracked["bbox_y1"],
        "patched_tracked_x2": patched_tracked["bbox_x2"], "patched_tracked_y2": patched_tracked["bbox_y2"],
        "patched_tracked_nms_survived": patched_tracked["nms_survived"],
        "tracked_score_delta": patched_tracked["decoded_score"] - clean_target["decoded_score"],
        "tracked_logit_delta": patched_tracked["class_logit"] - clean_target["class_logit"],
        "tracked_box_iou_clean": tracked_iou, "winner_iou_clean": winner_iou,
        "mechanism_mode": mode, "error": None,
    })
    if patched_winner is not None:
        row.update({
            "patched_winner_flat": patched_winner["flat_index"],
            "patched_winner_level": patched_winner["level_index"],
            "patched_winner_y": patched_winner["y_index"], "patched_winner_x": patched_winner["x_index"],
            "patched_winner_score": patched_winner["decoded_score"],
            "patched_winner_logit": patched_winner["class_logit"],
            "patched_winner_x1": patched_winner["bbox_x1"], "patched_winner_y1": patched_winner["bbox_y1"],
            "patched_winner_x2": patched_winner["bbox_x2"], "patched_winner_y2": patched_winner["bbox_y2"],
            "winner_same_flat": int(patched_winner["flat_index"] == clean_target["flat_index"]),
            "winner_level_changed": int(patched_winner["level_index"] != clean_target["level_index"]),
            "patched_winner_minus_tracked": patched_winner["decoded_score"] - patched_tracked["decoded_score"],
        })
    candidates = clean_candidates + patched_candidates
    lineage = _lineage_rows(example_id, clean_candidates, patched_candidates, config.lineage_top_k)
    return row, candidates, lineage


def _summary_outputs(conn: sqlite3.Connection, run_dir: Path, config: CandidateTraceConfig) -> tuple[Path, Path]:
    examples = pd.read_sql_query("SELECT * FROM examples WHERE error IS NULL", conn)
    if examples.empty:
        payload = {"n_examples": 0, "status": "empty"}
        return write_json(run_dir / "summary.json", payload), write_markdown(run_dir / "analysis_digest.md", ["# Candidate tracing", "", "No completed examples."])

    mode_summary = (
        examples.groupby(["mechanism_mode", "success"], dropna=False)
        .size().rename("n").reset_index()
    )
    mode_summary["fraction_within_success"] = mode_summary["n"] / mode_summary.groupby("success")["n"].transform("sum")
    mode_summary.to_csv(run_dir / "mode_summary.csv", index=False)

    level_summary = (
        examples.groupby(["clean_target_level", "patched_winner_level", "success"], dropna=False)
        .size().rename("n").reset_index()
    )
    level_summary.to_csv(run_dir / "level_transition_summary.csv", index=False)

    score_summary = (
        examples.groupby(["mechanism_mode", "success"], dropna=False)
        .agg(
            n=("example_id", "size"), mean_drop=("confidence_drop", "mean"),
            mean_tracked_logit_delta=("tracked_logit_delta", "mean"),
            mean_winner_minus_tracked=("patched_winner_minus_tracked", "mean"),
            tracked_nms_survival_rate=("patched_tracked_nms_survived", "mean"),
            mean_winner_iou_clean=("winner_iou_clean", "mean"),
        ).reset_index()
    )
    score_summary.to_csv(run_dir / "score_summary.csv", index=False)

    payload = {
        "status": "complete",
        "n_examples": int(len(examples)),
        "n_success": int(examples["success"].sum()),
        "n_failure": int((1 - examples["success"]).sum()),
        "mean_cached_vs_recomputed_abs_error": {
            "clean": float(examples["clean_conf_abs_error"].mean()),
            "patched": float(examples["patch_conf_abs_error"].mean()),
        },
        "mode_counts": {str(key): int(value) for key, value in examples["mechanism_mode"].value_counts(dropna=False).items()},
        "mode_fractions": {str(key): float(value) for key, value in examples["mechanism_mode"].value_counts(normalize=True, dropna=False).items()},
        "winner_changed_fraction": float((examples["winner_same_flat"].fillna(0) == 0).mean()),
        "cross_scale_fraction": float(examples["winner_level_changed"].fillna(0).mean()),
        "tracked_patch_nms_survival_rate": float(examples["patched_tracked_nms_survived"].mean()),
        "clean_target_nms_survival_rate": float(examples["clean_target_nms_survived"].mean()),
        "mean_clean_target_vs_post_nms_abs_error": float(
            (examples["clean_target_score"] - examples["conf_clean_recomputed"]).abs().mean()
        ),
        "database": str(run_dir / "candidate_tracing.sqlite"),
        "config": asdict(config),
    }
    summary_path = write_json(run_dir / "summary.json", payload)
    most_common = examples["mechanism_mode"].value_counts().head(5)
    digest_lines = [
        "# Candidate tracing digest", "",
        f"- examples: {len(examples)} ({int(examples['success'].sum())} success, {int((1-examples['success']).sum())} fail)",
        f"- winner changed: {payload['winner_changed_fraction']:.3f}",
        f"- cross-scale winner change: {payload['cross_scale_fraction']:.3f}",
        f"- tracked candidate survives patched NMS: {payload['tracked_patch_nms_survival_rate']:.3f}",
        f"- clean target survives clean NMS: {payload['clean_target_nms_survival_rate']:.3f}",
        f"- mean clean target/post-NMS score error: {payload['mean_clean_target_vs_post_nms_abs_error']:.6f}",
        f"- mean cached/recomputed confidence error: clean={payload['mean_cached_vs_recomputed_abs_error']['clean']:.6f}, patched={payload['mean_cached_vs_recomputed_abs_error']['patched']:.6f}",
        "", "## Most common observable modes", "",
    ]
    digest_lines.extend(f"- {name}: {int(count)}" for name, count in most_common.items())
    digest_lines.extend(["", "Read summary.json and the three small CSV summaries before opening SQLite."])
    digest_path = write_markdown(run_dir / "analysis_digest.md", digest_lines)
    return summary_path, digest_path


def run_candidate_tracing(exp, config: CandidateTraceConfig | None = None, *, force: bool = False) -> CandidateTraceResult:
    config = config or CandidateTraceConfig()
    cache = exp.get_cache()
    selected = list(cache.examples)
    if config.max_examples is not None:
        selected = selected[: int(config.max_examples)]
    payload = {
        "attack_cache_key": getattr(cache, "cache_key", None),
        "paths": [_example_id(item) for item in selected],
        **asdict(config),
    }
    run_dir = Path(config.output_dir) / f"candidate_trace_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = StorageBudget(Path(config.output_dir), config.max_output_gb)
    budget.check()
    db_path = run_dir / "candidate_tracing.sqlite"
    if force and db_path.exists():
        db_path.unlink()
    conn = connect_db(db_path)
    _create_schema(conn)
    upsert_metadata(conn, {"config": asdict(config), "payload": payload, "attack_cache_key": getattr(cache, "cache_key", None)})
    completed = {row[0] for row in conn.execute("SELECT example_id FROM examples WHERE error IS NULL")}
    selected = [item for item in selected if _example_id(item) not in completed]

    yolo, model = exp.load_model()
    from segmentig_detector.yolo_utils import ensure_predictor, get_detect_module

    predictor = ensure_predictor(
        yolo, imgsz=int(exp.config.attack.imgsz), conf=float(config.nms_conf),
        iou=float(config.nms_iou), device=exp.config.attack.device,
    )
    detect = get_detect_module(model, exp.config.detect_layer)
    nc = int(getattr(detect, "nc", 80))
    reg_max = int(getattr(detect, "reg_max", 16))
    progress = None
    if config.progress:
        try:
            from tqdm.auto import tqdm
            progress = tqdm(total=len(selected), desc="candidate tracing", unit="img")
        except Exception:
            progress = None

    try:
        for start in range(0, len(selected), max(1, int(config.batch_size))):
            batch_examples = selected[start : start + max(1, int(config.batch_size))]
            valid_examples, clean_images, patched_images = [], [], []
            error_rows = []
            for example in batch_examples:
                try:
                    clean, patched, _bbox = exp._images_for_example(example)
                    valid_examples.append(example)
                    clean_images.append(clean)
                    patched_images.append(patched)
                except Exception as exc:  # noqa: BLE001
                    row = {column: None for column in EXAMPLE_COLUMNS}
                    row.update({"example_id": _example_id(example), "path": str(example.path), "success": int(example.success), "confidence_drop": float(example.drop), "error": f"{type(exc).__name__}: {exc}"})
                    error_rows.append(row)

            if error_rows:
                _insert_rows(conn, "examples", EXAMPLE_COLUMNS, error_rows)
            if valid_examples:
                clean_inputs = _preprocess_batch(predictor, clean_images, model=model)
                patched_inputs = _preprocess_batch(predictor, patched_images, model=model)
                clean_decoded, clean_levels, clean_nms = _forward_batch(model, clean_inputs, class_id=int(valid_examples[0].target_class_id), config=config, nc=nc)
                patched_decoded, patched_levels, patched_nms = _forward_batch(model, patched_inputs, class_id=int(valid_examples[0].target_class_id), config=config, nc=nc)
                slices = _level_slices(clean_levels)
                example_rows, candidate_rows, lineage_rows = [], [], []
                for batch_index, example in enumerate(valid_examples):
                    try:
                        row, candidates, lineage = _process_pair(
                            example=example, batch_index=batch_index,
                            clean_decoded=clean_decoded, clean_levels=clean_levels, clean_nms=clean_nms,
                            patched_decoded=patched_decoded, patched_levels=patched_levels, patched_nms=patched_nms,
                            slices=slices, class_id=int(example.target_class_id), reg_max=reg_max,
                            imgsz=int(exp.config.attack.imgsz), config=config,
                        )
                        example_rows.append(row)
                        candidate_rows.extend(candidates)
                        lineage_rows.extend(lineage)
                    except Exception as exc:  # noqa: BLE001
                        row = {column: None for column in EXAMPLE_COLUMNS}
                        row.update({"example_id": _example_id(example), "path": str(example.path), "success": int(example.success), "confidence_drop": float(example.drop), "error": f"{type(exc).__name__}: {exc}"})
                        example_rows.append(row)
                _insert_rows(conn, "examples", EXAMPLE_COLUMNS, example_rows)
                _insert_rows(conn, "candidates", CANDIDATE_COLUMNS, candidate_rows)
                _insert_rows(conn, "lineage", LINEAGE_COLUMNS, lineage_rows)
            conn.commit()
            budget.check(extra_bytes=100 * 1024**2)
            if progress is not None:
                progress.update(len(batch_examples))
            release_accelerator_memory()
    finally:
        if progress is not None:
            progress.close()
        summary_path, digest_path = _summary_outputs(conn, run_dir, config)
        conn.close()

    return CandidateTraceResult(
        run_dir=run_dir, db_path=db_path, summary_path=summary_path,
        digest_path=digest_path, config=config,
    )


def load_trace_tables(result_or_db: CandidateTraceResult | str | Path, *, candidates: bool = False, lineage: bool = False):
    db_path = result_or_db.db_path if isinstance(result_or_db, CandidateTraceResult) else Path(result_or_db)
    conn = connect_db(db_path)
    try:
        out = {"examples": pd.read_sql_query("SELECT * FROM examples WHERE error IS NULL", conn)}
        if candidates:
            out["candidates"] = pd.read_sql_query("SELECT * FROM candidates", conn)
        if lineage:
            out["lineage"] = pd.read_sql_query("SELECT * FROM lineage", conn)
        return out
    finally:
        conn.close()


def load_example_candidates(result_or_db: CandidateTraceResult | str | Path, example_id: str) -> pd.DataFrame:
    db_path = result_or_db.db_path if isinstance(result_or_db, CandidateTraceResult) else Path(result_or_db)
    conn = connect_db(db_path)
    try:
        return pd.read_sql_query(
            "SELECT * FROM candidates WHERE example_id=? ORDER BY variant, rank",
            conn, params=(str(example_id),),
        )
    finally:
        conn.close()
