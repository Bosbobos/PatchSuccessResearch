from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .yolo_utils import (
    DetectionChoice,
    box_iou_xyxy,
    detect_train_levels,
    raw_inference_tensor,
    xywh_to_xyxy,
)


@dataclass(slots=True)
class FixedDetectorTarget:
    class_id: int
    level_index: int
    y_index: int
    x_index: int
    flat_index: int
    level_hw: tuple[int, int]
    stride_hw: tuple[float, float]
    matched_iou: float
    matched_class_score: float
    detection_confidence: float
    detection_bbox_xyxy_orig: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _level_slices(levels: list[torch.Tensor]) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    offset = 0
    for level_index, tensor in enumerate(levels):
        h, w = int(tensor.shape[-2]), int(tensor.shape[-1])
        n = h * w
        out.append({"level_index": level_index, "h": h, "w": w, "start": offset, "end": offset + n})
        offset += n
    return out


def _flat_index_to_level(flat_index: int, slices: list[dict[str, int]]) -> tuple[int, int, int, int]:
    for item in slices:
        if item["start"] <= int(flat_index) < item["end"]:
            local = int(flat_index) - item["start"]
            y = local // item["w"]
            x = local % item["w"]
            return int(item["level_index"]), int(y), int(x), int(local)
    raise IndexError(f"flat_index={flat_index} is outside prediction slices.")


@torch.no_grad()
def select_fixed_detector_target(
    model: torch.nn.Module,
    im: torch.Tensor,
    detection: DetectionChoice,
    *,
    detect_name: str = "model.23",
    orig_hw: tuple[int, int] | None = None,
) -> FixedDetectorTarget:
    """Match a post-NMS person detection to one decoded pre-NMS head position."""

    raw = raw_inference_tensor(model, im)
    levels, _detect = detect_train_levels(model, im, detect_name=detect_name)
    slices = _level_slices(levels)
    if sum(item["end"] - item["start"] for item in slices) != int(raw.shape[-1]):
        raise RuntimeError(
            "Decoded raw prediction count does not match Detect train levels: "
            f"raw N={raw.shape[-1]}, level N={sum(item['end'] - item['start'] for item in slices)}"
        )

    class_id = int(detection.class_id)
    xywh = raw[0, 0:4, :].transpose(0, 1)
    boxes_im = xywh_to_xyxy(xywh)
    boxes = boxes_im
    if orig_hw is not None:
        from ultralytics.utils import ops

        boxes = ops.scale_boxes(tuple(im.shape[-2:]), boxes_im.clone(), tuple(orig_hw))
    class_scores = raw[0, 4 + class_id, :]
    target_box = torch.tensor(detection.bbox_xyxy_orig, device=boxes.device, dtype=boxes.dtype).reshape(1, 4)
    ious = box_iou_xyxy(boxes, target_box).squeeze(1)

    # Prefer strong geometric match. Score breaks ties using decoded class score.
    rank_score = ious + 0.05 * class_scores.clamp(min=0.0, max=1.0)
    flat_index = int(torch.argmax(rank_score).item())
    level_index, y_index, x_index, _local = _flat_index_to_level(flat_index, slices)
    level = levels[level_index]
    level_h, level_w = int(level.shape[-2]), int(level.shape[-1])
    im_h, im_w = int(im.shape[-2]), int(im.shape[-1])

    return FixedDetectorTarget(
        class_id=class_id,
        level_index=level_index,
        y_index=y_index,
        x_index=x_index,
        flat_index=flat_index,
        level_hw=(level_h, level_w),
        stride_hw=(float(im_h) / float(level_h), float(im_w) / float(level_w)),
        matched_iou=float(ious[flat_index].detach().cpu().item()),
        matched_class_score=float(class_scores[flat_index].detach().cpu().item()),
        detection_confidence=float(detection.confidence),
        detection_bbox_xyxy_orig=detection.bbox_xyxy_orig,
    )


def _dfl_width_height(
    level_tensor: torch.Tensor,
    target: FixedDetectorTarget,
    *,
    reg_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dist_logits = level_tensor[
        :,
        : 4 * int(reg_max),
        int(target.y_index),
        int(target.x_index),
    ].reshape(level_tensor.shape[0], 4, int(reg_max))
    bins = torch.arange(int(reg_max), device=level_tensor.device, dtype=level_tensor.dtype)
    distances = torch.softmax(dist_logits, dim=2).matmul(bins)
    left, top, right, bottom = distances[:, 0], distances[:, 1], distances[:, 2], distances[:, 3]
    stride_h, stride_w = float(target.stride_hw[0]), float(target.stride_hw[1])
    width = (left + right) * stride_w
    height = (top + bottom) * stride_h
    return width, height


def detector_target_scalar(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: FixedDetectorTarget,
    *,
    mode: str,
    imgsz: int = 640,
    detect_name: str = "model.23",
    wh_weight: float = 10.0,
) -> torch.Tensor:
    """Strict pre-sigmoid detector scalar for a fixed head level/cell."""

    levels, detect = detect_train_levels(model, x, detect_name=detect_name)
    if int(target.level_index) >= len(levels):
        raise IndexError(f"Target level {target.level_index} is unavailable; got {len(levels)} levels.")
    level = levels[int(target.level_index)]
    reg_max = int(getattr(detect, "reg_max", 16))
    nc = int(getattr(detect, "nc", 80))
    class_channel = 4 * reg_max + int(target.class_id)
    if class_channel >= int(level.shape[1]):
        raise IndexError(
            f"class_channel={class_channel} exceeds level channels={level.shape[1]} "
            f"(reg_max={reg_max}, nc={nc}, class_id={target.class_id})."
        )

    class_logit = level[:, class_channel, int(target.y_index), int(target.x_index)]
    if mode == "class_only":
        return class_logit.sum()
    if mode in {
        "width",
        "height",
        "width_height_normalized",
        "class_plus_width_height_normalized",
        "class_width_height",
    }:
        width, height = _dfl_width_height(level, target, reg_max=reg_max)
    if mode == "width":
        return width.sum()
    if mode == "height":
        return height.sum()
    if mode == "width_height_normalized":
        return ((width + height) / float(imgsz)).sum()
    if mode == "class_plus_width_height_normalized":
        return (class_logit + ((width + height) / float(imgsz))).sum()
    if mode == "class_width_height":
        normalized_wh = (width + height) / float(imgsz)
        return (class_logit + float(wh_weight) * normalized_wh).sum()
    raise ValueError(f"Unsupported detector target mode: {mode!r}")


@torch.no_grad()
def detector_target_components(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: FixedDetectorTarget,
    *,
    imgsz: int = 640,
    detect_name: str = "model.23",
) -> dict[str, float]:
    """Return scalar components used by the fixed detector target."""

    levels, detect = detect_train_levels(model, x, detect_name=detect_name)
    level = levels[int(target.level_index)]
    reg_max = int(getattr(detect, "reg_max", 16))
    class_channel = 4 * reg_max + int(target.class_id)
    class_logit = level[:, class_channel, int(target.y_index), int(target.x_index)]
    width, height = _dfl_width_height(level, target, reg_max=reg_max)
    wh_norm_sum = (width + height) / float(imgsz)
    return {
        "class_logit": float(class_logit.detach().cpu().mean().item()),
        "width": float(width.detach().cpu().mean().item()),
        "height": float(height.detach().cpu().mean().item()),
        "width_plus_height_over_imgsz": float(wh_norm_sum.detach().cpu().mean().item()),
    }
