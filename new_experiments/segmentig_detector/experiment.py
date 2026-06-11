from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .layer_segmentig import compute_layer_segmentig_scalar
from .targets import detector_target_components, detector_target_scalar, select_fixed_detector_target
from .visualization import save_result_figure, spatial_importance_map
from .yolo_utils import (
    get_class_id,
    get_module_by_name,
    get_torch_model,
    list_image_paths,
    load_rgb,
    load_yolo,
    preprocess_pil,
    select_images_with_person,
)


@dataclass(slots=True)
class SegmentIGDetectorConfig:
    weights: str = "yolo11s.pt"
    dataset_dir: str = "datasets/inria_train"
    output_dir: str = "new_experiments/outputs/segmentig_detector_layer22"
    target_layer: str = "model.22"
    detect_layer: str = "model.23"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    device: str | None = None
    n_images: int = 3
    n_steps: int = 64
    alpha_batch_size: int = 4
    segment_start: float = 0.0
    segment_end: float = 0.1
    wh_weight: float = 10.0


def _baseline_like(inputs: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(inputs)


def _metadata_jsonable(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, tuple):
            out[key] = list(value)
        elif isinstance(value, dict):
            out[key] = _metadata_jsonable(value)
        else:
            out[key] = value
    return out


def run_segmentig_for_image(
    yolo,
    image_path: str | Path,
    *,
    config: SegmentIGDetectorConfig,
    preselected_detection=None,
) -> dict[str, Any]:
    model = get_torch_model(yolo)
    model.eval()
    if config.device is not None:
        model.to(torch.device(config.device))

    names = getattr(yolo, "names", getattr(model, "names", {}))
    person_class_id = get_class_id(names, "person", 0)
    if preselected_detection is None:
        detections = select_images_with_person(
            yolo,
            [image_path],
            count=1,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            device=config.device,
            person_class_id=person_class_id,
        )
        detection = detections[0]
    else:
        detection = preselected_detection

    image = load_rgb(image_path)
    pack = preprocess_pil(
        yolo,
        image,
        imgsz=config.imgsz,
        conf=config.conf,
        iou=config.iou,
        device=config.device,
    )
    inputs = pack["im"].to(
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )
    baselines = _baseline_like(inputs)
    target_layer = get_module_by_name(model, config.target_layer)
    fixed_target = select_fixed_detector_target(
        model,
        inputs,
        detection,
        detect_name=config.detect_layer,
        orig_hw=pack["orig_hw"],
    )
    target_components = detector_target_components(
        model,
        inputs,
        fixed_target,
        imgsz=config.imgsz,
        detect_name=config.detect_layer,
    )

    def target_class_logit(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return detector_target_scalar(
            m,
            x,
            fixed_target,
            mode="class_only",
            imgsz=config.imgsz,
            detect_name=config.detect_layer,
            wh_weight=config.wh_weight,
        )

    def target_width(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return detector_target_scalar(
            m,
            x,
            fixed_target,
            mode="width",
            imgsz=config.imgsz,
            detect_name=config.detect_layer,
            wh_weight=config.wh_weight,
        )

    def target_height(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return detector_target_scalar(
            m,
            x,
            fixed_target,
            mode="height",
            imgsz=config.imgsz,
            detect_name=config.detect_layer,
            wh_weight=config.wh_weight,
        )

    def target_wh_norm(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return detector_target_scalar(
            m,
            x,
            fixed_target,
            mode="width_height_normalized",
            imgsz=config.imgsz,
            detect_name=config.detect_layer,
            wh_weight=config.wh_weight,
        )

    def target_class_plus_wh_norm(m: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return detector_target_scalar(
            m,
            x,
            fixed_target,
            mode="class_plus_width_height_normalized",
            imgsz=config.imgsz,
            detect_name=config.detect_layer,
            wh_weight=config.wh_weight,
        )

    class_logit_result = compute_layer_segmentig_scalar(
        model,
        inputs,
        baselines,
        target_fn=target_class_logit,
        layer=target_layer,
        n_steps=config.n_steps,
        alpha_batch_size=config.alpha_batch_size,
        segment_start=config.segment_start,
        segment_end=config.segment_end,
    )
    width_result = compute_layer_segmentig_scalar(
        model,
        inputs,
        baselines,
        target_fn=target_width,
        layer=target_layer,
        n_steps=config.n_steps,
        alpha_batch_size=config.alpha_batch_size,
        segment_start=config.segment_start,
        segment_end=config.segment_end,
    )
    height_result = compute_layer_segmentig_scalar(
        model,
        inputs,
        baselines,
        target_fn=target_height,
        layer=target_layer,
        n_steps=config.n_steps,
        alpha_batch_size=config.alpha_batch_size,
        segment_start=config.segment_start,
        segment_end=config.segment_end,
    )
    wh_norm_result = compute_layer_segmentig_scalar(
        model,
        inputs,
        baselines,
        target_fn=target_wh_norm,
        layer=target_layer,
        n_steps=config.n_steps,
        alpha_batch_size=config.alpha_batch_size,
        segment_start=config.segment_start,
        segment_end=config.segment_end,
    )
    class_plus_wh_norm_result = compute_layer_segmentig_scalar(
        model,
        inputs,
        baselines,
        target_fn=target_class_plus_wh_norm,
        layer=target_layer,
        n_steps=config.n_steps,
        alpha_batch_size=config.alpha_batch_size,
        segment_start=config.segment_start,
        segment_end=config.segment_end,
    )

    class_logit_map = spatial_importance_map(class_logit_result.attribution_bchw())
    width_map = spatial_importance_map(width_result.attribution_bchw())
    height_map = spatial_importance_map(height_result.attribution_bchw())
    wh_norm_map = spatial_importance_map(wh_norm_result.attribution_bchw())
    class_plus_wh_norm_map = spatial_importance_map(class_plus_wh_norm_result.attribution_bchw())

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    png_path = output_dir / f"{stem}_segmentig_layer22.png"
    json_path = output_dir / f"{stem}_segmentig_layer22.json"

    metadata = {
        "image_path": str(image_path),
        "weights": config.weights,
        "target_layer": config.target_layer,
        "detect_layer": config.detect_layer,
        "imgsz": config.imgsz,
        "segment": [config.segment_start, config.segment_end],
        "n_steps": config.n_steps,
        "alpha_batch_size": config.alpha_batch_size,
        "wh_weight": config.wh_weight,
        "person_class_id": int(person_class_id),
        "detection_confidence": float(detection.confidence),
        "detection_bbox_xyxy_orig": list(detection.bbox_xyxy_orig),
        "fixed_target": fixed_target.to_dict(),
        "target_components": target_components,
        "targets_rendered": [
            "class_logit",
            "width",
            "height",
            "width_height_normalized",
            "class_plus_width_height_normalized",
        ],
        "activation_shape": list(width_result.activation_shape),
        "class_logit_alpha_count": int(class_logit_result.alpha_values.numel()),
        "width_alpha_count": int(width_result.alpha_values.numel()),
        "height_alpha_count": int(height_result.alpha_values.numel()),
        "width_height_normalized_alpha_count": int(wh_norm_result.alpha_values.numel()),
        "class_plus_width_height_normalized_alpha_count": int(class_plus_wh_norm_result.alpha_values.numel()),
        "class_logit_map_has_signal": bool(float(abs(class_logit_map).max()) > 0.0),
        "width_map_has_signal": bool(float(abs(width_map).max()) > 0.0),
        "height_map_has_signal": bool(float(abs(height_map).max()) > 0.0),
        "width_height_normalized_map_has_signal": bool(float(abs(wh_norm_map).max()) > 0.0),
        "class_plus_width_height_normalized_map_has_signal": bool(
            float(abs(class_plus_wh_norm_map).max()) > 0.0
        ),
        "png_path": str(png_path),
        "json_path": str(json_path),
    }
    metadata = _metadata_jsonable(metadata)

    save_result_figure(
        image,
        detection.bbox_xyxy_orig,
        class_logit_map,
        width_map,
        height_map,
        wh_norm_map,
        class_plus_wh_norm_map,
        metadata,
        png_path,
    )
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "metadata": metadata,
        "class_logit": class_logit_result,
        "width": width_result,
        "height": height_result,
        "width_height_normalized": wh_norm_result,
        "class_plus_width_height_normalized": class_plus_wh_norm_result,
        "class_logit_map": class_logit_map,
        "width_map": width_map,
        "height_map": height_map,
        "width_height_normalized_map": wh_norm_map,
        "class_plus_width_height_normalized_map": class_plus_wh_norm_map,
    }


def run_segmentig_detector_experiment(config: SegmentIGDetectorConfig | None = None) -> list[dict[str, Any]]:
    config = config or SegmentIGDetectorConfig()
    yolo = load_yolo(config.weights)
    model = get_torch_model(yolo)
    if config.device is not None:
        model.to(torch.device(config.device))
    names = getattr(yolo, "names", getattr(model, "names", {}))
    person_class_id = get_class_id(names, "person", 0)
    image_paths = list_image_paths(config.dataset_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {config.dataset_dir!r}.")

    detections = select_images_with_person(
        yolo,
        image_paths,
        count=config.n_images,
        imgsz=config.imgsz,
        conf=config.conf,
        iou=config.iou,
        device=config.device,
        person_class_id=person_class_id,
    )

    results: list[dict[str, Any]] = []
    for detection in detections:
        results.append(
            run_segmentig_for_image(
                yolo,
                detection.image_path,
                config=config,
                preselected_detection=detection,
            )
        )

    summary = {
        "config": asdict(config),
        "results": [item["metadata"] for item in results],
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results
