from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .candidate_routing import _box_iou, _xywh_to_xyxy
from .common import REPO_ROOT, load_experiment, stable_hash


DEFAULT_OUTPUT_DIR = REPO_ROOT / "CandidateRoutingAndAttackPath" / "pixel_patch_outputs"


@dataclass(slots=True)
class MechanismAwarePatchConfig:
    """Configuration for an offline, universal pixel-patch experiment."""

    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "cpu"
    require_device: bool = False
    patch_size: int = 160
    patch_xy: tuple[int, int] = (0, 0)
    train_examples: int = 64
    eval_examples: int = 64
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 0.08
    smoothmax_temperature: float = 0.35
    dynamic_iou_temperature: float = 0.07
    dynamic_iou_weight: float = 4.0
    match_iou: float = 0.50
    detection_conf: float = 0.25
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    tv_weight: float = 0.002
    init_patch: str | None = None
    seed: int = 211
    method_version: int = 2


def total_variation(patch):
    """Mean anisotropic TV; normalized so it is independent of patch dimensions."""

    horizontal = (patch[..., :, 1:] - patch[..., :, :-1]).abs().mean()
    vertical = (patch[..., 1:, :] - patch[..., :-1, :]).abs().mean()
    return horizontal + vertical


def overlay_patch(images, patch, xy: tuple[int, int]):
    """Paste a differentiable CHW patch into a BCHW image batch."""

    x, y = int(xy[0]), int(xy[1])
    _, _, image_h, image_w = images.shape
    _, _, patch_h, patch_w = patch.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image_w, x + patch_w), min(image_h, y + patch_h)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Patch at {xy} does not overlap images of size {(image_w, image_h)}")
    patch_x1, patch_y1 = x1 - x, y1 - y
    out = images.clone()
    out[..., y1:y2, x1:x2] = patch[..., patch_y1:patch_y1 + y2 - y1, patch_x1:patch_x1 + x2 - x1]
    return out


def dynamic_score_geometry_loss(
    decoded,
    target_boxes,
    class_ids,
    *,
    match_iou: float,
    iou_temperature: float,
    iou_weight: float,
    smoothmax_temperature: float,
):
    """Smooth maximum over all candidates that can still represent each clean target.

    This is the pixel-space counterpart of the best activation-space objective in
    ``07_AttackDirection``. Geometry is differentiable, so the candidate reserve is
    allowed to reroute while the patch is optimized.
    """

    import torch

    boxes = _xywh_to_xyxy(decoded[:, :4, :].transpose(1, 2))
    ious = torch.stack(
        [_box_iou(boxes[index], target_boxes[index:index + 1]).reshape(-1)
         for index in range(int(decoded.shape[0]))],
        dim=0,
    )
    batch_indices = torch.arange(decoded.shape[0], device=decoded.device)
    scores = decoded[batch_indices, 4 + class_ids, :].clamp(1e-6, 1.0 - 1e-6)
    logits = torch.logit(scores)
    membership = torch.sigmoid((ious - float(match_iou)) / float(iou_temperature))
    risk = logits + float(iou_weight) * torch.log(membership.clamp_min(1e-6))
    temperature = float(smoothmax_temperature)
    per_image = temperature * (
        torch.logsumexp(risk / temperature, dim=1) - math.log(int(risk.shape[1]))
    )
    return per_image.mean(), {
        "mean_risk": per_image.detach().mean(),
        "mean_best_iou": ious.detach().amax(dim=1).mean(),
        "mean_best_target_score": (scores.detach() * (ious.detach() >= float(match_iou))).amax(dim=1).mean(),
    }


def _decoded_from_model(model, images):
    from segmentig_detector.yolo_utils import safe_model_forward
    import torch

    output = safe_model_forward(model, images)
    if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 3:
        raise RuntimeError(f"Expected decoded YOLO tensor [B,C,N], got {type(output)}")
    return output


def _pil_to_tensor(image, *, device, dtype):
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(device=device, dtype=dtype)


def _example_records(exp) -> list[dict[str, Any]]:
    records = []
    for example in exp.get_cache().examples:
        detection = getattr(example, "clean_detection", None)
        if not detection or detection.get("bbox_xyxy_orig") is None:
            continue
        records.append({
            "example": example,
            "path": str(example.path),
            "class_id": int(detection["class_id"]),
            "target_box": tuple(float(value) for value in detection["bbox_xyxy_orig"]),
        })
    return records


def split_records(
    records: Iterable[dict[str, Any]],
    *,
    train_examples: int,
    eval_examples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(records)
    rng = random.Random(int(seed))
    rng.shuffle(rows)
    train = rows[: min(int(train_examples), len(rows))]
    remaining = rows[len(train):]
    evaluation = remaining[: min(int(eval_examples), len(remaining))]
    if not train:
        raise RuntimeError("No training examples with a clean target detection were found.")
    if not evaluation:
        raise RuntimeError("No disjoint evaluation examples remain; reduce train_examples.")
    return train, evaluation


def _load_batch(exp, records, *, device, dtype):
    import torch
    from PIL import Image
    from new_experiments.patch_success_analysis.patching import letterbox_pil

    images = []
    for record in records:
        with Image.open(record["path"]) as image:
            clean = letterbox_pil(image.convert("RGB"), int(exp.config.attack.imgsz))
        images.append(_pil_to_tensor(clean, device=device, dtype=dtype))
    targets = torch.as_tensor(
        [record["target_box"] for record in records], device=device, dtype=torch.float32
    )
    class_ids = torch.as_tensor(
        [record["class_id"] for record in records], device=device, dtype=torch.long
    )
    return torch.stack(images, dim=0), targets, class_ids


def _initial_patch(config: MechanismAwarePatchConfig, *, device, dtype):
    import torch
    from PIL import Image

    size = int(config.patch_size)
    if config.init_patch:
        with Image.open(config.init_patch) as image:
            image = image.convert("RGB").resize((size, size))
        initial = _pil_to_tensor(image, device=device, dtype=dtype).unsqueeze(0)
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
        initial = torch.rand((1, 3, size, size), generator=generator).to(device=device, dtype=dtype)
    eps = 1e-4
    return torch.logit(initial.clamp(eps, 1.0 - eps)).detach().requires_grad_(True)


def _save_patch(path: Path, patch) -> None:
    from PIL import Image

    array = (
        patch.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255.0
    ).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _target_detection_metrics(decoded, target_boxes, class_ids, config):
    import torch
    from ultralytics.utils.nms import non_max_suppression

    detections = non_max_suppression(
        decoded.detach().clone(),
        conf_thres=float(config.nms_conf),
        iou_thres=float(config.nms_iou),
        classes=None,
        max_det=int(config.nms_max_det),
        nc=int(decoded.shape[1] - 4),
    )
    rows = []
    for index, detected in enumerate(detections):
        class_id = int(class_ids[index])
        same_class = detected[detected[:, 5].long() == class_id] if len(detected) else detected
        confidence = max_iou = 0.0
        if len(same_class):
            ious = _box_iou(same_class[:, :4], target_boxes[index:index + 1]).reshape(-1)
            max_iou = float(ious.max().cpu())
            valid = torch.nonzero(ious >= float(config.match_iou), as_tuple=False).reshape(-1)
            if len(valid):
                confidence = float(same_class[valid, 4].max().cpu())
        visible = confidence >= float(config.detection_conf) and max_iou >= float(config.match_iou)
        rows.append({
            "target_conf": confidence,
            "target_max_iou": max_iou,
            "target_visible": int(visible),
            "target_hidden": int(not visible),
        })
    return rows


def evaluate_patch(exp, model, records, patch, config) -> list[dict[str, Any]]:
    import torch

    parameter = next(model.parameters())
    output: list[dict[str, Any]] = []
    for start in range(0, len(records), int(config.batch_size)):
        chunk = records[start:start + int(config.batch_size)]
        images, targets, classes = _load_batch(
            exp, chunk, device=parameter.device, dtype=parameter.dtype
        )
        with torch.inference_mode():
            clean_decoded = _decoded_from_model(model, images)
            patched_decoded = _decoded_from_model(
                model, overlay_patch(images, patch, config.patch_xy)
            )
        clean_rows = _target_detection_metrics(clean_decoded, targets, classes, config)
        patch_rows = _target_detection_metrics(patched_decoded, targets, classes, config)
        for record, clean, patched in zip(chunk, clean_rows, patch_rows, strict=True):
            output.append({
                "path": record["path"],
                "clean_target_visible": clean["target_visible"],
                "clean_target_conf": clean["target_conf"],
                "patched_target_visible": patched["target_visible"],
                "patched_target_conf": patched["target_conf"],
                "target_hidden": patched["target_hidden"],
            })
    return output


def run_mechanism_aware_patch(config: MechanismAwarePatchConfig | None = None) -> Path:
    import pandas as pd
    import torch

    config = config or MechanismAwarePatchConfig()
    started = time.time()
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    exp, cache_path = load_experiment(
        prefer_device=config.device, require_device=bool(config.require_device)
    )
    _yolo, model = exp.load_model()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameter = next(model.parameters())
    train, evaluation = split_records(
        _example_records(exp),
        train_examples=config.train_examples,
        eval_examples=config.eval_examples,
        seed=config.seed,
    )
    payload = {
        **asdict(config),
        "cache_path": str(cache_path),
        "train_paths": [row["path"] for row in train],
        "eval_paths": [row["path"] for row in evaluation],
    }
    run_dir = Path(config.output_dir) / f"mechanism_patch_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logits = _initial_patch(config, device=parameter.device, dtype=parameter.dtype)
    initial_patch = torch.sigmoid(logits).detach().clone()
    optimizer = torch.optim.Adam([logits], lr=float(config.learning_rate))
    history = []
    for epoch in range(int(config.epochs)):
        order = np.random.default_rng(int(config.seed) + epoch).permutation(len(train))
        for batch_index, start in enumerate(range(0, len(order), int(config.batch_size))):
            indices = order[start:start + int(config.batch_size)]
            chunk = [train[int(index)] for index in indices]
            images, targets, classes = _load_batch(
                exp, chunk, device=parameter.device, dtype=parameter.dtype
            )
            patch = torch.sigmoid(logits)
            decoded = _decoded_from_model(model, overlay_patch(images, patch, config.patch_xy))
            mechanism_loss, diagnostics = dynamic_score_geometry_loss(
                decoded,
                targets,
                classes,
                match_iou=config.match_iou,
                iou_temperature=config.dynamic_iou_temperature,
                iou_weight=config.dynamic_iou_weight,
                smoothmax_temperature=config.smoothmax_temperature,
            )
            tv = total_variation(patch)
            loss = mechanism_loss + float(config.tv_weight) * tv
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            history.append({
                "epoch": epoch + 1,
                "batch": batch_index + 1,
                "loss": float(loss.detach().cpu()),
                "mechanism_loss": float(mechanism_loss.detach().cpu()),
                "tv": float(tv.detach().cpu()),
                **{key: float(value.cpu()) for key, value in diagnostics.items()},
            })
        _save_patch(run_dir / "patch_latest.png", torch.sigmoid(logits))
    final_patch = torch.sigmoid(logits).detach()
    _save_patch(run_dir / "patch.png", final_patch)
    baseline_rows = evaluate_patch(exp, model, evaluation, initial_patch, config)
    evaluation_rows = evaluate_patch(exp, model, evaluation, final_patch, config)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(run_dir / "baseline_evaluation.csv", index=False)
    pd.DataFrame(evaluation_rows).to_csv(run_dir / "evaluation.csv", index=False)
    clean_visible = sum(row["clean_target_visible"] for row in evaluation_rows)
    eligible = [row for row in evaluation_rows if row["clean_target_visible"]]
    baseline_by_path = {row["path"]: row for row in baseline_rows}
    baseline_hidden = sum(
        baseline_by_path[row["path"]]["target_hidden"] for row in eligible
    )
    hidden = sum(row["target_hidden"] for row in eligible)
    baseline_rate = baseline_hidden / max(len(eligible), 1)
    hiding_rate = hidden / max(len(eligible), 1)
    summary = {
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "cache_path": str(cache_path),
        "train_examples": len(train),
        "eval_examples": len(evaluation),
        "clean_visible_eval_examples": clean_visible,
        "baseline_hidden_eval_examples": baseline_hidden,
        "baseline_target_hiding_rate": baseline_rate,
        "hidden_eval_examples": hidden,
        "target_hiding_rate": hiding_rate,
        "absolute_hiding_rate_gain": hiding_rate - baseline_rate,
        "patch_path": str(run_dir / "patch.png"),
        "config": asdict(config),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "analysis_digest.md").write_text(
        "# Mechanism-aware pixel patch\n\n"
        f"- train/eval: {len(train)}/{len(evaluation)} (disjoint)\n"
        f"- clean-visible eval targets: {clean_visible}\n"
        f"- initial patch hidden: {baseline_hidden}/{max(len(eligible), 1)} "
        f"({baseline_rate:.3f})\n"
        f"- hidden after patch: {hidden}/{max(len(eligible), 1)} "
        f"({hiding_rate:.3f})\n"
        f"- absolute hiding-rate gain: {hiding_rate - baseline_rate:+.3f}\n"
        f"- elapsed: {summary['elapsed_seconds']:.1f} s\n\n"
        "Objective: differentiable smooth maximum over the dynamic clean-target "
        "candidate reserve, including candidate geometry, plus TV regularization.\n",
        encoding="utf-8",
    )
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an offline mechanism-aware YOLO patch.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-examples", type=int, default=64)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--patch-size", type=int, default=160)
    parser.add_argument("--init-patch")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.train_examples = min(args.train_examples, 8)
        args.eval_examples = min(args.eval_examples, 8)
        args.epochs = 1
    config = MechanismAwarePatchConfig(
        output_dir=args.output_dir,
        device=args.device,
        train_examples=args.train_examples,
        eval_examples=args.eval_examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patch_size=args.patch_size,
        init_patch=args.init_patch,
    )
    print(run_mechanism_aware_patch(config))


if __name__ == "__main__":
    main()
