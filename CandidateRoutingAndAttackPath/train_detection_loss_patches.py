from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader


_REMOTE_VENDOR = Path(__file__).resolve().parent.parent / ".vendor"
if _REMOTE_VENDOR.is_dir():
    sys.path.insert(0, str(_REMOTE_VENDOR))

try:
    from depatch_yolo11 import ImageFolderDataset
    from train_fixed_corner_patches import prepare_scene_disjoint_split
except ImportError:
    from surface_dropout import ImageFolderDataset
    from surface_protocol import prepare_scene_disjoint_split

from ultralytics import YOLO


VARIANT_ALIASES = {
    "person_adv_patch": "person_adv_patch",
    "general_adv_patch": "general_adv_patch",
    "person_dpatch": "person_dpatch",
    "general_dpatch": "general_dpatch",
    "class0_score": "person_adv_patch",
    "allclass_score": "general_adv_patch",
    "class0_joint": "person_dpatch",
    "allclass_joint": "general_dpatch",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {requested}")
    return device


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def duty_cycle_sleep(active_seconds: float, duty_cycle: float) -> None:
    if duty_cycle >= 1.0:
        return
    sleep_seconds = active_seconds * (1.0 / duty_cycle - 1.0)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)


def save_png(patch: Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        patch.detach()
        .clamp(0.0, 1.0)
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray((array * 255.0).round().astype(np.uint8)).save(path)


def save_checkpoint(
    patch: Tensor,
    logits: Tensor,
    path: Path,
    *,
    step: int,
    config: dict,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "patch": patch.detach().clamp(0.0, 1.0).cpu(),
            "patch_logits": logits.detach().cpu(),
            "step": step,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_predictions(model: nn.Module, images: Tensor) -> tuple[Tensor, Tensor]:
    output = model(images)
    predictions = output[0] if isinstance(output, (tuple, list)) else output
    if predictions.ndim != 3 or predictions.shape[1] < 5:
        raise RuntimeError(f"Unexpected detector output: {tuple(predictions.shape)}")
    boxes_xywh = predictions[:, :4, :].permute(0, 2, 1)
    scores = predictions[:, 4:, :]
    x, y, width, height = boxes_xywh.unbind(dim=-1)
    boxes_xyxy = torch.stack(
        (
            x - width / 2.0,
            y - height / 2.0,
            x + width / 2.0,
            y + height / 2.0,
        ),
        dim=-1,
    )
    return boxes_xyxy, scores


def pairwise_iou(boxes: Tensor, targets: Tensor) -> Tensor:
    left_top = torch.maximum(boxes[:, None, :2], targets[None, :, :2])
    right_bottom = torch.minimum(boxes[:, None, 2:], targets[None, :, 2:])
    intersection_wh = (right_bottom - left_top).clamp(min=0.0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area_boxes = (
        (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
        * (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
    )
    area_targets = (
        (targets[:, 2] - targets[:, 0]).clamp(min=0.0)
        * (targets[:, 3] - targets[:, 1]).clamp(min=0.0)
    )
    union = area_boxes[:, None] + area_targets[None, :] - intersection
    return intersection / union.clamp(min=1e-7)


def aligned_ciou(boxes: Tensor, targets: Tensor) -> Tensor:
    left_top = torch.maximum(boxes[:, :2], targets[:, :2])
    right_bottom = torch.minimum(boxes[:, 2:], targets[:, 2:])
    intersection_wh = (right_bottom - left_top).clamp(min=0.0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]

    box_wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-7)
    target_wh = (targets[:, 2:] - targets[:, :2]).clamp(min=1e-7)
    union = (
        box_wh[:, 0] * box_wh[:, 1]
        + target_wh[:, 0] * target_wh[:, 1]
        - intersection
    )
    iou = intersection / union.clamp(min=1e-7)

    box_center = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    target_center = (targets[:, :2] + targets[:, 2:]) / 2.0
    center_distance = ((box_center - target_center) ** 2).sum(dim=1)
    enclosing_left_top = torch.minimum(boxes[:, :2], targets[:, :2])
    enclosing_right_bottom = torch.maximum(boxes[:, 2:], targets[:, 2:])
    enclosing_diagonal = (
        (enclosing_right_bottom - enclosing_left_top) ** 2
    ).sum(dim=1).clamp(min=1e-7)

    v = (
        4.0
        / math.pi**2
        * (
            torch.atan(target_wh[:, 0] / target_wh[:, 1])
            - torch.atan(box_wh[:, 0] / box_wh[:, 1])
        )
        ** 2
    )
    with torch.no_grad():
        alpha = v / (1.0 - iou + v).clamp(min=1e-7)
    return iou - center_distance / enclosing_diagonal - alpha * v


def overlay_top_left(images: Tensor, patch: Tensor) -> Tensor:
    output = images.clone()
    patch_height = min(patch.shape[-2], images.shape[-2])
    patch_width = min(patch.shape[-1], images.shape[-1])
    output[:, :, :patch_height, :patch_width] = patch[
        None, :, :patch_height, :patch_width
    ]
    return output


def confidence_loss(scores: Tensor, *, scope: str, topk: int) -> Tensor:
    if scope == "person":
        candidates = scores[:, 0, :]
    elif scope == "general":
        candidates = scores.flatten(start_dim=1)
    else:
        raise ValueError(scope)
    count = min(topk, candidates.shape[1])
    return candidates.topk(count, dim=1).values.mean()


def regression_ciou(
    boxes_batch: Tensor,
    scores_batch: Tensor,
    targets_batch: list[dict],
    *,
    scope: str,
    assignment_score_weight: float,
) -> tuple[Tensor, int]:
    matched_boxes: list[Tensor] = []
    matched_targets: list[Tensor] = []
    for image_index, target_payload in enumerate(targets_batch):
        target_boxes_np = np.asarray(target_payload["boxes"], dtype=np.float32).reshape(-1, 4)
        target_classes_np = np.asarray(target_payload["classes"], dtype=np.int64).reshape(-1)
        if scope == "person":
            keep = target_classes_np == 0
            target_boxes_np = target_boxes_np[keep]
            target_classes_np = target_classes_np[keep]
        if len(target_boxes_np) == 0:
            continue

        target_boxes = torch.as_tensor(
            target_boxes_np,
            device=boxes_batch.device,
            dtype=boxes_batch.dtype,
        )
        target_classes = torch.as_tensor(
            target_classes_np,
            device=boxes_batch.device,
            dtype=torch.long,
        )
        predicted_boxes = boxes_batch[image_index]
        ious = pairwise_iou(predicted_boxes, target_boxes)
        class_scores = scores_batch[image_index, target_classes, :].transpose(0, 1)
        assignment = (
            ious.detach() + assignment_score_weight * class_scores.detach()
        ).argmax(dim=0)
        matched_boxes.append(predicted_boxes[assignment])
        matched_targets.append(target_boxes)

    if not matched_boxes:
        return boxes_batch.new_tensor(0.0), 0
    predicted = torch.cat(matched_boxes, dim=0)
    targets = torch.cat(matched_targets, dim=0)
    return aligned_ciou(predicted, targets).mean(), len(targets)


def build_all_class_target_cache(
    dataset: ImageFolderDataset,
    *,
    cache_path: Path,
    weights: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    conf_thres: float,
    duty_cycle: float,
) -> dict[str, dict]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            if payload["weights"] != str(weights):
                raise RuntimeError("Target cache uses different detector weights")
            if payload["conf_thres"] != conf_thres:
                raise RuntimeError("Target cache uses a different confidence threshold")
            print(f"Loaded {len(payload['targets'])} shared all-class targets", flush=True)
            return payload["targets"]

        predictor = YOLO(str(weights))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        targets: dict[str, dict] = {}
        completed = 0
        for images, paths in loader:
            active_started = time.perf_counter()
            results = predictor.predict(
                images,
                imgsz=640,
                conf=conf_thres,
                device=str(device),
                verbose=False,
            )
            cuda_sync(device)
            active_seconds = time.perf_counter() - active_started
            for path, result in zip(paths, results):
                if result.boxes is None:
                    boxes: list = []
                    classes: list = []
                    confidences: list = []
                else:
                    boxes = result.boxes.xyxy.detach().cpu().float().tolist()
                    classes = result.boxes.cls.detach().cpu().long().tolist()
                    confidences = result.boxes.conf.detach().cpu().float().tolist()
                targets[path] = {
                    "boxes": boxes,
                    "classes": classes,
                    "confidences": confidences,
                }
            completed += len(paths)
            print(f"all_class_targets={completed}/{len(dataset)}", flush=True)
            duty_cycle_sleep(active_seconds, duty_cycle)

        payload = {
            "version": 1,
            "weights": str(weights),
            "conf_thres": conf_thres,
            "targets": targets,
        }
        temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False))
        temporary_path.replace(cache_path)
        del predictor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Saved {len(targets)} shared all-class targets", flush=True)
        return targets


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    prepare_scene_disjoint_split(
        args.dataset_dir,
        args.split_root,
        train_images=args.train_images,
        eval_images=args.eval_images,
        seed=args.seed,
    )
    dataset = ImageFolderDataset(args.split_root / "train")
    device = select_device(args.device)
    targets = build_all_class_target_cache(
        dataset,
        cache_path=args.split_root / "all_class_targets.json",
        weights=args.weights,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conf_thres=args.conf_thres,
        duty_cycle=args.duty_cycle,
    )

    output_dir = args.output_root / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = output_dir / "completed.json"
    if completed_path.exists() and not args.force:
        print(f"Already complete: {completed_path}", flush=True)
        return output_dir

    detector = YOLO(str(args.weights))
    model: nn.Module = detector.model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Target-cache construction and model loading may consume RNG state. Reset
    # immediately before initialization so all four variants start identically.
    seed_everything(args.seed)
    initial_patch = (
        torch.rand(3, args.patch_size, args.patch_size, device=device) * 0.10 + 0.45
    ).clamp(1e-4, 1.0 - 1e-4)
    patch_logits = nn.Parameter(torch.logit(initial_patch))
    initial_hash = hashlib.sha256(
        torch.sigmoid(patch_logits).detach().cpu().numpy().astype(np.float32).tobytes()
    ).hexdigest()
    optimizer = torch.optim.Adam([patch_logits], lr=args.lr)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )

    scope = "person" if args.method.startswith("person_") else "general"
    use_regression = args.method.endswith("dpatch")
    history: list[dict] = []
    step = 0
    epoch = 0
    started = time.perf_counter()
    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    while step < args.steps:
        epoch += 1
        for images, paths in loader:
            if step >= args.steps:
                break
            active_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device, non_blocking=True)
            patch = torch.sigmoid(patch_logits)
            modified = overlay_top_left(images, patch)
            boxes, scores = model_predictions(model, modified)
            loss_confidence = confidence_loss(scores, scope=scope, topk=args.topk)
            if use_regression:
                batch_targets = [targets[path] for path in paths]
                mean_ciou, target_count = regression_ciou(
                    boxes,
                    scores,
                    batch_targets,
                    scope=scope,
                    assignment_score_weight=args.assignment_score_weight,
                )
                loss = loss_confidence + args.box_weight * mean_ciou
            else:
                mean_ciou = loss_confidence.new_tensor(float("nan"))
                target_count = 0
                loss = loss_confidence
            loss.backward()
            optimizer.step()
            cuda_sync(device)
            active_seconds = time.perf_counter() - active_started
            step += 1

            row = {
                "step": step,
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "confidence_loss": float(loss_confidence.detach().cpu()),
                "mean_ciou": float(mean_ciou.detach().cpu()),
                "target_count": target_count,
                "active_seconds": active_seconds,
            }
            history.append(row)
            if step == 1 or step % args.log_interval == 0 or step == args.steps:
                elapsed = time.perf_counter() - started
                eta = elapsed / step * (args.steps - step)
                print(
                    f"step={step:04d}/{args.steps} loss={row['loss']:.6f} "
                    f"confidence={row['confidence_loss']:.6f} "
                    f"ciou={row['mean_ciou']:.6f} targets={target_count} "
                    f"active={active_seconds:.3f}s elapsed={elapsed / 60:.1f}m "
                    f"eta={eta / 60:.1f}m",
                    flush=True,
                )
            if step % args.save_interval == 0 or step == args.steps:
                save_png(torch.sigmoid(patch_logits), output_dir / "latest.png")
                save_checkpoint(
                    torch.sigmoid(patch_logits),
                    patch_logits,
                    output_dir / "latest.pt",
                    step=step,
                    config=config_payload,
                    metrics=row,
                )
                write_history(output_dir / "history.csv", history)

            del images, modified, boxes, scores, loss_confidence, mean_ciou, loss
            if step % args.cleanup_interval == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            duty_cycle_sleep(active_seconds, args.duty_cycle)

    final_patch = torch.sigmoid(patch_logits)
    save_png(final_patch, output_dir / "final.png")
    save_checkpoint(
        final_patch,
        patch_logits,
        output_dir / "final.pt",
        step=step,
        config=config_payload,
        metrics=history[-1],
    )
    completed = {
        "version": 1,
        "variant": args.variant,
        "method": args.method,
        "steps": step,
        "patch_size": [args.patch_size, args.patch_size],
        "position_xy": [0, 0],
        "loss_components": (
            ["confidence", "bbox_ciou"] if use_regression else ["confidence"]
        ),
        "augmentations": [],
        "regularizers": [],
        "initial_patch_sha256": initial_hash,
        "config": config_payload,
        "last_metrics": history[-1],
        "wall_seconds": time.perf_counter() - started,
    }
    completed_path.write_text(json.dumps(completed, indent=2, ensure_ascii=False))
    print(f"Completed: {completed_path}", flush=True)
    return output_dir


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-corner confidence and confidence-plus-regression protocol."
    )
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_ALIASES))
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--train-images", type=int, default=500)
    parser.add_argument("--eval-images", type=int, default=300)
    parser.add_argument("--patch-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--box-weight", type=float, default=1.0)
    parser.add_argument("--assignment-score-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--cleanup-interval", type=int, default=100)
    parser.add_argument("--duty-cycle", type=float, default=0.20)
    parser.add_argument("--allow-smoke-protocol", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for field in ("dataset_dir", "split_root", "output_root", "weights"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    args.method = VARIANT_ALIASES[args.variant]
    if not args.dataset_dir.is_dir():
        parser.error(f"Dataset not found: {args.dataset_dir}")
    if not args.weights.is_file():
        parser.error(f"Weights not found: {args.weights}")
    if args.patch_size != 160:
        parser.error("This protocol requires a 160x160 patch")
    if not args.allow_smoke_protocol and args.steps != 1000:
        parser.error("This protocol requires exactly 1000 optimizer steps")
    if not args.allow_smoke_protocol and args.train_images != 500:
        parser.error("This protocol requires exactly 500 training images")
    if not 0.0 < args.duty_cycle <= 1.0:
        parser.error("duty-cycle must be in (0, 1]")
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
