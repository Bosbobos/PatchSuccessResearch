from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import Tensor, nn

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class DifferentiableYoloLoss(v8DetectionLoss):
    """Ultralytics detection objective retaining gradients per component."""

    def __call__(
        self, preds, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [
                feature.view(feats[0].shape[0], self.no, -1)
                for feature in feats
            ],
            2,
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        image_size = (
            torch.tensor(
                feats[0].shape[2:], device=self.device, dtype=dtype
            )
            * self.stride[0]
        )
        anchor_points, stride_tensor = make_anchors(
            feats, self.stride, 0.5
        )
        targets = torch.cat(
            (
                batch["batch_idx"].view(-1, 1),
                batch["cls"].view(-1, 1),
                batch["bboxes"],
            ),
            1,
        )
        targets = self.preprocess(
            targets,
            batch_size,
            scale_tensor=image_size[[1, 0, 1, 0]],
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum()
            / target_scores_sum
        )
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        weighted = loss * batch_size
        return weighted.sum(), {
            "loss_box": weighted[0],
            "loss_cls": weighted[1],
            "loss_dfl": weighted[2],
        }


def freeze_training_state(model: nn.Module) -> None:
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(
            module,
            (
                nn.modules.batchnorm._BatchNorm,
                nn.Dropout,
                nn.Dropout2d,
                nn.Dropout3d,
            ),
        ):
            module.eval()


def list_images(root: Path) -> list[Path]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {root}")
    return paths


def load_uint8(path: Path, image_size: int) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize(
                (image_size, image_size), Image.Resampling.BICUBIC
            )
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def make_global_batches(
    paths: list[Path], batch_size: int
) -> list[list[Path]]:
    return [
        paths[start : start + batch_size]
        for start in range(0, len(paths), batch_size)
    ]


def detections_to_batch(
    detections: list[Tensor],
    *,
    device: torch.device,
    image_size: int,
) -> dict[str, Tensor]:
    batch_indices: list[Tensor] = []
    classes: list[Tensor] = []
    boxes: list[Tensor] = []
    for image_index, detection in enumerate(detections):
        if not len(detection):
            continue
        boxes_xyxy = detection[:, :4].detach()
        class_ids = detection[:, 5:6].detach()
        xywh = boxes_xyxy.clone()
        xywh[:, 2:] -= xywh[:, :2]
        xywh[:, :2] += xywh[:, 2:] / 2.0
        xywh /= float(image_size)
        batch_indices.append(
            torch.full(
                (len(xywh),),
                image_index,
                device=device,
                dtype=torch.float32,
            )
        )
        classes.append(class_ids.to(dtype=torch.float32))
        boxes.append(xywh.to(dtype=torch.float32))
    if not boxes:
        return {
            "batch_idx": torch.zeros(0, device=device),
            "cls": torch.zeros((0, 1), device=device),
            "bboxes": torch.zeros((0, 4), device=device),
        }
    return {
        "batch_idx": torch.cat(batch_indices),
        "cls": torch.cat(classes),
        "bboxes": torch.cat(boxes),
    }


def brightness_for(
    seed: int, iteration: int, global_batch_index: int
) -> float:
    mixed = (
        int(seed) * 1_000_003
        + int(iteration) * 100_003
        + int(global_batch_index) * 1_009
    )
    return random.Random(mixed).uniform(0.7, 1.0)


def quantized_brightness(
    images: Tensor, brightness: float, learning_rate: float
) -> Tensor:
    transformed = images * float(brightness)
    quantized = (
        torch.round(transformed / float(learning_rate))
        * float(learning_rate)
    )
    return transformed + (quantized - transformed).detach()


def save_surface(surface: Tensor, path: Path) -> None:
    array = (
        surface.detach()
        .clamp(0.0, 1.0)
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray((array * 255.0).astype(np.uint8)).save(path)


def write_history(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distributed_main(args: argparse.Namespace) -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    paths = list_images(args.train_root)
    if args.max_images:
        paths = paths[: args.max_images]
    if len(paths) != args.expected_images:
        raise RuntimeError(
            f"Expected {args.expected_images} images, found {len(paths)}"
        )
    global_batches = make_global_batches(paths, args.batch_size)
    assigned = [
        (index, batch)
        for index, batch in enumerate(global_batches)
        if index % world_size == rank
    ]
    resident_batches = [
        (
            index,
            torch.stack(
                [
                    load_uint8(path, args.image_size)
                    for path in batch_paths
                ]
            ),
        )
        for index, batch_paths in assigned
    ]

    detector = YOLO(str(args.weights)).model.to(device)
    detector.args = DetectionPredictor().args
    freeze_training_state(detector)
    criterion = DifferentiableYoloLoss(detector)
    criterion.hyp.box = 7.5
    criterion.hyp.cls = 0.5
    criterion.hyp.dfl = 1.5

    surface = (
        (
            torch.randint(
                0,
                255,
                (3, args.surface_size, args.surface_size),
                device=device,
                generator=torch.Generator(device=device).manual_seed(
                    args.seed
                ),
            ).float()
            / 255.0
        )
        .detach()
        .requires_grad_(True)
    )
    with torch.no_grad():
        dist.broadcast(surface, src=0)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    history: list[dict] = []
    started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        active_started = time.perf_counter()
        if surface.grad is not None:
            surface.grad.zero_()
        local_components = torch.zeros(3, device=device)
        local_targets = torch.zeros(1, device=device)

        for global_batch_index, images_uint8 in resident_batches:
            images = images_uint8.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            images /= 255.0
            brightness = brightness_for(
                args.seed, iteration, global_batch_index
            )
            clean = quantized_brightness(
                images, brightness, args.learning_rate
            )

            detector.eval()
            with torch.no_grad():
                clean_output = detector(clean)
                clean_predictions = (
                    clean_output[0]
                    if isinstance(clean_output, (tuple, list))
                    else clean_output
                )
                detections = non_max_suppression(
                    clean_predictions,
                    conf_thres=args.target_confidence,
                    iou_thres=args.target_iou,
                    max_det=args.target_max_det,
                )
            local_targets += sum(len(item) for item in detections)

            freeze_training_state(detector)
            changed = images.clone()
            changed[
                :, :, : args.surface_size, : args.surface_size
            ] = surface[None]
            changed = quantized_brightness(
                changed, brightness, args.learning_rate
            )
            target = detections_to_batch(
                detections,
                device=device,
                image_size=args.image_size,
            )
            predictions = detector(changed)
            _, components = criterion(predictions, target)
            score = (
                components["loss_box"]
                + components["loss_cls"]
                + components["loss_dfl"]
            )
            score.backward()
            local_components += torch.stack(
                [
                    components["loss_box"].detach(),
                    components["loss_cls"].detach(),
                    components["loss_dfl"].detach(),
                ]
            )

        if surface.grad is None:
            raise RuntimeError("Surface gradient is missing")
        dist.all_reduce(surface.grad, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_components, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_targets, op=dist.ReduceOp.SUM)
        with torch.no_grad():
            surface.add_(
                float(args.learning_rate) * surface.grad.sign()
            )
            surface.clamp_(0.0, 1.0)

        torch.cuda.synchronize()
        active_seconds = time.perf_counter() - active_started
        active_tensor = torch.tensor(
            [active_seconds], device=device, dtype=torch.float64
        )
        dist.all_reduce(active_tensor, op=dist.ReduceOp.MAX)
        active_seconds = float(active_tensor.item())

        if rank == 0:
            elapsed = time.perf_counter() - started
            projected_total = (
                elapsed / iteration * args.iterations
                if iteration
                else math.nan
            )
            row = {
                "iteration": iteration,
                "loss_box_sum": float(local_components[0].item()),
                "loss_cls_sum": float(local_components[1].item()),
                "loss_dfl_sum": float(local_components[2].item()),
                "target_count": int(local_targets.item()),
                "active_seconds": active_seconds,
                "elapsed_seconds": elapsed,
                "eta_seconds": max(0.0, projected_total - elapsed),
            }
            history.append(row)
            if (
                iteration == 1
                or iteration % args.log_interval == 0
                or iteration == args.iterations
            ):
                print(
                    f"iteration={iteration:04d}/{args.iterations} "
                    f"box={row['loss_box_sum']:.3f} "
                    f"cls={row['loss_cls_sum']:.3f} "
                    f"dfl={row['loss_dfl_sum']:.3f} "
                    f"targets={row['target_count']} "
                    f"active={active_seconds:.2f}s "
                    f"elapsed={elapsed / 3600:.2f}h "
                    f"eta={row['eta_seconds'] / 3600:.2f}h",
                    flush=True,
                )
            if (
                iteration % args.save_interval == 0
                or iteration == args.iterations
            ):
                save_surface(
                    surface, args.output_dir / "latest_surface.png"
                )
                torch.save(
                    {
                        "surface": surface.detach().cpu(),
                        "iteration": iteration,
                        "metrics": row,
                    },
                    args.output_dir / "latest_state.pt",
                )
                write_history(
                    args.output_dir / "training_history.csv", history
                )

        sleep_seconds = active_seconds * (
            1.0 / args.duty_cycle - 1.0
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        dist.barrier()

    if rank == 0:
        save_surface(surface, args.output_dir / "final_surface.png")
        torch.save(
            {
                "surface": surface.detach().cpu(),
                "iteration": args.iterations,
                "metrics": history[-1],
            },
            args.output_dir / "final_state.pt",
        )
        metadata = {
            "version": 1,
            "method": "distributed_full_pool_sign_alignment",
            "world_size": world_size,
            "train_images": len(paths),
            "global_batches": len(global_batches),
            "batch_size": args.batch_size,
            "iterations": args.iterations,
            "image_size": args.image_size,
            "surface_size": args.surface_size,
            "position_xy": [0, 0],
            "learning_rate": args.learning_rate,
            "brightness_range": [0.7, 1.0],
            "rotation_weights": [1, 0, 0, 0],
            "crop_range": [0, 0],
            "sample_size": 1,
            "selected_losses": [
                "loss_cls",
                "loss_box",
                "loss_dfl",
            ],
            "target_source": "dynamic_reference_predictions",
            "target_confidence": args.target_confidence,
            "target_iou": args.target_iou,
            "target_max_det": args.target_max_det,
            "duty_cycle": args.duty_cycle,
            "seed": args.seed,
            "weights": str(args.weights),
            "train_root": str(args.train_root),
            "wall_seconds": time.perf_counter() - started,
            "last_metrics": history[-1],
        }
        (args.output_dir / "completed.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )
        print(
            f"Completed: {args.output_dir / 'completed.json'}",
            flush=True,
        )
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed full-pool tactile surface alignment."
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=500)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--surface-size", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--target-confidence", type=float, default=0.25)
    parser.add_argument("--target-iou", type=float, default=0.7)
    parser.add_argument("--target-max-det", type=int, default=300)
    parser.add_argument("--duty-cycle", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=10)
    args = parser.parse_args()
    if not 0 < args.duty_cycle <= 1:
        parser.error("--duty-cycle must be in (0, 1]")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.iterations != 1000:
        parser.error("This protocol requires --iterations 1000")
    if args.surface_size != 160:
        parser.error("This protocol requires --surface-size 160")
    return args


if __name__ == "__main__":
    distributed_main(parse_args())
