from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


_REMOTE_VENDOR = Path(__file__).resolve().parent.parent / ".vendor"
if _REMOTE_VENDOR.is_dir():
    sys.path.insert(0, str(_REMOTE_VENDOR))

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


VARIANTS = {
    "adversarial_patch": ("loss_cls",),
    "dpatch": ("loss_box", "loss_cls", "loss_dfl"),
    "class_native": ("loss_cls",),
    "class_box_native": ("loss_box", "loss_cls", "loss_dfl"),
}


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path):
        self.paths = sorted(
            path
            for path in root.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        if array.shape != (640, 640, 3):
            raise ValueError(f"Expected 640x640 RGB image: {path}")
        return (
            torch.from_numpy(array).permute(2, 0, 1).contiguous(),
            str(path),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def duty_cycle_sleep(active_seconds: float, duty_cycle: float) -> None:
    if duty_cycle < 1.0:
        time.sleep(active_seconds * (1.0 / duty_cycle - 1.0))


def save_png(patch: Tensor, path: Path) -> None:
    array = (
        patch.detach()
        .clamp(0.0, 1.0)
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray((array * 255.0).round().astype(np.uint8)).save(path)


def write_history(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


class DifferentiableYoloLoss(v8DetectionLoss):
    """Ultralytics v8+ detection loss retaining gradients per component."""

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


def target_batch(
    paths: list[str] | tuple[str, ...],
    targets: dict[str, dict],
    *,
    device: torch.device,
    image_size: int,
) -> dict[str, Tensor]:
    batch_indices: list[Tensor] = []
    classes: list[Tensor] = []
    boxes: list[Tensor] = []
    for image_index, path in enumerate(paths):
        payload = targets[str(path)]
        boxes_xyxy = torch.as_tensor(
            payload["boxes"], device=device, dtype=torch.float32
        ).reshape(-1, 4)
        class_ids = torch.as_tensor(
            payload["classes"], device=device, dtype=torch.float32
        ).reshape(-1, 1)
        if not len(boxes_xyxy):
            continue
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
        classes.append(class_ids)
        boxes.append(xywh)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled ART-style native YOLO objective protocol."
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--duty-cycle", type=float, default=0.20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = args.output_root / args.variant
    completed_path = output_dir / "completed.json"
    if completed_path.exists() and not args.force:
        print(f"Already complete: {completed_path}", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageFolderDataset(args.split_root / "train")
    if len(dataset) != 500:
        raise RuntimeError(f"Expected 500 train images, found {len(dataset)}")
    target_payload = json.loads(args.target_cache.read_text())
    targets = target_payload["targets"]
    missing = [str(path) for path in dataset.paths if str(path) not in targets]
    if missing:
        raise RuntimeError(f"Target cache misses {len(missing)} paths")

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    detector = YOLO(str(args.weights)).model.to(device)
    # This is the same initialization performed by ART's
    # PyTorchYoloLossWrapper for Ultralytics v8+ models.
    detector.args = DetectionPredictor().args
    freeze_training_state(detector)
    criterion = DifferentiableYoloLoss(detector)

    patch = nn.Parameter(
        torch.full(
            (3, args.patch_size, args.patch_size),
            0.5,
            device=device,
        )
    )
    optimizer = torch.optim.Adam([patch], lr=args.lr)
    selected = VARIANTS[args.variant]
    history: list[dict] = []
    started = time.perf_counter()
    step = 0
    epoch = 0
    while step < args.steps:
        epoch += 1
        for images, paths in loader:
            if step >= args.steps:
                break
            active_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device, non_blocking=True)
            changed = images.clone()
            changed[
                :, :, : args.patch_size, : args.patch_size
            ] = patch.clamp(0.0, 1.0)[None]
            batch = target_batch(
                paths,
                targets,
                device=device,
                image_size=int(images.shape[-1]),
            )
            predictions = detector(changed)
            _, components = criterion(predictions, batch)
            attack_score = sum(components[name] for name in selected)
            optimization_loss = -attack_score
            optimization_loss.backward()
            optimizer.step()
            with torch.no_grad():
                patch.clamp_(0.0, 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            active_seconds = time.perf_counter() - active_started
            step += 1
            row = {
                "step": step,
                "epoch": epoch,
                "attack_score": float(attack_score.detach().cpu()),
                "loss_box": float(components["loss_box"].detach().cpu()),
                "loss_cls": float(components["loss_cls"].detach().cpu()),
                "loss_dfl": float(components["loss_dfl"].detach().cpu()),
                "active_seconds": active_seconds,
            }
            history.append(row)
            if (
                step == 1
                or step % args.log_interval == 0
                or step == args.steps
            ):
                elapsed = time.perf_counter() - started
                eta = elapsed / step * (args.steps - step)
                print(
                    f"step={step:04d}/{args.steps} "
                    f"score={row['attack_score']:.5f} "
                    f"box={row['loss_box']:.5f} "
                    f"cls={row['loss_cls']:.5f} "
                    f"dfl={row['loss_dfl']:.5f} "
                    f"active={active_seconds:.3f}s "
                    f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                    flush=True,
                )
            if step % args.save_interval == 0 or step == args.steps:
                save_png(patch, output_dir / "latest.png")
                torch.save(
                    {
                        "patch": patch.detach().cpu(),
                        "step": step,
                        "variant": args.variant,
                        "selected_losses": selected,
                        "metrics": row,
                    },
                    output_dir / "latest.pt",
                )
                write_history(output_dir / "history.csv", history)
            del images, changed, predictions, batch, components
            if step % 100 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            duty_cycle_sleep(active_seconds, args.duty_cycle)

    save_png(patch, output_dir / "final.png")
    torch.save(
        {
            "patch": patch.detach().cpu(),
            "step": step,
            "variant": args.variant,
            "selected_losses": selected,
            "metrics": history[-1],
        },
        output_dir / "final.pt",
    )
    completed = {
        "version": 1,
        "variant": args.variant,
        "protocol": "ART AdversarialPatchPyTorch optimizer semantics with differentiable native YOLO losses",
        "steps_are_minibatch_updates": True,
        "train_images": len(dataset),
        "patch_size": [args.patch_size, args.patch_size],
        "position_xy": [0, 0],
        "selected_losses": list(selected),
        "optimizer": "Adam",
        "lr": args.lr,
        "augmentations": [],
        "regularizers": [],
        "clean_pseudo_targets": str(args.target_cache),
        "last_metrics": history[-1],
        "wall_seconds": time.perf_counter() - started,
    }
    completed_path.write_text(
        json.dumps(completed, indent=2, ensure_ascii=False)
    )
    print(f"Completed: {completed_path}", flush=True)


if __name__ == "__main__":
    main()
