from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader


_REMOTE_VENDOR = Path(__file__).resolve().parent.parent / ".vendor"
if _REMOTE_VENDOR.is_dir():
    sys.path.insert(0, str(_REMOTE_VENDOR))


try:
    from depatch_yolo11 import (
        DePatchTrainer,
        PatchTrainerConfig,
        non_printability_loss,
        save_patch_png,
        save_patch_pt,
        total_variation_loss,
    )
    from robust_dpatch_yolo11 import RobustDPatchTrainer
except ImportError:
    # Neutral filenames used in the shared remote workspace.
    from surface_dropout import (
        DePatchTrainer,
        PatchTrainerConfig,
        non_printability_loss,
        save_patch_png,
        save_patch_pt,
        total_variation_loss,
    )
    from surface_transform import RobustDPatchTrainer


def _safe_release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps_backend = getattr(torch.backends, "mps", None)
    if (
        mps_backend is not None
        and mps_backend.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "empty_cache")
    ):
        torch.mps.empty_cache()


# Upstream currently calls torch.mps.empty_cache() whenever the symbol exists,
# which raises on CUDA-only builds. Keep the imported trainers otherwise intact.
DePatchTrainer.release_memory = staticmethod(_safe_release_memory)
RobustDPatchTrainer.release_memory = staticmethod(_safe_release_memory)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SCENE_RE = re.compile(r"(?:train2017_)?image(\d+)", re.IGNORECASE)
VARIANT_ALIASES = {
    "depatch": "depatch",
    "robust_dpatch": "robust_dpatch",
    "adversarial_patch": "adversarial_patch",
    "surface_dropout": "depatch",
    "surface_transform": "robust_dpatch",
    "surface_baseline": "adversarial_patch",
}


class AdversarialPatchTrainer(DePatchTrainer):
    """Plain detector-patch baseline: EOT colour/noise plus TV, without DePatch PDS/TPS."""

    def __init__(self, config: PatchTrainerConfig):
        config.decoupling = False
        config.enable_tps = False
        config.enable_tc = False
        super().__init__(config)

    def pds_params(self, epoch: int | None = None) -> tuple[int, float]:
        return 1, 0.0


def _scene_id(path: Path) -> str:
    match = SCENE_RE.search(path.stem)
    if match:
        return f"coco:{int(match.group(1))}"
    # Conservative fallback: files with an unknown naming scheme remain separate.
    return f"file:{path.stem}"


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(dataset_dir: Path) -> list[Path]:
    paths = sorted(
        path.resolve()
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {dataset_dir}")
    return paths


def prepare_scene_disjoint_split(
    dataset_dir: Path,
    split_root: Path,
    *,
    train_images: int,
    eval_images: int,
    seed: int,
) -> dict:
    manifest_path = split_root / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        expected = {
            "dataset_dir": str(dataset_dir.resolve()),
            "train_images": train_images,
            "eval_images": eval_images,
            "seed": seed,
        }
        actual = {key: payload[key] for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"Existing split does not match requested protocol: {actual} != {expected}"
            )
        return payload

    all_paths = discover_images(dataset_dir)
    grouped: dict[str, list[Path]] = {}
    for path in all_paths:
        grouped.setdefault(_scene_id(path), []).append(path)
    required = train_images + eval_images
    if len(grouped) < required:
        raise RuntimeError(
            f"Need {required} unique scenes, found only {len(grouped)} in {dataset_dir}"
        )

    scene_ids = sorted(grouped, key=lambda value: _stable_key(seed, value))

    def representative(scene_id: str) -> Path:
        return min(grouped[scene_id], key=lambda p: _stable_key(seed, p.name))

    train_scene_ids = scene_ids[:train_images]
    eval_scene_ids = scene_ids[train_images : train_images + eval_images]
    train_paths = [representative(scene_id) for scene_id in train_scene_ids]
    eval_paths = [representative(scene_id) for scene_id in eval_scene_ids]

    split_root.mkdir(parents=True, exist_ok=True)
    for split_name, selected_paths in (("train", train_paths), ("eval", eval_paths)):
        target_dir = split_root / split_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(selected_paths):
            link = target_dir / f"{index:04d}_{source.name}"
            if not link.exists():
                link.symlink_to(source)

    payload = {
        "version": 1,
        "dataset_dir": str(dataset_dir.resolve()),
        "train_images": train_images,
        "eval_images": eval_images,
        "seed": seed,
        "selection": "sha256(seed, scene_id), one deterministic crop per COCO image_id",
        "scene_disjoint": True,
        "train": [
            {
                "scene_id": scene_id,
                "path": str(path),
                "sha256": _sha256(path),
            }
            for scene_id, path in zip(train_scene_ids, train_paths)
        ],
        "eval": [
            {
                "scene_id": scene_id,
                "path": str(path),
                "sha256": _sha256(path),
            }
            for scene_id, path in zip(eval_scene_ids, eval_paths)
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_ultralytics_geometry_cache(model: torch.nn.Module) -> None:
    """Force Detect heads to rebuild anchors outside torch.inference_mode()."""
    for module in model.modules():
        if module.__class__.__name__ in {"Detect", "Segment", "Pose", "OBB"} and hasattr(
            module, "shape"
        ):
            module.shape = None


def _duty_cycle_sleep(active_seconds: float, duty_cycle: float) -> None:
    if duty_cycle >= 1.0:
        return
    sleep_seconds = active_seconds * (1.0 / duty_cycle - 1.0)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)


def cache_clean_boxes_shared(
    trainer: DePatchTrainer,
    *,
    cache_path: Path,
    duty_cycle: float,
    batch_size: int,
    num_workers: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text())
            if payload["weights"] != str(Path(trainer.weights).expanduser().resolve()):
                raise RuntimeError("Shared target cache was built with different detector weights")
            if payload["class_id"] != trainer.config.class_id:
                raise RuntimeError("Shared target cache was built for a different class_id")
            if payload["conf_thres"] != trainer.config.conf_thres:
                raise RuntimeError("Shared target cache was built with a different conf_thres")
            trainer.clean_cache.update(
                {
                    path: np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
                    for path, boxes in payload["boxes"].items()
                }
            )
            print(f"Loaded {len(trainer.clean_cache)} shared clean targets", flush=True)
            return

        loader = DataLoader(
            trainer.train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=trainer.device.type == "cuda",
        )
        total = 0
        for images, paths in loader:
            active_started = time.perf_counter()
            detected = trainer.detect_person_boxes(images)
            _cuda_sync(trainer.device)
            active_seconds = time.perf_counter() - active_started
            for path, boxes in zip(paths, detected):
                trainer.clean_cache[path] = boxes
            total += len(paths)
            print(f"clean_targets={total}/{len(trainer.train_dataset)}", flush=True)
            _duty_cycle_sleep(active_seconds, duty_cycle)
        payload = {
            "version": 1,
            "weights": str(Path(trainer.weights).expanduser().resolve()),
            "class_id": trainer.config.class_id,
            "conf_thres": trainer.config.conf_thres,
            "boxes": {
                path: boxes.astype(float).tolist()
                for path, boxes in trainer.clean_cache.items()
            },
        }
        temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False))
        temporary_path.replace(cache_path)
        print(f"Saved {len(trainer.clean_cache)} shared clean targets", flush=True)


def place_fixed_top_left(
    trainer: DePatchTrainer,
    images: Tensor,
    *,
    augment: bool,
    size: int,
) -> Tensor:
    output = images.clone()
    target_size = min(size, images.shape[-2], images.shape[-1])
    for index in range(images.shape[0]):
        if augment:
            surface, alpha = trainer.transform_patch_for_training()
        else:
            surface, alpha = trainer.patch, None
        trainer._overlay_patch_at_(
            output,
            index,
            cx=target_size / 2.0,
            cy=target_size / 2.0,
            target_size=target_size,
            angle_degrees=0.0,
            patch_tensor=surface,
            alpha_mask=alpha,
        )
    return output.clamp(0.0, 1.0)


def build_trainer(args: argparse.Namespace, output_dir: Path) -> DePatchTrainer:
    steps_per_epoch = math.ceil(args.train_images / args.batch_size)
    virtual_epochs = max(1, math.ceil(args.steps / steps_per_epoch))
    config = PatchTrainerConfig(
        train_dir=str(args.split_root / "train"),
        val_dir=str(args.split_root / "eval"),
        weights=args.weights,
        device=args.device,
        output_dir=str(output_dir),
        epochs=virtual_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_size_ratio=args.patch_size / 640.0,
        fixed_patch_size=args.patch_size,
        placement_mode="random_image",
        class_id=args.class_id,
        conf_thres=args.conf_thres,
        nps_weight=args.nps_weight,
        tv_weight=args.tv_weight,
        seed=args.seed,
        num_workers=args.num_workers,
        cleanup_batch_interval=args.cleanup_batch_interval,
        eval_interval=virtual_epochs + 1,
    )
    if args.method == "depatch":
        return DePatchTrainer(config)
    if args.method == "robust_dpatch":
        return RobustDPatchTrainer(config)
    if args.method == "adversarial_patch":
        return AdversarialPatchTrainer(config)
    raise ValueError(args.method)


def _write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> Path:
    split_payload = prepare_scene_disjoint_split(
        args.dataset_dir,
        args.split_root,
        train_images=args.train_images,
        eval_images=args.eval_images,
        seed=args.seed,
    )
    output_dir = args.output_root / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_path = output_dir / "completed.json"
    if completed_path.exists() and not args.force:
        print(f"Already complete: {completed_path}", flush=True)
        return output_dir

    trainer = build_trainer(args, output_dir)
    initial_hash = hashlib.sha256(
        trainer.patch.detach().cpu().numpy().astype(np.float32).tobytes()
    ).hexdigest()
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        trainer.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=trainer.device.type == "cuda",
        generator=loader_generator,
    )
    optimizer = torch.optim.Adam([trainer.patch_logits], lr=args.lr)

    print("Preparing shared clean target boxes...", flush=True)
    cache_clean_boxes_shared(
        trainer,
        cache_path=args.split_root / "clean_target_boxes.json",
        duty_cycle=args.duty_cycle,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    reset_ultralytics_geometry_cache(trainer.model)
    history: list[dict] = []
    global_step = 0
    epoch = 0
    started = time.perf_counter()
    while global_step < args.steps:
        epoch += 1
        trainer.current_epoch = epoch
        for images, paths in loader:
            if global_step >= args.steps:
                break
            active_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            clean_boxes = [
                trainer.get_or_detect_boxes(trainer.train_dataset, images[i], path)
                for i, path in enumerate(paths)
            ]
            images = images.to(trainer.device, non_blocking=True)
            transformed = place_fixed_top_left(
                trainer,
                images,
                augment=True,
                size=args.patch_size,
            )
            objective = trainer.suppression_loss(transformed, clean_boxes)
            nps = (
                non_printability_loss(trainer.patch, trainer.printable_colors)
                if trainer.config.nps_weight
                else trainer.patch.new_tensor(0.0)
            )
            tv = (
                total_variation_loss(trainer.patch)
                if trainer.config.tv_weight
                else trainer.patch.new_tensor(0.0)
            )
            loss = objective + trainer.config.nps_weight * nps + trainer.config.tv_weight * tv
            loss.backward()
            optimizer.step()
            _cuda_sync(trainer.device)
            active_seconds = time.perf_counter() - active_started
            global_step += 1

            row = {
                "step": global_step,
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "objective": float(objective.detach().cpu()),
                "nps": float(nps.detach().cpu()),
                "tv": float(tv.detach().cpu()),
                "active_seconds": active_seconds,
            }
            history.append(row)
            if global_step == 1 or global_step % args.log_interval == 0 or global_step == args.steps:
                elapsed = time.perf_counter() - started
                eta = elapsed / global_step * (args.steps - global_step)
                print(
                    f"step={global_step:04d}/{args.steps} "
                    f"loss={row['loss']:.6f} objective={row['objective']:.6f} "
                    f"active={active_seconds:.3f}s elapsed={elapsed / 60:.1f}m "
                    f"eta={eta / 60:.1f}m",
                    flush=True,
                )
            if global_step % args.save_interval == 0 or global_step == args.steps:
                metrics = {**row, "variant": args.variant}
                save_patch_png(trainer.patch, output_dir / "latest.png")
                save_patch_pt(
                    trainer.patch,
                    output_dir / "latest.pt",
                    trainer.config,
                    epoch,
                    metrics,
                )
                _write_history(output_dir / "history.csv", history)

            del images, transformed, objective, nps, tv, loss
            trainer.release_memory()
            _duty_cycle_sleep(active_seconds, args.duty_cycle)

    final_metrics = history[-1]
    save_patch_png(trainer.patch, output_dir / "final.png")
    save_patch_pt(
        trainer.patch,
        output_dir / "final.pt",
        trainer.config,
        epoch,
        final_metrics,
    )
    completed = {
        "version": 1,
        "variant": args.variant,
        "steps": args.steps,
        "patch_size": [args.patch_size, args.patch_size],
        "position_xy": [0, 0],
        "initial_patch_sha256": initial_hash,
        "train_scene_ids_sha256": hashlib.sha256(
            "\n".join(row["scene_id"] for row in split_payload["train"]).encode("utf-8")
        ).hexdigest(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "trainer_config": asdict(trainer.config),
        "last_metrics": final_metrics,
        "wall_seconds": time.perf_counter() - started,
    }
    completed_path.write_text(json.dumps(completed, indent=2, ensure_ascii=False))
    print(f"Completed: {completed_path}", flush=True)
    return output_dir


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified fixed-corner training protocol for three defensive stress surfaces."
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(VARIANT_ALIASES),
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--train-images", type=int, default=500)
    parser.add_argument("--eval-images", type=int, default=300)
    parser.add_argument("--patch-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--nps-weight", type=float, default=0.0)
    parser.add_argument("--tv-weight", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cleanup-batch-interval", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument(
        "--duty-cycle",
        type=float,
        default=0.20,
        help="Maximum active CUDA wall-time fraction for this process.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for name in ("dataset_dir", "split_root", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not args.dataset_dir.is_dir():
        parser.error(f"Dataset directory not found: {args.dataset_dir}")
    if args.steps <= 0 or args.train_images <= 0 or args.eval_images <= 0:
        parser.error("steps/train-images/eval-images must be positive")
    if not 0.0 < args.duty_cycle <= 1.0:
        parser.error("duty-cycle must be in (0, 1]")
    if args.patch_size != 160:
        parser.error("This controlled protocol requires --patch-size 160")
    args.method = VARIANT_ALIASES[args.variant]
    return args


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
