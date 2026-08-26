from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


_REMOTE_VENDOR = Path(__file__).resolve().parent.parent / ".vendor"
if _REMOTE_VENDOR.is_dir():
    sys.path.insert(0, str(_REMOTE_VENDOR))

from ultralytics import YOLO


def load_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def parse_patch(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--patch must be NAME=/path/to/final.png")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("patch name cannot be empty")
    return name, Path(raw_path).expanduser()


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.zeros(0, dtype=np.float32)
    left_top = np.maximum(box[None, :2], boxes[:, :2])
    right_bottom = np.minimum(box[None, 2:], boxes[:, 2:])
    wh = np.maximum(0.0, right_bottom - left_top)
    intersection = wh[:, 0] * wh[:, 1]
    area_a = max(0.0, float(box[2] - box[0])) * max(
        0.0, float(box[3] - box[1])
    )
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    return intersection / np.maximum(1e-9, area_a + area_b - intersection)


def result_arrays(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if result.boxes is None or len(result.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )
    return (
        result.boxes.xyxy.detach().float().cpu().numpy(),
        result.boxes.conf.detach().float().cpu().numpy(),
        result.boxes.cls.detach().cpu().numpy().astype(np.int64),
    )


def predict(
    detector: YOLO,
    images: torch.Tensor,
    *,
    device: str,
    inference_conf: float,
    nms_iou: float,
    duty_cycle: float,
):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    results = detector.predict(
        source=images,
        imgsz=int(images.shape[-1]),
        conf=inference_conf,
        iou=nms_iou,
        device=device,
        verbose=False,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    active = time.perf_counter() - started
    if duty_cycle < 1.0:
        time.sleep(active * (1.0 / duty_cycle - 1.0))
    return results


def instance_rows(
    *,
    clean_result,
    changed_result,
    scene_id: str,
    path: str,
    patch: str,
    detection_conf: float,
    match_iou: float,
    suppression_drop: float,
) -> tuple[list[dict], dict]:
    clean_boxes, clean_scores, clean_classes = result_arrays(clean_result)
    changed_boxes, changed_scores, changed_classes = result_arrays(changed_result)
    clean_keep = clean_scores >= detection_conf
    clean_boxes = clean_boxes[clean_keep]
    clean_scores = clean_scores[clean_keep]
    clean_classes = clean_classes[clean_keep]
    changed_keep = changed_scores >= detection_conf

    rows: list[dict] = []
    for index, (box, score, class_id) in enumerate(
        zip(clean_boxes, clean_scores, clean_classes, strict=True)
    ):
        same_class = changed_classes == class_id
        candidate_boxes = changed_boxes[same_class]
        candidate_scores = changed_scores[same_class]
        ious = box_iou_one_to_many(box, candidate_boxes)
        if len(ious):
            candidate_index = int(np.argmax(ious + 1e-4 * candidate_scores))
            best_iou = float(ious[candidate_index])
            candidate_conf = float(candidate_scores[candidate_index])
        else:
            best_iou = 0.0
            candidate_conf = 0.0
        matched_conf = candidate_conf if best_iou >= match_iou else 0.0
        visible = best_iou >= match_iou and candidate_conf >= detection_conf
        rows.append(
            {
                "scene_id": scene_id,
                "path": path,
                "patch": patch,
                "clean_index": index,
                "class_id": int(class_id),
                "is_person": int(class_id == 0),
                "clean_conf": float(score),
                "candidate_conf": candidate_conf,
                "matched_conf": matched_conf,
                "best_iou": best_iou,
                "visible": bool(visible),
                "hidden": not bool(visible),
                "confidence_drop": float(score) - matched_conf,
                "suppressed_0p3": float(score) - matched_conf
                >= suppression_drop,
                "geometry_shift": candidate_conf >= detection_conf
                and best_iou < match_iou,
                "clean_x1": float(box[0]),
                "clean_y1": float(box[1]),
                "clean_x2": float(box[2]),
                "clean_y2": float(box[3]),
            }
        )

    introduced = 0
    for box, class_id in zip(
        changed_boxes[changed_keep], changed_classes[changed_keep], strict=True
    ):
        same_class = clean_classes == class_id
        if not same_class.any():
            introduced += 1
            continue
        if float(box_iou_one_to_many(box, clean_boxes[same_class]).max()) < match_iou:
            introduced += 1

    person_indices = np.flatnonzero(clean_classes == 0)
    top_person_index = (
        int(person_indices[np.argmax(clean_scores[person_indices])])
        if len(person_indices)
        else None
    )
    top_all_index = int(np.argmax(clean_scores)) if len(clean_scores) else None

    def top_metrics(index: int | None, prefix: str) -> dict:
        if index is None:
            return {
                f"{prefix}_eligible": False,
                f"{prefix}_hidden": np.nan,
                f"{prefix}_suppressed_0p3": np.nan,
                f"{prefix}_confidence_drop": np.nan,
                f"{prefix}_best_iou": np.nan,
                f"{prefix}_geometry_shift": np.nan,
            }
        row = rows[index]
        return {
            f"{prefix}_eligible": True,
            f"{prefix}_hidden": row["hidden"],
            f"{prefix}_suppressed_0p3": row["suppressed_0p3"],
            f"{prefix}_confidence_drop": row["confidence_drop"],
            f"{prefix}_best_iou": row["best_iou"],
            f"{prefix}_geometry_shift": row["geometry_shift"],
        }

    image_row = {
        "scene_id": scene_id,
        "path": path,
        "patch": patch,
        "clean_all_count": int(len(clean_scores)),
        "changed_all_count": int(changed_keep.sum()),
        "clean_person_count": int((clean_classes == 0).sum()),
        "changed_person_count": int(
            ((changed_classes == 0) & changed_keep).sum()
        ),
        "introduced_count": introduced,
        **top_metrics(top_person_index, "top_person"),
        **top_metrics(top_all_index, "top_all"),
    }
    return rows, image_row


def mean_or_nan(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def summarize(
    instances: pd.DataFrame, images: pd.DataFrame, patch_names: list[str]
) -> pd.DataFrame:
    summaries: list[dict] = []
    for patch in patch_names:
        ins = instances[instances.patch.eq(patch)]
        img = images[images.patch.eq(patch)]
        person_ins = ins[ins.is_person.eq(1)]
        top_person = img[img.top_person_eligible.astype(bool)]
        top_all = img[img.top_all_eligible.astype(bool)]
        summaries.append(
            {
                "patch": patch,
                "eval_images": int(len(img)),
                "clean_instances": int(len(ins)),
                "clean_person_instances": int(len(person_ins)),
                "top_person_hidden_rate": mean_or_nan(
                    top_person.top_person_hidden
                ),
                "top_person_suppressed_0p3_rate": mean_or_nan(
                    top_person.top_person_suppressed_0p3
                ),
                "top_person_mean_confidence_drop": mean_or_nan(
                    top_person.top_person_confidence_drop
                ),
                "top_person_mean_best_iou": mean_or_nan(
                    top_person.top_person_best_iou
                ),
                "top_person_geometry_shift_rate": mean_or_nan(
                    top_person.top_person_geometry_shift
                ),
                "person_instance_hidden_rate": mean_or_nan(person_ins.hidden),
                "person_instance_suppressed_0p3_rate": mean_or_nan(
                    person_ins.suppressed_0p3
                ),
                "person_instance_mean_best_iou": mean_or_nan(
                    person_ins.best_iou
                ),
                "person_instance_geometry_shift_rate": mean_or_nan(
                    person_ins.geometry_shift
                ),
                "top_all_hidden_rate": mean_or_nan(top_all.top_all_hidden),
                "top_all_suppressed_0p3_rate": mean_or_nan(
                    top_all.top_all_suppressed_0p3
                ),
                "all_instance_hidden_rate": mean_or_nan(ins.hidden),
                "all_instance_suppressed_0p3_rate": mean_or_nan(
                    ins.suppressed_0p3
                ),
                "all_instance_mean_best_iou": mean_or_nan(ins.best_iou),
                "all_instance_geometry_shift_rate": mean_or_nan(
                    ins.geometry_shift
                ),
                "mean_all_count_delta": mean_or_nan(
                    img.changed_all_count - img.clean_all_count
                ),
                "mean_person_count_delta": mean_or_nan(
                    img.changed_person_count - img.clean_person_count
                ),
                "mean_introduced_detections": mean_or_nan(
                    img.introduced_count
                ),
            }
        )
    return pd.DataFrame(summaries)


def save_plot(summary: pd.DataFrame, patch_tensors: dict[str, torch.Tensor], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = summary.patch.tolist()
    labels = [name.replace("_", "\n") for name in names]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    x = np.arange(len(names))
    width = 0.36
    axes[0, 0].bar(
        x - width / 2,
        summary.top_person_hidden_rate,
        width,
        label="top person",
    )
    axes[0, 0].bar(
        x + width / 2,
        summary.top_all_hidden_rate,
        width,
        label="top any class",
    )
    axes[0, 0].set(
        title="Top-detection hidden rate",
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1),
    )
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(
        x - width / 2,
        summary.person_instance_hidden_rate,
        width,
        label="person instances",
    )
    axes[0, 1].bar(
        x + width / 2,
        summary.all_instance_hidden_rate,
        width,
        label="all instances",
    )
    axes[0, 1].set(
        title="Clean-instance hidden rate",
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1),
    )
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].bar(
        x - width / 2,
        summary.person_instance_mean_best_iou,
        width,
        label="person instances",
    )
    axes[1, 0].bar(
        x + width / 2,
        summary.all_instance_mean_best_iou,
        width,
        label="all instances",
    )
    axes[1, 0].set(
        title="Best same-class IoU (lower means more geometry change)",
        xticks=x,
        xticklabels=labels,
        ylim=(0, 1),
    )
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.25)

    canvas = np.concatenate(
        [
            patch_tensors[name].permute(1, 2, 0).numpy()
            for name in names
        ],
        axis=1,
    )
    axes[1, 1].imshow(np.clip(canvas, 0, 1))
    axes[1, 1].set_title("Learned surfaces: " + " | ".join(names))
    axes[1, 1].axis("off")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leak-free paired holdout evaluation of four fixed-corner surfaces."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--patch", action="append", type=parse_patch, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inference-conf", type=float, default=0.01)
    parser.add_argument("--detection-conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--nms-time-per-image", type=float, default=0.05)
    parser.add_argument("--suppression-drop", type=float, default=0.30)
    parser.add_argument("--duty-cycle", type=float, default=0.20)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    eval_records = manifest["eval"]
    train_scene_ids = {row["scene_id"] for row in manifest["train"]}
    eval_scene_ids = {row["scene_id"] for row in eval_records}
    overlap = train_scene_ids & eval_scene_ids
    if overlap:
        raise RuntimeError(f"train/eval scene leak: {len(overlap)} overlapping scenes")
    if len(eval_scene_ids) != len(eval_records):
        raise RuntimeError("eval split contains duplicate scene ids")

    patch_tensors = {name: load_tensor(path) for name, path in args.patch}
    if len(patch_tensors) != len(args.patch):
        raise ValueError("patch names must be unique")
    patch_shapes = {name: tuple(tensor.shape) for name, tensor in patch_tensors.items()}
    for name, tensor in patch_tensors.items():
        if tensor.shape[0] != 3:
            raise ValueError(f"{name} is not RGB: {tuple(tensor.shape)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.nms_time_per_image != 0.05:
        from ultralytics.utils import nms as nms_module

        original_nms = nms_module.non_max_suppression

        @functools.wraps(original_nms)
        def non_max_suppression_with_time(*nms_args, **nms_kwargs):
            nms_kwargs.setdefault(
                "max_time_img", float(args.nms_time_per_image)
            )
            return original_nms(*nms_args, **nms_kwargs)

        nms_module.non_max_suppression = non_max_suppression_with_time

    detector = YOLO(str(args.weights))
    instance_records: list[dict] = []
    image_records: list[dict] = []
    started = time.perf_counter()
    for start in range(0, len(eval_records), args.batch_size):
        records = eval_records[start : start + args.batch_size]
        clean = torch.stack([load_tensor(Path(row["path"])) for row in records])
        clean_results = predict(
            detector,
            clean,
            device=args.device,
            inference_conf=args.inference_conf,
            nms_iou=args.nms_iou,
            duty_cycle=args.duty_cycle,
        )
        for patch_name, patch in patch_tensors.items():
            changed = clean.clone()
            height = min(changed.shape[-2], patch.shape[-2])
            width = min(changed.shape[-1], patch.shape[-1])
            changed[:, :, :height, :width] = patch[None, :, :height, :width]
            changed_results = predict(
                detector,
                changed,
                device=args.device,
                inference_conf=args.inference_conf,
                nms_iou=args.nms_iou,
                duty_cycle=args.duty_cycle,
            )
            for record, clean_result, changed_result in zip(
                records, clean_results, changed_results, strict=True
            ):
                rows, image_row = instance_rows(
                    clean_result=clean_result,
                    changed_result=changed_result,
                    scene_id=record["scene_id"],
                    path=record["path"],
                    patch=patch_name,
                    detection_conf=args.detection_conf,
                    match_iou=args.match_iou,
                    suppression_drop=args.suppression_drop,
                )
                instance_records.extend(rows)
                image_records.append(image_row)
        completed = min(start + len(records), len(eval_records))
        elapsed = time.perf_counter() - started
        eta = elapsed / completed * (len(eval_records) - completed)
        print(
            f"evaluated={completed}/{len(eval_records)} elapsed={elapsed / 60:.1f}m "
            f"eta={eta / 60:.1f}m",
            flush=True,
        )

    instances = pd.DataFrame(instance_records)
    images = pd.DataFrame(image_records)
    patch_names = list(patch_tensors)
    summary = summarize(instances, images, patch_names)
    instances.to_csv(args.output_dir / "instance_details.csv", index=False)
    images.to_csv(args.output_dir / "image_details.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    save_plot(summary, patch_tensors, args.output_dir / "comparison.png")
    metadata = {
        "version": 1,
        "paired_clean_reference": True,
        "ground_truth_semantics": "clean detector detections are paired pseudo-targets",
        "scene_disjoint": True,
        "train_scenes": len(train_scene_ids),
        "eval_scenes": len(eval_scene_ids),
        "scene_overlap": len(overlap),
        "patch_shapes_chw": patch_shapes,
        "position_xy": [0, 0],
        "inference_conf": args.inference_conf,
        "detection_conf": args.detection_conf,
        "match_iou": args.match_iou,
        "nms_iou": args.nms_iou,
        "nms_time_per_image": args.nms_time_per_image,
        "suppression_drop": args.suppression_drop,
        "duty_cycle": args.duty_cycle,
        "elapsed_seconds": time.perf_counter() - started,
        "metric_notes": {
            "hidden": "no same-class patched detection with IoU>=match_iou and confidence>=detection_conf",
            "geometry_shift": "best same-class candidate has confidence>=detection_conf but IoU<match_iou",
            "introduced": "patched detection above detection_conf with no clean same-class IoU>=match_iou",
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Completed: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
