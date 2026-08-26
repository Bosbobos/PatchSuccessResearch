"""Distributed paired holdout evaluation for one fixed-corner surface.

The script is intentionally a thin torch.distributed wrapper around the
existing object_response_evaluation metrics so the single-GPU and four-GPU
reports have identical semantics.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--surface-name", default="surface_alignment")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inference-conf", type=float, default=0.01)
    parser.add_argument("--detection-conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--nms-time-per-image", type=float, default=1.0)
    parser.add_argument("--suppression-drop", type=float, default=0.30)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    source_dir = Path(__file__).resolve().parent.parent / "surface_reference_training"
    sys.path.insert(0, str(source_dir))
    from object_response_evaluation import (
        instance_rows,
        load_tensor,
        save_plot,
        summarize,
    )
    from ultralytics import YOLO
    from ultralytics.utils import nms as nms_module

    original_nms = nms_module.non_max_suppression

    @functools.wraps(original_nms)
    def non_max_suppression_with_time(*nms_args, **nms_kwargs):
        nms_kwargs.setdefault(
            "max_time_img", float(args.nms_time_per_image)
        )
        return original_nms(*nms_args, **nms_kwargs)

    nms_module.non_max_suppression = non_max_suppression_with_time

    manifest = json.loads(args.manifest.read_text())
    train_ids = {row["scene_id"] for row in manifest["train"]}
    eval_records = manifest["eval"]
    eval_ids = {row["scene_id"] for row in eval_records}
    overlap = train_ids & eval_ids
    if overlap:
        raise RuntimeError(f"train/eval scene leak: {len(overlap)}")
    if len(eval_ids) != len(eval_records):
        raise RuntimeError("duplicate eval scene ids")

    local_records = eval_records[rank::world_size]
    surface = load_tensor(args.surface)
    detector = YOLO(str(args.weights))
    instance_records: list[dict] = []
    image_records: list[dict] = []
    started = time.perf_counter()

    for start in range(0, len(local_records), args.batch_size):
        records = local_records[start : start + args.batch_size]
        clean = torch.stack(
            [load_tensor(Path(record["path"])) for record in records]
        )
        changed = clean.clone()
        height = min(changed.shape[-2], surface.shape[-2])
        width = min(changed.shape[-1], surface.shape[-1])
        changed[:, :, :height, :width] = surface[None, :, :height, :width]

        clean_results = detector.predict(
            source=clean,
            imgsz=int(clean.shape[-1]),
            conf=args.inference_conf,
            iou=args.nms_iou,
            device=device,
            verbose=False,
        )
        changed_results = detector.predict(
            source=changed,
            imgsz=int(changed.shape[-1]),
            conf=args.inference_conf,
            iou=args.nms_iou,
            device=device,
            verbose=False,
        )
        for record, clean_result, changed_result in zip(
            records, clean_results, changed_results, strict=True
        ):
            rows, image_row = instance_rows(
                clean_result=clean_result,
                changed_result=changed_result,
                scene_id=record["scene_id"],
                path=record["path"],
                patch=args.surface_name,
                detection_conf=args.detection_conf,
                match_iou=args.match_iou,
                suppression_drop=args.suppression_drop,
            )
            instance_records.extend(rows)
            image_records.append(image_row)
        print(
            f"rank={rank} evaluated={min(start + len(records), len(local_records))}"
            f"/{len(local_records)}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(instance_records).to_csv(
        args.output_dir / f"instance_details_rank{rank}.csv", index=False
    )
    pd.DataFrame(image_records).to_csv(
        args.output_dir / f"image_details_rank{rank}.csv", index=False
    )
    dist.barrier()

    if rank == 0:
        instances = pd.concat(
            [
                pd.read_csv(args.output_dir / f"instance_details_rank{i}.csv")
                for i in range(world_size)
            ],
            ignore_index=True,
        )
        images = pd.concat(
            [
                pd.read_csv(args.output_dir / f"image_details_rank{i}.csv")
                for i in range(world_size)
            ],
            ignore_index=True,
        )
        instances = instances.sort_values(
            ["scene_id", "clean_index"], kind="stable"
        ).reset_index(drop=True)
        images = images.sort_values("scene_id", kind="stable").reset_index(
            drop=True
        )
        summary = summarize(instances, images, [args.surface_name])
        instances.to_csv(args.output_dir / "instance_details.csv", index=False)
        images.to_csv(args.output_dir / "image_details.csv", index=False)
        summary.to_csv(args.output_dir / "summary.csv", index=False)
        save_plot(
            summary,
            {args.surface_name: surface.cpu()},
            args.output_dir / "comparison.png",
        )
        metadata = {
            "version": 1,
            "world_size": world_size,
            "paired_clean_reference": True,
            "ground_truth_semantics": (
                "clean detector detections are paired pseudo-targets"
            ),
            "scene_disjoint": True,
            "train_scenes": len(train_ids),
            "eval_scenes": len(eval_ids),
            "scene_overlap": len(overlap),
            "surface_shape_chw": list(surface.shape),
            "position_xy": [0, 0],
            "inference_conf": args.inference_conf,
            "detection_conf": args.detection_conf,
            "match_iou": args.match_iou,
            "nms_iou": args.nms_iou,
            "nms_time_per_image": args.nms_time_per_image,
            "suppression_drop": args.suppression_drop,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )
        print(summary.to_string(index=False), flush=True)
        print(f"Completed: {args.output_dir}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
