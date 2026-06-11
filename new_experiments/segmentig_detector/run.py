from __future__ import annotations

import argparse
from dataclasses import replace

from .experiment import SegmentIGDetectorConfig, run_segmentig_detector_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SegmentIG for YOLO person detections on model.22.")
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--dataset-dir", default="datasets/inria_train")
    parser.add_argument("--output-dir", default="new_experiments/outputs/segmentig_detector_layer22")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-images", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--alpha-batch-size", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        SegmentIGDetectorConfig(),
        weights=args.weights,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        device=args.device,
        n_images=args.n_images,
        n_steps=args.n_steps,
        alpha_batch_size=args.alpha_batch_size,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )
    results = run_segmentig_detector_experiment(config)
    for item in results:
        metadata = item["metadata"]
        c = metadata["target_components"]
        print(metadata["image_path"], "->", metadata["png_path"])
        print(
            "  target components:",
            f"class_logit={c['class_logit']:.6f}",
            f"width={c['width']:.6f}",
            f"height={c['height']:.6f}",
            f"(width+height)/imgsz={c['width_plus_height_over_imgsz']:.6f}",
        )


if __name__ == "__main__":
    main()
