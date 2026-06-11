from __future__ import annotations

from dataclasses import replace

from .experiment import SegmentIGDetectorConfig, run_segmentig_detector_experiment


def main() -> None:
    config = replace(
        SegmentIGDetectorConfig(),
        n_images=1,
        n_steps=4,
        alpha_batch_size=2,
        output_dir="new_experiments/outputs/segmentig_detector_layer22_smoke",
    )
    results = run_segmentig_detector_experiment(config)
    metadata = results[0]["metadata"]
    if metadata["activation_shape"][1:] != [20, 20]:
        print("warning: expected layer spatial shape [20, 20], got", metadata["activation_shape"])
    if (
        not metadata["class_logit_map_has_signal"]
        or not metadata["width_map_has_signal"]
        or not metadata["height_map_has_signal"]
        or not metadata["width_height_normalized_map_has_signal"]
        or not metadata["class_plus_width_height_normalized_map_has_signal"]
    ):
        raise RuntimeError("Smoke run produced an empty SegmentIG heatmap.")
    print("smoke ok:", metadata["png_path"])


if __name__ == "__main__":
    main()
