# SegmentIG detector experiments

This folder contains the detector SegmentIG implementation for YOLO `model.22`.
The current visualization renders five signed `[-1, 1]` spatial-square maps:
`class_logit`, `width`, `height`, `(width + height) / imgsz`, and
`class_logit + (width + height) / imgsz`.

Run a one-image smoke check in an environment with `torch`, `ultralytics`, `PIL`,
`numpy`, and `matplotlib`:

```bash
python -m segmentig_detector.smoke_test
```

Run the full three-image experiment:

```bash
python -m segmentig_detector.run \
  --weights yolo11s.pt \
  --dataset-dir datasets/inria_train \
  --n-images 3 \
  --n-steps 64 \
  --alpha-batch-size 4
```

The notebook `segmentig_detector_visualization.ipynb` calls the same Python
functions and displays the generated PNG collages.
