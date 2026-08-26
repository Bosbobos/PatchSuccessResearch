# Four-loss fixed-corner protocol

- Training images: the existing 500-image scene-disjoint manifest
- Independent evaluation images: the existing 300-image holdout
- Patch size and position: 160×160 at `(0, 0)`
- Optimizer: Adam, learning rate 0.03
- Batch size: 16
- Optimizer steps: 1000
- Initial tensor and shuffle seed: 7, identical across variants
- Augmentations: none
- Regularizers: none
- Per-process CUDA duty cycle: 0.20

## Losses

`Person Adv Patch` minimizes the mean of the top-200 YOLO person-class
confidences. `General Adv Patch` minimizes the mean of the top-200
confidences after flattening every class and anchor.

The DPatch variants add a bounding-box term. For every clean target, the
currently best matching anchor is selected using detached IoU plus detached
target-class confidence. The patch minimizes the aligned CIoU of that anchor
and the clean target box. Minimizing CIoU is equivalent to maximizing the
untargeted regression error `1 - CIoU`, following the original DPatch idea of
maximizing both classification and bounding-box regression loss.

| Remote name | Local research name |
|---|---|
| `class0_score` | Person Adv Patch |
| `allclass_score` | General Adv Patch |
| `class0_joint` | Person DPatch |
| `allclass_joint` | General DPatch |

The remote scheduler requires pre-launch GPU utilization at or below 80% and
at least 20 GiB free. There is intentionally no limit on pre-existing allocated
GPU memory. Each active protocol process still receives a distinct GPU and a
CUDA duty-cycle of 0.20.
