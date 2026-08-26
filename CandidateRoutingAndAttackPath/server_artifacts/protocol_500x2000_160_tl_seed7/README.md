# Fixed-corner training protocol

- Source implementation: `Bosbobos/DePatch-RobustDPatch-training`
- Source commit: `7fcb90d2f9b84ee66c37ecac47c05549ba24841b`
- Detector: YOLO11s, COCO `person` class
- Training set: 500 deterministic, unique COCO source scenes
- Independent evaluation set: 300 deterministic, unique source scenes
- Split seed: 7
- Optimizer budget: 2000 updates per method
- Batch size: 16
- Patch dimensions: 160×160 pixels
- Placement: fixed top-left corner, `(x, y) = (0, 0)`
- Per-process CUDA duty-cycle: 0.20
- Initial tensor: identical across all methods

The authoritative scene-disjoint split is in `manifest.json`. The initial
scheduler snapshot is retained locally; final PNG/PT checkpoints, histories,
completion metadata, and final logs must be pulled from the server after all
three jobs finish.

Remote-to-local method mapping:

| Remote name | Local research name |
|---|---|
| `surface_dropout` | DePatch |
| `surface_transform` | RobustDPatch |
| `surface_baseline` | ordinary adversarial patch |
