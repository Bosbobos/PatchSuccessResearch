from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LayerDelta:
    layer_name: str
    clean_shape: tuple[int, ...]
    patched_shape: tuple[int, ...]
    delta: Any


class ActivationCapture:
    def __init__(self, model, layer_names: list[str]):
        self.activations: dict[str, Any] = {}
        self._handles = []
        modules = dict(model.named_modules())
        for name in layer_names:
            if name not in modules:
                raise KeyError(f"Layer {name!r} was not found.")
            self._handles.append(modules[name].register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            import torch

            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach()
            elif isinstance(output, (list, tuple)):
                tensor = next((item for item in output if isinstance(item, torch.Tensor)), None)
                if tensor is not None:
                    self.activations[name] = tensor.detach()

        return hook

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.remove()


def capture_activations(model, x, layer_names: list[str]) -> dict[str, Any]:
    from segmentig_detector.yolo_utils import safe_model_forward

    with ActivationCapture(model, layer_names) as capture:
        safe_model_forward(model, x)
        return dict(capture.activations)


def compute_layer_deltas(model, clean_x, patched_x, layer_names: list[str]) -> dict[str, LayerDelta]:
    clean = capture_activations(model, clean_x, layer_names)
    patched = capture_activations(model, patched_x, layer_names)
    out: dict[str, LayerDelta] = {}
    for name in layer_names:
        if name in clean and name in patched:
            delta = patched[name] - clean[name]
            out[name] = LayerDelta(
                layer_name=name,
                clean_shape=tuple(int(v) for v in clean[name].shape),
                patched_shape=tuple(int(v) for v in patched[name].shape),
                delta=delta.detach(),
            )
    return out


def reduce_chw_to_hw(tensor, *, mode: str = "l2"):
    import torch

    x = tensor.detach()
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected [C,H,W] or [1,C,H,W], got {tuple(x.shape)}")
    if mode == "mean_abs":
        return x.abs().mean(dim=0)
    if mode == "signed_mean":
        return x.mean(dim=0)
    if mode == "max_abs":
        return x.abs().max(dim=0).values
    if mode == "l2":
        return torch.sqrt(torch.mean(x * x, dim=0).clamp_min(0.0))
    raise ValueError(f"Unsupported reduction mode: {mode!r}")


def robust_normalize(values, *, signed: bool = False, q: float = 99.0, eps: float = 1e-12):
    import numpy as np

    arr = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    arr = arr.astype("float32", copy=False)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype="float32")
    if signed:
        scale = np.nanpercentile(np.abs(arr[finite]), float(q))
        return np.clip(arr / max(float(scale), float(eps)), -1.0, 1.0)
    lo, hi = 0.0, np.nanpercentile(arr[finite], float(q))
    return np.clip((arr - lo) / max(float(hi - lo), float(eps)), 0.0, 1.0)


def patch_mask_on_feature_grid(
    patch_bbox_xyxy: tuple[float, float, float, float] | None,
    *,
    grid_hw: tuple[int, int],
    imgsz: int,
):
    import numpy as np

    h, w = int(grid_hw[0]), int(grid_hw[1])
    mask = np.zeros((h, w), dtype=bool)
    if patch_bbox_xyxy is None:
        return mask
    x1, y1, x2, y2 = [float(v) for v in patch_bbox_xyxy]
    gx1 = max(0, min(w, int(np.floor(x1 / float(imgsz) * w))))
    gx2 = max(0, min(w, int(np.ceil(x2 / float(imgsz) * w))))
    gy1 = max(0, min(h, int(np.floor(y1 / float(imgsz) * h))))
    gy2 = max(0, min(h, int(np.ceil(y2 / float(imgsz) * h))))
    if gx2 > gx1 and gy2 > gy1:
        mask[gy1:gy2, gx1:gx2] = True
    return mask


def delta_spread_metrics(delta, *, patch_bbox_xyxy=None, imgsz: int = 640, topk_frac: float = 0.05) -> dict[str, float]:
    import numpy as np

    hw = reduce_chw_to_hw(delta, mode="l2").detach().cpu().numpy()
    flat = hw.reshape(-1).astype("float64")
    total = float(np.sum(flat) + 1e-12)
    grid_hw = (int(hw.shape[0]), int(hw.shape[1]))
    roi_mask = patch_mask_on_feature_grid(patch_bbox_xyxy, grid_hw=grid_hw, imgsz=int(imgsz))
    k = max(1, int(round(float(topk_frac) * flat.size)))
    top_idx = np.argpartition(-flat, kth=min(k - 1, flat.size - 1))[:k]
    return {
        "delta_l2_rms": float(np.sqrt(np.mean(flat * flat))),
        "delta_abs_mean": float(np.mean(np.abs(flat))),
        "delta_max": float(np.max(flat)) if flat.size else 0.0,
        "delta_gini": gini(flat),
        "roi_energy_frac": float(np.sum(hw[roi_mask]) / total) if roi_mask.any() else 0.0,
        "topk_energy_frac": float(np.sum(flat[top_idx]) / total),
    }


def gini(values) -> float:
    import numpy as np

    x = np.asarray(values, dtype="float64").reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    x = np.abs(x)
    if np.allclose(x.sum(), 0.0):
        return 0.0
    x.sort()
    n = x.size
    index = np.arange(1, n + 1, dtype="float64")
    return float((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))
