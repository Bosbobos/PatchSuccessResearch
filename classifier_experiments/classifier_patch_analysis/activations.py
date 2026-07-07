from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LayerMaps:
    clean_activation_chw: Any
    patched_activation_chw: Any
    delta_chw: Any
    importance_chw: Any


class ActivationCapture:
    def __init__(self, model, layer_name: str, *, keep_grad: bool = False):
        self.activation = None
        self.keep_grad = bool(keep_grad)
        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise KeyError(f"Layer {layer_name!r} was not found. Available display layers usually are model.0..model.9.")
        self.handle = modules[layer_name].register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        import torch

        tensor = output if isinstance(output, torch.Tensor) else None
        if tensor is None and isinstance(output, (list, tuple)):
            tensor = next((item for item in output if isinstance(item, torch.Tensor)), None)
        if tensor is None:
            return
        if self.keep_grad:
            tensor.retain_grad()
            self.activation = tensor
        else:
            self.activation = tensor.detach()

    def remove(self) -> None:
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.remove()


def capture_activation(model, x, layer_name: str):
    from .modeling import forward_logits

    with ActivationCapture(model, layer_name) as capture:
        _ = forward_logits(model, x)
        if capture.activation is None:
            raise RuntimeError(f"Layer {layer_name!r} did not produce an activation.")
        return capture.activation.detach()


def gradient_x_activation_importance(model, x, layer_name: str):
    from .modeling import forward_logits

    import torch

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        x = x.detach().clone().requires_grad_(True)
        with ActivationCapture(model, layer_name, keep_grad=True) as capture:
            logits = forward_logits(model, x)
            logits.sum().backward()
            if capture.activation is None or capture.activation.grad is None:
                raise RuntimeError(f"Could not compute activation gradient for {layer_name!r}.")
            return capture.activation.detach() * capture.activation.grad.detach()


def reduce_chw_to_hw(tensor, *, mode: str = "l2"):
    import numpy as np
    import torch

    x = tensor.detach() if hasattr(tensor, "detach") else torch.as_tensor(np.asarray(tensor))
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected CHW or BCHW tensor, got {tuple(x.shape)}")
    if mode == "mean_abs":
        return x.abs().mean(dim=0)
    if mode == "signed_mean":
        return x.mean(dim=0)
    if mode == "max_abs":
        return x.abs().max(dim=0).values
    if mode == "l2":
        return (x * x).mean(dim=0).clamp_min(0.0).sqrt()
    raise ValueError(f"Unsupported reduction mode: {mode!r}")


def patch_mask_on_feature_grid(
    patch_bbox_xyxy: tuple[float, float, float, float] | None,
    *,
    grid_hw: tuple[int, int],
    img_size: int,
):
    import numpy as np

    h, w = int(grid_hw[0]), int(grid_hw[1])
    mask = np.zeros((h, w), dtype=bool)
    if patch_bbox_xyxy is None:
        return mask
    x1, y1, x2, y2 = [float(v) for v in patch_bbox_xyxy]
    gx1 = max(0, min(w, int(np.floor(x1 / float(img_size) * w))))
    gx2 = max(0, min(w, int(np.ceil(x2 / float(img_size) * w))))
    gy1 = max(0, min(h, int(np.floor(y1 / float(img_size) * h))))
    gy2 = max(0, min(h, int(np.ceil(y2 / float(img_size) * h))))
    if gx2 > gx1 and gy2 > gy1:
        mask[gy1:gy2, gx1:gx2] = True
    return mask


def gini(values) -> float:
    import numpy as np

    x = np.asarray(values, dtype="float64").reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    x = np.abs(x)
    if float(x.sum()) <= 0.0:
        return 0.0
    x.sort()
    n = x.size
    idx = np.arange(1, n + 1, dtype="float64")
    return float(np.sum((2 * idx - n - 1) * x) / (n * np.sum(x)))


def delta_spread_metrics(delta_chw, *, patch_bbox_xyxy=None, img_size: int = 224, topk_frac: float = 0.05):
    import numpy as np

    hw = reduce_chw_to_hw(delta_chw, mode="l2").detach().cpu().numpy().astype("float64", copy=False)
    flat = hw.reshape(-1)
    total = float(flat.sum() + 1e-12)
    mask = patch_mask_on_feature_grid(patch_bbox_xyxy, grid_hw=hw.shape, img_size=int(img_size))
    outside = flat.reshape(hw.shape)[~mask] if mask.any() else flat
    k = max(1, int(round(float(topk_frac) * flat.size)))
    top_idx = np.argpartition(-flat, kth=min(k - 1, flat.size - 1))[:k]
    return {
        "delta_l2_rms": float(np.sqrt(np.mean(flat * flat))),
        "delta_abs_mean": float(np.mean(np.abs(flat))),
        "delta_max": float(np.max(flat)) if flat.size else 0.0,
        "delta_gini": gini(flat),
        "patch_roi_energy_frac": float(hw[mask].sum() / total) if mask.any() else 0.0,
        "outside_patch_energy_frac": float(outside.sum() / total) if outside.size else 0.0,
        "top5pct_energy_frac": float(flat[top_idx].sum() / total),
    }


def _as_chw(values):
    import numpy as np

    arr = np.asarray(values, dtype="float32")
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW values, got shape={arr.shape}")
    return arr
