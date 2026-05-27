from __future__ import annotations

from typing import Any


def apply_patch_to_image(base_pil, patch_pil, position_xy: tuple[int, int] = (0, 0)):
    from PIL import Image

    if not isinstance(base_pil, Image.Image):
        raise TypeError("base_pil must be a PIL image.")
    base = base_pil.convert("RGB")
    patch = patch_pil.convert("RGB")
    base_w, base_h = base.size
    patch_w, patch_h = patch.size
    x, y = int(position_xy[0]), int(position_xy[1])
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(base_w, x + patch_w), min(base_h, y + patch_h)
    if x2 <= x1 or y2 <= y1:
        return base.copy(), None, 0
    crop = patch.crop((x1 - x, y1 - y, x1 - x + (x2 - x1), y1 - y + (y2 - y1)))
    out = base.copy()
    out.paste(crop, (x1, y1))
    return out, (float(x1), float(y1), float(x2), float(y2)), int((x2 - x1) * (y2 - y1))


def letterbox_pil(pil, imgsz: int, color: tuple[int, int, int] = (114, 114, 114)):
    image = pil.convert("RGB")
    w, h = image.size
    scale = min(float(imgsz) / float(h), float(imgsz) / float(w))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = image.resize((new_w, new_h))
    from PIL import Image

    out = Image.new("RGB", (int(imgsz), int(imgsz)), color)
    out.paste(resized, ((int(imgsz) - new_w) // 2, (int(imgsz) - new_h) // 2))
    return out


def build_clean_and_patched_letterboxed(base_pil, patch_pil, cfg: Any):
    clean_lb = letterbox_pil(base_pil, imgsz=int(cfg.imgsz))
    if bool(getattr(cfg, "apply_patch_after_letterbox", True)):
        patched_lb, bbox, _area = apply_patch_to_image(clean_lb, patch_pil, getattr(cfg, "patch_xy", (0, 0)))
        return clean_lb, patched_lb, bbox
    patched_orig, _bbox_orig, _area = apply_patch_to_image(base_pil, patch_pil, getattr(cfg, "patch_xy", (0, 0)))
    patched_lb = letterbox_pil(patched_orig, imgsz=int(cfg.imgsz))
    return clean_lb, patched_lb, None


def make_black_baseline_like(pil):
    from PIL import Image

    return Image.new("RGB", pil.size, (0, 0, 0))


def pil_to_torch_rgb01(pil, *, device=None, dtype=None):
    import numpy as np
    import torch

    arr = np.asarray(pil.convert("RGB")).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    if dtype is not None or device is not None:
        tensor = tensor.to(device=device, dtype=dtype)
    return tensor
