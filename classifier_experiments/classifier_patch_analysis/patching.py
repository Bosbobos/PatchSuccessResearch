from __future__ import annotations

from pathlib import Path
from typing import Any


def load_rgb_image(path: str | Path, *, img_size: int = 224):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").resize((int(img_size), int(img_size)))


def overlay_top_left_patch(image: Any, patch: Any, *, xy: tuple[int, int] = (0, 0)):
    image = image.convert("RGB").copy()
    patch = patch.convert("RGB")
    x, y = int(xy[0]), int(xy[1])
    w, h = patch.size
    if x < 0 or y < 0 or x + w > image.size[0] or y + h > image.size[1]:
        raise ValueError(f"Patch bbox {(x, y, x + w, y + h)} is outside image size {image.size}.")
    image.paste(patch, (x, y))
    return image, (x, y, x + w, y + h)
