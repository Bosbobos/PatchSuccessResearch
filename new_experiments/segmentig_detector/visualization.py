from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


def normalize_map(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = values - float(values.min())
    vmax = float(values.max())
    if vmax < eps:
        return np.zeros_like(values, dtype=np.float32)
    return values / vmax


def spatial_importance_map(attribution_bchw: torch.Tensor) -> np.ndarray:
    """Convert neuron importances [1,C,H,W] directly to signed HxW squares.

    Each spatial square gets the signed sum of all channel attributions at that
    location, then the whole map is scaled by max absolute value to [-1, 1].
    No absolute value, min-max normalization, smoothing, or other transforms are
    applied.
    """
    attr = attribution_bchw.detach().cpu().to(torch.float32)
    if attr.ndim != 4:
        raise ValueError(f"Expected attribution [B,C,H,W], got {tuple(attr.shape)}")
    spatial = attr[0].sum(dim=0)
    arr = spatial.numpy().astype(np.float32)
    max_abs = float(np.max(np.abs(arr)))
    if max_abs < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / max_abs


def draw_bbox(image: Image.Image, bbox_xyxy: tuple[float, float, float, float], label: str) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
    draw.text((x1 + 3, max(0.0, y1 - 14.0)), label, fill=(255, 0, 0))
    return out


def save_result_figure(
    image: Image.Image,
    bbox_xyxy: tuple[float, float, float, float],
    class_logit_map: np.ndarray,
    width_map: np.ndarray,
    height_map: np.ndarray,
    wh_norm_map: np.ndarray,
    class_plus_wh_norm_map: np.ndarray,
    metadata: dict[str, Any],
    save_path: str | Path,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    label = f"person {metadata.get('detection_confidence', 0.0):.3f}"
    boxed = draw_bbox(image, bbox_xyxy, label)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5), constrained_layout=True)
    axes_flat = axes.reshape(-1)
    axes_flat[0].imshow(boxed)
    axes_flat[0].set_title(Path(str(metadata.get("image_path", "image"))).name)
    axes_flat[0].axis("off")

    axes_flat[1].imshow(image)
    axes_flat[1].imshow(
        class_logit_map,
        cmap="coolwarm",
        alpha=0.55,
        interpolation="nearest",
        vmin=-1.0,
        vmax=1.0,
        extent=(0, image.width, image.height, 0),
    )
    axes_flat[1].set_title("SegmentIG: class logit")
    axes_flat[1].axis("off")

    axes_flat[2].imshow(image)
    axes_flat[2].imshow(
        width_map,
        cmap="coolwarm",
        alpha=0.55,
        interpolation="nearest",
        vmin=-1.0,
        vmax=1.0,
        extent=(0, image.width, image.height, 0),
    )
    axes_flat[2].set_title("SegmentIG: width")
    axes_flat[2].axis("off")

    axes_flat[3].imshow(image)
    axes_flat[3].imshow(
        height_map,
        cmap="coolwarm",
        alpha=0.55,
        interpolation="nearest",
        vmin=-1.0,
        vmax=1.0,
        extent=(0, image.width, image.height, 0),
    )
    axes_flat[3].set_title("SegmentIG: height")
    axes_flat[3].axis("off")

    axes_flat[4].imshow(image)
    axes_flat[4].imshow(
        wh_norm_map,
        cmap="coolwarm",
        alpha=0.55,
        interpolation="nearest",
        vmin=-1.0,
        vmax=1.0,
        extent=(0, image.width, image.height, 0),
    )
    axes_flat[4].set_title("SegmentIG: (width + height) / imgsz")
    axes_flat[4].axis("off")

    axes_flat[5].imshow(image)
    axes_flat[5].imshow(
        class_plus_wh_norm_map,
        cmap="coolwarm",
        alpha=0.55,
        interpolation="nearest",
        vmin=-1.0,
        vmax=1.0,
        extent=(0, image.width, image.height, 0),
    )
    axes_flat[5].set_title("SegmentIG: logit + norm wh")
    axes_flat[5].axis("off")

    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
