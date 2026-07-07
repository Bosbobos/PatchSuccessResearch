from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import list_image_paths, stable_hash


def grid_positions(*, img_size: int, patch_size: tuple[int, int], step: int) -> list[tuple[int, int]]:
    pw, ph = int(patch_size[0]), int(patch_size[1])
    max_x = int(img_size) - pw
    max_y = int(img_size) - ph
    if max_x < 0 or max_y < 0:
        raise ValueError(f"Patch size {patch_size} does not fit image size {img_size}.")
    xs = list(range(0, max_x + 1, int(step)))
    ys = list(range(0, max_y + 1, int(step)))
    if xs[-1] != max_x:
        xs.append(max_x)
    if ys[-1] != max_y:
        ys.append(max_y)
    return [(x, y) for y in ys for x in xs]


def _cache_path(exp, *, positions: list[tuple[int, int]], max_examples: int | None, batch_size: int) -> Path:
    payload = {
        "attack_cache_key": exp.config.attack.cache_key(),
        "positions": positions,
        "max_examples": max_examples,
        "batch_size": int(batch_size),
        "success_drop_threshold": float(exp.config.attack.success_drop_threshold),
        "metric_definition": "drop_gt_threshold_by_fixed_patch_position",
        "method_version": 1,
    }
    return Path(exp.derived_cache_dir) / f"patch_position_sweep_{stable_hash(payload)}.pkl"


def compute_or_load_position_sweep(
    exp,
    *,
    step: int = 16,
    positions: list[tuple[int, int]] | None = None,
    max_examples: int | None = None,
    batch_size: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    from .modeling import forward_logits, preprocess_pil_batch
    from .patching import load_rgb_image, overlay_top_left_patch

    patch = Image.open(exp.config.attack.patch_path).convert("RGB")
    positions = positions or grid_positions(
        img_size=int(exp.config.attack.img_size),
        patch_size=patch.size,
        step=int(step),
    )
    batch_size = int(batch_size or exp.config.attack.inference_batch_size)
    cache_path = _cache_path(exp, positions=positions, max_examples=max_examples, batch_size=batch_size)
    if cache_path.exists() and not force:
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(cache_path)
        return cached

    image_paths = list_image_paths(exp.config.attack.dataset_path)
    if max_examples is not None:
        image_paths = image_paths[: int(max_examples)]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {exp.config.attack.dataset_path}.")

    model = exp.load_model()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    clean_probs_parts: list[torch.Tensor] = []
    clean_images_cache = []
    progress = None
    try:
        from tqdm.auto import tqdm

        progress = tqdm(total=len(image_paths), desc="clean logits", unit="img")
    except Exception:
        progress = None
    for start in range(0, len(image_paths), batch_size):
        paths = image_paths[start : start + batch_size]
        images = [load_rgb_image(path, img_size=int(exp.config.attack.img_size)) for path in paths]
        clean_images_cache.extend(images)
        x = preprocess_pil_batch(images, img_size=int(exp.config.attack.img_size), device=device, dtype=dtype)
        with torch.no_grad():
            clean_probs_parts.append(torch.sigmoid(forward_logits(model, x)).detach().cpu())
        if progress is not None:
            progress.update(len(paths))
    if progress is not None:
        progress.close()
    clean_probs = torch.cat(clean_probs_parts, dim=0)

    rows = []
    all_position_values = []
    pos_bar = None
    try:
        from tqdm.auto import tqdm

        pos_bar = tqdm(positions, desc="patch positions", unit="pos")
    except Exception:
        pos_bar = positions

    for x_pos, y_pos in pos_bar:
        patched_probs_parts: list[torch.Tensor] = []
        for start in range(0, len(clean_images_cache), batch_size):
            images = clean_images_cache[start : start + batch_size]
            patched = [overlay_top_left_patch(image, patch, xy=(int(x_pos), int(y_pos)))[0] for image in images]
            px = preprocess_pil_batch(patched, img_size=int(exp.config.attack.img_size), device=device, dtype=dtype)
            with torch.no_grad():
                patched_probs_parts.append(torch.sigmoid(forward_logits(model, px)).detach().cpu())
        patched_probs = torch.cat(patched_probs_parts, dim=0)
        drops = clean_probs - patched_probs
        successes = drops > float(exp.config.attack.success_drop_threshold)
        row = {
            "x": int(x_pos),
            "y": int(y_pos),
            "patch_x_center": float(x_pos + patch.size[0] / 2.0),
            "patch_y_center": float(y_pos + patch.size[1] / 2.0),
            "n": int(len(drops)),
            "successful": int(successes.sum().item()),
            "unsuccessful": int((~successes).sum().item()),
            "ASR": float(successes.float().mean().item()),
            "mean_drop": float(drops.mean().item()),
            "median_drop": float(drops.median().item()),
            "mean_conf_clean": float(clean_probs.mean().item()),
            "mean_conf_patch": float(patched_probs.mean().item()),
        }
        rows.append(row)
        all_position_values.append(
            {
                "x": int(x_pos),
                "y": int(y_pos),
                "patched_probs": patched_probs.numpy().astype("float32", copy=False),
                "drops": drops.numpy().astype("float32", copy=False),
                "successes": successes.numpy().astype(bool, copy=False),
            }
        )

    rows_df = pd.DataFrame(rows).sort_values(["y", "x"]).reset_index(drop=True)
    result = {
        "rows_df": rows_df,
        "positions": positions,
        "paths": [str(path) for path in image_paths],
        "clean_probs": clean_probs.numpy().astype("float32", copy=False),
        "position_values": all_position_values,
        "patch_size": patch.size,
        "cache_path": str(cache_path),
        "loaded_from_cache": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(result, fh)
    return result
