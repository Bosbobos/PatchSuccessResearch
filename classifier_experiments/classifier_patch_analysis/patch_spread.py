from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .activations import capture_activation, reduce_chw_to_hw
from .utils import stable_hash


def jpeg_zigzag_indices(h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for s in range(h + w - 1):
        diag = [(y, s - y) for y in range(h) if 0 <= s - y < w]
        if s % 2 == 0:
            diag.reverse()
        out.extend(diag)
    return out


def centered_spiral_indices(h: int, w: int) -> list[tuple[int, int]]:
    h, w = int(h), int(w)
    center_rows = [h // 2] if h % 2 else [h // 2 - 1, h // 2]
    center_cols = [w // 2] if w % 2 else [w // 2 - 1, w // 2]
    start = (center_rows[0], center_cols[0])
    offsets = [(0, 0)]
    max_radius = max(h, w)
    for radius in range(1, max_radius + 1):
        for dc in range(-radius + 1, radius + 1):
            offsets.append((-radius, dc))
        for dr in range(-radius + 1, radius + 1):
            offsets.append((dr, radius))
        for dc in range(radius - 1, -radius - 1, -1):
            offsets.append((radius, dc))
        for dr in range(radius - 1, -radius - 1, -1):
            offsets.append((dr, -radius))
    out = []
    seen = set()
    for dr, dc in offsets:
        y, x = start[0] + dr, start[1] + dc
        if 0 <= y < h and 0 <= x < w and (y, x) not in seen:
            seen.add((y, x))
            out.append((y, x))
        if len(out) == h * w:
            break
    return out


def centered_square_ring_indices(h: int, w: int) -> list[tuple[int, int]]:
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    coords = [(y, x) for y in range(h) for x in range(w)]
    return sorted(coords, key=lambda p: (max(abs(p[0] - cy), abs(p[1] - cx)), p[0], p[1]))


def centered_square_ring_profile(hw: np.ndarray) -> np.ndarray:
    values = np.asarray(hw, dtype="float64")
    h, w = values.shape
    center_rows = [h // 2] if h % 2 else [h // 2 - 1, h // 2]
    center_cols = [w // 2] if w % 2 else [w // 2 - 1, w // 2]
    max_radius = max(
        max(min(abs(r - cr) for cr in center_rows) for r in range(h)),
        max(min(abs(c - cc) for cc in center_cols) for c in range(w)),
    )
    out = []
    for radius in range(max_radius + 1):
        vals = []
        for r in range(h):
            row_dist = min(abs(r - cr) for cr in center_rows)
            for c in range(w):
                col_dist = min(abs(c - cc) for cc in center_cols)
                if max(row_dist, col_dist) == radius:
                    vals.append(abs(float(values[r, c])))
        out.append(float(np.mean(vals)) if vals else 0.0)
    return np.asarray(out, dtype="float64")


def centered_square_ring_pixel_counts(h: int, w: int) -> np.ndarray:
    values = np.ones((int(h), int(w)), dtype="float64")
    center_rows = [h // 2] if h % 2 else [h // 2 - 1, h // 2]
    center_cols = [w // 2] if w % 2 else [w // 2 - 1, w // 2]
    max_radius = max(
        max(min(abs(r - cr) for cr in center_rows) for r in range(h)),
        max(min(abs(c - cc) for cc in center_cols) for c in range(w)),
    )
    counts = []
    for radius in range(max_radius + 1):
        count = 0
        for r in range(h):
            row_dist = min(abs(r - cr) for cr in center_rows)
            for c in range(w):
                col_dist = min(abs(c - cc) for cc in center_cols)
                if max(row_dist, col_dist) == radius:
                    count += int(values[r, c])
        counts.append(count)
    return np.asarray(counts, dtype=int)


def _profile_from_hw(hw: np.ndarray, order: list[tuple[int, int]]) -> np.ndarray:
    vals = np.asarray([hw[y, x] for y, x in order], dtype="float64")
    return vals


def _cache_path(exp, examples, *, layers: list[str], n_points: int, build_missing_layer_maps: bool) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [item.path for item in examples],
        "layers": layers,
        "n_points": int(n_points),
        "build_missing_layer_maps": bool(build_missing_layer_maps),
        "profiles": ["jpeg_zigzag", "center_spiral", "centered_square_rings"],
        "method_version": 3,
    }
    return Path(exp.derived_cache_dir) / f"patch_spread_profiles_{stable_hash(payload)}.pkl"


def _delta_cache_path(exp, example, *, layer_name: str) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "path": example.path,
        "drop": float(example.drop),
        "success": bool(example.success),
        "layer_name": layer_name,
        "img_size": int(exp.config.attack.img_size),
        "patch_bbox": list(example.patch_bbox),
        "artifact": "activation_delta_only",
        "method_version": 1,
    }
    out = Path(exp.derived_cache_dir) / "patch_spread_deltas" / layer_name.replace(".", "_")
    out.mkdir(parents=True, exist_ok=True)
    return out / f"patch_spread_delta_{stable_hash(payload)}.npz"


def _compute_or_load_delta_map(exp, example, *, layer_name: str, build_missing: bool):
    path = _delta_cache_path(exp, example, layer_name=layer_name)
    if path.exists():
        data = np.load(path, allow_pickle=False)
        return {"delta_chw": data["delta_chw"], "cache_path": str(path), "loaded_from_cache": True}

    full_map = exp._load_layer_map_cache(example, layer_name=layer_name)
    if full_map is not None:
        return {"delta_chw": full_map["delta_chw"], "cache_path": full_map["cache_path"], "loaded_from_cache": True}

    if not build_missing:
        return None

    import torch

    model = exp.load_model()
    clean_img, patched_img, _patch_bbox = exp._images_for_example(example)
    clean_x = exp._preprocess(clean_img)
    patched_x = exp._preprocess(patched_img)
    with torch.no_grad():
        clean_act = capture_activation(model, clean_x, layer_name)
        patched_act = capture_activation(model, patched_x, layer_name)
    delta = (patched_act - clean_act)[0].detach().cpu().numpy().astype("float32", copy=False)
    np.savez_compressed(path, delta_chw=delta)
    return {"delta_chw": delta, "cache_path": str(path), "loaded_from_cache": False}


def compute_or_load_patch_spread_profiles(
    exp,
    *,
    layers: list[str] | None = None,
    max_examples: int | None = None,
    n_points: int = 128,
    build_missing_layer_maps: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    cache = exp.get_cache()
    examples = list(cache.examples)
    if max_examples is not None:
        examples = examples[: int(max_examples)]
    layers = list(layers or exp.all_display_layer_names())
    path = _cache_path(
        exp,
        examples,
        layers=layers,
        n_points=int(n_points),
        build_missing_layer_maps=bool(build_missing_layer_maps),
    )
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached

    delta_status = {"computed": 0, "loaded": 0, "missing": 0}
    rows = []
    progress = None
    try:
        from tqdm.auto import tqdm

        progress = tqdm(total=len(layers) * len(examples), desc="patch-spread deltas", unit="img-layer")
    except Exception:
        progress = None
    try:
        for layer in layers:
            for example in examples:
                maps = _compute_or_load_delta_map(
                    exp,
                    example,
                    layer_name=layer,
                    build_missing=bool(build_missing_layer_maps),
                )
                if maps is None:
                    delta_status["missing"] += 1
                    if progress is not None:
                        progress.update(1)
                    continue
                if maps.get("loaded_from_cache"):
                    delta_status["loaded"] += 1
                else:
                    delta_status["computed"] += 1
                hw = reduce_chw_to_hw(maps["delta_chw"], mode="l2").detach().cpu().numpy()
                h, w = hw.shape
                profiles = {
                    "jpeg_zigzag": _profile_from_hw(hw, jpeg_zigzag_indices(h, w)),
                    "center_spiral": _profile_from_hw(hw, centered_spiral_indices(h, w)),
                    "centered_square_rings": centered_square_ring_profile(hw),
                }
                for profile_name, profile in profiles.items():
                    for idx, value in enumerate(profile):
                        if profile_name == "jpeg_zigzag":
                            x_value = float((idx + 1) / max(1, len(profile)))
                        else:
                            x_value = float(idx + 1)
                        rows.append(
                            {
                                "path": example.path,
                                "layer": layer,
                                "success": bool(example.success),
                                "drop": float(example.drop),
                                "profile": profile_name,
                                "step": int(idx),
                                "x": x_value,
                                "value": float(value),
                            }
                        )
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(delta_status)
    finally:
        if progress is not None:
            progress.close()
    if not rows:
        raise RuntimeError(
            "No patch-spread rows were produced. There are no cached full layer maps or patch-spread "
            f"delta maps for layers={layers}, and build_missing_layer_maps={build_missing_layer_maps}."
        )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["layer", "success", "profile", "step", "x"], as_index=False)["value"]
        .agg(["mean", "std", "var", "count"])
        .reset_index()
    )
    result = {
        "profiles_df": df,
        "summary_df": summary,
        "delta_status": delta_status,
        "cache_path": str(path),
        "loaded_from_cache": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh)
    return result
