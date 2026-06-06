from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


def select_balanced_examples(exp, *, per_class: int):
    cache = exp.get_cache()
    selected = cache.successes[: int(per_class)] + cache.failures[: int(per_class)]
    if len(cache.successes) < int(per_class) or len(cache.failures) < int(per_class):
        raise RuntimeError(f"Need {per_class} success/fail examples, got {len(cache.successes)} / {len(cache.failures)}")
    return selected


def channel_summary_cache_path(exp, examples, *, layer_name: str) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [item.path for item in examples],
        "target_layer": layer_name,
        "detect_layer": exp.config.detect_layer,
        "target_mode": exp.config.target_mode,
        "n_steps": int(exp.config.n_steps),
        "alpha_batch_size": int(exp.config.alpha_batch_size),
        "imgsz": int(exp.config.attack.imgsz),
        "method_version": 1,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
    return exp.derived_cache_dir / f"channel_summary_{key}.pkl"


def _as_chw(array):
    import numpy as np

    arr = np.asarray(array, dtype="float64")
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected [C,H,W] or [1,C,H,W], got {arr.shape}")
    return arr


def channel_vectors(delta_chw, importance_chw, *, variant: str) -> tuple[Any, Any]:
    import numpy as np

    delta = _as_chw(delta_chw)
    importance = _as_chw(importance_chw)
    c = min(delta.shape[0], importance.shape[0])
    h = min(delta.shape[1], importance.shape[1])
    w = min(delta.shape[2], importance.shape[2])
    delta = delta[:c, :h, :w]
    importance = importance[:c, :h, :w]
    if variant == "unsigned":
        return np.abs(importance).sum(axis=(1, 2)), np.abs(delta).sum(axis=(1, 2))
    if variant == "signed":
        return importance.sum(axis=(1, 2)), delta.sum(axis=(1, 2))
    raise ValueError(f"Unknown variant: {variant!r}")


def compute_or_load_channel_summary(
    exp,
    examples,
    *,
    layer_name: str = "model.22",
    force: bool = False,
) -> dict[str, Any]:
    import numpy as np

    from .yolo import get_module_by_name

    selected = list(examples)
    cache_path = channel_summary_cache_path(exp, selected, layer_name=layer_name)
    if cache_path.exists() and not force:
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(cache_path)
        return cached

    model = None
    layer = None
    sums: dict[str, dict[str, Any]] = {
        "unsigned": {"importance": None, "delta": None},
        "signed": {"importance": None, "delta": None},
    }
    example_rows: list[dict[str, Any]] = []
    for idx, example in enumerate(selected, start=1):
        layer_maps = exp._load_layer_map_cache(example, layer_name=layer_name, include_clean_activation=False)
        if layer_maps is None:
            if model is None or layer is None:
                _yolo, model = exp.load_model()
                layer = get_module_by_name(model, layer_name)
            ctx = exp._context_for_example(example, image_variant="clean")
            layer_maps = exp._compute_or_load_segmentig_layer_maps(
                example,
                ctx,
                model=model,
                layer=layer,
                layer_name=layer_name,
                include_clean_activation=False,
            )
        delta_chw = layer_maps["delta_chw"]
        importance_chw = layer_maps["segmentig_chw"]
        for variant in ("unsigned", "signed"):
            importance_vec, delta_vec = channel_vectors(delta_chw, importance_chw, variant=variant)
            if sums[variant]["importance"] is None:
                sums[variant]["importance"] = np.zeros_like(importance_vec, dtype="float64")
                sums[variant]["delta"] = np.zeros_like(delta_vec, dtype="float64")
            sums[variant]["importance"][: importance_vec.size] += importance_vec
            sums[variant]["delta"][: delta_vec.size] += delta_vec
        example_rows.append(
            {
                "path": example.path,
                "success": bool(example.success),
                "drop": float(example.drop),
                "layer_maps_cache_path": str(layer_maps["cache_path"]),
                "layer_maps_loaded_from_cache": bool(layer_maps["loaded_from_cache"]),
            }
        )
        if idx % max(1, int(getattr(exp.config, "metrics_batch_size", 64))) == 0:
            exp._release_batch_memory()
    exp._release_batch_memory()

    first_variant = sums["unsigned"]["importance"]
    if first_variant is None:
        channels = np.asarray([], dtype=int)
    else:
        channels = np.arange(first_variant.size, dtype=int)
    result = {
        "layer_name": layer_name,
        "channels": channels,
        "variants": sums,
        "examples": example_rows,
        "n_examples": len(example_rows),
        "n_success": sum(row["success"] for row in example_rows),
        "n_fail": sum(not row["success"] for row in example_rows),
        "cache_path": str(cache_path),
        "loaded_from_cache": False,
    }
    with cache_path.open("wb") as f:
        pickle.dump(result, f)
    return result


def channel_summary_frame(summary: dict[str, Any], *, variant: str, sorted_by_importance: bool = False):
    import numpy as np
    import pandas as pd

    channels = np.asarray(summary["channels"], dtype=int)
    importance = np.asarray(summary["variants"][variant]["importance"], dtype="float64")
    delta = np.asarray(summary["variants"][variant]["delta"], dtype="float64")
    order = np.argsort(-importance, kind="stable") if sorted_by_importance else np.arange(channels.size)
    return pd.DataFrame(
        {
            "rank": np.arange(1, order.size + 1, dtype=int),
            "channel": channels[order],
            "importance_sum": importance[order],
            "delta_sum": delta[order],
        }
    )


def plot_channel_summary_grid(summary: dict[str, Any], *, variant: str):
    import matplotlib.pyplot as plt
    import numpy as np

    if variant not in summary["variants"]:
        raise ValueError(f"Unknown variant: {variant!r}")
    layer_name = summary["layer_name"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.5), constrained_layout=True)
    specs = [
        (False, "channel order"),
        (True, "sorted by importance desc"),
    ]
    colors = {"importance": "#4C78A8", "delta": "#F58518"}
    for row_idx, (is_sorted, row_label) in enumerate(specs):
        frame = channel_summary_frame(summary, variant=variant, sorted_by_importance=is_sorted)
        x = np.arange(len(frame), dtype=int)
        for col_idx, (column, label, color) in enumerate(
            [
                ("importance_sum", "importance", colors["importance"]),
                ("delta_sum", "delta", colors["delta"]),
            ]
        ):
            ax = axes[row_idx, col_idx]
            ax.bar(x, frame[column], color=color, width=0.9)
            ax.axhline(0.0, color="0.25", linewidth=0.8)
            ax.set_title(f"{label} | {row_label}")
            ax.set_xlabel("channel" if not is_sorted else "rank by importance")
            ax.set_ylabel(f"{variant} sum")
            if not is_sorted:
                ax.set_xticks(x[:: max(1, len(x) // 16)])
                ax.set_xticklabels(frame["channel"].iloc[:: max(1, len(x) // 16)])
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"{layer_name}: channel-level {variant} sums "
        f"({summary['n_examples']} images: {summary['n_success']} success / {summary['n_fail']} fail)"
    )
    return fig


def channel_example_scores(exp, examples, *, layer_name: str, channel: int):
    import numpy as np
    import pandas as pd

    rows = []
    for example in examples:
        layer_maps = exp._load_layer_map_cache(example, layer_name=layer_name, include_clean_activation=False)
        if layer_maps is None:
            raise FileNotFoundError(f"Layer map cache is missing for {example.path}")
        importance = _as_chw(layer_maps["segmentig_chw"])
        delta = _as_chw(layer_maps["delta_chw"])
        ch = int(channel)
        if ch < 0 or ch >= importance.shape[0]:
            raise IndexError(f"Channel {ch} is outside importance shape {importance.shape}")
        rows.append(
            {
                "path": example.path,
                "success": bool(example.success),
                "drop": float(example.drop),
                "channel": ch,
                "importance_abs_sum": float(np.abs(importance[ch]).sum()),
                "importance_signed_sum": float(importance[ch].sum()),
                "delta_abs_sum": float(np.abs(delta[ch]).sum()),
                "delta_signed_sum": float(delta[ch].sum()),
                "layer_maps_cache_path": str(layer_maps["cache_path"]),
            }
        )
    return pd.DataFrame(rows)


def _normalize_positive(values, *, q: float = 99.0):
    import numpy as np

    arr = np.asarray(values, dtype="float64")
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype="float64")
    scale = float(np.nanpercentile(arr[finite], float(q)))
    return np.clip(arr / max(scale, 1e-12), 0.0, 1.0)


def _normalize_signed(values, *, q: float = 99.0):
    import numpy as np

    arr = np.asarray(values, dtype="float64")
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype="float64")
    scale = float(np.nanpercentile(np.abs(arr[finite]), float(q)))
    return np.clip(arr / max(scale, 1e-12), -1.0, 1.0)


def _resize_map(values, size_xy):
    import numpy as np
    from PIL import Image

    arr = np.asarray(values, dtype="float32")
    image = Image.fromarray(arr, mode="F")
    return np.asarray(image.resize(size_xy, resample=Image.Resampling.BILINEAR), dtype="float64")


def plot_channel_importance_overlays(
    exp,
    examples,
    *,
    layer_name: str,
    channel: int,
    max_examples: int = 5,
):
    import matplotlib.pyplot as plt
    import numpy as np

    scores = channel_example_scores(exp, examples, layer_name=layer_name, channel=int(channel))
    top = scores.sort_values("importance_abs_sum", ascending=False).head(int(max_examples))
    fig, axes = plt.subplots(len(top), 3, figsize=(13, max(3.0, 3.0 * len(top))), squeeze=False, constrained_layout=True)
    by_path = {item.path: item for item in examples}
    for row_idx, row in enumerate(top.itertuples(index=False)):
        example = by_path[row.path]
        clean_lb, _patched_lb, _patch_bbox = exp._images_for_example(example)
        layer_maps = exp._load_layer_map_cache(example, layer_name=layer_name, include_clean_activation=False)
        if layer_maps is None:
            raise FileNotFoundError(f"Layer map cache is missing for {example.path}")
        importance = _as_chw(layer_maps["segmentig_chw"])[int(channel)]
        abs_norm = _resize_map(_normalize_positive(np.abs(importance)), clean_lb.size)
        signed_norm = _resize_map(_normalize_signed(importance), clean_lb.size)
        signed_alpha = np.clip(np.abs(signed_norm), 0.0, 1.0) * 0.65

        axes[row_idx, 0].imshow(clean_lb)
        axes[row_idx, 0].set_title(
            f"{'success' if row.success else 'fail'} | drop={row.drop:.3f}\n{Path(row.path).name}",
            fontsize=9,
        )
        axes[row_idx, 1].imshow(clean_lb)
        axes[row_idx, 1].imshow(abs_norm, cmap="magma", alpha=np.clip(abs_norm, 0.0, 1.0) * 0.65, vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f"abs importance overlay\nsum={row.importance_abs_sum:.1f}", fontsize=9)
        axes[row_idx, 2].imshow(clean_lb)
        axes[row_idx, 2].imshow(signed_norm, cmap="coolwarm", alpha=signed_alpha, vmin=-1, vmax=1)
        axes[row_idx, 2].set_title(f"signed importance overlay\nsum={row.importance_signed_sum:.1f}", fontsize=9)
        for col in range(3):
            axes[row_idx, col].axis("off")
    fig.suptitle(f"{layer_name} channel {int(channel)}: top-{len(top)} images by abs importance")
    return fig, top.reset_index(drop=True)


def top_channels_by_importance(summary: dict[str, Any], *, percent: float):
    import numpy as np

    channels = np.asarray(summary["channels"], dtype=int)
    importance = np.asarray(summary["variants"]["unsigned"]["importance"], dtype="float64")
    if float(percent) >= 100:
        k = channels.size
    else:
        k = max(1, int(round(float(percent) / 100.0 * channels.size)))
    order = np.argsort(-importance, kind="stable")[:k]
    return channels[order]


def ranked_neuron_delta_by_channel_fraction(summary: dict[str, Any], examples, *, exp, fractions=(1, 10, 100)):
    import numpy as np

    layer_name = str(summary["layer_name"])
    rows: dict[str, dict[str, Any]] = {}
    for percent in fractions:
        selected_channels = top_channels_by_importance(summary, percent=float(percent))
        importance_success = None
        importance_fail = None
        delta_success = None
        delta_fail = None
        delta_abs_success = None
        delta_abs_fail = None
        n_success = 0
        n_fail = 0
        flat_neuron_ids = None
        for example in examples:
            layer_maps = exp._load_layer_map_cache(example, layer_name=layer_name, include_clean_activation=False)
            if layer_maps is None:
                raise FileNotFoundError(f"Layer map cache is missing for {example.path}")
            importance = _as_chw(layer_maps["segmentig_chw"])[selected_channels]
            delta = _as_chw(layer_maps["delta_chw"])[selected_channels]
            imp_flat = np.abs(importance).reshape(-1)
            delta_flat = delta.reshape(-1)
            delta_abs_flat = np.abs(delta_flat)
            if flat_neuron_ids is None:
                h, w = importance.shape[1], importance.shape[2]
                channel_ids = np.repeat(selected_channels, h * w)
                spatial_ids = np.tile(np.arange(h * w, dtype=int), selected_channels.size)
                flat_neuron_ids = channel_ids.astype("int64") * 1_000_000 + spatial_ids.astype("int64")
            if bool(example.success):
                importance_success = imp_flat.copy() if importance_success is None else importance_success + imp_flat
                delta_success = delta_flat.copy() if delta_success is None else delta_success + delta_flat
                delta_abs_success = delta_abs_flat.copy() if delta_abs_success is None else delta_abs_success + delta_abs_flat
                n_success += 1
            else:
                importance_fail = imp_flat.copy() if importance_fail is None else importance_fail + imp_flat
                delta_fail = delta_flat.copy() if delta_fail is None else delta_fail + delta_flat
                delta_abs_fail = delta_abs_flat.copy() if delta_abs_fail is None else delta_abs_fail + delta_abs_flat
                n_fail += 1
        if importance_success is None or importance_fail is None:
            raise RuntimeError("Need both success and fail examples.")
        total_importance = importance_success + importance_fail
        order = np.argsort(-total_importance, kind="stable")
        rows[f"top_{int(percent)}pct_channels" if percent < 100 else "all_channels"] = {
            "percent": float(percent),
            "channels": selected_channels,
            "neuron_ids": flat_neuron_ids[order],
            "importance": total_importance[order],
            "success_delta_mean": (delta_success / max(1, n_success))[order],
            "fail_delta_mean": (delta_fail / max(1, n_fail))[order],
            "success_delta_abs_mean": (delta_abs_success / max(1, n_success))[order],
            "fail_delta_abs_mean": (delta_abs_fail / max(1, n_fail))[order],
            "n_success": n_success,
            "n_fail": n_fail,
        }
    return rows


def _analysis_cache_path(exp, examples, *, layer_name: str, analysis_name: str, fractions):
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [item.path for item in examples],
        "target_layer": layer_name,
        "detect_layer": exp.config.detect_layer,
        "target_mode": exp.config.target_mode,
        "n_steps": int(exp.config.n_steps),
        "alpha_batch_size": int(exp.config.alpha_batch_size),
        "imgsz": int(exp.config.attack.imgsz),
        "analysis_name": str(analysis_name),
        "fractions": [float(v) for v in fractions],
        "method_version": 1,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
    return exp.derived_cache_dir / f"{analysis_name}_{key}.pkl"


def compute_or_load_ranked_neuron_delta_by_channel_fraction(
    exp,
    summary: dict[str, Any],
    examples,
    *,
    fractions=(1, 10, 100),
    force: bool = False,
):
    layer_name = str(summary["layer_name"])
    selected = list(examples)
    cache_path = _analysis_cache_path(
        exp,
        selected,
        layer_name=layer_name,
        analysis_name="ranked_neuron_delta_abs_v2",
        fractions=fractions,
    )
    if cache_path.exists() and not force:
        with cache_path.open("rb") as f:
            out = pickle.load(f)
        out["loaded_from_cache"] = True
        out["cache_path"] = str(cache_path)
        return out
    ranked = ranked_neuron_delta_by_channel_fraction(summary, selected, exp=exp, fractions=fractions)
    out = {"ranked": ranked, "loaded_from_cache": False, "cache_path": str(cache_path)}
    with cache_path.open("wb") as f:
        pickle.dump(out, f)
    return out


def _sample_line_points(values, *, max_points: int = 6000):
    import numpy as np

    arr = np.asarray(values, dtype="float64")
    if arr.size <= int(max_points):
        return np.arange(arr.size), arr
    idx = np.linspace(0, arr.size - 1, int(max_points), dtype=int)
    return idx, arr[idx]


def _moving_average_np(values, window: int):
    import numpy as np

    arr = np.asarray(values, dtype="float64")
    if arr.size == 0 or int(window) <= 1:
        return arr
    window = min(int(window), arr.size)
    kernel = np.ones(window, dtype="float64") / float(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_ranked_neuron_delta_by_channel_fraction(ranked: dict[str, dict[str, Any]], *, smooth_window: int = 401):
    import matplotlib.pyplot as plt
    import numpy as np

    items = list(ranked.items())
    fig, axes = plt.subplots(6, len(items), figsize=(6.2 * len(items), 19.5), squeeze=False, constrained_layout=True)
    colors = {"success": "#4C78A8", "fail": "#F58518"}
    for col, (name, data) in enumerate(items):
        x_success, y_success = _sample_line_points(data["success_delta_mean"])
        x_fail, y_fail = _sample_line_points(data["fail_delta_mean"])
        success_abs = data.get("success_delta_abs_mean", np.abs(data["success_delta_mean"]))
        fail_abs = data.get("fail_delta_abs_mean", np.abs(data["fail_delta_mean"]))
        x_success_abs, y_success_abs = _sample_line_points(success_abs)
        x_fail_abs, y_fail_abs = _sample_line_points(fail_abs)
        title = f"{data['percent']:.0f}% channels ({len(data['channels'])} ch, {len(data['importance'])} neurons)"
        axes[0, col].scatter(x_success, y_success, s=2, alpha=0.35, color=colors["success"])
        axes[0, col].plot(np.arange(len(data["success_delta_mean"])), _moving_average_np(data["success_delta_mean"], smooth_window), color="black", linewidth=1.1)
        axes[0, col].axhline(0, color="0.4", linewidth=0.8)
        axes[0, col].set_title(f"success | {title}", fontsize=10)
        axes[0, col].set_ylabel("mean signed delta")
        axes[0, col].grid(alpha=0.25)

        axes[1, col].scatter(x_fail, y_fail, s=2, alpha=0.35, color=colors["fail"])
        axes[1, col].plot(np.arange(len(data["fail_delta_mean"])), _moving_average_np(data["fail_delta_mean"], smooth_window), color="black", linewidth=1.1)
        axes[1, col].axhline(0, color="0.4", linewidth=0.8)
        axes[1, col].set_title(f"fail | {title}", fontsize=10)
        axes[1, col].set_ylabel("mean signed delta")
        axes[1, col].grid(alpha=0.25)

        axes[2, col].plot(_moving_average_np(data["success_delta_mean"], smooth_window), color=colors["success"], linewidth=1.3, label="success")
        axes[2, col].plot(_moving_average_np(data["fail_delta_mean"], smooth_window), color=colors["fail"], linewidth=1.3, label="fail")
        axes[2, col].axhline(0, color="0.4", linewidth=0.8)
        axes[2, col].set_title(f"combined smoothed | {title}", fontsize=10)
        axes[2, col].set_xlabel("rank by neuron importance")
        axes[2, col].set_ylabel("mean signed delta")
        axes[2, col].grid(alpha=0.25)
        axes[2, col].legend()

        axes[3, col].scatter(x_success_abs, y_success_abs, s=2, alpha=0.35, color=colors["success"])
        axes[3, col].plot(np.arange(len(success_abs)), _moving_average_np(success_abs, smooth_window), color="black", linewidth=1.1)
        axes[3, col].set_title(f"success abs | {title}", fontsize=10)
        axes[3, col].set_ylabel("mean abs delta")
        axes[3, col].grid(alpha=0.25)

        axes[4, col].scatter(x_fail_abs, y_fail_abs, s=2, alpha=0.35, color=colors["fail"])
        axes[4, col].plot(np.arange(len(fail_abs)), _moving_average_np(fail_abs, smooth_window), color="black", linewidth=1.1)
        axes[4, col].set_title(f"fail abs | {title}", fontsize=10)
        axes[4, col].set_ylabel("mean abs delta")
        axes[4, col].grid(alpha=0.25)

        axes[5, col].plot(
            _moving_average_np(success_abs, smooth_window),
            color=colors["success"],
            linewidth=1.3,
            label="success mean |delta|",
        )
        axes[5, col].plot(
            _moving_average_np(fail_abs, smooth_window),
            color=colors["fail"],
            linewidth=1.3,
            label="fail mean |delta|",
        )
        axes[5, col].set_title(f"combined smoothed mean abs | {title}", fontsize=10)
        axes[5, col].set_xlabel("rank by neuron importance")
        axes[5, col].set_ylabel("mean abs delta")
        axes[5, col].grid(alpha=0.25)
        axes[5, col].legend()
    fig.suptitle("Neuron deltas ranked by unsigned importance within selected top channels")
    return fig


def square_spiral_offsets(height: int, width: int):
    offsets = [(0, 0)]
    max_radius = max(int(height), int(width))
    for radius in range(1, max_radius + 1):
        y = -radius
        for x in range(-radius + 1, radius + 1):
            offsets.append((y, x))
        x = radius
        for y in range(-radius + 1, radius + 1):
            offsets.append((y, x))
        y = radius
        for x in range(radius - 1, -radius - 1, -1):
            offsets.append((y, x))
        x = -radius
        for y in range(radius - 1, -radius - 1, -1):
            offsets.append((y, x))
        if len(offsets) >= int(height) * int(width):
            break
    return offsets[: int(height) * int(width)]


def object_center_on_grid(example, *, grid_hw: tuple[int, int], imgsz: int):
    detection = example.clean_detection or {}
    bbox = detection.get("bbox_xyxy_orig")
    if not bbox:
        return (int(grid_hw[0]) // 2, int(grid_hw[1]) // 2)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    h, w = int(grid_hw[0]), int(grid_hw[1])
    row = int(round(cy / float(imgsz) * (h - 1)))
    col = int(round(cx / float(imgsz) * (w - 1)))
    return (row, col)


def spiral_profile(values_hw, *, center_rc: tuple[int, int], allow_oob_zero: bool):
    import numpy as np

    values = np.asarray(values_hw, dtype="float64")
    h, w = values.shape
    center_r, center_c = int(center_rc[0]), int(center_rc[1])
    out = []
    for dr, dc in square_spiral_offsets(h, w):
        r = center_r + int(dr)
        c = center_c + int(dc)
        if 0 <= r < h and 0 <= c < w:
            out.append(float(values[r, c]))
        elif allow_oob_zero:
            out.append(0.0)
    return np.asarray(out, dtype="float64")


def spiral_delta_profiles_by_channel_fraction(summary: dict[str, Any], examples, *, exp, fractions=(1, 10, 100)):
    import numpy as np

    layer_name = str(summary["layer_name"])
    rows: dict[str, dict[str, Any]] = {}
    for percent in fractions:
        selected_channels = top_channels_by_importance(summary, percent=float(percent))
        records = []
        for example in examples:
            layer_maps = exp._load_layer_map_cache(example, layer_name=layer_name, include_clean_activation=False)
            if layer_maps is None:
                raise FileNotFoundError(f"Layer map cache is missing for {example.path}")
            delta = _as_chw(layer_maps["delta_chw"])[selected_channels]
            importance = _as_chw(layer_maps["segmentig_chw"])[selected_channels]
            spatial = np.mean(np.abs(delta), axis=0)
            importance_spatial = np.mean(np.abs(importance), axis=0)
            h, w = spatial.shape
            image_center = (h // 2, w // 2)
            object_center = object_center_on_grid(example, grid_hw=(h, w), imgsz=int(exp.config.attack.imgsz))
            records.append(
                {
                    "success": bool(example.success),
                    "drop": float(example.drop),
                    "image_center": spiral_profile(spatial, center_rc=image_center, allow_oob_zero=False),
                    "object_center": spiral_profile(spatial, center_rc=object_center, allow_oob_zero=True),
                    "image_center_importance": spiral_profile(importance_spatial, center_rc=image_center, allow_oob_zero=False),
                    "object_center_importance": spiral_profile(importance_spatial, center_rc=object_center, allow_oob_zero=True),
                }
            )
        rows[f"top_{int(percent)}pct_channels" if percent < 100 else "all_channels"] = {
            "percent": float(percent),
            "channels": selected_channels,
            "records": records,
        }
    return rows


def compute_or_load_spiral_delta_profiles_by_channel_fraction(
    exp,
    summary: dict[str, Any],
    examples,
    *,
    fractions=(1, 10, 100),
    force: bool = False,
):
    layer_name = str(summary["layer_name"])
    selected = list(examples)
    cache_path = _analysis_cache_path(
        exp,
        selected,
        layer_name=layer_name,
        analysis_name="spiral_delta_profiles_with_importance_v2",
        fractions=fractions,
    )
    if cache_path.exists() and not force:
        with cache_path.open("rb") as f:
            out = pickle.load(f)
        out["loaded_from_cache"] = True
        out["cache_path"] = str(cache_path)
        return out
    profiles = spiral_delta_profiles_by_channel_fraction(summary, selected, exp=exp, fractions=fractions)
    out = {"profiles": profiles, "loaded_from_cache": False, "cache_path": str(cache_path)}
    with cache_path.open("wb") as f:
        pickle.dump(out, f)
    return out


def _mean_profiles(records, *, key: str, success: bool):
    import numpy as np

    profiles = [np.asarray(row[key], dtype="float64") for row in records if bool(row["success"]) == bool(success)]
    min_len = min(profile.size for profile in profiles)
    stack = np.stack([profile[:min_len] for profile in profiles], axis=0)
    return stack.mean(axis=0), stack.std(axis=0)


def plot_spiral_delta_profiles(spiral_profiles: dict[str, dict[str, Any]]):
    import matplotlib.pyplot as plt
    import numpy as np

    items = list(spiral_profiles.items())
    fig, axes = plt.subplots(4, len(items), figsize=(6.2 * len(items), 13.5), squeeze=False, constrained_layout=True)
    colors = {True: "#4C78A8", False: "#F58518"}
    labels = {True: "success", False: "fail"}
    for col, (_name, data) in enumerate(items):
        for row_idx, key in enumerate(("image_center", "object_center")):
            ax = axes[row_idx, col]
            for success in (True, False):
                mean, std = _mean_profiles(data["records"], key=key, success=success)
                x = np.arange(mean.size, dtype=int)
                ax.plot(x, mean, color=colors[success], label=labels[success])
                ax.fill_between(x, mean - std, mean + std, color=colors[success], alpha=0.15)
            ax.set_title(
                f"{'image center' if key == 'image_center' else 'object center'} | "
                f"{data['percent']:.0f}% channels ({len(data['channels'])} ch)"
            )
            ax.set_xlabel("spiral step")
            ax.set_ylabel("mean abs delta")
            ax.grid(alpha=0.25)
            ax.legend()
        for row_idx, key in enumerate(("image_center_importance", "object_center_importance"), start=2):
            ax = axes[row_idx, col]
            delta_key = "image_center" if key == "image_center_importance" else "object_center"
            for success in (True, False):
                delta_mean, _delta_std = _mean_profiles(data["records"], key=delta_key, success=success)
                x_delta = np.arange(delta_mean.size, dtype=int)
                ax.plot(x_delta, delta_mean, color=colors[success], linewidth=1.3, label=f"{labels[success]} delta")
            profiles = [
                np.asarray(row[key], dtype="float64")
                for row in data["records"]
                if key in row
            ]
            if profiles:
                min_len = min(profile.size for profile in profiles)
                mean = np.stack([profile[:min_len] for profile in profiles], axis=0).mean(axis=0)
                x = np.arange(mean.size, dtype=int)
                ax.plot(x, mean, color="black", linestyle="--", linewidth=1.7, label="importance")
            ax.set_title(
                f"{'image center importance' if key == 'image_center_importance' else 'object center importance'} | "
                f"{data['percent']:.0f}% channels ({len(data['channels'])} ch)"
            )
            ax.set_xlabel("spiral step")
            ax.set_ylabel("mean abs delta / importance")
            ax.grid(alpha=0.25)
            ax.legend()
    fig.suptitle("Patch spread and importance by square spiral from image center and object center")
    return fig


def centered_square_ring_profile(values_hw):
    import numpy as np

    values = np.asarray(values_hw, dtype="float64")
    h, w = values.shape
    center_rows = [h // 2] if h % 2 else [h // 2 - 1, h // 2]
    center_cols = [w // 2] if w % 2 else [w // 2 - 1, w // 2]
    max_radius = max(
        max(min(abs(r - cr) for cr in center_rows) for r in range(h)),
        max(min(abs(c - cc) for cc in center_cols) for c in range(w)),
    )
    out = []
    for radius in range(max_radius + 1):
        total = 0.0
        count = 0
        for r in range(h):
            row_dist = min(abs(r - cr) for cr in center_rows)
            for c in range(w):
                col_dist = min(abs(c - cc) for cc in center_cols)
                if max(row_dist, col_dist) == radius:
                    total += abs(float(values[r, c]))
                    count += 1
        out.append(total / max(1, count))
    return np.asarray(out, dtype="float64")


def centered_square_ring_pixel_counts(height: int, width: int):
    import numpy as np

    h, w = int(height), int(width)
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
                    count += 1
        counts.append(count)
    return np.asarray(counts, dtype=int)


def square_ring_delta_profiles_by_channel_fraction(summary: dict[str, Any], examples, *, exp, fractions=(1, 10, 100)):
    import numpy as np

    layer_name = str(summary["layer_name"])
    rows: dict[str, dict[str, Any]] = {}
    for percent in fractions:
        selected_channels = top_channels_by_importance(summary, percent=float(percent))
        records = []
        for example in examples:
            layer_maps = exp._load_layer_map_cache(example, layer_name=layer_name, include_clean_activation=False)
            if layer_maps is None:
                raise FileNotFoundError(f"Layer map cache is missing for {example.path}")
            delta = _as_chw(layer_maps["delta_chw"])[selected_channels]
            importance = _as_chw(layer_maps["segmentig_chw"])[selected_channels]
            spatial = np.sum(np.abs(delta), axis=0)
            importance_spatial = np.sum(np.abs(importance), axis=0)
            ring_counts = centered_square_ring_pixel_counts(*spatial.shape)
            records.append(
                {
                    "success": bool(example.success),
                    "drop": float(example.drop),
                    "image_center_square_rings": centered_square_ring_profile(spatial),
                    "image_center_square_ring_importance": centered_square_ring_profile(importance_spatial),
                    "ring_pixel_counts": ring_counts,
                }
            )
        rows[f"top_{int(percent)}pct_channels" if percent < 100 else "all_channels"] = {
            "percent": float(percent),
            "channels": selected_channels,
            "records": records,
        }
    return rows


def compute_or_load_square_ring_delta_profiles_by_channel_fraction(
    exp,
    summary: dict[str, Any],
    examples,
    *,
    fractions=(1, 10, 100),
    force: bool = False,
):
    layer_name = str(summary["layer_name"])
    selected = list(examples)
    cache_path = _analysis_cache_path(
        exp,
        selected,
        layer_name=layer_name,
        analysis_name="square_ring_delta_profiles_pixel_norm_v4",
        fractions=fractions,
    )
    if cache_path.exists() and not force:
        with cache_path.open("rb") as f:
            out = pickle.load(f)
        out["loaded_from_cache"] = True
        out["cache_path"] = str(cache_path)
        return out
    profiles = square_ring_delta_profiles_by_channel_fraction(summary, selected, exp=exp, fractions=fractions)
    out = {"profiles": profiles, "loaded_from_cache": False, "cache_path": str(cache_path)}
    with cache_path.open("wb") as f:
        pickle.dump(out, f)
    return out


def plot_square_ring_delta_profiles(square_ring_profiles: dict[str, dict[str, Any]]):
    import matplotlib.pyplot as plt
    import numpy as np

    items = list(square_ring_profiles.items())
    fig, axes = plt.subplots(2, len(items), figsize=(6.2 * len(items), 8.2), squeeze=False, constrained_layout=True)
    colors = {True: "#4C78A8", False: "#F58518"}
    labels = {True: "success", False: "fail"}
    for col, (_name, data) in enumerate(items):
        ax = axes[0, col]
        for success in (True, False):
            mean, std = _mean_profiles(data["records"], key="image_center_square_rings", success=success)
            x = np.arange(1, mean.size + 1, dtype=int)
            ax.plot(x, mean, marker="o", color=colors[success], label=labels[success])
            ax.fill_between(x, mean - std, mean + std, color=colors[success], alpha=0.15)
        ax.set_title(f"delta | {data['percent']:.0f}% channels ({len(data['channels'])} ch)")
        ax.set_xlabel("square ring index (1 = 4 center pixels)")
        ax.set_ylabel("mean abs delta per pixel in new ring")
        if data["records"]:
            counts = data["records"][0].get("ring_pixel_counts")
            if counts is not None:
                subtitle = ", ".join(str(int(v)) for v in counts)
                ax.text(0.01, 0.98, f"pixels/ring: {subtitle}", transform=ax.transAxes, va="top", fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend()

        ax = axes[1, col]
        for success in (True, False):
            delta_mean, _delta_std = _mean_profiles(data["records"], key="image_center_square_rings", success=success)
            x_delta = np.arange(1, delta_mean.size + 1, dtype=int)
            ax.plot(x_delta, delta_mean, marker="o", color=colors[success], label=f"{labels[success]} delta")
        importance_profiles = [
            np.asarray(row["image_center_square_ring_importance"], dtype="float64")
            for row in data["records"]
            if "image_center_square_ring_importance" in row
        ]
        if importance_profiles:
            min_len = min(profile.size for profile in importance_profiles)
            importance_mean = np.stack([profile[:min_len] for profile in importance_profiles], axis=0).mean(axis=0)
            imp_x = np.arange(1, importance_mean.size + 1, dtype=int)
            ax.plot(
                imp_x,
                importance_mean,
                color="black",
                linestyle="--",
                marker="D",
                markersize=4,
                linewidth=1.7,
                label="importance",
            )
        ax.set_title(f"importance | {data['percent']:.0f}% channels ({len(data['channels'])} ch)")
        ax.set_xlabel("square ring index (1 = 4 center pixels)")
        ax.set_ylabel("mean abs delta / importance per pixel")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Patch spread and importance by centered square rings, normalized by ring pixels")
    return fig
