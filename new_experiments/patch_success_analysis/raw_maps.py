from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _as_chw(values) -> np.ndarray:
    arr = np.asarray(values, dtype="float32")
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={arr.shape}")
    return arr


def reduce_raw_map(chw, *, mode: str) -> np.ndarray:
    arr = _as_chw(chw)
    if mode == "signed_mean":
        return arr.mean(axis=0)
    if mode == "mean_abs":
        return np.abs(arr).mean(axis=0)
    if mode == "l2":
        return np.sqrt(np.mean(arr * arr, axis=0))
    if mode == "sum":
        return arr.sum(axis=0)
    if mode == "abs_sum":
        return np.abs(arr).sum(axis=0)
    raise ValueError(f"Unsupported raw map reduction mode: {mode!r}")


def raw_tensor_stats(values) -> dict[str, float]:
    arr = np.asarray(values, dtype="float64").reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "min": float("nan"),
            "p01": float("nan"),
            "p50": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "mean_abs": float("nan"),
            "l2_rms": float("nan"),
        }
    return {
        "min": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "mean_abs": float(np.mean(np.abs(finite))),
        "l2_rms": float(np.sqrt(np.mean(finite * finite))),
    }


def load_cached_raw_map_rows(exp, examples, *, layer_name: str = "model.22") -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for example in examples:
        layer_maps = exp._load_layer_map_cache(example, layer_name=layer_name, include_clean_activation=False)
        if layer_maps is None:
            skipped.append(
                {
                    "path": example.path,
                    "success": bool(example.success),
                    "drop": float(example.drop),
                    "reason": "missing layer_maps cache",
                }
            )
            continue
        delta = _as_chw(layer_maps["delta_chw"])
        conductance = _as_chw(layer_maps["segmentig_chw"])
        delta_stats = raw_tensor_stats(delta)
        conductance_stats = raw_tensor_stats(conductance)
        row = {
            "path": example.path,
            "success": bool(example.success),
            "drop": float(example.drop),
            "conf_clean": float(example.conf_clean),
            "conf_patch": float(example.conf_patch),
            "layer_maps_cache_path": str(layer_maps["cache_path"]),
            "layer_maps_loaded_from_cache": bool(layer_maps["loaded_from_cache"]),
            "delta_chw": delta,
            "conductance_chw": conductance,
        }
        row.update({f"delta_{key}": value for key, value in delta_stats.items()})
        row.update({f"conductance_{key}": value for key, value in conductance_stats.items()})
        rows.append(row)
    return pd.DataFrame(rows), skipped


def raw_value_limits(values, *, signed: bool) -> tuple[float | None, float | None]:
    arr = np.asarray(values, dtype="float64")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    if signed:
        vmax = float(np.max(np.abs(finite)))
        return -vmax, vmax
    return float(np.min(finite)), float(np.max(finite))


def plot_raw_example_maps(row, *, clean_img=None, patched_img=None):
    import matplotlib.pyplot as plt

    delta = _as_chw(row["delta_chw"])
    conductance = _as_chw(row["conductance_chw"])
    panels = [
        ("delta signed mean", reduce_raw_map(delta, mode="signed_mean"), "coolwarm", True),
        ("delta L2 RMS", reduce_raw_map(delta, mode="l2"), "magma", False),
        ("conductance signed mean", reduce_raw_map(conductance, mode="signed_mean"), "coolwarm", True),
        ("conductance mean |value|", reduce_raw_map(conductance, mode="mean_abs"), "viridis", False),
    ]
    image_cols = int(clean_img is not None) + int(patched_img is not None)
    ncols = image_cols + len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 4.2), squeeze=False, constrained_layout=True)
    col = 0
    if clean_img is not None:
        axes[0, col].imshow(clean_img)
        axes[0, col].set_title("clean")
        axes[0, col].axis("off")
        col += 1
    if patched_img is not None:
        axes[0, col].imshow(patched_img)
        axes[0, col].set_title("patched")
        axes[0, col].axis("off")
        col += 1
    for title, values, cmap, signed in panels:
        vmin, vmax = raw_value_limits(values, signed=signed)
        im = axes[0, col].imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"{title}\nraw range [{np.nanmin(values):.3g}, {np.nanmax(values):.3g}]")
        axes[0, col].axis("off")
        fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
        col += 1
    fig.suptitle(
        f"{'success' if bool(row['success']) else 'fail'} | "
        f"drop={float(row['drop']):.3f}, clean={float(row['conf_clean']):.3f}, patch={float(row['conf_patch']):.3f}",
        fontsize=12,
    )
    return fig


def plot_raw_value_distributions(rows_df: pd.DataFrame, *, max_examples: int | None = None):
    import matplotlib.pyplot as plt

    df = rows_df if max_examples is None else rows_df.head(int(max_examples))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    specs = [
        ("delta_chw", "delta raw values", axes[0, 0]),
        ("conductance_chw", "conductance raw values", axes[0, 1]),
        ("delta_mean_abs", "per-image mean |delta|", axes[1, 0]),
        ("conductance_mean_abs", "per-image mean |conductance|", axes[1, 1]),
    ]
    for key, title, ax in specs:
        for success, label, color in [(False, "fail", "#F58518"), (True, "success", "#4C78A8")]:
            sub = df[df["success"].astype(bool) == success]
            if sub.empty:
                continue
            if key.endswith("_chw"):
                values = np.concatenate([np.asarray(v, dtype="float32").reshape(-1) for v in sub[key].to_list()])
                if values.size > 200_000:
                    idx = np.linspace(0, values.size - 1, 200_000).astype(int)
                    values = values[idx]
            else:
                values = sub[key].to_numpy(dtype="float64")
            ax.hist(values[np.isfinite(values)], bins=80, alpha=0.48, density=True, label=label, color=color)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
    return fig


def plot_raw_average_maps(rows_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for row_idx, (success, label) in enumerate([(False, "fail"), (True, "success")]):
        sub = rows_df[rows_df["success"].astype(bool) == success]
        if sub.empty:
            for ax in axes[row_idx]:
                ax.axis("off")
            continue
        delta_maps = np.stack([reduce_raw_map(v, mode="signed_mean") for v in sub["delta_chw"]], axis=0)
        delta_abs_maps = np.stack([reduce_raw_map(v, mode="l2") for v in sub["delta_chw"]], axis=0)
        cond_maps = np.stack([reduce_raw_map(v, mode="signed_mean") for v in sub["conductance_chw"]], axis=0)
        cond_abs_maps = np.stack([reduce_raw_map(v, mode="mean_abs") for v in sub["conductance_chw"]], axis=0)
        panels = [
            (f"{label}: mean delta signed", delta_maps.mean(axis=0), "coolwarm", True),
            (f"{label}: std delta signed", delta_maps.std(axis=0), "magma", False),
            (f"{label}: mean conductance signed", cond_maps.mean(axis=0), "coolwarm", True),
            (f"{label}: std conductance signed", cond_maps.std(axis=0), "viridis", False),
        ]
        for col, (title, values, cmap, signed) in enumerate(panels):
            vmin, vmax = raw_value_limits(values, signed=signed)
            im = axes[row_idx, col].imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row_idx, col].set_title(f"{title}\nraw range [{np.nanmin(values):.3g}, {np.nanmax(values):.3g}]")
            axes[row_idx, col].axis("off")
            fig.colorbar(im, ax=axes[row_idx, col], fraction=0.046, pad=0.04)
    return fig


def top_neuron_sets(delta_chw, conductance_chw, *, fractions=(0.01, 0.10), top_ns=(200,)) -> dict[str, dict[str, Any]]:
    delta = _as_chw(delta_chw)
    conductance = _as_chw(conductance_chw)
    n = min(delta.size, conductance.size)
    d = delta.reshape(-1)[:n]
    c = conductance.reshape(-1)[:n]

    def top_indices(values, k: int) -> np.ndarray:
        k = max(1, min(int(k), values.size))
        idx = np.argpartition(-np.abs(values), kth=k - 1)[:k]
        return idx[np.argsort(-np.abs(values[idx]))]

    out: dict[str, dict[str, Any]] = {}
    for top_n in top_ns:
        k = max(1, min(int(top_n), n))
        out[f"top{int(top_n)}"] = {
            "k": k,
            "by_conductance": top_indices(c, k),
            "by_delta": top_indices(d, k),
        }
    for frac in fractions:
        k = max(1, min(int(round(float(frac) * n)), n))
        label = f"top{int(round(float(frac) * 100))}pct"
        out[label] = {
            "k": k,
            "by_conductance": top_indices(c, k),
            "by_delta": top_indices(d, k),
        }
    return out


def top_set_value_table(delta_chw, conductance_chw, indices: np.ndarray, *, source: str, max_rows: int = 200) -> pd.DataFrame:
    delta = _as_chw(delta_chw)
    conductance = _as_chw(conductance_chw)
    c, h, w = delta.shape
    flat_delta = delta.reshape(-1)
    flat_conductance = conductance.reshape(-1)
    rows = []
    for rank, flat_idx in enumerate(np.asarray(indices[: int(max_rows)], dtype=int), start=1):
        ch = int(flat_idx // (h * w))
        rem = int(flat_idx % (h * w))
        y = int(rem // w)
        x = int(rem % w)
        rows.append(
            {
                "source": source,
                "rank": rank,
                "flat_idx": int(flat_idx),
                "channel": ch,
                "y": y,
                "x": x,
                "delta": float(flat_delta[flat_idx]),
                "abs_delta": float(abs(flat_delta[flat_idx])),
                "conductance": float(flat_conductance[flat_idx]),
                "abs_conductance": float(abs(flat_conductance[flat_idx])),
            }
        )
    return pd.DataFrame(rows)


def top_set_spatial_mask(chw, indices: np.ndarray) -> np.ndarray:
    arr = _as_chw(chw)
    _c, h, w = arr.shape
    mask = np.zeros((h, w), dtype="float32")
    for flat_idx in np.asarray(indices, dtype=int):
        rem = int(flat_idx % (h * w))
        y = int(rem // w)
        x = int(rem % w)
        mask[y, x] += 1.0
    return mask


def plot_top_conductance_delta_cross(row, *, subset_key: str = "top200", clean_img=None, patched_img=None):
    import matplotlib.pyplot as plt

    delta = _as_chw(row["delta_chw"])
    conductance = _as_chw(row["conductance_chw"])
    sets = top_neuron_sets(delta, conductance)
    if subset_key not in sets:
        raise KeyError(f"Unknown subset_key={subset_key!r}; available={list(sets)}")
    entry = sets[subset_key]
    idx_c = entry["by_conductance"]
    idx_d = entry["by_delta"]
    c_set = set(int(v) for v in idx_c)
    d_set = set(int(v) for v in idx_d)
    overlap = len(c_set & d_set)
    k = int(entry["k"])

    conductance_hw = reduce_raw_map(conductance, mode="mean_abs")
    delta_hw = reduce_raw_map(delta, mode="l2")
    mask_c = top_set_spatial_mask(delta, idx_c)
    mask_d = top_set_spatial_mask(delta, idx_d)

    image_cols = int(clean_img is not None) + int(patched_img is not None)
    fig, axes = plt.subplots(2, 4 + image_cols, figsize=(4.0 * (4 + image_cols), 8.2), squeeze=False, constrained_layout=True)

    for row_idx in range(2):
        col = 0
        if clean_img is not None:
            axes[row_idx, col].imshow(clean_img)
            axes[row_idx, col].set_title("clean")
            axes[row_idx, col].axis("off")
            col += 1
        if patched_img is not None:
            axes[row_idx, col].imshow(patched_img)
            axes[row_idx, col].set_title("patched")
            axes[row_idx, col].axis("off")
            col += 1
        if row_idx == 0:
            panels = [
                (f"|conductance| spatial\nsource for {subset_key}", conductance_hw, "viridis", False),
                (f"pixels hit by top {k}\nconductance neurons", mask_c, "magma", False),
                ("delta L2 spatial", delta_hw, "magma", False),
                (f"delta values on top conductance\nfirst 200 ranks", np.abs(delta.reshape(-1)[idx_c[:200]]), None, False),
            ]
        else:
            panels = [
                (f"|delta| spatial\nsource for {subset_key}", delta_hw, "magma", False),
                (f"pixels hit by top {k}\ndelta neurons", mask_d, "magma", False),
                ("|conductance| spatial", conductance_hw, "viridis", False),
                (f"conductance values on top delta\nfirst 200 ranks", np.abs(conductance.reshape(-1)[idx_d[:200]]), None, False),
            ]
        for title, values, cmap, signed in panels:
            if values.ndim == 1:
                axes[row_idx, col].plot(np.arange(1, values.size + 1), values)
                axes[row_idx, col].set_title(title)
                axes[row_idx, col].set_xlabel("rank")
                axes[row_idx, col].grid(alpha=0.25)
            else:
                vmin, vmax = raw_value_limits(values, signed=signed)
                im = axes[row_idx, col].imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
                axes[row_idx, col].set_title(title)
                axes[row_idx, col].axis("off")
                fig.colorbar(im, ax=axes[row_idx, col], fraction=0.046, pad=0.04)
            col += 1
    fig.suptitle(
        f"{subset_key}: overlap={overlap}/{k} ({overlap / max(1, k):.3f}) | "
        f"drop={float(row['drop']):.3f}",
        fontsize=13,
    )
    return fig
