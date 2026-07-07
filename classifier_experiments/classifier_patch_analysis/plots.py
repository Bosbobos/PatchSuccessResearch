from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _bar_label_fontsize(n_rows: int) -> float:
    return max(6.0, min(9.0, 180.0 / max(1, int(n_rows))))


def _quality_label(row, score_col: str) -> str:
    score = row.get(score_col, np.nan)
    if not np.isfinite(float(score)):
        return ""
    recall = row.get("balanced_recall", np.nan)
    specificity = row.get("balanced_specificity", np.nan)
    if np.isfinite(float(recall)) and np.isfinite(float(specificity)):
        return f"{float(score):.3f} ({float(recall):.3f}/{float(specificity):.3f})"
    return f"{float(score):.3f}"


def _annotate_barh(ax, y, values, labels, *, fontsize: float):
    for yi, value, label in zip(y, values, labels):
        if np.isfinite(float(value)) and label:
            ax.text(float(value) + 0.012, yi, label, va="center", ha="left", fontsize=fontsize, clip_on=False)


def savefig(fig, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def plot_attack_distributions(cache):
    import matplotlib.pyplot as plt

    rows = pd.DataFrame([item.to_dict() for item in cache.examples])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, col, title in zip(
        axes,
        ["conf_clean", "conf_patch", "drop"],
        ["clean confidence", "patched confidence", "drop = clean - patched"],
        strict=True,
    ):
        for success, group in rows.groupby("success"):
            label = "success" if bool(success) else "fail"
            ax.hist(group[col], bins=40, alpha=0.55, label=f"{label} n={len(group)}")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
    return fig


def plot_label_counts(cache):
    import matplotlib.pyplot as plt

    counts = pd.Series([item.success for item in cache.examples]).map({True: "success", False: "fail"}).value_counts()
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.bar(counts.index, counts.values, color=["#4C78A8", "#F58518"][: len(counts)])
    ax.set_title("success/fail counts")
    ax.set_ylabel("images")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_quality_leaderboard(quality: pd.DataFrame, *, top_n: int = 20):
    import matplotlib.pyplot as plt

    df = quality.head(int(top_n)).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(df))), constrained_layout=True)
    score_col = "best_balanced_accuracy" if "best_balanced_accuracy" in df.columns else "best_f1"
    score_label = "best balanced accuracy" if score_col == "best_balanced_accuracy" else "best F1"
    y = np.arange(len(df))
    values = df[score_col].to_numpy(dtype="float64")
    ax.barh(y, values, color="#4C78A8")
    ax.set_yticks(y, df["metric"].astype(str).tolist())
    labels = [_quality_label(row, score_col) for _, row in df.iterrows()]
    _annotate_barh(ax, y, values, labels, fontsize=_bar_label_fontsize(len(df)))
    ax.set_xlabel(score_label)
    ax.set_title("success/fail metric leaderboard")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    return fig


def plot_regression_leaderboard(regression: pd.DataFrame, *, top_n: int = 20):
    import matplotlib.pyplot as plt

    df = regression.head(int(top_n)).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(df))), constrained_layout=True)
    colors = np.where(df["spearman"].to_numpy(dtype="float64") >= 0.0, "#4C78A8", "#F58518")
    y = np.arange(len(df))
    values = df["main_score"].to_numpy(dtype="float64")
    ax.barh(y, values, color=colors)
    ax.set_yticks(y, df["metric"].astype(str).tolist())
    labels = [f"{value:.3f}" if np.isfinite(value) else "" for value in values]
    _annotate_barh(ax, y, values, labels, fontsize=_bar_label_fontsize(len(df)))
    ax.set_xlabel("|Spearman| (main)")
    ax.set_title("drop correlation leaderboard by |Spearman|")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    return fig


def plot_layer_spread_map(summary_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    df = summary_df.copy()
    if df.empty:
        raise ValueError("summary_df is empty")
    pivot = df.pivot_table(index="layer", columns="success", values="mean_outside_patch_energy_frac", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(pivot))), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype="float64"), aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xticks(range(len(pivot.columns)), ["fail" if not bool(v) else "success" for v in pivot.columns])
    ax.set_title("patch spread outside source ROI by layer")
    fig.colorbar(im, ax=ax, label="mean outside patch energy fraction")
    return fig


def plot_profile_summary(summary_df: pd.DataFrame, *, layer: str | None = None):
    import matplotlib.pyplot as plt

    df = summary_df.copy()
    if layer is not None:
        df = df[df["layer"] == layer]
    profile_order = ["jpeg_zigzag", "center_spiral", "centered_square_rings"]
    profiles = [item for item in profile_order if item in set(df["profile"])]
    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4), constrained_layout=True, squeeze=False)
    xlabels = {
        "jpeg_zigzag": "normalized JPEG zig-zag cell index",
        "center_spiral": "center square spiral step",
        "centered_square_rings": "square ring index (1 = center)",
    }
    titles = {
        "jpeg_zigzag": "JPEG zig-zag delta profile",
        "center_spiral": "center spiral delta profile",
        "centered_square_rings": "centered square rings",
    }
    for ax, profile in zip(axes[0], profiles, strict=True):
        sub = df[df["profile"] == profile]
        for success, group in sub.groupby("success"):
            label = "success" if bool(success) else "fail"
            x = group["x"].to_numpy(dtype="float64")
            mean = group["mean"].to_numpy(dtype="float64")
            std = group["std"].fillna(0.0).to_numpy(dtype="float64")
            ax.plot(x, mean, label=label)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_title(titles.get(profile, profile))
        ax.set_xlabel(xlabels.get(profile, "profile step"))
        ax.set_ylabel("mean abs delta")
        ax.grid(alpha=0.25)
        ax.legend()
    return fig


def plot_zigzag_cumulative_summary(profiles_df: pd.DataFrame, *, layer: str | None = None, with_std: bool = True):
    import matplotlib.pyplot as plt

    df = profiles_df[profiles_df["profile"] == "jpeg_zigzag"].copy()
    if layer is not None:
        df = df[df["layer"] == layer]
    if df.empty:
        raise ValueError("No jpeg_zigzag rows are available.")

    records = []
    for (path, success), group in df.groupby(["path", "success"], sort=False):
        values = group.sort_values("step")["value"].to_numpy(dtype="float64")
        mass = np.abs(values)
        total = float(mass.sum() + 1e-12)
        cumulative = np.cumsum(mass) / total
        x = np.linspace(1.0 / len(cumulative), 1.0, len(cumulative), dtype="float64")
        for idx, value in enumerate(cumulative):
            records.append({"path": path, "success": bool(success), "step": idx, "x": x[idx], "value": float(value)})
    cdf = pd.DataFrame(records)
    summary = cdf.groupby(["success", "step", "x"], as_index=False)["value"].agg(["mean", "std", "count"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for success, group in summary.groupby("success"):
        label = "success" if bool(success) else "fail"
        x = group["x"].to_numpy(dtype="float64")
        mean = group["mean"].to_numpy(dtype="float64")
        std = group["std"].fillna(0.0).to_numpy(dtype="float64")
        ax.plot(x, mean, label=label)
        if with_std:
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_title("normalized cumulative spread, JPEG zig-zag")
    ax.set_xlabel("normalized JPEG zig-zag cell index")
    ax.set_ylabel("cumulative |delta| mass")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


def plot_all_layer_zigzag_single_example(exp, example, *, layers: list[str] | None = None, build_missing: bool = True):
    import matplotlib.pyplot as plt

    from .activations import reduce_chw_to_hw
    from .patch_spread import _compute_or_load_delta_map, _profile_from_hw, jpeg_zigzag_indices

    layers = list(layers or exp.all_display_layer_names())
    rows = []
    for layer in layers:
        maps = _compute_or_load_delta_map(exp, example, layer_name=layer, build_missing=build_missing)
        if maps is None:
            continue
        hw = reduce_chw_to_hw(maps["delta_chw"], mode="l2").detach().cpu().numpy()
        profile = _profile_from_hw(hw, jpeg_zigzag_indices(*hw.shape))
        mass = np.abs(profile)
        total = float(mass.sum() + 1e-12)
        cumulative = np.cumsum(mass) / total
        auc = float(np.trapz(cumulative, np.linspace(1.0 / len(cumulative), 1.0, len(cumulative))))
        rows.append(
            {
                "layer": layer,
                "x": np.linspace(1.0 / len(profile), 1.0, len(profile), dtype="float64"),
                "profile": profile,
                "auc": auc,
            }
        )
    if not rows:
        raise RuntimeError("No layer delta profiles are available for the selected example.")

    fig, axes = plt.subplots(len(rows), 1, figsize=(12, max(8, 1.7 * len(rows))), sharex=False, constrained_layout=True)
    if len(rows) == 1:
        axes = [axes]
    for ax, row in zip(axes, rows, strict=True):
        ax.plot(row["x"], row["profile"], color="#4C78A8", linewidth=1.0)
        ax.set_title(f"{row['layer']} | mean abs zig-zag profile | cumulative AUC={row['auc']:.3f}", loc="left", fontsize=10)
        ax.set_ylabel("mean abs delta")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("normalized JPEG zig-zag cell index")
    fig.suptitle(f"All-layer patch spread for one success example: {Path(example.path).name}", y=1.002)
    return fig


def plot_all_layer_delta_maps_single_example(
    exp,
    example,
    *,
    layers: list[str] | None = None,
    build_missing: bool = True,
    title: str | None = None,
    fractions: tuple[float, ...] = (1.0, 10.0, 100.0),
):
    import matplotlib.pyplot as plt

    from .activations import reduce_chw_to_hw

    layers = list(layers or exp.all_display_layer_names())
    clean_img, _patched_img, _bbox = exp._images_for_example(example)
    image_arr = np.asarray(clean_img.convert("RGB"))
    maps_by_layer = []
    for layer in layers:
        if build_missing:
            maps = exp.compute_layer_map(example, layer_name=layer, force=False)
        else:
            maps = exp._load_layer_map_cache(example, layer_name=layer)
        if maps is None:
            continue
        delta = np.asarray(maps["delta_chw"], dtype="float32")
        importance = np.asarray(maps["importance_chw"], dtype="float32")
        channel_scores = np.abs(importance).sum(axis=(1, 2))
        order = np.argsort(-channel_scores, kind="stable")
        maps_by_layer.append((layer, delta, order))
    if not maps_by_layer:
        raise RuntimeError("No layer delta maps are available for the selected example.")

    n_maps = len(maps_by_layer)
    n_cols = n_maps
    n_rows = 1 + len(fractions)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.25 * n_cols, 2.45 * n_rows),
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={"height_ratios": [1.15, *([1.0] * len(fractions))]},
    )
    for ax in axes.reshape(-1):
        ax.axis("off")

    axes[0, 0].imshow(image_arr)
    axes[0, 0].set_title("clean image")
    axes[0, 0].axis("off")
    info_ax = axes[0, 1] if n_cols > 1 else axes[0, 0]
    info_ax.text(
        0.0,
        0.5,
        f"drop={example.drop:.3f}\nclean={example.conf_clean:.3f}\npatched={example.conf_patch:.3f}\npath={Path(example.path).name}",
        va="center",
        ha="left",
        fontsize=10,
    )
    info_ax.axis("off")

    for col_idx, (layer, delta, order) in enumerate(maps_by_layer):
        axes[0, col_idx].set_xlabel(layer)
        for row_idx, fraction in enumerate(fractions, start=1):
            if float(fraction) >= 100.0:
                keep = order
            else:
                k = max(1, int(round(float(fraction) / 100.0 * delta.shape[0])))
                keep = order[: min(k, delta.shape[0])]
            filtered = np.zeros_like(delta)
            filtered[keep] = delta[keep]
            hw = reduce_chw_to_hw(filtered, mode="mean_abs").detach().cpu().numpy()
            ax = axes[row_idx, col_idx]
            ax.imshow(_normalize_map(hw), cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0)
            if row_idx == 1:
                ax.set_title(layer, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"top {fraction:g}%\nfilters", rotation=0, labelpad=28, va="center")
            ax.text(
                0.02,
                0.98,
                f"ch={len(keep)}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1.5, "edgecolor": "none"},
            )
            ax.axis("off")
    fig.suptitle(title or f"All-layer top-filter mean abs delta maps: {Path(example.path).name}")
    return fig


def plot_metric_vs_drop(rows_df: pd.DataFrame, metric: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    colors = np.where(rows_df["success"].to_numpy(dtype=bool), "#4C78A8", "#F58518")
    ax.scatter(rows_df[metric], rows_df["drop"], c=colors, alpha=0.7, s=18)
    ax.set_xlabel(metric)
    ax.set_ylabel("drop")
    ax.set_title(f"{metric} vs drop")
    ax.grid(alpha=0.25)
    return fig


def plot_position_heatmap(rows_df: pd.DataFrame, *, value_col: str = "ASR"):
    import matplotlib.pyplot as plt

    pivot = rows_df.pivot(index="y", columns="x", values=value_col).sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(
        pivot.to_numpy(dtype="float64"),
        origin="upper",
        cmap="viridis",
        extent=[
            float(pivot.columns.min()),
            float(pivot.columns.max()),
            float(pivot.index.max()),
            float(pivot.index.min()),
        ],
        aspect="equal",
    )
    ax.set_title(f"patch position vs {value_col}")
    ax.set_xlabel("patch x")
    ax.set_ylabel("patch y")
    fig.colorbar(im, ax=ax, label=value_col)
    return fig


def _normalize_map(values):
    arr = np.asarray(values, dtype="float64")
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype="float32")
    lo = float(np.nanpercentile(arr[finite], 1.0))
    hi = float(np.nanpercentile(arr[finite], 99.0))
    if hi <= lo:
        hi = float(np.nanmax(arr[finite]))
        lo = float(np.nanmin(arr[finite]))
    if hi <= lo:
        return np.zeros_like(arr, dtype="float32")
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype("float32", copy=False)


def plot_importance_example_grid(
    exp,
    examples,
    *,
    layer_name: str = "model.9",
    title: str = "importance examples",
    max_examples: int = 20,
):
    import matplotlib.pyplot as plt
    from PIL import Image

    from .activations import reduce_chw_to_hw

    selected = list(examples)[: int(max_examples)]
    if not selected:
        raise ValueError("No examples were provided.")
    fig, axes = plt.subplots(len(selected), 3, figsize=(10.5, max(3.0, 2.6 * len(selected))), constrained_layout=True)
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_idx, example in enumerate(selected):
        maps = exp.compute_layer_map(example, layer_name=layer_name, force=False)
        clean_img, _patched_img, _bbox = exp._images_for_example(example)
        image_arr = np.asarray(clean_img.convert("RGB"))
        importance_hw = reduce_chw_to_hw(maps["importance_chw"], mode="l2").detach().cpu().numpy()
        importance_img = Image.fromarray((_normalize_map(importance_hw) * 255).astype("uint8")).resize(
            clean_img.size,
            resample=Image.Resampling.NEAREST,
        )
        importance_arr = np.asarray(importance_img, dtype="float32") / 255.0

        ax_img, ax_map, ax_overlay = axes[row_idx]
        ax_img.imshow(image_arr)
        ax_img.set_ylabel(f"{row_idx + 1}", rotation=0, labelpad=12, va="center")
        ax_img.set_title("image" if row_idx == 0 else "")
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        ax_map.imshow(importance_arr, cmap="magma", vmin=0.0, vmax=1.0)
        ax_map.set_title("clean importance" if row_idx == 0 else "")
        ax_map.set_xticks([])
        ax_map.set_yticks([])

        ax_overlay.imshow(image_arr)
        ax_overlay.imshow(importance_arr, cmap="magma", alpha=0.45, vmin=0.0, vmax=1.0)
        ax_overlay.set_title("overlay" if row_idx == 0 else "")
        ax_overlay.set_xticks([])
        ax_overlay.set_yticks([])
        ax_overlay.text(
            0.01,
            0.99,
            f"drop={example.drop:.3f}\nclean={example.conf_clean:.3f}\npatch={example.conf_patch:.3f}",
            transform=ax_overlay.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
        )
    fig.suptitle(title)
    return fig


def plot_channel_filtered_delta_examples(
    exp,
    examples,
    *,
    layer_name: str = "model.9",
    fractions: tuple[float, ...] = (1.0, 10.0, 100.0),
    max_examples: int = 10,
    title: str = "channel-filtered delta maps",
):
    import matplotlib.pyplot as plt
    from PIL import Image

    from .activations import reduce_chw_to_hw

    selected = list(examples)[: int(max_examples)]
    if not selected:
        raise ValueError("No examples were provided.")
    n_cols = 1 + len(fractions)
    fig, axes = plt.subplots(
        len(selected),
        n_cols,
        figsize=(3.1 * n_cols, max(3.0, 2.7 * len(selected))),
        constrained_layout=True,
        squeeze=False,
    )
    for row_idx, example in enumerate(selected):
        maps = exp.compute_layer_map(example, layer_name=layer_name, force=False)
        clean_img, _patched_img, _bbox = exp._images_for_example(example)
        image_arr = np.asarray(clean_img.convert("RGB"))
        delta = np.asarray(maps["delta_chw"], dtype="float32")
        importance = np.asarray(maps["importance_chw"], dtype="float32")
        channel_scores = np.abs(importance).sum(axis=(1, 2))
        order = np.argsort(-channel_scores, kind="stable")

        axes[row_idx, 0].imshow(image_arr)
        axes[row_idx, 0].set_ylabel(f"{row_idx + 1}", rotation=0, labelpad=12, va="center")
        axes[row_idx, 0].set_title("image" if row_idx == 0 else "")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        for col_idx, fraction in enumerate(fractions, start=1):
            if float(fraction) >= 100.0:
                keep = order
            else:
                k = max(1, int(round(float(fraction) / 100.0 * delta.shape[0])))
                keep = order[: min(k, delta.shape[0])]
            filtered = np.zeros_like(delta)
            filtered[keep] = delta[keep]
            hw = reduce_chw_to_hw(filtered, mode="mean_abs").detach().cpu().numpy()
            hw_norm = _normalize_map(hw)
            axes[row_idx, col_idx].imshow(hw_norm, cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0)
            axes[row_idx, col_idx].set_title(f"top {fraction:g}% filters" if row_idx == 0 else "")
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            axes[row_idx, col_idx].text(
                0.01,
                0.99,
                f"drop={example.drop:.3f}\nch={len(keep)}",
                transform=axes[row_idx, col_idx].transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
            )
    fig.suptitle(title)
    return fig
