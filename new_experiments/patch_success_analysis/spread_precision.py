from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _ensure_plot_cache_env() -> None:
    cache_root = Path(tempfile.gettempdir()) / "patch_success_plot_cache"
    mpl = cache_root / "matplotlib"
    xdg = cache_root / "xdg"
    mpl.mkdir(parents=True, exist_ok=True)
    xdg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg))


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


def _cache_path(exp, examples, *, layer_name: str, top_percent: float):
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [item.path for item in examples],
        "target_layer": layer_name,
        "detect_layer": exp.config.detect_layer,
        "target_mode": exp.config.target_mode,
        "top_percent": float(top_percent),
        "method_version": 2,
    }
    return exp.derived_cache_dir / f"spread_vs_precision_{_cache_key(payload)}.pkl"


def _as_chw(values) -> np.ndarray:
    arr = np.asarray(values, dtype="float32")
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={arr.shape}")
    return arr


def object_bbox_delta_metrics(delta_chw, object_bbox_xyxy, *, imgsz: int = 640) -> dict[str, float]:
    from .activations import patch_mask_on_feature_grid, reduce_chw_to_hw

    delta = _as_chw(delta_chw)
    hw_l2 = reduce_chw_to_hw(delta, mode="l2").detach().cpu().numpy().astype("float64", copy=False)
    hw_signed = reduce_chw_to_hw(delta, mode="signed_mean").detach().cpu().numpy().astype("float64", copy=False)
    total = float(np.sum(hw_l2) + 1e-12)
    mask = patch_mask_on_feature_grid(object_bbox_xyxy, grid_hw=hw_l2.shape, imgsz=int(imgsz))
    if not mask.any():
        return {
            "object_bbox_delta_frac": 0.0,
            "object_bbox_delta_sum": 0.0,
            "object_bbox_delta_mean": 0.0,
            "object_bbox_delta_max": 0.0,
            "object_bbox_delta_signed_mean": 0.0,
            "object_bbox_delta_signed_sum": 0.0,
            "object_bbox_feature_pixels": 0.0,
        }
    vals = hw_l2[mask]
    signed_vals = hw_signed[mask]
    return {
        "object_bbox_delta_frac": float(vals.sum() / total),
        "object_bbox_delta_sum": float(vals.sum()),
        "object_bbox_delta_mean": float(vals.mean()),
        "object_bbox_delta_max": float(vals.max()),
        "object_bbox_delta_signed_mean": float(signed_vals.mean()),
        "object_bbox_delta_signed_sum": float(signed_vals.sum()),
        "object_bbox_feature_pixels": float(mask.sum()),
    }


def patch_excluded_spread_metrics(delta_chw, importance_chw, *, object_bbox_xyxy=None, patch_bbox_xyxy=None, imgsz: int = 640) -> dict[str, float]:
    from .activations import patch_mask_on_feature_grid, reduce_chw_to_hw

    delta = _as_chw(delta_chw)
    importance = _as_chw(importance_chw)
    hw_l2 = reduce_chw_to_hw(delta, mode="l2").detach().cpu().numpy().astype("float64", copy=False)
    hw_signed = reduce_chw_to_hw(delta, mode="signed_mean").detach().cpu().numpy().astype("float64", copy=False)
    grid_hw = hw_l2.shape
    patch_mask = patch_mask_on_feature_grid(patch_bbox_xyxy, grid_hw=grid_hw, imgsz=int(imgsz))
    valid_mask = ~patch_mask
    valid_values = hw_l2[valid_mask]
    denom = float(valid_values.sum() + 1e-12)
    if valid_values.size == 0:
        valid_values = np.asarray([0.0], dtype="float64")
    top_k = max(1, int(round(0.05 * valid_values.size)))
    top_idx = np.argpartition(-valid_values.reshape(-1), kth=min(top_k - 1, valid_values.size - 1))[:top_k]

    object_mask = patch_mask_on_feature_grid(object_bbox_xyxy, grid_hw=grid_hw, imgsz=int(imgsz))
    object_valid_mask = object_mask & valid_mask
    object_values = hw_l2[object_valid_mask]
    object_signed = hw_signed[object_valid_mask]

    out = {
        "delta_l2_rms_no_patch": float(np.sqrt(np.mean(valid_values * valid_values))),
        "delta_abs_mean_no_patch": float(np.mean(np.abs(valid_values))),
        "delta_max_no_patch": float(np.max(valid_values)),
        "delta_gini_no_patch": _gini_np(valid_values),
        "top5pct_energy_frac_no_patch": float(valid_values.reshape(-1)[top_idx].sum() / denom),
        "patch_roi_energy_sum": float(hw_l2[patch_mask].sum()) if patch_mask.any() else 0.0,
        "patch_roi_energy_frac_total": float(hw_l2[patch_mask].sum() / (hw_l2.sum() + 1e-12)) if patch_mask.any() else 0.0,
        "object_bbox_delta_frac_no_patch": float(object_values.sum() / denom) if object_values.size else 0.0,
        "object_bbox_delta_sum_no_patch": float(object_values.sum()) if object_values.size else 0.0,
        "object_bbox_delta_mean_no_patch": float(object_values.mean()) if object_values.size else 0.0,
        "object_bbox_delta_max_no_patch": float(object_values.max()) if object_values.size else 0.0,
        "object_bbox_delta_signed_mean_no_patch": float(object_signed.mean()) if object_signed.size else 0.0,
        "object_bbox_feature_pixels_no_patch": float(object_valid_mask.sum()),
    }

    flat_delta = delta.reshape(delta.shape[0], -1)
    flat_importance = importance.reshape(importance.shape[0], -1)
    patch_flat = patch_mask.reshape(-1)
    valid_flat = ~patch_flat
    out["delta_total_abs_no_patch"] = float(np.abs(flat_delta[:, valid_flat]).sum())
    out["delta_total_signed_no_patch"] = float(flat_delta[:, valid_flat].sum())
    out["importance_total_abs_no_patch"] = float(np.abs(flat_importance[:, valid_flat]).sum())
    return out


def _gini_np(values) -> float:
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


def _clean_bbox(example) -> tuple[float, float, float, float] | None:
    detection = getattr(example, "clean_detection", None)
    if not detection:
        return None
    bbox = detection.get("bbox_xyxy_orig")
    if bbox is None or len(bbox) != 4:
        return None
    return tuple(float(v) for v in bbox)


def _prefix(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def compute_or_load_spread_vs_precision(
    exp,
    *,
    layer_name: str | None = None,
    max_examples: int | None = None,
    top_percent: float = 5.0,
    force: bool = False,
) -> dict[str, Any]:
    from .activations import delta_spread_metrics
    from .metrics import alignment_metrics, metric_quality_rows, segmentig_soft_alignment_metrics
    from .regression_metrics import regression_similarity_table

    cache = exp.get_cache()
    examples = list(cache.examples)
    if max_examples is not None:
        examples = examples[: int(max_examples)]
    layer_name = layer_name or exp.config.target_layer
    path = _cache_path(exp, examples, layer_name=layer_name, top_percent=float(top_percent))
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached

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
        importance = _as_chw(layer_maps["segmentig_chw"])
        d_flat = delta.reshape(-1)
        a_flat = importance.reshape(-1)

        spread = {}
        spread.update(delta_spread_metrics(delta, patch_bbox_xyxy=example.patch_bbox_lb, imgsz=int(exp.config.attack.imgsz)))
        spread.update(object_bbox_delta_metrics(delta, _clean_bbox(example), imgsz=int(exp.config.attack.imgsz)))
        spread["delta_total_abs"] = float(np.abs(delta).sum())
        spread["delta_total_signed"] = float(delta.sum())
        spread["delta_nonzero_frac"] = float(np.mean(np.abs(delta.reshape(-1)) > 0.0))

        precision = {}
        precision.update(alignment_metrics(d_flat, a_flat, top_percent=float(top_percent)))
        precision.update(segmentig_soft_alignment_metrics(d_flat, a_flat))

        row = {
            "path": example.path,
            "success": bool(example.success),
            "drop": float(example.drop),
            "conf_clean": float(example.conf_clean),
            "conf_patch": float(example.conf_patch),
            "layer_maps_cache_path": str(layer_maps["cache_path"]),
            "layer_maps_loaded_from_cache": bool(layer_maps["loaded_from_cache"]),
            **_prefix("spread", spread),
            **_prefix("precision", precision),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No spread-vs-precision rows; skipped={len(skipped)}")

    rows_df = pd.DataFrame(rows)
    spread_cols = [col for col in rows_df.columns if col.startswith("spread_")]
    precision_cols = [col for col in rows_df.columns if col.startswith("precision_")]
    labels = rows_df["success"].astype(bool).to_numpy()
    quality = pd.DataFrame(metric_quality_rows(labels, {col: rows_df[col].to_numpy() for col in spread_cols + precision_cols}))
    quality["family"] = np.where(quality["metric"].str.startswith("spread_"), "spread", "precision")
    quality = quality.sort_values(["best_balanced_accuracy", "roc_auc", "best_accuracy"], ascending=False).reset_index(drop=True)

    regression = regression_similarity_table(
        rows_df[["path", "success", "drop", *spread_cols, *precision_cols]],
        target_col="drop",
        metric_cols=spread_cols + precision_cols,
    )
    regression["family"] = np.where(regression["metric"].str.startswith("spread_"), "spread", "precision")
    regression = regression.sort_values(["main_score", "abs_spearman", "abs_pearson"], ascending=False).reset_index(drop=True)

    result = {
        "rows": rows,
        "spread_cols": spread_cols,
        "precision_cols": precision_cols,
        "quality": quality,
        "regression": regression,
        "skipped": skipped,
        "cache_path": str(path),
        "loaded_from_cache": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh)
    return result


def plot_family_quality_side_by_side(quality_df: pd.DataFrame, *, top_n: int = 15):
    _ensure_plot_cache_env()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, family, color in [(axes[0], "spread", "#4C78A8"), (axes[1], "precision", "#F58518")]:
        score_col = "best_balanced_accuracy" if "best_balanced_accuracy" in quality_df.columns else "best_accuracy"
        sub = quality_df[quality_df["family"] == family].head(int(top_n)).sort_values(score_col)
        y = np.arange(len(sub))
        values = sub[score_col].to_numpy(dtype="float64")
        ax.barh(y, values, color=color, alpha=0.85, label="best balanced accuracy")
        ax.set_yticks(y, sub["metric"].astype(str).tolist())
        value_labels = [_quality_label(row, score_col) for _, row in sub.iterrows()]
        _annotate_barh(ax, y, values, value_labels, fontsize=_bar_label_fontsize(len(sub)))
        ax.scatter(sub["roc_auc"], y, color="black", s=22, label="ROC-AUC", zorder=3)
        ax.set_title(f"{family}: success/fail classification")
        ax.set_xlabel("score")
        ax.set_xlim(0, 1.28)
        ax.grid(axis="x", alpha=0.25)
        ax.legend()
    return fig


def plot_family_quality_combined_bar(quality_df: pd.DataFrame, *, top_n: int = 25, score_col: str = "best_balanced_accuracy"):
    _ensure_plot_cache_env()
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if quality_df.empty:
        raise ValueError("quality_df is empty")
    score_col = str(score_col)
    if score_col not in quality_df.columns:
        raise KeyError(f"{score_col!r} was not found in quality_df")
    score_label = "best balanced accuracy" if score_col == "best_balanced_accuracy" else score_col
    family_colors = {"spread": "#4C78A8", "precision": "#F58518"}
    sort_cols = [score_col]
    ascending = [False]
    if score_col != "best_balanced_accuracy" and "best_balanced_accuracy" in quality_df.columns:
        sort_cols.append("best_balanced_accuracy")
        ascending.append(False)
    if score_col != "best_accuracy" and "best_accuracy" in quality_df.columns:
        sort_cols.append("best_accuracy")
        ascending.append(False)
    if score_col != "roc_auc" and "roc_auc" in quality_df.columns:
        sort_cols.append("roc_auc")
        ascending.append(False)
    sub = quality_df.sort_values(sort_cols, ascending=ascending).head(int(top_n)).copy()
    sub = sub.sort_values(score_col)
    labels = sub["metric"].astype(str) + " | " + sub["family"].astype(str)
    colors = [family_colors.get(str(family), "#777777") for family in sub["family"]]

    fig_h = max(5.0, 0.34 * len(sub) + 1.4)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    y = np.arange(len(sub))
    values = sub[score_col].to_numpy(dtype="float64")
    ax.barh(y, values, color=colors, alpha=0.88)
    if "roc_auc" in sub.columns and score_col != "roc_auc":
        ax.scatter(sub["roc_auc"].to_numpy(dtype="float64"), y, color="black", s=24, label="ROC-AUC", zorder=3)
    ax.set_yticks(y, labels.tolist())
    value_labels = [_quality_label(row, score_col) for _, row in sub.iterrows()]
    _annotate_barh(ax, y, values, value_labels, fontsize=_bar_label_fontsize(len(sub)))
    ax.set_xlabel(score_label)
    ax.set_title(f"Combined spread/precision classification metrics by {score_label}")
    ax.set_xlim(0, 1.28)
    ax.grid(axis="x", alpha=0.25)
    handles = [mpatches.Patch(color=color, label=family) for family, color in family_colors.items()]
    if "roc_auc" in sub.columns and score_col != "roc_auc":
        handles.append(ax.collections[0])
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    return fig


def plot_family_regression_side_by_side(regression_df: pd.DataFrame, *, top_n: int = 15):
    _ensure_plot_cache_env()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, family, color in [(axes[0], "spread", "#4C78A8"), (axes[1], "precision", "#F58518")]:
        sub = regression_df[regression_df["family"] == family].head(int(top_n)).sort_values("main_score")
        y = np.arange(len(sub))
        values = sub["main_score"].to_numpy(dtype="float64")
        ax.barh(y, values, color=color, alpha=0.85, label="|Spearman| (main)")
        ax.set_yticks(y, sub["metric"].astype(str).tolist())
        value_labels = [f"{value:.3f}" if np.isfinite(value) else "" for value in values]
        _annotate_barh(ax, y, values, value_labels, fontsize=_bar_label_fontsize(len(sub)))
        ax.scatter(sub["abs_spearman"], y, color="black", s=22, label="|Spearman|", zorder=3)
        ax.set_title(f"{family}: drop regression similarity")
        ax.set_xlabel("|Spearman|")
        ax.set_xlim(0, 1.12)
        ax.grid(axis="x", alpha=0.25)
        ax.legend()
    return fig


def plot_family_regression_combined_bar(regression_df: pd.DataFrame, *, top_n: int = 25, score_col: str = "main_score"):
    _ensure_plot_cache_env()
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if regression_df.empty:
        raise ValueError("regression_df is empty")
    score_col = str(score_col)
    if score_col not in regression_df.columns:
        raise KeyError(f"{score_col!r} was not found in regression_df")
    score_label = "|Spearman| (main)" if score_col == "main_score" else score_col
    family_colors = {"spread": "#4C78A8", "precision": "#F58518"}
    sort_cols = [score_col]
    ascending = [False]
    if score_col != "abs_spearman" and "abs_spearman" in regression_df.columns:
        sort_cols.append("abs_spearman")
        ascending.append(False)
    if score_col != "abs_pearson" and "abs_pearson" in regression_df.columns:
        sort_cols.append("abs_pearson")
        ascending.append(False)
    sub = regression_df.sort_values(sort_cols, ascending=ascending).head(int(top_n)).copy()
    sub = sub.sort_values(score_col)
    labels = sub["metric"].astype(str) + " | " + sub["family"].astype(str)
    colors = [family_colors.get(str(family), "#777777") for family in sub["family"]]

    fig_h = max(5.0, 0.34 * len(sub) + 1.4)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    y = np.arange(len(sub))
    values = sub[score_col].to_numpy(dtype="float64")
    ax.barh(y, values, color=colors, alpha=0.88)
    if "abs_spearman" in sub.columns and score_col != "abs_spearman":
        ax.scatter(sub["abs_spearman"].to_numpy(dtype="float64"), y, color="black", s=24, label="|Spearman|", zorder=3)
    ax.set_yticks(y, labels.tolist())
    value_labels = [f"{value:.3f}" if np.isfinite(value) else "" for value in values]
    _annotate_barh(ax, y, values, value_labels, fontsize=_bar_label_fontsize(len(sub)))
    ax.set_xlabel(score_label)
    ax.set_title(f"Combined spread/precision regression metrics by {score_label}")
    ax.set_xlim(0, 1.12)
    ax.grid(axis="x", alpha=0.25)
    handles = [mpatches.Patch(color=color, label=family) for family, color in family_colors.items()]
    if "abs_spearman" in sub.columns and score_col != "abs_spearman":
        handles.append(ax.collections[0])
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    return fig


def plot_spread_precision_scatter(rows_df: pd.DataFrame, *, spread_metric: str, precision_metric: str):
    _ensure_plot_cache_env()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    colors = np.where(rows_df["success"].astype(bool), "#4C78A8", "#F58518")
    axes[0].scatter(rows_df[spread_metric], rows_df[precision_metric], c=colors, alpha=0.55, s=18)
    axes[0].set_xlabel(spread_metric)
    axes[0].set_ylabel(precision_metric)
    axes[0].set_title("spread vs precision")
    axes[0].grid(alpha=0.25)

    sc = axes[1].scatter(rows_df[spread_metric], rows_df[precision_metric], c=rows_df["drop"], cmap="viridis", alpha=0.65, s=20)
    axes[1].set_xlabel(spread_metric)
    axes[1].set_ylabel(precision_metric)
    axes[1].set_title("colored by drop")
    axes[1].grid(alpha=0.25)
    fig.colorbar(sc, ax=axes[1], label="drop")
    return fig


def top_spread_precision_pairs(regression_df: pd.DataFrame, *, n: int = 3) -> pd.DataFrame:
    spread = regression_df[regression_df["family"] == "spread"].copy()
    precision = regression_df[regression_df["family"] == "precision"].copy()
    if spread.empty or precision.empty:
        return pd.DataFrame(columns=["spread_metric", "precision_metric", "pair_score"])
    spread = spread.sort_values(["abs_spearman", "abs_pearson"], ascending=False).head(int(n)).reset_index(drop=True)
    precision = precision.sort_values(["abs_spearman", "abs_pearson"], ascending=False).head(int(n)).reset_index(drop=True)
    pair_rows = []
    for idx in range(min(len(spread), len(precision), int(n))):
        pair_rows.append(
            {
                "spread_metric": spread.loc[idx, "metric"],
                "precision_metric": precision.loc[idx, "metric"],
                "spread_abs_spearman": float(spread.loc[idx, "abs_spearman"]),
                "spread_abs_pearson": float(spread.loc[idx, "abs_pearson"]),
                "precision_abs_spearman": float(precision.loc[idx, "abs_spearman"]),
                "precision_abs_pearson": float(precision.loc[idx, "abs_pearson"]),
                "pair_score": float(
                    np.nanmean(
                        [
                            spread.loc[idx, "abs_spearman"],
                            spread.loc[idx, "abs_pearson"],
                            precision.loc[idx, "abs_spearman"],
                            precision.loc[idx, "abs_pearson"],
                        ]
                    )
                ),
            }
        )
    return pd.DataFrame(pair_rows)


def plot_family_correlation_bars(regression_df: pd.DataFrame, *, corr_col: str, top_n: int = 15):
    _ensure_plot_cache_env()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, family, color in [(axes[0], "spread", "#4C78A8"), (axes[1], "precision", "#F58518")]:
        sub = regression_df[regression_df["family"] == family].sort_values(corr_col, ascending=False).head(int(top_n))
        sub = sub.sort_values(corr_col)
        ax.barh(sub["metric"], sub[corr_col], color=color, alpha=0.85)
        ax.set_title(f"{family}: {corr_col} with drop")
        ax.set_xlabel(corr_col)
        ax.grid(axis="x", alpha=0.25)
    return fig


def _top_channels_by_example_importance(importance_chw: np.ndarray, *, percent: float) -> np.ndarray:
    importance = np.asarray(importance_chw, dtype="float32")
    channel_scores = np.abs(importance).reshape(importance.shape[0], -1).sum(axis=1)
    if float(percent) >= 100:
        k = int(channel_scores.size)
    else:
        k = max(1, int(round(float(percent) / 100.0 * channel_scores.size)))
    order = np.argsort(-channel_scores, kind="stable")
    return order[:k]


def _raw_limits(values, *, signed: bool) -> tuple[float | None, float | None]:
    arr = np.asarray(values, dtype="float64")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    if signed:
        vmax = float(np.max(np.abs(finite)))
        return -vmax, vmax
    return float(np.min(finite)), float(np.max(finite))


def _score_increase(example) -> float:
    return float(example.conf_patch) - float(example.conf_clean)


def plot_top_confident_channel_filtered_maps(
    exp,
    examples,
    *,
    layer_name: str = "model.22",
    top_n_per_group: int = 10,
    fractions=(1, 10, 100),
    include_success: bool = True,
    include_fail: bool = True,
    fail_max_score_increase: float | None = None,
    fail_min_score_increase: float | None = None,
    title: str | None = None,
):
    _ensure_plot_cache_env()
    import matplotlib.pyplot as plt

    selected_success = (
        sorted([item for item in examples if item.success], key=lambda item: float(item.drop), reverse=True)[: int(top_n_per_group)]
        if include_success
        else []
    )
    fail_candidates = [item for item in examples if not item.success]
    if fail_max_score_increase is not None:
        fail_candidates = [item for item in fail_candidates if _score_increase(item) <= float(fail_max_score_increase)]
    if fail_min_score_increase is not None:
        fail_candidates = [item for item in fail_candidates if _score_increase(item) > float(fail_min_score_increase)]
    selected_fail = (
        sorted(fail_candidates, key=lambda item: float(item.drop))[: int(top_n_per_group)]
        if include_fail
        else []
    )
    records = [("success", item) for item in selected_success] + [("fail", item) for item in selected_fail]
    n_rows = len(records)
    if n_rows == 0:
        raise ValueError("No examples selected for channel-filtered visualization")
    n_cols = 2 + 2 * len(tuple(fractions))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.15 * n_cols, max(3.0, 2.75 * n_rows)), squeeze=False, constrained_layout=True)
    skipped = []
    for row_idx, (group, example) in enumerate(records):
        try:
            clean_lb, patched_lb, _patch_bbox = exp._images_for_example(example)
            layer_maps = exp._load_layer_map_cache(example, layer_name=layer_name, include_clean_activation=False)
            if layer_maps is None:
                raise FileNotFoundError("missing layer_maps cache")
            delta = np.asarray(layer_maps["delta_chw"], dtype="float32")
            importance = np.asarray(layer_maps["segmentig_chw"], dtype="float32")
            axes[row_idx, 0].imshow(clean_lb)
            axes[row_idx, 0].set_title(
                f"{group} clean\nrank {row_idx + 1}, drop={example.drop:.3f}, Δscore={_score_increase(example):+.3f}"
            )
            axes[row_idx, 1].imshow(patched_lb)
            axes[row_idx, 1].set_title(f"{group} patched\nclean={example.conf_clean:.3f}, patch={example.conf_patch:.3f}")
            for ax in axes[row_idx, :2]:
                ax.axis("off")
            for frac_idx, percent in enumerate(fractions):
                channels = _top_channels_by_example_importance(importance, percent=float(percent))
                delta_map = np.mean(np.abs(delta[channels]), axis=0)
                importance_map = np.mean(np.abs(importance[channels]), axis=0)
                delta_ax = axes[row_idx, 2 + frac_idx]
                imp_ax = axes[row_idx, 2 + len(tuple(fractions)) + frac_idx]
                vmin, vmax = _raw_limits(delta_map, signed=False)
                im = delta_ax.imshow(delta_map, cmap="magma", vmin=vmin, vmax=vmax)
                delta_ax.set_title(f"delta | top {percent}% imp ch\n{len(channels)} ch raw")
                delta_ax.axis("off")
                fig.colorbar(im, ax=delta_ax, fraction=0.046, pad=0.04)
                vmin, vmax = _raw_limits(importance_map, signed=False)
                im = imp_ax.imshow(importance_map, cmap="viridis", vmin=vmin, vmax=vmax)
                imp_ax.set_title(f"importance | top {percent}% imp ch\n{len(channels)} ch raw")
                imp_ax.axis("off")
                fig.colorbar(im, ax=imp_ax, fraction=0.046, pad=0.04)
        except Exception as exc:  # noqa: BLE001 - visualize the rest.
            skipped.append({"path": example.path, "group": group, "reason": f"{type(exc).__name__}: {exc}"})
            for ax in axes[row_idx]:
                ax.axis("off")
            axes[row_idx, 0].set_title(f"{group}: skipped\n{type(exc).__name__}: {exc}")
    fig.suptitle(title or f"Top-{top_n_per_group} confident success/fail: raw delta and importance maps by top importance channels")
    fig._spread_precision_skipped = skipped  # attach lightweight diagnostic for notebook access
    return fig


def _patch_excluded_cache_path(exp, examples, *, layer_name: str, fractions) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [item.path for item in examples],
        "target_layer": layer_name,
        "detect_layer": exp.config.detect_layer,
        "target_mode": exp.config.target_mode,
        "fractions": [float(v) for v in fractions],
        "method_version": 2,
    }
    return exp.derived_cache_dir / f"patch_excluded_spread_metrics_{_cache_key(payload)}.pkl"


def compute_or_load_patch_excluded_spread_metrics(
    exp,
    examples=None,
    *,
    layer_name: str = "model.22",
    fractions=(1, 10, 100),
    force: bool = False,
) -> dict[str, Any]:
    cache = exp.get_cache()
    selected = list(examples or cache.examples)
    path = _patch_excluded_cache_path(exp, selected, layer_name=layer_name, fractions=fractions)
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached

    rows = []
    skipped = []
    for example in selected:
        layer_maps = exp._load_layer_map_cache(example, layer_name=layer_name, include_clean_activation=False)
        if layer_maps is None:
            skipped.append({"path": example.path, "success": bool(example.success), "drop": float(example.drop), "reason": "missing layer_maps cache"})
            continue
        delta = _as_chw(layer_maps["delta_chw"])
        importance = _as_chw(layer_maps["segmentig_chw"])
        object_bbox = _clean_bbox(example)
        channel_scores = np.abs(importance).reshape(importance.shape[0], -1).sum(axis=1)
        order = np.argsort(-channel_scores, kind="stable")
        for percent in fractions:
            if float(percent) >= 100:
                channels = order
                subset = "all_channels"
            else:
                k = max(1, int(round(float(percent) / 100.0 * importance.shape[0])))
                channels = order[:k]
                subset = f"top_{int(percent)}pct_channels"
            metrics = patch_excluded_spread_metrics(
                delta[channels],
                importance[channels],
                object_bbox_xyxy=object_bbox,
                patch_bbox_xyxy=example.patch_bbox_lb,
                imgsz=int(exp.config.attack.imgsz),
            )
            rows.append(
                {
                    "path": example.path,
                    "success": bool(example.success),
                    "drop": float(example.drop),
                    "conf_clean": float(example.conf_clean),
                    "conf_patch": float(example.conf_patch),
                    "subset": subset,
                    "percent": float(percent),
                    "n_channels": int(len(channels)),
                    **metrics,
                }
            )
    df = pd.DataFrame(rows)
    result = {
        "rows_df": df,
        "skipped": skipped,
        "cache_path": str(path),
        "loaded_from_cache": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh)
    return result


def plot_patch_excluded_spread_summary(df: pd.DataFrame):
    _ensure_plot_cache_env()
    import matplotlib.pyplot as plt

    metrics = [
        "delta_l2_rms_no_patch",
        "delta_abs_mean_no_patch",
        "object_bbox_delta_frac_no_patch",
        "object_bbox_delta_mean_no_patch",
        "top5pct_energy_frac_no_patch",
        "patch_roi_energy_frac_total",
    ]
    metrics = [metric for metric in metrics if metric in df.columns]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, max(3.2, 3.0 * len(metrics))), squeeze=False, constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        summary = df.groupby(["subset", "success"])[metric].mean().reset_index()
        subsets = list(dict.fromkeys(df["subset"].tolist()))
        x = np.arange(len(subsets))
        width = 0.36
        for offset, success, label, color in [(-width / 2, False, "fail", "#F58518"), (width / 2, True, "success", "#4C78A8")]:
            values = []
            for subset in subsets:
                sub = summary[(summary["subset"] == subset) & (summary["success"].astype(bool) == success)]
                values.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
            ax.bar(x + offset, values, width=width, label=label, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(subsets, rotation=20, ha="right")
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    return fig
