from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _as_chw(values) -> np.ndarray:
    arr = np.asarray(values, dtype="float32")
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={arr.shape}")
    return arr


def _top_neuron_indices_by_importance(importance_chw: np.ndarray, *, top_percent: float) -> np.ndarray:
    importance = np.abs(np.asarray(importance_chw, dtype="float64")).reshape(-1)
    if importance.size == 0 or float(top_percent) <= 0.0:
        return np.asarray([], dtype=int)
    if float(top_percent) >= 100.0:
        k = int(importance.size)
    else:
        k = max(1, int(round(float(top_percent) / 100.0 * importance.size)))
    k = min(k, importance.size)
    idx = np.argpartition(-importance, kth=k - 1)[:k]
    return idx[np.argsort(-importance[idx], kind="stable")]


def _percent_label(top_percent: float) -> str:
    return f"top{float(top_percent):g}".replace(".", "p")


def psnr_metric_name(top_percent: float, *, aggregate: str = "sum") -> str:
    return f"psnr_{_percent_label(top_percent)}_importance_{aggregate}"


def psnr_importance_top_metric(
    delta_chw,
    importance_chw,
    *,
    top_percent: float = 1.0,
    max_value: float = 255.0,
    eps: float = 1e-12,
) -> dict[str, float]:
    """PSNR-like sum over top-importance neurons: sum(log10(max_value^2 / delta^2))."""

    delta = _as_chw(delta_chw).reshape(-1).astype("float64", copy=False)
    importance = _as_chw(importance_chw).reshape(-1).astype("float64", copy=False)
    n = min(delta.size, importance.size)
    if n <= 0:
        return {
            "psnr_top1_importance_sum": float("nan"),
            "psnr_top1_importance_mean": float("nan"),
            "psnr_top1_importance_min": float("nan"),
            "psnr_top1_importance_max": float("nan"),
            "psnr_top1_importance_zero_frac": float("nan"),
            "psnr_top1_importance_k": 0.0,
        }

    delta = delta[:n]
    importance = importance[:n]
    label = _percent_label(float(top_percent))
    idx = _top_neuron_indices_by_importance(importance, top_percent=float(top_percent))
    if idx.size == 0:
        return {
            f"psnr_{label}_importance_sum": 0.0,
            f"psnr_{label}_importance_mean": float("nan"),
            f"psnr_{label}_importance_min": float("nan"),
            f"psnr_{label}_importance_max": float("nan"),
            f"psnr_{label}_importance_zero_frac": float("nan"),
            f"psnr_{label}_importance_k": 0.0,
        }
    d_top = delta[idx]
    d2 = np.maximum(d_top * d_top, float(eps))
    values = np.log10((float(max_value) * float(max_value)) / d2)
    return {
        f"psnr_{label}_importance_sum": float(np.sum(values)),
        f"psnr_{label}_importance_mean": float(np.mean(values)),
        f"psnr_{label}_importance_min": float(np.min(values)),
        f"psnr_{label}_importance_max": float(np.max(values)),
        f"psnr_{label}_importance_zero_frac": float(np.mean(np.abs(d_top) <= np.sqrt(float(eps)))),
        f"psnr_{label}_importance_k": float(idx.size),
    }


def _psnr_cache_path(
    exp,
    examples,
    *,
    layer_name: str,
    top_percent: float,
    max_value: float,
    eps: float,
) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [item.path for item in examples],
        "target_layer": layer_name,
        "detect_layer": exp.config.detect_layer,
        "target_mode": exp.config.target_mode,
        "top_percent": float(top_percent),
        "max_value": float(max_value),
        "eps": float(eps),
        "method_version": 2,
    }
    return exp.derived_cache_dir / f"psnr_importance_top_metrics_{_cache_key(payload)}.pkl"


def _psnr_sweep_cache_path(
    exp,
    examples,
    *,
    layer_name: str,
    percentages: tuple[float, ...],
    max_value: float,
    eps: float,
) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [item.path for item in examples],
        "target_layer": layer_name,
        "detect_layer": exp.config.detect_layer,
        "target_mode": exp.config.target_mode,
        "percentages": [float(v) for v in percentages],
        "max_value": float(max_value),
        "eps": float(eps),
        "method_version": 2,
    }
    return exp.derived_cache_dir / f"psnr_importance_percent_sweep_{_cache_key(payload)}.pkl"


def _psnr_importance_sweep_for_arrays(
    delta_chw,
    importance_chw,
    *,
    percentages: tuple[float, ...],
    max_value: float,
    eps: float,
) -> dict[str, float]:
    delta = _as_chw(delta_chw).reshape(-1).astype("float64", copy=False)
    importance = _as_chw(importance_chw).reshape(-1).astype("float64", copy=False)
    n = min(delta.size, importance.size)
    if n <= 0:
        return {psnr_metric_name(percent, aggregate="sum"): float("nan") for percent in percentages}

    delta = delta[:n]
    importance = np.abs(importance[:n])
    order = np.argsort(-importance, kind="stable")
    ranked_delta = delta[order]
    ranked_d2 = np.maximum(ranked_delta * ranked_delta, float(eps))
    ranked_values = np.log10((float(max_value) * float(max_value)) / ranked_d2)
    cumulative_sum = np.cumsum(ranked_values, dtype="float64")
    zero_flags = np.abs(ranked_delta) <= np.sqrt(float(eps))
    cumulative_zero = np.cumsum(zero_flags, dtype="float64")

    out: dict[str, float] = {}
    for percent in percentages:
        percent_float = float(percent)
        label = _percent_label(percent_float)
        if percent_float <= 0.0:
            k = 0
        elif percent_float >= 100.0:
            k = n
        else:
            k = max(1, int(round(percent_float / 100.0 * n)))
            k = min(k, n)
        if k <= 0:
            out[f"psnr_{label}_importance_sum"] = 0.0
            out[f"psnr_{label}_importance_mean"] = float("nan")
            out[f"psnr_{label}_importance_zero_frac"] = float("nan")
            out[f"psnr_{label}_importance_k"] = 0.0
            continue
        total = float(cumulative_sum[k - 1])
        out[f"psnr_{label}_importance_sum"] = total
        out[f"psnr_{label}_importance_mean"] = float(total / k)
        out[f"psnr_{label}_importance_zero_frac"] = float(cumulative_zero[k - 1] / k)
        out[f"psnr_{label}_importance_k"] = float(k)
    return out


def _directional_metric_quality_rows(labels, metrics_by_name: dict[str, object]) -> list[dict[str, Any]]:
    from .metrics import metric_quality_rows

    rows = metric_quality_rows(labels, metrics_by_name)
    for row in rows:
        raw_auc = float(row["roc_auc"])
        row["raw_roc_auc"] = raw_auc
        if np.isfinite(raw_auc) and int(row.get("best_direction", 1)) == -1:
            row["roc_auc"] = float(1.0 - raw_auc)
        else:
            row["roc_auc"] = raw_auc
    return rows


def compute_or_load_psnr_importance_top_metrics(
    exp,
    examples=None,
    *,
    layer_name: str = "model.22",
    top_percent: float = 1.0,
    max_value: float = 255.0,
    eps: float = 1e-12,
    force: bool = False,
) -> dict[str, Any]:
    from .regression_metrics import regression_similarity_table

    cache = exp.get_cache()
    selected = list(examples or cache.examples)
    path = _psnr_cache_path(
        exp,
        selected,
        layer_name=layer_name,
        top_percent=float(top_percent),
        max_value=float(max_value),
        eps=float(eps),
    )
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for example in selected:
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
        try:
            metrics = psnr_importance_top_metric(
                layer_maps["delta_chw"],
                layer_maps["segmentig_chw"],
                top_percent=float(top_percent),
                max_value=float(max_value),
                eps=float(eps),
            )
        except Exception as exc:  # noqa: BLE001 - keep the rest of the cached dataset usable.
            skipped.append(
                {
                    "path": example.path,
                    "success": bool(example.success),
                    "drop": float(example.drop),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        rows.append(
            {
                "path": example.path,
                "success": bool(example.success),
                "drop": float(example.drop),
                "conf_clean": float(example.conf_clean),
                "conf_patch": float(example.conf_patch),
                "layer_maps_cache_path": str(layer_maps["cache_path"]),
                **metrics,
            }
        )

    if not rows:
        raise RuntimeError(f"No PSNR rows; skipped={len(skipped)}")

    rows_df = pd.DataFrame(rows)
    metric_cols = [col for col in rows_df.columns if col.startswith("psnr_") and col != "psnr_top1_importance_k"]
    labels = rows_df["success"].astype(bool).to_numpy()
    quality = pd.DataFrame(_directional_metric_quality_rows(labels, {col: rows_df[col].to_numpy() for col in metric_cols}))
    quality["family"] = "psnr_top_importance"
    quality = quality.sort_values(["best_accuracy", "roc_auc"], ascending=False).reset_index(drop=True)
    regression = regression_similarity_table(
        rows_df[["path", "success", "drop", *metric_cols]],
        target_col="drop",
        metric_cols=metric_cols,
    )
    regression["family"] = "psnr_top_importance"
    regression = regression.sort_values(["main_score", "abs_spearman", "abs_pearson"], ascending=False).reset_index(drop=True)

    result = {
        "rows": rows,
        "metric_cols": metric_cols,
        "quality": quality,
        "regression": regression,
        "skipped": skipped,
        "cache_path": str(path),
        "loaded_from_cache": False,
        "params": {
            "layer_name": layer_name,
            "top_percent": float(top_percent),
            "max_value": float(max_value),
            "eps": float(eps),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh)
    return result


def compute_or_load_psnr_importance_percent_sweep(
    exp,
    examples=None,
    *,
    layer_name: str = "model.22",
    percentages=None,
    max_value: float = 255.0,
    eps: float = 1e-12,
    force: bool = False,
) -> dict[str, Any]:
    from .regression_metrics import regression_similarity_table

    cache = exp.get_cache()
    selected = list(examples or cache.examples)
    if percentages is None:
        percentages = np.round(np.arange(0.0, 10.0 + 0.1, 0.1), 1)
    percentages_tuple = tuple(float(v) for v in percentages)
    path = _psnr_sweep_cache_path(
        exp,
        selected,
        layer_name=layer_name,
        percentages=percentages_tuple,
        max_value=float(max_value),
        eps=float(eps),
    )
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for example in selected:
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
        try:
            metrics = _psnr_importance_sweep_for_arrays(
                layer_maps["delta_chw"],
                layer_maps["segmentig_chw"],
                percentages=percentages_tuple,
                max_value=float(max_value),
                eps=float(eps),
            )
        except Exception as exc:  # noqa: BLE001 - keep the rest of the cached dataset usable.
            skipped.append(
                {
                    "path": example.path,
                    "success": bool(example.success),
                    "drop": float(example.drop),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        rows.append(
            {
                "path": example.path,
                "success": bool(example.success),
                "drop": float(example.drop),
                "conf_clean": float(example.conf_clean),
                "conf_patch": float(example.conf_patch),
                "layer_maps_cache_path": str(layer_maps["cache_path"]),
                **metrics,
            }
        )

    if not rows:
        raise RuntimeError(f"No PSNR percent-sweep rows; skipped={len(skipped)}")

    rows_df = pd.DataFrame(rows)
    sum_cols = [psnr_metric_name(percent, aggregate="sum") for percent in percentages_tuple]
    mean_cols = [psnr_metric_name(percent, aggregate="mean") for percent in percentages_tuple]
    metric_cols = [col for col in [*sum_cols, *mean_cols] if col in rows_df.columns]
    labels = rows_df["success"].astype(bool).to_numpy()
    quality = pd.DataFrame(_directional_metric_quality_rows(labels, {col: rows_df[col].to_numpy() for col in metric_cols}))
    quality["family"] = "psnr_percent_sweep"
    quality["top_percent"] = quality["metric"].map(_metric_top_percent)
    quality["aggregate"] = quality["metric"].map(_metric_aggregate)
    quality = quality.sort_values(["best_accuracy", "roc_auc"], ascending=False).reset_index(drop=True)

    regression = regression_similarity_table(
        rows_df[["path", "success", "drop", *metric_cols]],
        target_col="drop",
        metric_cols=metric_cols,
    )
    regression["family"] = "psnr_percent_sweep"
    regression["top_percent"] = regression["metric"].map(_metric_top_percent)
    regression["aggregate"] = regression["metric"].map(_metric_aggregate)
    regression = regression.sort_values(["main_score", "abs_spearman", "abs_pearson"], ascending=False).reset_index(drop=True)

    result = {
        "rows": rows,
        "percentages": percentages_tuple,
        "metric_cols": metric_cols,
        "quality": quality,
        "regression": regression,
        "skipped": skipped,
        "cache_path": str(path),
        "loaded_from_cache": False,
        "params": {
            "layer_name": layer_name,
            "percentages": percentages_tuple,
            "max_value": float(max_value),
            "eps": float(eps),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh)
    return result


def _metric_top_percent(metric: str) -> float:
    text = str(metric)
    prefix = "psnr_top"
    suffix = "_importance_"
    if not text.startswith(prefix) or suffix not in text:
        return float("nan")
    raw = text[len(prefix) : text.index(suffix)].replace("p", ".")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _metric_aggregate(metric: str) -> str:
    text = str(metric)
    marker = "_importance_"
    if marker not in text:
        return ""
    return text.split(marker, 1)[1]


def compare_with_baselines(
    psnr_table: pd.DataFrame,
    baseline_table: pd.DataFrame,
    *,
    score_col: str,
    top_baselines: int = 10,
) -> pd.DataFrame:
    baseline = baseline_table.copy()
    psnr = psnr_table.copy()
    baseline["family"] = baseline.get("family", "previous")
    psnr["family"] = psnr.get("family", "psnr_top_importance")
    out = pd.concat(
        [
            psnr,
            baseline.sort_values(score_col, ascending=False).head(int(top_baselines)),
        ],
        ignore_index=True,
        sort=False,
    )
    return out.sort_values(score_col, ascending=False).reset_index(drop=True)


def plot_metric_distribution(rows_df: pd.DataFrame, *, metric: str):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    success = rows_df[rows_df["success"].astype(bool)]
    fail = rows_df[~rows_df["success"].astype(bool)]
    axes[0].hist(success[metric], bins=40, alpha=0.6, label="success", color="#4C78A8", density=True)
    axes[0].hist(fail[metric], bins=40, alpha=0.6, label="fail", color="#F58518", density=True)
    axes[0].set_title(f"{metric}: success/fail distribution")
    axes[0].set_xlabel(metric)
    axes[0].set_ylabel("density")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    colors = np.where(rows_df["success"].astype(bool), "#4C78A8", "#F58518")
    axes[1].scatter(rows_df[metric], rows_df["drop"], c=colors, alpha=0.55, s=18)
    axes[1].set_title(f"{metric} vs drop")
    axes[1].set_xlabel(metric)
    axes[1].set_ylabel("drop")
    axes[1].grid(alpha=0.25)
    return fig


def plot_metric_roc(rows_df: pd.DataFrame, quality_df: pd.DataFrame, *, metric: str):
    import matplotlib.pyplot as plt

    from .metrics import roc_curve_points

    row = quality_df[quality_df["metric"] == metric]
    if row.empty:
        raise KeyError(f"Metric {metric!r} was not found in quality_df")
    row = row.iloc[0]
    curve = roc_curve_points(rows_df["success"].astype(bool).to_numpy(), rows_df[metric].to_numpy(), direction=int(row["best_direction"]))
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    ax.plot(curve["fpr"], curve["tpr"], color="#4C78A8", lw=2)
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", alpha=0.55)
    ax.set_title(
        f"{metric}\nROC-AUC={float(row['roc_auc']):.3f}, best acc={float(row['best_accuracy']):.3f}"
    )
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.grid(alpha=0.25)
    return fig


def plot_comparison_bars(table: pd.DataFrame, *, score_col: str, title: str, top_n: int = 15):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("table is empty")
    sub = table.sort_values(score_col, ascending=False).head(int(top_n)).sort_values(score_col)
    colors_by_family = {
        "psnr_top_importance": "#D62728",
        "previous": "#777777",
        "spread": "#4C78A8",
        "precision": "#F58518",
        "segmentig": "#54A24B",
    }
    families = sub.get("family", pd.Series("previous", index=sub.index)).astype(str)
    colors = [colors_by_family.get(family, "#777777") for family in families]
    labels = sub["metric"].astype(str) + " | " + families

    fig_h = max(5.0, 0.34 * len(sub) + 1.4)
    fig, ax = plt.subplots(figsize=(12, fig_h), constrained_layout=True)
    y = np.arange(len(sub))
    ax.barh(y, sub[score_col].to_numpy(dtype="float64"), color=colors, alpha=0.88)
    ax.set_yticks(y, labels.tolist())
    ax.set_xlabel(score_col)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    handles = [
        mpatches.Patch(color=color, label=family)
        for family, color in colors_by_family.items()
        if family in set(families)
    ]
    ax.legend(handles=handles, loc="lower right")
    for yi, value in zip(y, sub[score_col].to_numpy(dtype="float64")):
        if np.isfinite(value):
            ax.text(value, yi, f" {value:.3f}", va="center", fontsize=9)
    return fig


def plot_percent_sweep_accuracy(quality_df: pd.DataFrame, *, score_col: str = "best_accuracy"):
    import matplotlib.pyplot as plt

    if quality_df.empty:
        raise ValueError("quality_df is empty")
    if score_col not in quality_df.columns:
        raise KeyError(f"{score_col!r} was not found in quality_df")

    df = quality_df.copy()
    if "top_percent" not in df.columns:
        df["top_percent"] = df["metric"].map(_metric_top_percent)
    if "aggregate" not in df.columns:
        df["aggregate"] = df["metric"].map(_metric_aggregate)
    df = df[np.isfinite(df["top_percent"])].copy()

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    styles = {
        "sum": {"color": "#D62728", "linestyle": "-", "marker": "o", "label": "sum"},
        "mean": {"color": "#4C78A8", "linestyle": "--", "marker": ".", "label": "mean"},
    }
    for aggregate, style in styles.items():
        sub = df[df["aggregate"] == aggregate].sort_values("top_percent")
        if sub.empty:
            continue
        ax.plot(
            sub["top_percent"].to_numpy(dtype="float64"),
            sub[score_col].to_numpy(dtype="float64"),
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4,
            linewidth=1.8,
            label=style["label"],
        )
        best_idx = sub[score_col].astype("float64").idxmax()
        best = sub.loc[best_idx]
        ax.scatter([best["top_percent"]], [best[score_col]], color=style["color"], s=72, edgecolor="black", zorder=4)
        ax.annotate(
            f"{float(best['top_percent']):.1f}%\\n{float(best[score_col]):.3f}",
            xy=(float(best["top_percent"]), float(best[score_col])),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title(f"PSNR-like metric: {score_col} vs top-importance percent")
    ax.set_xlabel("top importance neurons, %")
    ax.set_ylabel(score_col)
    ax.grid(alpha=0.25)
    ax.legend()
    return fig
