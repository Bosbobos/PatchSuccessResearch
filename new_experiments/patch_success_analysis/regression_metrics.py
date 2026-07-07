from __future__ import annotations

import hashlib
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EXCLUDE_COLUMNS = {
    "success",
    "drop",
    "conf_clean",
    "conf_patch",
    "target_class_id",
}


def _bar_label_fontsize(n_rows: int) -> float:
    return max(6.0, min(9.0, 180.0 / max(1, int(n_rows))))


def regression_metric_columns(
    rows_df: pd.DataFrame,
    *,
    target_col: str = "drop",
    exclude_columns: set[str] | None = None,
) -> list[str]:
    exclude = set(DEFAULT_EXCLUDE_COLUMNS)
    if exclude_columns:
        exclude.update(exclude_columns)
    exclude.add(target_col)
    cols: list[str] = []
    for col in rows_df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_bool_dtype(rows_df[col]):
            continue
        values = pd.to_numeric(rows_df[col], errors="coerce")
        if values.notna().sum() >= 3 and float(values.std(skipna=True) or 0.0) > 0.0:
            cols.append(col)
    return cols


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    try:
        value = x.corr(y, method=method)
    except Exception:
        return float("nan")
    return float(value) if pd.notna(value) else float("nan")


def _linear_fit_stats(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x.size < 3 or x_std <= 0.0 or y_std <= 0.0:
        return {
            "linear_slope": float("nan"),
            "linear_intercept": float("nan"),
            "linear_r2": float("nan"),
            "calibrated_mae": float("nan"),
            "calibrated_rmse": float("nan"),
            "calibrated_nrmse": float("nan"),
        }
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * x + intercept
    err = pred - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "linear_r2": float(r2),
        "calibrated_mae": mae,
        "calibrated_rmse": rmse,
        "calibrated_nrmse": float(rmse / (y_std + 1e-12)),
    }


def _mutual_info_scores(feature_df: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            from sklearn.feature_selection import mutual_info_regression
    except Exception:
        return {col: float("nan") for col in feature_df.columns}

    clean = pd.concat([target.rename("__target__"), feature_df], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 5:
        return {col: float("nan") for col in feature_df.columns}
    x = clean[feature_df.columns].to_numpy(dtype="float64")
    y = clean["__target__"].to_numpy(dtype="float64")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            mi = mutual_info_regression(x, y, random_state=17)
    except Exception:
        return {col: float("nan") for col in feature_df.columns}
    return {col: float(value) for col, value in zip(feature_df.columns, mi)}


def _distance_corr(x: np.ndarray, y: np.ndarray, *, max_n: int = 1000) -> float:
    if x.size < 3 or float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    if x.size > max_n:
        idx = np.linspace(0, x.size - 1, int(max_n)).astype(int)
        x = x[idx]
        y = y[idx]
    x_dist = np.abs(x[:, None] - x[None, :])
    y_dist = np.abs(y[:, None] - y[None, :])
    x_centered = x_dist - x_dist.mean(axis=0, keepdims=True) - x_dist.mean(axis=1, keepdims=True) + x_dist.mean()
    y_centered = y_dist - y_dist.mean(axis=0, keepdims=True) - y_dist.mean(axis=1, keepdims=True) + y_dist.mean()
    dcov2 = float(np.mean(x_centered * y_centered))
    dvar_x = float(np.mean(x_centered * x_centered))
    dvar_y = float(np.mean(y_centered * y_centered))
    if dvar_x <= 0.0 or dvar_y <= 0.0:
        return float("nan")
    return float(np.sqrt(max(dcov2, 0.0) / np.sqrt(dvar_x * dvar_y)))


def regression_similarity_table(
    rows_df: pd.DataFrame,
    *,
    target_col: str = "drop",
    metric_cols: list[str] | None = None,
) -> pd.DataFrame:
    if target_col not in rows_df.columns:
        raise KeyError(f"Missing target column {target_col!r}")
    target = pd.to_numeric(rows_df[target_col], errors="coerce")
    metric_cols = metric_cols or regression_metric_columns(rows_df, target_col=target_col)
    feature_df = rows_df[metric_cols].apply(pd.to_numeric, errors="coerce")
    mi_by_col = _mutual_info_scores(feature_df, target)
    target_std = float(target.std(skipna=True) or 0.0)

    out = []
    for col in metric_cols:
        x = feature_df[col]
        clean = pd.concat([target.rename(target_col), x.rename(col)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 3:
            continue
        x_clean = clean[col].to_numpy(dtype="float64")
        y_clean = clean[target_col].to_numpy(dtype="float64")
        pearson = _safe_corr(clean[col], clean[target_col], "pearson")
        spearman = _safe_corr(clean[col], clean[target_col], "spearman")
        kendall = _safe_corr(clean[col], clean[target_col], "kendall")
        fit = _linear_fit_stats(x_clean, y_clean)
        distance_corr = _distance_corr(x_clean, y_clean)
        out.append(
            {
                "metric": col,
                "n": int(len(clean)),
                "coverage": float(len(clean) / max(1, len(rows_df))),
                "pearson": pearson,
                "spearman": spearman,
                "kendall": kendall,
                "abs_pearson": abs(pearson) if np.isfinite(pearson) else float("nan"),
                "abs_spearman": abs(spearman) if np.isfinite(spearman) else float("nan"),
                "abs_kendall": abs(kendall) if np.isfinite(kendall) else float("nan"),
                "mutual_info": mi_by_col.get(col, float("nan")),
                "distance_corr": distance_corr,
                "metric_mean": float(np.mean(x_clean)),
                "metric_std": float(np.std(x_clean)),
                "target_std": target_std,
                **fit,
            }
        )
    df = pd.DataFrame(out)
    if df.empty:
        return df
    mi_max = float(df["mutual_info"].max(skipna=True) or 0.0)
    df["mutual_info_norm"] = df["mutual_info"] / mi_max if mi_max > 0.0 else np.nan
    df["calibrated_error_score"] = 1.0 / (1.0 + df["calibrated_nrmse"])
    # Primary ranking metric: monotonic agreement with drop.
    # Other dependency diagnostics remain in the table but no longer enter the
    # main leaderboard score.
    df["main_score"] = df["abs_spearman"]
    return df.sort_values(["main_score", "abs_spearman", "abs_pearson"], ascending=False).reset_index(drop=True)


def _cache_path(exp: Any, rows_df: pd.DataFrame, *, target_col: str, metric_cols: list[str]) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "target_col": target_col,
        "metric_cols": list(metric_cols),
        "n_rows": int(len(rows_df)),
        "paths_head": rows_df.get("path", pd.Series(dtype=str)).head(20).astype(str).tolist(),
        "method_version": 3,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
    return exp.derived_cache_dir / f"drop_regression_similarity_{key}.pkl"


def compute_or_load_drop_regression_similarity(
    exp: Any,
    rows_df: pd.DataFrame,
    *,
    target_col: str = "drop",
    metric_cols: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    metric_cols = metric_cols or regression_metric_columns(rows_df, target_col=target_col)
    path = _cache_path(exp, rows_df, target_col=target_col, metric_cols=metric_cols)
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached

    table = regression_similarity_table(rows_df, target_col=target_col, metric_cols=metric_cols)
    result = {
        "target_col": target_col,
        "metric_cols": metric_cols,
        "similarity_df": table,
        "n_rows": int(len(rows_df)),
        "n_metrics": int(len(metric_cols)),
        "loaded_from_cache": False,
        "cache_path": str(path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh)
    return result


def plot_similarity_score_bars(similarity_df: pd.DataFrame, *, score_col: str = "main_score"):
    import matplotlib.pyplot as plt

    df = similarity_df.sort_values(score_col, ascending=True)
    score_label = "|Spearman| (main)" if score_col == "main_score" else score_col
    height = max(7.0, min(42.0, 0.19 * len(df) + 1.8))
    fig, ax = plt.subplots(figsize=(13, height), constrained_layout=True)
    colors = np.where(df["spearman"].to_numpy(dtype="float64") >= 0.0, "#4C78A8", "#F58518")
    y = np.arange(len(df))
    values = df[score_col].to_numpy(dtype="float64")
    ax.barh(y, values, color=colors, alpha=0.85)
    ax.set_yticks(y, df["metric"].astype(str).tolist())
    fontsize = _bar_label_fontsize(len(df))
    for yi, value in zip(y, values):
        if np.isfinite(value):
            ax.text(float(value) + 0.012, yi, f"{float(value):.3f}", va="center", ha="left", fontsize=fontsize, clip_on=False)
    ax.set_title(f"All scalar metrics ranked by {score_label}")
    ax.set_xlabel(score_label)
    if score_col in {"main_score", "abs_spearman"}:
        ax.set_xlim(0, 1.12)
    ax.grid(axis="x", alpha=0.25)
    return fig


def plot_similarity_heatmap(similarity_df: pd.DataFrame, *, top_n: int | None = None):
    import matplotlib.pyplot as plt

    cols = ["pearson", "spearman", "kendall", "linear_r2", "mutual_info_norm", "distance_corr", "calibrated_error_score"]
    df = similarity_df.copy()
    if top_n is not None:
        df = df.head(int(top_n))
    data = df[cols].to_numpy(dtype="float64")
    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.27 * len(df) + 1.5)), constrained_layout=True)
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["metric"], fontsize=8)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_title("Drop similarity metrics")
    fig.colorbar(im, ax=ax, shrink=0.85)
    return fig


def plot_top_metric_scatter(rows_df: pd.DataFrame, similarity_df: pd.DataFrame, *, target_col: str = "drop", top_n: int = 12):
    import math
    import matplotlib.pyplot as plt

    top = similarity_df.head(int(top_n)).copy()
    n = len(top)
    ncols = 3
    nrows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.8 * nrows), squeeze=False, constrained_layout=True)
    y = pd.to_numeric(rows_df[target_col], errors="coerce")
    success = rows_df.get("success", pd.Series(False, index=rows_df.index)).astype(bool)
    for ax, (_, meta) in zip(axes.ravel(), top.iterrows()):
        metric = meta["metric"]
        x = pd.to_numeric(rows_df[metric], errors="coerce")
        clean = pd.concat([x.rename("x"), y.rename("y"), success.rename("success")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        ax.scatter(clean.loc[~clean["success"], "x"], clean.loc[~clean["success"], "y"], s=14, alpha=0.45, label="fail", color="#F58518")
        ax.scatter(clean.loc[clean["success"], "x"], clean.loc[clean["success"], "y"], s=14, alpha=0.45, label="success", color="#4C78A8")
        if len(clean) >= 3 and float(clean["x"].std()) > 0.0:
            xs = np.linspace(float(clean["x"].min()), float(clean["x"].max()), 100)
            slope = float(meta.get("linear_slope", np.nan))
            intercept = float(meta.get("linear_intercept", np.nan))
            if np.isfinite(slope) and np.isfinite(intercept):
                ax.plot(xs, slope * xs + intercept, color="black", linewidth=1.2)
        ax.set_title(f"{metric}\nρ={meta['spearman']:.3f}, r={meta['pearson']:.3f}, R2={meta['linear_r2']:.3f}", fontsize=9)
        ax.set_xlabel(metric)
        ax.set_ylabel(target_col)
        ax.grid(alpha=0.22)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    return fig
