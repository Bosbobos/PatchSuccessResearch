from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EXCLUDE_COLUMNS = {"success", "drop", "conf_clean", "conf_patch", "clean_logit", "patched_logit"}


def regression_metric_columns(rows_df: pd.DataFrame, *, target_col: str = "drop") -> list[str]:
    cols = []
    for col in rows_df.columns:
        if col in DEFAULT_EXCLUDE_COLUMNS or col == target_col:
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
    if x.size < 3 or float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return {"linear_slope": float("nan"), "linear_intercept": float("nan"), "linear_r2": float("nan")}
    slope, intercept = np.polyfit(x, y, deg=1)
    pred = slope * x + intercept
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {"linear_slope": float(slope), "linear_intercept": float(intercept), "linear_r2": float(1.0 - ss_res / (ss_tot + 1e-12))}


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
    rows = []
    for col in metric_cols:
        feature = pd.to_numeric(rows_df[col], errors="coerce")
        clean = pd.concat([target.rename(target_col), feature.rename(col)], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 3:
            continue
        x = clean[col].to_numpy(dtype="float64")
        y = clean[target_col].to_numpy(dtype="float64")
        pearson = _safe_corr(clean[col], clean[target_col], "pearson")
        spearman = _safe_corr(clean[col], clean[target_col], "spearman")
        kendall = _safe_corr(clean[col], clean[target_col], "kendall")
        row = {
            "metric": col,
            "n": int(len(clean)),
            "coverage": float(len(clean) / max(1, len(rows_df))),
            "pearson": pearson,
            "spearman": spearman,
            "kendall": kendall,
            "abs_pearson": abs(pearson) if np.isfinite(pearson) else float("nan"),
            "abs_spearman": abs(spearman) if np.isfinite(spearman) else float("nan"),
            "abs_kendall": abs(kendall) if np.isfinite(kendall) else float("nan"),
            **_linear_fit_stats(x, y),
        }
        # Primary ranking metric: monotonic agreement with drop.
        row["main_score"] = row["abs_spearman"]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["main_score", "abs_spearman", "abs_pearson"], ascending=False).reset_index(drop=True)


def compute_or_load_drop_regression_similarity(
    exp: Any,
    rows_df: pd.DataFrame,
    *,
    target_col: str = "drop",
    metric_cols: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    metric_cols = metric_cols or regression_metric_columns(rows_df, target_col=target_col)
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "target_col": target_col,
        "metric_cols": metric_cols,
        "n_rows": int(len(rows_df)),
        "method_version": 2,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
    path = Path(exp.derived_cache_dir) / f"drop_regression_similarity_{key}.pkl"
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached
    table = regression_similarity_table(rows_df, target_col=target_col, metric_cols=metric_cols)
    result = {"similarity_df": table, "metric_cols": metric_cols, "cache_path": str(path), "loaded_from_cache": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh)
    return result
