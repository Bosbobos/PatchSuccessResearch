from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLASSIFICATION_SCORE_COLUMNS = {
    "best_balanced_accuracy",
    "best_accuracy",
    "balanced_precision",
    "balanced_recall",
    "balanced_specificity",
    "balanced_f1",
    "balanced_accuracy_plain",
    "balanced_threshold",
    "balanced_direction",
    "roc_auc",
    "raw_roc_auc",
    "roc_auc_effective",
    "best_threshold",
    "best_direction",
    "train_acc",
    "val_acc",
    "train_auc",
    "val_auc",
    "cv_acc",
    "cv_auc",
    "all_best_acc",
    "all_auc",
}

REGRESSION_SCORE_COLUMNS = {
    "n",
    "coverage",
    "pearson",
    "spearman",
    "kendall",
    "abs_pearson",
    "abs_spearman",
    "abs_kendall",
    "mutual_info",
    "distance_corr",
    "linear_r2",
    "calibrated_mae",
    "calibrated_rmse",
    "calibrated_nrmse",
    "mutual_info_norm",
    "calibrated_error_score",
    "main_score",
}

META_COLUMNS = {
    "path",
    "success",
    "drop",
    "conf_clean",
    "conf_patch",
    "clean_logit",
    "patched_logit",
    "layer_maps_cache_path",
    "cov_cache_path",
    "method_maps_cache_path",
}


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


def _use_plain_accuracy_as_balanced_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "best_accuracy" not in df.columns:
        return df
    out = df.copy()
    accuracy = pd.to_numeric(out["best_accuracy"], errors="coerce")
    out["best_balanced_accuracy"] = accuracy
    if "balanced_accuracy_plain" in out.columns:
        out["balanced_accuracy_plain"] = accuracy
    return out


def _annotate_barh(ax, y, values, labels, *, fontsize: float):
    for yi, value, label in zip(y, values, labels):
        if np.isfinite(float(value)) and label:
            ax.text(float(value) + 0.012, yi, label, va="center", ha="left", fontsize=fontsize, clip_on=False)


def _cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _leaderboard_cache_path(exp, cache_files: list[Path]) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "cache_files": [
            {
                "path": str(path),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "size": int(path.stat().st_size),
            }
            for path in cache_files
            if path.exists()
        ],
        "method_version": 3,
    }
    return exp.derived_cache_dir / f"metric_leaderboard_{_cache_key(payload)}.pkl"


def _candidate_cache_files(cache_dir: Path) -> list[Path]:
    patterns = [
        "segmentig_success_failure_metrics_*.pkl",
        "success_failure_metrics_*.pkl",
        "spread_vs_precision_*.pkl",
        "patch_excluded_spread_metrics_*.pkl",
        "attribution_method_suite_*.pkl",
        "psnr_importance_top_metrics_*.pkl",
        "psnr_importance_percent_sweep_*.pkl",
        "drop_regression_similarity_*.pkl",
        "binmetrics_*.pkl",
        "covariance_split_metrics/*.pkl",
        "covariance_threshold_sweep_focus/*.pkl",
        "importance_energy_focus_metrics/*.pkl",
    ]
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(cache_dir.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            files.append(path)
    return files


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _as_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, tuple):
        return pd.DataFrame(list(value))
    return pd.DataFrame()


def _source_name(path: Path) -> str:
    name = path.name
    for prefix in (
        "segmentig_success_failure_metrics",
        "success_failure_metrics",
        "spread_vs_precision",
        "patch_excluded_spread_metrics",
        "attribution_method_suite",
        "psnr_importance_top_metrics",
        "psnr_importance_percent_sweep",
        "drop_regression_similarity",
        "binmetrics_bin_granularity_v1",
        "binmetrics_elasticnet_bin_logreg_v1",
        "binmetrics_handcrafted_selection_v1",
        "binmetrics_robust_product_compression_v1",
        "binmetrics_unregularized_logreg_capacity_v1",
        "covariance_split_metrics",
        "covariance_threshold_sweep_focus",
        "importance_energy_focus_metrics",
    ):
        if name.startswith(prefix):
            return prefix
    return name.rsplit("_", 1)[0]


def _source_family(source: str, table: pd.DataFrame) -> str:
    if source.startswith("psnr_"):
        return "psnr"
    if source.startswith("spread_vs_precision"):
        return "spread_vs_precision"
    if source.startswith("patch_excluded_spread"):
        return "patch_excluded_spread"
    if source.startswith("covariance"):
        return "covariance"
    if source.startswith("importance_energy_focus"):
        return "covariance_focus"
    if source.startswith("attribution_method_suite"):
        return "attribution_methods"
    if source.startswith("binmetrics"):
        return "binmetrics"
    if source.startswith("drop_regression_similarity"):
        return "segmentig_regression"
    if source.startswith("segmentig") or source.startswith("success_failure"):
        return "segmentig"
    family = table.get("family")
    if family is not None and len(family):
        return str(family.iloc[0])
    return source


def _decorate_table(
    table: pd.DataFrame,
    *,
    source: str,
    source_table: str,
    cache_file: Path,
    n_rows: int | None,
    task: str,
) -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    if "metric" not in out.columns:
        if "method" in out.columns:
            out["metric"] = out["method"].astype(str)
        elif "name" in out.columns:
            out["metric"] = out["name"].astype(str)
        elif "variant" in out.columns:
            out["metric"] = out["variant"].astype(str)
        else:
            out["metric"] = [f"metric_{idx}" for idx in range(len(out))]
    if "family" not in out.columns:
        out["family"] = _source_family(source, out)
    out["source_family"] = _source_family(source, out)
    out["source"] = source
    out["source_table"] = source_table
    out["cache_file"] = str(cache_file)
    out["cache_mtime"] = float(cache_file.stat().st_mtime)
    out["n_examples"] = float(n_rows) if n_rows is not None else np.nan
    out["task"] = task
    qualifier_cols = [
        col
        for col in (
            "method",
            "family",
            "cov_method",
            "sigma",
            "cov_group",
            "threshold_label",
            "threshold_kind",
            "aggregate",
            "top_percent",
            "subset",
            "percent",
            "kind",
            "model_kind",
            "source",
        )
        if col in out.columns
    ]
    qualifier = out[qualifier_cols].astype(str).agg("|".join, axis=1) if qualifier_cols else ""
    out["metric_key"] = out["source"].astype(str) + "::" + out["metric"].astype(str)
    if qualifier_cols:
        out["metric_key"] = out["metric_key"] + "::" + qualifier
    out["display_metric"] = out["metric"].astype(str)
    return out


def _numeric_metric_columns(rows_df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in rows_df.columns:
        if col in META_COLUMNS:
            continue
        if pd.api.types.is_bool_dtype(rows_df[col]):
            continue
        values = pd.to_numeric(rows_df[col], errors="coerce")
        if values.notna().sum() >= 3 and float(values.std(skipna=True) or 0.0) > 0.0:
            cols.append(col)
    return cols


def _quality_from_rows(rows_df: pd.DataFrame, *, metric_names: list[str] | None = None) -> pd.DataFrame:
    if rows_df.empty or "success" not in rows_df.columns:
        return pd.DataFrame()
    from .metrics import metric_quality_rows

    if metric_names is None:
        metric_cols = _numeric_metric_columns(rows_df)
        metric_cols = [col for col in metric_cols if col not in {"drop", "conf_clean", "conf_patch", "clean_logit", "patched_logit"}]
    else:
        metric_cols = [col for col in metric_names if col in rows_df.columns]
    if not metric_cols:
        return pd.DataFrame()
    labels = rows_df["success"].astype(bool).to_numpy()
    table = pd.DataFrame(metric_quality_rows(labels, {col: rows_df[col].to_numpy() for col in metric_cols}))
    if table.empty:
        return table
    table["raw_roc_auc"] = table["roc_auc"]
    inverse = table["best_direction"].astype(int) == -1
    table.loc[inverse, "roc_auc"] = 1.0 - table.loc[inverse, "raw_roc_auc"]
    return table


def _regression_from_rows(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty or "drop" not in rows_df.columns:
        return pd.DataFrame()
    from .regression_metrics import regression_similarity_table

    metric_cols = _numeric_metric_columns(rows_df)
    metric_cols = [col for col in metric_cols if col not in {"conf_clean", "conf_patch"}]
    if not metric_cols:
        return pd.DataFrame()
    return regression_similarity_table(rows_df[["drop", *metric_cols]], target_col="drop", metric_cols=metric_cols)


def _extract_classification_tables(payload: dict[str, Any], path: Path) -> list[tuple[str, pd.DataFrame]]:
    tables: list[tuple[str, pd.DataFrame]] = []
    for key in ("quality", "all_quality", "classification"):
        if key in payload:
            tables.append((key, _as_dataframe(payload[key])))
    metric_names: list[str] | None = None
    for _, table in tables:
        if not table.empty and "metric" in table.columns:
            metric_names = table["metric"].astype(str).tolist()
            break

    if "overall_val_df" in payload:
        table = _as_dataframe(payload["overall_val_df"]).rename(
            columns={"method": "metric", "val_acc": "best_accuracy", "val_auc": "roc_auc"}
        )
        tables.append(("overall_val_df", table))
    if "selection_df" in payload:
        table = _as_dataframe(payload["selection_df"]).rename(
            columns={"name": "metric", "val_acc": "best_accuracy", "val_auc": "roc_auc"}
        )
        tables.append(("selection_df", table))
    if "bin_logreg_df" in payload:
        table = _as_dataframe(payload["bin_logreg_df"]).rename(
            columns={"variant": "metric", "val_acc": "best_accuracy", "val_auc": "roc_auc"}
        )
        tables.append(("bin_logreg_df", table))

    return [(name, table) for name, table in tables if not table.empty]


def _extract_regression_tables(payload: dict[str, Any]) -> list[tuple[str, pd.DataFrame]]:
    tables: list[tuple[str, pd.DataFrame]] = []
    for key in ("regression", "all_regression", "similarity_df"):
        if key in payload:
            tables.append((key, _as_dataframe(payload[key])))
    for rows_key in ("rows_df", "rows"):
        if rows_key in payload:
            regression = _regression_from_rows(_as_dataframe(payload[rows_key]))
            if not regression.empty:
                tables.append((f"{rows_key}_regression", regression))
    return [(name, table) for name, table in tables if not table.empty]


def _n_rows(payload: dict[str, Any]) -> int | None:
    for key in ("rows", "rows_df"):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, pd.DataFrame):
            return int(len(value))
        if isinstance(value, list | tuple):
            return int(len(value))
    if "n_rows" in payload:
        try:
            return int(payload["n_rows"])
        except Exception:
            return None
    return None


def _deduplicate_leaderboard(df: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    if df.empty or "metric_key" not in df.columns:
        return df
    sort_cols = ["metric_key", "n_examples", "cache_mtime"]
    ascending = [True, False, False]
    if score_col in df.columns:
        sort_cols.insert(1, score_col)
        ascending.insert(1, False)
    return (
        df.sort_values(sort_cols, ascending=ascending)
        .drop_duplicates("metric_key", keep="first")
        .reset_index(drop=True)
    )


def compute_or_load_metric_leaderboard(exp, *, force: bool = False) -> dict[str, Any]:
    cache_files = _candidate_cache_files(exp.derived_cache_dir)
    path = _leaderboard_cache_path(exp, cache_files)
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached

    classification_parts: list[pd.DataFrame] = []
    regression_parts: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []

    for cache_file in cache_files:
        try:
            payload = _read_pickle(cache_file)
        except Exception as exc:  # noqa: BLE001 - corrupt/legacy cache should not stop the leaderboard.
            skipped.append({"cache_file": str(cache_file), "reason": f"load {type(exc).__name__}: {exc}"})
            continue
        if not isinstance(payload, dict):
            skipped.append({"cache_file": str(cache_file), "reason": f"unsupported payload {type(payload).__name__}"})
            continue
        source = _source_name(cache_file)
        n_rows = _n_rows(payload)
        try:
            for table_name, table in _extract_classification_tables(payload, cache_file):
                classification_parts.append(
                    _decorate_table(
                        table,
                        source=source,
                        source_table=table_name,
                        cache_file=cache_file,
                        n_rows=n_rows,
                        task="classification",
                    )
                )
            for table_name, table in _extract_regression_tables(payload):
                regression_parts.append(
                    _decorate_table(
                        table,
                        source=source,
                        source_table=table_name,
                        cache_file=cache_file,
                        n_rows=n_rows,
                        task="regression",
                    )
                )
        except Exception as exc:  # noqa: BLE001 - keep collecting other caches.
            skipped.append({"cache_file": str(cache_file), "reason": f"extract {type(exc).__name__}: {exc}"})

    classification_all = pd.concat(classification_parts, ignore_index=True, sort=False) if classification_parts else pd.DataFrame()
    regression_all = pd.concat(regression_parts, ignore_index=True, sort=False) if regression_parts else pd.DataFrame()
    if not classification_all.empty and "best_balanced_accuracy" not in classification_all.columns:
        classification_all["best_balanced_accuracy"] = np.nan
    classification_all = _use_plain_accuracy_as_balanced_accuracy(classification_all)
    if not regression_all.empty and "abs_spearman" in regression_all.columns:
        regression_all["main_score"] = regression_all["abs_spearman"]
    classification = _deduplicate_leaderboard(classification_all, score_col="best_balanced_accuracy")
    regression = _deduplicate_leaderboard(regression_all, score_col="main_score")

    result = {
        "classification": classification.sort_values(["best_balanced_accuracy", "roc_auc", "best_accuracy"], ascending=False, na_position="last").reset_index(drop=True)
        if not classification.empty and "best_balanced_accuracy" in classification.columns
        else classification,
        "regression": regression.sort_values(["main_score", "abs_spearman", "abs_pearson"], ascending=False, na_position="last").reset_index(drop=True)
        if not regression.empty and "main_score" in regression.columns
        else regression,
        "classification_all": classification_all,
        "regression_all": regression_all,
        "cache_files": [str(path) for path in cache_files],
        "skipped": pd.DataFrame(skipped),
        "loaded_from_cache": False,
        "cache_path": str(path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh)
    return result


def plot_leaderboard_bars(
    table: pd.DataFrame,
    *,
    score_col: str,
    top_n: int = 30,
    title: str | None = None,
):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("table is empty")
    if score_col not in table.columns:
        raise KeyError(f"{score_col!r} was not found in table")

    sub = table.sort_values(score_col, ascending=False).head(int(top_n)).copy()
    sub = sub.sort_values(score_col)
    families = sub.get("source_family", pd.Series("unknown", index=sub.index)).astype(str)
    palette = {
        "psnr": "#D62728",
        "segmentig": "#54A24B",
        "segmentig_regression": "#54A24B",
        "spread_vs_precision": "#4C78A8",
        "patch_excluded_spread": "#72B7B2",
        "covariance": "#B279A2",
        "covariance_focus": "#9D755D",
        "attribution_methods": "#F58518",
        "binmetrics": "#E45756",
    }
    colors = [palette.get(family, "#777777") for family in families]
    labels = sub["metric"].astype(str)
    if "family" in sub.columns:
        labels = labels + " | " + sub["family"].astype(str)
    labels = labels + " | " + families

    fig_h = max(5.0, 0.34 * len(sub) + 1.4)
    fig, ax = plt.subplots(figsize=(13, fig_h), constrained_layout=True)
    y = np.arange(len(sub))
    values = sub[score_col].to_numpy(dtype="float64")
    ax.barh(y, values, color=colors, alpha=0.88)
    ax.set_yticks(y, labels.tolist())
    if score_col == "best_balanced_accuracy":
        value_labels = [_quality_label(row, score_col) for _, row in sub.iterrows()]
        ax.set_xlim(0, 1.28)
    else:
        value_labels = [f"{value:.3f}" if np.isfinite(value) else "" for value in values]
        if score_col in {"main_score", "abs_spearman"}:
            ax.set_xlim(0, 1.12)
    _annotate_barh(ax, y, values, value_labels, fontsize=_bar_label_fontsize(len(sub)))
    ax.set_xlabel(score_col)
    ax.set_title(title or f"Metric leaderboard by {score_col}")
    ax.grid(axis="x", alpha=0.25)
    handles = [mpatches.Patch(color=color, label=family) for family, color in palette.items() if family in set(families)]
    ax.legend(handles=handles, loc="lower right")
    return fig


def best_by_source_family(table: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    if table.empty:
        return table
    return (
        table.sort_values(["source_family", score_col], ascending=[True, False], na_position="last")
        .groupby("source_family", as_index=False)
        .head(1)
        .sort_values(score_col, ascending=False, na_position="last")
        .reset_index(drop=True)
    )
