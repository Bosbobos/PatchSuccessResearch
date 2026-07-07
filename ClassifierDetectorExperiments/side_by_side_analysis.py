from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "patch_success_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    tqdm = None


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


def _annotate_vertical_bars(ax, *, fontsize: float):
    for patch in ax.patches:
        height = float(patch.get_height())
        if np.isfinite(height):
            ax.text(
                patch.get_x() + patch.get_width() / 2.0,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                rotation=0,
                clip_on=False,
            )


@dataclass(frozen=True)
class ExperimentRoots:
    repo_root: Path
    classifier_output: Path
    classifier_cache: Path
    detector_output: Path | None
    detector_cache: Path | None
    comparison_output: Path


def resolve_roots(repo_root: str | Path | None = None) -> ExperimentRoots:
    repo = Path(repo_root or Path.cwd()).resolve()
    classifier_output = repo / "classifier_experiments" / "outputs" / "classifier_patch_analysis"
    classifier_cache = classifier_output / "cache"

    detector_candidates = [
        repo / "new_experiments" / "outputs" / "patch_success_analysis",
        repo / "new_experiments" / "new_experiments" / "outputs" / "patch_success_analysis",
    ]
    detector_output = None
    source_cache_prefixes = (
        "success_failure_metrics_",
        "segmentig_success_failure_metrics_",
        "spread_vs_precision_",
        "psnr_importance_",
        "binmetrics_",
        "covariance_",
        "importance_energy_focus_",
    )
    for candidate in detector_candidates:
        cache = candidate / "cache"
        if cache.exists() and any(path.name.startswith(source_cache_prefixes) for path in cache.rglob("*.pkl")):
            detector_output = candidate
            break
    if detector_output is None:
        detector_output = next((p for p in detector_candidates if p.exists()), None)
    detector_cache = detector_output / "cache" if detector_output is not None else None

    comparison_output = repo / "ClassifierDetectorExperiments" / "outputs"
    comparison_output.mkdir(parents=True, exist_ok=True)
    classifier_cache.mkdir(parents=True, exist_ok=True)
    if detector_cache is not None:
        detector_cache.mkdir(parents=True, exist_ok=True)

    return ExperimentRoots(
        repo_root=repo,
        classifier_output=classifier_output,
        classifier_cache=classifier_cache,
        detector_output=detector_output,
        detector_cache=detector_cache,
        comparison_output=comparison_output,
    )


def _cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _write_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh)


def _as_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, tuple):
        return pd.DataFrame(list(value))
    return pd.DataFrame()


def _latest_file(paths: list[Path]) -> Path | None:
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return max(paths, key=lambda p: (p.stat().st_mtime_ns, p.stat().st_size))


def _load_table_csv(path: Path, *, source: str, experiment: str, table: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "table", table)
    df.insert(0, "experiment", experiment)
    df.insert(0, "source", source)
    df["cache_file"] = str(path)
    return df


def _standardize_quality(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "best_accuracy" in out.columns and "best_f1" not in out.columns:
        out["best_f1"] = np.nan
    if "best_balanced_accuracy" not in out.columns:
        out["best_balanced_accuracy"] = np.nan
    for col in [
        "best_balanced_accuracy",
        "best_f1",
        "best_precision",
        "best_recall",
        "best_accuracy",
        "roc_auc",
        "best_threshold",
        "best_direction",
        "balanced_threshold",
        "balanced_direction",
    ]:
        if col not in out.columns:
            out[col] = np.nan
    return out


def sanitize_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = _standardize_quality(df)
    if "table" in out.columns:
        out = out[~out["table"].astype(str).eq("rows_quality")].copy()
    if "metric" in out.columns:
        out = out[~out["metric"].astype(str).isin({"clean_logit", "patched_logit"})].copy()
    if "best_balanced_accuracy" in out.columns:
        out = out[pd.to_numeric(out["best_balanced_accuracy"], errors="coerce").notna()].copy()
    return out.reset_index(drop=True)


def filter_importance_only_metrics(quality: pd.DataFrame, regression: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality = sanitize_quality_table(quality)
    if quality.empty or "metric" not in quality.columns:
        q = pd.DataFrame()
    else:
        metric = quality["metric"].astype(str)
        experiment = quality["experiment"].astype(str) if "experiment" in quality.columns else pd.Series("", index=quality.index)
        mask = experiment.eq("psnr") & metric.str.match(r"^psnr_top.*_importance_(sum|mean|min|zero_frac)$")
        q = quality[mask].copy()
        if not q.empty:
            q = (
                q.sort_values(["source", "metric", "best_balanced_accuracy", "roc_auc"], ascending=[True, True, False, False], na_position="last")
                .drop_duplicates(["source", "metric"], keep="first")
                .sort_values("best_balanced_accuracy", ascending=False, na_position="last")
                .reset_index(drop=True)
            )

    if regression is None or regression.empty or "metric" not in regression.columns:
        r = pd.DataFrame()
    else:
        r = regression.copy()
        metric = r["metric"].astype(str)
        experiment = r["experiment"].astype(str) if "experiment" in r.columns else pd.Series("", index=r.index)
        mask = experiment.eq("psnr") & metric.str.match(r"^psnr_top.*_importance_(sum|mean|min|zero_frac)$")
        r = r[mask].copy()
        if not r.empty:
            sort_cols = [col for col in ["source", "metric", "main_score", "abs_spearman"] if col in r.columns]
            ascending = [True, True] + [False] * max(0, len(sort_cols) - 2)
            r = (
                r.sort_values(sort_cols, ascending=ascending, na_position="last")
                .drop_duplicates(["source", "metric"], keep="first")
                .sort_values("main_score", ascending=False, na_position="last")
                .reset_index(drop=True)
            )
    return q, r


def _use_plain_accuracy_for_detector_quality(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "source" not in df.columns or "best_accuracy" not in df.columns:
        return df
    out = df.copy()
    detector_mask = out["source"].astype(str).eq("detector")
    if not detector_mask.any():
        return out
    accuracy = pd.to_numeric(out.loc[detector_mask, "best_accuracy"], errors="coerce")
    out.loc[detector_mask, "best_balanced_accuracy"] = accuracy
    if "balanced_accuracy_plain" in out.columns:
        out.loc[detector_mask, "balanced_accuracy_plain"] = accuracy
    return out


def use_plain_accuracy_for_detector_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Public wrapper for notebooks that load quality CSV files directly."""
    return _use_plain_accuracy_for_detector_quality(_standardize_quality(df))


def _standardize_regression(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["pearson", "spearman", "kendall", "abs_pearson", "abs_spearman", "main_score", "n", "coverage"]:
        if col not in out.columns:
            out[col] = np.nan
    if "main_score" in out.columns:
        out["main_score"] = out["abs_spearman"]
    return out


_ROW_META_COLUMNS = {
    "path",
    "success",
    "drop",
    "conf_clean",
    "conf_patch",
    "clean_logit",
    "patched_logit",
    "layer_maps_cache_path",
    "cache_file",
    "source",
    "experiment",
    "table",
}


def _numeric_metric_columns(rows: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in rows.columns:
        if col in _ROW_META_COLUMNS:
            continue
        if pd.api.types.is_bool_dtype(rows[col]):
            continue
        values = pd.to_numeric(rows[col], errors="coerce")
        if values.notna().sum() >= 3 and float(values.std(skipna=True) or 0.0) > 0.0:
            cols.append(col)
    return cols


def _quality_from_rows(rows: pd.DataFrame, *, metric_names: list[str] | None = None) -> pd.DataFrame:
    if rows.empty or "success" not in rows.columns:
        return pd.DataFrame()
    labels = rows["success"].astype(bool).to_numpy()
    if metric_names is None:
        metric_cols = [
            col
            for col in _numeric_metric_columns(rows)
            if col not in {"drop", "conf_clean", "conf_patch", "clean_logit", "patched_logit"}
        ]
    else:
        metric_cols = [col for col in metric_names if col in rows.columns]
    out: list[dict[str, Any]] = []
    for metric in metric_cols:
        scores = pd.to_numeric(rows[metric], errors="coerce").to_numpy(dtype=float)
        best = best_f1_threshold(labels, scores)
        out.append({"metric": metric, "roc_auc": roc_auc_score_manual(labels, scores), **best})
    return pd.DataFrame(out)


def _detector_cache_files(cache_dir: Path | None) -> list[Path]:
    if cache_dir is None or not cache_dir.exists():
        return []
    patterns = [
        "success_failure_metrics_*.pkl",
        "segmentig_success_failure_metrics_*.pkl",
        "spread_vs_precision_*.pkl",
        "psnr_importance_top_metrics_*.pkl",
        "psnr_importance_percent_sweep_*.pkl",
        "binmetrics_*.pkl",
        "covariance_split_metrics/*.pkl",
        "covariance_threshold_sweep_focus/*.pkl",
        "importance_energy_focus_metrics/*.pkl",
    ]
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(cache_dir.glob(pattern)):
            if path not in seen:
                files.append(path)
                seen.add(path)
    return files


def _source_family_from_name(path: Path) -> str:
    name = path.name
    if name.startswith("spread_vs_precision"):
        return "spread_vs_precision"
    if name.startswith("psnr_importance"):
        return "psnr"
    if name.startswith("binmetrics"):
        return "binmetrics"
    if name.startswith("covariance"):
        return "covariance"
    if name.startswith("importance_energy_focus"):
        return "covariance_focus"
    if name.startswith("segmentig") or name.startswith("success_failure"):
        return "segmentig"
    return name.rsplit("_", 1)[0]


def _rows_are_coco_people(rows: pd.DataFrame) -> bool:
    if rows.empty or "path" not in rows.columns:
        return False
    paths = rows["path"].dropna().astype(str)
    if paths.empty:
        return False
    return bool(paths.str.contains("COCO_people", case=False, regex=False).all())


def load_detector_artifact_tables(roots: ExperimentRoots | None = None) -> dict[str, pd.DataFrame]:
    roots = roots or resolve_roots()
    quality_tables: list[pd.DataFrame] = []
    regression_tables: list[pd.DataFrame] = []
    row_tables: list[pd.DataFrame] = []

    for path in _detector_cache_files(roots.detector_cache):
        try:
            obj = _read_pickle(path)
        except Exception as exc:  # noqa: BLE001
            print(f"skip unreadable detector cache {path}: {exc}")
            continue
        if not isinstance(obj, dict):
            continue
        rows_for_filter = _as_dataframe(obj.get("rows"))
        if not _rows_are_coco_people(rows_for_filter):
            continue
        experiment = _source_family_from_name(path)
        legacy_quality = _as_dataframe(obj.get("quality"))
        legacy_metric_names = (
            legacy_quality["metric"].astype(str).tolist()
            if not legacy_quality.empty and "metric" in legacy_quality.columns
            else None
        )
        for key, bucket in [("quality", quality_tables), ("regression", regression_tables), ("rows", row_tables)]:
            table = _as_dataframe(obj.get(key))
            if table.empty:
                continue
            table.insert(0, "table", key)
            table.insert(0, "experiment", experiment)
            table.insert(0, "source", "detector")
            table["cache_file"] = str(path)
            bucket.append(table)

    return {
        "quality": _use_plain_accuracy_for_detector_quality(_standardize_quality(pd.concat(quality_tables, ignore_index=True))) if quality_tables else pd.DataFrame(),
        "regression": _standardize_regression(pd.concat(regression_tables, ignore_index=True)) if regression_tables else pd.DataFrame(),
        "rows": pd.concat(row_tables, ignore_index=True) if row_tables else pd.DataFrame(),
    }


def load_classifier_artifact_tables(roots: ExperimentRoots | None = None) -> dict[str, pd.DataFrame]:
    roots = roots or resolve_roots()
    output = roots.classifier_output
    quality = [
        _load_table_csv(output / "spread_vs_precision_quality.csv", source="classifier", experiment="spread_vs_precision", table="quality"),
        _load_table_csv(output / "psnr_quality_balanced_accuracy.csv", source="classifier", experiment="psnr", table="quality"),
    ]
    regression = [
        _load_table_csv(output / "spread_vs_precision_drop_regression.csv", source="classifier", experiment="spread_vs_precision", table="regression"),
        _load_table_csv(output / "psnr_drop_regression.csv", source="classifier", experiment="psnr", table="regression"),
    ]
    rows = [
        _load_table_csv(output / "spread_vs_precision_rows.csv", source="classifier", experiment="spread_vs_precision", table="rows"),
        _load_table_csv(output / "psnr_rows.csv", source="classifier", experiment="psnr", table="rows"),
    ]
    row_frames = [df for df in rows if not df.empty]
    return {
        "quality": _standardize_quality(pd.concat([df for df in quality if not df.empty], ignore_index=True)) if any(not df.empty for df in quality) else pd.DataFrame(),
        "regression": _standardize_regression(pd.concat([df for df in regression if not df.empty], ignore_index=True)) if any(not df.empty for df in regression) else pd.DataFrame(),
        "rows": pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame(),
    }


def _load_saved_all_metrics(roots: ExperimentRoots) -> dict[str, pd.DataFrame] | None:
    quality_path = roots.comparison_output / "all_metrics_quality.csv"
    regression_path = roots.comparison_output / "all_metrics_regression.csv"
    rows_path = roots.comparison_output / "all_metrics_rows.csv"
    if not quality_path.exists() or not regression_path.exists() or not rows_path.exists():
        return None
    quality = pd.read_csv(quality_path, low_memory=False)
    regression = pd.read_csv(regression_path, low_memory=False)
    rows = pd.read_csv(rows_path, low_memory=False)
    quality = _use_plain_accuracy_for_detector_quality(_standardize_quality(quality))
    return {"quality": quality, "regression": regression, "rows": rows, "roots": roots, "loaded_from_saved_tables": True}


def collect_all_metrics(repo_root: str | Path | None = None, *, force_rebuild: bool = False) -> dict[str, pd.DataFrame]:
    roots = resolve_roots(repo_root)
    if not force_rebuild:
        saved = _load_saved_all_metrics(roots)
        if saved is not None:
            return saved

    classifier = load_classifier_artifact_tables(roots)
    detector = load_detector_artifact_tables(roots)
    quality_parts = [classifier["quality"], detector["quality"]]
    regression_parts = [classifier["regression"], detector["regression"]]
    rows_parts = [classifier["rows"], detector["rows"]]

    quality = pd.concat([df for df in quality_parts if not df.empty], ignore_index=True) if any(not df.empty for df in quality_parts) else pd.DataFrame()
    regression = pd.concat([df for df in regression_parts if not df.empty], ignore_index=True) if any(not df.empty for df in regression_parts) else pd.DataFrame()
    rows = pd.concat([df for df in rows_parts if not df.empty], ignore_index=True) if any(not df.empty for df in rows_parts) else pd.DataFrame()

    if not quality.empty:
        quality = quality[pd.to_numeric(quality["best_balanced_accuracy"], errors="coerce").notna()].copy()
        sort_cols = [col for col in ["best_balanced_accuracy", "best_f1", "best_accuracy", "roc_auc"] if col in quality.columns]
        quality = quality.sort_values(sort_cols, ascending=False, na_position="last").reset_index(drop=True)
    if not regression.empty:
        sort_cols = [col for col in ["main_score", "abs_spearman", "abs_pearson"] if col in regression.columns]
        regression = regression.sort_values(sort_cols, ascending=False, na_position="last").reset_index(drop=True)

    return {"quality": quality, "regression": regression, "rows": rows, "roots": roots, "loaded_from_saved_tables": False}


def roc_auc_score_manual(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve_points(labels: np.ndarray, scores: np.ndarray, *, direction: int = 1) -> dict[str, list[float]]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float) * (1 if int(direction) == 1 else -1)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    if len(scores) == 0:
        return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [float("inf"), float("-inf")]}
    thresholds = np.r_[np.inf, np.unique(scores)[::-1], -np.inf]
    pos = max(1, int(labels.sum()))
    neg = max(1, int((~labels).sum()))
    fpr: list[float] = []
    tpr: list[float] = []
    for threshold in thresholds:
        pred = scores >= threshold
        tpr.append(float((pred & labels).sum() / pos))
        fpr.append(float((pred & ~labels).sum() / neg))
    return {"fpr": fpr, "tpr": tpr, "thresholds": [float(v) for v in thresholds]}


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    if len(scores) == 0 or labels.sum() == 0:
        return {
            "best_f1": float("nan"),
            "best_precision": float("nan"),
            "best_recall": float("nan"),
            "best_threshold": float("nan"),
            "best_direction": 1,
            "best_accuracy": float("nan"),
            "best_balanced_accuracy": float("nan"),
            "balanced_threshold": float("nan"),
            "balanced_direction": 1,
        }
    n = float(len(scores))
    pos = float(labels.sum())
    neg = float((~labels).sum())
    if neg == 0:
        neg = 0.0

    best: dict[str, float | int] = {
        "best_f1": -1.0,
        "best_precision": 0.0,
        "best_recall": 0.0,
        "best_threshold": float(scores[0]),
        "best_direction": 1,
        "best_accuracy": 0.0,
    }
    best_balanced: dict[str, float | int] = {
        "best_balanced_accuracy": -1.0,
        "balanced_precision": 0.0,
        "balanced_recall": 0.0,
        "balanced_specificity": 0.0,
        "balanced_f1": 0.0,
        "balanced_accuracy_plain": 0.0,
        "balanced_threshold": float(scores[0]),
        "balanced_direction": 1,
    }

    def _update_from_arrays(
        *,
        thresholds: np.ndarray,
        tp: np.ndarray,
        fp: np.ndarray,
        fn: np.ndarray,
        tn: np.ndarray,
        direction: int,
    ) -> None:
        nonlocal best, best_balanced
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
        specificity = np.divide(tn, tn + fp, out=np.zeros_like(tp, dtype=float), where=(tn + fp) > 0)
        f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) > 0)
        accuracy = (tp + tn) / max(1.0, n)
        balanced_accuracy = 0.5 * (recall + specificity)

        f1_idx = int(np.nanargmax(f1))
        if float(f1[f1_idx]) > float(best["best_f1"]):
            best = {
                "best_f1": float(f1[f1_idx]),
                "best_precision": float(precision[f1_idx]),
                "best_recall": float(recall[f1_idx]),
                "best_threshold": float(thresholds[f1_idx]),
                "best_direction": int(direction),
                "best_accuracy": float(accuracy[f1_idx]),
            }

        order = np.lexsort((accuracy, recall, precision, f1, balanced_accuracy))
        ba_idx = int(order[-1])
        if (float(balanced_accuracy[ba_idx]), float(f1[ba_idx]), float(precision[ba_idx]), float(recall[ba_idx]), float(accuracy[ba_idx])) > (
            float(best_balanced["best_balanced_accuracy"]),
            float(best_balanced["balanced_f1"]),
            float(best_balanced["balanced_precision"]),
            float(best_balanced["balanced_recall"]),
            float(best_balanced["balanced_accuracy_plain"]),
        ):
            best_balanced = {
                "best_balanced_accuracy": float(balanced_accuracy[ba_idx]),
                "balanced_precision": float(precision[ba_idx]),
                "balanced_recall": float(recall[ba_idx]),
                "balanced_specificity": float(specificity[ba_idx]),
                "balanced_f1": float(f1[ba_idx]),
                "balanced_accuracy_plain": float(accuracy[ba_idx]),
                "balanced_threshold": float(thresholds[ba_idx]),
                "balanced_direction": int(direction),
            }

    # direction=1: predict success when score >= threshold.
    desc_order = np.argsort(scores)[::-1]
    desc_scores = scores[desc_order]
    desc_labels = labels[desc_order]
    group_ends = np.r_[np.flatnonzero(desc_scores[1:] != desc_scores[:-1]), len(desc_scores) - 1]
    tp_ge = np.cumsum(desc_labels)[group_ends].astype(float)
    pred_ge = (group_ends + 1).astype(float)
    fp_ge = pred_ge - tp_ge
    fn_ge = pos - tp_ge
    tn_ge = neg - fp_ge
    _update_from_arrays(thresholds=desc_scores[group_ends], tp=tp_ge, fp=fp_ge, fn=fn_ge, tn=tn_ge, direction=1)

    # direction=-1: predict success when score <= threshold.
    asc_order = np.argsort(scores)
    asc_scores = scores[asc_order]
    asc_labels = labels[asc_order]
    group_ends = np.r_[np.flatnonzero(asc_scores[1:] != asc_scores[:-1]), len(asc_scores) - 1]
    tp_le = np.cumsum(asc_labels)[group_ends].astype(float)
    pred_le = (group_ends + 1).astype(float)
    fp_le = pred_le - tp_le
    fn_le = pos - tp_le
    tn_le = neg - fp_le
    _update_from_arrays(thresholds=asc_scores[group_ends], tp=tp_le, fp=fp_le, fn=fn_le, tn=tn_le, direction=-1)

    best.update(best_balanced)
    return best


def _apply_threshold_metrics(labels: np.ndarray, scores: np.ndarray, *, threshold: float, direction: int) -> dict[str, float | int]:
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    if len(scores) == 0:
        return {
            "best_balanced_accuracy": float("nan"),
            "balanced_precision": float("nan"),
            "balanced_recall": float("nan"),
            "balanced_specificity": float("nan"),
            "balanced_f1": float("nan"),
            "balanced_accuracy_plain": float("nan"),
            "best_accuracy": float("nan"),
            "best_f1": float("nan"),
            "best_precision": float("nan"),
            "best_recall": float("nan"),
            "best_threshold": float(threshold),
            "best_direction": int(direction),
            "balanced_threshold": float(threshold),
            "balanced_direction": int(direction),
        }
    pred = scores >= threshold if direction == 1 else scores <= threshold
    tp = float(np.sum(pred & labels))
    fp = float(np.sum(pred & ~labels))
    fn = float(np.sum(~pred & labels))
    tn = float(np.sum(~pred & ~labels))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(1.0, tp + fp + fn + tn)
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "best_balanced_accuracy": float(balanced_accuracy),
        "balanced_precision": float(precision),
        "balanced_recall": float(recall),
        "balanced_specificity": float(specificity),
        "balanced_f1": float(f1),
        "balanced_accuracy_plain": float(accuracy),
        "best_accuracy": float(accuracy),
        "best_f1": float(f1),
        "best_precision": float(precision),
        "best_recall": float(recall),
        "best_threshold": float(threshold),
        "best_direction": int(direction),
        "balanced_threshold": float(threshold),
        "balanced_direction": int(direction),
    }


def _corr(a: np.ndarray, b: np.ndarray, *, spearman: bool = False) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if spearman:
        a = pd.Series(a).rank(method="average").to_numpy()
        b = pd.Series(b).rank(method="average").to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _find_metric_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _load_l1_l2_source_rows(source: str, roots: ExperimentRoots) -> tuple[pd.DataFrame, Path | None]:
    if source == "classifier":
        path = roots.classifier_output / "spread_vs_precision_rows.csv"
        if not path.exists():
            return pd.DataFrame(), None
        return pd.read_csv(path), path

    if roots.detector_cache is None:
        return pd.DataFrame(), None
    cache = _latest_file(sorted(roots.detector_cache.glob("spread_vs_precision_*.pkl")))
    if cache is None:
        return pd.DataFrame(), None
    obj = _read_pickle(cache)
    return _as_dataframe(obj.get("rows") if isinstance(obj, dict) else None), cache


def _build_l1_l2_for_source(source: str, roots: ExperimentRoots) -> dict[str, Any]:
    rows, source_path = _load_l1_l2_source_rows(source, roots)
    warnings: list[str] = []
    if rows.empty:
        warnings.append(f"{source}: spread_vs_precision rows not found")
        return {"source": source, "rows": pd.DataFrame(), "quality": pd.DataFrame(), "regression": pd.DataFrame(), "warnings": warnings, "source_path": source_path}

    l1_col = _find_metric_column(rows, ["spread_delta_abs_mean", "delta_abs_mean", "delta_abs_mean_no_patch"])
    l2_col = _find_metric_column(rows, ["spread_delta_l2_rms", "delta_l2_rms", "delta_l2_rms_no_patch"])
    if l1_col is None or l2_col is None:
        warnings.append(f"{source}: cannot find L1/L2 columns in spread rows")
        return {"source": source, "rows": pd.DataFrame(), "quality": pd.DataFrame(), "regression": pd.DataFrame(), "warnings": warnings, "source_path": source_path}

    required = ["success", "drop", l1_col, l2_col]
    out = rows[[col for col in ["path", *required] if col in rows.columns]].copy()
    out = out.rename(columns={l1_col: "delta_l1_mean_abs", l2_col: "delta_l2_rms"})
    out["success"] = out["success"].astype(bool)
    out["source"] = source
    out["l1_column"] = l1_col
    out["l2_column"] = l2_col

    q_rows: list[dict[str, Any]] = []
    r_rows: list[dict[str, Any]] = []
    for metric in ["delta_l1_mean_abs", "delta_l2_rms"]:
        scores = out[metric].to_numpy(dtype=float)
        labels = out["success"].to_numpy(dtype=bool)
        best = best_f1_threshold(labels, scores)
        q_rows.append({"source": source, "experiment": "l1_vs_l2", "metric": metric, "roc_auc": roc_auc_score_manual(labels, scores), **best})
        drop = out["drop"].to_numpy(dtype=float)
        pearson = _corr(scores, drop)
        spearman = _corr(scores, drop, spearman=True)
        r_rows.append({
            "source": source,
            "experiment": "l1_vs_l2",
            "metric": metric,
            "n": int(np.isfinite(scores).sum()),
            "pearson": pearson,
            "spearman": spearman,
            "abs_pearson": abs(pearson) if np.isfinite(pearson) else np.nan,
            "abs_spearman": abs(spearman) if np.isfinite(spearman) else np.nan,
            "main_score": abs(spearman) if np.isfinite(spearman) else np.nan,
        })

    return {
        "source": source,
        "rows": out,
        "quality": pd.DataFrame(q_rows),
        "regression": pd.DataFrame(r_rows),
        "warnings": warnings,
        "source_path": source_path,
    }


def compute_or_load_l1_l2_comparison(repo_root: str | Path | None = None, *, force: bool = False) -> dict[str, Any]:
    roots = resolve_roots(repo_root)
    outputs: dict[str, Any] = {}
    warnings: list[str] = []
    for source, cache_dir in [("classifier", roots.classifier_cache), ("detector", roots.detector_cache)]:
        source_rows, source_path = _load_l1_l2_source_rows(source, roots)
        payload = {
            "method": "l1_l2_spread_rows_v3_balanced_accuracy_spearman_main_score",
            "source": source,
            "source_path": str(source_path),
            "mtime_ns": source_path.stat().st_mtime_ns if source_path and source_path.exists() else None,
            "size": source_path.stat().st_size if source_path and source_path.exists() else None,
            "columns": list(source_rows.columns) if not source_rows.empty else [],
        }
        key = _cache_key(payload)
        cache_path = (cache_dir or roots.comparison_output / "cache") / f"l1_l2_comparison_{key}.pkl"
        if cache_path.exists() and not force:
            result = _read_pickle(cache_path)
            result["loaded_from_cache"] = True
        else:
            result = _build_l1_l2_for_source(source, roots)
            result["loaded_from_cache"] = False
            result["cache_path"] = str(cache_path)
            _write_pickle(cache_path, result)
        outputs[source] = result
        warnings.extend(result.get("warnings", []))

    row_parts = [outputs[s]["rows"] for s in outputs if not outputs[s]["rows"].empty]
    quality_parts = [outputs[s]["quality"] for s in outputs if not outputs[s]["quality"].empty]
    regression_parts = [outputs[s]["regression"] for s in outputs if not outputs[s]["regression"].empty]
    return {
        "roots": roots,
        "sources": outputs,
        "rows": pd.concat(row_parts, ignore_index=True) if row_parts else pd.DataFrame(),
        "quality": pd.concat(quality_parts, ignore_index=True) if quality_parts else pd.DataFrame(),
        "regression": pd.concat(regression_parts, ignore_index=True) if regression_parts else pd.DataFrame(),
        "warnings": warnings,
    }


def _resolve_existing_path(path_value: Any, roots: ExperimentRoots) -> Path | None:
    if path_value is None or (isinstance(path_value, float) and not np.isfinite(path_value)):
        return None
    raw = str(path_value)
    candidates = [Path(raw)]
    if not Path(raw).is_absolute():
        candidates.extend(
            [
                roots.repo_root / raw,
                roots.repo_root / "new_experiments" / raw,
                roots.repo_root / raw.replace("new_experiments/outputs/", "new_experiments/new_experiments/outputs/"),
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _classifier_layer_map_manifest(roots: ExperimentRoots, *, layer_name: str = "model.9") -> dict[str, Path]:
    layer_dir = roots.classifier_cache / "layer_maps" / layer_name.replace(".", "_")
    manifest: dict[str, Path] = {}
    if not layer_dir.exists():
        return manifest
    files = sorted(layer_dir.glob("*.npz"))
    iterator = tqdm(files, desc=f"classifier {layer_name} layer-map manifest", leave=False) if tqdm is not None else files
    for path in iterator:
        try:
            with np.load(path, allow_pickle=True) as data:
                if "metadata" not in data.files:
                    continue
                meta = json.loads(str(data["metadata"].item()))
        except Exception:  # noqa: BLE001
            continue
        image_path = str(meta.get("path", ""))
        if image_path:
            manifest[image_path] = path.resolve()
            manifest[Path(image_path).name] = path.resolve()
    return manifest


def _load_sparse_source_rows(source: str, roots: ExperimentRoots, *, layer_name: str = "model.9") -> tuple[pd.DataFrame, Path | None, list[str]]:
    warnings: list[str] = []
    if source == "classifier":
        rows_path = roots.classifier_output / "spread_vs_precision_rows.csv"
        if not rows_path.exists():
            return pd.DataFrame(), None, [f"classifier rows not found: {rows_path}"]
        rows = pd.read_csv(rows_path)
        manifest = _classifier_layer_map_manifest(roots, layer_name=layer_name)
        if not manifest:
            warnings.append(f"classifier layer maps not found for {layer_name}")
        rows["layer_maps_cache_path"] = [
            str(manifest.get(str(path)) or manifest.get(Path(str(path)).name) or "")
            for path in rows.get("path", pd.Series(dtype=str))
        ]
        return rows, rows_path, warnings

    if roots.detector_cache is None:
        return pd.DataFrame(), None, ["detector cache directory not found"]
    cache_path = _latest_file(sorted(roots.detector_cache.glob("spread_vs_precision_*.pkl")))
    if cache_path is None:
        return pd.DataFrame(), None, [f"detector spread_vs_precision cache not found in {roots.detector_cache}"]
    obj = _read_pickle(cache_path)
    rows = _as_dataframe(obj.get("rows") if isinstance(obj, dict) else None)
    if rows.empty:
        return pd.DataFrame(), cache_path, [f"detector rows are empty in {cache_path}"]
    rows["layer_maps_cache_path"] = [str(_resolve_existing_path(value, roots) or "") for value in rows["layer_maps_cache_path"]]
    missing = int(rows["layer_maps_cache_path"].eq("").sum())
    if missing:
        warnings.append(f"detector missing layer maps: {missing} / {len(rows)}")
    return rows, cache_path, warnings


def _load_delta_importance(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=True) as data:
            if "delta_chw" not in data.files:
                return None
            delta = np.asarray(data["delta_chw"], dtype=np.float32)
            if "importance_chw" in data.files:
                importance = np.asarray(data["importance_chw"], dtype=np.float32)
            elif "segmentig_chw" in data.files:
                importance = np.asarray(data["segmentig_chw"], dtype=np.float32)
            else:
                importance = np.abs(delta)
    except Exception:  # noqa: BLE001
        return None
    if delta.shape != importance.shape:
        return None
    return delta.reshape(-1), importance.reshape(-1)


def _load_importance_only(path: Path) -> np.ndarray | None:
    try:
        with np.load(path, allow_pickle=True) as data:
            if "importance_chw" in data.files:
                importance = np.asarray(data["importance_chw"], dtype=np.float32)
            elif "segmentig_chw" in data.files:
                importance = np.asarray(data["segmentig_chw"], dtype=np.float32)
            else:
                return None
    except Exception:  # noqa: BLE001
        return None
    return np.abs(importance.reshape(-1).astype(np.float64, copy=False))


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    values = np.abs(values)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    sorted_values = np.sort(values)
    n = sorted_values.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * sorted_values) / (n * total)) - ((n + 1.0) / n))


def _importance_only_features_for_map(
    importance: np.ndarray,
    *,
    top_percents: tuple[float, ...],
    eps: float = 1e-12,
) -> dict[str, float]:
    importance = np.asarray(importance, dtype=np.float64)
    importance = np.abs(importance[np.isfinite(importance)])
    if importance.size == 0:
        return {}
    n = int(importance.size)
    total = float(importance.sum())
    max_value = float(importance.max())
    if total <= 0.0 or max_value <= 0.0:
        mass = np.full(n, 1.0 / n, dtype=np.float64)
        unit = np.zeros(n, dtype=np.float64)
    else:
        mass = importance / total
        unit = importance / max_value
    order = np.argsort(mass)[::-1]
    ranked_mass = mass[order]
    ranked_unit = unit[order]
    entropy = float(-np.sum(mass * np.log(np.maximum(mass, eps))))
    entropy_norm = entropy / np.log(n) if n > 1 else 0.0
    out: dict[str, float] = {
        "importance_only_gini": _gini(importance),
        "importance_only_entropy_norm": float(entropy_norm),
        "importance_only_effective_neuron_frac": float(np.exp(entropy) / n),
        "importance_only_max_to_mean": float(max_value / (float(importance.mean()) + eps)),
        "importance_only_p99_to_median": float(np.quantile(importance, 0.99) / (float(np.median(importance)) + eps)),
        "importance_only_top1_neuron_mass_frac": float(ranked_mass[0]),
        "importance_only_top10_neuron_mass_frac": float(ranked_mass[: min(10, n)].sum()),
    }
    psnr_values = np.log10(1.0 / np.maximum(ranked_unit * ranked_unit, eps))
    for percent in top_percents:
        k = max(1, min(n, int(np.ceil(n * float(percent) / 100.0))))
        label = _percent_label(float(percent))
        top_mass = ranked_mass[:k]
        top_unit = ranked_unit[:k]
        top_psnr = psnr_values[:k]
        out[f"importance_top{label}pct_mass_frac"] = float(top_mass.sum())
        out[f"importance_top{label}pct_mean_unit"] = float(top_unit.mean())
        out[f"importance_top{label}pct_min_unit"] = float(top_unit.min())
        out[f"importance_top{label}pct_sum_unit"] = float(top_unit.sum())
        out[f"importance_psnr_top{label}pct_sum"] = float(top_psnr.sum())
        out[f"importance_psnr_top{label}pct_mean"] = float(top_psnr.mean())
        out[f"importance_psnr_top{label}pct_min"] = float(top_psnr.min())
        out[f"importance_top{label}pct_k"] = float(k)
    return out


def _sparse_features_for_map(delta: np.ndarray, importance: np.ndarray, *, top_counts: tuple[int, ...]) -> dict[str, float]:
    abs_delta = np.abs(delta)
    abs_importance = np.abs(importance)
    order = np.argsort(abs_importance)[::-1]
    out: dict[str, float] = {
        "all_delta_abs_p99": float(np.quantile(abs_delta, 0.99)),
        "all_delta_abs_p999": float(np.quantile(abs_delta, 0.999)),
        "all_delta_abs_max": float(np.max(abs_delta)),
    }
    top10_abs: np.ndarray | None = None
    rank_limit = int(max(top_counts))
    rank_abs = abs_delta[order[: min(rank_limit, len(order))]]
    for rank, value in enumerate(rank_abs, start=1):
        out[f"top_rank{rank}_abs_delta"] = float(value)
    for requested_k in top_counts:
        k = int(max(1, min(requested_k, len(order))))
        top_abs = abs_delta[order[:k]]
        top_signed = delta[order[:k]]
        if k == 10:
            top10_abs = top_abs
        out[f"top{k}_max_abs_delta"] = float(np.max(top_abs))
        out[f"top{k}_min_abs_delta"] = float(np.min(top_abs))
        out[f"top{k}_mean_abs_delta"] = float(np.mean(top_abs))
        out[f"top{k}_signed_max_delta"] = float(np.max(top_signed))
        out[f"top{k}_signed_min_delta"] = float(np.min(top_signed))
        out[f"top{k}_abs_delta_cv"] = float(np.std(top_abs) / (np.mean(top_abs) + 1e-12))
    if top10_abs is None:
        k = int(max(1, min(10, len(order))))
        top10_abs = abs_delta[order[:k]]
    out["top1_to_top10_mean_abs_ratio"] = float(out["top1_max_abs_delta"] / (np.mean(top10_abs) + 1e-12))
    out["top3_to_top10_mean_abs_ratio"] = float(out.get("top3_mean_abs_delta", np.mean(top10_abs)) / (np.mean(top10_abs) + 1e-12))
    return out


def _percent_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _sparse_percent_features_for_map(delta: np.ndarray, importance: np.ndarray, *, top_percents: tuple[float, ...]) -> dict[str, float]:
    abs_delta = np.abs(delta)
    abs_importance = np.abs(importance)
    order = np.argsort(abs_importance)[::-1]
    n_neurons = max(1, len(order))
    out: dict[str, float] = {}
    for percent in top_percents:
        k = max(1, min(n_neurons, int(np.ceil(n_neurons * float(percent) / 100.0))))
        label = _percent_label(float(percent))
        top_abs = abs_delta[order[:k]]
        out[f"top{label}pct_max_abs_delta"] = float(np.max(top_abs))
        out[f"top{label}pct_min_abs_delta"] = float(np.min(top_abs))
        out[f"top{label}pct_mean_abs_delta"] = float(np.mean(top_abs))
        out[f"top{label}pct_abs_delta_cv"] = float(np.std(top_abs) / (np.mean(top_abs) + 1e-12))
        out[f"top{label}pct_k"] = float(k)
    return out


def _build_sparse_rows_for_source(
    source: str,
    roots: ExperimentRoots,
    *,
    layer_name: str,
    top_counts: tuple[int, ...],
    top_percents: tuple[float, ...],
    max_examples: int | None,
) -> tuple[pd.DataFrame, Path | None, list[str]]:
    rows, source_path, warnings = _load_sparse_source_rows(source, roots, layer_name=layer_name)
    if rows.empty:
        return pd.DataFrame(), source_path, warnings
    needed = [col for col in ["path", "success", "drop", "conf_clean", "conf_patch", "layer_maps_cache_path"] if col in rows.columns]
    rows = rows[needed].copy()
    rows = rows[rows["layer_maps_cache_path"].astype(str).ne("")]
    if max_examples is not None:
        rows = rows.head(int(max_examples))
    feature_rows: list[dict[str, Any]] = []
    skipped = 0
    iterator = tqdm(rows.itertuples(index=False), total=len(rows), desc=f"{source} sparse maps", leave=False) if tqdm is not None else rows.itertuples(index=False)
    for item in iterator:
        row = item._asdict()
        path = Path(str(row["layer_maps_cache_path"]))
        loaded = _load_delta_importance(path)
        if loaded is None:
            skipped += 1
            continue
        delta, importance = loaded
        feature = _sparse_features_for_map(delta, importance, top_counts=top_counts)
        feature.update(_sparse_percent_features_for_map(delta, importance, top_percents=top_percents))
        feature.update(
            {
                "source": source,
                "path": row.get("path"),
                "success": bool(row.get("success")),
                "drop": float(row.get("drop", np.nan)),
                "conf_clean": float(row.get("conf_clean", np.nan)),
                "conf_patch": float(row.get("conf_patch", np.nan)),
                "layer_maps_cache_path": str(path),
            }
        )
        feature_rows.append(feature)
    if skipped:
        warnings.append(f"{source} skipped invalid layer maps: {skipped}")
    return pd.DataFrame(feature_rows), source_path, warnings


def _sparse_source_signature(source: str, roots: ExperimentRoots, *, layer_name: str) -> dict[str, Any]:
    if source == "classifier":
        rows_path = roots.classifier_output / "spread_vs_precision_rows.csv"
        layer_dir = roots.classifier_cache / "layer_maps" / layer_name.replace(".", "_")
        layer_files = list(layer_dir.glob("*.npz")) if layer_dir.exists() else []
        latest_layer_mtime = max((p.stat().st_mtime_ns for p in layer_files), default=None)
        return {
            "source_path": str(rows_path),
            "mtime_ns": rows_path.stat().st_mtime_ns if rows_path.exists() else None,
            "size": rows_path.stat().st_size if rows_path.exists() else None,
            "layer_dir": str(layer_dir),
            "layer_map_count": len(layer_files),
            "latest_layer_mtime_ns": latest_layer_mtime,
        }
    cache_path = _latest_file(sorted(roots.detector_cache.glob("spread_vs_precision_*.pkl"))) if roots.detector_cache else None
    layer_dir = roots.detector_cache / "layer_maps" if roots.detector_cache else None
    layer_files = list(layer_dir.glob("*.npz")) if layer_dir and layer_dir.exists() else []
    latest_layer_mtime = max((p.stat().st_mtime_ns for p in layer_files), default=None)
    return {
        "source_path": str(cache_path),
        "mtime_ns": cache_path.stat().st_mtime_ns if cache_path and cache_path.exists() else None,
        "size": cache_path.stat().st_size if cache_path and cache_path.exists() else None,
        "layer_dir": str(layer_dir),
        "layer_map_count": len(layer_files),
        "latest_layer_mtime_ns": latest_layer_mtime,
    }


def _stratified_train_test_split(labels: np.ndarray, *, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(bool)
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for value in (False, True):
        idx = np.flatnonzero(labels == value)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        n_test = min(max(n_test, 1 if len(idx) > 1 else 0), max(0, len(idx) - 1))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def _fit_count_threshold_metric(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    test_values: np.ndarray,
    test_labels: np.ndarray,
    *,
    k: int,
) -> dict[str, float | int]:
    values = np.asarray(train_values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    delta_thresholds = np.unique(np.quantile(finite, np.linspace(0.1, 0.95, 18)))
    best_train: dict[str, Any] | None = None
    best_test: dict[str, Any] | None = None
    for delta_threshold in delta_thresholds:
        train_counts = np.sum(np.asarray(train_values, dtype=float) > delta_threshold, axis=1).astype(float)
        test_counts = np.sum(np.asarray(test_values, dtype=float) > delta_threshold, axis=1).astype(float)
        fit = best_f1_threshold(train_labels, train_counts)
        applied = _apply_threshold_metrics(
            test_labels,
            test_counts,
            threshold=float(fit["balanced_threshold"]),
            direction=int(fit["balanced_direction"]),
        )
        train_ba = float(fit.get("best_balanced_accuracy", np.nan))
        if best_train is None or train_ba > float(best_train.get("best_balanced_accuracy", -np.inf)):
            best_train = dict(fit)
            best_train["activation_delta_threshold"] = float(delta_threshold)
            best_test = dict(applied)
    if best_train is None or best_test is None:
        return {}
    out = {f"train_{key}": value for key, value in best_train.items()}
    out.update(best_test)
    return out


def _evaluate_sparse_quality(
    rows: pd.DataFrame,
    *,
    top_counts: tuple[int, ...],
    top_percents: tuple[float, ...],
    seed: int,
    test_size: float,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    quality_rows: list[dict[str, Any]] = []
    for source, sub in rows.groupby("source", sort=True):
        labels = sub["success"].astype(bool).to_numpy()
        if labels.sum() == 0 or (~labels).sum() == 0:
            continue
        train_idx, test_idx = _stratified_train_test_split(labels, test_size=test_size, seed=seed)
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        for k in top_counts:
            max_col = f"top{k}_max_abs_delta"
            min_col = f"top{k}_min_abs_delta"
            mean_col = f"top{k}_mean_abs_delta"
            cv_col = f"top{k}_abs_delta_cv"
            for metric, family in [
                (max_col, "top_window_max"),
                (min_col, "top_window_min"),
                (mean_col, "top_window_mean"),
                (cv_col, "top_window_cv"),
            ]:
                if metric not in sub.columns:
                    continue
                scores = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
                fit = best_f1_threshold(train_labels, scores[train_idx])
                test_metrics = _apply_threshold_metrics(
                    test_labels,
                    scores[test_idx],
                    threshold=float(fit["balanced_threshold"]),
                    direction=int(fit["balanced_direction"]),
                )
                quality_rows.append(
                    {
                        "source": source,
                        "experiment": "sparse_neuron_metrics",
                        "metric": metric,
                        "metric_family": family,
                        "top_k": int(k),
                        "split_policy": f"stratified_seed_{seed}_test_{test_size}",
                        "train_n": int(len(train_idx)),
                        "test_n": int(len(test_idx)),
                        "train_best_balanced_accuracy": float(fit["best_balanced_accuracy"]),
                        "train_balanced_recall": float(fit["balanced_recall"]),
                        "train_balanced_specificity": float(fit["balanced_specificity"]),
                        "roc_auc": roc_auc_score_manual(test_labels, scores[test_idx]),
                        **test_metrics,
                    }
                )
            count_source_col = f"top{k}_mean_abs_delta"
            rank_cols = [f"top_rank{rank}_abs_delta" for rank in range(1, int(k) + 1) if f"top_rank{rank}_abs_delta" in sub.columns]
            if count_source_col in sub.columns and len(rank_cols) == int(k):
                rank_values = sub[rank_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                fitted = _fit_count_threshold_metric(rank_values[train_idx], train_labels, rank_values[test_idx], test_labels, k=int(k))
                if fitted:
                    quality_rows.append(
                        {
                            "source": source,
                            "experiment": "sparse_neuron_metrics",
                            "metric": f"top{k}_count_abs_delta_gt_train_threshold",
                            "metric_family": "affected_count_threshold",
                            "top_k": int(k),
                            "split_policy": f"stratified_seed_{seed}_test_{test_size}",
                            "train_n": int(len(train_idx)),
                            "test_n": int(len(test_idx)),
                            "roc_auc": float("nan"),
                            **fitted,
                        }
                    )
        for percent in top_percents:
            label = _percent_label(float(percent))
            for metric, family in [
                (f"top{label}pct_max_abs_delta", "top_percent_max"),
                (f"top{label}pct_min_abs_delta", "top_percent_min"),
                (f"top{label}pct_mean_abs_delta", "top_percent_mean"),
                (f"top{label}pct_abs_delta_cv", "top_percent_cv"),
            ]:
                if metric not in sub.columns:
                    continue
                scores = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
                fit = best_f1_threshold(train_labels, scores[train_idx])
                test_metrics = _apply_threshold_metrics(
                    test_labels,
                    scores[test_idx],
                    threshold=float(fit["balanced_threshold"]),
                    direction=int(fit["balanced_direction"]),
                )
                quality_rows.append(
                    {
                        "source": source,
                        "experiment": "sparse_neuron_metrics",
                        "metric": metric,
                        "metric_family": family,
                        "top_k": np.nan,
                        "top_percent": float(percent),
                        "split_policy": f"stratified_seed_{seed}_test_{test_size}",
                        "train_n": int(len(train_idx)),
                        "test_n": int(len(test_idx)),
                        "train_best_balanced_accuracy": float(fit["best_balanced_accuracy"]),
                        "train_balanced_recall": float(fit["balanced_recall"]),
                        "train_balanced_specificity": float(fit["balanced_specificity"]),
                        "roc_auc": roc_auc_score_manual(test_labels, scores[test_idx]),
                        **test_metrics,
                    }
                )
        for metric in ["top1_to_top10_mean_abs_ratio", "top3_to_top10_mean_abs_ratio", "all_delta_abs_p99", "all_delta_abs_p999", "all_delta_abs_max"]:
            if metric not in sub.columns:
                continue
            scores = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            fit = best_f1_threshold(train_labels, scores[train_idx])
            test_metrics = _apply_threshold_metrics(
                test_labels,
                scores[test_idx],
                threshold=float(fit["balanced_threshold"]),
                direction=int(fit["balanced_direction"]),
            )
            quality_rows.append(
                {
                    "source": source,
                    "experiment": "sparse_neuron_metrics",
                    "metric": metric,
                    "metric_family": "tail_concentration" if "ratio" in metric else "global_tail",
                    "top_k": 10 if "top10" in metric else np.nan,
                    "split_policy": f"stratified_seed_{seed}_test_{test_size}",
                    "train_n": int(len(train_idx)),
                    "test_n": int(len(test_idx)),
                    "train_best_balanced_accuracy": float(fit["best_balanced_accuracy"]),
                    "train_balanced_recall": float(fit["balanced_recall"]),
                    "train_balanced_specificity": float(fit["balanced_specificity"]),
                    "roc_auc": roc_auc_score_manual(test_labels, scores[test_idx]),
                    **test_metrics,
                }
            )
    return _standardize_quality(pd.DataFrame(quality_rows))


def _sparse_regression(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "drop" not in rows.columns:
        return pd.DataFrame()
    meta = {"source", "path", "success", "drop", "conf_clean", "conf_patch", "layer_maps_cache_path"}
    metric_cols = [col for col in rows.columns if col not in meta and pd.api.types.is_numeric_dtype(rows[col])]
    out: list[dict[str, Any]] = []
    for source, sub in rows.groupby("source", sort=True):
        drop = pd.to_numeric(sub["drop"], errors="coerce").to_numpy(dtype=float)
        for metric in metric_cols:
            scores = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            pearson = _corr(scores, drop)
            spearman = _corr(scores, drop, spearman=True)
            out.append(
                {
                    "source": source,
                    "experiment": "sparse_neuron_metrics",
                    "metric": metric,
                    "n": int(np.isfinite(scores).sum()),
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_pearson": abs(pearson) if np.isfinite(pearson) else np.nan,
                    "abs_spearman": abs(spearman) if np.isfinite(spearman) else np.nan,
                    "main_score": abs(spearman) if np.isfinite(spearman) else np.nan,
                }
            )
    return _standardize_regression(pd.DataFrame(out))


def compute_or_load_sparse_neuron_metrics(
    repo_root: str | Path | None = None,
    *,
    force: bool = False,
    layer_name: str = "model.9",
    top_counts: tuple[int, ...] = (1, 2, 3, 5, 10),
    top_percents: tuple[float, ...] = tuple(np.round(np.arange(0.1, 10.0 + 1e-9, 0.1), 1)),
    seed: int = 17,
    test_size: float = 0.4,
    max_examples: int | None = None,
) -> dict[str, Any]:
    roots = resolve_roots(repo_root)
    source_payloads = {source: _sparse_source_signature(source, roots, layer_name=layer_name) for source in ("classifier", "detector")}
    payload = {
        "method": "sparse_neuron_tail_metrics_v1_train_threshold_test_balanced_accuracy",
        "sources": source_payloads,
        "layer_name": layer_name,
        "top_counts": list(top_counts),
        "top_percents": [float(v) for v in top_percents],
        "seed": seed,
        "test_size": test_size,
        "max_examples": max_examples,
    }
    key = _cache_key(payload)
    cache_path = roots.comparison_output / "cache" / f"sparse_neuron_metrics_{key}.pkl"
    if cache_path.exists() and not force:
        result = _read_pickle(cache_path)
        result["loaded_from_cache"] = True
        return result

    warnings: list[str] = []
    row_parts: list[pd.DataFrame] = []
    for source in ("classifier", "detector"):
        source_rows, _, source_warnings = _build_sparse_rows_for_source(
            source,
            roots,
            layer_name=layer_name,
            top_counts=top_counts,
            top_percents=top_percents,
            max_examples=max_examples,
        )
        warnings.extend(source_warnings)
        if not source_rows.empty:
            row_parts.append(source_rows)
    rows = pd.concat(row_parts, ignore_index=True) if row_parts else pd.DataFrame()
    quality = _evaluate_sparse_quality(rows, top_counts=top_counts, top_percents=top_percents, seed=seed, test_size=test_size)
    regression = _sparse_regression(rows)
    result = {
        "roots": roots,
        "rows": rows,
        "quality": quality,
        "regression": regression,
        "warnings": warnings,
        "cache_path": str(cache_path),
        "loaded_from_cache": False,
        "config": payload,
    }
    _write_pickle(cache_path, result)
    return result


def _importance_metric_family(metric: str) -> str:
    if metric.startswith("importance_psnr_") and metric.endswith("_sum"):
        return "importance_psnr_sum"
    if metric.startswith("importance_psnr_") and metric.endswith("_mean"):
        return "importance_psnr_mean"
    if metric.startswith("importance_psnr_") and metric.endswith("_min"):
        return "importance_psnr_min"
    if metric.endswith("_mass_frac"):
        return "importance_mass"
    if metric.endswith("_mean_unit"):
        return "importance_mean_unit"
    if metric.endswith("_min_unit"):
        return "importance_min_unit"
    if metric.endswith("_sum_unit"):
        return "importance_sum_unit"
    return "importance_concentration"


def _importance_metric_top_percent(metric: str) -> float:
    import re

    match = re.search(r"top([0-9]+(?:p[0-9]+)?)pct", str(metric))
    if not match:
        return float("nan")
    return float(match.group(1).replace("p", "."))


def _quality_and_regression_for_importance_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rows.empty or "success" not in rows.columns:
        return pd.DataFrame(), pd.DataFrame()
    meta = {"source", "path", "success", "drop", "conf_clean", "conf_patch", "layer_maps_cache_path"}
    metric_cols = [
        col
        for col in rows.columns
        if col not in meta
        and pd.api.types.is_numeric_dtype(rows[col])
        and not col.endswith("_k")
    ]
    quality_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    for source, sub in rows.groupby("source", sort=True):
        labels = sub["success"].astype(bool).to_numpy()
        drop = pd.to_numeric(sub["drop"], errors="coerce").to_numpy(dtype=float) if "drop" in sub.columns else np.full(len(sub), np.nan)
        for metric in metric_cols:
            scores = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            best = best_f1_threshold(labels, scores)
            quality_rows.append(
                {
                    "source": source,
                    "experiment": "importance_only_tensor",
                    "metric": metric,
                    "metric_family": _importance_metric_family(metric),
                    "top_percent": _importance_metric_top_percent(metric),
                    "roc_auc": roc_auc_score_manual(labels, scores),
                    **best,
                }
            )
            pearson = _corr(scores, drop)
            spearman = _corr(scores, drop, spearman=True)
            regression_rows.append(
                {
                    "source": source,
                    "experiment": "importance_only_tensor",
                    "metric": metric,
                    "metric_family": _importance_metric_family(metric),
                    "top_percent": _importance_metric_top_percent(metric),
                    "n": int(np.isfinite(scores).sum()),
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_pearson": abs(pearson) if np.isfinite(pearson) else np.nan,
                    "abs_spearman": abs(spearman) if np.isfinite(spearman) else np.nan,
                    "main_score": abs(spearman) if np.isfinite(spearman) else np.nan,
                }
            )
    return _standardize_quality(pd.DataFrame(quality_rows)), _standardize_regression(pd.DataFrame(regression_rows))


def _build_importance_only_rows_for_source(
    source: str,
    roots: ExperimentRoots,
    *,
    layer_name: str,
    top_percents: tuple[float, ...],
    max_examples: int | None,
) -> tuple[pd.DataFrame, Path | None, list[str]]:
    rows, source_path, warnings = _load_sparse_source_rows(source, roots, layer_name=layer_name)
    if rows.empty:
        return pd.DataFrame(), source_path, warnings
    if source == "detector" and "path" in rows.columns:
        rows = rows[rows["path"].astype(str).str.contains("COCO_people", case=False, regex=False)].copy()
    needed = [col for col in ["path", "success", "drop", "conf_clean", "conf_patch", "layer_maps_cache_path"] if col in rows.columns]
    rows = rows[needed].copy()
    rows = rows[rows["layer_maps_cache_path"].astype(str).ne("")]
    if max_examples is not None:
        rows = rows.head(int(max_examples))
    feature_rows: list[dict[str, Any]] = []
    skipped = 0
    iterator = tqdm(rows.itertuples(index=False), total=len(rows), desc=f"{source} importance-only maps", leave=False) if tqdm is not None else rows.itertuples(index=False)
    for item in iterator:
        row = item._asdict()
        path = Path(str(row["layer_maps_cache_path"]))
        importance = _load_importance_only(path)
        if importance is None:
            skipped += 1
            continue
        feature = _importance_only_features_for_map(importance, top_percents=top_percents)
        if not feature:
            skipped += 1
            continue
        feature.update(
            {
                "source": source,
                "path": row.get("path"),
                "success": bool(row.get("success")),
                "drop": float(row.get("drop", np.nan)),
                "conf_clean": float(row.get("conf_clean", np.nan)),
                "conf_patch": float(row.get("conf_patch", np.nan)),
                "layer_maps_cache_path": str(path),
            }
        )
        feature_rows.append(feature)
    if skipped:
        warnings.append(f"{source} skipped invalid importance maps: {skipped}")
    return pd.DataFrame(feature_rows), source_path, warnings


def compute_or_load_importance_only_tensor_metrics(
    repo_root: str | Path | None = None,
    *,
    force: bool = False,
    layer_name: str = "model.9",
    top_percents: tuple[float, ...] | None = None,
    max_examples: int | None = None,
) -> dict[str, Any]:
    roots = resolve_roots(repo_root)
    if top_percents is None:
        top_percents = tuple(round(float(v), 1) for v in np.arange(0.1, 10.0 + 1e-9, 0.1))
    source_payloads = {source: _sparse_source_signature(source, roots, layer_name=layer_name) for source in ("classifier", "detector")}
    payload = {
        "method": "importance_only_tensor_metrics_v1_no_delta",
        "sources": source_payloads,
        "layer_name": layer_name,
        "top_percents": list(top_percents),
        "max_examples": max_examples,
    }
    key = _cache_key(payload)
    cache_path = roots.comparison_output / "cache" / f"importance_only_tensor_metrics_{key}.pkl"
    if cache_path.exists() and not force:
        result = _read_pickle(cache_path)
        result["loaded_from_cache"] = True
        return result

    warnings: list[str] = []
    row_parts: list[pd.DataFrame] = []
    for source in ("classifier", "detector"):
        source_rows, _, source_warnings = _build_importance_only_rows_for_source(
            source,
            roots,
            layer_name=layer_name,
            top_percents=tuple(top_percents),
            max_examples=max_examples,
        )
        warnings.extend(source_warnings)
        if not source_rows.empty:
            row_parts.append(source_rows)
    rows = pd.concat(row_parts, ignore_index=True) if row_parts else pd.DataFrame()
    quality, regression = _quality_and_regression_for_importance_rows(rows)
    result = {
        "roots": roots,
        "rows": rows,
        "quality": sanitize_quality_table(quality),
        "regression": regression,
        "warnings": warnings,
        "cache_path": str(cache_path),
        "loaded_from_cache": False,
        "config": payload,
    }
    _write_pickle(cache_path, result)
    return result


def plot_importance_only_percent_sweep(quality: pd.DataFrame, *, metric_families: tuple[str, ...] = ("importance_psnr_sum", "importance_mass")):
    fig, axes = plt.subplots(len(metric_families), 1, figsize=(12, 4.2 * len(metric_families)), squeeze=False)
    if quality.empty:
        axes[0, 0].text(0.5, 0.5, "No importance-only quality rows", ha="center", va="center")
        axes[0, 0].axis("off")
        return fig
    for ax, family in zip(axes[:, 0], metric_families):
        sub = quality[quality.get("metric_family", "").astype(str).eq(family)].copy()
        sub = sub[pd.to_numeric(sub.get("top_percent", np.nan), errors="coerce").notna()] if not sub.empty else sub
        if sub.empty:
            ax.text(0.5, 0.5, f"No {family} rows", ha="center", va="center")
            ax.axis("off")
            continue
        sub["top_percent"] = pd.to_numeric(sub["top_percent"], errors="coerce")
        for source, group in sub.groupby("source", sort=True):
            curve = group.groupby("top_percent")["best_balanced_accuracy"].max().sort_index()
            ax.plot(curve.index, curve.values, marker=".", linewidth=1.6, label=str(source))
            best_idx = curve.idxmax()
            best_value = curve.loc[best_idx]
            ax.scatter([best_idx], [best_value], s=48, edgecolor="black", zorder=4)
            ax.text(best_idx, best_value + 0.008, f"{best_idx:.1f}%\n{best_value:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{family}: best balanced accuracy by top-percent boundary")
        ax.set_xlabel("top important neurons, %")
        ax.set_ylabel("best balanced accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    return fig


def save_tables(tables: dict[str, pd.DataFrame], output_dir: Path, *, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            if "quality" in name:
                df = sanitize_quality_table(df)
            df.to_csv(output_dir / f"{prefix}_{name}.csv", index=False)


def plot_l1_l2_scatter(rows: pd.DataFrame):
    sources = list(rows["source"].dropna().unique()) if not rows.empty else []
    fig, axes = plt.subplots(1, max(1, len(sources)), figsize=(6 * max(1, len(sources)), 5), squeeze=False)
    if not sources:
        axes[0, 0].text(0.5, 0.5, "No L1/L2 rows found", ha="center", va="center")
        axes[0, 0].axis("off")
        return fig
    for ax, source in zip(axes[0], sources):
        sub = rows[rows["source"] == source]
        colors = np.where(sub["success"].to_numpy(dtype=bool), "#54A24B", "#E45756")
        ax.scatter(sub["delta_l1_mean_abs"], sub["delta_l2_rms"], c=colors, alpha=0.45, s=12, linewidths=0)
        ax.set_title(source)
        ax.set_xlabel("L1 mean abs delta")
        ax.set_ylabel("L2 RMS delta")
        ax.grid(alpha=0.25)
    return fig


def plot_l1_l2_quality(quality: pd.DataFrame, regression: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    if quality.empty:
        axes[0].text(0.5, 0.5, "No quality rows", ha="center", va="center")
        axes[0].axis("off")
    else:
        score_col = "best_balanced_accuracy" if "best_balanced_accuracy" in quality.columns else "best_f1"
        pivot = quality.pivot_table(index="source", columns="metric", values=score_col, aggfunc="max")
        pivot.plot(kind="bar", ax=axes[0], color=["#4C78A8", "#F58518"])
        _annotate_vertical_bars(axes[0], fontsize=8)
        axes[0].set_title("Best balanced accuracy: success/fail")
        axes[0].set_xlabel("")
        axes[0].set_ylim(0, 1.12)
        axes[0].grid(axis="y", alpha=0.25)
    if regression.empty:
        axes[1].text(0.5, 0.5, "No regression rows", ha="center", va="center")
        axes[1].axis("off")
    else:
        pivot = regression.pivot_table(index="source", columns="metric", values="abs_spearman", aggfunc="max")
        pivot.plot(kind="bar", ax=axes[1], color=["#4C78A8", "#F58518"])
        _annotate_vertical_bars(axes[1], fontsize=8)
        axes[1].set_title("|Spearman| vs drop")
        axes[1].set_xlabel("")
        axes[1].set_ylim(0, 1.12)
        axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_quality_lines(
    quality: pd.DataFrame,
    *,
    families: list[str],
    title: str,
    ylabel: str = "test balanced accuracy",
):
    fig, ax = plt.subplots(figsize=(9, 5))
    source_colors = {"classifier": "#4C78A8", "detector": "#F58518"}
    if quality.empty:
        ax.text(0.5, 0.5, "No quality rows", ha="center", va="center")
        ax.axis("off")
        return fig
    sub = quality[quality["metric_family"].isin(families)].copy() if "metric_family" in quality.columns else pd.DataFrame()
    use_percent = not sub.empty and "top_percent" in sub.columns and pd.to_numeric(sub["top_percent"], errors="coerce").notna().any()
    x_col = "top_percent" if use_percent else "top_k"
    sub = sub[pd.to_numeric(sub.get(x_col, np.nan), errors="coerce").notna()] if not sub.empty else sub
    if sub.empty:
        ax.text(0.5, 0.5, "No matching sparse metrics", ha="center", va="center")
        ax.axis("off")
        return fig
    sub[x_col] = pd.to_numeric(sub[x_col], errors="coerce")
    notes: list[str] = []
    for (source, family), group in sub.groupby(["source", "metric_family"], sort=True):
        grouped = group.groupby(x_col)["best_balanced_accuracy"]
        agg = grouped.max().sort_index()
        sigma = grouped.std(ddof=0).reindex(agg.index).fillna(0.0)
        ax.errorbar(
            agg.index,
            agg.values,
            yerr=sigma.values,
            marker="o",
            markersize=3 if use_percent else 5,
            linewidth=2,
            capsize=2,
            color=source_colors.get(str(source), None),
            label=f"{source} / {family}",
        )
        for x, y, s in zip(agg.index, agg.values, sigma.values):
            if np.isfinite(float(y)):
                if use_percent and float(x) not in {float(agg.index.min()), float(agg.index.max())} and abs((float(x) * 10) % 10) > 1e-9:
                    continue
                label = f"{float(y):.3f}\nСКО {float(s):.3f}" if float(s) > 0.0 else f"{float(y):.3f}"
                ax.text(x, y + 0.012, label, ha="center", va="bottom", fontsize=7 if use_percent else 8)
        curve_sigma = float(np.nanstd(agg.values, ddof=0)) if len(agg) else float("nan")
        final_score = float(agg.iloc[-1]) if len(agg) else float("nan")
        notes.append(f"{source}/{family}: final={final_score:.3f}, СКО={curve_sigma:.3f}")
    ax.set_title(title)
    ax.set_xlabel("top important neurons, %" if use_percent else "top-k important neurons")
    ax.set_ylabel(ylabel)
    ticks = sorted(sub[x_col].unique())
    if use_percent and len(ticks) > 20:
        ticks = [value for value in ticks if abs((float(value) * 10) % 10) < 1e-9]
    ax.set_xticks(ticks)
    ax.set_ylim(0, 1.12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    if notes:
        ax.text(
            0.01,
            0.01,
            "\n".join(notes[:8]),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.86},
        )
    fig.tight_layout()
    return fig


def plot_sparse_count_threshold_quality(quality: pd.DataFrame):
    return _plot_quality_lines(
        quality,
        families=["affected_count_threshold"],
        title="Affected-neuron count: train-selected activation threshold, test score",
    )


def plot_sparse_top_window_quality(quality: pd.DataFrame):
    return _plot_quality_lines(
        quality,
        families=["top_percent_max", "top_percent_min"],
        title="Top-percent max/min delta: train-selected decision threshold, test score",
    )


def plot_sparse_percent_mean_quality(quality: pd.DataFrame):
    return _plot_quality_lines(
        quality,
        families=["top_percent_mean", "top_percent_cv"],
        title="Top-percent mean/CV delta: train-selected decision threshold, test score",
    )


def plot_sparse_extra_quality(quality: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    if quality.empty or "metric_family" not in quality.columns:
        axes[0].text(0.5, 0.5, "No quality rows", ha="center", va="center")
        axes[0].axis("off")
        axes[1].axis("off")
        return fig
    for ax, family, title in [
        (axes[0], "tail_concentration", "Tail concentration ratios"),
        (axes[1], "global_tail", "Global activation-tail diagnostics"),
    ]:
        sub = quality[quality["metric_family"].eq(family)].copy()
        if sub.empty:
            ax.text(0.5, 0.5, f"No {family} rows", ha="center", va="center")
            ax.axis("off")
            continue
        pivot = sub.pivot_table(index="metric", columns="source", values="best_balanced_accuracy", aggfunc="max")
        pivot = pivot.sort_values(list(pivot.columns), ascending=False).head(12)
        pivot.plot(kind="barh", ax=ax, color=["#4C78A8", "#F58518"])
        for patch in ax.patches:
            width = float(patch.get_width())
            if np.isfinite(width):
                ax.text(width + 0.012, patch.get_y() + patch.get_height() / 2.0, f"{width:.3f}", va="center", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("test balanced accuracy")
        ax.set_xlim(0, 1.18)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_sparse_metric_leaderboard(
    quality: pd.DataFrame,
    regression: pd.DataFrame | None = None,
    *,
    existing_quality: pd.DataFrame | None = None,
    top_n: int = 24,
):
    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    parts = []
    if existing_quality is not None and not existing_quality.empty:
        existing = use_plain_accuracy_for_detector_quality(existing_quality)
        existing["metric_origin"] = "existing"
        parts.append(existing)
    if quality is not None and not quality.empty:
        new = quality.copy()
        new["metric_origin"] = "new_sparse"
        parts.append(new)
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if combined.empty:
        axes[0].text(0.5, 0.5, "No quality rows", ha="center", va="center")
        axes[0].axis("off")
    else:
        q = combined.copy()
        q["score"] = pd.to_numeric(q["best_balanced_accuracy"], errors="coerce")
        q["label"] = q["metric_origin"].astype(str) + " / " + q["source"].astype(str) + " / " + q["experiment"].astype(str) + " / " + q["metric"].astype(str)
        q = q.sort_values("score", ascending=False, na_position="last").head(top_n).iloc[::-1]
        y = np.arange(len(q))
        color_map = {"new_sparse": "#54A24B", "existing": "#9D755D"}
        colors = [color_map.get(origin, "#4C78A8") for origin in q["metric_origin"]]
        values = q["score"].to_numpy(dtype=float)
        axes[0].barh(y, values, color=colors)
        axes[0].set_yticks(y, q["label"].tolist())
        labels = [_quality_label(row, "score") for _, row in q.iterrows()]
        _annotate_barh(axes[0], y, values, labels, fontsize=_bar_label_fontsize(len(q)))
        axes[0].set_title("All quality metrics by test balanced accuracy")
        axes[0].set_xlabel("balanced accuracy")
        axes[0].set_xlim(0, 1.28)
        axes[0].grid(axis="x", alpha=0.25)

    if regression is None or regression.empty:
        axes[1].text(0.5, 0.5, "No sparse drop correlations", ha="center", va="center")
        axes[1].axis("off")
    else:
        r = regression.copy()
        r["score"] = pd.to_numeric(r["main_score"], errors="coerce")
        r["label"] = r["source"].astype(str) + " / " + r["metric"].astype(str)
        r = r.sort_values("score", ascending=False, na_position="last").head(top_n).iloc[::-1]
        y = np.arange(len(r))
        values = r["score"].to_numpy(dtype=float)
        axes[1].barh(y, values, color=np.where(r["source"].eq("classifier"), "#4C78A8", "#F58518"))
        axes[1].set_yticks(y, r["label"].tolist())
        labels = [f"{value:.3f}" if np.isfinite(value) else "" for value in values]
        _annotate_barh(axes[1], y, values, labels, fontsize=_bar_label_fontsize(len(r)))
        axes[1].set_title("New sparse metrics: |Spearman| vs drop")
        axes[1].set_xlabel("|Spearman|")
        axes[1].set_xlim(0, 1.12)
        axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_metric_leaderboards(quality: pd.DataFrame, regression: pd.DataFrame, *, top_n: int = 20):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    if quality.empty:
        axes[0].text(0.5, 0.5, "No quality metrics", ha="center", va="center")
        axes[0].axis("off")
    else:
        q = quality.copy()
        if "best_balanced_accuracy" not in q.columns:
            q["best_balanced_accuracy"] = np.nan
        q["score"] = q["best_balanced_accuracy"]
        q["label"] = q["source"] + " / " + q["experiment"].astype(str) + " / " + q["metric"].astype(str)
        q = q.sort_values("score", ascending=False, na_position="last").head(top_n).iloc[::-1]
        y = np.arange(len(q))
        values = q["score"].to_numpy(dtype="float64")
        axes[0].barh(y, values, color=np.where(q["source"].eq("classifier"), "#4C78A8", "#F58518"))
        axes[0].set_yticks(y, q["label"].tolist())
        labels = [_quality_label(row, "score") for _, row in q.iterrows()]
        _annotate_barh(axes[0], y, values, labels, fontsize=_bar_label_fontsize(len(q)))
        axes[0].set_title("Quality leaderboard: best balanced accuracy")
        axes[0].set_xlabel("best balanced accuracy")
        axes[0].set_xlim(0, 1.28)
        axes[0].grid(axis="x", alpha=0.25)
    if regression.empty:
        axes[1].text(0.5, 0.5, "No drop correlations", ha="center", va="center")
        axes[1].axis("off")
    else:
        r = regression.copy()
        if "main_score" not in r.columns:
            r["main_score"] = r["abs_spearman"]
        r["label"] = r["source"] + " / " + r["experiment"].astype(str) + " / " + r["metric"].astype(str)
        r = r.sort_values("main_score", ascending=False, na_position="last").head(top_n).iloc[::-1]
        y = np.arange(len(r))
        values = r["main_score"].to_numpy(dtype="float64")
        axes[1].barh(y, values, color=np.where(r["source"].eq("classifier"), "#4C78A8", "#F58518"))
        axes[1].set_yticks(y, r["label"].tolist())
        labels = [f"{value:.3f}" if np.isfinite(value) else "" for value in values]
        _annotate_barh(axes[1], y, values, labels, fontsize=_bar_label_fontsize(len(r)))
        axes[1].set_title("Drop correlation leaderboard by |Spearman|")
        axes[1].set_xlabel("|Spearman| (main)")
        axes[1].set_xlim(0, 1.12)
        axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def _regression_lookup(regression: pd.DataFrame | None) -> pd.DataFrame:
    if regression is None or regression.empty:
        return pd.DataFrame()
    required = {"source", "experiment", "metric"}
    if not required.issubset(regression.columns):
        return pd.DataFrame()
    out = regression.copy()
    if "abs_spearman" not in out.columns and "spearman" in out.columns:
        out["abs_spearman"] = pd.to_numeric(out["spearman"], errors="coerce").abs()
    out["abs_spearman"] = pd.to_numeric(out.get("abs_spearman", np.nan), errors="coerce")
    out = (
        out.sort_values("abs_spearman", ascending=False, na_position="last")
        .drop_duplicates(["source", "experiment", "metric"], keep="first")
        .reset_index(drop=True)
    )
    return out[["source", "experiment", "metric", "spearman", "abs_spearman"] if "spearman" in out.columns else ["source", "experiment", "metric", "abs_spearman"]]


def plot_quality_metric_triplets(quality: pd.DataFrame, regression: pd.DataFrame | None = None, *, top_n: int = 25):
    fig, ax = plt.subplots(figsize=(13, max(6.5, 0.4 * int(top_n) + 1.8)))
    if quality.empty:
        ax.text(0.5, 0.5, "No quality metrics", ha="center", va="center")
        ax.axis("off")
        return fig

    q = sanitize_quality_table(quality).copy()
    needed = ["best_balanced_accuracy", "best_accuracy", "best_f1"]
    for col in needed:
        if col not in q.columns:
            q[col] = np.nan
        q[col] = pd.to_numeric(q[col], errors="coerce")
    experiment_short = (
        q["experiment"]
        .astype(str)
        .replace(
            {
                "spread_vs_precision": "spread",
                "sparse_neuron_metrics": "sparse",
            }
        )
    )
    q["label"] = q["source"].astype(str) + " | " + experiment_short + " | " + q["metric"].astype(str)
    q = q.sort_values("best_balanced_accuracy", ascending=False, na_position="last").head(int(top_n)).iloc[::-1]
    reg = _regression_lookup(regression)
    if not reg.empty:
        q = q.merge(reg, on=["source", "experiment", "metric"], how="left")
    else:
        q["abs_spearman"] = np.nan
    q["rank"] = np.arange(len(q), dtype=float)

    long = q.melt(
        id_vars=["label", "rank", "source", "experiment", "metric"],
        value_vars=["best_balanced_accuracy", "best_accuracy", "best_f1", "abs_spearman"],
        var_name="score_name",
        value_name="score",
    )
    name_map = {
        "best_balanced_accuracy": "balanced accuracy",
        "best_accuracy": "accuracy",
        "best_f1": "F1",
        "abs_spearman": "|Spearman|",
    }
    long["score_name"] = long["score_name"].map(name_map)
    palette = {
        "balanced accuracy": "#2F6B9A",
        "accuracy": "#E6862A",
        "F1": "#3B8C5A",
        "|Spearman|": "#8E5EA2",
    }

    finite_scores = long["score"].replace([np.inf, -np.inf], np.nan).dropna()
    x_min = max(0.0, float(np.floor((finite_scores.min() - 0.035) * 20.0) / 20.0)) if len(finite_scores) else 0.0
    x_max = min(1.02, float(np.ceil((finite_scores.max() + 0.035) * 20.0) / 20.0)) if len(finite_scores) else 1.0
    if x_max - x_min < 0.15:
        x_min = max(0.0, x_max - 0.15)
    text_x = min(1.015, x_max + 0.012)

    for _, row in q.iterrows():
        values = [row["best_balanced_accuracy"], row["best_accuracy"], row["best_f1"]]
        if np.isfinite(float(row.get("abs_spearman", np.nan))):
            values.append(row["abs_spearman"])
        finite = [float(v) for v in values if np.isfinite(float(v))]
        if len(finite) >= 2:
            ax.plot([min(finite), max(finite)], [row["rank"], row["rank"]], color="#D8D8D8", linewidth=2.0, zorder=1)

    for _, row in q.iterrows():
        bacc = float(row["best_balanced_accuracy"])
        acc = float(row["best_accuracy"])
        f1 = float(row["best_f1"])
        spearman_abs = float(row.get("abs_spearman", np.nan))
        spearman = float(row.get("spearman", np.nan))
        rank = float(row["rank"])
        if np.isfinite(bacc) and np.isfinite(acc) and abs(bacc - acc) < 1e-9:
            ax.plot(
                [bacc],
                [rank],
                marker="o",
                markersize=7.2,
                linestyle="None",
                fillstyle="left",
                markerfacecolor=palette["balanced accuracy"],
                markerfacecoloralt=palette["accuracy"],
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=4,
            )
        else:
            if np.isfinite(bacc):
                ax.scatter([bacc], [rank], s=58, color=palette["balanced accuracy"], edgecolors="white", linewidths=0.7, zorder=3)
            if np.isfinite(acc):
                ax.scatter([acc], [rank], s=58, color=palette["accuracy"], edgecolors="white", linewidths=0.7, zorder=3)
        if np.isfinite(f1):
            ax.scatter([f1], [rank], s=58, color=palette["F1"], edgecolors="white", linewidths=0.7, zorder=3)
        if np.isfinite(spearman_abs):
            ax.scatter([spearman_abs], [rank], s=58, color=palette["|Spearman|"], edgecolors="white", linewidths=0.7, zorder=3)
        if np.isfinite(bacc):
            suffix = f"{bacc:.3f}  ({acc:.3f}/{f1:.3f}"
            if np.isfinite(spearman_abs):
                if np.isfinite(spearman):
                    suffix += f"/S={spearman:+.3f}"
                else:
                    suffix += f"/|S|={spearman_abs:.3f}"
            suffix += ")"
            ax.text(text_x, row["rank"], suffix, va="center", ha="left", fontsize=_bar_label_fontsize(len(q)))

    ax.set_yticks(q["rank"].to_numpy(), q["label"].tolist())
    ax.set_xlim(x_min, min(1.12, text_x + 0.12))
    ax.set_ylim(-0.75, len(q) - 0.25)
    ax.set_xlabel("score")
    ax.set_ylabel("")
    ax.set_title("Quality leaderboard sorted by balanced accuracy", pad=26)
    ax.grid(axis="x", alpha=0.25)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette["balanced accuracy"], markeredgecolor="white", markersize=7.2, label="balanced accuracy"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette["accuracy"], markeredgecolor="white", markersize=7.2, label="accuracy"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette["F1"], markeredgecolor="white", markersize=7.2, label="F1"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette["|Spearman|"], markeredgecolor="white", markersize=7.2, label="|Spearman|"),
    ]
    ax.legend(handles=handles, title="", loc="lower center", bbox_to_anchor=(0.5, 1.055), ncol=4, frameon=False)
    ax.text(
        text_x,
        len(q) - 0.05,
        "bacc (acc/F1/S)",
        va="bottom",
        ha="left",
        fontsize=8,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_correlation_metric_triplets(regression: pd.DataFrame, *, top_n: int = 25):
    fig, ax = plt.subplots(figsize=(13, max(6.5, 0.4 * int(top_n) + 1.8)))
    if regression.empty:
        ax.text(0.5, 0.5, "No regression metrics", ha="center", va="center")
        ax.axis("off")
        return fig

    r = regression.copy()
    for col in ["pearson", "spearman", "kendall"]:
        if col not in r.columns:
            r[col] = np.nan
        r[col] = pd.to_numeric(r[col], errors="coerce")
        r[f"abs_{col}"] = r[col].abs()
    if "abs_spearman" not in r.columns:
        r["abs_spearman"] = r["spearman"].abs()
    r["abs_spearman"] = pd.to_numeric(r["abs_spearman"], errors="coerce")
    r = r[pd.to_numeric(r["abs_spearman"], errors="coerce").notna()].copy()
    experiment_short = (
        r["experiment"]
        .astype(str)
        .replace(
            {
                "spread_vs_precision": "spread",
                "sparse_neuron_metrics": "sparse",
            }
        )
    )
    r["label"] = r["source"].astype(str) + " | " + experiment_short + " | " + r["metric"].astype(str)
    r = r.sort_values("abs_spearman", ascending=False, na_position="last").head(int(top_n)).iloc[::-1]
    r["rank"] = np.arange(len(r), dtype=float)

    long = r.melt(
        id_vars=["label", "rank", "source", "experiment", "metric"],
        value_vars=["abs_pearson", "abs_spearman", "abs_kendall"],
        var_name="corr_name",
        value_name="corr",
    )
    name_map = {"abs_pearson": "|Pearson|", "abs_spearman": "|Spearman|", "abs_kendall": "|Kendall|"}
    long["corr_name"] = long["corr_name"].map(name_map)
    palette = {"|Pearson|": "#2F6B9A", "|Spearman|": "#8E5EA2", "|Kendall|": "#3B8C5A"}

    for _, row in r.iterrows():
        values = [row["abs_pearson"], row["abs_spearman"], row["abs_kendall"]]
        finite = [float(v) for v in values if np.isfinite(float(v))]
        if len(finite) >= 2:
            ax.plot([min(finite), max(finite)], [row["rank"], row["rank"]], color="#D8D8D8", linewidth=2.0, zorder=1)

    for corr_name, group in long.groupby("corr_name", sort=False):
        ax.scatter(group["corr"], group["rank"], s=58, label=corr_name, color=palette.get(corr_name), edgecolors="white", linewidths=0.7, zorder=3)

    text_x = 1.035
    for _, row in r.iterrows():
        p = float(row["abs_pearson"])
        s = float(row["abs_spearman"])
        k = float(row["abs_kendall"])
        ax.text(text_x, row["rank"], f"{p:.3f} / {s:.3f} / {k:.3f}", va="center", ha="left", fontsize=_bar_label_fontsize(len(r)))

    ax.set_yticks(r["rank"].to_numpy(), r["label"].tolist())
    ax.set_xlim(0.0, 1.25)
    ax.set_ylim(-0.75, len(r) - 0.25)
    ax.set_xlabel("absolute correlation with drop")
    ax.set_ylabel("")
    ax.set_title("Drop correlation leaderboard sorted by |Spearman|", pad=26)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(title="", loc="lower center", bbox_to_anchor=(0.5, 1.055), ncol=3, frameon=False)
    ax.text(
        text_x,
        len(r) - 0.05,
        "|Pearson| / |Spearman| / |Kendall|",
        va="bottom",
        ha="left",
        fontsize=8,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_side_by_side_metric_counts(quality: pd.DataFrame, regression: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, df, title in [(axes[0], quality, "Quality metric counts"), (axes[1], regression, "Regression metric counts")]:
        if df.empty:
            ax.text(0.5, 0.5, "No rows", ha="center", va="center")
            ax.axis("off")
            continue
        counts = df.groupby(["source", "experiment"]).size().unstack(fill_value=0)
        counts.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def select_best_metrics(quality: pd.DataFrame, *, per_source: int = 5) -> pd.DataFrame:
    q = sanitize_quality_table(quality).copy()
    if q.empty:
        return q
    for col in ["best_balanced_accuracy", "roc_auc", "roc_auc_effective", "best_f1", "best_accuracy"]:
        if col not in q.columns:
            q[col] = np.nan
        q[col] = pd.to_numeric(q[col], errors="coerce")
    q["roc_auc_effective"] = q.apply(_oriented_auc, axis=1)
    q = q[pd.to_numeric(q["best_balanced_accuracy"], errors="coerce").notna()].copy()
    q = q.sort_values(
        ["source", "best_balanced_accuracy", "roc_auc_effective", "best_f1", "best_accuracy"],
        ascending=[True, False, False, False, False],
        na_position="last",
    )
    q = q.drop_duplicates(["source", "experiment", "metric"], keep="first")
    return q.groupby("source", group_keys=False, sort=True).head(int(per_source)).reset_index(drop=True)


def load_metric_rows_for_selected(
    rows_path: str | Path,
    selected: pd.DataFrame,
    *,
    extra_columns: tuple[str, ...] = ("path", "success", "drop", "conf_clean", "conf_patch"),
) -> pd.DataFrame:
    rows_path = Path(rows_path)
    header = pd.read_csv(rows_path, nrows=0, low_memory=False)
    available = set(header.columns)
    metric_cols = [str(metric) for metric in selected.get("metric", pd.Series(dtype=str)).dropna().unique() if str(metric) in available]
    base_cols = [col for col in ("source", "experiment", *extra_columns) if col in available]
    usecols = list(dict.fromkeys([*base_cols, *metric_cols]))
    if not usecols:
        return pd.DataFrame()
    return pd.read_csv(rows_path, usecols=usecols, low_memory=False)


def _metric_scores_for_row(rows: pd.DataFrame, metric_row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    metric = str(metric_row["metric"])
    if metric not in rows.columns:
        return np.asarray([], dtype=bool), np.asarray([], dtype=float)
    sub = rows[
        rows["source"].astype(str).eq(str(metric_row["source"]))
        & rows["experiment"].astype(str).eq(str(metric_row["experiment"]))
    ].copy()
    if sub.empty:
        return np.asarray([], dtype=bool), np.asarray([], dtype=float)
    labels = sub["success"].astype(bool).to_numpy()
    scores = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(scores)
    return labels[mask], scores[mask]


def _threshold_from_quality_row(row: pd.Series) -> tuple[float, int]:
    threshold = row.get("balanced_threshold", np.nan)
    direction = row.get("balanced_direction", np.nan)
    if not np.isfinite(float(threshold)) if threshold is not None else True:
        threshold = row.get("best_threshold", np.nan)
    if not np.isfinite(float(direction)) if direction is not None else True:
        direction = row.get("best_direction", 1)
    return float(threshold), int(direction)


def best_metric_confusion_table(selected: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for _, metric_row in selected.iterrows():
        labels, scores = _metric_scores_for_row(rows, metric_row)
        threshold, direction = _threshold_from_quality_row(metric_row)
        if len(scores) == 0 or not np.isfinite(threshold):
            tp = fp = tn = fn = np.nan
        else:
            pred = scores >= threshold if int(direction) == 1 else scores <= threshold
            tp = int(np.sum(pred & labels))
            fp = int(np.sum(pred & ~labels))
            fn = int(np.sum(~pred & labels))
            tn = int(np.sum(~pred & ~labels))
        item = metric_row.to_dict()
        item.update({"threshold_used": threshold, "direction_used": direction, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n_eval": int(len(scores))})
        out.append(item)
    return pd.DataFrame(out)


def _short_metric_label(row: pd.Series) -> str:
    experiment = str(row.get("experiment", ""))
    metric = str(row.get("metric", ""))
    return f"{experiment} | {metric}"


def _oriented_auc(row: pd.Series) -> float:
    raw_auc = row.get("roc_auc", np.nan)
    if raw_auc is None or not np.isfinite(float(raw_auc)):
        effective_auc = row.get("roc_auc_effective", np.nan)
        return float(effective_auc) if effective_auc is not None and np.isfinite(float(effective_auc)) else float("nan")
    raw_auc = float(raw_auc)
    return float(max(raw_auc, 1.0 - raw_auc))


def _roc_direction_from_auc(row: pd.Series) -> int:
    raw_auc = row.get("roc_auc", np.nan)
    if raw_auc is None or not np.isfinite(float(raw_auc)):
        return int(row.get("best_direction", row.get("balanced_direction", 1)) or 1)
    return 1 if float(raw_auc) >= 0.5 else -1


def plot_best_metrics_quality(selected: pd.DataFrame):
    if selected.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No selected metrics", ha="center", va="center")
        ax.axis("off")
        return fig
    sources = selected["source"].astype(str).drop_duplicates().tolist()
    fig, axes = plt.subplots(len(sources), 1, figsize=(14, max(4.2, 3.2 * len(sources))), squeeze=False)
    palette = {
        "best_balanced_accuracy": "#2F6B9A",
        "best_accuracy": "#E6862A",
        "best_f1": "#3B8C5A",
        "roc_auc_effective": "#8E5EA2",
    }
    for ax, source in zip(axes[:, 0], sources):
        sub = selected[selected["source"].astype(str).eq(source)].copy().iloc[::-1]
        labels = [_short_metric_label(row) for _, row in sub.iterrows()]
        y = np.arange(len(sub), dtype=float)
        bacc = pd.to_numeric(sub["best_balanced_accuracy"], errors="coerce").to_numpy(dtype=float)
        ax.barh(y, bacc, color=palette["best_balanced_accuracy"], alpha=0.88, label="balanced accuracy")
        for col, label, marker in [
            ("best_accuracy", "accuracy", "o"),
            ("best_f1", "F1", "D"),
            ("roc_auc_effective", "ROC-AUC", "s"),
        ]:
            values = pd.to_numeric(sub[col] if col in sub.columns else np.nan, errors="coerce").to_numpy(dtype=float)
            if col == "roc_auc_effective" and np.isnan(values).all():
                values = np.asarray([_oriented_auc(row) for _, row in sub.iterrows()], dtype=float)
            ax.scatter(values, y, s=42, marker=marker, color=palette[col], edgecolors="white", linewidths=0.7, label=label, zorder=3)
        labels_right = []
        for _, row in sub.iterrows():
            auc = _oriented_auc(row)
            labels_right.append(
                f"{float(row.get('best_balanced_accuracy', np.nan)):.3f} "
                f"({float(row.get('best_accuracy', np.nan)):.3f}/"
                f"{float(row.get('best_f1', np.nan)):.3f}/"
                f"{auc:.3f})"
            )
        _annotate_barh(ax, y, bacc, labels_right, fontsize=_bar_label_fontsize(len(sub)))
        ax.set_yticks(y, labels)
        ax.set_xlim(0, 1.22)
        ax.set_title(f"{source}: top quality metrics")
        ax.set_xlabel("score")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
        ax.text(1.02, len(sub) - 0.35, "bacc (acc/F1/AUC)", ha="left", va="bottom", fontsize=8, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_best_metric_distributions_and_roc(selected: pd.DataFrame, rows: pd.DataFrame, *, source: str):
    sub = selected[selected["source"].astype(str).eq(str(source))].copy()
    n = len(sub)
    fig, axes = plt.subplots(max(1, n), 2, figsize=(14, max(4.0, 3.4 * max(1, n))), squeeze=False)
    if n == 0:
        axes[0, 0].text(0.5, 0.5, f"No selected {source} metrics", ha="center", va="center")
        axes[0, 0].axis("off")
        axes[0, 1].axis("off")
        return fig
    for row_idx, (_, metric_row) in enumerate(sub.iterrows()):
        metric = str(metric_row["metric"])
        labels, scores = _metric_scores_for_row(rows, metric_row)
        threshold, direction = _threshold_from_quality_row(metric_row)
        ax_hist, ax_roc = axes[row_idx]
        if len(scores) == 0:
            ax_hist.text(0.5, 0.5, f"No rows for {metric}", ha="center", va="center")
            ax_hist.axis("off")
            ax_roc.axis("off")
            continue
        fail_scores = scores[~labels]
        success_scores = scores[labels]
        bins = min(60, max(15, int(np.sqrt(len(scores)))))
        ax_hist.hist(fail_scores, bins=bins, density=True, alpha=0.55, color="#4C78A8", label=f"fail n={len(fail_scores)}")
        ax_hist.hist(success_scores, bins=bins, density=True, alpha=0.55, color="#F58518", label=f"success n={len(success_scores)}")
        if np.isfinite(threshold):
            ax_hist.axvline(threshold, color="black", linestyle="--", linewidth=1.2, alpha=0.75, label=f"thr={threshold:.3g}")
        ax_hist.set_title(f"{source}: {_short_metric_label(metric_row)}")
        ax_hist.set_xlabel(metric)
        ax_hist.set_ylabel("density")
        ax_hist.grid(alpha=0.25)
        ax_hist.legend(fontsize=8)

        curve = roc_curve_points(labels, scores, direction=_roc_direction_from_auc(metric_row))
        auc = _oriented_auc(metric_row)
        ax_roc.plot(curve["fpr"], curve["tpr"], color="#2F6B9A", linewidth=2)
        ax_roc.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, alpha=0.55)
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        ax_roc.set_title(f"ROC-AUC={auc:.3f}, bacc={float(metric_row.get('best_balanced_accuracy', np.nan)):.3f}")
        ax_roc.set_xlabel("false positive rate")
        ax_roc.set_ylabel("true positive rate")
        ax_roc.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_best_metric_confusion(confusion: pd.DataFrame):
    if confusion.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No confusion rows", ha="center", va="center")
        ax.axis("off")
        return fig
    sources = confusion["source"].astype(str).drop_duplicates().tolist()
    fig, axes = plt.subplots(len(sources), 1, figsize=(14, max(4.2, 3.4 * len(sources))), squeeze=False)
    colors = {"tp": "#3B8C5A", "tn": "#4C78A8", "fp": "#E6862A", "fn": "#C44E52"}
    for ax, source in zip(axes[:, 0], sources):
        sub = confusion[confusion["source"].astype(str).eq(source)].copy()
        labels = [_short_metric_label(row) for _, row in sub.iterrows()]
        x = np.arange(len(sub), dtype=float)
        width = 0.18
        offsets = {"tp": -1.5 * width, "tn": -0.5 * width, "fp": 0.5 * width, "fn": 1.5 * width}
        max_value = 0.0
        for key in ["tp", "tn", "fp", "fn"]:
            values = pd.to_numeric(sub[key], errors="coerce").fillna(0).to_numpy(dtype=float)
            max_value = max(max_value, float(np.max(values)) if len(values) else 0.0)
            bars = ax.bar(x + offsets[key], values, width=width, color=colors[key], label=key.upper(), alpha=0.9)
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(1.0, max_value * 0.01), f"{int(value)}", ha="center", va="bottom", fontsize=_bar_label_fontsize(len(sub)))
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel("examples")
        ax.set_title(f"{source}: TP/TN/FP/FN at selected threshold")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=4, frameon=False)
    fig.tight_layout()
    return fig
