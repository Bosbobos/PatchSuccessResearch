from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .spread_precision import compute_or_load_layer_maps
from .utils import stable_hash


def _as_chw(values) -> np.ndarray:
    arr = np.asarray(values, dtype="float32")
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW values, got shape={arr.shape}")
    return arr


def _percent_label(percent: float) -> str:
    return f"top{float(percent):g}".replace(".", "p")


def psnr_importance_sweep(delta_chw, importance_chw, *, percentages: tuple[float, ...], max_value: float = 255.0, eps: float = 1e-12):
    delta = _as_chw(delta_chw).reshape(-1).astype("float64", copy=False)
    importance = np.abs(_as_chw(importance_chw).reshape(-1).astype("float64", copy=False))
    n = min(delta.size, importance.size)
    delta, importance = delta[:n], importance[:n]
    if n == 0:
        return {}
    order = np.argsort(-importance, kind="stable")
    ranked_delta = delta[order]
    values = np.log10((float(max_value) * float(max_value)) / np.maximum(ranked_delta * ranked_delta, float(eps)))
    out = {}
    for percent in percentages:
        label = _percent_label(percent)
        k = n if float(percent) >= 100.0 else max(1, int(round(float(percent) / 100.0 * n)))
        k = min(k, n)
        top = values[:k]
        out[f"psnr_{label}_importance_sum"] = float(top.sum())
        out[f"psnr_{label}_importance_mean"] = float(top.mean())
        out[f"psnr_{label}_importance_min"] = float(top.min())
        out[f"psnr_{label}_importance_zero_frac"] = float(np.mean(np.abs(ranked_delta[:k]) <= np.sqrt(float(eps))))
        out[f"psnr_{label}_importance_k"] = float(k)
    return out


def _cache_path(
    exp,
    examples,
    *,
    layer_name: str,
    percentages: tuple[float, ...],
    max_value: float,
    eps: float,
    build_missing_layer_maps: bool,
) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [item.path for item in examples],
        "layer_name": layer_name,
        "percentages": [float(v) for v in percentages],
        "max_value": float(max_value),
        "eps": float(eps),
        "build_missing_layer_maps": bool(build_missing_layer_maps),
        "quality_metric": "best_balanced_accuracy",
        "method_version": 3,
    }
    return Path(exp.derived_cache_dir) / f"psnr_metrics_{stable_hash(payload)}.pkl"


def compute_or_load_psnr_metrics(
    exp,
    *,
    layer_name: str | None = None,
    percentages: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0),
    max_examples: int | None = None,
    max_value: float = 255.0,
    eps: float = 1e-12,
    build_missing_layer_maps: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    from .metrics import metric_quality_rows
    from .regression_metrics import regression_similarity_table

    cache = exp.get_cache()
    examples = list(cache.examples)
    if max_examples is not None:
        examples = examples[: int(max_examples)]
    layer_name = layer_name or exp.config.target_layer
    path = _cache_path(
        exp,
        examples,
        layer_name=layer_name,
        percentages=tuple(percentages),
        max_value=max_value,
        eps=eps,
        build_missing_layer_maps=bool(build_missing_layer_maps),
    )
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached

    # PSNR uses only the selected target layer. Missing maps for this layer may
    # be built, but unrelated display layers are never requested here.
    if build_missing_layer_maps:
        compute_or_load_layer_maps(exp, layer_name=layer_name, max_examples=max_examples, force=False)
    rows = []
    skipped = []
    for example in examples:
        maps = exp._load_layer_map_cache(example, layer_name=layer_name)
        if maps is None:
            skipped.append({"path": example.path, "reason": "missing layer map"})
            continue
        metrics = psnr_importance_sweep(
            maps["delta_chw"],
            maps["importance_chw"],
            percentages=tuple(percentages),
            max_value=float(max_value),
            eps=float(eps),
        )
        rows.append(
            {
                "path": example.path,
                "success": bool(example.success),
                "drop": float(example.drop),
                "conf_clean": float(example.conf_clean),
                "conf_patch": float(example.conf_patch),
                **metrics,
            }
        )
    rows_df = pd.DataFrame(rows)
    metric_cols = [col for col in rows_df.columns if col.startswith("psnr_")]
    quality = pd.DataFrame(metric_quality_rows(rows_df["success"].to_numpy(dtype=bool), {col: rows_df[col].to_numpy() for col in metric_cols}))
    quality = quality.sort_values(["best_balanced_accuracy", "roc_auc", "best_f1"], ascending=False).reset_index(drop=True)
    regression = regression_similarity_table(rows_df[["path", "success", "drop", *metric_cols]], metric_cols=metric_cols)
    result = {
        "rows_df": rows_df,
        "metric_cols": metric_cols,
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
