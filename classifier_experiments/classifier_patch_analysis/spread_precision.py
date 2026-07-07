from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import stable_hash


def _as_chw(values) -> np.ndarray:
    arr = np.asarray(values, dtype="float32")
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={arr.shape}")
    return arr


def compute_or_load_layer_maps(
    exp,
    *,
    layer_name: str | None = None,
    layers: list[str] | tuple[str, ...] | None = None,
    max_examples: int | None = None,
    force: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    cache = exp.get_cache()
    selected = list(cache.examples)
    if max_examples is not None:
        selected = selected[: int(max_examples)]
    layers = list(layers or [layer_name or exp.config.target_layer])
    out = {"n_examples": len(selected), "layers": layers, "computed": 0, "loaded": 0, "failed": []}
    progress = None
    try:
        from tqdm.auto import tqdm

        progress = tqdm(total=len(selected) * len(layers), desc="layer maps", unit="map")
    except Exception:
        progress = None
    for layer in layers:
        for example in selected:
            try:
                path = exp._layer_map_cache_path(example, layer_name=layer)
                existed = path.exists()
                result = exp.compute_layer_map(example, layer_name=layer, force=force)
                if existed and not force and isinstance(result, dict) and result.get("loaded_from_cache"):
                    out["loaded"] += 1
                else:
                    out["computed"] += 1
            except Exception as exc:  # noqa: BLE001
                out["failed"].append({"path": example.path, "layer": layer, "error": f"{type(exc).__name__}: {exc}"})
                if fail_fast:
                    if progress is not None:
                        progress.close()
                    raise RuntimeError(
                        f"Failed to compute layer map for {example.path} at {layer}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()
    return out


def _cache_path(exp, examples, *, layer_name: str, top_percent: float) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [item.path for item in examples],
        "layer_name": layer_name,
        "top_percent": float(top_percent),
        "success_drop_threshold": float(exp.config.attack.success_drop_threshold),
        "metric": "best_balanced_accuracy",
        "method_version": 3,
    }
    return Path(exp.derived_cache_dir) / f"spread_vs_precision_{stable_hash(payload)}.pkl"


def compute_or_load_spread_vs_precision(
    exp,
    *,
    layer_name: str | None = None,
    max_examples: int | None = None,
    top_percent: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    from .activations import delta_spread_metrics
    from .metrics import alignment_metrics, metric_quality_rows
    from .regression_metrics import regression_similarity_table

    cache = exp.get_cache()
    examples = list(cache.examples)
    if max_examples is not None:
        examples = examples[: int(max_examples)]
    layer_name = layer_name or exp.config.target_layer
    top_percent = float(top_percent if top_percent is not None else exp.config.top_percent)
    path = _cache_path(exp, examples, layer_name=layer_name, top_percent=top_percent)
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached

    compute_or_load_layer_maps(exp, layer_name=layer_name, max_examples=max_examples, force=False)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for example in examples:
        maps = exp._load_layer_map_cache(example, layer_name=layer_name)
        if maps is None:
            skipped.append({"path": example.path, "reason": "missing layer map"})
            continue
        delta = _as_chw(maps["delta_chw"])
        importance = _as_chw(maps["importance_chw"])
        spread = delta_spread_metrics(
            delta,
            patch_bbox_xyxy=example.patch_bbox,
            img_size=int(exp.config.attack.img_size),
        )
        precision = alignment_metrics(delta.reshape(-1), importance.reshape(-1), top_percent=top_percent)
        rows.append(
            {
                "path": example.path,
                "success": bool(example.success),
                "drop": float(example.drop),
                "conf_clean": float(example.conf_clean),
                "conf_patch": float(example.conf_patch),
                "clean_logit": float(example.clean_logit),
                "patched_logit": float(example.patched_logit),
                **{f"spread_{key}": value for key, value in spread.items()},
                **{f"precision_{key}": value for key, value in precision.items()},
            }
        )
    if not rows:
        raise RuntimeError(f"No spread-vs-precision rows; skipped={len(skipped)}")
    rows_df = pd.DataFrame(rows)
    metric_cols = [col for col in rows_df.columns if col.startswith("spread_") or col.startswith("precision_")]
    quality = pd.DataFrame(metric_quality_rows(rows_df["success"].to_numpy(dtype=bool), {col: rows_df[col].to_numpy() for col in metric_cols}))
    quality["family"] = np.where(quality["metric"].str.startswith("spread_"), "spread", "precision")
    quality = quality.sort_values(["best_balanced_accuracy", "roc_auc", "best_f1"], ascending=False).reset_index(drop=True)
    regression = regression_similarity_table(rows_df[["path", "success", "drop", *metric_cols]], metric_cols=metric_cols)
    if not regression.empty:
        regression["family"] = np.where(regression["metric"].str.startswith("spread_"), "spread", "precision")
    result = {
        "rows": rows,
        "rows_df": rows_df,
        "metric_cols": metric_cols,
        "quality": quality,
        "regression": regression,
        "skipped": skipped,
        "loaded_from_cache": False,
        "cache_path": str(path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh)
    return result


def compute_layer_spread_summary(exp, *, layers: list[str] | None = None, max_examples: int | None = None, force: bool = False):
    rows = []
    layers = layers or exp.all_display_layer_names()
    for layer in layers:
        data = compute_or_load_spread_vs_precision(exp, layer_name=layer, max_examples=max_examples, force=force)
        df = data["rows_df"]
        for success, group in df.groupby("success"):
            rows.append(
                {
                    "layer": layer,
                    "success": bool(success),
                    "n": int(len(group)),
                    "mean_drop": float(group["drop"].mean()),
                    "mean_outside_patch_energy_frac": float(group["spread_outside_patch_energy_frac"].mean()),
                    "std_outside_patch_energy_frac": float(group["spread_outside_patch_energy_frac"].std(ddof=0)),
                    "mean_delta_gini": float(group["spread_delta_gini"].mean()),
                    "std_delta_gini": float(group["spread_delta_gini"].std(ddof=0)),
                }
            )
    return pd.DataFrame(rows)
