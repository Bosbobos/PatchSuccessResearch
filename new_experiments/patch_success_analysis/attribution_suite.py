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


METHODS = ("segmentig", "odam", "layercam", "full_layerig")


def _ensure_plot_cache_env() -> None:
    cache_root = Path(tempfile.gettempdir()) / "patch_success_plot_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def _json_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _method_maps_cache_path(exp, example, *, layer_name: str) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "path": example.path,
        "drop": float(example.drop),
        "success": bool(example.success),
        "target_layer": layer_name,
        "detect_layer": exp.config.detect_layer,
        "target_mode": exp.config.target_mode,
        "n_steps": int(exp.config.n_steps),
        "alpha_batch_size": int(exp.config.alpha_batch_size),
        "imgsz": int(exp.config.attack.imgsz),
        "methods": ["odam", "layercam", "full_layerig"],
        "method_version": 2,
    }
    out = exp.derived_cache_dir / "attribution_method_maps"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"attribution_method_maps_{_json_key(payload)}.npz"


def _suite_cache_path(exp, examples, *, layer_name: str, top_percent: float) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [item.path for item in examples],
        "target_layer": layer_name,
        "detect_layer": exp.config.detect_layer,
        "target_mode": exp.config.target_mode,
        "n_steps": int(exp.config.n_steps),
        "alpha_batch_size": int(exp.config.alpha_batch_size),
        "top_percent": float(top_percent),
        "methods": list(METHODS),
        "method_version": 2,
    }
    return exp.derived_cache_dir / f"attribution_method_suite_{_json_key(payload)}.pkl"


def _load_npz(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _to_chw(tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    return np.asarray(arr, dtype="float32")


def _relu_hw(chw: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(chw, dtype="float32").sum(axis=0), 0.0).astype("float32", copy=False)


def _compute_or_load_cam_method_maps(exp, example, *, model, layer, layer_name: str) -> dict[str, Any]:
    path = _method_maps_cache_path(exp, example, layer_name=layer_name)
    cached = _load_npz(path)
    if cached is not None:
        cached["cache_path"] = str(path)
        cached["loaded_from_cache"] = True
        return cached

    import torch

    from .attributions import _activation_and_gradient, compute_layer_ig_attribution

    ctx = exp._context_for_example(example, image_variant="clean")
    activation, grad = _activation_and_gradient(model, ctx["inputs"], target_fn=ctx["target_fn"], layer=layer)

    # ODAM official implementation uses ReLU(sum_c grad_cij * A_cij) as the heat map.
    # Keep the channel-wise product too, because our scalar metrics work on CxHxW neurons.
    odam_chw = _to_chw(grad * activation.detach())
    layercam_chw = _to_chw(torch.relu(grad) * activation.detach())

    full_layerig = compute_layer_ig_attribution(
        model,
        ctx["inputs"],
        ctx["baselines"],
        target_fn=ctx["target_fn"],
        layer=layer,
        layer_name=layer_name,
        method="Full LayerIG",
        n_steps=int(exp.config.n_steps),
        alpha_batch_size=int(exp.config.alpha_batch_size),
        segment_start=0.0,
        segment_end=1.0,
    )
    full_layerig_chw = _to_chw(full_layerig.chw())

    arrays = {
        "odam_chw": odam_chw,
        "odam_hw": _relu_hw(odam_chw),
        "layercam_chw": layercam_chw,
        "layercam_hw": _relu_hw(layercam_chw),
        "full_layerig_chw": full_layerig_chw,
        "full_layerig_hw": _relu_hw(full_layerig_chw),
    }
    np.savez_compressed(path, **arrays)
    arrays["cache_path"] = str(path)
    arrays["loaded_from_cache"] = False
    return arrays


def _get_segmentig_layer_maps(exp, example, *, model, layer, layer_name: str) -> dict[str, Any]:
    layer_maps = exp._load_layer_map_cache(example, layer_name=layer_name, include_clean_activation=False)
    if layer_maps is not None:
        return layer_maps
    ctx = exp._context_for_example(example, image_variant="clean")
    return exp._compute_or_load_segmentig_layer_maps(
        example,
        ctx,
        model=model,
        layer=layer,
        layer_name=layer_name,
        include_clean_activation=False,
    )


def _method_metric_names(method: str, row: dict[str, Any]) -> list[str]:
    prefix = f"{method}_"
    return [
        key
        for key in row
        if key.startswith(prefix)
        and (
            key.startswith(f"{prefix}align_")
            or key.startswith(f"{prefix}importance_")
            or key.startswith(f"{prefix}delta_energy_in_importance_top")
            or key.startswith(f"{prefix}delta_importance_product_")
            or key.startswith(f"{prefix}delta_energy_importance_bins_")
            or key.startswith(f"{prefix}delta_energy_importance_binfrac_")
            or key.startswith(f"{prefix}hand_")
        )
    ]


def _compute_method_metrics(delta_chw: np.ndarray, importance_chw: np.ndarray, *, example, exp, top_percent: float) -> dict[str, float]:
    from .metrics import (
        alignment_metrics,
        handcrafted_delta_importance_features,
        importance_rank_bin_energy_fractions,
        segmentig_soft_alignment_metrics,
    )

    delta_flat = np.asarray(delta_chw).reshape(-1)
    importance_flat = np.asarray(importance_chw).reshape(-1)
    out = alignment_metrics(delta_flat, importance_flat, top_percent=float(top_percent))
    out.update(segmentig_soft_alignment_metrics(delta_flat, importance_flat))
    out.update(
        handcrafted_delta_importance_features(
            delta_chw,
            importance_chw,
            patch_bbox_xyxy=example.patch_bbox_lb,
            imgsz=int(exp.config.attack.imgsz),
        )
    )
    bin_fractions = importance_rank_bin_energy_fractions(delta_flat, importance_flat, n_bins=100)
    out.update({f"delta_energy_importance_binfrac_{idx:03d}": float(value) for idx, value in enumerate(bin_fractions, start=1)})
    return out


def _add_prefixed_metrics(row: dict[str, Any], method: str, metrics: dict[str, float]) -> None:
    for name, value in metrics.items():
        row[f"{method}_{name}"] = value


def compute_or_load_attribution_method_suite(
    exp,
    *,
    layer_name: str | None = None,
    max_examples: int | None = None,
    top_percent: float = 5.0,
    force: bool = False,
) -> dict[str, Any]:
    _ensure_plot_cache_env()
    import matplotlib.pyplot as plt

    from .activations import delta_spread_metrics
    from .metrics import metric_quality_rows
    from .plots import plot_metric_distribution_and_roc
    from .regression_metrics import regression_similarity_table
    from .yolo import get_module_by_name

    cache = exp.get_cache()
    examples = list(cache.examples)
    if max_examples is not None:
        examples = examples[: int(max_examples)]
    layer_name = layer_name or exp.config.target_layer
    cache_path = _suite_cache_path(exp, examples, layer_name=layer_name, top_percent=float(top_percent))
    if cache_path.exists() and not force:
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(cache_path)
        return cached

    _yolo, model = exp.load_model()
    layer = get_module_by_name(model, layer_name)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    metrics_batch_size = max(1, int(getattr(exp.config, "metrics_batch_size", 64)))

    for idx, example in enumerate(examples, start=1):
        try:
            segmentig_maps = _get_segmentig_layer_maps(exp, example, model=model, layer=layer, layer_name=layer_name)
            method_maps = _compute_or_load_cam_method_maps(exp, example, model=model, layer=layer, layer_name=layer_name)
            delta_chw = np.asarray(segmentig_maps["delta_chw"], dtype="float32")
            row: dict[str, Any] = {
                "path": example.path,
                "success": bool(example.success),
                "conf_clean": float(example.conf_clean),
                "conf_patch": float(example.conf_patch),
                "drop": float(example.drop),
                "layer_maps_cache_path": segmentig_maps["cache_path"],
                "method_maps_cache_path": method_maps["cache_path"],
                "layer_maps_loaded_from_cache": bool(segmentig_maps["loaded_from_cache"]),
                "method_maps_loaded_from_cache": bool(method_maps["loaded_from_cache"]),
                **delta_spread_metrics(delta_chw, patch_bbox_xyxy=example.patch_bbox_lb, imgsz=int(exp.config.attack.imgsz)),
            }
            importance_by_method = {
                "segmentig": np.asarray(segmentig_maps["segmentig_chw"], dtype="float32"),
                "odam": np.asarray(method_maps["odam_chw"], dtype="float32"),
                "layercam": np.asarray(method_maps["layercam_chw"], dtype="float32"),
                "full_layerig": np.asarray(method_maps["full_layerig_chw"], dtype="float32"),
            }
            for method, importance_chw in importance_by_method.items():
                _add_prefixed_metrics(
                    row,
                    method,
                    _compute_method_metrics(delta_chw, importance_chw, example=example, exp=exp, top_percent=float(top_percent)),
                )
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - one bad example should not stop the suite.
            skipped.append(
                {
                    "path": example.path,
                    "success": bool(example.success),
                    "drop": float(example.drop),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if idx % metrics_batch_size == 0:
                exp._release_batch_memory()
    exp._release_batch_memory()
    if not rows:
        raise RuntimeError(f"No valid attribution method rows; skipped={len(skipped)}")

    labels = [row["success"] for row in rows]
    rows_df = pd.DataFrame(rows)
    quality_by_method: dict[str, pd.DataFrame] = {}
    regression_by_method: dict[str, pd.DataFrame] = {}
    figure_dir = exp.figures_dir / "attribution_methods"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for method in METHODS:
        metric_names = _method_metric_names(method, rows[0])
        quality = pd.DataFrame(metric_quality_rows(labels, {name: [row[name] for row in rows] for name in metric_names}))
        if not quality.empty:
            quality = quality.sort_values(["best_balanced_accuracy", "roc_auc", "best_accuracy"], ascending=False).reset_index(drop=True)
            for item in quality.head(20).to_dict("records"):
                name = item["metric"]
                path = figure_dir / f"{method}_metric_{name}.png"
                fig = plot_metric_distribution_and_roc(
                    labels,
                    rows_df[name].to_numpy(),
                    metric_name=name,
                    auc=float(item["roc_auc"]),
                    best_accuracy=float(item["best_accuracy"]),
                    direction=int(item["best_direction"]),
                    save_path=path,
                )
                plt.close(fig)
                quality.loc[quality["metric"] == name, "figure_path"] = str(path)
        quality_by_method[method] = quality
        regression = regression_similarity_table(rows_df[["path", "success", "drop", *metric_names]], target_col="drop", metric_cols=metric_names)
        regression_by_method[method] = regression

    all_quality = pd.concat([df.assign(method=method) for method, df in quality_by_method.items()], ignore_index=True)
    all_regression = pd.concat([df.assign(method=method) for method, df in regression_by_method.items()], ignore_index=True)
    result = {
        "rows": rows,
        "quality_by_method": quality_by_method,
        "regression_by_method": regression_by_method,
        "all_quality": all_quality.sort_values(["best_balanced_accuracy", "roc_auc", "best_accuracy"], ascending=False).reset_index(drop=True),
        "all_regression": all_regression.sort_values(["main_score", "abs_spearman", "abs_pearson"], ascending=False).reset_index(drop=True),
        "skipped": skipped,
        "cache_path": str(cache_path),
        "loaded_from_cache": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(result, fh)
    return result


def plot_method_classification_summary(all_quality: pd.DataFrame, *, top_n: int = 12):
    _ensure_plot_cache_env()
    import matplotlib.pyplot as plt

    df = pd.DataFrame(all_quality).head(int(top_n)).copy()
    fig, ax = plt.subplots(figsize=(14, 5.8), constrained_layout=True)
    labels = [f"{row.method}\n{row.metric}" for row in df.itertuples(index=False)]
    x = np.arange(len(df))
    score_col = "best_balanced_accuracy" if "best_balanced_accuracy" in df.columns else "best_accuracy"
    score_label = "best balanced accuracy" if score_col == "best_balanced_accuracy" else "best accuracy"
    ax.bar(x - 0.18, df[score_col], width=0.36, label=score_label, color="#4C78A8")
    ax.bar(x + 0.18, df["roc_auc"], width=0.36, label="ROC-AUC", color="#72B7B2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
    ax.set_title("Overall best attribution-method metrics for success/fail classification")
    ax.set_ylabel("score")
    ax.set_ylim(max(0.0, float(np.nanmin(df[[score_col, "roc_auc"]].to_numpy())) - 0.05), 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return fig


def plot_method_regression_summary(all_regression: pd.DataFrame, *, top_n: int = 12):
    _ensure_plot_cache_env()
    import matplotlib.pyplot as plt

    df = pd.DataFrame(all_regression).head(int(top_n)).copy()
    fig, ax = plt.subplots(figsize=(14, 5.8), constrained_layout=True)
    labels = [f"{row.method}\n{row.metric}" for row in df.itertuples(index=False)]
    x = np.arange(len(df))
    ax.bar(x - 0.22, df["main_score"], width=0.22, label="|Spearman| (main)", color="#4C78A8")
    ax.bar(x, df["abs_spearman"], width=0.22, label="|Spearman|", color="#F58518")
    ax.bar(x + 0.22, df["abs_pearson"], width=0.22, label="|Pearson|", color="#72B7B2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
    ax.set_title("Overall best attribution-method metrics for drop regression similarity")
    ax.set_ylabel("score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return fig
