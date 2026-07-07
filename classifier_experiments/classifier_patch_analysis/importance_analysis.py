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
        raise ValueError(f"Expected CHW values, got shape={arr.shape}")
    return arr


def _cumulative_quartiles(values, *, signed: bool) -> tuple[np.ndarray, dict[str, int]]:
    vals = np.asarray(values, dtype="float64")
    if vals.size == 0:
        return np.asarray([], dtype=int), {"q25": 0, "q50": 0, "q75": 0}
    if signed:
        total = float(vals.sum())
        if abs(total) > 1e-12 and np.all(np.diff(np.cumsum(vals) / total) >= -1e-12):
            cumulative = np.cumsum(vals) / total
        else:
            mass = np.abs(vals)
            cumulative = np.cumsum(mass) / float(mass.sum() + 1e-12)
    else:
        mass = np.abs(vals)
        cumulative = np.cumsum(mass) / float(mass.sum() + 1e-12)
    bins = np.searchsorted([0.25, 0.5, 0.75], cumulative, side="left") + 1
    positions = {
        "q25": int(np.searchsorted(cumulative, 0.25, side="left") + 1),
        "q50": int(np.searchsorted(cumulative, 0.5, side="left") + 1),
        "q75": int(np.searchsorted(cumulative, 0.75, side="left") + 1),
    }
    return bins.astype(int), positions


def _cache_path(exp, examples, *, layer_name: str) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [item.path for item in examples],
        "drops": [float(item.drop) for item in examples],
        "success": [bool(item.success) for item in examples],
        "layer_name": layer_name,
        "importance_target": "clean_one_logit_person_score",
        "method_version": 2,
    }
    return Path(exp.derived_cache_dir) / f"importance_rankings_{stable_hash(payload)}.pkl"


def compute_or_load_importance_rankings(
    exp,
    examples,
    *,
    layer_name: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    layer_name = layer_name or exp.config.target_layer
    selected = list(examples)
    if not selected:
        raise ValueError("No examples were provided.")
    path = _cache_path(exp, selected, layer_name=layer_name)
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached

    neuron_abs_sum = None
    neuron_signed_sum = None
    channel_abs_sum = None
    channel_signed_sum = None
    n = 0
    map_paths = []
    for example in selected:
        maps = exp.compute_layer_map(example, layer_name=layer_name, force=False)
        importance = _as_chw(maps["importance_chw"]).astype("float64", copy=False)
        abs_importance = np.abs(importance)
        if neuron_abs_sum is None:
            neuron_abs_sum = np.zeros_like(abs_importance, dtype="float64")
            neuron_signed_sum = np.zeros_like(importance, dtype="float64")
            channel_abs_sum = np.zeros(importance.shape[0], dtype="float64")
            channel_signed_sum = np.zeros(importance.shape[0], dtype="float64")
        neuron_abs_sum += abs_importance
        neuron_signed_sum += importance
        channel_abs_sum += abs_importance.mean(axis=(1, 2))
        channel_signed_sum += importance.mean(axis=(1, 2))
        map_paths.append(maps["cache_path"])
        n += 1

    assert neuron_abs_sum is not None
    neuron_abs_mean = neuron_abs_sum / max(1, n)
    neuron_signed_mean = neuron_signed_sum / max(1, n)
    channel_abs_mean = channel_abs_sum / max(1, n)
    channel_signed_mean = channel_signed_sum / max(1, n)

    c, h, w = neuron_abs_mean.shape
    coords = np.indices((c, h, w)).reshape(3, -1).T
    neuron_df = pd.DataFrame(
        {
            "channel": coords[:, 0].astype(int),
            "y": coords[:, 1].astype(int),
            "x": coords[:, 2].astype(int),
            "importance_abs_mean": neuron_abs_mean.reshape(-1),
            "importance_signed_mean": neuron_signed_mean.reshape(-1),
        }
    )
    neuron_abs_df = neuron_df.sort_values("importance_abs_mean", ascending=False).reset_index(drop=True)
    neuron_abs_df.insert(0, "rank", np.arange(1, len(neuron_abs_df) + 1, dtype=int))
    neuron_abs_df["cumulative_importance_frac"] = (
        neuron_abs_df["importance_abs_mean"].abs().cumsum() / (neuron_abs_df["importance_abs_mean"].abs().sum() + 1e-12)
    )
    neuron_abs_df["sum_quartile"] = _cumulative_quartiles(neuron_abs_df["importance_abs_mean"], signed=False)[0]

    neuron_signed_df = neuron_df.sort_values("importance_signed_mean", ascending=False).reset_index(drop=True)
    neuron_signed_df.insert(0, "rank", np.arange(1, len(neuron_signed_df) + 1, dtype=int))
    signed_total = float(neuron_signed_df["importance_signed_mean"].sum())
    if abs(signed_total) > 1e-12:
        neuron_signed_df["cumulative_importance_frac"] = neuron_signed_df["importance_signed_mean"].cumsum() / signed_total
    else:
        neuron_signed_df["cumulative_importance_frac"] = (
            neuron_signed_df["importance_signed_mean"].abs().cumsum()
            / (neuron_signed_df["importance_signed_mean"].abs().sum() + 1e-12)
        )
    neuron_signed_df["sum_quartile"] = _cumulative_quartiles(neuron_signed_df["importance_signed_mean"], signed=True)[0]

    filter_df = pd.DataFrame(
        {
            "channel": np.arange(c, dtype=int),
            "filter_abs_mean": channel_abs_mean,
            "filter_signed_mean": channel_signed_mean,
        }
    )
    filter_abs_df = filter_df.sort_values("filter_abs_mean", ascending=False).reset_index(drop=True)
    filter_abs_df.insert(0, "rank", np.arange(1, len(filter_abs_df) + 1, dtype=int))
    filter_abs_df["cumulative_importance_frac"] = (
        filter_abs_df["filter_abs_mean"].abs().cumsum() / (filter_abs_df["filter_abs_mean"].abs().sum() + 1e-12)
    )
    filter_abs_df["sum_quartile"] = _cumulative_quartiles(filter_abs_df["filter_abs_mean"], signed=False)[0]

    filter_signed_df = filter_df.sort_values("filter_signed_mean", ascending=False).reset_index(drop=True)
    filter_signed_df.insert(0, "rank", np.arange(1, len(filter_signed_df) + 1, dtype=int))
    signed_total = float(filter_signed_df["filter_signed_mean"].sum())
    if abs(signed_total) > 1e-12:
        filter_signed_df["cumulative_importance_frac"] = filter_signed_df["filter_signed_mean"].cumsum() / signed_total
    else:
        filter_signed_df["cumulative_importance_frac"] = (
            filter_signed_df["filter_signed_mean"].abs().cumsum()
            / (filter_signed_df["filter_signed_mean"].abs().sum() + 1e-12)
        )
    filter_signed_df["sum_quartile"] = _cumulative_quartiles(filter_signed_df["filter_signed_mean"], signed=True)[0]

    result = {
        "layer_name": layer_name,
        "n_examples": n,
        "map_paths": map_paths,
        "neuron_abs_df": neuron_abs_df,
        "neuron_signed_df": neuron_signed_df,
        "filter_abs_df": filter_abs_df,
        "filter_signed_df": filter_signed_df,
        "quartile_definition": "sum_quartile is based on cumulative fraction of total importance mass, not count rank",
        "cache_path": str(path),
        "loaded_from_cache": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh)
    return result


def plot_importance_rankings(rankings: dict[str, Any]):
    import matplotlib.pyplot as plt

    panels = [
        ("neuron_abs_df", "importance_abs_mean", False, "neurons | abs mean importance"),
        ("neuron_signed_df", "importance_signed_mean", True, "neurons | signed mean importance"),
        ("filter_abs_df", "filter_abs_mean", False, "filters | abs mean importance"),
        ("filter_signed_df", "filter_signed_mean", True, "filters | signed mean importance"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for ax, (df_key, value_col, signed, title) in zip(axes.reshape(-1), panels, strict=True):
        df = rankings[df_key]
        x = np.arange(1, len(df) + 1, dtype=int)
        ax.plot(x, df[value_col].to_numpy(dtype="float64"), linewidth=1.2)
        _quartiles, positions = _cumulative_quartiles(df[value_col], signed=signed)
        for label, pos in [("25%", positions["q25"]), ("50%", positions["q50"]), ("75%", positions["q75"])]:
            ax.axvline(pos, color="#999999", linestyle="--", linewidth=0.9)
            ax.text(pos, ax.get_ylim()[1], label, va="top", ha="right", fontsize=8, color="#555555")
        ax.set_title(title)
        ax.set_xlabel("sorted rank")
        ax.set_ylabel(value_col)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Importance rankings, {rankings['layer_name']}, n={rankings['n_examples']}")
    return fig
