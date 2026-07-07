from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "patch_success_matplotlib"))

import numpy as np
import pandas as pd

from .importance_analysis import compute_or_load_importance_rankings
from .modeling import forward_logits, preprocess_pil_batch
from .utils import stable_hash


def build_neuron_schedule(
    n_neurons: int,
    *,
    max_percent: float = 100.0,
    step_percent: float = 0.1,
) -> pd.DataFrame:
    """Build a uniform top-neuron percent sweep schedule."""
    n_neurons = int(n_neurons)
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive.")
    if float(step_percent) <= 0.0:
        raise ValueError("step_percent must be positive.")
    max_k = max(1, min(n_neurons, int(np.ceil(n_neurons * float(max_percent) / 100.0))))
    step_size = max(1, int(np.ceil(n_neurons * float(step_percent) / 100.0)))

    starts: list[int] = []
    ends: list[int] = []
    k = 0
    while k < max_k:
        starts.append(k)
        k = min(max_k, k + step_size)
        ends.append(k)

    return pd.DataFrame(
        {
            "step": np.arange(1, len(ends) + 1, dtype=int),
            "start_rank": np.asarray(starts, dtype=int) + 1,
            "end_rank": np.asarray(ends, dtype=int),
            "n_changed": np.asarray(ends, dtype=int),
            "changed_frac": np.asarray(ends, dtype="float64") / float(n_neurons),
            "changed_percent": np.asarray(ends, dtype="float64") / float(n_neurons) * 100.0,
            "step_size": np.asarray(ends, dtype=int) - np.asarray(starts, dtype=int),
        }
    )


def _cache_path(exp, examples, rankings: dict[str, Any], *, layer_name: str, max_percent: float, step_percent: float) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [item.path for item in examples],
        "drops": [float(item.drop) for item in examples],
        "success": [bool(item.success) for item in examples],
        "layer_name": layer_name,
        "rankings_cache_path": rankings.get("cache_path", ""),
        "max_percent": float(max_percent),
        "step_percent": float(step_percent),
        "interventions": ["zero", "opposite_extreme"],
        "curves": ["single_step", "cumulative"],
        "metric": "clean_logit_minus_intervened_logit_raw_and_abs",
        "method_version": 3,
    }
    return Path(exp.derived_cache_dir) / f"sanity_check_{stable_hash(payload)}.pkl"


def _examples_to_batch(exp, examples):
    clean_images = [exp._images_for_example(example)[0] for example in examples]
    model = exp.load_model()
    param = next(model.parameters())
    return preprocess_pil_batch(
        clean_images,
        img_size=int(exp.config.attack.img_size),
        device=param.device,
        dtype=param.dtype,
    )


def _ranked_neuron_arrays(rankings: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    df = rankings["neuron_abs_df"].copy()
    c = df["channel"].to_numpy(dtype=int)
    y = df["y"].to_numpy(dtype=int)
    x = df["x"].to_numpy(dtype=int)
    signed = df["importance_signed_mean"].to_numpy(dtype="float32")
    shape = (
        int(c.max()) + 1,
        int(y.max()) + 1,
        int(x.max()) + 1,
    )
    flat = (c * shape[1] * shape[2] + y * shape[2] + x).astype(int)
    return flat, signed, shape


class _LayerIntervention:
    def __init__(
        self,
        model,
        layer_name: str,
        *,
        flat_indices: np.ndarray,
        signed_importance: np.ndarray,
        mode: str,
    ):
        import torch

        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise KeyError(f"Layer {layer_name!r} was not found.")
        if mode not in {"zero", "opposite_extreme"}:
            raise ValueError(f"Unsupported intervention mode: {mode!r}")
        self.mode = mode
        self.flat_indices = torch.as_tensor(flat_indices, dtype=torch.long)
        self.signed_importance = torch.as_tensor(signed_importance, dtype=torch.float32)
        self.handle = modules[layer_name].register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        import torch

        if not isinstance(output, torch.Tensor):
            return output
        if output.ndim != 4:
            return output
        out = output.clone()
        flat = out.reshape(out.shape[0], -1)
        idx = self.flat_indices.to(device=out.device)
        if idx.numel() == 0:
            return out
        if self.mode == "zero":
            flat[:, idx] = 0.0
        else:
            signs = self.signed_importance.to(device=out.device, dtype=out.dtype)
            layer_min = flat.min(dim=1).values[:, None]
            layer_max = flat.max(dim=1).values[:, None]
            values = torch.where(signs[None, :] >= 0.0, layer_min, layer_max)
            flat[:, idx] = values
        return out

    def remove(self) -> None:
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.remove()


def _evaluate_logits(model, x, layer_name: str, flat_indices, signed_importance, mode: str):
    import torch

    with torch.no_grad():
        with _LayerIntervention(
            model,
            layer_name,
            flat_indices=np.asarray(flat_indices, dtype=int),
            signed_importance=np.asarray(signed_importance, dtype="float32"),
            mode=mode,
        ):
            return forward_logits(model, x).detach().cpu().numpy().astype("float64")


def _normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if x.size < 2:
        return float(y[0]) if y.size else 0.0
    width = float(x[-1] - x[0])
    if width <= 0.0:
        return float(np.nanmean(y))
    return float(np.trapz(y, x) / width)


def compute_or_load_sanity_check(
    exp,
    examples,
    *,
    layer_name: str | None = None,
    max_percent: float = 100.0,
    step_percent: float = 0.1,
    force: bool = False,
) -> dict[str, Any]:
    layer_name = layer_name or exp.config.target_layer
    selected = list(examples)
    if not selected:
        raise ValueError("No examples were provided.")
    rankings = compute_or_load_importance_rankings(exp, selected, layer_name=layer_name, force=False)
    path = _cache_path(exp, selected, rankings, layer_name=layer_name, max_percent=max_percent, step_percent=step_percent)
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached

    model = exp.load_model()
    x = _examples_to_batch(exp, selected)
    import torch

    with torch.no_grad():
        baseline_logits = forward_logits(model, x).detach().cpu().numpy().astype("float64")

    flat_order, signed_order, shape = _ranked_neuron_arrays(rankings)
    schedule = build_neuron_schedule(flat_order.size, max_percent=max_percent, step_percent=step_percent)
    denom = np.maximum(np.abs(baseline_logits), 1e-12)
    baseline_mean = float(np.mean(baseline_logits))
    baseline_std = float(np.std(baseline_logits, ddof=1)) if baseline_logits.size > 1 else 0.0
    rows: list[dict[str, Any]] = []

    for mode in ["zero", "opposite_extreme"]:
        for _, step in schedule.iterrows():
            start = int(step["start_rank"]) - 1
            end = int(step["end_rank"])
            for curve, sl in [
                ("single_step", slice(start, end)),
                ("cumulative", slice(0, end)),
            ]:
                logits = _evaluate_logits(model, x, layer_name, flat_order[sl], signed_order[sl], mode)
                logit_drop = baseline_logits - logits
                abs_logit_drop = np.abs(logit_drop)
                norm_drop = logit_drop / denom
                rows.append(
                    {
                        "intervention": mode,
                        "curve": curve,
                        "step": int(step["step"]),
                        "start_rank": int(step["start_rank"]),
                        "end_rank": int(step["end_rank"]),
                        "step_size": int(step["step_size"]),
                        "n_changed": int(step["n_changed"]),
                        "changed_frac": float(step["changed_frac"]),
                        "changed_percent": float(step["changed_percent"]),
                        "source": "classifier",
                        "mean_logit_drop": float(np.mean(logit_drop)),
                        "std_logit_drop": float(np.std(logit_drop, ddof=1)) if logit_drop.size > 1 else 0.0,
                        "median_logit_drop": float(np.median(logit_drop)),
                        "mean_abs_logit_drop": float(np.mean(abs_logit_drop)),
                        "std_abs_logit_drop": float(np.std(abs_logit_drop, ddof=1)) if abs_logit_drop.size > 1 else 0.0,
                        "median_abs_logit_drop": float(np.median(abs_logit_drop)),
                        "mean_normalized_drop": float(np.mean(norm_drop)),
                        "std_normalized_drop": float(np.std(norm_drop, ddof=1)) if norm_drop.size > 1 else 0.0,
                        "median_normalized_drop": float(np.median(norm_drop)),
                        "mean_intervened_logit": float(np.mean(logits)),
                        "mean_baseline_logit": baseline_mean,
                        "std_baseline_logit": baseline_std,
                    }
                )

    rows_df = pd.DataFrame(rows)
    summary_rows = []
    for (intervention, curve), group in rows_df.groupby(["intervention", "curve"], sort=False):
        group = group.sort_values("changed_frac")
        summary_rows.append(
            {
                "intervention": intervention,
                "curve": curve,
                "auc_mean_normalized_drop": _normalized_auc(
                    group["changed_frac"].to_numpy(),
                    group["mean_normalized_drop"].to_numpy(),
                ),
                "auc_mean_logit_drop": _normalized_auc(
                    group["changed_frac"].to_numpy(),
                    group["mean_logit_drop"].to_numpy(),
                ),
                "auc_mean_abs_logit_drop": _normalized_auc(
                    group["changed_frac"].to_numpy(),
                    group["mean_abs_logit_drop"].to_numpy(),
                ),
                "max_mean_logit_drop": float(group["mean_logit_drop"].max()),
                "max_mean_normalized_drop": float(group["mean_normalized_drop"].max()),
                "mean_std_logit_drop": float(group["std_logit_drop"].mean()) if "std_logit_drop" in group else np.nan,
                "mean_std_normalized_drop": float(group["std_normalized_drop"].mean()) if "std_normalized_drop" in group else np.nan,
                "final_mean_logit_drop": float(group["mean_logit_drop"].iloc[-1]),
                "final_std_logit_drop": float(group["std_logit_drop"].iloc[-1]) if "std_logit_drop" in group else np.nan,
                "final_mean_normalized_drop": float(group["mean_normalized_drop"].iloc[-1]),
                "final_std_normalized_drop": float(group["std_normalized_drop"].iloc[-1]) if "std_normalized_drop" in group else np.nan,
                "final_changed_percent": float(group["changed_percent"].iloc[-1]),
                "mean_baseline_logit": float(group["mean_baseline_logit"].iloc[0]) if "mean_baseline_logit" in group else np.nan,
                "std_baseline_logit": float(group["std_baseline_logit"].iloc[0]) if "std_baseline_logit" in group else np.nan,
                "n_steps": int(len(group)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    result = {
        "layer_name": layer_name,
        "activation_shape_chw": shape,
        "n_examples": len(selected),
        "n_neurons": int(flat_order.size),
        "max_percent": float(max_percent),
        "step_percent": float(step_percent),
        "rows": rows_df,
        "summary": summary_df,
        "schedule": schedule,
        "rankings_cache_path": rankings.get("cache_path", ""),
        "cache_path": str(path),
        "loaded_from_cache": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh)
    return result


def _plot_sanity_check_curves(
    rows: pd.DataFrame,
    *,
    value_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    show_logit0: bool,
):
    import matplotlib.pyplot as plt

    colors = {"classifier": "#4C78A8", "detector": "#F58518"}
    linestyles = {"zero": "-", "opposite_extreme": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, curve in zip(axes, ["single_step", "cumulative"], strict=True):
        sub = rows[rows["curve"] == curve].sort_values("changed_frac")
        notes: list[str] = []
        group_cols = ["source", "intervention"] if "source" in sub.columns else ["intervention"]
        for key, group in sub.groupby(group_cols, sort=False):
            if isinstance(key, tuple):
                source, intervention = str(key[0]), str(key[1])
            else:
                source, intervention = "classifier", str(key)
            x = group["changed_percent"].to_numpy(dtype="float64")
            y = group[value_col].to_numpy(dtype="float64")
            std = (
                group[std_col].fillna(0.0).to_numpy(dtype="float64")
                if std_col in group.columns
                else np.zeros_like(y)
            )
            label = f"{source} / {intervention}"
            ax.plot(
                x,
                y,
                color=colors.get(source, "#4C78A8"),
                linestyle=linestyles.get(intervention, "-"),
                linewidth=1.5,
                label=label,
            )
            ax.fill_between(x, y - std, y + std, color=colors.get(source, "#4C78A8"), alpha=0.12, linewidth=0)
            auc = _normalized_auc(group["changed_frac"].to_numpy(), y)
            sigma = float(np.nanmean(std)) if std.size else float("nan")
            base = float(group["mean_baseline_logit"].iloc[0]) if "mean_baseline_logit" in group.columns else float("nan")
            base_std = float(group["std_baseline_logit"].iloc[0]) if "std_baseline_logit" in group.columns else float("nan")
            notes.append(f"{label}: AUC={auc:.3f}, СКО={sigma:.3f}, logit0={base:.3f}±{base_std:.3f}")
        if show_logit0 and "mean_baseline_logit" in sub.columns:
            baseline = float(pd.to_numeric(sub["mean_baseline_logit"], errors="coerce").dropna().iloc[0])
            ax.axhline(baseline, color="#777777", linestyle="--", linewidth=0.9, label="logit0 threshold")
            ax.text(0.995, baseline, f" logit0={baseline:.3f}", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color="#555555")
        ax.axhline(0.0, color="#BBBBBB", linestyle=":", linewidth=0.9)
        ax.set_title(curve.replace("_", " "))
        ax.set_xlabel("top important neurons, %")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        if notes:
            ax.text(
                0.01,
                0.99,
                "\n".join(notes[:6]),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.86},
            )
    fig.suptitle(title)
    return fig


def plot_sanity_check_curves(rows: pd.DataFrame):
    return _plot_sanity_check_curves(
        rows,
        value_col="mean_logit_drop",
        std_col="std_logit_drop",
        ylabel="mean logit drop = clean logit - intervened logit",
        title="Sanity check: raw logit drop after top-importance neuron interventions",
        show_logit0=True,
    )


def plot_sanity_check_unsigned_curves(rows: pd.DataFrame):
    return _plot_sanity_check_curves(
        rows,
        value_col="mean_abs_logit_drop",
        std_col="std_abs_logit_drop",
        ylabel="mean |logit drop|",
        title="Sanity check: unsigned raw logit change",
        show_logit0=False,
    )
