from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class CovarianceMethodConfig:
    name: str
    n_steps: int
    segment_start: float
    segment_end: float


SEGMENTIG_COVARIANCE = CovarianceMethodConfig(
    name="segmentig",
    n_steps=64,
    segment_start=0.0,
    segment_end=0.1,
)
FULL_IG_32_COVARIANCE = CovarianceMethodConfig(
    name="full_ig_32",
    n_steps=32,
    segment_start=0.0,
    segment_end=1.0,
)


def covariance_cache_dir(exp, method: CovarianceMethodConfig, *, layer_name: str | None = None) -> Path:
    layer = layer_name or exp.config.target_layer
    out = exp.derived_cache_dir / "covariance" / _safe_name(layer) / _safe_name(method.name)
    out.mkdir(parents=True, exist_ok=True)
    return out


def covariance_cache_path(exp, example, method: CovarianceMethodConfig, *, layer_name: str | None = None) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "path": str(example.path),
        "drop": float(example.drop),
        "success": bool(example.success),
        "target_layer": layer_name or exp.config.target_layer,
        "detect_layer": exp.config.detect_layer,
        "target_mode": exp.config.target_mode,
        "imgsz": int(exp.config.attack.imgsz),
        "method": method.name,
        "n_steps": int(method.n_steps),
        "segment_start": float(method.segment_start),
        "segment_end": float(method.segment_end),
        "version": 4,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
    return covariance_cache_dir(exp, method, layer_name=layer_name) / f"covariance_{key}.npz"


def compute_layer_path_covariance(
    model,
    inputs,
    baselines,
    *,
    target_fn,
    layer,
    n_steps: int,
    segment_start: float,
    segment_end: float,
    alpha_batch_size: int = 4,
    clear_every: int = 8,
    return_components: bool = False,
    compute_corr: bool = False,
) -> dict[str, Any]:
    """Compute per-neuron covariance of layer conductance integrand along an alpha path.

    For each selected alpha and each layer element j:
      a_j(alpha) = d target / d activation_j
      b_j(alpha) = J_activation_j(x_alpha) @ (inputs - baselines)

    The returned covariance is E[a*b] - E[a]E[b], with the same shape as the layer activation.
    """
    import gc

    import torch
    from torch.func import jvp

    from .attributions import _ActivationHook

    x = inputs.detach()
    x0 = baselines.detach()
    delta_x = (x - x0).contiguous()
    alphas = _midpoint_alphas(
        int(n_steps),
        segment_start=float(segment_start),
        segment_end=float(segment_end),
    )

    hook = _ActivationHook(layer)

    def clear_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def forward_with_activation(x_in):
        hook.clear()
        score = target_fn(model, x_in)
        if getattr(score, "ndim", 0) != 0:
            score = score.sum()
        activation = hook.get()
        return score, activation

    def flat_activation(x_in):
        _score, activation = forward_with_activation(x_in)
        return activation.reshape(activation.shape[0], -1)

    try:
        sum_a = None
        sum_b = None
        sum_ab = None
        sum_a2 = None
        sum_b2 = None
        activation_shape = None
        count = 0
        batch_size = int(x.shape[0])
        alpha_batch_size = max(1, int(alpha_batch_size))
        for start in range(0, int(alphas.size), alpha_batch_size):
            alpha_batch_np = alphas[start : start + alpha_batch_size]
            alpha_count = int(alpha_batch_np.size)
            alpha_batch = torch.as_tensor(alpha_batch_np, device=x.device, dtype=x.dtype)

            view_shape = (alpha_count,) + (1,) * x.ndim
            x_alpha = x0.unsqueeze(0) + alpha_batch.reshape(view_shape) * delta_x.unsqueeze(0)
            x_alpha = x_alpha.reshape((alpha_count * batch_size,) + tuple(x.shape[1:]))
            x_alpha = x_alpha.contiguous().detach().requires_grad_(True)

            score, activation = forward_with_activation(x_alpha)
            if activation_shape is None:
                activation_shape = tuple(int(v) for v in activation.shape[1:])
            grad_y = torch.autograd.grad(
                score,
                activation,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad_y is None:
                grad_y = torch.zeros_like(activation)

            delta_batch = delta_x.unsqueeze(0).expand(alpha_count, -1, *([-1] * (x.ndim - 1)))
            delta_batch = delta_batch.reshape_as(x_alpha).contiguous()
            _, dir_flat = jvp(flat_activation, (x_alpha,), (delta_batch,))
            dir_y = dir_flat.reshape_as(activation)

            a_flat = grad_y.detach().reshape(alpha_count, batch_size, -1)
            b_flat = dir_y.detach().reshape(alpha_count, batch_size, -1)
            ab_flat = a_flat * b_flat

            cur_sum_a = a_flat.sum(dim=0)
            cur_sum_b = b_flat.sum(dim=0)
            cur_sum_ab = ab_flat.sum(dim=0)
            if compute_corr or return_components:
                cur_sum_a2 = (a_flat * a_flat).sum(dim=0)
                cur_sum_b2 = (b_flat * b_flat).sum(dim=0)
            else:
                cur_sum_a2 = None
                cur_sum_b2 = None
            if sum_a is None:
                sum_a = cur_sum_a
                sum_b = cur_sum_b
                sum_ab = cur_sum_ab
                sum_a2 = cur_sum_a2
                sum_b2 = cur_sum_b2
            else:
                sum_a = sum_a + cur_sum_a
                sum_b = sum_b + cur_sum_b
                sum_ab = sum_ab + cur_sum_ab
                if cur_sum_a2 is not None:
                    sum_a2 = sum_a2 + cur_sum_a2
                    sum_b2 = sum_b2 + cur_sum_b2
            count += alpha_count

            del x_alpha, score, activation, grad_y, dir_y, delta_batch
            del a_flat, b_flat, ab_flat, cur_sum_a, cur_sum_b, cur_sum_ab, cur_sum_a2, cur_sum_b2
            hook.clear()
            if clear_every > 0 and count % int(clear_every) == 0:
                clear_memory()

        if count == 0 or sum_a is None or activation_shape is None:
            raise RuntimeError("No alpha samples were evaluated.")

        denom = float(count)
        mean_a = sum_a / denom
        mean_b = sum_b / denom
        mean_ab = sum_ab / denom
        cov = mean_ab - mean_a * mean_b
        cov = cov.reshape((batch_size,) + activation_shape)
        corr = None
        var_a = None
        var_b = None
        if compute_corr or return_components:
            var_a = (sum_a2 / denom) - mean_a * mean_a
            var_b = (sum_b2 / denom) - mean_b * mean_b
            corr = cov.reshape(batch_size, -1) / (
                torch.sqrt(var_a.clamp_min(0.0)) * torch.sqrt(var_b.clamp_min(0.0)) + 1e-12
            )
            corr = torch.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
            corr = corr.reshape((batch_size,) + activation_shape)

        out = {
            "cov": cov.detach(),
            "corr": corr.detach() if corr is not None else None,
            "activation_shape": activation_shape,
            "alphas": alphas.astype("float32", copy=False),
        }
        if return_components:
            out.update(
                {
                    "mean_a": mean_a.reshape((batch_size,) + activation_shape).detach(),
                    "mean_b": mean_b.reshape((batch_size,) + activation_shape).detach(),
                    "mean_ab": mean_ab.reshape((batch_size,) + activation_shape).detach(),
                    "var_a": var_a.reshape((batch_size,) + activation_shape).detach(),
                    "var_b": var_b.reshape((batch_size,) + activation_shape).detach(),
                }
            )
        return out
    finally:
        hook.remove()
        clear_memory()


def compute_or_load_covariance_for_example(
    exp,
    example,
    method: CovarianceMethodConfig,
    *,
    model,
    layer,
    layer_name: str | None = None,
    ctx: dict[str, Any] | None = None,
    force: bool = False,
    save_components: bool = False,
    clear_every: int = 8,
    alpha_batch_size: int = 4,
    compute_corr: bool = False,
) -> dict[str, Any]:
    path = covariance_cache_path(exp, example, method, layer_name=layer_name)
    if path.exists() and not force:
        cached = _load_covariance_npz(path)
        cached["loaded_from_cache"] = True
        return cached

    if ctx is None:
        ctx = exp._context_for_example(example, image_variant="clean")
    result = compute_layer_path_covariance(
        model,
        ctx["inputs"],
        ctx["baselines"],
        target_fn=ctx["target_fn"],
        layer=layer,
        n_steps=int(method.n_steps),
        segment_start=float(method.segment_start),
        segment_end=float(method.segment_end),
        alpha_batch_size=int(alpha_batch_size),
        clear_every=int(clear_every),
        return_components=bool(save_components),
        compute_corr=bool(compute_corr),
    )

    def as_chw(tensor):
        arr = tensor.detach().cpu().numpy()
        if arr.ndim == 4:
            arr = arr[0]
        return arr.astype("float32", copy=False)

    cov_chw = as_chw(result["cov"]).astype("float16")
    if result.get("corr") is None:
        corr_chw = np.asarray([np.nan], dtype="float16")
    else:
        corr_chw = as_chw(result["corr"]).astype("float16")
    payload = {
        "cov_chw": cov_chw,
        "corr_chw": corr_chw,
        "activation_shape": np.asarray(result["activation_shape"], dtype="int64"),
        "alphas": np.asarray(result["alphas"], dtype="float32"),
        "drop": np.asarray([float(example.drop)], dtype="float32"),
        "success": np.asarray([bool(example.success)], dtype="bool"),
        "path": np.asarray([str(example.path)]),
    }
    if save_components:
        for key in ("mean_a", "mean_b", "mean_ab", "var_a", "var_b"):
            payload[key] = as_chw(result[key]).astype("float16")
    np.savez_compressed(path, **payload)
    return {
        "cov_chw": payload["cov_chw"].astype("float32", copy=False),
        "corr_chw": payload["corr_chw"].astype("float32", copy=False),
        "activation_shape": tuple(int(v) for v in payload["activation_shape"].tolist()),
        "alphas": payload["alphas"],
        "cache_path": str(path),
        "loaded_from_cache": False,
    }


def _load_covariance_npz(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return {
            "cov_chw": data["cov_chw"].astype("float32", copy=False),
            "corr_chw": data["corr_chw"].astype("float32", copy=False),
            "activation_shape": tuple(int(v) for v in data["activation_shape"].tolist()),
            "alphas": data["alphas"].astype("float32", copy=False),
            "cache_path": str(path),
        }


def compute_or_load_covariance_dataset(
    exp,
    *,
    methods: tuple[CovarianceMethodConfig, ...] = (SEGMENTIG_COVARIANCE, FULL_IG_32_COVARIANCE),
    examples: list[Any] | None = None,
    max_examples: int | None = None,
    layer_name: str | None = None,
    force: bool = False,
    save_components: bool = False,
    clear_every: int = 8,
    alpha_batch_size: int | None = None,
    compute_corr: bool = False,
) -> dict[str, Any]:
    from tqdm.auto import tqdm

    selected = exp._selected_examples(examples, max_examples)
    _yolo, model = exp.load_model()
    layer_key = layer_name or exp.config.target_layer
    modules = dict(model.named_modules())
    if layer_key not in modules:
        raise KeyError(f"Layer {layer_key!r} was not found.")
    layer = modules[layer_key]

    rows = []
    errors = []
    effective_alpha_batch_size = int(alpha_batch_size or getattr(exp.config, "alpha_batch_size", 4))
    iterator = tqdm(selected, desc="covariance", unit="img")
    for example in iterator:
        cached_or_missing = []
        for method in methods:
            cache_path = covariance_cache_path(exp, example, method, layer_name=layer_key)
            if cache_path.exists() and not force:
                try:
                    maps = _load_covariance_npz(cache_path)
                    maps["loaded_from_cache"] = True
                    rows.append(_covariance_row(example, method, maps))
                except Exception as exc:
                    errors.append({"path": str(example.path), "method": method.name, "error": repr(exc)})
                    iterator.set_postfix(skipped=len(errors), last_error=type(exc).__name__)
            else:
                cached_or_missing.append(method)
        if not cached_or_missing:
            continue

        try:
            ctx = exp._context_for_example(example, image_variant="clean")
        except Exception as exc:
            for method in cached_or_missing:
                errors.append({"path": str(example.path), "method": method.name, "error": repr(exc)})
            iterator.set_postfix(skipped=len(errors), last_error=type(exc).__name__)
            exp._release_batch_memory()
            continue

        for method in cached_or_missing:
            try:
                maps = compute_or_load_covariance_for_example(
                    exp,
                    example,
                    method,
                    model=model,
                    layer=layer,
                    layer_name=layer_key,
                    ctx=ctx,
                    force=force,
                    save_components=save_components,
                    clear_every=clear_every,
                    alpha_batch_size=effective_alpha_batch_size,
                    compute_corr=bool(compute_corr),
                )
                rows.append(_covariance_row(example, method, maps))
            except Exception as exc:
                errors.append({"path": str(example.path), "method": method.name, "error": repr(exc)})
                iterator.set_postfix(skipped=len(errors), last_error=type(exc).__name__)
        exp._release_batch_memory()

    return {
        "rows": pd.DataFrame(rows),
        "errors": pd.DataFrame(errors),
        "methods": methods,
        "layer_name": layer_key,
        "n_examples": len(selected),
    }


def _covariance_row(example, method: CovarianceMethodConfig, maps: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(example.path),
        "success": bool(example.success),
        "drop": float(example.drop),
        "method": method.name,
        "n_steps": int(method.n_steps),
        "segment_start": float(method.segment_start),
        "segment_end": float(method.segment_end),
        "cache_path": maps["cache_path"],
        "loaded_from_cache": bool(maps["loaded_from_cache"]),
        **covariance_tensor_summary(maps["cov_chw"], prefix="cov"),
        **covariance_tensor_summary(maps["corr_chw"], prefix="corr"),
    }


def covariance_tensor_summary(values, *, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype="float32")
    flat = arr.reshape(-1)
    finite = np.isfinite(flat)
    if not finite.any():
        return {
            f"{prefix}_{name}": float("nan")
            for name in ("mean", "std", "mean_abs", "max_abs", "sum", "sum_abs", "positive_frac")
        }
    v = flat[finite].astype("float64", copy=False)
    return {
        f"{prefix}_mean": float(v.mean()),
        f"{prefix}_std": float(v.std()),
        f"{prefix}_mean_abs": float(np.abs(v).mean()),
        f"{prefix}_max_abs": float(np.abs(v).max()),
        f"{prefix}_sum": float(v.sum()),
        f"{prefix}_sum_abs": float(np.abs(v).sum()),
        f"{prefix}_positive_frac": float((v > 0).mean()),
    }


def aggregate_covariance_by_rank(
    rows: pd.DataFrame,
    *,
    method: str,
    n_bins: int = 200,
    signed: bool = False,
) -> pd.DataFrame:
    acc_sum = None
    acc_sq = None
    count = 0
    for cache_path in rows.loc[rows["method"] == method, "cache_path"].astype(str):
        with np.load(cache_path, allow_pickle=False) as data:
            cov = data["cov_chw"].astype("float32", copy=False).reshape(-1)
        score = cov if signed else np.abs(cov)
        order = np.argsort(score)[::-1]
        ranked = cov[order] if signed else score[order]
        binned = _binned_means(ranked, n_bins=n_bins)
        if acc_sum is None:
            acc_sum = np.zeros_like(binned, dtype="float64")
            acc_sq = np.zeros_like(binned, dtype="float64")
        acc_sum += binned
        acc_sq += binned * binned
        count += 1
    if count == 0 or acc_sum is None or acc_sq is None:
        return pd.DataFrame(columns=["method", "rank_bin", "rank_frac", "mean", "std", "n"])
    mean = acc_sum / count
    std = np.sqrt(np.maximum(acc_sq / count - mean * mean, 0.0))
    return pd.DataFrame(
        {
            "method": method,
            "rank_bin": np.arange(1, n_bins + 1),
            "rank_frac": (np.arange(n_bins) + 0.5) / float(n_bins),
            "mean": mean,
            "std": std,
            "n": count,
        }
    )


def aggregate_covariance_distribution(
    rows: pd.DataFrame,
    *,
    methods: tuple[str, ...] | None = None,
    bins: int = 200,
    value_mode: str = "abs",
    sample_per_image: int = 20000,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    method_names = methods or tuple(rows["method"].dropna().unique().tolist())
    out = []
    for method in method_names:
        samples = []
        for cache_path in rows.loc[rows["method"] == method, "cache_path"].astype(str):
            with np.load(cache_path, allow_pickle=False) as data:
                cov = data["cov_chw"].astype("float32", copy=False).reshape(-1)
            if value_mode == "abs":
                values = np.abs(cov)
            elif value_mode == "signed":
                values = cov
            elif value_mode == "log_abs":
                values = np.log1p(np.abs(cov))
            else:
                raise ValueError(f"Unsupported value_mode: {value_mode!r}")
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            if sample_per_image and finite.size > int(sample_per_image):
                idx = rng.choice(finite.size, size=int(sample_per_image), replace=False)
                finite = finite[idx]
            samples.append(finite.astype("float64", copy=False))
        if not samples:
            continue
        vals = np.concatenate(samples)
        hist, edges = np.histogram(vals, bins=int(bins), density=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        out.append(pd.DataFrame({"method": method, "x": centers, "density": hist, "value_mode": value_mode}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["method", "x", "density", "value_mode"])


def sample_covariance_values(
    rows: pd.DataFrame,
    *,
    methods: tuple[str, ...] | None = None,
    value_mode: str = "signed",
    sample_per_image: int = 20000,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    method_names = methods or tuple(rows["method"].dropna().unique().tolist())
    out = []
    for method in method_names:
        parts = []
        for cache_path in rows.loc[rows["method"] == method, "cache_path"].astype(str):
            with np.load(cache_path, allow_pickle=False) as data:
                cov = data["cov_chw"].astype("float32", copy=False).reshape(-1)
            if value_mode == "abs":
                values = np.abs(cov)
            elif value_mode == "signed":
                values = cov
            elif value_mode == "log_abs":
                values = np.log1p(np.abs(cov))
            else:
                raise ValueError(f"Unsupported value_mode: {value_mode!r}")
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            if sample_per_image and finite.size > int(sample_per_image):
                idx = rng.choice(finite.size, size=int(sample_per_image), replace=False)
                finite = finite[idx]
            parts.append(finite.astype("float32", copy=False))
        if parts:
            out.append(pd.DataFrame({"method": method, "value": np.concatenate(parts), "value_mode": value_mode}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["method", "value", "value_mode"])


def plot_covariance_overview(rows: pd.DataFrame):
    import matplotlib.pyplot as plt

    metric_cols = ["cov_mean_abs", "cov_max_abs", "cov_sum_abs", "corr_mean_abs"]
    grouped = rows.groupby("method")[metric_cols].agg(["mean", "std"]).reset_index()
    methods = grouped["method"].astype(str).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    for ax, metric in zip(axes, metric_cols):
        means = grouped[(metric, "mean")].to_numpy(dtype="float64")
        stds = grouped[(metric, "std")].to_numpy(dtype="float64")
        ax.bar(methods, means, yerr=stds, capsize=4, color="#4C78A8", alpha=0.85)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Per-image covariance summaries, mean ± std")
    fig.tight_layout()
    return fig


def plot_covariance_distributions(dist_df: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for method, sub in dist_df.groupby("method"):
        ax.plot(sub["x"], sub["density"], label=str(method), linewidth=2)
    ax.set_xlabel(dist_df["value_mode"].iloc[0] if len(dist_df) else "value")
    ax.set_ylabel("density")
    ax.set_title("Covariance value distribution")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_covariance_zero_zoom(
    values_df: pd.DataFrame,
    *,
    q: float = 99.0,
    bins: int = 200,
    symmetric: bool = True,
    drop_zeros: bool = False,
    zero_eps: float = 0.0,
):
    import matplotlib.pyplot as plt

    if values_df.empty:
        raise ValueError("values_df is empty")
    finite = values_df[np.isfinite(values_df["value"].to_numpy(dtype="float64"))].copy()
    if finite.empty:
        raise ValueError("values_df has no finite values")
    zero_mask = np.abs(finite["value"].to_numpy(dtype="float64")) <= float(zero_eps)
    zero_frac_by_method = finite.assign(_is_zero=zero_mask).groupby("method")["_is_zero"].mean().to_dict()
    if drop_zeros:
        finite = finite.loc[~zero_mask].copy()
        if finite.empty:
            raise ValueError("No non-zero covariance values remain after drop_zeros=True")
    if symmetric:
        limit = float(np.nanpercentile(np.abs(finite["value"].to_numpy(dtype="float64")), float(q)))
        lo, hi = -limit, limit
    else:
        tail = (100.0 - float(q)) / 2.0
        lo = float(np.nanpercentile(finite["value"], tail))
        hi = float(np.nanpercentile(finite["value"], 100.0 - tail))
    zoom = finite[(finite["value"] >= lo) & (finite["value"] <= hi)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for method, sub in zoom.groupby("method"):
        axes[0].hist(sub["value"], bins=int(bins), density=True, alpha=0.35, label=str(method))
        xs = np.sort(sub["value"].to_numpy(dtype="float64"))
        if xs.size:
            ys = np.linspace(0.0, 1.0, xs.size, endpoint=True)
            axes[1].plot(xs, ys, label=str(method), linewidth=2)
    zero_note = ", zero bin removed" if drop_zeros else ""
    axes[0].set_title(f"Signed covariance near zero ({q:g}% central window{zero_note})")
    axes[0].set_xlabel("covariance")
    axes[0].set_ylabel("density")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_title("ECDF near zero")
    axes[1].set_xlabel("covariance")
    axes[1].set_ylabel("cumulative fraction")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    if drop_zeros and zero_frac_by_method:
        note = "zero fraction: " + ", ".join(f"{k}={v:.2%}" for k, v in zero_frac_by_method.items())
        fig.text(0.01, 0.01, note, fontsize=9, ha="left", va="bottom")
    fig.tight_layout()
    return fig


def plot_covariance_rank_curves(rank_df: pd.DataFrame, *, title: str = "Covariance by per-image rank"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for method, sub in rank_df.groupby("method"):
        x = sub["rank_frac"].to_numpy(dtype="float64")
        y = sub["mean"].to_numpy(dtype="float64")
        s = sub["std"].to_numpy(dtype="float64")
        ax.plot(x, y, label=str(method), linewidth=2)
        ax.fill_between(x, y - s, y + s, alpha=0.15)
    ax.set_xlabel("rank fraction after sorting by |covariance|")
    ax.set_ylabel("mean |covariance|")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def covariance_sigma_thresholds(
    rows: pd.DataFrame,
    *,
    sigmas: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
    zero_eps: float = 0.0,
) -> pd.DataFrame:
    """Estimate global |covariance| thresholds per method as mean + k * std.

    Exact zeros are excluded from mean/std by default, but the resulting thresholds
    are later applied to all neurons, so zero-covariance neurons naturally land in
    the low-covariance group.
    """
    out: list[dict[str, float | str]] = []
    if rows.empty:
        return pd.DataFrame(
            columns=["method", "sigma", "threshold", "mean_abs_nonzero", "std_abs_nonzero", "nonzero_count", "zero_frac"]
        )
    for method, sub in rows.groupby("method"):
        total_count = 0
        nonzero_count = 0
        sum_abs = 0.0
        sum_abs2 = 0.0
        for cache_path in sub["cache_path"].astype(str):
            with np.load(cache_path, allow_pickle=False) as data:
                values = np.abs(data["cov_chw"].astype("float32", copy=False).reshape(-1))
            finite = values[np.isfinite(values)]
            total_count += int(finite.size)
            nonzero = finite[finite > float(zero_eps)].astype("float64", copy=False)
            nonzero_count += int(nonzero.size)
            sum_abs += float(nonzero.sum())
            sum_abs2 += float(np.dot(nonzero, nonzero))
        if nonzero_count == 0:
            mean = float("nan")
            std = float("nan")
        else:
            mean = sum_abs / float(nonzero_count)
            var = max(0.0, sum_abs2 / float(nonzero_count) - mean * mean)
            std = float(np.sqrt(var))
        zero_frac = 1.0 - float(nonzero_count) / float(total_count) if total_count else float("nan")
        for sigma in sigmas:
            out.append(
                {
                    "method": str(method),
                    "sigma": float(sigma),
                    "threshold": float(mean + float(sigma) * std) if np.isfinite(mean) and np.isfinite(std) else float("nan"),
                    "mean_abs_nonzero": float(mean),
                    "std_abs_nonzero": float(std),
                    "nonzero_count": float(nonzero_count),
                    "total_count": float(total_count),
                    "zero_frac": float(zero_frac),
                }
            )
    return pd.DataFrame(out)


def compute_or_load_covariance_split_metrics(
    exp,
    cov_rows: pd.DataFrame,
    examples: list[Any],
    *,
    sigmas: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0),
    layer_name: str | None = None,
    top_percent: float = 5.0,
    rank_bins: int = 100,
    include_spatial: bool = True,
    zero_eps: float = 0.0,
    force: bool = False,
) -> dict[str, Any]:
    from tqdm.auto import tqdm

    from .activations import delta_spread_metrics
    from .metrics import (
        alignment_metrics,
        handcrafted_delta_importance_features,
        importance_rank_bin_energy_fractions,
        metric_quality_rows,
        segmentig_soft_alignment_metrics,
    )
    from .spread_precision import _clean_bbox, object_bbox_delta_metrics, patch_excluded_spread_metrics

    layer_key = layer_name or exp.config.target_layer
    selected = list(examples)
    requested_sigmas = tuple(float(value) for value in sigmas)
    cov_rows = cov_rows.copy()
    path = _covariance_split_metrics_cache_path(
        exp,
        cov_rows,
        selected,
        sigmas=requested_sigmas,
        layer_name=layer_key,
        top_percent=float(top_percent),
        rank_bins=int(rank_bins),
        include_spatial=bool(include_spatial),
        zero_eps=float(zero_eps),
    )
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached

    cached_chunks = (
        []
        if force
        else _load_covariance_split_metric_chunks(
            exp,
            cov_rows,
            selected,
            requested_sigmas=requested_sigmas,
            layer_name=layer_key,
            top_percent=float(top_percent),
            rank_bins=int(rank_bins),
            include_spatial=bool(include_spatial),
            zero_eps=float(zero_eps),
        )
    )
    cached_sigmas = {
        float(sigma)
        for chunk in cached_chunks
        for sigma in chunk["rows"].get("sigma", pd.Series(dtype="float64")).dropna().unique().tolist()
    }
    sigmas_to_compute = tuple(sigma for sigma in requested_sigmas if sigma not in cached_sigmas)
    if cached_chunks and not sigmas_to_compute and not force:
        combined = _combine_covariance_split_metric_results(cached_chunks, cache_path=path)
        combined["loaded_from_cache"] = True
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(combined, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return combined

    thresholds = covariance_sigma_thresholds(cov_rows, sigmas=sigmas_to_compute, zero_eps=float(zero_eps))
    threshold_by_key = {
        (str(row.method), float(row.sigma)): float(row.threshold)
        for row in thresholds.itertuples(index=False)
        if np.isfinite(float(row.threshold))
    }
    cov_by_path: dict[str, list[dict[str, Any]]] = {}
    for record in cov_rows.to_dict("records"):
        cov_by_path.setdefault(str(record["path"]), []).append(record)

    rows_out: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    imgsz = int(exp.config.attack.imgsz)
    iterator = tqdm(selected, desc="cov split metrics", unit="img")
    for example in iterator:
        ex_path = str(example.path)
        method_records = cov_by_path.get(ex_path, [])
        if not method_records:
            skipped.append({"path": ex_path, "reason": "missing covariance row"})
            continue
        layer_maps = exp._load_layer_map_cache(example, layer_name=layer_key, include_clean_activation=False)
        if layer_maps is None:
            skipped.append({"path": ex_path, "reason": "missing layer_maps cache"})
            continue
        try:
            delta_chw = _as_chw_float32(layer_maps["delta_chw"])
            importance_chw = _as_chw_float32(layer_maps["segmentig_chw"])
        except Exception as exc:
            skipped.append({"path": ex_path, "reason": f"bad layer_maps: {exc!r}"})
            continue

        d_flat_full = delta_chw.reshape(-1)
        a_flat_full = importance_chw.reshape(-1)
        for record in method_records:
            method = str(record["method"])
            try:
                with np.load(str(record["cache_path"]), allow_pickle=False) as data:
                    cov_chw = _as_chw_float32(data["cov_chw"])
            except Exception as exc:
                skipped.append({"path": ex_path, "method": method, "reason": f"bad covariance cache: {exc!r}"})
                continue

            n = min(cov_chw.size, d_flat_full.size, a_flat_full.size)
            cov_flat = cov_chw.reshape(-1)[:n]
            d_flat = d_flat_full[:n]
            a_flat = a_flat_full[:n]
            cov_abs = np.abs(cov_flat)
            finite_cov = np.isfinite(cov_abs)
            cov_mass_total = float(np.nansum(cov_abs[finite_cov]) + 1e-12)
            delta_mass_total = float(np.nansum(np.abs(d_flat)) + 1e-12)
            importance_mass_total = float(np.nansum(np.abs(a_flat)) + 1e-12)

            for sigma in sigmas_to_compute:
                threshold = threshold_by_key.get((method, float(sigma)), float("nan"))
                if not np.isfinite(threshold):
                    continue
                masks = {
                    "low": finite_cov & (cov_abs < float(threshold)),
                    "high": finite_cov & (cov_abs >= float(threshold)),
                }
                for group, mask in masks.items():
                    selected_count = int(mask.sum())
                    metric_values: dict[str, float] = {
                        "cov_selected_count": float(selected_count),
                        "cov_selected_frac": float(selected_count / max(1, n)),
                        "cov_abs_mass_frac": float(np.nansum(cov_abs[mask]) / cov_mass_total),
                        "delta_abs_mass_frac": float(np.nansum(np.abs(d_flat[mask])) / delta_mass_total),
                        "importance_abs_mass_frac": float(np.nansum(np.abs(a_flat[mask])) / importance_mass_total),
                    }
                    if selected_count > 0:
                        sub_d = d_flat[mask]
                        sub_a = a_flat[mask]
                        metric_values.update(importance_energy_focus_metrics(sub_d, sub_a))
                        metric_values.update(_prefix("align", alignment_metrics(sub_d, sub_a, top_percent=float(top_percent))))
                        metric_values.update(_prefix("soft", segmentig_soft_alignment_metrics(sub_d, sub_a)))
                        bin_fracs = importance_rank_bin_energy_fractions(sub_d, sub_a, n_bins=int(rank_bins))
                        for i, value in enumerate(bin_fracs, start=1):
                            metric_values[f"rank_bin_{i:03d}_delta_frac"] = float(value)

                        if include_spatial:
                            mask_chw = mask.reshape(cov_chw.reshape(-1)[:n].shape)
                            full_mask = np.zeros(delta_chw.size, dtype=bool)
                            full_mask[:n] = mask_chw
                            full_mask = full_mask.reshape(delta_chw.shape)
                            masked_delta = np.where(full_mask, delta_chw, 0.0).astype("float32", copy=False)
                            masked_importance = np.where(full_mask, importance_chw, 0.0).astype("float32", copy=False)
                            spatial = {}
                            spatial.update(
                                _prefix(
                                    "masked_spread",
                                    delta_spread_metrics(
                                        masked_delta,
                                        patch_bbox_xyxy=example.patch_bbox_lb,
                                        imgsz=imgsz,
                                    ),
                                )
                            )
                            spatial.update(
                                _prefix(
                                    "masked",
                                    object_bbox_delta_metrics(masked_delta, _clean_bbox(example), imgsz=imgsz),
                                )
                            )
                            spatial.update(
                                _prefix(
                                    "masked_nopatch",
                                    patch_excluded_spread_metrics(
                                        masked_delta,
                                        masked_importance,
                                        object_bbox_xyxy=_clean_bbox(example),
                                        patch_bbox_xyxy=example.patch_bbox_lb,
                                        imgsz=imgsz,
                                    ),
                                )
                            )
                            spatial.update(
                                _prefix(
                                    "masked",
                                    handcrafted_delta_importance_features(
                                        masked_delta,
                                        masked_importance,
                                        patch_bbox_xyxy=example.patch_bbox_lb,
                                        imgsz=imgsz,
                                        rank_bins=int(rank_bins),
                                    ),
                                )
                            )
                            metric_values.update(spatial)

                    rows_out.append(
                        {
                            "path": ex_path,
                            "success": bool(example.success),
                            "drop": float(example.drop),
                            "conf_clean": float(example.conf_clean),
                            "conf_patch": float(example.conf_patch),
                            "cov_method": method,
                            "sigma": float(sigma),
                            "cov_group": group,
                            "cov_threshold": float(threshold),
                            "cov_cache_path": str(record["cache_path"]),
                            "layer_maps_cache_path": str(layer_maps.get("cache_path", "")),
                            **metric_values,
                        }
                    )

    rows_df = pd.DataFrame(rows_out)
    if rows_df.empty:
        raise RuntimeError(f"No covariance split metric rows; skipped={len(skipped)}")
    meta_cols = {
        "path",
        "success",
        "drop",
        "conf_clean",
        "conf_patch",
        "cov_method",
        "sigma",
        "cov_group",
        "cov_threshold",
        "cov_cache_path",
        "layer_maps_cache_path",
    }
    metric_cols = [
        col
        for col in rows_df.columns
        if col not in meta_cols and pd.api.types.is_numeric_dtype(rows_df[col])
    ]
    quality_parts = []
    for (method, sigma, group), sub in rows_df.groupby(["cov_method", "sigma", "cov_group"]):
        labels = sub["success"].astype(bool).to_numpy()
        metrics_by_name = {col: sub[col].to_numpy(dtype="float64", copy=False) for col in metric_cols}
        quality = pd.DataFrame(metric_quality_rows(labels, metrics_by_name))
        if quality.empty:
            continue
        quality["cov_method"] = str(method)
        quality["sigma"] = float(sigma)
        quality["cov_group"] = str(group)
        quality["roc_auc_effective"] = quality["roc_auc"].map(lambda v: max(float(v), 1.0 - float(v)) if np.isfinite(v) else np.nan)
        quality_parts.append(quality)
    quality_df = (
        pd.concat(quality_parts, ignore_index=True)
        if quality_parts
        else pd.DataFrame(columns=["metric", "roc_auc", "best_balanced_accuracy", "best_accuracy", "best_threshold", "best_direction"])
    )
    if not quality_df.empty:
        quality_df = quality_df.sort_values(["best_balanced_accuracy", "roc_auc_effective", "best_accuracy"], ascending=False).reset_index(drop=True)

    result = {
        "rows": rows_df,
        "quality": quality_df,
        "thresholds": thresholds,
        "skipped": pd.DataFrame(skipped),
        "metric_cols": metric_cols,
        "loaded_from_cache": False,
        "cache_path": str(path),
    }
    for sigma in sigmas_to_compute:
        sigma_result = _subset_covariance_split_result(result, sigma=float(sigma))
        sigma_path = _covariance_split_metrics_cache_path(
            exp,
            cov_rows,
            selected,
            sigmas=(float(sigma),),
            layer_name=layer_key,
            top_percent=float(top_percent),
            rank_bins=int(rank_bins),
            include_spatial=bool(include_spatial),
            zero_eps=float(zero_eps),
        )
        sigma_result["cache_path"] = str(sigma_path)
        sigma_path.parent.mkdir(parents=True, exist_ok=True)
        with sigma_path.open("wb") as fh:
            pickle.dump(sigma_result, fh, protocol=pickle.HIGHEST_PROTOCOL)

    if cached_chunks:
        result = _combine_covariance_split_metric_results([*cached_chunks, result], cache_path=path)
        result["loaded_from_cache"] = False

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def compute_or_load_importance_energy_focus_metrics(
    exp,
    examples: list[Any],
    *,
    layer_name: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    from tqdm.auto import tqdm

    from .metrics import metric_quality_rows

    layer_key = layer_name or exp.config.target_layer
    selected = list(examples)
    path = _importance_energy_focus_cache_path(exp, selected, layer_name=layer_key)
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for example in tqdm(selected, desc="importance energy focus", unit="img"):
        layer_maps = exp._load_layer_map_cache(example, layer_name=layer_key, include_clean_activation=False)
        if layer_maps is None:
            skipped.append({"path": str(example.path), "reason": "missing layer_maps cache"})
            continue
        try:
            delta = _as_chw_float32(layer_maps["delta_chw"]).reshape(-1)
            importance = _as_chw_float32(layer_maps["segmentig_chw"]).reshape(-1)
        except Exception as exc:
            skipped.append({"path": str(example.path), "reason": f"bad layer_maps: {exc!r}"})
            continue
        rows.append(
            {
                "path": str(example.path),
                "success": bool(example.success),
                "drop": float(example.drop),
                "conf_clean": float(example.conf_clean),
                "conf_patch": float(example.conf_patch),
                "layer_maps_cache_path": str(layer_maps.get("cache_path", "")),
                **importance_energy_focus_metrics(delta, importance),
            }
        )
    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        raise RuntimeError(f"No importance energy focus rows; skipped={len(skipped)}")
    meta_cols = {"path", "success", "drop", "conf_clean", "conf_patch", "layer_maps_cache_path"}
    metric_cols = [col for col in rows_df.columns if col not in meta_cols and pd.api.types.is_numeric_dtype(rows_df[col])]
    quality = pd.DataFrame(
        metric_quality_rows(
            rows_df["success"].astype(bool).to_numpy(),
            {col: rows_df[col].to_numpy(dtype="float64", copy=False) for col in metric_cols},
        )
    )
    if not quality.empty:
        quality["roc_auc_effective"] = quality["roc_auc"].map(lambda v: max(float(v), 1.0 - float(v)) if np.isfinite(v) else np.nan)
        quality = quality.sort_values(["best_balanced_accuracy", "roc_auc_effective", "best_accuracy"], ascending=False).reset_index(drop=True)
    result = {
        "rows": rows_df,
        "quality": quality,
        "skipped": pd.DataFrame(skipped),
        "metric_cols": metric_cols,
        "loaded_from_cache": False,
        "cache_path": str(path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def covariance_threshold_candidate_table(
    cov_rows: pd.DataFrame,
    *,
    sigmas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
    quantiles: tuple[float, ...] = (50.0, 60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 92.5, 95.0, 97.0, 97.5, 98.0, 99.0, 99.2, 99.5, 99.7, 99.9),
    top_fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
    sample_per_image: int = 20000,
    seed: int = 0,
    zero_eps: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    samples: list[np.ndarray] = []
    total_count = 0
    nonzero_count = 0
    sum_abs = 0.0
    sum_abs2 = 0.0
    for cache_path in cov_rows["cache_path"].astype(str):
        with np.load(cache_path, allow_pickle=False) as data:
            values = np.abs(data["cov_chw"].astype("float32", copy=False).reshape(-1))
        finite = values[np.isfinite(values)]
        total_count += int(finite.size)
        nonzero = finite[finite > float(zero_eps)].astype("float64", copy=False)
        nonzero_count += int(nonzero.size)
        sum_abs += float(nonzero.sum())
        sum_abs2 += float(np.dot(nonzero, nonzero))
        if nonzero.size:
            if sample_per_image and nonzero.size > int(sample_per_image):
                idx = rng.choice(nonzero.size, size=int(sample_per_image), replace=False)
                nonzero = nonzero[idx]
            samples.append(nonzero.astype("float32", copy=False))
    if nonzero_count == 0 or not samples:
        return pd.DataFrame(columns=["threshold_kind", "threshold_label", "threshold", "source_value"])
    sampled = np.concatenate(samples).astype("float64", copy=False)
    mean = sum_abs / float(nonzero_count)
    std = float(np.sqrt(max(0.0, sum_abs2 / float(nonzero_count) - mean * mean)))
    rows: list[dict[str, Any]] = []
    for sigma in sigmas:
        threshold = float(mean + float(sigma) * std)
        rows.append(
            {
                "threshold_kind": "sigma",
                "threshold_label": f"mean+{float(sigma):g}sigma",
                "threshold": threshold,
                "source_value": float(sigma),
                "mean_abs_nonzero": float(mean),
                "std_abs_nonzero": float(std),
                "nonzero_count": float(nonzero_count),
                "total_count": float(total_count),
                "zero_frac": 1.0 - float(nonzero_count) / float(total_count) if total_count else float("nan"),
            }
        )
    for q in quantiles:
        threshold = float(np.nanpercentile(sampled, float(q)))
        rows.append(
            {
                "threshold_kind": "quantile",
                "threshold_label": f"q{float(q):g}",
                "threshold": threshold,
                "source_value": float(q),
                "mean_abs_nonzero": float(mean),
                "std_abs_nonzero": float(std),
                "nonzero_count": float(nonzero_count),
                "total_count": float(total_count),
                "zero_frac": 1.0 - float(nonzero_count) / float(total_count) if total_count else float("nan"),
            }
        )
    for frac in top_fractions:
        q = 100.0 - float(frac)
        threshold = float(np.nanpercentile(sampled, q))
        rows.append(
            {
                "threshold_kind": "top_fraction",
                "threshold_label": f"top{float(frac):g}pct",
                "threshold": threshold,
                "source_value": float(frac),
                "mean_abs_nonzero": float(mean),
                "std_abs_nonzero": float(std),
                "nonzero_count": float(nonzero_count),
                "total_count": float(total_count),
                "zero_frac": 1.0 - float(nonzero_count) / float(total_count) if total_count else float("nan"),
            }
        )
    out = pd.DataFrame(rows)
    out = out[np.isfinite(out["threshold"].to_numpy(dtype="float64")) & (out["threshold"] > float(zero_eps))].copy()
    out["_rounded_threshold"] = out["threshold"].map(lambda value: round(float(value), 10))
    out = out.drop_duplicates("_rounded_threshold", keep="first").drop(columns=["_rounded_threshold"])
    return out.sort_values("threshold").reset_index(drop=True)


def compute_or_load_covariance_threshold_sweep_focus(
    exp,
    cov_rows: pd.DataFrame,
    examples: list[Any],
    *,
    layer_name: str | None = None,
    candidates: pd.DataFrame | None = None,
    metric_cols: tuple[str, ...] = (
        "attack_energy_top5_importance_frac",
        "attack_energy_top5_importance_sum",
        "attack_energy_top10_0p1bin_linear_frac",
        "attack_energy_top10_0p1bin_linear_sum",
    ),
    sample_per_image: int = 20000,
    seed: int = 0,
    zero_eps: float = 0.0,
    force: bool = False,
) -> dict[str, Any]:
    from tqdm.auto import tqdm

    from .metrics import metric_quality_rows
    from .regression_metrics import regression_similarity_table

    layer_key = layer_name or exp.config.target_layer
    selected = list(examples)
    if candidates is None:
        candidates = covariance_threshold_candidate_table(
            cov_rows,
            sample_per_image=int(sample_per_image),
            seed=int(seed),
            zero_eps=float(zero_eps),
        )
    candidates = candidates.copy().reset_index(drop=True)
    candidates["threshold_id"] = np.arange(len(candidates), dtype=int)
    path = _covariance_threshold_sweep_focus_cache_path(
        exp,
        cov_rows,
        selected,
        candidates=candidates,
        layer_name=layer_key,
        metric_cols=metric_cols,
        zero_eps=float(zero_eps),
    )
    if path.exists() and not force:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        cached["loaded_from_cache"] = True
        cached["cache_path"] = str(path)
        return cached

    cov_by_path = {str(row["path"]): row for row in cov_rows.to_dict("records")}
    candidate_records = candidates.to_dict("records")
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for example in tqdm(selected, desc="cov threshold sweep focus", unit="img"):
        record = cov_by_path.get(str(example.path))
        if record is None:
            skipped.append({"path": str(example.path), "reason": "missing covariance row"})
            continue
        layer_maps = exp._load_layer_map_cache(example, layer_name=layer_key, include_clean_activation=False)
        if layer_maps is None:
            skipped.append({"path": str(example.path), "reason": "missing layer_maps cache"})
            continue
        try:
            delta = _as_chw_float32(layer_maps["delta_chw"]).reshape(-1)
            importance = _as_chw_float32(layer_maps["segmentig_chw"]).reshape(-1)
            with np.load(str(record["cache_path"]), allow_pickle=False) as data:
                cov = _as_chw_float32(data["cov_chw"]).reshape(-1)
        except Exception as exc:
            skipped.append({"path": str(example.path), "reason": repr(exc)})
            continue
        n = min(delta.size, importance.size, cov.size)
        d = delta[:n]
        a = importance[:n]
        cov_abs = np.abs(cov[:n])
        finite = np.isfinite(cov_abs)
        for candidate in candidate_records:
            threshold = float(candidate["threshold"])
            masks = {
                "low": finite & (cov_abs < threshold),
                "high": finite & (cov_abs >= threshold),
            }
            for group, mask in masks.items():
                selected_count = int(mask.sum())
                metrics = {col: float("nan") for col in metric_cols}
                if selected_count > 0:
                    metrics.update({key: value for key, value in importance_energy_focus_metrics(d[mask], a[mask]).items() if key in metric_cols})
                rows.append(
                    {
                        "path": str(example.path),
                        "success": bool(example.success),
                        "drop": float(example.drop),
                        "conf_clean": float(example.conf_clean),
                        "conf_patch": float(example.conf_patch),
                        "cov_method": str(record["method"]),
                        "threshold_id": int(candidate["threshold_id"]),
                        "threshold_kind": str(candidate["threshold_kind"]),
                        "threshold_label": str(candidate["threshold_label"]),
                        "threshold": threshold,
                        "cov_group": group,
                        "cov_selected_count": float(selected_count),
                        "cov_selected_frac": float(selected_count / max(1, n)),
                        **metrics,
                    }
                )
    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        raise RuntimeError(f"No threshold sweep rows; skipped={len(skipped)}")

    class_parts: list[pd.DataFrame] = []
    reg_parts: list[pd.DataFrame] = []
    group_cols = ["cov_method", "threshold_id", "threshold_kind", "threshold_label", "threshold", "cov_group"]
    for key, sub in rows_df.groupby(group_cols):
        labels = sub["success"].astype(bool).to_numpy()
        class_df = pd.DataFrame(metric_quality_rows(labels, {col: sub[col].to_numpy(dtype="float64", copy=False) for col in metric_cols}))
        if not class_df.empty:
            for col, value in zip(group_cols, key):
                class_df[col] = value
            class_df["roc_auc_effective"] = class_df["roc_auc"].map(lambda v: max(float(v), 1.0 - float(v)) if np.isfinite(v) else np.nan)
            class_parts.append(class_df)
        reg_df = regression_similarity_table(sub[["drop", *metric_cols]], target_col="drop", metric_cols=list(metric_cols))
        if not reg_df.empty:
            for col, value in zip(group_cols, key):
                reg_df[col] = value
            reg_parts.append(reg_df)
    classification = pd.concat(class_parts, ignore_index=True) if class_parts else pd.DataFrame()
    if not classification.empty:
        classification = classification.sort_values(["best_balanced_accuracy", "roc_auc_effective", "best_accuracy"], ascending=False).reset_index(drop=True)
    regression = pd.concat(reg_parts, ignore_index=True) if reg_parts else pd.DataFrame()
    if not regression.empty:
        regression = regression.sort_values(["main_score", "abs_spearman", "abs_pearson"], ascending=False).reset_index(drop=True)

    result = {
        "rows": rows_df,
        "classification": classification,
        "regression": regression,
        "candidates": candidates,
        "skipped": pd.DataFrame(skipped),
        "metric_cols": list(metric_cols),
        "loaded_from_cache": False,
        "cache_path": str(path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def importance_energy_focus_metrics(delta_flat, importance_flat) -> dict[str, float]:
    d = np.asarray(delta_flat, dtype="float64").reshape(-1)
    a = np.asarray(importance_flat, dtype="float64").reshape(-1)
    n = min(d.size, a.size)
    if n == 0:
        return {
            "attack_energy_top5_importance_frac": float("nan"),
            "attack_energy_top5_importance_sum": float("nan"),
            "attack_energy_top10_0p1bin_linear_frac": float("nan"),
            "attack_energy_top10_0p1bin_linear_sum": float("nan"),
            "attack_energy_total_abs_sum": float("nan"),
        }
    d_abs = np.abs(d[:n])
    a_abs = np.abs(a[:n])
    total = float(d_abs.sum() + 1e-12)
    order = np.argsort(-a_abs, kind="stable")

    top5_k = max(1, int(round(0.05 * n)))
    top5_sum = float(d_abs[order[:top5_k]].sum())

    n_bins = 1000
    top_bins = 100
    top10_k = max(1, int(round(0.10 * n)))
    ranked_delta = d_abs[order[:top10_k]]
    bin_ids = np.floor(np.arange(top10_k, dtype="float64") * top_bins / top10_k).astype(int)
    bin_ids = np.clip(bin_ids, 0, top_bins - 1)
    bin_weights = np.linspace(1.0, 1.0 / top_bins, top_bins, dtype="float64")
    weighted_top10_sum = float(np.sum(ranked_delta * bin_weights[bin_ids]))
    return {
        "attack_energy_top5_importance_frac": float(top5_sum / total),
        "attack_energy_top5_importance_sum": top5_sum,
        "attack_energy_top10_0p1bin_linear_frac": float(weighted_top10_sum / total),
        "attack_energy_top10_0p1bin_linear_sum": weighted_top10_sum,
        "attack_energy_total_abs_sum": float(d_abs.sum()),
    }


def plot_focus_raw_vs_fraction(
    rows: pd.DataFrame,
    *,
    metric_base: str,
    title: str | None = None,
    color_col: str | None = "success",
):
    import matplotlib.pyplot as plt

    frac_col = f"{metric_base}_frac"
    sum_col = f"{metric_base}_sum"
    if frac_col not in rows.columns or sum_col not in rows.columns:
        raise KeyError(f"Expected columns {frac_col!r} and {sum_col!r}")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    if color_col and color_col in rows.columns:
        if color_col == "success":
            colors = np.where(rows[color_col].astype(bool).to_numpy(), "#4C78A8", "#F58518")
            ax.scatter(rows[frac_col], rows[sum_col], c=colors, alpha=0.6, s=18)
            handles = [
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C78A8", label="success", markersize=7),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#F58518", label="fail", markersize=7),
            ]
            ax.legend(handles=handles)
        else:
            sc = ax.scatter(rows[frac_col], rows[sum_col], c=rows[color_col], cmap="viridis", alpha=0.65, s=18)
            fig.colorbar(sc, ax=ax, label=color_col)
    else:
        ax.scatter(rows[frac_col], rows[sum_col], alpha=0.6, s=18)
    ax.set_xlabel(frac_col)
    ax.set_ylabel(sum_col)
    ax.set_title(title or f"{sum_col} vs {frac_col}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def focus_quality_comparison_table(
    split_quality: pd.DataFrame,
    overall_quality: pd.DataFrame,
    *,
    metric_cols: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    metric_set = set(str(col) for col in metric_cols)
    if overall_quality is not None and not overall_quality.empty:
        overall = overall_quality[overall_quality["metric"].astype(str).isin(metric_set)].copy()
        if not overall.empty:
            overall["variant"] = "overall"
            overall["variant_type"] = "overall"
            overall["cov_method"] = "none"
            overall["sigma"] = np.nan
            overall["cov_group"] = "none"
            parts.append(overall)
    if split_quality is not None and not split_quality.empty:
        split = split_quality[split_quality["metric"].astype(str).isin(metric_set)].copy()
        if not split.empty:
            split["variant"] = (
                split["cov_method"].astype(str)
                + " | "
                + split["sigma"].map(lambda value: f"{float(value):g}σ")
                + " | "
                + split["cov_group"].astype(str)
            )
            split["variant_type"] = split["cov_group"].astype(str)
            parts.append(split)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    if "roc_auc_effective" not in out.columns and "roc_auc" in out.columns:
        out["roc_auc_effective"] = out["roc_auc"].map(lambda v: max(float(v), 1.0 - float(v)) if np.isfinite(v) else np.nan)
    preferred = [
        "metric",
        "variant",
        "variant_type",
        "best_accuracy",
        "roc_auc",
        "roc_auc_effective",
        "best_threshold",
        "best_direction",
        "cov_method",
        "sigma",
        "cov_group",
    ]
    cols = [col for col in preferred if col in out.columns] + [col for col in out.columns if col not in preferred]
    return out[cols].reset_index(drop=True)


def plot_focus_quality_comparison_bars(
    comparison: pd.DataFrame,
    *,
    score_col: str = "best_balanced_accuracy",
    title: str | None = None,
    metrics: list[str] | tuple[str, ...] | None = None,
):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if comparison.empty:
        raise ValueError("comparison is empty")
    score_col = str(score_col)
    if score_col not in comparison.columns:
        raise KeyError(f"{score_col!r} was not found in comparison")
    plot_df = comparison.copy()
    if metrics is not None:
        metric_set = set(str(metric) for metric in metrics)
        plot_df = plot_df[plot_df["metric"].astype(str).isin(metric_set)].copy()
    if plot_df.empty:
        raise ValueError("No rows remain after filtering metrics")
    type_order = {"overall": 0, "low": 1, "high": 2}
    plot_df["_type_order"] = plot_df["variant_type"].map(lambda value: type_order.get(str(value), 9))
    plot_df["_sigma_order"] = plot_df["sigma"].fillna(-1).astype(float) if "sigma" in plot_df.columns else -1.0
    plot_df = plot_df.sort_values(["metric", "_type_order", "_sigma_order", "variant"]).reset_index(drop=True)
    labels = plot_df["metric"].astype(str) + " | " + plot_df["variant"].astype(str)
    colors_by_type = {"overall": "#6B7280", "low": "#4C78A8", "high": "#F58518"}
    colors = [colors_by_type.get(str(value), "#9CA3AF") for value in plot_df["variant_type"]]

    fig_h = max(5.0, 0.28 * len(plot_df) + 1.4)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df[score_col].to_numpy(dtype="float64"), color=colors, alpha=0.88)
    ax.set_yticks(y, labels.tolist())
    ax.invert_yaxis()
    ax.set_xlabel(score_col)
    ax.set_title(title or f"Focused metrics: overall vs low/high covariance by {score_col}")
    ax.grid(axis="x", alpha=0.25)
    handles = [mpatches.Patch(color=color, label=label) for label, color in colors_by_type.items()]
    ax.legend(handles=handles, loc="lower right")
    for yi, value in zip(y, plot_df[score_col].to_numpy(dtype="float64")):
        if np.isfinite(value):
            ax.text(value, yi, f" {value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_threshold_sweep_top_bars(
    table: pd.DataFrame,
    *,
    score_col: str,
    top_n: int = 25,
    title: str | None = None,
):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("table is empty")
    if score_col not in table.columns:
        raise KeyError(f"{score_col!r} was not found in table")
    plot_df = table.sort_values(score_col, ascending=False).head(int(top_n)).copy()
    plot_df = plot_df.sort_values(score_col)
    labels = (
        plot_df["metric"].astype(str)
        + " | "
        + plot_df["threshold_label"].astype(str)
        + " | "
        + plot_df["cov_group"].astype(str)
    )
    colors_by_group = {"low": "#4C78A8", "high": "#F58518"}
    colors = [colors_by_group.get(str(group), "#9CA3AF") for group in plot_df["cov_group"]]
    fig_h = max(5.0, 0.32 * len(plot_df) + 1.4)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df[score_col].to_numpy(dtype="float64"), color=colors, alpha=0.88)
    ax.set_yticks(y, labels.tolist())
    ax.set_xlabel(score_col)
    ax.set_title(title or f"Threshold sweep top metrics by {score_col}")
    ax.grid(axis="x", alpha=0.25)
    handles = [mpatches.Patch(color=color, label=label) for label, color in colors_by_group.items()]
    ax.legend(handles=handles, loc="lower right")
    for yi, value in zip(y, plot_df[score_col].to_numpy(dtype="float64")):
        if np.isfinite(value):
            ax.text(value, yi, f" {value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_covariance_split_quality_overview(
    quality: pd.DataFrame,
    *,
    score_col: str = "best_balanced_accuracy",
    top_n: int = 25,
    title: str | None = None,
):
    import matplotlib.pyplot as plt

    if quality.empty:
        raise ValueError("quality is empty")
    score_col = str(score_col)
    if score_col not in quality.columns:
        raise KeyError(f"{score_col!r} was not found in quality")
    plot_df = quality.sort_values(score_col, ascending=False).head(int(top_n)).copy()
    plot_df["label"] = (
        plot_df["metric"].astype(str)
        + " | "
        + plot_df["cov_method"].astype(str)
        + " | "
        + plot_df["sigma"].map(lambda v: f"{float(v):g}σ")
        + " | "
        + plot_df["cov_group"].astype(str)
    )
    fig_h = max(5.0, 0.35 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df[score_col].to_numpy(dtype="float64"), color="#4C78A8", alpha=0.9)
    ax.set_yticks(y, plot_df["label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel(score_col)
    ax.set_title(title or f"Top covariance split metrics by {score_col}")
    ax.grid(axis="x", alpha=0.25)
    for yi, value in zip(y, plot_df[score_col].to_numpy(dtype="float64")):
        if np.isfinite(value):
            ax.text(value, yi, f" {value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_best_covariance_split_metric_details(
    rows: pd.DataFrame,
    quality: pd.DataFrame,
    *,
    top_n: int = 6,
    score_col: str = "best_balanced_accuracy",
):
    from .plots import plot_metric_distribution_and_roc

    if rows.empty or quality.empty:
        return []
    figs = []
    top = quality.sort_values(str(score_col), ascending=False).head(int(top_n))
    for item in top.itertuples(index=False):
        sub = rows[
            (rows["cov_method"].astype(str) == str(item.cov_method))
            & (rows["sigma"].astype(float) == float(item.sigma))
            & (rows["cov_group"].astype(str) == str(item.cov_group))
        ]
        if sub.empty or str(item.metric) not in sub.columns:
            continue
        metric_name = f"{item.metric} | {item.cov_method} | {float(item.sigma):g}σ | {item.cov_group}"
        raw_auc = float(item.roc_auc)
        if np.isfinite(raw_auc) and int(item.best_direction) == -1:
            raw_auc = 1.0 - raw_auc
        figs.append(
            plot_metric_distribution_and_roc(
                sub["success"].astype(bool).to_numpy(),
                sub[str(item.metric)].to_numpy(dtype="float64", copy=False),
                metric_name=metric_name,
                auc=raw_auc,
                best_accuracy=float(item.best_accuracy),
                direction=int(item.best_direction),
            )
        )
    return figs


def _load_covariance_split_metric_chunks(
    exp,
    cov_rows: pd.DataFrame,
    examples: list[Any],
    *,
    requested_sigmas: tuple[float, ...],
    layer_name: str,
    top_percent: float,
    rank_bins: int,
    include_spatial: bool,
    zero_eps: float,
) -> list[dict[str, Any]]:
    selected_paths = {str(example.path) for example in examples}
    cov_cache_paths = set(cov_rows.get("cache_path", pd.Series(dtype=str)).astype(str).tolist())
    required_cols = {
        "attack_energy_top5_importance_frac",
        "attack_energy_top5_importance_sum",
        "attack_energy_top10_0p1bin_linear_frac",
        "attack_energy_top10_0p1bin_linear_sum",
    }
    candidates: list[Path] = []
    for sigma in requested_sigmas:
        candidates.append(
            _covariance_split_metrics_cache_path(
                exp,
                cov_rows,
                examples,
                sigmas=(float(sigma),),
                layer_name=layer_name,
                top_percent=float(top_percent),
                rank_bins=int(rank_bins),
                include_spatial=bool(include_spatial),
                zero_eps=float(zero_eps),
            )
        )
    cache_dir = exp.derived_cache_dir / "covariance_split_metrics"
    if cache_dir.exists():
        candidates.extend(sorted(cache_dir.glob("covariance_split_metrics_*.pkl")))

    chunks: list[dict[str, Any]] = []
    covered: set[float] = set()
    seen_paths: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen_paths or not candidate.exists():
            continue
        seen_paths.add(candidate_key)
        try:
            with candidate.open("rb") as fh:
                cached = pickle.load(fh)
        except Exception:
            continue
        rows = cached.get("rows")
        if not isinstance(rows, pd.DataFrame) or rows.empty:
            continue
        if not required_cols.issubset(set(rows.columns)):
            continue
        row_paths = set(rows.get("path", pd.Series(dtype=str)).astype(str).unique().tolist())
        if row_paths != selected_paths:
            continue
        row_cov_paths = set(rows.get("cov_cache_path", pd.Series(dtype=str)).astype(str).unique().tolist())
        if row_cov_paths and cov_cache_paths and not row_cov_paths.issubset(cov_cache_paths):
            continue
        available = [float(sigma) for sigma in rows["sigma"].dropna().unique().tolist() if float(sigma) in requested_sigmas]
        available = [sigma for sigma in available if sigma not in covered]
        if not available:
            continue
        rows_sub = rows[rows["sigma"].astype(float).isin(available)].copy()
        quality = cached.get("quality")
        if isinstance(quality, pd.DataFrame) and not quality.empty and "sigma" in quality.columns:
            quality_sub = quality[quality["sigma"].astype(float).isin(available)].copy()
        else:
            quality_sub = pd.DataFrame()
        thresholds = cached.get("thresholds")
        if isinstance(thresholds, pd.DataFrame) and not thresholds.empty and "sigma" in thresholds.columns:
            thresholds_sub = thresholds[thresholds["sigma"].astype(float).isin(available)].copy()
        else:
            thresholds_sub = pd.DataFrame()
        skipped = cached.get("skipped")
        skipped_sub = skipped.copy() if isinstance(skipped, pd.DataFrame) else pd.DataFrame()
        metric_cols = cached.get("metric_cols")
        if not metric_cols:
            meta = {
                "path",
                "success",
                "drop",
                "conf_clean",
                "conf_patch",
                "cov_method",
                "sigma",
                "cov_group",
                "cov_threshold",
                "cov_cache_path",
                "layer_maps_cache_path",
            }
            metric_cols = [col for col in rows_sub.columns if col not in meta and pd.api.types.is_numeric_dtype(rows_sub[col])]
        chunks.append(
            {
                "rows": rows_sub,
                "quality": quality_sub,
                "thresholds": thresholds_sub,
                "skipped": skipped_sub,
                "metric_cols": list(metric_cols),
                "loaded_from_cache": True,
                "cache_path": str(candidate),
            }
        )
        covered.update(available)
        if all(float(sigma) in covered for sigma in requested_sigmas):
            break
    return chunks


def _combine_covariance_split_metric_results(chunks: list[dict[str, Any]], *, cache_path: Path) -> dict[str, Any]:
    rows_parts = [chunk["rows"] for chunk in chunks if isinstance(chunk.get("rows"), pd.DataFrame) and not chunk["rows"].empty]
    quality_parts = [
        chunk["quality"] for chunk in chunks if isinstance(chunk.get("quality"), pd.DataFrame) and not chunk["quality"].empty
    ]
    threshold_parts = [
        chunk["thresholds"]
        for chunk in chunks
        if isinstance(chunk.get("thresholds"), pd.DataFrame) and not chunk["thresholds"].empty
    ]
    skipped_parts = [
        chunk["skipped"] for chunk in chunks if isinstance(chunk.get("skipped"), pd.DataFrame) and not chunk["skipped"].empty
    ]
    rows = pd.concat(rows_parts, ignore_index=True) if rows_parts else pd.DataFrame()
    if not rows.empty:
        dedupe_cols = [col for col in ["path", "cov_method", "sigma", "cov_group"] if col in rows.columns]
        rows = rows.drop_duplicates(dedupe_cols, keep="last").sort_values(dedupe_cols).reset_index(drop=True)
    quality = pd.concat(quality_parts, ignore_index=True) if quality_parts else pd.DataFrame()
    if not quality.empty:
        dedupe_cols = [col for col in ["metric", "cov_method", "sigma", "cov_group"] if col in quality.columns]
        quality = quality.drop_duplicates(dedupe_cols, keep="last")
        sort_cols = [col for col in ["best_accuracy", "roc_auc_effective"] if col in quality.columns]
        if sort_cols:
            quality = quality.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    thresholds = pd.concat(threshold_parts, ignore_index=True) if threshold_parts else pd.DataFrame()
    if not thresholds.empty:
        dedupe_cols = [col for col in ["method", "sigma"] if col in thresholds.columns]
        thresholds = thresholds.drop_duplicates(dedupe_cols, keep="last").sort_values(dedupe_cols).reset_index(drop=True)
    skipped = pd.concat(skipped_parts, ignore_index=True) if skipped_parts else pd.DataFrame()
    metric_cols = sorted({col for chunk in chunks for col in chunk.get("metric_cols", [])})
    return {
        "rows": rows,
        "quality": quality,
        "thresholds": thresholds,
        "skipped": skipped,
        "metric_cols": metric_cols,
        "loaded_from_cache": False,
        "cache_path": str(cache_path),
    }


def _subset_covariance_split_result(result: dict[str, Any], *, sigma: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("rows", "quality", "thresholds"):
        value = result.get(key)
        if isinstance(value, pd.DataFrame) and not value.empty and "sigma" in value.columns:
            out[key] = value[value["sigma"].astype(float) == float(sigma)].copy()
        elif isinstance(value, pd.DataFrame):
            out[key] = value.copy()
        else:
            out[key] = pd.DataFrame()
    skipped = result.get("skipped")
    out["skipped"] = skipped.copy() if isinstance(skipped, pd.DataFrame) else pd.DataFrame()
    out["metric_cols"] = list(result.get("metric_cols", []))
    out["loaded_from_cache"] = False
    out["cache_path"] = str(result.get("cache_path", ""))
    return out


def _midpoint_alphas(n_steps: int, *, segment_start: float, segment_end: float) -> np.ndarray:
    if int(n_steps) <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if not 0.0 <= float(segment_start) < float(segment_end) <= 1.0:
        raise ValueError(f"segment must satisfy 0 <= start < end <= 1, got [{segment_start}, {segment_end}]")
    width = float(segment_end) - float(segment_start)
    return float(segment_start) + width * ((np.arange(int(n_steps), dtype=np.float64) + 0.5) / float(n_steps))


def _segment_index_mask(alphas, start: float, end: float) -> np.ndarray:
    arr = np.asarray(alphas, dtype=np.float64)
    if float(end) < 1.0:
        return np.nonzero((arr >= float(start)) & (arr < float(end)))[0]
    return np.nonzero((arr >= float(start)) & (arr <= float(end)))[0]


def _binned_means(values, *, n_bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    bins = np.array_split(arr, int(n_bins))
    return np.asarray([float(np.nanmean(part)) if len(part) else float("nan") for part in bins], dtype="float64")


def _as_chw_float32(values) -> np.ndarray:
    arr = np.asarray(values, dtype="float32")
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={arr.shape}")
    return arr


def _prefix(prefix: str, values: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        try:
            out[f"{prefix}_{key}"] = float(value)
        except Exception:
            out[f"{prefix}_{key}"] = float("nan")
    return out


def _covariance_split_metrics_cache_path(
    exp,
    cov_rows: pd.DataFrame,
    examples: list[Any],
    *,
    sigmas: tuple[float, ...],
    layer_name: str,
    top_percent: float,
    rank_bins: int,
    include_spatial: bool,
    zero_eps: float,
) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [str(example.path) for example in examples],
        "cov_cache_paths": sorted(str(path) for path in cov_rows.get("cache_path", pd.Series(dtype=str)).astype(str).tolist()),
        "target_layer": str(layer_name),
        "target_mode": exp.config.target_mode,
        "detect_layer": exp.config.detect_layer,
        "sigmas": [float(v) for v in sigmas],
        "top_percent": float(top_percent),
        "rank_bins": int(rank_bins),
        "include_spatial": bool(include_spatial),
        "zero_eps": float(zero_eps),
        "version": 2,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
    out = exp.derived_cache_dir / "covariance_split_metrics"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"covariance_split_metrics_{key}.pkl"


def _importance_energy_focus_cache_path(exp, examples: list[Any], *, layer_name: str) -> Path:
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [str(example.path) for example in examples],
        "target_layer": str(layer_name),
        "target_mode": exp.config.target_mode,
        "detect_layer": exp.config.detect_layer,
        "version": 1,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
    out = exp.derived_cache_dir / "importance_energy_focus_metrics"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"importance_energy_focus_metrics_{key}.pkl"


def _covariance_threshold_sweep_focus_cache_path(
    exp,
    cov_rows: pd.DataFrame,
    examples: list[Any],
    *,
    candidates: pd.DataFrame,
    layer_name: str,
    metric_cols: tuple[str, ...],
    zero_eps: float,
) -> Path:
    candidate_payload = candidates[
        ["threshold_kind", "threshold_label", "threshold"]
    ].to_dict("records") if not candidates.empty else []
    payload = {
        "attack_cache_key": exp.get_cache().cache_key,
        "paths": [str(example.path) for example in examples],
        "cov_cache_paths": sorted(str(path) for path in cov_rows.get("cache_path", pd.Series(dtype=str)).astype(str).tolist()),
        "target_layer": str(layer_name),
        "target_mode": exp.config.target_mode,
        "detect_layer": exp.config.detect_layer,
        "metric_cols": list(metric_cols),
        "candidates": candidate_payload,
        "zero_eps": float(zero_eps),
        "version": 1,
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
    out = exp.derived_cache_dir / "covariance_threshold_sweep_focus"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"covariance_threshold_sweep_focus_{key}.pkl"


def _safe_name(value: str) -> str:
    return str(value).replace(".", "_").replace("/", "_").replace(" ", "_")
