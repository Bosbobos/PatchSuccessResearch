from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "patch_success_matplotlib"))

DEFAULT_STEP_FRACS: tuple[float, ...] = (
    0.0,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.50,
    1.0,
)


@dataclass(slots=True)
class ExampleActivations:
    clean_chw: np.ndarray
    patched_chw: np.ndarray
    delta_chw: np.ndarray
    importance_chw: np.ndarray
    activation_shape: tuple[int, int, int]
    cache_path: str


def stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def select_success_examples(exp, *, max_examples: int = 20):
    cache = exp.get_cache()
    out = []
    for example in cache.successes:
        if example.clean_detection is None:
            continue
        if float(example.conf_patch) >= float(example.conf_clean):
            continue
        out.append(example)
        if len(out) >= int(max_examples):
            break
    return out


def device_available(device: str | None) -> bool:
    if device is None:
        return True
    try:
        import torch

        if device == "mps":
            return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        if device == "cuda":
            return bool(torch.cuda.is_available())
    except Exception:
        return False
    return True


def default_device(prefer: str = "mps", *, require: bool = False) -> str | None:
    if prefer and device_available(prefer):
        return prefer
    if require and prefer:
        raise RuntimeError(
            f"Requested device {prefer!r}, but it is not available in this Python environment."
        )
    return None


def resolve_attack_config():
    from new_experiments.patch_success_analysis.data import AttackConfig, load_attack_cache

    device = default_device("mps")
    candidates = [
        AttackConfig(device=device),
        AttackConfig(output_dir="new_experiments/new_experiments/outputs/patch_success_analysis", device=device),
    ]
    for config in candidates:
        if load_attack_cache(config) is not None:
            return config
    return candidates[0]


def _normalize_legacy_path(value: Any, repo_root: Path) -> Any:
    if isinstance(value, (list, tuple)):
        return [_normalize_legacy_path(item, repo_root) for item in value]
    if not isinstance(value, str):
        return value
    raw = value
    if raw.startswith("../"):
        raw = raw[3:]
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    candidate = repo_root / path
    return str(candidate if candidate.exists() else path)


def _attack_config_from_cache(cache, *, cache_path: Path, repo_root: Path):
    from new_experiments.patch_success_analysis.data import AttackConfig

    cfg = dict(cache.config)
    output_dir = cache_path.parent.parent
    cfg["output_dir"] = str(output_dir)
    for key in ("dataset_path", "patch_path", "model_path"):
        if key in cfg:
            cfg[key] = _normalize_legacy_path(cfg[key], repo_root)
    device = default_device("mps")
    if device is not None:
        cfg["device"] = device
    allowed = set(AttackConfig.__dataclass_fields__)
    cfg = {key: value for key, value in cfg.items() if key in allowed}
    return AttackConfig(**cfg)


def load_existing_experiment(
    *,
    repo_root: str | Path | None = None,
    prefer_dataset: str = "COCO_people",
    prefer_device: str = "mps",
    require_device: bool = False,
):
    import sys

    from new_experiments.patch_success_analysis.experiments import ExperimentConfig, PatchSuccessExperiment

    repo = Path(repo_root or Path.cwd()).resolve()
    for package_root in (repo, repo / "new_experiments"):
        package_root_str = str(package_root)
        if package_root_str not in sys.path:
            sys.path.insert(0, package_root_str)
    roots = [
        repo / "new_experiments" / "outputs" / "patch_success_analysis" / "cache",
        repo / "new_experiments" / "new_experiments" / "outputs" / "patch_success_analysis" / "cache",
    ]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(path for path in root.glob("attack_pool_*.pkl") if not path.name.endswith("_summary.pkl"))
    if not paths:
        cfg = resolve_attack_config()
        device = default_device(prefer_device, require=require_device)
        if device is not None:
            cfg.device = device
        exp = PatchSuccessExperiment(ExperimentConfig(attack=cfg))
        return exp, None

    def _rank(path: Path) -> tuple[int, int, int]:
        try:
            with path.open("rb") as fh:
                cache = pickle.load(fh)
            if not hasattr(cache, "examples") or not hasattr(cache, "config"):
                return (0, 0, 0)
            config_text = json.dumps(cache.config, ensure_ascii=True)
            preferred = int(prefer_dataset in config_text)
            return (preferred, len(cache.examples), int(path.stat().st_mtime_ns))
        except Exception:
            return (0, 0, 0)

    best_path = max(paths, key=_rank)
    with best_path.open("rb") as fh:
        cache = pickle.load(fh)
    if not hasattr(cache, "examples") or not hasattr(cache, "config"):
        raise TypeError(f"Unexpected cache object in {best_path}: {type(cache)}")
    cfg = _attack_config_from_cache(cache, cache_path=best_path, repo_root=repo)
    device = default_device(prefer_device, require=require_device)
    if device is not None:
        cfg.device = device
    for example in getattr(cache, "examples", []):
        if hasattr(example, "path"):
            example.path = _normalize_legacy_path(example.path, repo)
    exp = PatchSuccessExperiment(ExperimentConfig(attack=cfg))
    exp.cache = cache
    return exp, best_path


def ensure_example_activations(exp, example, *, layer_name: str, force: bool = False) -> ExampleActivations:
    from new_experiments.patch_success_analysis.yolo import get_module_by_name

    yolo, model = exp.load_model()
    layer = get_module_by_name(model, layer_name)
    ctx = exp._context_for_example(example, image_variant="clean")
    maps = exp._compute_or_load_segmentig_layer_maps(
        example,
        ctx,
        model=model,
        layer=layer,
        layer_name=layer_name,
        force=force,
        include_clean_activation=True,
    )
    clean = np.asarray(maps["clean_activation_chw"], dtype=np.float32)
    delta = np.asarray(maps["delta_chw"], dtype=np.float32)
    importance = np.asarray(maps["segmentig_chw"], dtype=np.float32)
    patched = clean + delta
    if clean.shape != patched.shape or clean.shape != importance.shape:
        raise ValueError(
            f"Activation shape mismatch: clean={clean.shape}, patched={patched.shape}, importance={importance.shape}"
        )
    if clean.ndim != 3:
        raise ValueError(f"Expected CHW activations, got shape={clean.shape}")
    return ExampleActivations(
        clean_chw=clean,
        patched_chw=patched,
        delta_chw=delta,
        importance_chw=importance,
        activation_shape=tuple(int(v) for v in clean.shape),
        cache_path=str(maps.get("cache_path", "")),
    )


def _bbox_from_clean_detection(example) -> tuple[float, float, float, float] | None:
    detection = getattr(example, "clean_detection", None)
    if not detection:
        return None
    bbox = detection.get("bbox_xyxy_orig")
    if bbox is None or len(bbox) != 4:
        return None
    return tuple(float(v) for v in bbox)


def spatial_masks(example, *, activation_shape: tuple[int, int, int], imgsz: int) -> dict[str, np.ndarray]:
    from new_experiments.patch_success_analysis.activations import patch_mask_on_feature_grid

    c, h, w = [int(v) for v in activation_shape]
    patch_hw = patch_mask_on_feature_grid(example.patch_bbox_lb, grid_hw=(h, w), imgsz=int(imgsz))
    object_hw = patch_mask_on_feature_grid(_bbox_from_clean_detection(example), grid_hw=(h, w), imgsz=int(imgsz))
    masks_hw = {
        "spatial_patch_region": patch_hw,
        "spatial_object_region": object_hw,
        "spatial_outside_patch": ~patch_hw,
        "spatial_outside_object": ~object_hw,
    }
    return {name: np.broadcast_to(mask[None, :, :], (c, h, w)).copy() for name, mask in masks_hw.items()}


def indices_for_fraction(order: np.ndarray, frac: float, n_total: int) -> np.ndarray:
    frac = float(frac)
    if frac <= 0.0:
        return np.asarray([], dtype=np.int64)
    k = max(1, int(math.ceil(float(n_total) * min(frac, 1.0))))
    return np.asarray(order[: min(k, int(n_total))], dtype=np.int64)


def guided_orders(
    activations: ExampleActivations,
    *,
    random_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    n_delta_bins: int = 10,
) -> dict[str, np.ndarray]:
    n = int(np.prod(activations.activation_shape))
    importance = np.abs(activations.importance_chw).reshape(-1)
    delta = np.abs(activations.delta_chw).reshape(-1)
    orders: dict[str, np.ndarray] = {
        "importance_desc": np.argsort(-importance, kind="stable"),
        "importance_asc": np.argsort(importance, kind="stable"),
        "delta_desc": np.argsort(-delta, kind="stable"),
        "delta_asc": np.argsort(delta, kind="stable"),
    }
    for seed in random_seeds:
        rng = np.random.default_rng(int(seed))
        orders[f"random_seed{seed}"] = rng.permutation(n)
        orders[f"random_delta_matched_seed{seed}"] = delta_matched_random_order(delta, seed=int(seed), n_bins=n_delta_bins)
    return orders


def delta_matched_random_order(delta_abs_flat: np.ndarray, *, seed: int, n_bins: int = 10) -> np.ndarray:
    delta = np.asarray(delta_abs_flat, dtype=np.float64).reshape(-1)
    n = int(delta.size)
    if n == 0:
        return np.asarray([], dtype=np.int64)
    desc = np.argsort(-delta, kind="stable")
    bins = np.array_split(desc, max(1, int(n_bins)))
    rng = np.random.default_rng(int(seed))
    shuffled_bins = []
    for bucket in bins:
        bucket = np.asarray(bucket, dtype=np.int64).copy()
        rng.shuffle(bucket)
        shuffled_bins.append(bucket)
    return np.concatenate(shuffled_bins) if shuffled_bins else np.arange(n, dtype=np.int64)


def flat_indices_to_mask(indices: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(int(np.prod(shape)), dtype=bool)
    if len(indices):
        mask[np.asarray(indices, dtype=np.int64)] = True
    return mask.reshape(shape)


def counterdelta_fractions(delta_chw: np.ndarray, mask_chw: np.ndarray, *, eps: float = 1e-12) -> dict[str, float]:
    delta = np.asarray(delta_chw, dtype=np.float64)
    mask = np.asarray(mask_chw, dtype=bool)
    selected = delta[mask]
    total_l1 = float(np.sum(np.abs(delta)))
    total_l2 = float(np.sqrt(np.sum(delta * delta)))
    selected_l1 = float(np.sum(np.abs(selected))) if selected.size else 0.0
    selected_l2 = float(np.sqrt(np.sum(selected * selected))) if selected.size else 0.0
    return {
        "counterdelta_l1": selected_l1,
        "counterdelta_l2": selected_l2,
        "counterdelta_l1_frac": selected_l1 / max(float(eps), total_l1),
        "counterdelta_l2_frac": selected_l2 / max(float(eps), total_l2),
    }


class ActivationRepairHook:
    def __init__(self, model, *, layer_name: str, clean_chw: np.ndarray, mask_chw: np.ndarray):
        import torch

        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise KeyError(f"Layer {layer_name!r} was not found.")
        self.clean = torch.as_tensor(np.asarray(clean_chw, dtype=np.float32))
        self.mask = torch.as_tensor(np.asarray(mask_chw, dtype=bool))
        self.handle = modules[layer_name].register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        import torch

        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            return output
        clean = self.clean.to(device=output.device, dtype=output.dtype)
        mask = self.mask.to(device=output.device)
        if tuple(clean.shape) != tuple(output.shape[1:]):
            raise RuntimeError(f"Repair shape mismatch: clean={tuple(clean.shape)}, output={tuple(output.shape[1:])}")
        out = output.clone()
        out[:, mask] = clean[mask].to(dtype=out.dtype)
        return out

    def remove(self) -> None:
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.remove()


def _inputs_for_variant(exp, example, *, image_variant: str):
    clean_lb, patched_lb, _patch_bbox = exp._images_for_example(example)
    image = patched_lb if image_variant == "patched" else clean_lb
    return exp._preprocess(image)["im"]


def fixed_target_class_logit(exp, example, *, image_variant: str, layer_name: str | None = None, repair: tuple[np.ndarray, np.ndarray] | None = None) -> float:
    from segmentig_detector.targets import detector_target_components

    _yolo, model = exp.load_model()
    clean_ctx = exp._context_for_example(example, image_variant="clean")
    inputs = clean_ctx["inputs"] if image_variant == "clean" else _inputs_for_variant(exp, example, image_variant=image_variant)
    hook = None
    try:
        if repair is not None:
            clean_chw, mask_chw = repair
            if layer_name is None:
                raise ValueError("layer_name is required when repair is provided")
            hook = ActivationRepairHook(model, layer_name=layer_name, clean_chw=clean_chw, mask_chw=mask_chw)
        components = detector_target_components(
            model,
            inputs,
            clean_ctx["fixed_target"],
            imgsz=int(exp.config.attack.imgsz),
            detect_name=exp.config.detect_layer,
        )
        return float(components["class_logit"])
    finally:
        if hook is not None:
            hook.remove()


def final_person_confidence_with_repair(
    exp,
    example,
    *,
    layer_name: str,
    clean_chw: np.ndarray,
    mask_chw: np.ndarray,
) -> float:
    from new_experiments.patch_success_analysis.yolo import confidence_from_result

    yolo, model = exp.load_model()
    _clean_lb, patched_lb, _patch_bbox = exp._images_for_example(example)
    with ActivationRepairHook(model, layer_name=layer_name, clean_chw=clean_chw, mask_chw=mask_chw):
        result = yolo.predict(
            source=np.asarray(patched_lb.convert("RGB"))[..., ::-1].copy(),
            imgsz=int(exp.config.attack.imgsz),
            conf=float(exp.config.attack.conf),
            device=exp.config.attack.device,
            verbose=False,
        )[0]
    return float(confidence_from_result(result, target_class_id=example.target_class_id))


def _example_id(example) -> str:
    return stable_hash({"path": example.path, "drop": float(example.drop), "success": bool(example.success)})


def run_example_repair(
    exp,
    example,
    *,
    layer_name: str = "model.22",
    step_fracs: tuple[float, ...] = DEFAULT_STEP_FRACS,
    random_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    evaluate_final_detection: bool = False,
    final_conf_threshold: float | None = None,
) -> pd.DataFrame:
    activations = ensure_example_activations(exp, example, layer_name=layer_name)
    shape = activations.activation_shape
    n_total = int(np.prod(shape))
    score_clean = fixed_target_class_logit(exp, example, image_variant="clean")
    score_patched = fixed_target_class_logit(exp, example, image_variant="patched")
    denom = max(1e-12, float(score_clean) - float(score_patched))
    final_conf_threshold = float(final_conf_threshold if final_conf_threshold is not None else exp.config.attack.success_thresh)

    row_base = {
        "example_id": _example_id(example),
        "path": example.path,
        "success": bool(example.success),
        "conf_clean": float(example.conf_clean),
        "conf_patch": float(example.conf_patch),
        "drop": float(example.drop),
        "layer_name": layer_name,
        "score_clean": float(score_clean),
        "score_patched": float(score_patched),
        "score_recovery_denominator": float(denom),
        "activation_c": int(shape[0]),
        "activation_h": int(shape[1]),
        "activation_w": int(shape[2]),
        "layer_maps_cache_path": activations.cache_path,
    }

    methods: list[tuple[str, str, Any]] = []
    for name, mask in spatial_masks(example, activation_shape=shape, imgsz=int(exp.config.attack.imgsz)).items():
        methods.append((name, "spatial", np.flatnonzero(mask.reshape(-1))))
    for name, order in guided_orders(activations, random_seeds=random_seeds).items():
        methods.append((name, "guided", order))

    rows = []
    full_mask = np.ones(shape, dtype=bool)
    zero_mask = np.zeros(shape, dtype=bool)
    for method_name, method_family, order_or_indices in methods:
        for frac in step_fracs:
            if method_family == "spatial":
                spatial_indices = np.asarray(order_or_indices, dtype=np.int64)
                k = int(math.ceil(len(spatial_indices) * min(max(float(frac), 0.0), 1.0)))
                selected = spatial_indices[:k]
                mask = flat_indices_to_mask(selected, shape)
            else:
                selected = indices_for_fraction(np.asarray(order_or_indices, dtype=np.int64), float(frac), n_total)
                mask = flat_indices_to_mask(selected, shape)
            if float(frac) >= 1.0 and method_family == "guided":
                mask = full_mask
            if float(frac) <= 0.0:
                mask = zero_mask

            score = fixed_target_class_logit(
                exp,
                example,
                image_variant="patched",
                layer_name=layer_name,
                repair=(activations.clean_chw, mask),
            )
            recovery = (float(score) - float(score_patched)) / denom
            final_conf = np.nan
            restored_detection: bool | float = np.nan
            if evaluate_final_detection:
                final_conf = final_person_confidence_with_repair(
                    exp,
                    example,
                    layer_name=layer_name,
                    clean_chw=activations.clean_chw,
                    mask_chw=mask,
                )
                restored_detection = bool(final_conf >= final_conf_threshold)
            budgets = counterdelta_fractions(activations.delta_chw, mask)
            rows.append(
                {
                    **row_base,
                    "method": method_name,
                    "method_family": method_family,
                    "step_frac": float(frac),
                    "n_elements": int(mask.sum()),
                    "n_elements_frac": float(mask.sum()) / float(n_total),
                    "score": float(score),
                    "recovery": float(recovery),
                    "final_conf": float(final_conf) if np.isfinite(final_conf) else np.nan,
                    "restored_detection": restored_detection,
                    **budgets,
                }
            )
    return pd.DataFrame(rows)


def run_repair_experiment(
    exp,
    *,
    layer_name: str = "model.22",
    max_examples: int = 20,
    step_fracs: tuple[float, ...] = DEFAULT_STEP_FRACS,
    random_seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    evaluate_final_detection: bool = False,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    examples = select_success_examples(exp, max_examples=max_examples)
    frames: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    iterator = enumerate(examples)
    progress_bar = False
    if show_progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=len(examples), desc=f"causal repair {layer_name}", unit="example")
            progress_bar = True
        except Exception:
            print(f"causal repair {layer_name}: {len(examples)} examples")
    for idx, example in iterator:
        if show_progress and not progress_bar:
            print(f"[{idx + 1}/{len(examples)}] {Path(str(example.path)).name}")
        try:
            frames.append(
                run_example_repair(
                    exp,
                    example,
                    layer_name=layer_name,
                    step_fracs=step_fracs,
                    random_seeds=random_seeds,
                    evaluate_final_detection=evaluate_final_detection,
                )
            )
            if show_progress and not progress_bar:
                print(f"[{idx + 1}/{len(examples)}] done")
        except Exception as exc:  # noqa: BLE001 - notebook should record skipped examples and continue.
            if show_progress and not progress_bar:
                print(f"[{idx + 1}/{len(examples)}] skipped: {type(exc).__name__}: {exc}")
            skipped.append(
                {
                    "path": example.path,
                    "success": bool(example.success),
                    "drop": float(example.drop),
                    "index": int(idx),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = summarize_repair_curves(rows)
    return {"rows": rows, "summary": summary, "skipped": pd.DataFrame(skipped)}


def summarize_repair_curves(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    out_rows = []
    group_cols = ["method", "method_family"]

    def _nanmean_or_nan(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else np.nan

    def _nanmedian_or_nan(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.median(arr)) if arr.size else np.nan

    for keys, group in rows.groupby(group_cols, dropna=False):
        method, family = keys
        per_example_auc = []
        budget_50 = []
        for _example_id, eg in group.groupby("example_id"):
            curve = (
                eg.sort_values("counterdelta_l2_frac")
                .drop_duplicates("counterdelta_l2_frac", keep="last")
                [["counterdelta_l2_frac", "recovery"]]
                .dropna()
            )
            if curve.empty:
                continue
            x = curve["counterdelta_l2_frac"].to_numpy(dtype=float)
            y = curve["recovery"].to_numpy(dtype=float)
            if x[0] > 0.0:
                x = np.r_[0.0, x]
                y = np.r_[0.0, y]
            if x[-1] < 1.0:
                x = np.r_[x, 1.0]
                y = np.r_[y, y[-1]]
            per_example_auc.append(float(np.trapz(y, x)))
            hit = curve[curve["recovery"] >= 0.5]
            budget_50.append(float(hit["counterdelta_l2_frac"].iloc[0]) if not hit.empty else np.nan)
        out_rows.append(
            {
                "method": method,
                "method_family": family,
                "n_examples": int(group["example_id"].nunique()),
                "mean_recovery_auc_l2": _nanmean_or_nan(per_example_auc),
                "std_recovery_auc_l2": float(np.nanstd(per_example_auc)) if per_example_auc else np.nan,
                "mean_budget_to_recovery_0p5": _nanmean_or_nan(budget_50),
                "median_budget_to_recovery_0p5": _nanmedian_or_nan(budget_50),
                "final_mean_recovery": float(group.sort_values("counterdelta_l2_frac").groupby("example_id").tail(1)["recovery"].mean()),
            }
        )
    return pd.DataFrame(out_rows).sort_values("mean_recovery_auc_l2", ascending=False, na_position="last").reset_index(drop=True)


def add_marginal_efficiency(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    out_parts = []
    for _, group in rows.groupby(["example_id", "method"], dropna=False):
        g = group.sort_values("counterdelta_l2_frac").copy()
        dx = g["counterdelta_l2_frac"].diff()
        dy = g["recovery"].diff()
        g["marginal_recovery_per_l2"] = dy / dx.clip(lower=1e-12)
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True)


def save_experiment_outputs(result: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(output / f"{name}.csv", index=False)


def plot_recovery_curves(rows: pd.DataFrame, *, x_col: str, title: str, output_path: str | Path | None = None):
    import matplotlib.pyplot as plt

    if rows.empty:
        raise ValueError("rows is empty")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    plot_df = rows.dropna(subset=[x_col, "recovery"]).copy()
    for method, group in plot_df.groupby("method"):
        curve = group.groupby(x_col, as_index=False)["recovery"].mean().sort_values(x_col)
        ax.plot(curve[x_col], curve["recovery"], marker="o", linewidth=1.6, markersize=3, label=str(method))
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlabel(x_col)
    ax.set_ylabel("normalized fixed-target logit recovery")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    if output_path is not None:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def plot_marginal_efficiency(rows: pd.DataFrame, *, output_path: str | Path | None = None):
    import matplotlib.pyplot as plt

    data = add_marginal_efficiency(rows).dropna(subset=["marginal_recovery_per_l2"])
    if data.empty:
        raise ValueError("No marginal efficiency rows to plot")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for method, group in data.groupby("method"):
        curve = group.groupby("counterdelta_l2_frac", as_index=False)["marginal_recovery_per_l2"].median()
        ax.plot(curve["counterdelta_l2_frac"], curve["marginal_recovery_per_l2"], marker="o", linewidth=1.4, markersize=3, label=str(method))
    ax.set_xlabel("counterdelta_l2_frac")
    ax.set_ylabel("median marginal recovery per L2 budget")
    ax.set_title("Marginal recovery efficiency")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    if output_path is not None:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def cache_result(result: dict[str, pd.DataFrame], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(result, fh)
