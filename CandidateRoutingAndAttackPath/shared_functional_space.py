from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .candidate_reserve import _cache_lookup, _candidate_frames, _evaluate_batch
from .causal_repair import _load_inputs
from .common import StorageBudget, load_experiment, release_accelerator_memory, stable_hash
from .followup_common import ATTACK_PATH_DB, MANIFEST_CSV, TRACE_DB, balanced_subset, write_summary
from .mechanism_followup import _head_branches, _raw_from_branches
from .shared_candidate_mechanism import _scatter_windows
from .score_functional_subspace import _integrated_candidate_jacobian


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "shared_functional_space_outputs"


@dataclass(slots=True)
class SharedFunctionalSpaceConfig:
    output_dir: str = str(OUTPUT_ROOT)
    device: str = "mps"
    require_device: bool = True
    examples_per_group: int = 100
    train_fraction: float = 0.60
    path_steps: int = 3
    window_radius: int = 2
    ranks: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    causal_ranks: tuple[int, ...] = (1, 4, 16, 32)
    target_iou: float = 0.50
    detection_conf: float = 0.25
    candidate_min_score: float = 0.01
    random_max_iou: float = 0.10
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    nms_max_time_img: float = 1.0
    seed: int = 613
    max_output_gb: float = 2.0
    method_version: int = 1


def _split_ids(selected: pd.DataFrame, train_fraction: float, seed: int) -> pd.DataFrame:
    parts = []
    for offset, (group, frame) in enumerate(selected.groupby("analysis_group", sort=False)):
        shuffled = frame.sample(frac=1.0, random_state=int(seed) + offset).reset_index(drop=True)
        n_train = min(max(int(round(len(shuffled) * float(train_fraction))), 1), len(shuffled) - 1)
        shuffled["split"] = "test"
        shuffled.loc[: n_train - 1, "split"] = "train"
        parts.append(shuffled)
    return pd.concat(parts, ignore_index=True)


def _window_rows(tensor, selection: pd.DataFrame, radius: int) -> np.ndarray:
    """Extract candidate-centred windows from [N,C,H,W] or [C,H,W]."""

    import torch.nn.functional as functional

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    padded = functional.pad(tensor, (radius, radius, radius, radius))
    width = 2 * int(radius) + 1
    rows = []
    for row_index, item in enumerate(selection.itertuples(index=False)):
        y, x = int(item.y_index), int(item.x_index)
        source_index = row_index if len(tensor) == len(selection) else 0
        rows.append(
            padded[source_index, :, y : y + width, x : x + width]
            .detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
        )
    channels = int(tensor.shape[1])
    return np.asarray(rows, dtype=np.float32).reshape(len(rows), channels * width * width)


def _safe_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _candidate_functional_rows(gradients: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """First-order projection of each local delta onto its candidate gradient."""

    numerator = np.sum(gradients.astype(np.float64) * deltas.astype(np.float64), axis=1)
    denominator = np.sum(np.square(gradients.astype(np.float64)), axis=1)
    scale = numerator / np.maximum(denominator, 1e-12)
    return (scale[:, None] * gradients).astype(np.float32)


def _fit_basis(matrix: np.ndarray, max_rank: int, seed: int, normalize: bool) -> np.ndarray:
    from sklearn.utils.extmath import randomized_svd

    matrix = np.asarray(matrix, dtype=np.float32)
    if normalize:
        matrix = _safe_normalize_rows(matrix)
    valid = np.linalg.norm(matrix.astype(np.float64), axis=1) > 1e-12
    matrix = matrix[valid]
    if not len(matrix):
        return np.zeros((0, matrix.shape[1]), dtype=np.float32)
    rank = min(int(max_rank), int(matrix.shape[0]), int(matrix.shape[1]))
    _u, _s, vh = randomized_svd(
        matrix,
        n_components=rank,
        n_iter=5,
        random_state=int(seed),
    )
    return vh.astype(np.float32)


def _random_basis(dimension: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    matrix = rng.standard_normal((int(dimension), int(rank)), dtype=np.float32)
    q, _r = np.linalg.qr(matrix, mode="reduced")
    return q.T.astype(np.float32)


def _project_rows(matrix: np.ndarray, basis: np.ndarray, rank: int) -> np.ndarray:
    use = min(int(rank), len(basis))
    if use <= 0:
        return np.zeros_like(matrix, dtype=np.float32)
    selected = basis[:use].astype(np.float32)
    return ((matrix @ selected.T) @ selected).astype(np.float32)


def _pool_key(group: str, setting: str) -> str:
    outcome = "visible" if str(group).startswith("visible_") else "hidden"
    if setting == "same_group":
        return str(group)
    if setting == "same_outcome":
        return outcome
    if setting == "cross_outcome":
        return "hidden" if outcome == "visible" else "visible"
    return "all"


def _collect_records(
    config: SharedFunctionalSpaceConfig,
    *,
    test_patch_path: str | Path | None = None,
):
    selected, _ = _load_inputs(Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None)
    selected = balanced_subset(selected, config.examples_per_group, seed=config.seed)
    selected = _split_ids(selected, config.train_fraction, config.seed + 17)
    frames = _candidate_frames(selected, config)
    exp, cache_path = load_experiment(
        prefer_device=config.device, require_device=bool(config.require_device)
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    cache = _cache_lookup(exp)
    test_patch = None
    if test_patch_path is not None:
        from PIL import Image

        test_patch = Image.open(Path(test_patch_path)).convert("RGB")
        if test_patch.size != (160, 160):
            raise ValueError(
                f"Cross-patch surface must be 160x160 for the fixed-corner protocol, "
                f"got {test_patch.size}"
            )
    records: list[dict] = []
    progress = tqdm(
        selected.itertuples(index=False),
        total=len(selected),
        desc="canonical Jacobians",
        unit="image",
    )
    for row in progress:
        example_id = str(row.example_id)
        if example_id not in frames:
            continue
        example = cache[example_id]
        clean_image, patched_image, _ = exp._images_for_example(example)
        if test_patch is not None and str(row.split) == "test":
            from new_experiments.patch_success_analysis.patching import apply_patch_to_image

            patched_image, _bbox, _area = apply_patch_to_image(
                clean_image,
                test_patch,
                position_xy=tuple(exp.config.attack.patch_xy),
            )
        pair = _preprocess_pair(exp, clean_image, patched_image)
        captured = _capture_detect_inputs(model, detect, pair)
        clean_inputs = [item[0:1] for item in captured]
        patched_inputs = [item[1:2] for item in captured]
        target_set = frames[example_id]["target_set"]
        for level in range(len(clean_inputs)):
            selection = target_set[target_set.level_index.astype(int).eq(level)].reset_index(drop=True)
            if selection.empty:
                continue
            gradients = _integrated_candidate_jacobian(
                detect.cv3[level],
                clean_inputs[level],
                patched_inputs[level],
                selection,
                int(row.class_id),
                config.path_steps,
            )
            gradient_windows = _window_rows(gradients, selection, config.window_radius)
            delta = (patched_inputs[level][0] - clean_inputs[level][0]).detach()
            delta_windows = _window_rows(delta, selection, config.window_radius)
            records.append(
                {
                    "example_id": example_id,
                    "analysis_group": str(row.analysis_group),
                    "split": str(row.split),
                    "level": int(level),
                    "class_id": int(row.class_id),
                    "selection": selection,
                    "gradients": gradient_windows,
                    "deltas": delta_windows,
                    "shape": tuple(int(value) for value in clean_inputs[level][0].shape),
                }
            )
        release_accelerator_memory()
    return selected, frames, records, exp, model, detect, cache, cache_path


def _training_pools(records: list[dict]) -> dict[tuple[str, int, str], np.ndarray]:
    rows: dict[tuple[str, int, str], list[np.ndarray]] = {}
    for item in records:
        if item["split"] != "train":
            continue
        group = item["analysis_group"]
        outcome = "visible" if group.startswith("visible_") else "hidden"
        for pool in ("all", group, outcome):
            rows.setdefault(("sensitivity", item["level"], pool), []).append(item["gradients"])
            functional = _candidate_functional_rows(item["gradients"], item["deltas"])
            rows.setdefault(("attack", item["level"], pool), []).append(functional)
    return {key: np.concatenate(values, axis=0) for key, values in rows.items()}


def _build_bases(records: list[dict], config: SharedFunctionalSpaceConfig):
    max_rank = max(config.ranks)
    pools = _training_pools(records)
    bases: dict[tuple[str, int, str], np.ndarray] = {}
    for offset, (key, matrix) in enumerate(sorted(pools.items(), key=lambda item: str(item[0]))):
        family = key[0]
        bases[key] = _fit_basis(
            matrix,
            max_rank=max_rank,
            seed=config.seed + 101 * offset,
            normalize=(family == "sensitivity"),
        )
    return bases


def _basis_for_record(
    item: dict,
    family: str,
    setting: str,
    bases: dict,
    config: SharedFunctionalSpaceConfig,
) -> np.ndarray:
    if setting == "oracle_image":
        source = (
            item["gradients"]
            if family == "sensitivity"
            else _candidate_functional_rows(item["gradients"], item["deltas"])
        )
        return _fit_basis(
            source,
            max_rank=max(config.ranks),
            seed=config.seed + int(item["level"]),
            normalize=(family == "sensitivity"),
        )
    if setting == "random":
        return _random_basis(
            item["deltas"].shape[1],
            max(config.ranks),
            config.seed + 1009 * int(item["level"]),
        )
    pool = _pool_key(item["analysis_group"], setting)
    return bases.get(
        (family, int(item["level"]), pool),
        np.zeros((0, item["deltas"].shape[1]), dtype=np.float32),
    )


def _analytic_rows(records: list[dict], bases: dict, config: SharedFunctionalSpaceConfig):
    rows = []
    settings = ("oracle_image", "same_group", "same_outcome", "pooled_all", "cross_outcome", "random")
    families = ("sensitivity", "attack")
    for item in records:
        if item["split"] != "test":
            continue
        gradients = item["gradients"].astype(np.float64)
        deltas = item["deltas"].astype(np.float64)
        true_effect = np.sum(gradients * deltas, axis=1)
        gradient_energy = np.sum(np.square(gradients), axis=1)
        delta_energy = np.sum(np.square(deltas), axis=1)
        for family in families:
            for setting in settings:
                if setting == "random" and family == "attack":
                    continue
                basis = _basis_for_record(item, family, setting, bases, config)
                for rank in config.ranks:
                    projected_gradient = _project_rows(item["gradients"], basis, rank).astype(np.float64)
                    projected_delta = _project_rows(item["deltas"], basis, rank).astype(np.float64)
                    predicted_effect = np.sum(gradients * projected_delta, axis=1)
                    for candidate_index in range(len(gradients)):
                        rows.append(
                            {
                                "example_id": item["example_id"],
                                "analysis_group": item["analysis_group"],
                                "level": item["level"],
                                "candidate_index": candidate_index,
                                "family": family,
                                "setting": setting,
                                "rank": int(rank),
                                "gradient_coverage": float(
                                    np.sum(np.square(projected_gradient[candidate_index]))
                                    / max(gradient_energy[candidate_index], 1e-12)
                                ),
                                "delta_energy_fraction": float(
                                    np.sum(np.square(projected_delta[candidate_index]))
                                    / max(delta_energy[candidate_index], 1e-12)
                                ),
                                "true_linear_effect": float(true_effect[candidate_index]),
                                "captured_linear_effect": float(predicted_effect[candidate_index]),
                                "effect_residual": float(
                                    true_effect[candidate_index] - predicted_effect[candidate_index]
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def _analytic_summary(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    keys = ["family", "setting", "rank"]
    for values, frame in rows.groupby(keys, sort=False):
        true = frame.true_linear_effect.to_numpy(float)
        predicted = frame.captured_linear_effect.to_numpy(float)
        denominator = max(float(np.sum(np.square(true))), 1e-12)
        output.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n_examples": int(frame.example_id.nunique()),
                "n_candidates": int(len(frame)),
                "mean_gradient_coverage": float(frame.gradient_coverage.mean()),
                "mean_delta_energy_fraction": float(frame.delta_energy_fraction.mean()),
                "effect_r2_zero": float(1.0 - np.sum(np.square(true - predicted)) / denominator),
                "effect_cosine": float(
                    np.dot(true, predicted)
                    / max(np.linalg.norm(true) * np.linalg.norm(predicted), 1e-12)
                ),
            }
        )
    return pd.DataFrame(output)


def _causal_conditions(config: SharedFunctionalSpaceConfig) -> list[tuple[str, str, int]]:
    settings = ("oracle_image", "same_group", "same_outcome", "pooled_all", "cross_outcome")
    conditions = [("none", "none", 0)]
    for family in ("sensitivity", "attack"):
        for setting in settings:
            for rank in config.causal_ranks:
                conditions.append((family, setting, int(rank)))
    for rank in config.causal_ranks:
        conditions.append(("sensitivity", "random", int(rank)))
    return conditions


def _component_maps(item_records: list[dict], bases: dict, conditions, config, base_inputs):
    import torch

    maps = {
        (family, setting, rank): [
            torch.zeros_like(level_input[0], dtype=torch.float32, device="cpu")
            for level_input in base_inputs
        ]
        for family, setting, rank in conditions
        if family != "none"
    }
    by_level = {int(item["level"]): item for item in item_records}
    for family, setting, rank in conditions:
        if family == "none":
            continue
        for level, item in by_level.items():
            basis = _basis_for_record(item, family, setting, bases, config)
            projected = _project_rows(item["deltas"], basis, rank)
            maps[(family, setting, rank)][level] = _scatter_windows(
                projected,
                item["selection"],
                item["shape"],
                config.window_radius,
            )
    return maps


def _causal_rows(
    selected,
    records,
    frames,
    bases,
    exp,
    model,
    detect,
    cache,
    config,
    *,
    test_patch_path: str | Path | None = None,
):
    import torch

    test = selected[selected.split.eq("test")].reset_index(drop=True)
    records_by_id: dict[str, list[dict]] = {}
    for item in records:
        if item["split"] == "test":
            records_by_id.setdefault(item["example_id"], []).append(item)
    conditions = _causal_conditions(config)
    test_patch = None
    if test_patch_path is not None:
        from PIL import Image

        test_patch = Image.open(Path(test_patch_path)).convert("RGB")
    output = []
    progress = tqdm(test.itertuples(index=False), total=len(test), desc="holdout repair", unit="image")
    for row in progress:
        example_id = str(row.example_id)
        if example_id not in records_by_id:
            continue
        example = cache[example_id]
        clean_image, patched_image, _ = exp._images_for_example(example)
        if test_patch is not None:
            from new_experiments.patch_success_analysis.patching import apply_patch_to_image

            patched_image, _bbox, _area = apply_patch_to_image(
                clean_image,
                test_patch,
                position_xy=tuple(exp.config.attack.patch_xy),
            )
        pair = _preprocess_pair(exp, clean_image, patched_image)
        captured = _capture_detect_inputs(model, detect, pair)
        patched_inputs = [item[1:2] for item in captured]
        maps = _component_maps(
            records_by_id[example_id], bases, conditions, config, patched_inputs
        )
        with torch.inference_mode():
            patched_box, _patched_cls, _ = _head_branches(detect, patched_inputs)
            for start in range(0, len(conditions), 8):
                chunk = conditions[start : start + 8]
                class_inputs = []
                names = []
                for family, setting, rank in chunk:
                    names.append((family, setting, rank))
                    if family == "none":
                        class_inputs.append(patched_inputs)
                    else:
                        level_maps = maps[(family, setting, rank)]
                        class_inputs.append(
                            [
                                patched_inputs[level]
                                - level_maps[level]
                                .to(patched_inputs[level].device, patched_inputs[level].dtype)
                                .unsqueeze(0)
                                for level in range(len(patched_inputs))
                            ]
                        )
                class_outputs = [
                    detect.cv3[level](torch.cat([values[level] for values in class_inputs], dim=0))
                    for level in range(len(patched_inputs))
                ]
                box_outputs = [item.repeat(len(chunk), 1, 1, 1) for item in patched_box]
                raw = _raw_from_branches(box_outputs, class_outputs)
                results = _evaluate_batch(detect, raw, row, config)
                for name, result in zip(names, results, strict=True):
                    family, setting, rank = name
                    result.update(
                        {
                            "example_id": example_id,
                            "analysis_group": str(row.analysis_group),
                            "family": family,
                            "setting": setting,
                            "rank": int(rank),
                        }
                    )
                    output.append(result)
        release_accelerator_memory()
    rows = pd.DataFrame(output)
    baseline = rows[rows.family.eq("none")][["example_id", "target_hidden"]].rename(
        columns={"target_hidden": "baseline_hidden"}
    )
    return rows.merge(baseline, on="example_id", validate="many_to_one")


def _causal_summary(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for values, frame in rows.groupby(["family", "setting", "rank"], sort=False):
        hidden = frame.baseline_hidden.astype(bool)
        output.append(
            {
                "family": values[0],
                "setting": values[1],
                "rank": int(values[2]),
                "n_examples": int(frame.example_id.nunique()),
                "n_hidden_baseline": int(hidden.sum()),
                "target_hidden_rate": float(frame.target_hidden.mean()),
                "recovery_rate": float(frame.loc[hidden, "target_detected"].mean())
                if hidden.any()
                else np.nan,
                "clean_endpoint_damage_rate": float(frame.loc[~hidden, "target_hidden"].mean())
                if (~hidden).any()
                else np.nan,
                "mean_pre_target_conf": float(frame.pre_target_conf.mean()),
            }
        )
    return pd.DataFrame(output)


def run_shared_functional_space(config: SharedFunctionalSpaceConfig | None = None) -> Path:
    config = config or SharedFunctionalSpaceConfig()
    started = time.time()
    StorageBudget(config.output_dir, config.max_output_gb).check()
    selected, frames, records, exp, model, detect, cache, cache_path = _collect_records(config)
    bases = _build_bases(records, config)
    analytic_rows = _analytic_rows(records, bases, config)
    analytic_summary = _analytic_summary(analytic_rows)
    causal_rows = _causal_rows(
        selected, records, frames, bases, exp, model, detect, cache, config
    )
    causal_summary = _causal_summary(causal_rows)
    payload = {**asdict(config), "example_ids": selected.example_id.tolist()}
    run_dir = Path(config.output_dir) / f"shared_space_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(run_dir / "split.csv", index=False)
    analytic_rows.to_csv(run_dir / "analytic_rows.csv", index=False)
    analytic_summary.to_csv(run_dir / "analytic_summary.csv", index=False)
    causal_rows.to_csv(run_dir / "causal_rows.csv", index=False)
    causal_summary.to_csv(run_dir / "causal_summary.csv", index=False)
    basis_manifest = []
    for (family, level, pool), basis in bases.items():
        basis_manifest.append(
            {
                "family": family,
                "level": level,
                "pool": pool,
                "available_rank": len(basis),
                "dimension": basis.shape[1] if basis.ndim == 2 else 0,
            }
        )
    pd.DataFrame(basis_manifest).to_csv(run_dir / "basis_manifest.csv", index=False)
    elapsed = time.time() - started
    pooled = analytic_summary[
        analytic_summary.setting.eq("pooled_all")
        & analytic_summary["rank"].eq(max(config.ranks))
    ]
    causal_pooled = causal_summary[
        causal_summary.setting.eq("pooled_all")
        & causal_summary["rank"].eq(max(config.causal_ranks))
    ]
    summary = {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "n_examples": int(selected.example_id.nunique()),
        "n_train": int(selected.split.eq("train").sum()),
        "n_test": int(selected.split.eq("test").sum()),
        "cache_path": str(cache_path),
        "config": asdict(config),
        "pooled_rank_max_analytic": pooled.to_dict("records"),
        "pooled_rank_max_causal": causal_pooled.to_dict("records"),
        "interpretation_rule": (
            "A shared space is supported only if pooled/cross-outcome holdout gradient coverage "
            "and causal recovery exceed the same-rank random basis while using a small delta-energy fraction."
        ),
        "scope": (
            "One detector and one fixed adversarial patch; cross-patch and cross-model transfer "
            "must be tested only if the across-image holdout result is positive."
        ),
    }
    write_summary(run_dir / "summary.json", summary)
    (Path(config.output_dir) / "LATEST.txt").write_text(str(run_dir.resolve()) + "\n", encoding="utf-8")
    StorageBudget(config.output_dir, config.max_output_gb).check()
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find a shared low-dimensional score-functional space across images."
    )
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    parser.add_argument("--examples-per-group", type=int, default=100)
    parser.add_argument("--path-steps", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    smoke = bool(args.smoke)
    config = SharedFunctionalSpaceConfig(
        output_dir=args.output_dir,
        device=args.device,
        require_device=args.device == "mps",
        examples_per_group=2 if smoke else int(args.examples_per_group),
        train_fraction=float(args.train_fraction),
        path_steps=1 if smoke else int(args.path_steps),
        ranks=(1, 2, 4) if smoke else (1, 2, 4, 8, 16, 32),
        causal_ranks=(1, 4) if smoke else (1, 4, 16, 32),
        method_version=9001 if smoke else 1,
    )
    print(run_shared_functional_space(config))
