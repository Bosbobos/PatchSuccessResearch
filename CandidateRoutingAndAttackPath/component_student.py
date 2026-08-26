from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .candidate_reserve import _cache_lookup, _evaluate_batch
from .causal_repair import _load_inputs
from .common import REPO_ROOT, load_experiment, stable_hash
from .component_targeted_patch import (
    ComponentTargetedPatchConfig,
    _balanced_train_eval,
    _record_lookup,
    build_teacher_cache,
)
from .followup_common import ATTACK_PATH_DB, MANIFEST_CSV, TRACE_DB
from .mechanism_aware_patch import dynamic_score_geometry_loss
from .mechanism_followup import _head_branches
from .self_counterfactual_defense import _all_class_nms, _detection_set_metrics


DEFAULT_OUTPUT_DIR = REPO_ROOT / "CandidateRoutingAndAttackPath" / "component_student_outputs"
FEATURE_SETS = ("activation", "local", "functional", "combined")


@dataclass(slots=True)
class ComponentStudentConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "cpu"
    require_device: bool = False
    train_examples_per_group: int = 4
    eval_examples_per_group: int = 4
    teacher_path_steps: int = 3
    feature_sets: tuple[str, ...] = FEATURE_SETS
    max_coordinates_per_image: int = 12000
    max_iter: int = 120
    max_leaf_nodes: int = 31
    learning_rate: float = 0.08
    l2_regularization: float = 1.0
    clean_behavior: str = "zero"
    target_iou: float = 0.50
    detection_conf: float = 0.25
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    nms_max_time_img: float = 1.0
    smoothmax_temperature: float = 0.35
    dynamic_iou_temperature: float = 0.07
    dynamic_iou_weight: float = 4.0
    init_patch: str = str(REPO_ROOT / "data" / "patch.png")
    seed: int = 811
    method_version: int = 1

    @property
    def match_iou(self) -> float:
        return float(self.target_iou)


def _capture_with_grad(model, detect, inputs):
    from segmentig_detector.yolo_utils import safe_model_forward

    captured: dict[str, Any] = {}

    def hook(_module, args):
        captured["levels"] = list(args[0])

    handle = detect.register_forward_pre_hook(hook)
    try:
        output = safe_model_forward(model, inputs)
    finally:
        handle.remove()
    decoded = output[0] if isinstance(output, (list, tuple)) else output
    return decoded, captured["levels"]


def _reference_statistics(exp, model, detect, records):
    sums = sums2 = counts = None
    examples = _cache_lookup(exp)
    for record in records:
        example = examples[record["example_id"]]
        clean_image, _patched, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, clean_image)
        levels = _capture_detect_inputs(model, detect, pair[:1])
        if sums is None:
            sums = [np.zeros(level.shape[1], dtype=np.float64) for level in levels]
            sums2 = [np.zeros(level.shape[1], dtype=np.float64) for level in levels]
            counts = [0 for _ in levels]
        for index, level in enumerate(levels):
            array = level[0].float().cpu().numpy()
            sums[index] += array.sum(axis=(1, 2))
            sums2[index] += np.square(array, dtype=np.float64).sum(axis=(1, 2))
            counts[index] += int(array.shape[1] * array.shape[2])
    means = [value / max(count, 1) for value, count in zip(sums, counts, strict=True)]
    stds = [
        np.sqrt(np.maximum(second / max(count, 1) - mean**2, 1e-6))
        for second, mean, count in zip(sums2, means, counts, strict=True)
    ]
    return means, stds


def _coordinate_arrays(level, indices, reference_mean, reference_std, gradient=None):
    import torch
    import torch.nn.functional as functional

    channels, height, width = level.shape
    if len(indices) == 0:
        empty = np.empty(0, dtype=np.float32)
        return {
            key: empty.copy()
            for key in (
                "z",
                "level",
                "channel",
                "y",
                "x",
                "local_mean_z",
                "local_std_z",
                "center_minus_local",
                "gradient",
                "z_gradient",
            )
        }
    flat = level.reshape(-1)
    indices_t = torch.as_tensor(indices, device=level.device, dtype=torch.long)
    values = flat[indices_t].detach().float().cpu().numpy()
    channel = indices // (height * width)
    spatial = indices % (height * width)
    y = spatial // width
    x = spatial % width
    z = (values - reference_mean[channel]) / reference_std[channel]
    tensor = level.unsqueeze(0).float()
    local_mean = functional.avg_pool2d(tensor, 3, 1, 1)[0]
    local_second = functional.avg_pool2d(tensor.square(), 3, 1, 1)[0]
    local_std = torch.sqrt((local_second - local_mean.square()).clamp_min(1e-6))
    local_mean_values = local_mean.reshape(-1)[indices_t].detach().cpu().numpy()
    local_std_values = local_std.reshape(-1)[indices_t].detach().cpu().numpy()
    output = {
        "z": z.astype(np.float32),
        "level": np.zeros(len(indices), dtype=np.float32),
        "channel": (channel / max(channels - 1, 1)).astype(np.float32),
        "y": (y / max(height - 1, 1)).astype(np.float32),
        "x": (x / max(width - 1, 1)).astype(np.float32),
        "local_mean_z": (
            (local_mean_values - reference_mean[channel]) / reference_std[channel]
        ).astype(np.float32),
        "local_std_z": (local_std_values / reference_std[channel]).astype(np.float32),
        "center_minus_local": (
            (values - local_mean_values) / reference_std[channel]
        ).astype(np.float32),
    }
    if gradient is None:
        gradient_values = np.zeros(len(indices), dtype=np.float32)
    else:
        gradient_values = (
            gradient.reshape(-1)[indices_t].detach().float().cpu().numpy()
        )
    gradient_scale = max(float(np.sqrt(np.mean(np.square(gradient_values)))), 1e-8)
    gradient_values = gradient_values / gradient_scale
    output["gradient"] = gradient_values.astype(np.float32)
    output["z_gradient"] = (z * gradient_values).astype(np.float32)
    return output


def _feature_matrix(parts: list[dict[str, np.ndarray]], feature_set: str):
    keys = ["z", "level", "channel", "y", "x"]
    if feature_set in {"local", "combined"}:
        keys += ["local_mean_z", "local_std_z", "center_minus_local"]
    if feature_set in {"functional", "combined"}:
        keys += ["gradient", "z_gradient"]
    return np.stack(
        [np.concatenate([part[key] for part in parts]) for key in keys], axis=1
    ).astype(np.float32), keys


def _endpoint_features(
    model,
    detect,
    inputs,
    record,
    target_box,
    class_id,
    reference_mean,
    reference_std,
    *,
    require_gradient: bool,
):
    import torch

    if require_gradient:
        _reset_detect_inference_cache(detect)
        inputs = inputs.detach().requires_grad_(True)
    decoded, levels = _capture_with_grad(model, detect, inputs)
    gradients = [None for _ in levels]
    if require_gradient:
        loss, _ = dynamic_score_geometry_loss(
            decoded,
            target_box,
            class_id,
            match_iou=0.50,
            iou_temperature=0.07,
            iou_weight=4.0,
            smoothmax_temperature=0.35,
        )
        gradients = torch.autograd.grad(loss, levels, retain_graph=False)
    parts = []
    targets = []
    with np.load(record["component_path"], allow_pickle=False) as data:
        for level_index, level in enumerate(levels):
            indices = data[f"indices_{level_index}"].astype(np.int64)
            arrays = _coordinate_arrays(
                level[0],
                indices,
                reference_mean[level_index],
                reference_std[level_index],
                gradients[level_index][0] if gradients[level_index] is not None else None,
            )
            arrays["level"].fill(level_index / max(len(levels) - 1, 1))
            parts.append(arrays)
            targets.append(data[f"component_{level_index}"].astype(np.float32))
    return parts, np.concatenate(targets), levels


def _sample_indices(target, maximum: int, seed: int):
    if len(target) <= maximum:
        return np.arange(len(target), dtype=np.int64)
    rng = np.random.default_rng(seed)
    half = maximum // 2
    top = np.argsort(-np.abs(target), kind="stable")[:half]
    remaining = np.setdiff1d(np.arange(len(target)), top, assume_unique=False)
    random = rng.choice(remaining, size=maximum - len(top), replace=False)
    return np.concatenate([top, random]).astype(np.int64)


def _subset_parts(parts, indices):
    sizes = [len(part["z"]) for part in parts]
    offsets = np.cumsum([0, *sizes])
    output = []
    for level, part in enumerate(parts):
        local = indices[(indices >= offsets[level]) & (indices < offsets[level + 1])]
        local = local - offsets[level]
        output.append({key: value[local] for key, value in part.items()})
    return output


def _build_training_dataset(
    exp, model, detect, records, reference_mean, reference_std, config
):
    examples = _cache_lookup(exp)
    endpoint_rows = []
    for record_index, record in enumerate(records):
        example = examples[record["example_id"]]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        import torch

        target = torch.as_tensor([record["target_box"]], device=pair.device, dtype=torch.float32)
        classes = torch.as_tensor([record["class_id"]], device=pair.device, dtype=torch.long)
        for endpoint_index, endpoint in enumerate(("patched", "clean")):
            parts, teacher, _levels = _endpoint_features(
                model,
                detect,
                pair[1:2] if endpoint == "patched" else pair[0:1],
                record,
                target,
                classes,
                reference_mean,
                reference_std,
                require_gradient=True,
            )
            labels = teacher if endpoint == "patched" else np.zeros_like(teacher)
            keep = _sample_indices(
                labels if endpoint == "patched" else teacher,
                int(config.max_coordinates_per_image),
                config.seed + 101 * record_index + endpoint_index,
            )
            endpoint_rows.append({
                "parts": _subset_parts(parts, keep),
                "labels": labels[keep],
                "endpoint": endpoint,
            })
    return endpoint_rows


def _fit_models(rows, config):
    from sklearn.ensemble import HistGradientBoostingRegressor

    target = np.concatenate([row["labels"] for row in rows])
    target_scale = max(float(np.std(target)), 1e-8)
    y = target / target_scale
    weights = 1.0 + 4.0 * np.minimum(np.abs(y), 5.0)
    models = {}
    feature_names = {}
    for feature_set in config.feature_sets:
        matrices = [
            _feature_matrix(row["parts"], feature_set)[0] for row in rows
        ]
        x = np.concatenate(matrices, axis=0)
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(config.learning_rate),
            max_iter=int(config.max_iter),
            max_leaf_nodes=int(config.max_leaf_nodes),
            l2_regularization=float(config.l2_regularization),
            random_state=int(config.seed),
        )
        model.fit(x, y, sample_weight=weights)
        models[feature_set] = model
        feature_names[feature_set] = _feature_matrix(rows[0]["parts"], feature_set)[1]
    return models, feature_names, target_scale


def _reset_detect_inference_cache(detect) -> None:
    """Do not reuse inference-only anchor tensors in the next gradient pass."""

    if hasattr(detect, "shape"):
        detect.shape = None


def _evaluate_students(
    exp,
    model,
    detect,
    records,
    models,
    reference_mean,
    reference_std,
    target_scale,
    config,
):
    import torch

    examples = _cache_lookup(exp)
    rows = []
    for record in records:
        example = examples[record["example_id"]]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        target = torch.as_tensor([record["target_box"]], device=pair.device, dtype=torch.float32)
        classes = torch.as_tensor([record["class_id"]], device=pair.device, dtype=torch.long)
        endpoint_data = {}
        for endpoint, tensor in (("patched", pair[1:2]), ("clean", pair[0:1])):
            parts, teacher, levels = _endpoint_features(
                model,
                detect,
                tensor,
                record,
                target,
                classes,
                reference_mean,
                reference_std,
                require_gradient=True,
            )
            endpoint_data[endpoint] = (parts, teacher, levels)
        with torch.no_grad():
            clean_box, clean_cls, clean_raw = _head_branches(
                detect, endpoint_data["clean"][2]
            )
            patched_box, patched_cls, patched_raw = _head_branches(
                detect, endpoint_data["patched"][2]
            )
            clean_nms = _all_class_nms(detect, clean_raw, config)[0]
        for feature_set, student in models.items():
            condition_inputs = {}
            component_metrics = {}
            for endpoint in ("patched", "clean"):
                parts, teacher, levels = endpoint_data[endpoint]
                matrix, _ = _feature_matrix(parts, feature_set)
                prediction = student.predict(matrix).astype(np.float32) * target_scale
                predicted_levels = []
                cursor = 0
                with np.load(record["component_path"], allow_pickle=False) as data:
                    for level_index, level in enumerate(levels):
                        indices_np = data[f"indices_{level_index}"].astype(np.int64)
                        count = len(indices_np)
                        values = prediction[cursor:cursor + count]
                        cursor += count
                        correction = torch.zeros_like(level[0]).reshape(-1)
                        indices = torch.as_tensor(
                            indices_np, device=level.device, dtype=torch.long
                        )
                        correction[indices] = torch.as_tensor(
                            values, device=level.device, dtype=level.dtype
                        )
                        predicted_levels.append(level - correction.reshape_as(level[0]).unsqueeze(0))
                condition_inputs[endpoint] = predicted_levels
                reference = teacher if endpoint == "patched" else np.zeros_like(teacher)
                denominator = max(
                    float(np.linalg.norm(prediction) * np.linalg.norm(reference)), 1e-12
                )
                component_metrics[endpoint] = {
                    "prediction_l2": float(np.linalg.norm(prediction)),
                    "teacher_l2": float(np.linalg.norm(reference)),
                    "component_cosine": float(np.dot(prediction, reference) / denominator),
                    "component_nrmse": float(
                        np.linalg.norm(prediction - reference)
                        / max(np.linalg.norm(reference), 1e-12)
                    ),
                }
            with torch.no_grad():
                conditions = {
                    "patched_observed": endpoint_data["patched"][2],
                    "patched_student": condition_inputs["patched"],
                    "clean_observed": endpoint_data["clean"][2],
                    "clean_student": condition_inputs["clean"],
                }
                names = list(conditions)
                batched = [
                    torch.cat([conditions[name][level] for name in names], dim=0)
                    for level in range(len(endpoint_data["clean"][2]))
                ]
                _box, _cls, raw = _head_branches(detect, batched)
                target_results = _evaluate_batch(detect, raw, record["row"], config)
                all_nms = _all_class_nms(detect, raw, config)
            by_name = dict(zip(names, target_results, strict=True))
            nms_by_name = dict(zip(names, all_nms, strict=True))
            clean_set = _detection_set_metrics(
                clean_nms, nms_by_name["clean_student"], config.target_iou
            )
            patched_teacher_l2 = component_metrics["patched"]["teacher_l2"]
            rows.append({
                "example_id": record["example_id"],
                "feature_set": feature_set,
                "patched_baseline_detected": by_name["patched_observed"]["target_detected"],
                "patched_corrected_detected": by_name["patched_student"]["target_detected"],
                "patched_baseline_conf": by_name["patched_observed"]["post_target_conf"],
                "patched_corrected_conf": by_name["patched_student"]["post_target_conf"],
                "clean_baseline_detected": by_name["clean_observed"]["target_detected"],
                "clean_corrected_detected": by_name["clean_student"]["target_detected"],
                "clean_detection_f1": clean_set["detection_f1"],
                "patched_component_cosine": component_metrics["patched"]["component_cosine"],
                "patched_component_nrmse": component_metrics["patched"]["component_nrmse"],
                "patched_teacher_l2": patched_teacher_l2,
                "clean_prediction_l2": component_metrics["clean"]["prediction_l2"],
                "clean_prediction_ratio": (
                    component_metrics["clean"]["prediction_l2"]
                    / max(patched_teacher_l2, 1e-12)
                ),
            })
        _reset_detect_inference_cache(detect)
    return pd.DataFrame(rows)


def run_component_student(config: ComponentStudentConfig | None = None) -> Path:
    config = config or ComponentStudentConfig()
    if config.clean_behavior != "zero":
        raise ValueError(
            "This experiment currently trains the conservative zero-on-clean behavior."
        )
    started = time.time()
    selected, _ = _load_inputs(
        Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None
    )
    split_config = ComponentTargetedPatchConfig(
        output_dir=str(
            REPO_ROOT / "CandidateRoutingAndAttackPath" / "component_patch_outputs"
        ),
        device=config.device,
        require_device=config.require_device,
        train_examples_per_group=config.train_examples_per_group,
        eval_examples_per_group=config.eval_examples_per_group,
        teacher_path_steps=config.teacher_path_steps,
        init_patch=config.init_patch,
        seed=613,
        method_version=2,
    )
    train_rows, eval_rows = _balanced_train_eval(selected, split_config)
    teacher_rows = pd.concat([train_rows, eval_rows], ignore_index=True)
    exp, attack_cache_path = load_experiment(
        prefer_device=config.device, require_device=config.require_device
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, detector = exp.load_model()
    detector.eval()
    detect = get_detect_module(detector, exp.config.detect_layer)
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    teacher_cache_dir, manifest = build_teacher_cache(
        split_config,
        exp=exp,
        model=detector,
        detect=detect,
        rows=teacher_rows,
        cache_path=attack_cache_path,
    )
    train = _record_lookup(exp, train_rows, manifest)
    evaluation = _record_lookup(exp, eval_rows, manifest)
    row_lookup = {
        str(row.example_id): row for row in teacher_rows.itertuples(index=False)
    }
    for record in [*train, *evaluation]:
        record["row"] = row_lookup[record["example_id"]]
    reference_mean, reference_std = _reference_statistics(
        exp, detector, detect, train
    )
    dataset = _build_training_dataset(
        exp, detector, detect, train, reference_mean, reference_std, config
    )
    students, feature_names, target_scale = _fit_models(dataset, config)
    results = _evaluate_students(
        exp,
        detector,
        detect,
        evaluation,
        students,
        reference_mean,
        reference_std,
        target_scale,
        config,
    )
    payload = {
        **asdict(config),
        "teacher_cache_dir": str(teacher_cache_dir),
        "train_ids": [item["example_id"] for item in train],
        "eval_ids": [item["example_id"] for item in evaluation],
    }
    run_dir = Path(config.output_dir) / f"component_student_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(run_dir / "student_rows.csv", index=False)
    summary = results.groupby("feature_set", as_index=False).agg(
        n=("example_id", "nunique"),
        baseline_target_rate=("patched_baseline_detected", "mean"),
        corrected_target_rate=("patched_corrected_detected", "mean"),
        baseline_target_conf=("patched_baseline_conf", "mean"),
        corrected_target_conf=("patched_corrected_conf", "mean"),
        clean_target_preservation=("clean_corrected_detected", "mean"),
        clean_detection_f1=("clean_detection_f1", "mean"),
        component_cosine=("patched_component_cosine", "mean"),
        component_nrmse=("patched_component_nrmse", "mean"),
        clean_prediction_l2=("clean_prediction_l2", "mean"),
        clean_prediction_ratio=("clean_prediction_ratio", "mean"),
    )
    summary.to_csv(run_dir / "student_summary.csv", index=False)
    for feature_set, student in students.items():
        joblib.dump(student, run_dir / f"student_{feature_set}.joblib")
    metadata = {
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "target_scale": target_scale,
        "feature_names": feature_names,
        "teacher_cache_dir": str(teacher_cache_dir),
        "config": asdict(config),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "analysis_digest.md").write_text(
        "# Single-endpoint component student\n\n"
        f"- train/eval: {len(train)}/{len(evaluation)}\n"
        f"- clean behavior: {config.clean_behavior}\n"
        f"- elapsed: {metadata['elapsed_seconds']:.1f} s\n\n"
        + summary.to_csv(index=False),
        encoding="utf-8",
    )
    return run_dir


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train single-endpoint students to predict a defensive correction."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-examples-per-group", type=int, default=4)
    parser.add_argument("--eval-examples-per-group", type=int, default=4)
    parser.add_argument("--max-coordinates-per-image", type=int, default=12000)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.smoke:
        args.train_examples_per_group = 1
        args.eval_examples_per_group = 1
        args.max_coordinates_per_image = 2000
        args.max_iter = 20
    config = ComponentStudentConfig(
        output_dir=args.output_dir,
        device=args.device,
        train_examples_per_group=args.train_examples_per_group,
        eval_examples_per_group=args.eval_examples_per_group,
        max_coordinates_per_image=args.max_coordinates_per_image,
        max_iter=args.max_iter,
    )
    print(run_component_student(config))


if __name__ == "__main__":
    main()
