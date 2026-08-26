from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .candidate_reserve import _cache_lookup
from .causal_repair import _load_inputs
from .common import REPO_ROOT, load_experiment, release_accelerator_memory, stable_hash
from .followup_common import ATTACK_PATH_DB, GROUP_ORDER, MANIFEST_CSV, TRACE_DB
from .full_success_closure import (
    FullSuccessClosureConfig,
    _candidate_closure,
    _functional_components,
    _local_indices,
)
from .mechanism_aware_patch import (
    _decoded_from_model,
    _initial_patch,
    _load_batch,
    _save_patch,
    dynamic_score_geometry_loss,
    evaluate_patch,
    overlay_patch,
    total_variation,
)
from .mechanism_followup import _head_branches


DEFAULT_OUTPUT_DIR = REPO_ROOT / "CandidateRoutingAndAttackPath" / "component_patch_outputs"


@dataclass(slots=True)
class ComponentTargetedPatchConfig:
    """Train a defensive challenge patch against the causal joint component."""

    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "cpu"
    require_device: bool = False
    train_examples_per_group: int = 4
    eval_examples_per_group: int = 4
    teacher_path_steps: int = 3
    patch_size: int = 160
    patch_xy: tuple[int, int] = (0, 0)
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 0.05
    objective: str = "hybrid"
    component_scale: float = 1.25
    component_weight: float = 5.0
    null_weight: float = 0.01
    dynamic_weight: float = 1.0
    tv_weight: float = 0.002
    smoothmax_temperature: float = 0.35
    dynamic_iou_temperature: float = 0.07
    dynamic_iou_weight: float = 4.0
    target_iou: float = 0.50
    candidate_min_score: float = 0.01
    detection_conf: float = 0.25
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    init_patch: str = str(REPO_ROOT / "data" / "patch.png")
    seed: int = 613
    method_version: int = 2

    @property
    def match_iou(self) -> float:
        """Compatibility with the canonical target-aware evaluator."""

        return float(self.target_iou)


def _balanced_train_eval(selected: pd.DataFrame, config: ComponentTargetedPatchConfig):
    train_parts = []
    eval_parts = []
    for group_index, group in enumerate(GROUP_ORDER):
        frame = selected[selected.analysis_group.eq(group)].sample(
            frac=1.0, random_state=int(config.seed) + group_index
        )
        train_n = min(int(config.train_examples_per_group), len(frame))
        eval_n = min(int(config.eval_examples_per_group), max(0, len(frame) - train_n))
        train_parts.append(frame.iloc[:train_n])
        eval_parts.append(frame.iloc[train_n:train_n + eval_n])
    train = pd.concat(train_parts, ignore_index=True)
    evaluation = pd.concat(eval_parts, ignore_index=True)
    if train.empty or evaluation.empty:
        raise RuntimeError("A non-empty disjoint train/eval split is required.")
    return train, evaluation


def _teacher_config(config: ComponentTargetedPatchConfig) -> FullSuccessClosureConfig:
    return FullSuccessClosureConfig(
        device=config.device,
        require_device=config.require_device,
        path_steps=config.teacher_path_steps,
        target_iou=config.target_iou,
        candidate_min_score=config.candidate_min_score,
        detection_conf=config.detection_conf,
        nms_conf=config.nms_conf,
        nms_iou=config.nms_iou,
        nms_max_det=config.nms_max_det,
        seed=config.seed,
    )


def _teacher_cache_dir(
    config: ComponentTargetedPatchConfig,
    rows: pd.DataFrame,
    cache_path: str | Path | None,
) -> Path:
    payload = {
        "method_version": config.method_version,
        "teacher_path_steps": config.teacher_path_steps,
        "target_iou": config.target_iou,
        "candidate_min_score": config.candidate_min_score,
        "init_patch": str(Path(config.init_patch).resolve()),
        "example_ids": rows.example_id.astype(str).tolist(),
        "cache_path": str(cache_path),
    }
    return Path(config.output_dir) / "teacher_cache" / f"joint_component_{stable_hash(payload)}"


def _component_file(cache_dir: Path, example_id: str) -> Path:
    return cache_dir / f"{example_id}.npz"


def _save_teacher_component(
    path: Path,
    *,
    clean_inputs,
    maps,
    selection: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    payload: dict[str, Any] = {
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }
    for level, clean in enumerate(clean_inputs):
        subset = selection[selection.level_index.astype(int).eq(level)].reset_index(drop=True)
        indices = (
            _local_indices(clean[0].shape, subset, radius=2)
            if not subset.empty else np.asarray([], dtype=np.int64)
        )
        clean_flat = clean[0].detach().float().cpu().numpy().reshape(-1)
        component_flat = maps["joint_rowspace"][level].float().cpu().numpy().reshape(-1)
        payload[f"indices_{level}"] = indices.astype(np.int64)
        payload[f"clean_{level}"] = clean_flat[indices].astype(np.float16)
        payload[f"component_{level}"] = component_flat[indices].astype(np.float16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def build_teacher_cache(
    config: ComponentTargetedPatchConfig,
    *,
    exp,
    model,
    detect,
    rows: pd.DataFrame,
    cache_path: str | Path | None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Build image-specific signed joint-row-space targets once, then reuse them."""

    cache_dir = _teacher_cache_dir(config, rows, cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    example_cache = _cache_lookup(exp)
    closure_config = _teacher_config(config)
    manifest = []
    for row in tqdm(
        rows.itertuples(index=False),
        total=len(rows),
        desc="teacher component cache",
        unit="scene",
    ):
        example_id = str(row.example_id)
        target = _component_file(cache_dir, example_id)
        if not target.exists():
            example = example_cache[example_id]
            clean_image, teacher_image, _ = exp._images_for_example(example)
            pair = _preprocess_pair(exp, clean_image, teacher_image)
            captured = _capture_detect_inputs(model, detect, pair)
            clean_inputs = [item[0:1] for item in captured]
            teacher_inputs = [item[1:2] for item in captured]
            import torch

            # These tensors are reused as constants inside path-Jacobian graphs.
            # ``inference_mode`` tensors cannot be saved for backward by PyTorch.
            with torch.no_grad():
                clean_box, clean_cls, clean_raw = _head_branches(detect, clean_inputs)
                teacher_box, teacher_cls, teacher_raw = _head_branches(detect, teacher_inputs)
                selection = _candidate_closure(
                    detect, clean_raw, teacher_raw, row, closure_config
                )
            maps, energy, metadata = _functional_components(
                detect,
                clean_inputs,
                teacher_inputs,
                clean_box,
                clean_cls,
                teacher_box,
                teacher_cls,
                selection,
                row,
                closure_config,
            )
            metadata.update({
                "example_id": example_id,
                "analysis_group": str(row.analysis_group),
                "joint_energy_fraction": float(energy["joint_rowspace"]),
                "candidate_count": int(len(selection)),
            })
            _save_teacher_component(
                target,
                clean_inputs=clean_inputs,
                maps=maps,
                selection=selection,
                metadata=metadata,
            )
            release_accelerator_memory()
        with np.load(target, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
        manifest.append({
            "example_id": example_id,
            "component_path": str(target),
            **metadata,
        })
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return cache_dir, manifest


def _record_lookup(exp, selected: pd.DataFrame, component_manifest):
    examples = _cache_lookup(exp)
    components = {
        str(item["example_id"]): item for item in component_manifest
    }
    records = []
    for row in selected.itertuples(index=False):
        example_id = str(row.example_id)
        example = examples[example_id]
        records.append({
            "example_id": example_id,
            "path": str(example.path),
            "class_id": int(row.class_id),
            "target_box": (
                float(row.clean_target_x1),
                float(row.clean_target_y1),
                float(row.clean_target_x2),
                float(row.clean_target_y2),
            ),
            "analysis_group": str(row.analysis_group),
            "component_path": components[example_id]["component_path"],
        })
    return records


def _capture_detect_inputs_with_grad(model, detect, images):
    captured: dict[str, Any] = {}

    def hook(_module, args):
        value = args[0]
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Expected Detect input list, got {type(value)}")
        captured["levels"] = list(value)

    handle = detect.register_forward_pre_hook(hook)
    try:
        decoded = _decoded_from_model(model, images)
    finally:
        handle.remove()
    if "levels" not in captured:
        raise RuntimeError("Detect inputs were not captured.")
    return decoded, captured["levels"]


def signed_component_losses(feature_levels, records, *, target_scale: float):
    """Match the signed teacher component and measure off-direction local energy."""

    import torch
    import torch.nn.functional as functional

    scale_losses = []
    null_losses = []
    coefficients = []
    cosines = []
    for batch_index, record in enumerate(records):
        delta_parts = []
        component_parts = []
        with np.load(record["component_path"], allow_pickle=False) as data:
            for level, features in enumerate(feature_levels):
                indices_np = data[f"indices_{level}"].astype(np.int64)
                if not len(indices_np):
                    continue
                indices = torch.as_tensor(indices_np, device=features.device, dtype=torch.long)
                clean = torch.as_tensor(
                    data[f"clean_{level}"].astype(np.float32),
                    device=features.device,
                    dtype=features.dtype,
                )
                component = torch.as_tensor(
                    data[f"component_{level}"].astype(np.float32),
                    device=features.device,
                    dtype=features.dtype,
                )
                current = features[batch_index].reshape(-1)[indices]
                delta_parts.append((current - clean).float())
                component_parts.append(component.float())
        if not delta_parts:
            continue
        delta = torch.cat(delta_parts)
        component = torch.cat(component_parts)
        component_energy = torch.sum(component.square()).clamp_min(1e-12)
        coefficient = torch.sum(delta * component) / component_energy
        projection = coefficient * component
        delta_energy = torch.sum(delta.square()).clamp_min(1e-12)
        cosine = torch.sum(delta * component) / (
            torch.sqrt(delta_energy) * torch.sqrt(component_energy)
        ).clamp_min(1e-12)
        scale_losses.append(
            functional.smooth_l1_loss(
                coefficient, coefficient.new_tensor(float(target_scale))
            )
        )
        null_losses.append(torch.sum((delta - projection).square()) / delta_energy)
        coefficients.append(coefficient.detach())
        cosines.append(cosine.detach())
    if not scale_losses:
        zero = feature_levels[0].sum() * 0.0
        return zero, zero, {"component_coefficient": zero.detach(), "component_cosine": zero.detach()}
    return (
        torch.stack(scale_losses).mean(),
        torch.stack(null_losses).mean(),
        {
            "component_coefficient": torch.stack(coefficients).mean(),
            "component_cosine": torch.stack(cosines).mean(),
        },
    )


def _loss_weights(config: ComponentTargetedPatchConfig) -> tuple[float, float, float]:
    if config.objective == "dynamic":
        return float(config.dynamic_weight), 0.0, 0.0
    if config.objective == "component":
        return 0.0, float(config.component_weight), 0.0
    if config.objective == "hybrid":
        return float(config.dynamic_weight), float(config.component_weight), 0.0
    if config.objective == "hybrid_null":
        return (
            float(config.dynamic_weight),
            float(config.component_weight),
            float(config.null_weight),
        )
    raise ValueError(
        "objective must be one of: dynamic, component, hybrid, hybrid_null"
    )


def evaluate_component_alignment(exp, model, detect, records, patch, config) -> dict[str, float]:
    """Measure signed teacher alignment on a held-out split."""

    import torch

    parameter = next(model.parameters())
    rows = []
    for start in range(0, len(records), int(config.batch_size)):
        chunk = records[start:start + int(config.batch_size)]
        images, _targets, _classes = _load_batch(
            exp, chunk, device=parameter.device, dtype=parameter.dtype
        )
        with torch.no_grad():
            _decoded, levels = _capture_detect_inputs_with_grad(
                model, detect, overlay_patch(images, patch, config.patch_xy)
            )
            component_loss, null_loss, diagnostics = signed_component_losses(
                levels, chunk, target_scale=config.component_scale
            )
        rows.append({
            "n": len(chunk),
            "component_loss": float(component_loss.cpu()),
            "null_loss": float(null_loss.cpu()),
            "component_coefficient": float(
                diagnostics["component_coefficient"].cpu()
            ),
            "component_cosine": float(diagnostics["component_cosine"].cpu()),
        })
    denominator = max(sum(row["n"] for row in rows), 1)
    return {
        key: sum(row["n"] * row[key] for row in rows) / denominator
        for key in (
            "component_loss",
            "null_loss",
            "component_coefficient",
            "component_cosine",
        )
    }


def run_component_targeted_patch(
    config: ComponentTargetedPatchConfig | None = None,
) -> Path:
    import torch

    config = config or ComponentTargetedPatchConfig()
    started = time.time()
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
    selected, _ = _load_inputs(
        Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None
    )
    train_rows, eval_rows = _balanced_train_eval(selected, config)
    teacher_rows = pd.concat([train_rows, eval_rows], ignore_index=True)
    exp, attack_cache_path = load_experiment(
        prefer_device=config.device, require_device=config.require_device
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameter = next(model.parameters())
    teacher_cache_dir, component_manifest = build_teacher_cache(
        config,
        exp=exp,
        model=model,
        detect=detect,
        rows=teacher_rows,
        cache_path=attack_cache_path,
    )
    train = _record_lookup(exp, train_rows, component_manifest)
    evaluation = _record_lookup(exp, eval_rows, component_manifest)
    payload = {
        **asdict(config),
        "attack_cache_path": str(attack_cache_path),
        "teacher_cache_dir": str(teacher_cache_dir),
        "train_ids": [item["example_id"] for item in train],
        "eval_ids": [item["example_id"] for item in evaluation],
    }
    run_dir = Path(config.output_dir) / f"component_patch_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logits = _initial_patch(config, device=parameter.device, dtype=parameter.dtype)
    initial_patch = torch.sigmoid(logits).detach().clone()
    optimizer = torch.optim.Adam([logits], lr=float(config.learning_rate))
    dynamic_weight, component_weight, null_weight = _loss_weights(config)
    history = []
    for epoch in range(int(config.epochs)):
        order = np.random.default_rng(int(config.seed) + epoch).permutation(len(train))
        for batch_index, start in enumerate(range(0, len(order), int(config.batch_size))):
            indices = order[start:start + int(config.batch_size)]
            chunk = [train[int(index)] for index in indices]
            images, targets, classes = _load_batch(
                exp, chunk, device=parameter.device, dtype=parameter.dtype
            )
            patch = torch.sigmoid(logits)
            patched_images = overlay_patch(images, patch, config.patch_xy)
            decoded, feature_levels = _capture_detect_inputs_with_grad(
                model, detect, patched_images
            )
            dynamic_loss, dynamic_diagnostics = dynamic_score_geometry_loss(
                decoded,
                targets,
                classes,
                match_iou=config.target_iou,
                iou_temperature=config.dynamic_iou_temperature,
                iou_weight=config.dynamic_iou_weight,
                smoothmax_temperature=config.smoothmax_temperature,
            )
            component_loss, null_loss, component_diagnostics = signed_component_losses(
                feature_levels, chunk, target_scale=config.component_scale
            )
            tv = total_variation(patch)
            loss = (
                dynamic_weight * dynamic_loss
                + component_weight * component_loss
                + null_weight * null_loss
                + float(config.tv_weight) * tv
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            history.append({
                "epoch": epoch + 1,
                "batch": batch_index + 1,
                "loss": float(loss.detach().cpu()),
                "dynamic_loss": float(dynamic_loss.detach().cpu()),
                "component_loss": float(component_loss.detach().cpu()),
                "null_loss": float(null_loss.detach().cpu()),
                "tv": float(tv.detach().cpu()),
                **{
                    key: float(value.cpu())
                    for key, value in {
                        **dynamic_diagnostics,
                        **component_diagnostics,
                    }.items()
                },
            })
        _save_patch(run_dir / "patch_latest.png", torch.sigmoid(logits))
    final_patch = torch.sigmoid(logits).detach()
    _save_patch(run_dir / "patch.png", final_patch)
    baseline_alignment = evaluate_component_alignment(
        exp, model, detect, evaluation, initial_patch, config
    )
    final_alignment = evaluate_component_alignment(
        exp, model, detect, evaluation, final_patch, config
    )
    baseline_rows = evaluate_patch(exp, model, evaluation, initial_patch, config)
    evaluation_rows = evaluate_patch(exp, model, evaluation, final_patch, config)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(run_dir / "baseline_evaluation.csv", index=False)
    pd.DataFrame(evaluation_rows).to_csv(run_dir / "evaluation.csv", index=False)
    eligible_paths = {
        row["path"] for row in evaluation_rows if row["clean_target_visible"]
    }
    baseline_hidden = sum(
        row["target_hidden"] for row in baseline_rows if row["path"] in eligible_paths
    )
    hidden = sum(
        row["target_hidden"] for row in evaluation_rows if row["path"] in eligible_paths
    )
    denominator = len(eligible_paths)
    baseline_rate = baseline_hidden / max(denominator, 1)
    hiding_rate = hidden / max(denominator, 1)
    teacher_energy = [
        float(item["joint_energy_fraction"]) for item in component_manifest
        if np.isfinite(float(item["joint_energy_fraction"]))
    ]
    summary = {
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "objective": config.objective,
        "train_examples": len(train),
        "eval_examples": len(evaluation),
        "clean_visible_eval_examples": denominator,
        "baseline_hidden_eval_examples": baseline_hidden,
        "baseline_target_hiding_rate": baseline_rate,
        "hidden_eval_examples": hidden,
        "target_hiding_rate": hiding_rate,
        "absolute_hiding_rate_gain": hiding_rate - baseline_rate,
        "mean_teacher_joint_energy_fraction": float(np.mean(teacher_energy)),
        "baseline_eval_component_coefficient": baseline_alignment["component_coefficient"],
        "baseline_eval_component_cosine": baseline_alignment["component_cosine"],
        "final_eval_component_coefficient": final_alignment["component_coefficient"],
        "final_eval_component_cosine": final_alignment["component_cosine"],
        "final_eval_component_loss": final_alignment["component_loss"],
        "final_eval_null_loss": final_alignment["null_loss"],
        "teacher_cache_dir": str(teacher_cache_dir),
        "patch_path": str(run_dir / "patch.png"),
        "config": asdict(config),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "analysis_digest.md").write_text(
        "# Component-targeted defensive challenge patch\n\n"
        f"- objective: {config.objective}\n"
        f"- train/eval: {len(train)}/{len(evaluation)} (disjoint)\n"
        f"- mean teacher joint energy fraction: "
        f"{summary['mean_teacher_joint_energy_fraction']:.5f}\n"
        f"- eval signed component coefficient: "
        f"{summary['baseline_eval_component_coefficient']:.3f} -> "
        f"{summary['final_eval_component_coefficient']:.3f}\n"
        f"- eval component cosine: "
        f"{summary['baseline_eval_component_cosine']:.3f} -> "
        f"{summary['final_eval_component_cosine']:.3f}\n"
        f"- baseline hiding: {baseline_hidden}/{max(denominator, 1)} "
        f"({baseline_rate:.3f})\n"
        f"- final hiding: {hidden}/{max(denominator, 1)} ({hiding_rate:.3f})\n"
        f"- absolute gain: {hiding_rate - baseline_rate:+.3f}\n"
        f"- elapsed: {summary['elapsed_seconds']:.1f} s\n",
        encoding="utf-8",
    )
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a defensive challenge patch against the causal joint component."
    )
    parser.add_argument(
        "--objective",
        choices=("dynamic", "component", "hybrid", "hybrid_null"),
        default="hybrid",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-examples-per-group", type=int, default=4)
    parser.add_argument("--eval-examples-per-group", type=int, default=4)
    parser.add_argument("--teacher-path-steps", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--component-scale", type=float, default=1.25)
    parser.add_argument("--component-weight", type=float, default=5.0)
    parser.add_argument("--null-weight", type=float, default=0.01)
    parser.add_argument("--dynamic-weight", type=float, default=1.0)
    parser.add_argument("--init-patch", default=str(REPO_ROOT / "data" / "patch.png"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.train_examples_per_group = 1
        args.eval_examples_per_group = 1
        args.teacher_path_steps = 1
        args.epochs = 1
    config = ComponentTargetedPatchConfig(
        output_dir=args.output_dir,
        device=args.device,
        train_examples_per_group=args.train_examples_per_group,
        eval_examples_per_group=args.eval_examples_per_group,
        teacher_path_steps=args.teacher_path_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        objective=args.objective,
        component_scale=args.component_scale,
        component_weight=args.component_weight,
        null_weight=args.null_weight,
        dynamic_weight=args.dynamic_weight,
        init_patch=args.init_patch,
    )
    print(run_component_targeted_patch(config))


if __name__ == "__main__":
    main()
