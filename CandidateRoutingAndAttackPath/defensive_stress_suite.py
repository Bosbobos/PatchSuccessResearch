from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .candidate_routing import _box_iou, _xywh_to_xyxy
from .causal_repair import _load_inputs
from .common import REPO_ROOT, load_experiment, stable_hash
from .component_targeted_patch import (
    ComponentTargetedPatchConfig,
    _balanced_train_eval,
    _capture_detect_inputs_with_grad,
    _record_lookup,
    build_teacher_cache,
    evaluate_component_alignment,
)
from .followup_common import ATTACK_PATH_DB, MANIFEST_CSV, TRACE_DB
from .mechanism_aware_patch import (
    _decoded_from_model,
    _initial_patch,
    _load_batch,
    _save_patch,
    evaluate_patch,
    overlay_patch,
    total_variation,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "CandidateRoutingAndAttackPath" / "stress_suite_outputs"

MECHANISMS = (
    "reserve_score",
    "geometry_only",
    "nms_only",
    "candidate_handoff",
    "cross_scale",
    "nonlinear_residual",
    "component_minimal",
    "tail_compact",
    "tail_diffuse",
    "dormant_geometry",
    "clean_signature_matched",
    "naturalistic",
)


@dataclass(slots=True)
class DefensiveStressSuiteConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "cpu"
    require_device: bool = False
    train_examples_per_group: int = 4
    eval_examples_per_group: int = 4
    teacher_path_steps: int = 3
    mechanisms: tuple[str, ...] = MECHANISMS
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 0.05
    patch_size: int = 160
    patch_xy: tuple[int, int] = (0, 0)
    target_iou: float = 0.50
    detection_conf: float = 0.25
    candidate_min_score: float = 0.01
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    smoothmax_temperature: float = 0.35
    iou_temperature: float = 0.07
    geometry_weight: float = 4.0
    component_scale: float = 1.0
    tv_weight: float = 0.002
    naturalistic_tv_weight: float = 0.05
    naturalistic_frequency_weight: float = 0.10
    init_patch: str = str(REPO_ROOT / "data" / "patch.png")
    seed: int = 613
    method_version: int = 1

    @property
    def match_iou(self) -> float:
        return float(self.target_iou)


def _smooth_max(values, temperature: float):
    import torch

    if not int(values.numel()):
        return values.new_tensor(-20.0)
    temperature = float(temperature)
    return temperature * (
        torch.logsumexp(values / temperature, dim=0) - math.log(int(values.numel()))
    )


def _target_state(decoded, clean_decoded, target_box, class_id: int, config):
    import torch

    boxes = _xywh_to_xyxy(decoded[:4].transpose(0, 1))
    clean_boxes = _xywh_to_xyxy(clean_decoded[:4].transpose(0, 1))
    ious = _box_iou(boxes, target_box.reshape(1, 4)).reshape(-1)
    clean_ious = _box_iou(clean_boxes, target_box.reshape(1, 4)).reshape(-1)
    scores = decoded[4 + int(class_id)].clamp(1e-6, 1 - 1e-6)
    clean_scores = clean_decoded[4 + int(class_id)].clamp(1e-6, 1 - 1e-6)
    logits = torch.logit(scores)
    clean_logits = torch.logit(clean_scores)
    target_mask = clean_ious >= float(config.target_iou)
    if not bool(target_mask.any()):
        target_mask[torch.argmax(clean_scores)] = True
    membership = torch.sigmoid(
        (ious - float(config.target_iou)) / float(config.iou_temperature)
    )
    risk = logits + float(config.geometry_weight) * torch.log(membership.clamp_min(1e-6))
    target_indices = torch.nonzero(target_mask, as_tuple=False).reshape(-1)
    dynamic = _smooth_max(risk[target_indices], config.smoothmax_temperature)
    return {
        "boxes": boxes,
        "ious": ious,
        "scores": scores,
        "logits": logits,
        "clean_logits": clean_logits,
        "target_mask": target_mask,
        "target_indices": target_indices,
        "dynamic": dynamic,
    }


def _component_state(feature_levels, batch_index: int, record: dict[str, Any]):
    import torch

    delta_parts = []
    component_parts = []
    target_cosine = 0.0
    with np.load(record["component_path"], allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        local_fraction = max(float(metadata.get("local_energy_fraction", 0.0)), 1e-12)
        joint_fraction = max(float(metadata.get("joint_energy_fraction", 0.0)), 0.0)
        target_cosine = math.sqrt(min(joint_fraction / local_fraction, 1.0))
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
            delta_parts.append((features[batch_index].reshape(-1)[indices] - clean).float())
            component_parts.append(component.float())
    delta = torch.cat(delta_parts)
    component = torch.cat(component_parts)
    component_energy = component.square().sum().clamp_min(1e-12)
    delta_energy = delta.square().sum().clamp_min(1e-12)
    coefficient = (delta * component).sum() / component_energy
    cosine = (delta * component).sum() / (
        torch.sqrt(delta_energy) * torch.sqrt(component_energy)
    ).clamp_min(1e-12)
    projection = coefficient * component
    null_fraction = (delta - projection).square().sum() / delta_energy
    contributions = (delta * component).abs()
    probabilities = contributions / contributions.sum().clamp_min(1e-12)
    nonzero = probabilities > 0
    entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    entropy = entropy / math.log(max(int(probabilities.numel()), 2))
    return {
        "coefficient": coefficient,
        "cosine": cosine,
        "null_fraction": null_fraction,
        "contribution_entropy": entropy,
        "target_cosine": cosine.new_tensor(target_cosine),
    }


def _level_ranges(feature_levels):
    ranges = []
    offset = 0
    for level in feature_levels:
        count = int(level.shape[-2] * level.shape[-1])
        ranges.append((offset, offset + count))
        offset += count
    return ranges


def mechanism_surrogate_loss(
    mechanism: str,
    decoded,
    clean_decoded,
    feature_levels,
    records,
    target_boxes,
    class_ids,
    config: DefensiveStressSuiteConfig,
):
    """Differentiable, mechanism-isolating defensive stress objectives."""

    import torch
    import torch.nn.functional as functional

    losses = []
    diagnostics: dict[str, list] = {}
    level_ranges = _level_ranges(feature_levels)
    threshold_logit = float(torch.logit(torch.tensor(config.detection_conf)))
    for index, record in enumerate(records):
        state = _target_state(
            decoded[index], clean_decoded[index], target_boxes[index],
            int(class_ids[index]), config,
        )
        component = _component_state(feature_levels, index, record)
        dynamic = state["dynamic"]
        target_indices = state["target_indices"]
        target_logits = state["logits"][target_indices]
        clean_target_logits = state["clean_logits"][target_indices]
        target_ious = state["ious"][target_indices]
        score_preservation = functional.mse_loss(target_logits, clean_target_logits)
        if mechanism == "reserve_score":
            loss = dynamic
        elif mechanism == "geometry_only":
            geometry = _smooth_max(target_ious, config.smoothmax_temperature)
            loss = geometry + 2.0 * score_preservation
        elif mechanism == "nms_only":
            chosen = target_indices[torch.argmax(target_logits)]
            target_logit = state["logits"][chosen]
            overlaps = _box_iou(
                state["boxes"], state["boxes"][chosen:chosen + 1]
            ).reshape(-1)
            competitor_mask = ~state["target_mask"]
            competitor_risk = (
                state["logits"]
                + 4.0 * torch.log(
                    torch.sigmoid(
                        (overlaps - float(config.nms_iou)) / config.iou_temperature
                    ).clamp_min(1e-6)
                )
            )[competitor_mask]
            competitor = _smooth_max(competitor_risk, config.smoothmax_temperature)
            keep_pre_nms = functional.relu(
                target_logit.new_tensor(threshold_logit) - target_logit
            )
            loss = functional.softplus(target_logit - competitor + 0.2) + 2.0 * keep_pre_nms
        elif mechanism == "candidate_handoff":
            probabilities = torch.softmax(target_logits, dim=0)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            entropy = entropy / math.log(max(int(probabilities.numel()), 2))
            loss = dynamic - 0.5 * entropy
        elif mechanism == "cross_scale":
            level_risks = []
            for start, end in level_ranges:
                local = target_indices[(target_indices >= start) & (target_indices < end)]
                if len(local):
                    level_risks.append(
                        _smooth_max(
                            state["logits"][local],
                            config.smoothmax_temperature,
                        )
                    )
            dispersion = (
                torch.stack(level_risks).std(unbiased=False)
                if len(level_risks) > 1 else dynamic.new_tensor(0.0)
            )
            loss = dynamic - 0.5 * dispersion
        elif mechanism == "nonlinear_residual":
            loss = dynamic + 2.0 * functional.smooth_l1_loss(
                component["coefficient"], component["coefficient"].new_tensor(1.0)
            )
        elif mechanism == "component_minimal":
            loss = dynamic + 0.5 * component["coefficient"].abs()
        elif mechanism == "tail_compact":
            loss = dynamic + 0.25 * component["contribution_entropy"]
        elif mechanism == "tail_diffuse":
            loss = dynamic - 0.25 * component["contribution_entropy"]
        elif mechanism == "dormant_geometry":
            max_iou = target_ious.max()
            geometry_preservation = functional.relu(max_iou.new_tensor(0.75) - max_iou)
            loss = _smooth_max(target_logits, config.smoothmax_temperature) + 5.0 * geometry_preservation.square()
        elif mechanism == "clean_signature_matched":
            signature = (
                functional.smooth_l1_loss(
                    component["coefficient"], component["coefficient"].new_tensor(1.0)
                )
                + functional.smooth_l1_loss(
                    component["cosine"], component["target_cosine"]
                )
            )
            loss = dynamic + 2.0 * signature
        elif mechanism == "naturalistic":
            loss = dynamic
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
        losses.append(loss)
        values = {
            "dynamic": dynamic,
            "component_coefficient": component["coefficient"],
            "component_cosine": component["cosine"],
            "null_fraction": component["null_fraction"],
            "contribution_entropy": component["contribution_entropy"],
            "max_target_iou": target_ious.max(),
            "max_target_score": state["scores"][target_indices].max(),
        }
        for key, value in values.items():
            diagnostics.setdefault(key, []).append(value.detach())
    return torch.stack(losses).mean(), {
        key: torch.stack(values).mean() for key, values in diagnostics.items()
    }


def _high_frequency_energy(patch):
    import torch.nn.functional as functional

    smooth = functional.avg_pool2d(patch, kernel_size=5, stride=1, padding=2)
    return (patch - smooth).square().mean()


def _appearance_metrics(patch) -> dict[str, float]:
    return {
        "patch_tv": float(total_variation(patch).detach().cpu()),
        "patch_high_frequency_energy": float(
            _high_frequency_energy(patch).detach().cpu()
        ),
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for item in frame[columns].itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(
                f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
                for value in item
            )
            + " |"
        )
    return "\n".join([header, separator, *rows])


def _reset_detect_inference_cache(detect) -> None:
    """Prevent inference-mode anchor tensors leaking into the next grad run."""

    if hasattr(detect, "shape"):
        detect.shape = None


def _train_one(
    mechanism,
    *,
    config,
    exp,
    model,
    detect,
    train,
    evaluation,
    parameter,
    suite_dir,
):
    import torch

    logits = _initial_patch(config, device=parameter.device, dtype=parameter.dtype)
    initial_patch = torch.sigmoid(logits).detach().clone()
    optimizer = torch.optim.Adam([logits], lr=float(config.learning_rate))
    history = []
    for epoch in range(int(config.epochs)):
        order = np.random.default_rng(
            int(config.seed) + epoch
        ).permutation(len(train))
        for batch_index, start in enumerate(range(0, len(order), int(config.batch_size))):
            indices = order[start:start + int(config.batch_size)]
            chunk = [train[int(item)] for item in indices]
            images, targets, classes = _load_batch(
                exp, chunk, device=parameter.device, dtype=parameter.dtype
            )
            with torch.no_grad():
                clean_decoded = _decoded_from_model(model, images)
            patch = torch.sigmoid(logits)
            decoded, feature_levels = _capture_detect_inputs_with_grad(
                model, detect, overlay_patch(images, patch, config.patch_xy)
            )
            surrogate, diagnostics = mechanism_surrogate_loss(
                mechanism,
                decoded,
                clean_decoded,
                feature_levels,
                chunk,
                targets,
                classes,
                config,
            )
            tv_weight = (
                config.naturalistic_tv_weight
                if mechanism == "naturalistic" else config.tv_weight
            )
            high_frequency = _high_frequency_energy(patch)
            frequency_weight = (
                config.naturalistic_frequency_weight
                if mechanism == "naturalistic" else 0.0
            )
            tv = total_variation(patch)
            loss = surrogate + float(tv_weight) * tv + float(frequency_weight) * high_frequency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            history.append({
                "epoch": epoch + 1,
                "batch": batch_index + 1,
                "loss": float(loss.detach().cpu()),
                "surrogate_loss": float(surrogate.detach().cpu()),
                "tv": float(tv.detach().cpu()),
                "high_frequency_energy": float(high_frequency.detach().cpu()),
                **{key: float(value.cpu()) for key, value in diagnostics.items()},
            })
    patch = torch.sigmoid(logits).detach()
    mechanism_dir = suite_dir / mechanism
    mechanism_dir.mkdir(parents=True, exist_ok=True)
    _save_patch(mechanism_dir / "patch.png", patch)
    pd.DataFrame(history).to_csv(mechanism_dir / "training_history.csv", index=False)
    alignment = evaluate_component_alignment(
        exp, model, detect, evaluation, patch, config
    )
    evaluation_rows = evaluate_patch(exp, model, evaluation, patch, config)
    _reset_detect_inference_cache(detect)
    pd.DataFrame(evaluation_rows).to_csv(mechanism_dir / "evaluation.csv", index=False)
    return initial_patch, patch, history, alignment, evaluation_rows


def run_defensive_stress_suite(
    config: DefensiveStressSuiteConfig | None = None,
) -> Path:
    import torch

    config = config or DefensiveStressSuiteConfig()
    unknown = sorted(set(config.mechanisms).difference(MECHANISMS))
    if unknown:
        raise ValueError(f"Unknown mechanisms: {unknown}")
    started = time.time()
    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))
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
        seed=config.seed,
        method_version=2,
    )
    train_rows, eval_rows = _balanced_train_eval(selected, split_config)
    teacher_rows = pd.concat([train_rows, eval_rows], ignore_index=True)
    exp, attack_cache_path = load_experiment(
        prefer_device=config.device, require_device=config.require_device
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    for model_parameter in model.parameters():
        model_parameter.requires_grad_(False)
    parameter = next(model.parameters())
    teacher_cache_dir, component_manifest = build_teacher_cache(
        split_config,
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
        "teacher_cache_dir": str(teacher_cache_dir),
        "train_ids": [item["example_id"] for item in train],
        "eval_ids": [item["example_id"] for item in evaluation],
    }
    suite_dir = Path(config.output_dir) / f"stress_suite_{stable_hash(payload)}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    initial_logits = _initial_patch(
        config, device=parameter.device, dtype=parameter.dtype
    )
    initial_patch = torch.sigmoid(initial_logits).detach()
    baseline_evaluation = evaluate_patch(
        exp, model, evaluation, initial_patch, config
    )
    _reset_detect_inference_cache(detect)
    baseline_alignment = evaluate_component_alignment(
        exp, model, detect, evaluation, initial_patch, config
    )
    eligible_paths = {
        row["path"] for row in baseline_evaluation if row["clean_target_visible"]
    }
    baseline_failures = sum(
        row["target_hidden"]
        for row in baseline_evaluation
        if row["path"] in eligible_paths
    )
    denominator = len(eligible_paths)
    comparison = []
    for mechanism in config.mechanisms:
        _initial, patch, history, alignment, evaluation_rows = _train_one(
            mechanism,
            config=config,
            exp=exp,
            model=model,
            detect=detect,
            train=train,
            evaluation=evaluation,
            parameter=parameter,
            suite_dir=suite_dir,
        )
        failures = sum(
            row["target_hidden"]
            for row in evaluation_rows
            if row["path"] in eligible_paths
        )
        appearance = _appearance_metrics(patch)
        comparison.append({
            "mechanism": mechanism,
            "n_eval": denominator,
            "baseline_failure_rate": baseline_failures / max(denominator, 1),
            "failure_reproduction_rate": failures / max(denominator, 1),
            "absolute_failure_gain": (
                failures - baseline_failures
            ) / max(denominator, 1),
            "component_coefficient_before": baseline_alignment["component_coefficient"],
            "component_coefficient_after": alignment["component_coefficient"],
            "component_cosine_before": baseline_alignment["component_cosine"],
            "component_cosine_after": alignment["component_cosine"],
            "final_dynamic_surrogate": float(history[-1]["dynamic"]),
            "final_null_fraction": float(history[-1]["null_fraction"]),
            "final_contribution_entropy": float(
                history[-1]["contribution_entropy"]
            ),
            "final_max_target_iou": float(history[-1]["max_target_iou"]),
            "final_max_target_score": float(history[-1]["max_target_score"]),
            **appearance,
        })
    comparison_frame = pd.DataFrame(comparison).sort_values(
        ["failure_reproduction_rate", "mechanism"], ascending=[False, True]
    )
    comparison_frame.to_csv(suite_dir / "mechanism_comparison.csv", index=False)
    summary = {
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "n_train": len(train),
        "n_eval": len(evaluation),
        "clean_visible_eval": denominator,
        "baseline_failure_rate": baseline_failures / max(denominator, 1),
        "teacher_cache_dir": str(teacher_cache_dir),
        "best_failure_mechanism": str(comparison_frame.iloc[0].mechanism),
        "best_failure_rate": float(
            comparison_frame.iloc[0].failure_reproduction_rate
        ),
        "config": asdict(config),
    }
    (suite_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    columns = [
        "mechanism",
        "failure_reproduction_rate",
        "component_coefficient_after",
        "component_cosine_after",
        "final_max_target_iou",
        "final_max_target_score",
        "patch_tv",
    ]
    (suite_dir / "analysis_digest.md").write_text(
        "# Defensive mechanism stress suite\n\n"
        f"- train/eval: {len(train)}/{len(evaluation)} (disjoint)\n"
        f"- clean-visible eval: {denominator}\n"
        f"- baseline failure rate: {summary['baseline_failure_rate']:.3f}\n"
        f"- elapsed: {summary['elapsed_seconds']:.1f} s\n\n"
        + _markdown_table(comparison_frame, columns)
        + "\n",
        encoding="utf-8",
    )
    return suite_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a defensive, mechanism-specific patch stress suite."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-examples-per-group", type=int, default=4)
    parser.add_argument("--eval-examples-per-group", type=int, default=4)
    parser.add_argument("--teacher-path-steps", type=int, default=3)
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        choices=MECHANISMS,
        default=list(MECHANISMS),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.epochs = 1
        args.train_examples_per_group = 1
        args.eval_examples_per_group = 1
        args.teacher_path_steps = 1
    config = DefensiveStressSuiteConfig(
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_examples_per_group=args.train_examples_per_group,
        eval_examples_per_group=args.eval_examples_per_group,
        teacher_path_steps=args.teacher_path_steps,
        mechanisms=tuple(args.mechanisms),
    )
    print(run_defensive_stress_suite(config))


if __name__ == "__main__":
    main()
