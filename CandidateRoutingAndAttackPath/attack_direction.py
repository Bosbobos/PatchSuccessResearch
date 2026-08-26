from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .candidate_routing import _box_iou, _xywh_to_xyxy
from .causal_repair import _load_inputs
from .common import load_experiment, release_accelerator_memory, stable_hash
from .followup_common import (
    ATTACK_PATH_DB,
    FOLLOWUP_DIR,
    MANIFEST_CSV,
    TRACE_DB,
    balanced_subset,
    write_summary,
)
from .mechanism_followup import _decode, _evaluate_decoded, _head_branches


@dataclass(slots=True)
class AttackOracleConfig:
    output_dir: str = str(FOLLOWUP_DIR)
    device: str = "cpu"
    require_device: bool = False
    examples_per_group: int = 6
    objectives: tuple[str, ...] = ("fixed_cell", "static_clean_set", "dynamic_score_geometry")
    steps: int = 30
    checkpoints: tuple[int, ...] = (0, 5, 10, 20, 30)
    epsilon_actual_l2_fraction: float = 1.0
    step_fraction: float = 0.075
    smoothmax_temperature: float = 0.35
    dynamic_iou_temperature: float = 0.07
    dynamic_iou_weight: float = 4.0
    max_candidates: int = 32
    candidate_min_score: float = 0.01
    detection_conf: float = 0.25
    match_iou: float = 0.50
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    seed: int = 131
    method_version: int = 1


def _flat_class_logits(raw, class_channel: int):
    import torch

    return torch.cat([level[:, class_channel].reshape(level.shape[0], -1) for level in raw], dim=1)


def _clean_candidate_flats(detect, clean_inputs, row, config: AttackOracleConfig):
    import torch

    class_id = int(row.class_id)
    with torch.no_grad():
        _box, _cls, raw = _head_branches(detect, clean_inputs)
        decoded = _decode(detect, raw)
        boxes = _xywh_to_xyxy(decoded[0, :4, :].transpose(0, 1))
        target = torch.as_tensor(
            [[row.clean_target_x1, row.clean_target_y1, row.clean_target_x2, row.clean_target_y2]],
            device=boxes.device, dtype=torch.float32,
        )
        ious = _box_iou(boxes, target).reshape(-1)
        scores = decoded[0, 4 + class_id]
        valid = torch.nonzero(
            (ious >= config.match_iou) & (scores >= config.candidate_min_score), as_tuple=False
        ).reshape(-1)
        candidates = valid[torch.argsort(scores[valid], descending=True)].detach().cpu().tolist()
    target_flat = int(row.clean_target_flat)
    if target_flat not in candidates:
        candidates.append(target_flat)
    return sorted(set(int(value) for value in candidates), key=lambda value: candidates.index(value))[: config.max_candidates]


def _objective_value(detect, raw, decoded, row, candidates, objective: str, config: AttackOracleConfig):
    import torch

    class_id = int(row.class_id)
    class_channel = 4 * int(detect.reg_max) + class_id
    if objective == "fixed_cell":
        return raw[int(row.clean_target_level)][0, class_channel, int(row.clean_target_y), int(row.clean_target_x)]
    logits = _flat_class_logits(raw, class_channel)[0]
    if objective == "static_clean_set":
        values = logits[torch.as_tensor(candidates, device=logits.device, dtype=torch.long)]
        temperature = float(config.smoothmax_temperature)
        return temperature * torch.logsumexp(values / temperature, dim=0)
    if objective == "dynamic_score_geometry":
        boxes = _xywh_to_xyxy(decoded[0, :4, :].transpose(0, 1))
        target = torch.as_tensor(
            [[row.clean_target_x1, row.clean_target_y1, row.clean_target_x2, row.clean_target_y2]],
            device=boxes.device, dtype=torch.float32,
        )
        ious = _box_iou(boxes, target).reshape(-1)
        membership = torch.sigmoid((ious - float(config.match_iou)) / float(config.dynamic_iou_temperature))
        risk = logits + float(config.dynamic_iou_weight) * torch.log(membership.clamp_min(1e-6))
        temperature = float(config.smoothmax_temperature)
        return temperature * torch.logsumexp(risk / temperature, dim=0)
    raise ValueError(f"Unknown objective: {objective}")


def _raw_for_inputs(detect, inputs):
    _box, _cls, raw = _head_branches(detect, inputs)
    return raw, _decode(detect, raw)


def _cosine(a, b) -> float:
    import torch

    numerator = torch.sum(a.float() * b.float())
    denominator = torch.linalg.vector_norm(a.float()) * torch.linalg.vector_norm(b.float())
    return float((numerator / denominator.clamp_min(1e-12)).detach().cpu())


def _optimize_one(detect, clean_inputs, patched_inputs, row, candidates, objective, config):
    import torch

    target_level = int(row.clean_target_level)
    actual_delta = (patched_inputs[target_level] - clean_inputs[target_level]).detach()
    actual_l2 = float(torch.linalg.vector_norm(actual_delta.float()).cpu())
    epsilon = max(actual_l2 * float(config.epsilon_actual_l2_fraction), 1e-6)
    step_size = epsilon * float(config.step_fraction)
    perturb = torch.zeros_like(clean_inputs[target_level], requires_grad=True)
    records = []
    checkpoints = set(int(value) for value in config.checkpoints)
    for step in range(int(config.steps) + 1):
        inputs = [item.detach().clone() for item in clean_inputs]
        inputs[target_level] = clean_inputs[target_level] + perturb
        raw, decoded = _raw_for_inputs(detect, inputs)
        objective_value = _objective_value(detect, raw, decoded, row, candidates, objective, config)
        if step in checkpoints:
            with torch.no_grad():
                endpoint = _evaluate_decoded(detect, [item.detach() for item in raw], row, config)
                current_l2 = float(torch.linalg.vector_norm(perturb.detach().float()).cpu())
                records.append({
                    "step": step, "objective": objective, "objective_value": float(objective_value.detach().cpu()),
                    "perturb_l2": current_l2, "actual_patch_delta_l2": actual_l2,
                    "relative_actual_l2": current_l2 / max(actual_l2, 1e-12),
                    "cosine_with_actual_patch_delta": _cosine(perturb.detach(), actual_delta) if current_l2 > 0 else 0.0,
                    **endpoint,
                })
        if step == int(config.steps):
            break
        gradient = torch.autograd.grad(objective_value, perturb, only_inputs=True)[0]
        with torch.no_grad():
            direction = gradient / torch.linalg.vector_norm(gradient.float()).clamp_min(1e-12)
            perturb -= step_size * direction.to(dtype=perturb.dtype)
            norm = torch.linalg.vector_norm(perturb.float())
            if float(norm.cpu()) > epsilon:
                perturb *= (epsilon / norm).to(dtype=perturb.dtype)
        perturb = perturb.detach().requires_grad_(True)
    return records


def _cache_lookup(exp):
    return {
        stable_hash({"path": str(item.path), "drop": float(item.drop), "success": bool(item.success)}): item
        for item in exp.get_cache().examples
    }


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby(["objective", "step"], as_index=False).agg(
        n=("example_id", "nunique"), target_hidden_rate=("target_hidden", "mean"),
        mean_relative_actual_l2=("relative_actual_l2", "mean"),
        mean_cosine_actual_delta=("cosine_with_actual_patch_delta", "mean"),
        mean_fixed_logit=("fixed_target_logit", "mean"), mean_fixed_iou=("fixed_box_iou_clean", "mean"),
        mean_post_conf=("post_target_conf", "mean"), mean_post_iou=("post_target_max_iou", "mean"),
        nms_only_hidden_rate=("nms_only_hidden", "mean"),
    )


def _first_hiding(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hidden = rows[rows.target_hidden == 1].sort_values(["example_id", "objective", "relative_actual_l2", "step"])
    first = hidden.groupby(["example_id", "analysis_group", "objective"], as_index=False).first()
    expected = rows[["example_id", "analysis_group", "objective"]].drop_duplicates()
    first = expected.merge(
        first[["example_id", "analysis_group", "objective", "step", "relative_actual_l2", "cosine_with_actual_patch_delta"]],
        on=["example_id", "analysis_group", "objective"], how="left", validate="one_to_one",
    )
    first["ever_hidden"] = first.step.notna().astype(int)
    summary = first.groupby("objective", as_index=False).agg(
        n=("example_id", "nunique"), hiding_success_rate=("ever_hidden", "mean"),
        mean_first_hiding_step=("step", "mean"), median_first_hiding_step=("step", "median"),
        mean_first_hiding_norm=("relative_actual_l2", "mean"),
        median_first_hiding_norm=("relative_actual_l2", "median"),
        mean_cosine_actual_delta=("cosine_with_actual_patch_delta", "mean"),
    )
    return first, summary


def run_attack_oracle(config: AttackOracleConfig | None = None) -> Path:
    config = config or AttackOracleConfig()
    started = time.time()
    selected, _top = _load_inputs(Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None)
    selected = balanced_subset(selected, config.examples_per_group, seed=config.seed)
    exp, cache_path = load_experiment(
        prefer_device=config.device,
        require_device=bool(config.require_device),
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    cache = _cache_lookup(exp)
    output = []
    for row in selected.itertuples(index=False):
        example = cache[str(row.example_id)]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        captured = _capture_detect_inputs(model, detect, pair)
        clean_inputs = [item[0:1].detach() for item in captured]
        patched_inputs = [item[1:2].detach() for item in captured]
        candidates = _clean_candidate_flats(detect, clean_inputs, row, config)
        for objective in config.objectives:
            records = _optimize_one(detect, clean_inputs, patched_inputs, row, candidates, objective, config)
            for record in records:
                record.update({
                    "example_id": str(row.example_id), "analysis_group": str(row.analysis_group),
                    "clean_candidate_count": len(candidates),
                })
                output.append(record)
        release_accelerator_memory()
    rows = pd.DataFrame(output)
    summary = _summarize(rows)
    first_hiding, first_hiding_summary = _first_hiding(rows)
    payload = {**asdict(config), "example_ids": selected.example_id.tolist()}
    run_dir = Path(config.output_dir) / f"attack_oracle_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(run_dir / "attack_oracle_rows.csv", index=False)
    summary.to_csv(run_dir / "attack_oracle_summary.csv", index=False)
    first_hiding.to_csv(run_dir / "attack_oracle_first_hiding.csv", index=False)
    first_hiding_summary.to_csv(run_dir / "attack_oracle_first_hiding_summary.csv", index=False)
    elapsed = time.time() - started
    final = summary[summary.step == int(config.steps)].set_index("objective")
    final_rates = final.target_hidden_rate.to_dict()
    final_norms = final.mean_relative_actual_l2.to_dict()
    write_summary(run_dir / "summary.json", {
        "status": "complete", "elapsed_seconds": elapsed, "n_examples": int(rows.example_id.nunique()),
        "final_hiding_rate": final_rates, "final_relative_norm": final_norms,
        "cache_path": str(cache_path), "config": asdict(config),
    })
    (run_dir / "analysis_digest.md").write_text(
        "# Activation-space attack oracle\n\n"
        f"- elapsed: {elapsed:.1f} s\n- examples: {rows.example_id.nunique()}\n"
        f"- final hiding rate: `{json.dumps(final_rates, ensure_ascii=False)}`\n"
        f"- final norm / actual patch delta norm: `{json.dumps(final_norms, ensure_ascii=False)}`\n\n"
        "This is an activation-space feasibility oracle, not a deployable pixel or physical attack.\n",
        encoding="utf-8",
    )
    return run_dir


if __name__ == "__main__":
    print(run_attack_oracle())
