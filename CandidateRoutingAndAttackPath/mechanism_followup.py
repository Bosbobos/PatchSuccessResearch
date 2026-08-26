from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .candidate_routing import (
    _box_iou,
    _flat_location,
    _level_slices,
    _match_post_nms_to_flat,
    _xywh_to_xyxy,
)
from .causal_repair import _load_inputs
from .common import load_experiment, release_accelerator_memory, stable_hash
from .followup_common import (
    ATTACK_PATH_DB,
    FOLLOWUP_DIR,
    MANIFEST_CSV,
    TRACE_DB,
    balanced_subset,
    bootstrap_ci,
    write_summary,
)


@dataclass(slots=True)
class MechanismFollowupConfig:
    output_dir: str = str(FOLLOWUP_DIR)
    device: str = "cpu"
    require_device: bool = False
    branch_examples_per_group: int | None = None
    path_examples_per_group: int = 50
    alphas: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    detection_conf: float = 0.25
    match_iou: float = 0.50
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    candidate_min_score: float = 0.01
    seed: int = 101
    method_version: int = 1


def _head_branches(detect, inputs):
    import torch

    box = [detect.cv2[index](item) for index, item in enumerate(inputs)]
    cls = [detect.cv3[index](item) for index, item in enumerate(inputs)]
    raw = [torch.cat((b, c), dim=1) for b, c in zip(box, cls, strict=True)]
    return box, cls, raw


def _raw_from_branches(box, cls):
    import torch

    return [torch.cat((b, c), dim=1) for b, c in zip(box, cls, strict=True)]


def _decode(detect, raw):
    return detect._inference([item.clone() for item in raw])


def _evaluate_decoded(detect, raw, row, config: MechanismFollowupConfig):
    import torch
    from ultralytics.utils.nms import non_max_suppression

    class_id = int(row.class_id)
    reg_max = int(detect.reg_max)
    class_channel = 4 * reg_max + class_id
    decoded = _decode(detect, raw)
    target_flat = int(row.clean_target_flat)
    target_level = int(row.clean_target_level)
    target_y = int(row.clean_target_y)
    target_x = int(row.clean_target_x)
    target_box = torch.as_tensor(
        [[row.clean_target_x1, row.clean_target_y1, row.clean_target_x2, row.clean_target_y2]],
        device=decoded.device,
        dtype=torch.float32,
    )
    boxes = _xywh_to_xyxy(decoded[0, :4, :].transpose(0, 1))
    scores = decoded[0, 4 + class_id, :]
    ious = _box_iou(boxes, target_box).reshape(-1)
    matched = torch.nonzero(ious >= float(config.match_iou), as_tuple=False).reshape(-1)
    pre_conf = float(scores[matched].max().cpu()) if len(matched) else 0.0
    pre_best_flat = int(matched[torch.argmax(scores[matched])].cpu()) if len(matched) else None
    pre_detected = int(pre_conf >= float(config.detection_conf))
    candidate_count = int(((ious >= float(config.match_iou)) & (scores >= float(config.candidate_min_score))).sum().cpu())

    nms = non_max_suppression(
        decoded.detach().clone(),
        conf_thres=float(config.nms_conf),
        iou_thres=float(config.nms_iou),
        classes=[class_id],
        max_det=int(config.nms_max_det),
        nc=int(detect.nc),
    )[0]
    post_conf = 0.0
    post_max_iou = 0.0
    matched_flat = None
    if len(nms):
        post_ious = _box_iou(nms[:, :4], target_box).reshape(-1)
        post_max_iou = float(post_ious.max().cpu())
        valid = torch.nonzero(post_ious >= float(config.match_iou), as_tuple=False).reshape(-1)
        if len(valid):
            chosen = int(valid[torch.argmax(nms[valid, 4])])
            post_conf = float(nms[chosen, 4].cpu())
            matched_flat = int(_match_post_nms_to_flat(0, nms[chosen], decoded, class_id))
    post_detected = int(post_conf >= float(config.detection_conf) and post_max_iou >= float(config.match_iou))
    fixed_box = boxes[target_flat:target_flat + 1]
    fixed_iou = float(_box_iou(fixed_box, target_box).reshape(-1)[0].cpu())
    slices = _level_slices(raw)
    matched_level = _flat_location(matched_flat, slices)[0] if matched_flat is not None else None
    return {
        "fixed_target_logit": float(raw[target_level][0, class_channel, target_y, target_x].float().cpu()),
        "fixed_target_score": float(scores[target_flat].cpu()),
        "fixed_box_iou_clean": fixed_iou,
        "pre_target_conf": pre_conf,
        "pre_target_detected": pre_detected,
        "pre_best_flat": pre_best_flat,
        "target_candidate_count": candidate_count,
        "post_target_conf": post_conf,
        "post_target_max_iou": post_max_iou,
        "post_target_detected": post_detected,
        "target_hidden": 1 - post_detected,
        "nms_only_hidden": int(pre_detected and not post_detected),
        "matched_flat": matched_flat,
        "matched_level": matched_level,
    }


def _cache_lookup(exp):
    return {
        stable_hash({"path": str(item.path), "drop": float(item.drop), "success": bool(item.success)}): item
        for item in exp.get_cache().examples
    }


def run_branch_factorial(exp, model, detect, selected, config: MechanismFollowupConfig) -> pd.DataFrame:
    import torch

    cache = _cache_lookup(exp)
    rows = []
    for row in selected.itertuples(index=False):
        example = cache[str(row.example_id)]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        captured = _capture_detect_inputs(model, detect, pair)
        clean_inputs = [item[0:1] for item in captured]
        patched_inputs = [item[1:2] for item in captured]
        with torch.inference_mode():
            clean_box, clean_cls, _ = _head_branches(detect, clean_inputs)
            patched_box, patched_cls, _ = _head_branches(detect, patched_inputs)
            variants = {
                "clean": _raw_from_branches(clean_box, clean_cls),
                "class_only_patched": _raw_from_branches(clean_box, patched_cls),
                "box_only_patched": _raw_from_branches(patched_box, clean_cls),
                "both_patched": _raw_from_branches(patched_box, patched_cls),
            }
            for condition, raw in variants.items():
                result = _evaluate_decoded(detect, raw, row, config)
                result.update({
                    "example_id": str(row.example_id),
                    "analysis_group": str(row.analysis_group),
                    "condition": condition,
                })
                rows.append(result)
        release_accelerator_memory()
    return pd.DataFrame(rows)


def _path_batch(model, detect, clean_tensor, patched_tensor, alphas):
    import torch

    alpha_tensor = torch.as_tensor(alphas, device=clean_tensor.device, dtype=clean_tensor.dtype).reshape(-1, 1, 1, 1)
    image_path = clean_tensor + alpha_tensor * (patched_tensor - clean_tensor)
    actual_inputs = _capture_detect_inputs(model, detect, image_path)
    clean_inputs = [item[0:1] for item in actual_inputs]
    patched_inputs = [item[-1:] for item in actual_inputs]
    activation_inputs = [
        clean + alpha_tensor * (patched - clean)
        for clean, patched in zip(clean_inputs, patched_inputs, strict=True)
    ]
    return actual_inputs, activation_inputs


def _evaluate_path_batch(detect, inputs, row, config, path_kind, alphas):
    import torch
    from ultralytics.utils.nms import non_max_suppression

    class_id = int(row.class_id)
    class_channel = 4 * int(detect.reg_max) + class_id
    target_level, target_y, target_x = int(row.clean_target_level), int(row.clean_target_y), int(row.clean_target_x)
    target_box = torch.as_tensor(
        [[row.clean_target_x1, row.clean_target_y1, row.clean_target_x2, row.clean_target_y2]],
        device=inputs[0].device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        _box, _cls, raw = _head_branches(detect, inputs)
        decoded = _decode(detect, raw)
        nms_rows = non_max_suppression(
            decoded.detach().clone(), conf_thres=config.nms_conf, iou_thres=config.nms_iou,
            classes=[class_id], max_det=config.nms_max_det, nc=int(detect.nc),
        )
    slices = _level_slices(raw)
    output = []
    for index, alpha in enumerate(alphas):
        boxes = _xywh_to_xyxy(decoded[index, :4, :].transpose(0, 1))
        scores = decoded[index, 4 + class_id, :]
        ious = _box_iou(boxes, target_box).reshape(-1)
        valid = torch.nonzero(ious >= config.match_iou, as_tuple=False).reshape(-1)
        best_flat = int(valid[torch.argmax(scores[valid])].cpu()) if len(valid) else None
        pre_conf = float(scores[valid].max().cpu()) if len(valid) else 0.0
        candidate_count = int(((ious >= config.match_iou) & (scores >= config.candidate_min_score)).sum().cpu())
        post_conf = post_max_iou = 0.0
        matched_flat = None
        detections = nms_rows[index]
        if len(detections):
            post_ious = _box_iou(detections[:, :4], target_box).reshape(-1)
            post_max_iou = float(post_ious.max().cpu())
            matched = torch.nonzero(post_ious >= config.match_iou, as_tuple=False).reshape(-1)
            if len(matched):
                chosen = int(matched[torch.argmax(detections[matched, 4])])
                post_conf = float(detections[chosen, 4].cpu())
                matched_flat = int(_match_post_nms_to_flat(index, detections[chosen], decoded, class_id))
        output.append({
            "example_id": str(row.example_id), "analysis_group": str(row.analysis_group),
            "path_kind": path_kind, "alpha": float(alpha),
            "fixed_target_logit": float(raw[target_level][index, class_channel, target_y, target_x].float().cpu()),
            "pre_target_conf": pre_conf, "pre_best_flat": best_flat,
            "target_candidate_count": candidate_count,
            "post_target_conf": post_conf, "post_target_max_iou": post_max_iou,
            "target_hidden": int(not (post_conf >= config.detection_conf and post_max_iou >= config.match_iou)),
            "matched_flat": matched_flat,
            "matched_level": _flat_location(matched_flat, slices)[0] if matched_flat is not None else None,
        })
    return output


def run_path_validation(exp, model, detect, selected, config: MechanismFollowupConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = _cache_lookup(exp)
    rows = []
    alphas = tuple(float(value) for value in config.alphas)
    for row in selected.itertuples(index=False):
        example = cache[str(row.example_id)]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        actual_inputs, activation_inputs = _path_batch(model, detect, pair[0:1], pair[1:2], alphas)
        rows.extend(_evaluate_path_batch(detect, actual_inputs, row, config, "image_opacity", alphas))
        rows.extend(_evaluate_path_batch(detect, activation_inputs, row, config, "activation_linear", alphas))
        release_accelerator_memory()
    detail = pd.DataFrame(rows)
    summaries = []
    for (example_id, group, path_kind), subset in detail.groupby(["example_id", "analysis_group", "path_kind"]):
        subset = subset.sort_values("alpha")
        alpha = subset.alpha.to_numpy(float)
        score = subset.fixed_target_logit.to_numpy(float)
        chord = score[0] + alpha * (score[-1] - score[0])
        scale = max(abs(score[-1] - score[0]), 1e-6)
        clean_best = subset.iloc[0].pre_best_flat
        flats = {int(value) for value in subset.pre_best_flat.dropna().tolist()}
        summaries.append({
            "example_id": example_id, "analysis_group": group, "path_kind": path_kind,
            "endpoint_logit_delta": float(score[-1] - score[0]),
            "max_abs_chord_deviation": float(np.max(np.abs(score - chord))),
            "normalized_chord_deviation": float(np.max(np.abs(score - chord)) / scale),
            "signed_chord_area": float(np.trapz(score - chord, alpha)),
            "unique_best_candidate_count": len(flats),
            "best_candidate_changed": int(any(value != clean_best for value in flats)),
            "max_candidate_count": int(subset.target_candidate_count.max()),
            "hiding_onset_alpha": float(subset.loc[subset.target_hidden == 1, "alpha"].min())
            if (subset.target_hidden == 1).any() else np.nan,
        })
    return detail, pd.DataFrame(summaries)


def _summarize_branch(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    clean = rows[rows.condition == "clean"][["example_id", "fixed_target_logit", "fixed_box_iou_clean"]].rename(
        columns={"fixed_target_logit": "clean_logit", "fixed_box_iou_clean": "clean_fixed_iou"}
    )
    enriched = rows.merge(clean, on="example_id", how="left", validate="many_to_one")
    enriched["fixed_logit_loss"] = enriched.clean_logit - enriched.fixed_target_logit
    enriched["fixed_iou_loss"] = enriched.clean_fixed_iou - enriched.fixed_box_iou_clean
    return enriched.groupby(["analysis_group", "condition"], as_index=False).agg(
        n=("example_id", "nunique"), target_hidden_rate=("target_hidden", "mean"),
        pre_hidden_rate=("pre_target_detected", lambda value: 1.0 - float(np.mean(value))),
        nms_only_hidden_rate=("nms_only_hidden", "mean"),
        mean_fixed_logit_loss=("fixed_logit_loss", "mean"),
        mean_fixed_iou_loss=("fixed_iou_loss", "mean"),
        mean_post_conf=("post_target_conf", "mean"), mean_post_max_iou=("post_target_max_iou", "mean"),
    )


def _summarize_path(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    output = []
    for (group, path_kind), subset in rows.groupby(["analysis_group", "path_kind"]):
        for metric in ("endpoint_logit_delta", "normalized_chord_deviation", "unique_best_candidate_count", "best_candidate_changed"):
            low, high = bootstrap_ci(subset[metric], seed=seed + len(output))
            output.append({
                "analysis_group": group, "path_kind": path_kind, "metric": metric,
                "n": len(subset), "mean": float(subset[metric].mean()),
                "median": float(subset[metric].median()), "ci95_low": low, "ci95_high": high,
            })
    return pd.DataFrame(output)


def run_mechanism_followups(config: MechanismFollowupConfig | None = None) -> Path:
    config = config or MechanismFollowupConfig()
    started = time.time()
    selected, _top = _load_inputs(Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None)
    branch_selected = balanced_subset(selected, config.branch_examples_per_group, seed=config.seed)
    path_selected = balanced_subset(selected, config.path_examples_per_group, seed=config.seed + 17)
    exp, cache_path = load_experiment(
        prefer_device=config.device,
        require_device=bool(config.require_device),
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    payload = {
        **asdict(config),
        "branch_ids": branch_selected.example_id.tolist(),
        "path_ids": path_selected.example_id.tolist(),
    }
    run_dir = Path(config.output_dir) / f"mechanism_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    branch_rows = run_branch_factorial(exp, model, detect, branch_selected, config)
    branch_summary = _summarize_branch(branch_rows, config.seed)
    path_detail, path_examples = run_path_validation(exp, model, detect, path_selected, config)
    path_summary = _summarize_path(path_examples, config.seed)
    branch_rows.to_csv(run_dir / "branch_factorial_rows.csv", index=False)
    branch_summary.to_csv(run_dir / "branch_factorial_summary.csv", index=False)
    path_detail.to_csv(run_dir / "path_detail.csv", index=False)
    path_examples.to_csv(run_dir / "path_example_summary.csv", index=False)
    path_summary.to_csv(run_dir / "path_group_summary.csv", index=False)
    elapsed = time.time() - started
    hidden = branch_summary[branch_summary.analysis_group.str.startswith("hidden")]
    reproduction = hidden.groupby("condition").target_hidden_rate.mean().to_dict()
    path_pivot = path_summary[path_summary.metric == "normalized_chord_deviation"].pivot(
        index="analysis_group", columns="path_kind", values="mean"
    ).to_dict()
    write_summary(run_dir / "summary.json", {
        "status": "complete", "elapsed_seconds": elapsed,
        "branch_examples": int(branch_rows.example_id.nunique()),
        "path_examples": int(path_detail.example_id.nunique()),
        "hidden_reproduction_mean": reproduction,
        "normalized_path_nonlinearity": path_pivot,
        "cache_path": str(cache_path), "config": asdict(config),
    })
    (run_dir / "analysis_digest.md").write_text(
        "# Mechanism follow-ups\n\n"
        f"- elapsed: {elapsed:.1f} s\n"
        f"- branch examples: {branch_rows.example_id.nunique()}\n"
        f"- path examples: {path_detail.example_id.nunique()}\n"
        f"- hidden reproduction by branch condition: `{json.dumps(reproduction, ensure_ascii=False)}`\n\n"
        "See branch_factorial_summary.csv and path_group_summary.csv first.\n",
        encoding="utf-8",
    )
    return run_dir


if __name__ == "__main__":
    print(run_mechanism_followups())
