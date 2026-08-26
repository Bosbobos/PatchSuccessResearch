from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .candidate_routing import _box_iou, _match_post_nms_to_flat, _xywh_to_xyxy
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
from .mechanism_followup import _decode, _head_branches, _raw_from_branches


@dataclass(slots=True)
class CandidateReserveConfig:
    output_dir: str = str(FOLLOWUP_DIR)
    device: str = "cpu"
    require_device: bool = False
    examples_per_group: int = 100
    budgets: tuple[int, ...] = (1, 2, 4, 6, 8, 9, 10, 12)
    target_iou: float = 0.50
    detection_conf: float = 0.25
    candidate_min_score: float = 0.01
    random_max_iou: float = 0.10
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    seed: int = 211
    method_version: int = 1


def _cache_lookup(exp):
    return {
        stable_hash({"path": str(item.path), "drop": float(item.drop), "success": bool(item.success)}): item
        for item in exp.get_cache().examples
    }


def _candidate_frames(selected: pd.DataFrame, config: CandidateReserveConfig) -> dict[str, dict]:
    with sqlite3.connect(TRACE_DB) as connection:
        candidates = pd.read_sql_query(
            "SELECT example_id,rank,flat_index,level_index,y_index,x_index,decoded_score,"
            "bbox_x1,bbox_y1,bbox_x2,bbox_y2 FROM candidates WHERE variant='clean'",
            connection,
        )
    selected_columns = selected[[
        "example_id", "clean_target_flat", "clean_target_level", "clean_target_y", "clean_target_x",
        "clean_target_x1", "clean_target_y1", "clean_target_x2", "clean_target_y2",
    ]]
    candidates = candidates[candidates.example_id.isin(selected.example_id)].merge(
        selected_columns, on="example_id", validate="many_to_one"
    )
    output: dict[str, dict] = {}
    for example_id, frame in candidates.groupby("example_id", sort=False):
        target_box = np.asarray([
            frame.clean_target_x1.iloc[0], frame.clean_target_y1.iloc[0],
            frame.clean_target_x2.iloc[0], frame.clean_target_y2.iloc[0],
        ], dtype=float)
        boxes = frame[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)
        left_top = np.maximum(boxes[:, :2], target_box[:2])
        right_bottom = np.minimum(boxes[:, 2:], target_box[2:])
        wh = np.maximum(0.0, right_bottom - left_top)
        intersection = wh[:, 0] * wh[:, 1]
        area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        area_target = max(0.0, target_box[2] - target_box[0]) * max(0.0, target_box[3] - target_box[1])
        local = frame.copy()
        local["target_iou"] = intersection / np.maximum(area_boxes + area_target - intersection, 1e-12)
        target_set = local[
            (local.target_iou >= config.target_iou)
            & (local.decoded_score >= config.candidate_min_score)
        ].sort_values(["decoded_score", "rank"], ascending=[False, True]).drop_duplicates("flat_index")
        random_pool = local[local.target_iou < config.random_max_iou].sort_values(
            ["decoded_score", "rank"], ascending=[False, True]
        ).drop_duplicates("flat_index")
        tracked_flat = int(local.clean_target_flat.iloc[0])
        tracked = target_set[target_set.flat_index == tracked_flat]
        if tracked.empty:
            tracked = pd.DataFrame([{
                "flat_index": tracked_flat,
                "level_index": int(local.clean_target_level.iloc[0]),
                "y_index": int(local.clean_target_y.iloc[0]),
                "x_index": int(local.clean_target_x.iloc[0]),
                "decoded_score": float(target_set.decoded_score.max()) if len(target_set) else 1.0,
            }])
        output[str(example_id)] = {
            "target_set": target_set.reset_index(drop=True),
            "random_pool": random_pool.reset_index(drop=True),
            "tracked": tracked.iloc[:1].reset_index(drop=True),
        }
    return output


def _matched_random(selection: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    if selection.empty or pool.empty:
        return pool.iloc[:0].copy()
    remaining = pool.copy()
    chosen = []
    for target in selection.itertuples(index=False):
        if remaining.empty:
            break
        same_level = remaining[remaining.level_index == int(target.level_index)]
        candidates = same_level if len(same_level) else remaining
        index = (candidates.decoded_score - float(target.decoded_score)).abs().idxmin()
        chosen.append(remaining.loc[index])
        remaining = remaining.drop(index)
    return pd.DataFrame(chosen).reset_index(drop=True) if chosen else pool.iloc[:0].copy()


def _selection_specs(frames: dict, config: CandidateReserveConfig) -> list[dict]:
    target_set = frames["target_set"]
    specs = [{
        "condition": "none", "selection_kind": "none", "budget_label": "0",
        "requested_k": 0, "selection": target_set.iloc[:0],
    }]
    selections = [("tracked", frames["tracked"]), ("top1", target_set.head(1))]
    selections.extend((f"top{k}", target_set.head(int(k))) for k in config.budgets if int(k) != 1)
    selections.append(("all", target_set))
    for label, selection in selections:
        requested = -1 if label == "all" else (1 if label in {"tracked", "top1"} else int(label[3:]))
        specs.append({
            "condition": label, "selection_kind": "target", "budget_label": label,
            "requested_k": requested, "selection": selection,
        })
        random_selection = _matched_random(selection, frames["random_pool"])
        specs.append({
            "condition": f"random_{label}", "selection_kind": "random",
            "budget_label": label, "requested_k": requested, "selection": random_selection,
        })
    return specs


def _replace_class_logits(base_cls, source_cls, selection: pd.DataFrame, class_id: int):
    output = [item.clone() for item in base_cls]
    for item in selection.itertuples(index=False):
        level, y, x = int(item.level_index), int(item.y_index), int(item.x_index)
        output[level][0, class_id, y, x] = source_cls[level][0, class_id, y, x]
    return output


def _batched_raw(box, base_cls, source_cls, specs: list[dict], class_id: int):
    raw_per_condition = []
    for spec in specs:
        cls = _replace_class_logits(base_cls, source_cls, spec["selection"], class_id)
        raw_per_condition.append(_raw_from_branches(box, cls))
    return [
        __import__("torch").cat([raw[level] for raw in raw_per_condition], dim=0)
        for level in range(len(box))
    ]


def _evaluate_batch(detect, raw, row, config: CandidateReserveConfig) -> list[dict]:
    import torch
    from ultralytics.utils.nms import non_max_suppression

    class_id = int(row.class_id)
    target_box = torch.as_tensor(
        [[row.clean_target_x1, row.clean_target_y1, row.clean_target_x2, row.clean_target_y2]],
        device=raw[0].device, dtype=torch.float32,
    )
    decoded = _decode(detect, raw)
    nms_rows = non_max_suppression(
        decoded.detach().clone(), conf_thres=config.nms_conf, iou_thres=config.nms_iou,
        classes=[class_id], max_det=config.nms_max_det, nc=int(detect.nc),
        # Ultralytics defaults to 0.05 s/image and silently stops processing the
        # remainder of a batched intervention once the global time limit is
        # exceeded. MPS occasionally crosses that limit for the 7-14 condition
        # batches used here, which turns an infrastructure timeout into an
        # apparent target-hiding event. Keep the same NMS semantics but give
        # every intervention enough time to be evaluated.
        max_time_img=float(getattr(config, "nms_max_time_img", 1.0)),
    )
    output = []
    for index, detections in enumerate(nms_rows):
        boxes = _xywh_to_xyxy(decoded[index, :4, :].transpose(0, 1))
        scores = decoded[index, 4 + class_id, :]
        ious = _box_iou(boxes, target_box).reshape(-1)
        valid_pre = torch.nonzero(ious >= config.target_iou, as_tuple=False).reshape(-1)
        pre_conf = float(scores[valid_pre].max().cpu()) if len(valid_pre) else 0.0
        pre_detected = int(pre_conf >= config.detection_conf)
        post_conf = 0.0
        post_iou = 0.0
        matched_flat = None
        if len(detections):
            post_ious = _box_iou(detections[:, :4], target_box).reshape(-1)
            post_iou = float(post_ious.max().cpu())
            valid_post = torch.nonzero(post_ious >= config.target_iou, as_tuple=False).reshape(-1)
            if len(valid_post):
                chosen = int(valid_post[torch.argmax(detections[valid_post, 4])])
                post_conf = float(detections[chosen, 4].cpu())
                matched_flat = int(_match_post_nms_to_flat(index, detections[chosen], decoded, class_id))
        detected = int(post_conf >= config.detection_conf and post_iou >= config.target_iou)
        output.append({
            "pre_target_conf": pre_conf, "pre_target_detected": pre_detected,
            "post_target_conf": post_conf, "post_target_iou": post_iou,
            "post_target_flat": matched_flat, "target_detected": detected,
            "target_hidden": 1 - detected, "nms_only_hidden": int(pre_detected and not detected),
            "tracked_score": float(scores[int(row.clean_target_flat)].cpu()),
        })
    return output


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = ["analysis_group", "direction", "selection_kind", "budget_label", "requested_k"]
    for values, frame in rows.groupby(keys, dropna=False, sort=False):
        baseline_hidden = frame.baseline_hidden.astype(bool)
        repair_denominator = int(baseline_hidden.sum())
        transplant_denominator = int((~baseline_hidden).sum())
        records.append({
            **dict(zip(keys, values, strict=True)), "n": frame.example_id.nunique(),
            "mean_actual_k": float(frame.actual_k.mean()),
            "target_hidden_rate": float(frame.target_hidden.mean()),
            "target_detection_rate": float(frame.target_detected.mean()),
            "nms_only_hidden_rate": float(frame.nms_only_hidden.mean()),
            "mean_post_conf": float(frame.post_target_conf.mean()),
            "repair_denominator": repair_denominator,
            "recovery_rate": float(frame.loc[baseline_hidden, "target_detected"].mean()) if repair_denominator else np.nan,
            "transplant_denominator": transplant_denominator,
            "reproduced_hiding_rate": float(frame.loc[~baseline_hidden, "target_hidden"].mean()) if transplant_denominator else np.nan,
        })
    return pd.DataFrame(records)


def run_candidate_reserve(config: CandidateReserveConfig | None = None) -> Path:
    config = config or CandidateReserveConfig()
    started = time.time()
    selected, _ = _load_inputs(Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None)
    selected = balanced_subset(selected, config.examples_per_group, seed=config.seed)
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
    output = []
    progress = tqdm(selected.itertuples(index=False), total=len(selected), desc="candidate reserve", unit="image")
    for row in progress:
        example_id = str(row.example_id)
        if example_id not in frames:
            continue
        example = cache[example_id]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        captured = _capture_detect_inputs(model, detect, pair)
        clean_inputs = [item[0:1] for item in captured]
        patched_inputs = [item[1:2] for item in captured]
        import torch

        with torch.inference_mode():
            clean_box, clean_cls, _ = _head_branches(detect, clean_inputs)
            patched_box, patched_cls, _ = _head_branches(detect, patched_inputs)
            specs = _selection_specs(frames[example_id], config)
            directions = {
                "repair_patched": _batched_raw(patched_box, patched_cls, clean_cls, specs, int(row.class_id)),
                "transplant_clean": _batched_raw(clean_box, clean_cls, patched_cls, specs, int(row.class_id)),
            }
            for direction, raw in directions.items():
                results = _evaluate_batch(detect, raw, row, config)
                for spec, result in zip(specs, results, strict=True):
                    result.update({
                        "example_id": example_id, "analysis_group": str(row.analysis_group),
                        "direction": direction, "condition": spec["condition"],
                        "selection_kind": spec["selection_kind"], "budget_label": spec["budget_label"],
                        "requested_k": int(spec["requested_k"]), "actual_k": len(spec["selection"]),
                        "clean_set_size": len(frames[example_id]["target_set"]),
                        "selection_contains_tracked": int(
                            int(row.clean_target_flat) in set(spec["selection"].flat_index.astype(int))
                        ) if len(spec["selection"]) else 0,
                    })
                    output.append(result)
        release_accelerator_memory()
    rows = pd.DataFrame(output)
    baseline = rows[rows.condition == "none"][["example_id", "direction", "target_hidden"]].rename(
        columns={"target_hidden": "baseline_hidden"}
    )
    rows = rows.merge(baseline, on=["example_id", "direction"], validate="many_to_one")
    summary = _summarize(rows)
    payload = {**asdict(config), "example_ids": selected.example_id.tolist()}
    run_dir = Path(config.output_dir) / f"candidate_reserve_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(run_dir / "candidate_reserve_rows.csv", index=False)
    summary.to_csv(run_dir / "candidate_reserve_summary.csv", index=False)
    elapsed = time.time() - started
    all_target = summary[(summary.selection_kind == "target") & (summary.budget_label == "all")]
    write_summary(run_dir / "summary.json", {
        "status": "complete", "elapsed_seconds": elapsed,
        "n_examples": int(rows.example_id.nunique()), "n_conditions": int(rows.condition.nunique()),
        "all_target_results": all_target.to_dict("records"),
        "cache_path": str(cache_path), "config": asdict(config),
        "limitations": [
            "The clean target set is limited to saved top-50 person candidates.",
            "Same-flat lineage is operational and does not guarantee unchanged semantic identity.",
            "Only person logits are replaced; geometry is deliberately held at the base endpoint.",
        ],
    })
    (run_dir / "analysis_digest.md").write_text(
        "# Candidate reserve causal intervention\n\n"
        f"- elapsed: {elapsed:.1f} s\n- examples: {rows.example_id.nunique()}\n"
        "- see `candidate_reserve_summary.csv` for repair and transplant dose curves\n",
        encoding="utf-8",
    )
    return run_dir


if __name__ == "__main__":
    print(run_candidate_reserve())
