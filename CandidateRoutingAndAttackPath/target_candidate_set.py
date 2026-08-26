from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .attack_path import _capture_detect_inputs, _path_integral, _preprocess_pair, _top_rows, TargetTerm
from .causal_repair import (
    _coords_from_top, _delta_top_coords, _load_inputs, _random_delta_matched_coords,
)
from .causal_transplant import _transplant_variant
from .candidate_routing import (
    _box_iou, _flat_location, _level_slices, _match_post_nms_to_flat,
    _parse_forward_output, _xywh_to_xyxy,
)
from .common import (
    DEFAULT_MAX_OUTPUT_GB, DEFAULT_OUTPUT_DIR, StorageBudget, connect_db,
    release_accelerator_memory, stable_hash, upsert_metadata, write_json, write_markdown,
)


@dataclass(slots=True)
class TargetCandidateSetConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    max_output_gb: float = DEFAULT_MAX_OUTPUT_GB
    max_examples: int | None = None
    n_steps: int = 8
    alpha_batch_size: int = 4
    candidate_iou: float = 0.50
    candidate_min_score: float = 0.01
    max_candidates: int = 32
    smoothmax_temperature: float = 0.50
    top_contributions: int = 1000
    k_values: tuple[int, ...] = (10, 50, 100, 250)
    random_repeats: int = 2
    forward_batch_size: int = 8
    detection_conf: float = 0.25
    match_iou: float = 0.50
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    seed: int = 83
    progress: bool = True
    method_version: int = 1


@dataclass(slots=True)
class TargetCandidateSetResult:
    run_dir: Path
    db_path: Path
    summary_path: Path
    digest_path: Path
    group_summary_path: Path
    handoff_summary_path: Path
    pairwise_path: Path
    config: TargetCandidateSetConfig


SET_EXAMPLE_COLUMNS = (
    "example_id", "analysis_group", "match_set", "candidate_count", "candidate_flats_json",
    "clean_set_score", "patched_set_score", "exact_set_delta", "path_sum",
    "completeness_error", "relative_completeness_error", "total_abs_contribution",
    "negative_contribution", "positive_contribution", "error",
)

RESULT_COLUMNS = (
    "example_id", "analysis_group", "match_set", "strategy", "k_requested", "repeat",
    "actual_k", "transplanted_delta_l1", "transplanted_delta_l2",
    "fixed_target_logit", "fixed_target_score", "target_set_score", "fixed_box_iou_clean",
    "target_match_conf", "target_max_iou", "target_detected", "target_hidden",
    "matched_flat", "matched_level", "matched_y", "matched_x", "matched_same_clean_flat",
    "matched_in_clean_set", "matched_grid_distance", "matched_level_changed",
    "fixed_logit_loss", "target_set_score_loss", "target_conf_loss", "target_iou_loss", "error",
)

TOP_COLUMNS = (
    "example_id", "global_rank", "level_index", "channel", "y_index", "x_index",
    "contribution", "abs_contribution", "delta_activation", "average_gradient", "sign",
)


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS set_examples (
        example_id TEXT PRIMARY KEY, analysis_group TEXT, match_set INTEGER, candidate_count INTEGER,
        candidate_flats_json TEXT, clean_set_score REAL, patched_set_score REAL,
        exact_set_delta REAL, path_sum REAL, completeness_error REAL,
        relative_completeness_error REAL, total_abs_contribution REAL,
        negative_contribution REAL, positive_contribution REAL, error TEXT)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS set_transplant_results (
        example_id TEXT, analysis_group TEXT, match_set INTEGER, strategy TEXT,
        k_requested INTEGER, repeat INTEGER, actual_k INTEGER,
        transplanted_delta_l1 REAL, transplanted_delta_l2 REAL,
        fixed_target_logit REAL, fixed_target_score REAL, target_set_score REAL,
        fixed_box_iou_clean REAL, target_match_conf REAL, target_max_iou REAL,
        target_detected INTEGER, target_hidden INTEGER, matched_flat INTEGER,
        matched_level INTEGER, matched_y INTEGER, matched_x INTEGER,
        matched_same_clean_flat INTEGER, matched_in_clean_set INTEGER,
        matched_grid_distance REAL, matched_level_changed INTEGER,
        fixed_logit_loss REAL, target_set_score_loss REAL, target_conf_loss REAL,
        target_iou_loss REAL, error TEXT,
        PRIMARY KEY(example_id, strategy, k_requested, repeat))"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS set_top_contributions (
        example_id TEXT, global_rank INTEGER, level_index INTEGER, channel INTEGER,
        y_index INTEGER, x_index INTEGER, contribution REAL, abs_contribution REAL,
        delta_activation REAL, average_gradient REAL, sign INTEGER,
        PRIMARY KEY(example_id, global_rank))"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_set_result_group ON set_transplant_results(analysis_group, strategy, k_requested)")
    conn.commit()


def _insert(conn, table: str, columns, rows) -> None:
    if not rows:
        return
    placeholders = ",".join("?" for _ in columns)
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
        [[row.get(column) for column in columns] for row in rows],
    )


def _set_vector(raw_levels, locations, class_channel: int, temperature: float):
    import torch

    values = torch.stack([
        raw_levels[level][:, int(class_channel), y, x] for level, y, x in locations
    ], dim=1)
    temperature = float(temperature)
    return temperature * (torch.logsumexp(values / temperature, dim=1) - math.log(values.shape[1]))


def _set_endpoint(detect, inputs, locations, class_channel, temperature) -> float:
    import torch

    with torch.no_grad():
        _decoded, raw = _parse_forward_output(detect([item.clone() for item in inputs]))
        value = _set_vector(raw, locations, class_channel, temperature)
    return float(value.detach().float().cpu()[0])


def _set_path_integral(detect, clean_inputs, deltas, locations, class_channel, config):
    import torch

    alphas = (torch.arange(config.n_steps, device=clean_inputs[0].device, dtype=torch.float32) + 0.5) / config.n_steps
    accumulated = [torch.zeros_like(delta, dtype=torch.float32) for delta in deltas]
    for start in range(0, config.n_steps, config.alpha_batch_size):
        alpha = alphas[start:start + config.alpha_batch_size]
        variables = []
        for clean, delta in zip(clean_inputs, deltas, strict=True):
            view = alpha.to(dtype=clean.dtype).reshape(-1, 1, 1, 1)
            variables.append((clean + view * delta).detach().requires_grad_(True))
        _decoded, raw = _parse_forward_output(detect(list(variables)))
        score = _set_vector(raw, locations, class_channel, config.smoothmax_temperature)
        grads = torch.autograd.grad(score.sum(), variables, allow_unused=True)
        for index, (gradient, delta) in enumerate(zip(grads, deltas, strict=True)):
            if gradient is not None:
                accumulated[index] += (gradient.detach().float() * delta.float()).sum(dim=0, keepdim=True)
    return [item / float(config.n_steps) for item in accumulated]


def _candidate_set(detect, clean_inputs, row, class_id: int, config):
    import torch

    with torch.no_grad():
        decoded, raw = _parse_forward_output(detect([item.clone() for item in clean_inputs]))
    clean_box = torch.as_tensor(
        [row.clean_target_x1, row.clean_target_y1, row.clean_target_x2, row.clean_target_y2],
        device=decoded.device, dtype=torch.float32,
    ).reshape(1, 4)
    boxes = _xywh_to_xyxy(decoded[0, :4, :].transpose(0, 1))
    ious = _box_iou(boxes, clean_box).reshape(-1)
    scores = decoded[0, 4 + int(class_id), :]
    valid = torch.nonzero(
        (ious >= float(config.candidate_iou)) & (scores >= float(config.candidate_min_score)),
        as_tuple=False,
    ).reshape(-1)
    target_flat = int(row.clean_target_flat)
    candidates = valid.detach().cpu().tolist()
    if target_flat not in candidates:
        candidates.append(target_flat)
    candidates = sorted(set(int(value) for value in candidates), key=lambda value: float(scores[value]), reverse=True)
    candidates = candidates[: max(1, int(config.max_candidates))]
    slices = _level_slices(raw)
    locations = [_flat_location(flat, slices) for flat in candidates]
    return candidates, locations


def _variant_specs(clean_inputs, patched_inputs, fixed_top, set_top, target_level, config, seed):
    yield {"strategy": "clean", "k_requested": 0, "repeat": 0, "coords": [], "inputs": clean_inputs}
    yield {"strategy": "patched_full_head", "k_requested": -1, "repeat": 0, "coords": None,
           "inputs": [item.detach().clone() for item in patched_inputs]}
    deltas = [patched - clean for clean, patched in zip(clean_inputs, patched_inputs, strict=True)]
    for k in config.k_values:
        fixed_negative = _coords_from_top(fixed_top, "top_negative", k)
        set_negative = _coords_from_top(set_top, "top_negative", k)
        selections = {
            "fixed_top_negative": fixed_negative,
            "set_top_negative": set_negative,
            "set_top_abs": _coords_from_top(set_top, "top_abs", k),
            "delta_magnitude": _delta_top_coords(deltas, target_level, k),
        }
        for strategy, coords in selections.items():
            inputs, l1, l2 = _transplant_variant(clean_inputs, patched_inputs, coords)
            yield {"strategy": strategy, "k_requested": k, "repeat": 0, "coords": coords,
                   "inputs": inputs, "l1": l1, "l2": l2}
        for repeat in range(config.random_repeats):
            rng = np.random.default_rng(seed + 1009 * k + repeat)
            coords = _random_delta_matched_coords(deltas, target_level, set_negative, rng)
            inputs, l1, l2 = _transplant_variant(clean_inputs, patched_inputs, coords)
            yield {"strategy": "random_delta_matched", "k_requested": k, "repeat": repeat,
                   "coords": coords, "inputs": inputs, "l1": l1, "l2": l2}


def _expected_keys(config):
    keys = {("clean", 0, 0), ("patched_full_head", -1, 0)}
    for k in config.k_values:
        keys.update((strategy, k, 0) for strategy in (
            "fixed_top_negative", "set_top_negative", "set_top_abs", "delta_magnitude",
        ))
        keys.update(("random_delta_matched", k, repeat) for repeat in range(config.random_repeats))
    return keys


def _evaluate(detect, specs, row, candidate_flats, locations, config, nc: int):
    import torch
    from ultralytics.utils.nms import non_max_suppression

    class_id = int(row.class_id)
    reg_max = int(getattr(detect, "reg_max", 16))
    class_channel = 4 * reg_max + class_id
    target_flat = int(row.clean_target_flat)
    clean_level, clean_y, clean_x = int(row.clean_target_level), int(row.clean_target_y), int(row.clean_target_x)
    clean_box = torch.as_tensor(
        [row.clean_target_x1, row.clean_target_y1, row.clean_target_x2, row.clean_target_y2],
        device=next(detect.parameters()).device, dtype=torch.float32,
    ).reshape(1, 4)
    output = []
    for start in range(0, len(specs), config.forward_batch_size):
        chunk = specs[start:start + config.forward_batch_size]
        levels = [torch.cat([spec["inputs"][level] for spec in chunk], dim=0) for level in range(len(chunk[0]["inputs"]))]
        with torch.inference_mode():
            decoded, raw = _parse_forward_output(detect(levels))
            nms = non_max_suppression(
                decoded.detach().clone(), conf_thres=config.nms_conf, iou_thres=config.nms_iou,
                classes=[class_id], max_det=config.nms_max_det, nc=nc,
            )
        slices = _level_slices(raw)
        fixed_boxes = _xywh_to_xyxy(decoded[:, :4, target_flat].reshape(-1, 4))
        set_scores = _set_vector(raw, locations, class_channel, config.smoothmax_temperature)
        for index, spec in enumerate(chunk):
            detections = nms[index]
            fixed_iou = float(_box_iou(fixed_boxes[index:index + 1], clean_box).reshape(-1)[0].cpu())
            target_conf = max_iou = 0.0
            matched_flat = matched_level = matched_y = matched_x = None
            if len(detections):
                ious = _box_iou(detections[:, :4], clean_box).reshape(-1)
                max_iou = float(ious.max().cpu())
                valid = torch.nonzero(ious >= config.match_iou, as_tuple=False).reshape(-1)
                if len(valid):
                    chosen = int(valid[torch.argmax(detections[valid, 4])])
                    target_conf = float(detections[chosen, 4].cpu())
                    matched_flat = _match_post_nms_to_flat(index, detections[chosen], decoded, class_id)
                    matched_level, matched_y, matched_x = _flat_location(matched_flat, slices)
            detected = int(target_conf >= config.detection_conf and max_iou >= config.match_iou)
            distance = None
            level_changed = None
            if matched_flat is not None:
                level_changed = int(matched_level != clean_level)
                distance = float(math.hypot(matched_y - clean_y, matched_x - clean_x)) if not level_changed else np.nan
            output.append({
                "strategy": spec["strategy"], "k_requested": int(spec["k_requested"]),
                "repeat": int(spec["repeat"]), "actual_k": len(spec["coords"]) if spec["coords"] is not None else -1,
                "transplanted_delta_l1": spec.get("l1", np.nan), "transplanted_delta_l2": spec.get("l2", np.nan),
                "fixed_target_logit": float(raw[clean_level][index, class_channel, clean_y, clean_x].float().cpu()),
                "fixed_target_score": float(decoded[index, 4 + class_id, target_flat].float().cpu()),
                "target_set_score": float(set_scores[index].float().cpu()), "fixed_box_iou_clean": fixed_iou,
                "target_match_conf": target_conf, "target_max_iou": max_iou,
                "target_detected": detected, "target_hidden": 1 - detected,
                "matched_flat": matched_flat, "matched_level": matched_level, "matched_y": matched_y, "matched_x": matched_x,
                "matched_same_clean_flat": int(matched_flat == target_flat) if matched_flat is not None else None,
                "matched_in_clean_set": int(matched_flat in candidate_flats) if matched_flat is not None else None,
                "matched_grid_distance": distance, "matched_level_changed": level_changed,
            })
    return output


def _pairwise(rows):
    from scipy.stats import wilcoxon

    unit = rows.groupby(["example_id", "analysis_group", "strategy", "k_requested"], as_index=False).agg(
        target_set_score_loss=("target_set_score_loss", "mean"), target_conf_loss=("target_conf_loss", "mean"),
        target_iou_loss=("target_iou_loss", "mean"), target_hidden=("target_hidden", "mean"),
    )
    output = []
    for group in unit.analysis_group.unique():
        for k in sorted(value for value in unit.k_requested.unique() if value > 0):
            primary = unit[(unit.analysis_group == group) & (unit.k_requested == k) & (unit.strategy == "set_top_negative")]
            for control in ("fixed_top_negative", "set_top_abs", "delta_magnitude", "random_delta_matched"):
                other = unit[(unit.analysis_group == group) & (unit.k_requested == k) & (unit.strategy == control)]
                for metric in ("target_set_score_loss", "target_conf_loss", "target_iou_loss", "target_hidden"):
                    paired = primary[["example_id", metric]].merge(other[["example_id", metric]], on="example_id", suffixes=("_set", "_control")).dropna()
                    difference = paired[f"{metric}_set"] - paired[f"{metric}_control"]
                    p = 1.0 if np.allclose(difference, 0) else float(wilcoxon(difference, zero_method="zsplit").pvalue)
                    output.append({"analysis_group": group, "k_requested": k, "control": control, "metric": metric,
                                   "n": len(difference), "set_mean": paired[f"{metric}_set"].mean(),
                                   "control_mean": paired[f"{metric}_control"].mean(),
                                   "mean_difference": difference.mean(), "p_value": p})
    result = pd.DataFrame(output)
    if not result.empty:
        order = np.argsort(result.p_value.to_numpy())
        adjusted = np.empty(len(result), dtype=float)
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, (len(result) - rank) * float(result.iloc[index].p_value))
            adjusted[index] = min(1.0, running)
        result["p_holm"] = adjusted
    return result


def _summaries(conn, run_dir, config, expected_examples, expected_rows):
    examples = pd.read_sql_query("SELECT * FROM set_examples WHERE error IS NULL", conn)
    rows = pd.read_sql_query("SELECT * FROM set_transplant_results WHERE error IS NULL", conn)
    group_path = run_dir / "candidate_set_group_summary.csv"
    handoff_path = run_dir / "candidate_handoff_summary.csv"
    pairwise_path = run_dir / "candidate_set_pairwise.csv"
    group = rows.groupby(["analysis_group", "strategy", "k_requested"], dropna=False).agg(
        n=("example_id", "nunique"), n_rows=("example_id", "size"), mean_actual_k=("actual_k", "mean"),
        target_hidden_rate=("target_hidden", "mean"), mean_fixed_logit_loss=("fixed_logit_loss", "mean"),
        mean_set_score_loss=("target_set_score_loss", "mean"), mean_conf_loss=("target_conf_loss", "mean"),
        mean_iou_loss=("target_iou_loss", "mean"), mean_transplanted_l1=("transplanted_delta_l1", "mean"),
    ).reset_index()
    path_by_group = examples.groupby("analysis_group", dropna=False).agg(
        mean_candidate_count=("candidate_count", "mean"),
        mean_exact_set_delta=("exact_set_delta", "mean"),
        mean_total_abs_contribution=("total_abs_contribution", "mean"),
        median_relative_completeness_error=("relative_completeness_error", "median"),
    ).reset_index()
    group = group.merge(path_by_group, on="analysis_group", how="left", validate="many_to_one")
    group.to_csv(group_path, index=False)
    handoff = rows.groupby(["analysis_group", "strategy", "k_requested"], dropna=False).agg(
        n=("example_id", "nunique"), target_hidden_rate=("target_hidden", "mean"),
        same_clean_flat_rate=("matched_same_clean_flat", "mean"), in_clean_set_rate=("matched_in_clean_set", "mean"),
        level_changed_rate=("matched_level_changed", "mean"), mean_grid_distance=("matched_grid_distance", "mean"),
    ).reset_index()
    handoff.to_csv(handoff_path, index=False)
    pairwise = _pairwise(rows)
    pairwise.to_csv(pairwise_path, index=False)
    errors = conn.execute("SELECT COUNT(*) FROM set_examples WHERE error IS NOT NULL").fetchone()[0]
    payload = {
        "status": "complete" if len(examples) == expected_examples and len(rows) == expected_rows else "partial",
        "n_examples": len(examples), "expected_examples": expected_examples, "n_rows": len(rows),
        "expected_rows": expected_rows, "error_rows": errors,
        "mean_candidate_count": float(examples.candidate_count.mean()) if len(examples) else None,
        "median_relative_completeness_error": float(examples.relative_completeness_error.median()) if len(examples) else None,
        "database": str(run_dir / "target_candidate_set.sqlite"), "config": asdict(config),
    }
    summary = write_json(run_dir / "summary.json", payload)
    mean_count = "n/a" if payload["mean_candidate_count"] is None else f"{payload['mean_candidate_count']:.2f}"
    completeness = "n/a" if payload["median_relative_completeness_error"] is None else f"{payload['median_relative_completeness_error']:.6f}"
    digest = write_markdown(run_dir / "analysis_digest.md", [
        "# Target candidate-set causal experiment", "", f"- status: {payload['status']}",
        f"- examples: {payload['n_examples']} / {payload['expected_examples']}",
        f"- intervention rows: {payload['n_rows']} / {payload['expected_rows']}",
        f"- errors: {payload['error_rows']}", f"- mean clean target-set size: {mean_count}",
        f"- median path relative completeness error: {completeness}", "",
        "Read candidate_set_group_summary.csv, candidate_handoff_summary.csv, and candidate_set_pairwise.csv first.",
    ])
    return summary, digest, group_path, handoff_path, pairwise_path


def run_target_candidate_set(exp, attack_path_db, trace_db, manifest_csv, config=None, *, force=False):
    from segmentig_detector.yolo_utils import get_detect_module

    config = config or TargetCandidateSetConfig()
    attack_path_db, trace_db, manifest_csv = map(Path, (attack_path_db, trace_db, manifest_csv))
    selected, fixed_top = _load_inputs(attack_path_db, trace_db, manifest_csv, config.max_examples)
    payload = {
        "attack_path_db": str(attack_path_db.resolve()), "attack_path_size": attack_path_db.stat().st_size,
        "trace_db": str(trace_db.resolve()), "trace_size": trace_db.stat().st_size,
        "manifest_csv": str(manifest_csv.resolve()), "manifest_size": manifest_csv.stat().st_size,
        "example_ids": selected.example_id.tolist(), **asdict(config),
    }
    run_dir = Path(config.output_dir) / f"target_candidate_set_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    StorageBudget(Path(config.output_dir), config.max_output_gb).check()
    db_path = run_dir / "target_candidate_set.sqlite"
    if force and db_path.exists(): db_path.unlink()
    conn = connect_db(db_path); _create_schema(conn); upsert_metadata(conn, {"config": asdict(config), "payload": payload})
    completed = {(str(a), str(b), int(c), int(d)) for a, b, c, d in conn.execute(
        "SELECT example_id,strategy,k_requested,repeat FROM set_transplant_results WHERE error IS NULL")}
    expected_keys = _expected_keys(config)
    _yolo, model = exp.load_model(); model.eval(); detect = get_detect_module(model, exp.config.detect_layer); detect.eval()
    nc, reg_max = int(detect.nc), int(detect.reg_max)
    cache = {stable_hash({"path": str(item.path), "drop": float(item.drop), "success": bool(item.success)}): item for item in exp.get_cache().examples}
    progress = None
    if config.progress:
        from tqdm.auto import tqdm
        progress = tqdm(total=len(selected), desc="target candidate set", unit="img")
    try:
        for row in selected.itertuples(index=False):
            example_id = str(row.example_id)
            done = {(s, k, r) for e, s, k, r in completed if e == example_id}
            if expected_keys.issubset(done):
                if progress: progress.update(1)
                continue
            try:
                example = cache[example_id]
                clean_image, patched_image, _ = exp._images_for_example(example)
                pair = _preprocess_pair(exp, clean_image, patched_image)
                captured = _capture_detect_inputs(model, detect, pair)
                clean_inputs = [item[0:1].detach() for item in captured]
                patched_inputs = [item[1:2].detach() for item in captured]
                deltas = [patched - clean for clean, patched in zip(clean_inputs, patched_inputs, strict=True)]
                class_id = int(row.class_id); class_channel = 4 * reg_max + class_id
                candidate_flats, locations = _candidate_set(detect, clean_inputs, row, class_id, config)
                clean_score = _set_endpoint(detect, clean_inputs, locations, class_channel, config.smoothmax_temperature)
                patched_score = _set_endpoint(detect, patched_inputs, locations, class_channel, config.smoothmax_temperature)
                contributions_t = _set_path_integral(detect, clean_inputs, deltas, locations, class_channel, config)
                contributions = [item.detach().float().cpu().numpy() for item in contributions_t]
                path_sum = float(sum(item.sum() for item in contributions)); exact = patched_score - clean_score
                top_rows = _top_rows(example_id, "target_set", contributions, deltas, config.top_contributions)
                for item in top_rows: item.pop("target_kind", None); item.pop("level_name", None)
                set_row = {"example_id": example_id, "analysis_group": str(row.analysis_group), "match_set": int(row.match_set),
                           "candidate_count": len(candidate_flats), "candidate_flats_json": json.dumps(candidate_flats),
                           "clean_set_score": clean_score, "patched_set_score": patched_score, "exact_set_delta": exact,
                           "path_sum": path_sum, "completeness_error": path_sum - exact,
                           "relative_completeness_error": abs(path_sum - exact) / max(abs(exact), 1e-8),
                           "total_abs_contribution": sum(float(np.abs(item).sum()) for item in contributions),
                           "negative_contribution": sum(float(item[item < 0].sum()) for item in contributions),
                           "positive_contribution": sum(float(item[item > 0].sum()) for item in contributions), "error": None}
                set_top = pd.DataFrame(top_rows)
                seed = int(stable_hash({"example_id": example_id, "seed": config.seed})[:12], 16)
                specs = list(_variant_specs(clean_inputs, patched_inputs, fixed_top[fixed_top.example_id.eq(example_id)],
                                            set_top, int(row.clean_target_level), config, seed))
                specs = [spec for spec in specs if (example_id, spec["strategy"], spec["k_requested"], spec["repeat"]) not in completed]
                evaluated = _evaluate(detect, specs, row, candidate_flats, locations, config, nc)
                baseline = next((item for item in evaluated if item["strategy"] == "clean"), None)
                if baseline is None:
                    baseline_spec = next(_variant_specs(clean_inputs, patched_inputs, fixed_top[fixed_top.example_id.eq(example_id)], set_top, int(row.clean_target_level), config, seed))
                    baseline = _evaluate(detect, [baseline_spec], row, candidate_flats, locations, config, nc)[0]
                output = []
                for item in evaluated:
                    item.update({"example_id": example_id, "analysis_group": str(row.analysis_group), "match_set": int(row.match_set),
                                 "fixed_logit_loss": baseline["fixed_target_logit"] - item["fixed_target_logit"],
                                 "target_set_score_loss": baseline["target_set_score"] - item["target_set_score"],
                                 "target_conf_loss": baseline["target_match_conf"] - item["target_match_conf"],
                                 "target_iou_loss": baseline["target_max_iou"] - item["target_max_iou"], "error": None})
                    output.append(item)
                _insert(conn, "set_examples", SET_EXAMPLE_COLUMNS, [set_row]); _insert(conn, "set_top_contributions", TOP_COLUMNS, top_rows)
                _insert(conn, "set_transplant_results", RESULT_COLUMNS, output); conn.commit()
                StorageBudget(Path(config.output_dir), config.max_output_gb).check(extra_bytes=50 * 1024**2)
            except Exception as exc:  # noqa: BLE001
                error = {column: None for column in SET_EXAMPLE_COLUMNS}; error.update({"example_id": example_id,
                    "analysis_group": str(row.analysis_group), "match_set": int(row.match_set), "error": f"{type(exc).__name__}: {exc}"})
                _insert(conn, "set_examples", SET_EXAMPLE_COLUMNS, [error]); conn.commit()
            finally:
                if progress: progress.update(1)
                release_accelerator_memory()
    finally:
        if progress: progress.close()
        summary, digest, group, handoff, pairwise = _summaries(
            conn, run_dir, config, len(selected), len(selected) * len(expected_keys))
        conn.close()
    return TargetCandidateSetResult(run_dir, db_path, summary, digest, group, handoff, pairwise, config)


def load_target_candidate_set(result_or_db):
    db_path = result_or_db.db_path if isinstance(result_or_db, TargetCandidateSetResult) else Path(result_or_db)
    conn = connect_db(db_path)
    try:
        return {"examples": pd.read_sql_query("SELECT * FROM set_examples WHERE error IS NULL", conn),
                "results": pd.read_sql_query("SELECT * FROM set_transplant_results WHERE error IS NULL", conn)}
    finally:
        conn.close()
