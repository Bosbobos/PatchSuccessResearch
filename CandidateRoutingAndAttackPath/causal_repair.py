from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .candidate_routing import _box_iou, _level_slices, _parse_forward_output, _xywh_to_xyxy
from .common import (
    DEFAULT_MAX_OUTPUT_GB,
    DEFAULT_OUTPUT_DIR,
    StorageBudget,
    connect_db,
    release_accelerator_memory,
    stable_hash,
    upsert_metadata,
    write_json,
    write_markdown,
)


@dataclass(slots=True)
class CausalRepairConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    max_output_gb: float = DEFAULT_MAX_OUTPUT_GB
    max_examples: int | None = None
    k_values: tuple[int, ...] = (10, 50, 100, 250, 500, 1000)
    random_repeats: int = 3
    forward_batch_size: int = 8
    detection_conf: float = 0.25
    match_iou: float = 0.50
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    seed: int = 53
    progress: bool = True
    method_version: int = 1


@dataclass(slots=True)
class CausalRepairResult:
    run_dir: Path
    db_path: Path
    summary_path: Path
    digest_path: Path
    group_summary_path: Path
    config: CausalRepairConfig


REPAIR_COLUMNS = (
    "example_id", "analysis_group", "match_set", "strategy", "k_requested", "repeat",
    "actual_k", "restored_delta_l1", "restored_delta_l2", "fixed_target_logit",
    "fixed_target_score", "fixed_box_iou_clean", "target_match_conf", "target_match_iou",
    "target_max_iou", "target_detected", "target_hidden", "global_winner_conf",
    "global_winner_iou_target", "target_logit_gain", "target_conf_gain", "target_iou_gain",
    "error",
)


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS repair_results (
            example_id TEXT, analysis_group TEXT, match_set INTEGER, strategy TEXT,
            k_requested INTEGER, repeat INTEGER, actual_k INTEGER,
            restored_delta_l1 REAL, restored_delta_l2 REAL,
            fixed_target_logit REAL, fixed_target_score REAL, fixed_box_iou_clean REAL,
            target_match_conf REAL, target_match_iou REAL, target_max_iou REAL,
            target_detected INTEGER, target_hidden INTEGER, global_winner_conf REAL,
            global_winner_iou_target REAL, target_logit_gain REAL, target_conf_gain REAL,
            target_iou_gain REAL, error TEXT,
            PRIMARY KEY(example_id, strategy, k_requested, repeat)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_repair_group ON repair_results(analysis_group, strategy, k_requested)")
    conn.commit()


def _insert_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    placeholders = ",".join("?" for _ in REPAIR_COLUMNS)
    sql = f"INSERT OR REPLACE INTO repair_results ({','.join(REPAIR_COLUMNS)}) VALUES ({placeholders})"
    conn.executemany(sql, [[row.get(column) for column in REPAIR_COLUMNS] for row in rows])


def _load_inputs(attack_path_db: Path, trace_db: Path, manifest_csv: Path, max_examples: int | None):
    attack_conn = connect_db(attack_path_db)
    try:
        examples = pd.read_sql_query(
            "SELECT example_id, mechanism_mode AS analysis_group FROM path_examples "
            "WHERE error IS NULL AND target_kind='tracked_clean'",
            attack_conn,
        )
        top = pd.read_sql_query(
            "SELECT * FROM top_contributions WHERE target_kind='tracked_clean' ORDER BY example_id, global_rank",
            attack_conn,
        )
    finally:
        attack_conn.close()
    trace_conn = connect_db(trace_db)
    try:
        trace = pd.read_sql_query("SELECT * FROM examples WHERE error IS NULL", trace_conn)
    finally:
        trace_conn.close()
    manifest = pd.read_csv(manifest_csv)
    selected = examples.merge(
        manifest[["example_id", "analysis_group", "match_set"]],
        on=["example_id", "analysis_group"], how="inner", validate="one_to_one",
    ).merge(trace, on="example_id", how="inner", validate="one_to_one")
    selected = selected.sort_values(["match_set", "analysis_group"]).reset_index(drop=True)
    if max_examples is not None:
        selected = selected.head(int(max_examples)).copy()
    top = top.loc[top.example_id.isin(selected.example_id)].copy()
    return selected, top


def _coords_from_top(rows: pd.DataFrame, strategy: str, k: int) -> list[tuple[int, int, int, int]]:
    if strategy == "top_negative":
        chosen = rows.loc[rows.contribution.lt(0)].sort_values("contribution").head(int(k))
    elif strategy == "top_positive":
        chosen = rows.loc[rows.contribution.gt(0)].sort_values("contribution", ascending=False).head(int(k))
    elif strategy == "top_abs":
        chosen = rows.sort_values("abs_contribution", ascending=False).head(int(k))
    else:
        raise ValueError(strategy)
    return [
        (int(row.level_index), int(row.channel), int(row.y_index), int(row.x_index))
        for row in chosen.itertuples(index=False)
    ]


def _delta_top_coords(deltas, target_level: int, k: int) -> list[tuple[int, int, int, int]]:
    values = deltas[int(target_level)][0].detach().float().abs().reshape(-1)
    actual = min(int(k), int(values.numel()))
    indices = values.topk(actual).indices.detach().cpu().numpy()
    _batch, _channels, height, width = deltas[int(target_level)].shape
    return [
        (int(target_level), int(index // (height * width)), int((index % (height * width)) // width), int(index % width))
        for index in indices
    ]


def _random_delta_matched_coords(
    deltas, target_level: int, reference: list[tuple[int, int, int, int]], rng: np.random.Generator,
) -> list[tuple[int, int, int, int]]:
    if not reference:
        return []
    level = int(target_level)
    values = deltas[level][0].detach().float().abs().reshape(-1).cpu().numpy()
    order = np.argsort(values)
    sorted_values = values[order]
    _batch, _channels, height, width = deltas[level].shape
    used: set[int] = set()
    chosen: list[tuple[int, int, int, int]] = []
    window = max(16, int(len(order) * 0.005))
    for ref_level, channel, y, x in reference:
        ref_flat = int(channel * height * width + y * width + x)
        target_value = abs(float(deltas[int(ref_level)][0, channel, y, x].detach().float().cpu()))
        center = int(np.searchsorted(sorted_values, target_value))
        lo, hi = max(0, center - window), min(len(order), center + window + 1)
        pool = [int(item) for item in order[lo:hi] if int(item) not in used and int(item) != ref_flat]
        if not pool:
            pool = [int(item) for item in order if int(item) not in used and int(item) != ref_flat]
        flat = int(rng.choice(pool))
        used.add(flat)
        chosen.append((level, flat // (height * width), (flat % (height * width)) // width, flat % width))
    return chosen


def _repair_variant(clean_inputs, patched_inputs, coords):
    repaired = [item.detach().clone() for item in patched_inputs]
    restored = []
    for level, channel, y, x in coords:
        delta = patched_inputs[level][0, channel, y, x] - clean_inputs[level][0, channel, y, x]
        restored.append(float(delta.detach().float().cpu()))
        repaired[level][0, channel, y, x] = clean_inputs[level][0, channel, y, x]
    array = np.asarray(restored, dtype=np.float64)
    return repaired, float(np.abs(array).sum()), float(np.sqrt(np.square(array).sum()))


def _variant_specs(clean_inputs, patched_inputs, top_rows, target_level, config, example_seed):
    yield {"strategy": "patched", "k_requested": 0, "repeat": 0, "coords": [], "inputs": patched_inputs}
    yield {"strategy": "oracle_target_level", "k_requested": -1, "repeat": 0, "coords": None,
           "inputs": [clean.detach().clone() if idx == int(target_level) else patched.detach().clone()
                      for idx, (clean, patched) in enumerate(zip(clean_inputs, patched_inputs, strict=True))]}
    yield {"strategy": "oracle_full_head", "k_requested": -1, "repeat": 0, "coords": None,
           "inputs": [item.detach().clone() for item in clean_inputs]}
    deltas = [patched - clean for clean, patched in zip(clean_inputs, patched_inputs, strict=True)]
    for k in config.k_values:
        negative = _coords_from_top(top_rows, "top_negative", int(k))
        for strategy in ("top_negative", "top_positive", "top_abs"):
            coords = negative if strategy == "top_negative" else _coords_from_top(top_rows, strategy, int(k))
            repaired, l1, l2 = _repair_variant(clean_inputs, patched_inputs, coords)
            yield {"strategy": strategy, "k_requested": int(k), "repeat": 0, "coords": coords,
                   "inputs": repaired, "restored_delta_l1": l1, "restored_delta_l2": l2}
        coords = _delta_top_coords(deltas, int(target_level), int(k))
        repaired, l1, l2 = _repair_variant(clean_inputs, patched_inputs, coords)
        yield {"strategy": "delta_magnitude", "k_requested": int(k), "repeat": 0, "coords": coords,
               "inputs": repaired, "restored_delta_l1": l1, "restored_delta_l2": l2}
        for repeat in range(int(config.random_repeats)):
            rng = np.random.default_rng(int(example_seed) + 1009 * int(k) + repeat)
            coords = _random_delta_matched_coords(deltas, int(target_level), negative, rng)
            repaired, l1, l2 = _repair_variant(clean_inputs, patched_inputs, coords)
            yield {"strategy": "random_delta_matched", "k_requested": int(k), "repeat": repeat,
                   "coords": coords, "inputs": repaired, "restored_delta_l1": l1, "restored_delta_l2": l2}


def _expected_keys(config: CausalRepairConfig) -> set[tuple[str, int, int]]:
    keys = {("patched", 0, 0), ("oracle_target_level", -1, 0), ("oracle_full_head", -1, 0)}
    for k in config.k_values:
        keys.update((strategy, int(k), 0) for strategy in ("top_negative", "top_positive", "top_abs", "delta_magnitude"))
        keys.update(("random_delta_matched", int(k), repeat) for repeat in range(int(config.random_repeats)))
    return keys


def _evaluate_variants(detect, specs, row, config, nc: int):
    import torch
    from ultralytics.utils.nms import non_max_suppression

    class_id = int(row.class_id)
    reg_max = int(getattr(detect, "reg_max", 16))
    class_channel = 4 * reg_max + class_id
    target_level, target_y, target_x = int(row.clean_target_level), int(row.clean_target_y), int(row.clean_target_x)
    target_flat = int(row.clean_target_flat)
    clean_box = torch.as_tensor(
        [row.clean_target_x1, row.clean_target_y1, row.clean_target_x2, row.clean_target_y2],
        device=next(detect.parameters()).device, dtype=torch.float32,
    ).reshape(1, 4)
    out_rows = []
    for start in range(0, len(specs), max(1, int(config.forward_batch_size))):
        chunk = specs[start:start + max(1, int(config.forward_batch_size))]
        levels = [torch.cat([item["inputs"][level] for item in chunk], dim=0) for level in range(len(chunk[0]["inputs"]))]
        with torch.inference_mode():
            decoded, raw_levels = _parse_forward_output(detect(levels))
            nms = non_max_suppression(
                decoded.detach().clone(), conf_thres=float(config.nms_conf), iou_thres=float(config.nms_iou),
                classes=[class_id], max_det=int(config.nms_max_det), nc=int(nc),
            )
        fixed_boxes = _xywh_to_xyxy(decoded[:, :4, target_flat].reshape(-1, 4))
        for index, spec in enumerate(chunk):
            detections = nms[index]
            fixed_iou = float(_box_iou(fixed_boxes[index:index + 1], clean_box).reshape(-1)[0].detach().cpu())
            target_conf, target_iou, max_iou = 0.0, 0.0, 0.0
            winner_conf, winner_iou = 0.0, 0.0
            if int(detections.shape[0]) > 0:
                ious = _box_iou(detections[:, :4], clean_box).reshape(-1)
                max_iou = float(ious.max().detach().cpu())
                winner_conf = float(detections[0, 4].detach().float().cpu())
                winner_iou = float(ious[0].detach().cpu())
                valid = torch.nonzero(ious >= float(config.match_iou), as_tuple=False).reshape(-1)
                if int(valid.numel()) > 0:
                    valid_conf = detections[valid, 4]
                    chosen = int(valid[int(torch.argmax(valid_conf).item())].item())
                    target_conf = float(detections[chosen, 4].detach().float().cpu())
                    target_iou = float(ious[chosen].detach().cpu())
            detected = int(target_iou >= float(config.match_iou) and target_conf >= float(config.detection_conf))
            out_rows.append({
                "strategy": spec["strategy"], "k_requested": int(spec["k_requested"]),
                "repeat": int(spec["repeat"]),
                "actual_k": int(len(spec["coords"])) if spec.get("coords") is not None else -1,
                "restored_delta_l1": float(spec.get("restored_delta_l1", np.nan)),
                "restored_delta_l2": float(spec.get("restored_delta_l2", np.nan)),
                "fixed_target_logit": float(raw_levels[target_level][index, class_channel, target_y, target_x].detach().float().cpu()),
                "fixed_target_score": float(decoded[index, 4 + class_id, target_flat].detach().float().cpu()),
                "fixed_box_iou_clean": fixed_iou, "target_match_conf": target_conf,
                "target_match_iou": target_iou, "target_max_iou": max_iou,
                "target_detected": detected, "target_hidden": 1 - detected,
                "global_winner_conf": winner_conf, "global_winner_iou_target": winner_iou,
            })
    return out_rows


def _summary_outputs(
    conn: sqlite3.Connection, run_dir: Path, config: CausalRepairConfig,
    *, expected_examples: int, expected_rows: int,
):
    rows = pd.read_sql_query("SELECT * FROM repair_results WHERE error IS NULL", conn)
    if rows.empty:
        summary_path = write_json(run_dir / "summary.json", {"status": "empty"})
        digest_path = write_markdown(run_dir / "analysis_digest.md", ["# Causal repair", "", "No completed rows."])
        return summary_path, digest_path, run_dir / "repair_group_summary.csv"
    group_summary = (
        rows.groupby(["analysis_group", "strategy", "k_requested"], dropna=False)
        .agg(
            n=("example_id", "nunique"), n_rows=("example_id", "size"),
            mean_actual_k=("actual_k", "mean"), rescue_rate=("target_detected", "mean"),
            mean_target_logit=("fixed_target_logit", "mean"), mean_logit_gain=("target_logit_gain", "mean"),
            mean_target_conf=("target_match_conf", "mean"), mean_conf_gain=("target_conf_gain", "mean"),
            mean_target_iou=("target_match_iou", "mean"), mean_iou_gain=("target_iou_gain", "mean"),
            mean_fixed_box_iou=("fixed_box_iou_clean", "mean"),
            mean_restored_delta_l1=("restored_delta_l1", "mean"),
        ).reset_index()
    )
    group_summary_path = run_dir / "repair_group_summary.csv"
    group_summary.to_csv(group_summary_path, index=False)
    error_rows = int(conn.execute("SELECT COUNT(*) FROM repair_results WHERE error IS NOT NULL").fetchone()[0])
    completed_examples = int(rows.example_id.nunique())
    completed_rows = int(len(rows))
    payload = {
        "status": "complete" if completed_examples == int(expected_examples) and completed_rows == int(expected_rows) else "partial",
        "n_examples": completed_examples, "expected_examples": int(expected_examples),
        "n_rows": completed_rows, "expected_rows": int(expected_rows), "error_rows": error_rows,
        "group_counts": {str(k): int(v) for k, v in rows.groupby("analysis_group").example_id.nunique().items()},
        "strategies": sorted(rows.strategy.unique().tolist()), "k_values": list(config.k_values),
        "database": str(run_dir / "causal_repair.sqlite"), "config": asdict(config),
    }
    summary_path = write_json(run_dir / "summary.json", payload)
    digest_path = write_markdown(run_dir / "analysis_digest.md", [
        "# Sign-selective causal repair", "",
        f"- status: {payload['status']}",
        f"- examples: {payload['n_examples']} / {payload['expected_examples']}",
        f"- intervention rows: {payload['n_rows']} / {payload['expected_rows']}",
        f"- error rows: {payload['error_rows']}",
        f"- groups: {payload['group_counts']}", f"- doses: {payload['k_values']}",
        "", "Primary endpoint: target rescue at IoU >= 0.5 and confidence >= 0.25.",
        "Primary contrast: top_negative versus random_delta_matched and delta_magnitude.",
        "Read repair_group_summary.csv before querying SQLite.",
    ])
    return summary_path, digest_path, group_summary_path


def run_causal_repair(
    exp,
    attack_path_db: str | Path,
    trace_db: str | Path,
    manifest_csv: str | Path,
    config: CausalRepairConfig | None = None,
    *,
    force: bool = False,
) -> CausalRepairResult:
    from segmentig_detector.yolo_utils import get_detect_module

    config = config or CausalRepairConfig()
    attack_path_db, trace_db, manifest_csv = Path(attack_path_db), Path(trace_db), Path(manifest_csv)
    selected, top = _load_inputs(attack_path_db, trace_db, manifest_csv, config.max_examples)
    payload = {
        "attack_path_db": str(attack_path_db.resolve()), "attack_path_size": attack_path_db.stat().st_size,
        "trace_db": str(trace_db.resolve()), "trace_size": trace_db.stat().st_size,
        "manifest_csv": str(manifest_csv.resolve()), "manifest_size": manifest_csv.stat().st_size,
        "example_ids": selected.example_id.tolist(), **asdict(config),
    }
    run_dir = Path(config.output_dir) / f"causal_repair_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = StorageBudget(Path(config.output_dir), config.max_output_gb)
    budget.check()
    db_path = run_dir / "causal_repair.sqlite"
    if force and db_path.exists():
        db_path.unlink()
    conn = connect_db(db_path)
    _create_schema(conn)
    upsert_metadata(conn, {"config": asdict(config), "payload": payload})
    completed = {
        (str(row[0]), str(row[1]), int(row[2]), int(row[3]))
        for row in conn.execute("SELECT example_id, strategy, k_requested, repeat FROM repair_results WHERE error IS NULL")
    }
    expected_variant_keys = _expected_keys(config)

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    nc = int(getattr(detect, "nc", 80))
    cache_examples = {
        stable_hash({"path": str(item.path), "drop": float(item.drop), "success": bool(item.success)}): item
        for item in exp.get_cache().examples
    }
    progress = None
    if config.progress:
        try:
            from tqdm.auto import tqdm
            progress = tqdm(total=len(selected), desc="causal repair", unit="img")
        except Exception:
            progress = None
    try:
        for row in selected.itertuples(index=False):
            example_id = str(row.example_id)
            completed_for_example = {
                (strategy, k, repeat) for done_id, strategy, k, repeat in completed if done_id == example_id
            }
            if expected_variant_keys.issubset(completed_for_example):
                if progress is not None:
                    progress.update(1)
                continue
            example = cache_examples.get(example_id)
            if example is None:
                continue
            try:
                clean_image, patched_image, _bbox = exp._images_for_example(example)
                pair_inputs = _preprocess_pair(exp, clean_image, patched_image)
                head_inputs = _capture_detect_inputs(model, detect, pair_inputs)
                clean_inputs = [item[0:1].detach() for item in head_inputs]
                patched_inputs = [item[1:2].detach() for item in head_inputs]
                seed = int(stable_hash({"example_id": example_id, "seed": config.seed})[:12], 16)
                specs = list(_variant_specs(
                    clean_inputs, patched_inputs, top.loc[top.example_id.eq(example_id)],
                    int(row.clean_target_level), config, seed,
                ))
                specs = [spec for spec in specs if (example_id, spec["strategy"], int(spec["k_requested"]), int(spec["repeat"])) not in completed]
                if specs:
                    evaluated = _evaluate_variants(detect, specs, row, config, nc)
                    baseline_rows = [item for item in evaluated if item["strategy"] == "patched"]
                    if baseline_rows:
                        baseline = baseline_rows[0]
                    else:
                        baseline_spec = next(_variant_specs(clean_inputs, patched_inputs, top.loc[top.example_id.eq(example_id)], int(row.clean_target_level), config, seed))
                        baseline = _evaluate_variants(detect, [baseline_spec], row, config, nc)[0]
                    output_rows = []
                    for item in evaluated:
                        item.update({
                            "example_id": example_id, "analysis_group": str(row.analysis_group),
                            "match_set": int(row.match_set),
                            "target_logit_gain": item["fixed_target_logit"] - baseline["fixed_target_logit"],
                            "target_conf_gain": item["target_match_conf"] - baseline["target_match_conf"],
                            "target_iou_gain": item["target_match_iou"] - baseline["target_match_iou"],
                            "error": None,
                        })
                        output_rows.append(item)
                    _insert_rows(conn, output_rows)
                    conn.commit()
                budget.check(extra_bytes=50 * 1024**2)
            except Exception as exc:  # noqa: BLE001
                error_row = {column: None for column in REPAIR_COLUMNS}
                error_row.update({
                    "example_id": example_id, "analysis_group": str(row.analysis_group),
                    "match_set": int(row.match_set), "strategy": "error", "k_requested": -999,
                    "repeat": 0, "error": f"{type(exc).__name__}: {exc}",
                })
                _insert_rows(conn, [error_row])
                conn.commit()
            finally:
                if progress is not None:
                    progress.update(1)
                release_accelerator_memory()
    finally:
        if progress is not None:
            progress.close()
        expected_rows = len(selected) * len(expected_variant_keys)
        summary_path, digest_path, group_summary_path = _summary_outputs(
            conn, run_dir, config, expected_examples=len(selected), expected_rows=expected_rows,
        )
        conn.close()
    return CausalRepairResult(run_dir, db_path, summary_path, digest_path, group_summary_path, config)


def load_causal_repair(result_or_db: CausalRepairResult | str | Path) -> pd.DataFrame:
    db_path = result_or_db.db_path if isinstance(result_or_db, CausalRepairResult) else Path(result_or_db)
    conn = connect_db(db_path)
    try:
        return pd.read_sql_query("SELECT * FROM repair_results WHERE error IS NULL", conn)
    finally:
        conn.close()
