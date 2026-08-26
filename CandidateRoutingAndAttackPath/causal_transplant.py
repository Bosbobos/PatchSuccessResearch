from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .causal_repair import (
    _coords_from_top,
    _delta_top_coords,
    _evaluate_variants,
    _load_inputs,
    _random_delta_matched_coords,
)
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
class CausalTransplantConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    max_output_gb: float = DEFAULT_MAX_OUTPUT_GB
    max_examples: int | None = None
    k_values: tuple[int, ...] = (10, 50, 100, 250)
    random_repeats: int = 3
    forward_batch_size: int = 8
    detection_conf: float = 0.25
    match_iou: float = 0.50
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    seed: int = 71
    progress: bool = True
    method_version: int = 1


@dataclass(slots=True)
class CausalTransplantResult:
    run_dir: Path
    db_path: Path
    summary_path: Path
    digest_path: Path
    group_summary_path: Path
    pairwise_path: Path
    config: CausalTransplantConfig


TRANSPLANT_COLUMNS = (
    "example_id", "analysis_group", "match_set", "strategy", "k_requested", "repeat",
    "actual_k", "transplanted_delta_l1", "transplanted_delta_l2", "fixed_target_logit",
    "fixed_target_score", "fixed_box_iou_clean", "target_match_conf", "target_match_iou",
    "target_max_iou", "target_detected", "target_hidden", "attack_success",
    "global_winner_conf", "global_winner_iou_target", "target_logit_loss", "target_conf_loss",
    "target_iou_loss", "fixed_box_iou_loss", "error",
)


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transplant_results (
            example_id TEXT, analysis_group TEXT, match_set INTEGER, strategy TEXT,
            k_requested INTEGER, repeat INTEGER, actual_k INTEGER,
            transplanted_delta_l1 REAL, transplanted_delta_l2 REAL,
            fixed_target_logit REAL, fixed_target_score REAL, fixed_box_iou_clean REAL,
            target_match_conf REAL, target_match_iou REAL, target_max_iou REAL,
            target_detected INTEGER, target_hidden INTEGER, attack_success INTEGER,
            global_winner_conf REAL, global_winner_iou_target REAL,
            target_logit_loss REAL, target_conf_loss REAL, target_iou_loss REAL,
            fixed_box_iou_loss REAL, error TEXT,
            PRIMARY KEY(example_id, strategy, k_requested, repeat)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_transplant_group "
        "ON transplant_results(analysis_group, strategy, k_requested)"
    )
    conn.commit()


def _insert_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    placeholders = ",".join("?" for _ in TRANSPLANT_COLUMNS)
    sql = (
        f"INSERT OR REPLACE INTO transplant_results ({','.join(TRANSPLANT_COLUMNS)}) "
        f"VALUES ({placeholders})"
    )
    conn.executemany(sql, [[row.get(column) for column in TRANSPLANT_COLUMNS] for row in rows])


def _transplant_variant(clean_inputs, patched_inputs, coords):
    transplanted = [item.detach().clone() for item in clean_inputs]
    moved = []
    for level, channel, y, x in coords:
        delta = patched_inputs[level][0, channel, y, x] - clean_inputs[level][0, channel, y, x]
        moved.append(float(delta.detach().float().cpu()))
        transplanted[level][0, channel, y, x] = patched_inputs[level][0, channel, y, x]
    array = np.asarray(moved, dtype=np.float64)
    return transplanted, float(np.abs(array).sum()), float(np.sqrt(np.square(array).sum()))


def _variant_specs(clean_inputs, patched_inputs, top_rows, target_level, config, example_seed):
    yield {"strategy": "clean", "k_requested": 0, "repeat": 0, "coords": [], "inputs": clean_inputs}
    yield {
        "strategy": "patched_target_level", "k_requested": -1, "repeat": 0, "coords": None,
        "inputs": [
            patched.detach().clone() if idx == int(target_level) else clean.detach().clone()
            for idx, (clean, patched) in enumerate(zip(clean_inputs, patched_inputs, strict=True))
        ],
    }
    yield {
        "strategy": "patched_full_head", "k_requested": -1, "repeat": 0, "coords": None,
        "inputs": [item.detach().clone() for item in patched_inputs],
    }
    deltas = [patched - clean for clean, patched in zip(clean_inputs, patched_inputs, strict=True)]
    for k in config.k_values:
        negative = _coords_from_top(top_rows, "top_negative", int(k))
        for strategy in ("top_negative", "top_positive", "top_abs"):
            coords = negative if strategy == "top_negative" else _coords_from_top(top_rows, strategy, int(k))
            inputs, l1, l2 = _transplant_variant(clean_inputs, patched_inputs, coords)
            yield {
                "strategy": strategy, "k_requested": int(k), "repeat": 0, "coords": coords,
                "inputs": inputs, "restored_delta_l1": l1, "restored_delta_l2": l2,
            }
        coords = _delta_top_coords(deltas, int(target_level), int(k))
        inputs, l1, l2 = _transplant_variant(clean_inputs, patched_inputs, coords)
        yield {
            "strategy": "delta_magnitude", "k_requested": int(k), "repeat": 0, "coords": coords,
            "inputs": inputs, "restored_delta_l1": l1, "restored_delta_l2": l2,
        }
        for repeat in range(int(config.random_repeats)):
            rng = np.random.default_rng(int(example_seed) + 1009 * int(k) + repeat)
            coords = _random_delta_matched_coords(deltas, int(target_level), negative, rng)
            inputs, l1, l2 = _transplant_variant(clean_inputs, patched_inputs, coords)
            yield {
                "strategy": "random_delta_matched", "k_requested": int(k), "repeat": repeat,
                "coords": coords, "inputs": inputs,
                "restored_delta_l1": l1, "restored_delta_l2": l2,
            }


def _expected_keys(config: CausalTransplantConfig) -> set[tuple[str, int, int]]:
    keys = {("clean", 0, 0), ("patched_target_level", -1, 0), ("patched_full_head", -1, 0)}
    for k in config.k_values:
        keys.update(
            (strategy, int(k), 0)
            for strategy in ("top_negative", "top_positive", "top_abs", "delta_magnitude")
        )
        keys.update(
            ("random_delta_matched", int(k), repeat)
            for repeat in range(int(config.random_repeats))
        )
    return keys


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(np.where(np.isfinite(p_values), p_values, np.inf))
    adjusted = np.full(len(p_values), np.nan)
    running = 0.0
    finite_n = int(np.isfinite(p_values).sum())
    for rank, index in enumerate(order[:finite_n]):
        running = max(running, (finite_n - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def _pairwise_summary(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    unit = rows.groupby(
        ["example_id", "analysis_group", "strategy", "k_requested"], as_index=False
    ).agg(
        target_logit_loss=("target_logit_loss", "mean"),
        target_conf_loss=("target_conf_loss", "mean"),
        target_iou_loss=("target_iou_loss", "mean"),
        fixed_box_iou_loss=("fixed_box_iou_loss", "mean"),
        attack_success=("attack_success", "mean"),
    )
    controls = ("top_abs", "top_positive", "delta_magnitude", "random_delta_matched")
    metrics = (
        "target_logit_loss", "target_conf_loss", "target_iou_loss",
        "fixed_box_iou_loss", "attack_success",
    )
    output = []
    for group in sorted(unit.analysis_group.unique()):
        for k in sorted(value for value in unit.k_requested.unique() if value > 0):
            primary = unit[(unit.analysis_group == group) & (unit.k_requested == k) & (unit.strategy == "top_negative")]
            for control in controls:
                other = unit[(unit.analysis_group == group) & (unit.k_requested == k) & (unit.strategy == control)]
                for metric in metrics:
                    paired = primary[["example_id", metric]].merge(
                        other[["example_id", metric]], on="example_id", suffixes=("_primary", "_control")
                    ).dropna()
                    difference = paired[f"{metric}_primary"].to_numpy() - paired[f"{metric}_control"].to_numpy()
                    if not len(difference):
                        continue
                    if np.allclose(difference, 0):
                        p_value = 1.0
                    else:
                        p_value = float(wilcoxon(difference, zero_method="zsplit").pvalue)
                    rng = np.random.default_rng(
                        int(seed) + int(k) + int(stable_hash({"group": group, "control": control, "metric": metric})[:8], 16)
                    )
                    boot = np.asarray([
                        rng.choice(difference, size=len(difference), replace=True).mean() for _ in range(2000)
                    ])
                    output.append({
                        "analysis_group": group, "k_requested": int(k), "control": control,
                        "metric": metric, "n": int(len(difference)),
                        "primary_mean": float(paired[f"{metric}_primary"].mean()),
                        "control_mean": float(paired[f"{metric}_control"].mean()),
                        "mean_difference": float(difference.mean()),
                        "median_difference": float(np.median(difference)),
                        "ci95_low": float(np.quantile(boot, 0.025)),
                        "ci95_high": float(np.quantile(boot, 0.975)),
                        "p_value": p_value,
                    })
    result = pd.DataFrame(output)
    if not result.empty:
        result["p_holm"] = _holm_adjust(result.p_value.to_numpy())
    return result


def _summary_outputs(
    conn: sqlite3.Connection,
    run_dir: Path,
    config: CausalTransplantConfig,
    *,
    expected_examples: int,
    expected_rows: int,
):
    rows = pd.read_sql_query("SELECT * FROM transplant_results WHERE error IS NULL", conn)
    group_summary_path = run_dir / "transplant_group_summary.csv"
    pairwise_path = run_dir / "transplant_pairwise.csv"
    if rows.empty:
        pd.DataFrame().to_csv(group_summary_path, index=False)
        pd.DataFrame().to_csv(pairwise_path, index=False)
        summary_path = write_json(run_dir / "summary.json", {"status": "empty"})
        digest_path = write_markdown(run_dir / "analysis_digest.md", ["# Causal transplant", "", "No completed rows."])
        return summary_path, digest_path, group_summary_path, pairwise_path
    group_summary = rows.groupby(
        ["analysis_group", "strategy", "k_requested"], dropna=False
    ).agg(
        n=("example_id", "nunique"), n_rows=("example_id", "size"),
        mean_actual_k=("actual_k", "mean"), attack_success_rate=("attack_success", "mean"),
        mean_target_logit=("fixed_target_logit", "mean"), mean_logit_loss=("target_logit_loss", "mean"),
        mean_target_conf=("target_match_conf", "mean"), mean_conf_loss=("target_conf_loss", "mean"),
        mean_target_max_iou=("target_max_iou", "mean"), mean_iou_loss=("target_iou_loss", "mean"),
        mean_fixed_box_iou=("fixed_box_iou_clean", "mean"),
        mean_fixed_box_iou_loss=("fixed_box_iou_loss", "mean"),
        mean_transplanted_delta_l1=("transplanted_delta_l1", "mean"),
    ).reset_index()
    group_summary.to_csv(group_summary_path, index=False)
    pairwise = _pairwise_summary(rows, config.seed)
    pairwise.to_csv(pairwise_path, index=False)
    error_rows = int(conn.execute("SELECT COUNT(*) FROM transplant_results WHERE error IS NOT NULL").fetchone()[0])
    completed_examples = int(rows.example_id.nunique())
    completed_rows = int(len(rows))
    payload = {
        "status": "complete" if completed_examples == expected_examples and completed_rows == expected_rows else "partial",
        "n_examples": completed_examples, "expected_examples": int(expected_examples),
        "n_rows": completed_rows, "expected_rows": int(expected_rows), "error_rows": error_rows,
        "group_counts": {str(k): int(v) for k, v in rows.groupby("analysis_group").example_id.nunique().items()},
        "strategies": sorted(rows.strategy.unique().tolist()), "k_values": list(config.k_values),
        "database": str(run_dir / "causal_transplant.sqlite"), "config": asdict(config),
    }
    summary_path = write_json(run_dir / "summary.json", payload)
    digest_path = write_markdown(run_dir / "analysis_digest.md", [
        "# Signed causal transplant", "",
        f"- status: {payload['status']}",
        f"- examples: {payload['n_examples']} / {payload['expected_examples']}",
        f"- intervention rows: {payload['n_rows']} / {payload['expected_rows']}",
        f"- error rows: {payload['error_rows']}",
        f"- groups: {payload['group_counts']}",
        f"- doses: {payload['k_values']}", "",
        "Primary endpoint: reproduction of target hiding after transplanting patched values into clean Detect inputs.",
        "Primary contrast: top_negative versus top_abs, delta_magnitude, random_delta_matched, and top_positive.",
        "Positive pairwise differences mean that top_negative reproduces more of the attack.",
        "Read transplant_group_summary.csv and transplant_pairwise.csv before querying SQLite.",
    ])
    return summary_path, digest_path, group_summary_path, pairwise_path


def run_causal_transplant(
    exp,
    attack_path_db: str | Path,
    trace_db: str | Path,
    manifest_csv: str | Path,
    config: CausalTransplantConfig | None = None,
    *,
    force: bool = False,
) -> CausalTransplantResult:
    from segmentig_detector.yolo_utils import get_detect_module

    config = config or CausalTransplantConfig()
    attack_path_db, trace_db, manifest_csv = Path(attack_path_db), Path(trace_db), Path(manifest_csv)
    selected, top = _load_inputs(attack_path_db, trace_db, manifest_csv, config.max_examples)
    payload = {
        "attack_path_db": str(attack_path_db.resolve()), "attack_path_size": attack_path_db.stat().st_size,
        "trace_db": str(trace_db.resolve()), "trace_size": trace_db.stat().st_size,
        "manifest_csv": str(manifest_csv.resolve()), "manifest_size": manifest_csv.stat().st_size,
        "example_ids": selected.example_id.tolist(), **asdict(config),
    }
    run_dir = Path(config.output_dir) / f"causal_transplant_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = StorageBudget(Path(config.output_dir), config.max_output_gb)
    budget.check()
    db_path = run_dir / "causal_transplant.sqlite"
    if force and db_path.exists():
        db_path.unlink()
    conn = connect_db(db_path)
    _create_schema(conn)
    upsert_metadata(conn, {"config": asdict(config), "payload": payload})
    completed = {
        (str(row[0]), str(row[1]), int(row[2]), int(row[3]))
        for row in conn.execute(
            "SELECT example_id, strategy, k_requested, repeat FROM transplant_results WHERE error IS NULL"
        )
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
            progress = tqdm(total=len(selected), desc="causal transplant", unit="img")
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
                specs = [
                    spec for spec in specs
                    if (example_id, spec["strategy"], int(spec["k_requested"]), int(spec["repeat"])) not in completed
                ]
                if specs:
                    evaluated = _evaluate_variants(detect, specs, row, config, nc)
                    baseline_rows = [item for item in evaluated if item["strategy"] == "clean"]
                    if baseline_rows:
                        baseline = baseline_rows[0]
                    else:
                        baseline_spec = next(_variant_specs(
                            clean_inputs, patched_inputs, top.loc[top.example_id.eq(example_id)],
                            int(row.clean_target_level), config, seed,
                        ))
                        baseline = _evaluate_variants(detect, [baseline_spec], row, config, nc)[0]
                    output_rows = []
                    for item in evaluated:
                        item["transplanted_delta_l1"] = item.pop("restored_delta_l1")
                        item["transplanted_delta_l2"] = item.pop("restored_delta_l2")
                        item.update({
                            "example_id": example_id, "analysis_group": str(row.analysis_group),
                            "match_set": int(row.match_set), "attack_success": int(item["target_hidden"]),
                            "target_logit_loss": baseline["fixed_target_logit"] - item["fixed_target_logit"],
                            "target_conf_loss": baseline["target_match_conf"] - item["target_match_conf"],
                            "target_iou_loss": baseline["target_max_iou"] - item["target_max_iou"],
                            "fixed_box_iou_loss": baseline["fixed_box_iou_clean"] - item["fixed_box_iou_clean"],
                            "error": None,
                        })
                        output_rows.append(item)
                    _insert_rows(conn, output_rows)
                    conn.commit()
                budget.check(extra_bytes=50 * 1024**2)
            except Exception as exc:  # noqa: BLE001
                error_row = {column: None for column in TRANSPLANT_COLUMNS}
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
        summary_path, digest_path, group_summary_path, pairwise_path = _summary_outputs(
            conn, run_dir, config, expected_examples=len(selected), expected_rows=expected_rows,
        )
        conn.close()
    return CausalTransplantResult(
        run_dir, db_path, summary_path, digest_path, group_summary_path, pairwise_path, config
    )


def load_causal_transplant(result_or_db: CausalTransplantResult | str | Path) -> pd.DataFrame:
    db_path = result_or_db.db_path if isinstance(result_or_db, CausalTransplantResult) else Path(result_or_db)
    conn = connect_db(db_path)
    try:
        return pd.read_sql_query("SELECT * FROM transplant_results WHERE error IS NULL", conn)
    finally:
        conn.close()
