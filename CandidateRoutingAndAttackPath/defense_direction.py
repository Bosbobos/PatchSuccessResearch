from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .common import stable_hash
from .followup_common import FOLLOWUP_DIR, MANIFEST_CSV, TRACE_DB, box_iou_numpy, write_summary


TARGET_LABELS = (
    TRACE_DB.parents[1]
    / "target_instance_all_metrics"
    / "target_instance_bc9b647edb5cdec7"
    / "target_instance_labels.csv"
)


@dataclass(slots=True)
class RoutePoolConfig:
    output_dir: str = str(FOLLOWUP_DIR)
    balanced_only: bool = True
    cluster_iou: float = 0.70
    max_cluster_members: int = 8
    detection_conf: float = 0.25
    target_iou: float = 0.50
    train_fraction: float = 0.60
    seed: int = 151
    strategies: tuple[str, ...] = ("max_box", "max_wbf_box", "mean_odds_wbf", "noisy_or_wbf")
    method_version: int = 1


def _split(example_id: str, config: RoutePoolConfig) -> str:
    value = int(stable_hash({"example_id": str(example_id), "seed": int(config.seed)}, length=12), 16)
    return "train" if (value % 10_000) < int(float(config.train_fraction) * 10_000) else "test"


def _greedy_clusters(frame: pd.DataFrame, config: RoutePoolConfig):
    if frame.empty:
        return []
    ordered = frame.sort_values("decoded_score", ascending=False).reset_index(drop=True)
    boxes = ordered[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)
    scores = ordered.decoded_score.to_numpy(float)
    remaining = list(range(len(ordered)))
    clusters = []
    while remaining:
        seed = remaining[0]
        ious = box_iou_numpy(boxes[seed:seed + 1], boxes[remaining]).reshape(-1)
        all_members = [remaining[index] for index, value in enumerate(ious) if value >= float(config.cluster_iou)]
        members = sorted(all_members, key=lambda index: scores[index], reverse=True)[: int(config.max_cluster_members)]
        member_set = set(all_members)
        remaining = [index for index in remaining if index not in member_set]
        clusters.append((members, boxes[members], scores[members]))
    return clusters


def _pool_cluster(members, boxes, scores, strategy: str):
    scores = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    boxes = np.asarray(boxes, dtype=float)
    weights = scores / max(float(scores.sum()), 1e-12)
    weighted_box = np.sum(boxes * weights[:, None], axis=0)
    if strategy == "max_box":
        return float(scores[0]), boxes[0]
    if strategy == "max_wbf_box":
        return float(scores[0]), weighted_box
    if strategy == "mean_odds_wbf":
        odds = scores / (1.0 - scores)
        mean_odds = float(np.mean(odds))
        return mean_odds / (1.0 + mean_odds), weighted_box
    if strategy == "noisy_or_wbf":
        return float(1.0 - np.prod(1.0 - scores)), weighted_box
    raise ValueError(strategy)


def _load_frames(config: RoutePoolConfig | None = None):
    config = config or RoutePoolConfig()
    conn = sqlite3.connect(TRACE_DB)
    candidates = pd.read_sql_query(
        "SELECT example_id,variant,rank,decoded_score,bbox_x1,bbox_y1,bbox_x2,bbox_y2,"
        "nms_survived,nms_rank,nms_conf "
        "FROM candidates ORDER BY example_id,variant,rank",
        conn,
    )
    examples = pd.read_sql_query(
        "SELECT example_id,clean_target_x1,clean_target_y1,clean_target_x2,clean_target_y2,error FROM examples",
        conn,
    )
    conn.close()
    labels = pd.read_csv(TARGET_LABELS)
    labels = labels[[
        "example_id", "target_eligible", "target_detected", "target_hidden", "outcome",
        "target_match_iou", "target_patched_conf", "patched_winner_is_target",
    ]]
    manifest = pd.read_csv(MANIFEST_CSV)[["example_id", "analysis_group", "match_set"]]
    examples = examples[(examples.error.isna()) & examples.example_id.isin(labels[labels.target_eligible == 1].example_id)].copy()
    examples = examples.merge(labels, on="example_id", how="left", validate="one_to_one")
    examples = examples.merge(manifest, on="example_id", how="left", validate="one_to_one")
    derived_group = np.select(
        [
            (examples.target_hidden == 1) & (examples.target_match_iou >= 0.5),
            (examples.target_hidden == 1) & (examples.target_match_iou < 0.5),
            (examples.target_hidden == 0) & (examples.patched_winner_is_target == 1),
            (examples.target_hidden == 0) & (examples.patched_winner_is_target != 1),
        ],
        [
            "hidden_low_conf_match", "hidden_no_iou_match",
            "visible_target_winner", "visible_non_target_winner",
        ],
        default=None,
    )
    examples["analysis_group"] = examples.analysis_group.fillna(pd.Series(derived_group, index=examples.index))
    if config.balanced_only:
        examples = examples[examples.analysis_group.notna()].copy()
        candidates = candidates[candidates.example_id.isin(examples.example_id)].copy()
    return candidates, examples


def _build_cluster_rows(candidates: pd.DataFrame, examples: pd.DataFrame, config: RoutePoolConfig) -> pd.DataFrame:
    target_boxes = examples.set_index("example_id")[[
        "clean_target_x1", "clean_target_y1", "clean_target_x2", "clean_target_y2"
    ]].to_dict("index")
    rows = []
    for (example_id, variant), frame in candidates.groupby(["example_id", "variant"], sort=False):
        if example_id not in target_boxes:
            continue
        target = np.asarray(list(target_boxes[example_id].values()), dtype=float).reshape(1, 4)
        for cluster_index, (members, boxes, scores) in enumerate(_greedy_clusters(frame, config)):
            for strategy in config.strategies:
                pooled_score, pooled_box = _pool_cluster(members, boxes, scores, strategy)
                rows.append({
                    "example_id": example_id, "variant": variant, "cluster_index": cluster_index,
                    "strategy": strategy, "member_count": len(members), "pooled_score": pooled_score,
                    "bbox_x1": pooled_box[0], "bbox_y1": pooled_box[1],
                    "bbox_x2": pooled_box[2], "bbox_y2": pooled_box[3],
                    "target_iou": float(box_iou_numpy(pooled_box.reshape(1, 4), target)[0, 0]),
                })
    return pd.DataFrame(rows)


def _calibrate(cluster_rows: pd.DataFrame, candidates: pd.DataFrame, examples: pd.DataFrame, config: RoutePoolConfig):
    train_ids = set(examples.loc[examples.split == "train", "example_id"])
    baseline = candidates[
        (candidates.example_id.isin(train_ids))
        & (candidates.variant == "clean")
        & (candidates.nms_survived == 1)
        & (candidates.decoded_score >= float(config.detection_conf))
    ]
    desired_count = int(baseline[["example_id", "nms_rank"]].drop_duplicates().shape[0])
    thresholds = []
    for strategy in config.strategies:
        scores = cluster_rows[
            (cluster_rows.example_id.isin(train_ids))
            & (cluster_rows.variant == "clean")
            & (cluster_rows.strategy == strategy)
        ].pooled_score.to_numpy(float)
        if not len(scores):
            threshold = np.nan
        elif desired_count <= 0:
            threshold = 1.0
        else:
            rank = min(desired_count, len(scores))
            threshold = float(np.partition(scores, len(scores) - rank)[len(scores) - rank])
        thresholds.append({
            "strategy": strategy, "threshold": threshold,
            "desired_clean_train_output_count": desired_count,
            "available_clean_train_clusters": int(len(scores)),
        })
    return pd.DataFrame(thresholds)


def _evaluate(cluster_rows, thresholds, candidates, examples, config):
    threshold_map = thresholds.set_index("strategy").threshold.to_dict()
    example_meta = examples.set_index("example_id")
    output = []
    for (example_id, variant, strategy), frame in cluster_rows.groupby(["example_id", "variant", "strategy"]):
        threshold = float(threshold_map[strategy])
        kept = frame[frame.pooled_score >= threshold]
        detected = int((kept.target_iou >= float(config.target_iou)).any())
        meta = example_meta.loc[example_id]
        output.append({
            "example_id": example_id, "split": meta.split, "variant": variant, "strategy": strategy,
            "threshold": threshold, "output_count": len(kept), "target_detected": detected,
            "target_hidden": 1 - detected, "best_target_iou": float(kept.target_iou.max()) if len(kept) else 0.0,
            "best_target_score": float(kept.loc[kept.target_iou.idxmax(), "pooled_score"]) if len(kept) else 0.0,
            "baseline_patched_target_detected": int(meta.target_detected),
            "baseline_patched_target_hidden": int(meta.target_hidden),
            "analysis_group": meta.analysis_group, "match_set": meta.match_set,
        })
    evaluated = pd.DataFrame(output)
    baseline_counts = candidates[
        (candidates.nms_survived == 1) & (candidates.decoded_score >= float(config.detection_conf))
    ].groupby(["example_id", "variant"]).nms_rank.nunique().rename("baseline_output_count").reset_index()
    evaluated = evaluated.merge(baseline_counts, on=["example_id", "variant"], how="left")
    evaluated["baseline_output_count"] = evaluated.baseline_output_count.fillna(0)
    return evaluated


def _summaries(evaluated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = evaluated[evaluated.split == "test"].copy()
    overall = test.groupby(["variant", "strategy"], as_index=False).agg(
        n=("example_id", "nunique"), target_detection_rate=("target_detected", "mean"),
        mean_output_count=("output_count", "mean"), mean_baseline_output_count=("baseline_output_count", "mean"),
        mean_best_target_iou=("best_target_iou", "mean"),
    )
    balanced = test[test.analysis_group.notna()].groupby(
        ["analysis_group", "variant", "strategy"], as_index=False
    ).agg(
        n=("example_id", "nunique"), target_detection_rate=("target_detected", "mean"),
        mean_output_count=("output_count", "mean"), mean_baseline_output_count=("baseline_output_count", "mean"),
        mean_best_target_iou=("best_target_iou", "mean"),
    )
    return overall, balanced


def run_route_pool(config: RoutePoolConfig | None = None) -> Path:
    config = config or RoutePoolConfig()
    started = time.time()
    candidates, examples = _load_frames(config)
    examples["split"] = examples.example_id.map(lambda value: _split(value, config))
    candidates = candidates[candidates.example_id.isin(examples.example_id)].copy()
    cluster_rows = _build_cluster_rows(candidates, examples, config)
    thresholds = _calibrate(cluster_rows, candidates, examples, config)
    evaluated = _evaluate(cluster_rows, thresholds, candidates, examples, config)
    overall, balanced = _summaries(evaluated)
    payload = asdict(config)
    run_dir = Path(config.output_dir) / f"route_pool_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    thresholds.to_csv(run_dir / "route_pool_thresholds.csv", index=False)
    evaluated.to_csv(run_dir / "route_pool_example_results.csv", index=False)
    overall.to_csv(run_dir / "route_pool_overall_summary.csv", index=False)
    balanced.to_csv(run_dir / "route_pool_balanced_summary.csv", index=False)
    elapsed = time.time() - started
    hidden_groups = balanced[
        (balanced.variant == "patched") & balanced.analysis_group.str.startswith("hidden", na=False)
    ]
    hidden_recovery = hidden_groups.groupby("strategy").target_detection_rate.mean().to_dict()
    clean = overall[overall.variant == "clean"].set_index("strategy")
    clean_detection = clean.target_detection_rate.to_dict()
    clean_count_ratio = (clean.mean_output_count / clean.mean_baseline_output_count.replace(0, np.nan)).to_dict()
    write_summary(run_dir / "summary.json", {
        "status": "complete", "elapsed_seconds": elapsed,
        "n_examples": int(examples.example_id.nunique()),
        "hidden_group_recovery": hidden_recovery,
        "clean_target_detection": clean_detection,
        "clean_output_count_ratio": clean_count_ratio,
        "config": asdict(config),
        "limitations": [
            "Only the top-50 person candidates saved by candidate tracing are available.",
            "Thresholds match clean output volume, but full COCO ground-truth false-positive AP is not available here.",
            "This is a post-hoc direction-selection experiment, not a certified defense evaluation.",
        ],
    })
    (run_dir / "analysis_digest.md").write_text(
        "# RoutePool direction-selection experiment\n\n"
        f"- elapsed: {elapsed:.1f} s\n- examples: {examples.example_id.nunique()}\n"
        f"- hidden-group recovery: `{hidden_recovery}`\n"
        f"- clean target detection: `{clean_detection}`\n"
        f"- clean output-count ratio: `{clean_count_ratio}`\n\n"
        "Thresholds are calibrated on the train split to match the number of clean post-NMS outputs.\n"
        "This experiment cannot replace a full COCO AP and adaptive-attack evaluation.\n",
        encoding="utf-8",
    )
    return run_dir


if __name__ == "__main__":
    print(run_route_pool())
