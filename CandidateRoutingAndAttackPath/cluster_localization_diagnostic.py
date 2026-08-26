from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .autonomous_negative_repair import _clusters_from_frame, _target_overlap
from .candidate_reserve import _cache_lookup
from .causal_repair import _load_inputs
from .common import StorageBudget, load_experiment, release_accelerator_memory, stable_hash
from .followup_common import (
    ATTACK_PATH_DB,
    FOLLOWUP_DIR,
    MANIFEST_CSV,
    TRACE_DB,
    balanced_subset,
    write_summary,
)
from .improved_component_defense import (
    ImprovedComponentDefenseConfig,
    _prefilter,
    _proposal_frame,
)
from .mechanism_followup import _head_branches
from .single_forward_component import _split_reference_evaluation


@dataclass(slots=True)
class ClusterLocalizationDiagnosticConfig(ImprovedComponentDefenseConfig):
    proposal_top_ks: tuple[int, ...] = (1000, 2200, 4000, 8400)
    cluster_top_ks: tuple[int, ...] = (1000, 2200)
    cluster_member_limits: tuple[int, ...] = (20, 100)
    diagnostic_min_score: float = 1e-8
    method_version: int = 1


def _target_candidate_stats(frame: pd.DataFrame, row) -> dict:
    boxes = frame[["x1", "y1", "x2", "y2"]].to_numpy(float)
    target = np.asarray([[
        row.clean_target_x1,
        row.clean_target_y1,
        row.clean_target_x2,
        row.clean_target_y2,
    ]], dtype=float)
    from .followup_common import box_iou_numpy

    ious = box_iou_numpy(target, boxes).reshape(-1)
    best = int(np.argmax(ious))
    matching = np.flatnonzero(ious >= 0.50)
    result = {
        "best_iou": float(ious[best]),
        "best_iou_score": float(frame.iloc[best].score),
        "best_iou_score_rank": int(best + 1),
        "best_iou_flat_index": int(frame.iloc[best].flat_index),
        "matching_candidate_n": int(len(matching)),
    }
    if len(matching):
        best_matching = matching[np.argmax(frame.iloc[matching].score.to_numpy(float))]
        result.update({
            "best_matching_score": float(frame.iloc[best_matching].score),
            "best_matching_score_rank": int(best_matching + 1),
            "best_matching_iou": float(ious[best_matching]),
        })
    else:
        result.update({
            "best_matching_score": np.nan,
            "best_matching_score_rank": np.nan,
            "best_matching_iou": np.nan,
        })
    return result


def _cluster_config(
    base: ClusterLocalizationDiagnosticConfig,
    top_k: int,
    members: int,
):
    return SimpleNamespace(
        cluster_min_score=float(base.diagnostic_min_score),
        cluster_candidate_limit=int(top_k),
        cluster_iou=float(base.cluster_iou),
        max_cluster_members=int(members),
        prefilter_per_strategy=int(base.prefilter_per_strategy),
    )


def run_cluster_localization_diagnostic(
    config: ClusterLocalizationDiagnosticConfig | None = None,
) -> Path:
    config = config or ClusterLocalizationDiagnosticConfig()
    started = time.time()
    StorageBudget(config.output_dir, config.max_output_gb).check()
    selected, _ = _load_inputs(
        Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None
    )
    selected = balanced_subset(
        selected, int(config.examples_per_group), seed=int(config.seed)
    )
    _reference, evaluation = _split_reference_evaluation(selected, config)

    previous = Path(config.output_dir) / (
        "improved_component_defense_90a24c45c5b058b6"
    )
    repairs = pd.read_csv(previous / "improved_repair_rows.csv")
    hidden_ids = set(
        repairs[
            repairs.input_kind.eq("patched")
            & repairs.condition.eq("observed")
            & repairs.source_hidden.astype(bool)
        ].example_id.astype(str)
    )
    evaluation = evaluation[
        evaluation.example_id.astype(str).isin(hidden_ids)
    ].copy()

    exp, cache_path = load_experiment(
        prefer_device=config.device, require_device=bool(config.require_device)
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    cache = _cache_lookup(exp)

    candidate_rows: list[dict] = []
    cluster_rows: list[dict] = []
    cluster_detail_rows: list[dict] = []
    full_config = ClusterLocalizationDiagnosticConfig(**asdict(config))
    full_config.person_top_k = max(config.proposal_top_ks)
    full_config.cluster_candidate_limit = max(config.proposal_top_ks)
    full_config.cluster_min_score = config.diagnostic_min_score

    for row in tqdm(
        evaluation.itertuples(index=False),
        total=len(evaluation),
        desc="cluster localization diagnostic",
        unit="hidden image",
    ):
        example_id = str(row.example_id)
        example = cache[example_id]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        endpoint = _capture_detect_inputs(model, detect, pair[1:2])
        import torch

        with torch.inference_mode():
            _box, _cls, raw = _head_branches(detect, endpoint)
            frame = _proposal_frame(
                detect, raw, int(row.class_id), "person", full_config
            )

        stats = _target_candidate_stats(frame, row)
        for top_k in config.proposal_top_ks:
            subset = frame.head(int(top_k))
            overlap = _target_overlap(subset, row)
            candidate_rows.append({
                "example_id": example_id,
                "analysis_group": str(row.analysis_group),
                "proposal_top_k": int(top_k),
                "target_candidate_available": int(overlap >= config.target_iou),
                "max_target_iou_in_top_k": float(overlap),
                **stats,
            })

        for top_k in config.cluster_top_ks:
            for members in config.cluster_member_limits:
                local = _cluster_config(config, top_k, members)
                clusters = _clusters_from_frame(frame, local)
                finalists = _prefilter(clusters, "person", local)
                finalist_keys = {
                    tuple(sorted(item["selection"].flat_index.astype(int)))
                    for item in finalists
                }
                target_any = [
                    _target_overlap(item["selection"], row) >= config.target_iou
                    for item in clusters
                ]
                target_final = [
                    _target_overlap(item["selection"], row) >= config.target_iou
                    for item in finalists
                ]
                chosen = max(
                    finalists,
                    key=lambda item: item["reserve_tension"],
                )
                chosen_target = (
                    _target_overlap(chosen["selection"], row) >= config.target_iou
                )
                for cluster_index, item in enumerate(clusters):
                    selection = item["selection"]
                    scores = selection.score.to_numpy(float)
                    boxes = selection[["x1", "y1", "x2", "y2"]].to_numpy(float)
                    widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6)
                    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6)
                    key = tuple(
                        sorted(selection.flat_index.astype(int))
                    )
                    cluster_detail_rows.append({
                        "example_id": example_id,
                        "analysis_group": str(row.analysis_group),
                        "cluster_top_k": int(top_k),
                        "max_cluster_members": int(members),
                        "cluster_index": int(cluster_index),
                        "is_finalist": int(key in finalist_keys),
                        "is_target_cluster": int(
                            _target_overlap(selection, row) >= config.target_iou
                        ),
                        "target_iou": float(_target_overlap(selection, row)),
                        "n_members": int(item["n_members"]),
                        "n_levels": int(selection.level_index.nunique()),
                        "max_score": float(item["max_score"]),
                        "mean_score": float(scores.mean()),
                        "median_score": float(np.median(scores)),
                        "score_sum": float(scores.sum()),
                        "noisy_or": float(item["noisy_or"]),
                        "reserve_tension": float(item["reserve_tension"]),
                        "median_box_area": float(np.median(widths * heights)),
                        "median_aspect_hw": float(np.median(heights / widths)),
                        "center_x_std": float(
                            np.std((boxes[:, 0] + boxes[:, 2]) * 0.5)
                        ),
                        "center_y_std": float(
                            np.std((boxes[:, 1] + boxes[:, 3]) * 0.5)
                        ),
                    })
                cluster_rows.append({
                    "example_id": example_id,
                    "analysis_group": str(row.analysis_group),
                    "cluster_top_k": int(top_k),
                    "max_cluster_members": int(members),
                    "n_clusters": int(len(clusters)),
                    "n_finalists": int(len(finalists)),
                    "target_in_any_cluster": int(any(target_any)),
                    "target_in_finalists": int(any(target_final)),
                    "target_chosen_reserve_tension": int(chosen_target),
                })
        del endpoint, raw
        release_accelerator_memory()

    payload = {
        **asdict(config),
        "evaluation_ids": evaluation.example_id.astype(str).tolist(),
    }
    run_dir = Path(config.output_dir) / (
        f"cluster_localization_diagnostic_{stable_hash(payload)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(candidate_rows)
    clusters = pd.DataFrame(cluster_rows)
    cluster_details = pd.DataFrame(cluster_detail_rows)
    candidates.to_csv(run_dir / "candidate_availability.csv", index=False)
    clusters.to_csv(run_dir / "cluster_discovery.csv", index=False)
    cluster_details.to_csv(run_dir / "cluster_details.csv", index=False)
    candidate_summary = candidates.groupby(
        "proposal_top_k", as_index=False
    ).agg(
        hidden_n=("example_id", "nunique"),
        target_candidate_available_n=("target_candidate_available", "sum"),
        median_max_target_iou=("max_target_iou_in_top_k", "median"),
    )
    candidate_summary.to_csv(
        run_dir / "candidate_availability_summary.csv", index=False
    )
    if clusters.empty:
        cluster_summary = pd.DataFrame(columns=[
            "cluster_top_k", "max_cluster_members", "hidden_n",
            "target_in_any_cluster_n", "target_in_finalists_n",
            "target_chosen_reserve_tension_n", "mean_clusters",
        ])
    else:
        cluster_summary = clusters.groupby(
            ["cluster_top_k", "max_cluster_members"], as_index=False
        ).agg(
            hidden_n=("example_id", "nunique"),
            target_in_any_cluster_n=("target_in_any_cluster", "sum"),
            target_in_finalists_n=("target_in_finalists", "sum"),
            target_chosen_reserve_tension_n=(
                "target_chosen_reserve_tension", "sum"
            ),
            mean_clusters=("n_clusters", "mean"),
        )
    cluster_summary.to_csv(
        run_dir / "cluster_discovery_summary.csv", index=False
    )
    elapsed = time.time() - started
    write_summary(run_dir / "summary.json", {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "hidden_examples": int(evaluation.example_id.nunique()),
        "cache_path": str(cache_path),
        "config": asdict(config),
    })
    StorageBudget(config.output_dir, config.max_output_gb).check()
    return run_dir


if __name__ == "__main__":
    print(run_cluster_localization_diagnostic())
