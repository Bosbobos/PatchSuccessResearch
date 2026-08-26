from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .candidate_reserve import _cache_lookup, _evaluate_batch
from .causal_repair import _load_inputs
from .common import StorageBudget, load_experiment, release_accelerator_memory, stable_hash
from .followup_common import (
    ATTACK_PATH_DB,
    FOLLOWUP_DIR,
    MANIFEST_CSV,
    TRACE_DB,
    balanced_subset,
    box_iou_numpy,
    write_summary,
)
from .mechanism_followup import _head_branches
from .self_counterfactual_defense import _decoded_frame
from .self_counterfactual_defense import _all_class_nms, _detection_set_metrics
from .single_forward_component import (
    SingleForwardComponentConfig,
    _collect_clean_channel_moments,
    _intervention_raw,
    _single_forward_maps,
    _split_reference_evaluation,
)


@dataclass(slots=True)
class AutonomousNegativeRepairConfig(SingleForwardComponentConfig):
    top_negative_k: tuple[int, ...] = (250, 500, 1000)
    cluster_selection_k: int = 500
    repair_top_clusters: tuple[int, ...] = (1, 2)
    cluster_iou: float = 0.50
    cluster_candidate_limit: int = 300
    cluster_min_score: float = 0.001
    max_cluster_members: int = 20
    prefilter_per_strategy: int = 2
    clean_evaluation_examples: int = 25
    cluster_ranking: str = "predicted_gain"
    method_version: int = 1


def _clusters_from_frame(frame: pd.DataFrame, config) -> list[dict]:
    proposal_column = (
        "proposal_score" if "proposal_score" in frame.columns else "score"
    )
    current = frame[
        frame[proposal_column] >= float(config.cluster_min_score)
    ].head(int(config.cluster_candidate_limit)).copy()
    current = current.sort_values(
        proposal_column, ascending=False
    ).reset_index(drop=True)
    if current.empty:
        return []
    boxes = current[["x1", "y1", "x2", "y2"]].to_numpy(float)
    remaining = list(range(len(current)))
    clusters = []
    while remaining:
        seed = remaining[0]
        ious = box_iou_numpy(boxes[seed:seed + 1], boxes[remaining]).reshape(-1)
        all_members = [
            remaining[index]
            for index, value in enumerate(ious)
            if value >= float(config.cluster_iou)
        ]
        member_set = set(all_members)
        remaining = [index for index in remaining if index not in member_set]
        members = all_members[: int(config.max_cluster_members)]
        columns = [
            "flat_index", "level_index", "y_index", "x_index", "score",
            "x1", "y1", "x2", "y2",
        ]
        if proposal_column != "score":
            columns.append(proposal_column)
        selection = current.iloc[members][columns].drop_duplicates(
            "flat_index"
        ).reset_index(drop=True)
        scores = selection.score.to_numpy(float)
        proposal_scores = selection[proposal_column].to_numpy(float)
        noisy_or = float(1.0 - np.prod(1.0 - np.clip(scores, 1e-8, 1 - 1e-8)))
        max_score = float(scores.max())
        clusters.append({
            "selection": selection,
            "n_members": int(len(selection)),
            "max_score": max_score,
            "max_proposal_score": float(proposal_scores.max()),
            "noisy_or": noisy_or,
            "reserve_tension": float(len(selection) * (1.0 - max_score)),
            "object_suppression_tension": float(
                len(selection)
                * (1.0 - max_score)
                * float(proposal_scores.max())
            ),
        })
    return clusters


def _prefilter_clusters(clusters: list[dict], config) -> list[dict]:
    chosen = []
    seen = set()
    for metric in ("noisy_or", "reserve_tension"):
        ordered = sorted(clusters, key=lambda item: item[metric], reverse=True)
        for item in ordered[: int(config.prefilter_per_strategy)]:
            key = tuple(sorted(item["selection"].flat_index.astype(int).tolist()))
            if key not in seen:
                chosen.append(item)
                seen.add(key)
    return chosen


def _target_overlap(selection: pd.DataFrame, row) -> float:
    boxes = selection[["x1", "y1", "x2", "y2"]].to_numpy(float)
    target = np.asarray([[
        row.clean_target_x1,
        row.clean_target_y1,
        row.clean_target_x2,
        row.clean_target_y2,
    ]], dtype=float)
    return float(box_iou_numpy(boxes, target).max()) if len(boxes) else 0.0


def run_autonomous_negative_repair(
    config: AutonomousNegativeRepairConfig | None = None,
) -> Path:
    config = config or AutonomousNegativeRepairConfig()
    started = time.time()
    StorageBudget(config.output_dir, config.max_output_gb).check()
    selected, _ = _load_inputs(
        Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None
    )
    selected = balanced_subset(
        selected, int(config.examples_per_group), seed=int(config.seed)
    )
    reference, evaluation = _split_reference_evaluation(selected, config)
    exp, cache_path = load_experiment(
        prefer_device=config.device, require_device=bool(config.require_device)
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    cache = _cache_lookup(exp)
    means, _stds = _collect_clean_channel_moments(
        exp, model, detect, cache, reference
    )
    records = []
    cluster_records = []
    clean_ids = set(
        evaluation.sample(
            n=min(int(config.clean_evaluation_examples), len(evaluation)),
            random_state=int(config.seed) + 41,
        ).example_id.astype(str)
    )
    for row in tqdm(
        evaluation.itertuples(index=False),
        total=len(evaluation),
        desc="autonomous negative repair",
        unit="image",
    ):
        example_id = str(row.example_id)
        example = cache[example_id]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        evaluation_clean_inputs = _capture_detect_inputs(model, detect, pair[0:1])
        with __import__("torch").inference_mode():
            _eval_box, _eval_cls, evaluation_clean_raw = _head_branches(
                detect, evaluation_clean_inputs
            )
            clean_reference = _all_class_nms(
                detect, evaluation_clean_raw, config
            )[0]
        del evaluation_clean_inputs, evaluation_clean_raw
        release_accelerator_memory()
        input_specs = [("patched", pair[1:2])]
        if example_id in clean_ids:
            input_specs.insert(0, ("clean", pair[0:1]))
        for input_kind, image in input_specs:
            captured = _capture_detect_inputs(model, detect, image)
            endpoint = [item[0:1] for item in captured]
            import torch

            with torch.inference_mode():
                _box, _cls, raw = _head_branches(detect, endpoint)
            frame = _decoded_frame(detect, raw, int(row.class_id))
            all_clusters = _clusters_from_frame(frame, config)
            finalists = _prefilter_clusters(all_clusters, config)
            scored = []
            for cluster_index, cluster in enumerate(finalists):
                maps, metadata = _single_forward_maps(
                    detect,
                    endpoint,
                    cluster["selection"],
                    int(row.class_id),
                    means,
                    config,
                    oracle_delta_inputs=None,
                )
                condition = f"top_negative_{int(config.cluster_selection_k)}"
                predicted_gain = float(metadata[f"{condition}_predicted_gain"])
                target_iou = _target_overlap(cluster["selection"], row)
                item = {
                    **cluster,
                    "cluster_index": int(cluster_index),
                    "maps": maps,
                    "metadata": metadata,
                    "predicted_gain": predicted_gain,
                    "target_iou": target_iou,
                }
                scored.append(item)
                cluster_records.append({
                    "example_id": example_id,
                    "analysis_group": str(row.analysis_group),
                    "input_kind": input_kind,
                    "cluster_index": int(cluster_index),
                    "n_all_clusters": int(len(all_clusters)),
                    "n_finalists": int(len(finalists)),
                    "n_members": int(cluster["n_members"]),
                    "max_score": float(cluster["max_score"]),
                    "noisy_or": float(cluster["noisy_or"]),
                    "reserve_tension": float(cluster["reserve_tension"]),
                    "predicted_gain": predicted_gain,
                    "target_iou": target_iou,
                    "is_target_cluster": int(target_iou >= float(config.target_iou)),
                    **{
                        key: value
                        for key, value in metadata.items()
                        if isinstance(value, (int, float, np.integer, np.floating))
                    },
                })
            if not scored:
                release_accelerator_memory()
                continue
            def ranking_value(item):
                if config.cluster_ranking == "predicted_gain":
                    return float(item["predicted_gain"])
                if config.cluster_ranking == "reserve_tension":
                    return float(item["reserve_tension"])
                if config.cluster_ranking == "mass_over_concentration":
                    concentration = float(
                        item["metadata"][
                            f"top_negative_{int(config.cluster_selection_k)}_gain_concentration"
                        ]
                    )
                    return float(
                        item["metadata"]["total_available_negative_gain"]
                        / max(concentration, 1e-6)
                    )
                raise ValueError(
                    f"Unknown cluster_ranking: {config.cluster_ranking}"
                )

            for item in scored:
                item["ranking_value"] = ranking_value(item)
            scored.sort(key=lambda item: item["ranking_value"], reverse=True)
            chosen = scored[0]
            intervention_maps = {}
            intervention_target_flags = {}
            for k in config.top_negative_k:
                source_name = f"top_negative_{int(k)}"
                name = f"top1_k{int(k)}"
                intervention_maps[name] = chosen["maps"][source_name]
                intervention_target_flags[name] = int(
                    chosen["target_iou"] >= float(config.target_iou)
                )
            for top_n in config.repair_top_clusters:
                if int(top_n) <= 1:
                    continue
                selected_clusters = scored[: min(int(top_n), len(scored))]
                source_name = f"top_negative_{int(config.cluster_selection_k)}"
                combined = []
                for level in range(len(endpoint)):
                    value = selected_clusters[0]["maps"][source_name][level].clone()
                    occupied = value.ne(0)
                    for item in selected_clusters[1:]:
                        candidate = item["maps"][source_name][level]
                        value = __import__("torch").where(
                            occupied, value, candidate
                        )
                        occupied |= candidate.ne(0)
                    combined.append(value)
                name = f"top{int(top_n)}_k{int(config.cluster_selection_k)}"
                intervention_maps[name] = combined
                intervention_target_flags[name] = int(
                    any(
                        item["target_iou"] >= float(config.target_iou)
                        for item in selected_clusters
                    )
                )
            names, intervention_raw = _intervention_raw(
                detect, endpoint, intervention_maps
            )
            with torch.inference_mode():
                results = _evaluate_batch(detect, intervention_raw, row, config)
                all_detections = _all_class_nms(
                    detect, intervention_raw, config
                )
            source_hidden = int(results[0]["target_hidden"])
            for name, result, detections in zip(
                names, results, all_detections, strict=True
            ):
                selected_contains_target = (
                    int(chosen["target_iou"] >= float(config.target_iou))
                    if name == "observed"
                    else int(intervention_target_flags[name])
                )
                result.update({
                    "example_id": example_id,
                    "analysis_group": str(row.analysis_group),
                    "input_kind": input_kind,
                    "condition": name,
                    "source_hidden": source_hidden,
                    "n_all_clusters": int(len(all_clusters)),
                    "n_finalists": int(len(finalists)),
                    "chosen_cluster_target_iou": float(chosen["target_iou"]),
                    "chosen_is_target_cluster": int(
                        chosen["target_iou"] >= float(config.target_iou)
                    ),
                    "selected_contains_target_cluster": selected_contains_target,
                    "chosen_predicted_gain": float(chosen["predicted_gain"]),
                    "chosen_ranking_value": float(chosen["ranking_value"]),
                    "cluster_ranking": str(config.cluster_ranking),
                    "chosen_n_members": int(chosen["n_members"]),
                    "chosen_max_score": float(chosen["max_score"]),
                    **chosen["metadata"],
                })
                for key, value in _detection_set_metrics(
                    clean_reference, detections, config.target_iou
                ).items():
                    result[f"detection_{key}"] = value
                records.append(result)
            del captured
            release_accelerator_memory()
    rows = pd.DataFrame(records)
    clusters = pd.DataFrame(cluster_records)
    hidden = rows[
        rows.input_kind.eq("patched") & rows.source_hidden.astype(bool)
    ]
    result_summary = rows.groupby(
        ["input_kind", "condition"], as_index=False
    ).agg(
        n=("example_id", "nunique"),
        target_detection_rate=("target_detected", "mean"),
        target_hidden_rate=("target_hidden", "mean"),
        chosen_target_cluster_rate=("chosen_is_target_cluster", "mean"),
        selected_target_cluster_rate=("selected_contains_target_cluster", "mean"),
        mean_post_target_conf=("post_target_conf", "mean"),
        mean_detection_f1=("detection_detection_f1", "mean"),
    )
    hidden_summary = hidden.groupby("condition", as_index=False).agg(
        hidden_n=("example_id", "nunique"),
        hidden_recovery_rate=("target_detected", "mean"),
        hidden_chosen_target_cluster_rate=("chosen_is_target_cluster", "mean"),
        hidden_selected_target_cluster_rate=("selected_contains_target_cluster", "mean"),
    )
    result_summary = result_summary.merge(
        hidden_summary, on="condition", how="left"
    )
    payload = {
        **asdict(config),
        "reference_ids": reference.example_id.astype(str).tolist(),
        "evaluation_ids": evaluation.example_id.astype(str).tolist(),
    }
    run_dir = Path(config.output_dir) / f"autonomous_negative_repair_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(run_dir / "autonomous_repair_rows.csv", index=False)
    clusters.to_csv(run_dir / "autonomous_cluster_rows.csv", index=False)
    result_summary.to_csv(run_dir / "autonomous_repair_summary.csv", index=False)
    elapsed = time.time() - started
    summary = {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "reference_examples": int(reference.example_id.nunique()),
        "evaluation_examples": int(evaluation.example_id.nunique()),
        "clean_evaluation_examples": int(len(clean_ids)),
        "cache_path": str(cache_path),
        "config": asdict(config),
        "limitations": [
            "Inference uses one unmodified image, Detect-input activations, gradients, and unpaired clean-population channel means.",
            "Candidate clusters and their ranking do not use the clean target or patch location.",
            "The clean target box is used only to score localization and recovery after the intervention.",
            "Only the person class and a fixed sparse coordinate budget are evaluated.",
        ],
    }
    write_summary(run_dir / "summary.json", summary)
    (run_dir / "analysis_digest.md").write_text(
        "\n".join([
            "# Autonomous single-forward negative repair",
            "",
            f"- elapsed: {elapsed:.1f} s",
            f"- reference examples: {summary['reference_examples']}",
            f"- evaluation examples: {summary['evaluation_examples']}",
            "- Read autonomous_repair_summary.csv first.",
        ]) + "\n",
        encoding="utf-8",
    )
    StorageBudget(config.output_dir, config.max_output_gb).check()
    return run_dir


if __name__ == "__main__":
    print(run_autonomous_negative_repair())
