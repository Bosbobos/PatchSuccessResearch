from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _preprocess_pair
from .candidate_reserve import _cache_lookup, _evaluate_batch
from .common import REPO_ROOT, load_experiment, release_accelerator_memory, stable_hash
from .component_aware_cluster_ranker import (
    _cluster_energy_fraction,
    _fit_pairwise_ranker,
    _greedy_marginal_indices,
    _pairwise_scores,
    _scene_features,
    _teacher_spatial_energy,
)
from .component_student import _capture_with_grad, _reset_detect_inference_cache
from .component_targeted_patch import _record_lookup
from .followup_common import GROUP_ORDER, TRACE_DB
from .large_component_student import (
    LargeComponentStudentConfig,
    _attacked_reference_statistics,
    _blind_candidate_indices,
    _build_training_rows,
)
from .learned_cluster_ranker import (
    _average_gradient_for_clusters,
    _clusters_for_levels,
)
from .leak_free_component_defense import (
    _student_condition,
)
from .self_counterfactual_defense import _all_class_nms, _detection_set_metrics
from .mechanism_followup import _head_branches
from .prepare_max_balanced_component_pool import LABELS_CSV
from .balanced_target_path import _assign_groups


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "CandidateRoutingAndAttackPath"
    / "no_iou_expansion_outputs"
)


@dataclass(slots=True)
class NoIouExpansionConfig:
    prior_run: str
    pool_run: str
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "mps"
    require_device: bool = False
    student_feature_set: str = "functional"
    final_per_group: int = 100
    final_groups: tuple[str, ...] = (
        "hidden_low_conf_match",
        "hidden_no_iou_match",
        "visible_target_winner",
    )
    proposal_top_k: int = 3000
    proposal_per_level_k: int = 900
    proposal_candidate_limit: int = 6500
    cluster_ious: tuple[float, ...] = (0.30, 0.50, 0.70)
    max_hypotheses: int = 320
    top_clusters: int = 5
    candidate_budget: int = 8000
    student_apply_top_k: int = 12000
    pair_samples_per_scene: int = 600
    pairwise_max_iter: int = 180
    pairwise_max_leaf_nodes: int = 31
    set_rank_weight: float = 0.65
    seed: int = 2907
    max_student_scenes: int | None = None
    max_ranker_scenes: int | None = None
    max_final_scenes: int | None = None
    method_version: int = 1


def _balanced_limit(rows: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or len(rows) <= int(limit):
        return rows.copy()
    groups = [
        group for group in GROUP_ORDER
        if rows.analysis_group.eq(group).any()
    ]
    base, remainder = divmod(int(limit), len(groups))
    parts = []
    for index, group in enumerate(groups):
        take = base + int(index < remainder)
        parts.append(rows[rows.analysis_group.eq(group)].head(take))
    return pd.concat(parts, ignore_index=True)


def _fresh_final_split(pool_run: Path, config: NoIouExpansionConfig) -> pd.DataFrame:
    used = pd.concat(
        [
            pd.read_csv(pool_run / "student_train_split.csv"),
            pd.read_csv(pool_run / "ranker_train_split.csv"),
            pd.read_csv(pool_run / "holdout_split.csv"),
        ],
        ignore_index=True,
    )
    used_paths = set(used.path.astype(str))
    with sqlite3.connect(TRACE_DB) as connection:
        trace = pd.read_sql_query(
            "SELECT * FROM examples WHERE error IS NULL", connection
        )
    labels = pd.read_csv(LABELS_CSV)
    labels = labels[labels.target_eligible.astype(bool)].copy()
    labels["analysis_group"] = _assign_groups(labels)
    labels = labels[
        ["example_id", "analysis_group"]
    ].dropna().drop_duplicates("example_id")
    trace = trace.merge(
        labels,
        on="example_id",
        how="inner",
        validate="one_to_one",
    )
    available = trace[~trace.path.astype(str).isin(used_paths)].copy()
    parts = []
    for index, group in enumerate(config.final_groups):
        current = (
            available[available.analysis_group.eq(group)]
            .drop_duplicates("path")
        )
        if len(current) < int(config.final_per_group):
            raise RuntimeError(
                f"Fresh group {group!r} has {len(current)} scenes; "
                f"{config.final_per_group} requested."
            )
        parts.append(
            current.sample(
                n=int(config.final_per_group),
                random_state=int(config.seed) + index,
            )
        )
    final = pd.concat(parts, ignore_index=True)
    if set(final.path.astype(str)) & used_paths:
        raise AssertionError("Fresh final split overlaps the previous pool.")
    return final


def _cluster_key(cluster) -> tuple[int, ...]:
    return tuple(sorted(cluster["selection"].flat_index.astype(int).tolist()))


def _expanded_clusters_for_levels(detect, levels, base_config, config):
    candidates = []
    seen = set()
    for cluster_iou in config.cluster_ious:
        current_config = replace(
            base_config,
            blind_person_top_k=int(config.proposal_top_k),
            blind_class_agnostic_top_k=int(config.proposal_top_k),
            blind_class_agnostic_per_level_k=int(
                config.proposal_per_level_k
            ),
            blind_cluster_candidate_limit=int(
                config.proposal_candidate_limit
            ),
            blind_cluster_min_score=1e-10,
            blind_cluster_iou=float(cluster_iou),
            blind_max_cluster_members=150,
        )
        for cluster in _clusters_for_levels(
            detect, levels, current_config
        ):
            key = _cluster_key(cluster)
            if key not in seen:
                item = dict(cluster)
                item["source_iou"] = float(cluster_iou)
                candidates.append(item)
                seen.add(key)
    if len(candidates) <= int(config.max_hypotheses):
        return candidates
    metrics = (
        "object_suppression_tension",
        "reserve_tension",
        "noisy_or",
        "max_proposal_score",
    )
    chosen = []
    chosen_keys = set()
    per_metric = max(1, int(config.max_hypotheses) // len(metrics))
    for metric in metrics:
        ordered = sorted(
            candidates, key=lambda item: item[metric], reverse=True
        )
        for cluster in ordered[:per_metric]:
            key = _cluster_key(cluster)
            if key not in chosen_keys:
                chosen.append(cluster)
                chosen_keys.add(key)
    if len(chosen) < int(config.max_hypotheses):
        ordered = sorted(
            candidates,
            key=lambda item: (
                item["object_suppression_tension"],
                item["reserve_tension"],
            ),
            reverse=True,
        )
        for cluster in ordered:
            key = _cluster_key(cluster)
            if key not in chosen_keys:
                chosen.append(cluster)
                chosen_keys.add(key)
            if len(chosen) >= int(config.max_hypotheses):
                break
    return chosen[: int(config.max_hypotheses)]


def _target_scale_from_training_rows(rows_by_mode) -> float:
    target = np.concatenate(
        [row["labels"] for row in rows_by_mode["blind_support"]]
    )
    nonzero = target[np.abs(target) > 1e-8]
    return max(
        float(np.std(nonzero if len(nonzero) else target)), 1e-8
    )


def _expanded_ranker_frame(
    exp,
    model,
    detect,
    records,
    student,
    target_scale,
    reference_mean,
    reference_std,
    base_config,
    config,
):
    examples = _cache_lookup(exp)
    rows = []
    for record in tqdm(
        records, desc="expanded ranker features", unit="scene"
    ):
        example = examples[record["example_id"]]
        _clean, patched, _ = exp._images_for_example(example)
        image = _preprocess_pair(exp, patched, patched)[:1]
        _reset_detect_inference_cache(detect)
        image = image.detach().requires_grad_(True)
        decoded, levels = _capture_with_grad(model, detect, image)
        clusters = _expanded_clusters_for_levels(
            detect, levels, base_config, config
        )
        features, _signal, _ = _scene_features(
            decoded,
            levels,
            clusters,
            int(image.shape[-1]),
            student,
            target_scale,
            reference_mean,
            reference_std,
            config,
        )
        from .large_component_student import _teacher_arrays

        teacher_indices, teacher_components = _teacher_arrays(
            record, levels
        )
        energy_maps, total_energy = _teacher_spatial_energy(
            teacher_indices, teacher_components, levels
        )
        features["teacher_energy_fraction"] = [
            _cluster_energy_fraction(
                cluster, energy_maps, total_energy, levels
            )
            for cluster in clusters
        ]
        features["example_id"] = record["example_id"]
        features["analysis_group"] = record["analysis_group"]
        rows.append(features)
        _reset_detect_inference_cache(detect)
        release_accelerator_memory()
    return pd.concat(rows, ignore_index=True)


def _correction_for_selection(
    decoded,
    levels,
    clusters,
    indices,
    student,
    target_scale,
    reference_mean,
    reference_std,
    config,
):
    gradients = _average_gradient_for_clusters(
        decoded, levels, [clusters[index] for index in indices]
    )
    candidates = _blind_candidate_indices(
        levels,
        gradients,
        reference_mean,
        reference_std,
        int(config.candidate_budget),
        "hybrid",
    )
    return _student_condition(
        student,
        levels,
        gradients,
        candidates,
        reference_mean,
        reference_std,
        target_scale,
        int(config.student_apply_top_k),
    )


def _conditions_for_image(
    model,
    detect,
    image,
    student,
    baseline_ranker,
    general_ranker,
    specialist_ranker,
    target_scale,
    reference_mean,
    reference_std,
    base_config,
    config,
):
    _reset_detect_inference_cache(detect)
    image = image.detach().requires_grad_(True)
    decoded, levels = _capture_with_grad(model, detect, image)
    default_clusters = _clusters_for_levels(detect, levels, base_config)
    default_features, _default_signal, _ = _scene_features(
        decoded,
        levels,
        default_clusters,
        int(image.shape[-1]),
        student,
        target_scale,
        reference_mean,
        reference_std,
        config,
    )
    # The saved baseline ranker is evaluated without teacher information.
    baseline_scores = _pairwise_scores(
        baseline_ranker, default_features
    )
    baseline_indices = np.argsort(-baseline_scores)[
        : int(config.top_clusters)
    ].astype(int).tolist()

    expanded = _expanded_clusters_for_levels(
        detect, levels, base_config, config
    )
    features, signal, _ = _scene_features(
        decoded,
        levels,
        expanded,
        int(image.shape[-1]),
        student,
        target_scale,
        reference_mean,
        reference_std,
        config,
    )
    general_scores = _pairwise_scores(general_ranker, features)
    specialist_scores = _pairwise_scores(specialist_ranker, features)
    general_indices = np.argsort(-general_scores)[
        : int(config.top_clusters)
    ].astype(int).tolist()
    specialist_indices = np.argsort(-specialist_scores)[
        : int(config.top_clusters)
    ].astype(int).tolist()
    set_aware_indices = _greedy_marginal_indices(
        expanded,
        levels,
        signal["fused_maps"],
        int(config.top_clusters),
        base_scores=specialist_scores,
        base_weight=float(config.set_rank_weight),
    )
    selections = {
        "baseline_pairwise_8k": (
            default_clusters, baseline_indices
        ),
        "expanded_general_top5_8k": (
            expanded, general_indices
        ),
        "expanded_specialist_top5_8k": (
            expanded, specialist_indices
        ),
        "expanded_specialist_setaware_top5_8k": (
            expanded, set_aware_indices
        ),
    }
    conditions = {}
    metadata = {}
    for name, (clusters, indices) in selections.items():
        conditions[name], metadata[name] = _correction_for_selection(
            decoded,
            levels,
            clusters,
            indices,
            student,
            target_scale,
            reference_mean,
            reference_std,
            config,
        )
        metadata[name]["hypothesis_count"] = int(len(clusters))
    return levels, conditions, metadata


def _evaluate(
    exp,
    model,
    detect,
    rows,
    student,
    baseline_ranker,
    general_ranker,
    specialist_ranker,
    target_scale,
    reference_mean,
    reference_std,
    base_config,
    config,
):
    import torch

    examples = _cache_lookup(exp)
    output = []
    for row in tqdm(
        rows.itertuples(index=False),
        total=len(rows),
        desc="fresh no-IoU final",
        unit="scene",
    ):
        example = examples[str(row.example_id)]
        clean, patched, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean, patched)
        for input_kind, image in (
            ("patched", pair[1:2]),
            ("clean", pair[0:1]),
        ):
            levels, conditions, metadata = _conditions_for_image(
                model,
                detect,
                image,
                student,
                baseline_ranker,
                general_ranker,
                specialist_ranker,
                target_scale,
                reference_mean,
                reference_std,
                base_config,
                config,
            )
            names = ["observed", *conditions]
            all_levels = {"observed": levels, **conditions}
            with torch.no_grad():
                batched = [
                    torch.cat(
                        [all_levels[name][level] for name in names], dim=0
                    )
                    for level in range(len(levels))
                ]
                _box, _cls, raw = _head_branches(detect, batched)
                target_results = _evaluate_batch(
                    detect, raw, row, base_config
                )
                nms_results = _all_class_nms(
                    detect, raw, base_config
                )
            target_by_name = dict(
                zip(names, target_results, strict=True)
            )
            nms_by_name = dict(zip(names, nms_results, strict=True))
            for name in conditions:
                full_set = _detection_set_metrics(
                    nms_by_name["observed"],
                    nms_by_name[name],
                    base_config.target_iou,
                )
                output.append({
                    "example_id": str(row.example_id),
                    "analysis_group": str(row.analysis_group),
                    "input_kind": input_kind,
                    "condition": name,
                    "baseline_target_detected": target_by_name[
                        "observed"
                    ]["target_detected"],
                    "corrected_target_detected": target_by_name[name][
                        "target_detected"
                    ],
                    "baseline_target_conf": target_by_name[
                        "observed"
                    ]["post_target_conf"],
                    "corrected_target_conf": target_by_name[name][
                        "post_target_conf"
                    ],
                    "full_detection_f1": full_set["detection_f1"],
                    **metadata[name],
                })
            _reset_detect_inference_cache(detect)
            release_accelerator_memory()
    return pd.DataFrame(output)


def _summary(evaluation):
    rows = []
    for condition, group in evaluation.groupby("condition"):
        patched = group[group.input_kind.eq("patched")]
        clean = group[group.input_kind.eq("clean")]
        for analysis_group in ("all", *sorted(
            patched.analysis_group.unique()
        )):
            current = (
                patched
                if analysis_group == "all"
                else patched[patched.analysis_group.eq(analysis_group)]
            )
            hidden = current.baseline_target_detected.eq(0)
            rows.append({
                "condition": condition,
                "analysis_group": analysis_group,
                "patched_n": int(len(current)),
                "hidden_n": int(hidden.sum()),
                "hidden_recovered_n": int(
                    (
                        hidden
                        & current.corrected_target_detected.eq(1)
                    ).sum()
                ),
                "baseline_lost_n": int(
                    (
                        current.baseline_target_detected.eq(1)
                        & current.corrected_target_detected.eq(0)
                    ).sum()
                ),
                "clean_target_change_n": int(
                    (
                        clean.corrected_target_detected
                        != clean.baseline_target_detected
                    ).sum()
                ) if analysis_group == "all" else np.nan,
                "clean_full_detection_f1": float(
                    clean.full_detection_f1.mean()
                ) if analysis_group == "all" else np.nan,
            })
    return pd.DataFrame(rows)


def run_no_iou_expansion(config: NoIouExpansionConfig) -> Path:
    started = time.time()
    prior_run = Path(config.prior_run).resolve()
    pool_run = Path(config.pool_run).resolve()
    pool_meta = json.loads(
        (pool_run / "run.json").read_text(encoding="utf-8")
    )
    if pool_meta["status"] != "complete":
        raise RuntimeError("The balanced pool must be complete.")
    student_rows = pd.read_csv(pool_run / "student_train_split.csv")
    ranker_rows = pd.read_csv(pool_run / "ranker_train_split.csv")
    previous_holdout = pd.read_csv(pool_run / "holdout_split.csv")
    final_rows = _fresh_final_split(pool_run, config)
    student_rows = _balanced_limit(
        student_rows, config.max_student_scenes
    )
    ranker_rows = _balanced_limit(
        ranker_rows, config.max_ranker_scenes
    )
    if config.max_final_scenes is not None:
        final_rows = final_rows.head(
            int(config.max_final_scenes)
        ).copy()
    prior_meta = json.loads(
        (prior_run / "run.json").read_text(encoding="utf-8")
    )
    if Path(prior_meta["config"]["pool_run"]).resolve() != pool_run:
        raise RuntimeError("Prior defense and pool-run do not match.")

    payload = {
        **asdict(config),
        "student_ids": student_rows.example_id.astype(str).tolist(),
        "ranker_ids": ranker_rows.example_id.astype(str).tolist(),
        "final_ids": final_rows.example_id.astype(str).tolist(),
    }
    run_dir = (
        Path(config.output_dir)
        / f"no_iou_expansion_{stable_hash(payload)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    final_rows.to_csv(run_dir / "fresh_final_split.csv", index=False)
    print("[1/5] Loading model and reconstructing student state...", flush=True)
    exp, _cache_path = load_experiment(
        prefer_device=config.device,
        require_device=config.require_device,
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    manifest = json.loads(
        (
            Path(pool_meta["teacher_cache_dir"]) / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    student_records = _record_lookup(exp, student_rows, manifest)
    ranker_records = _record_lookup(exp, ranker_rows, manifest)
    row_lookup = {
        str(row.example_id): row
        for row in pd.concat(
            [student_rows, ranker_rows], ignore_index=True
        ).itertuples(index=False)
    }
    for record in [*student_records, *ranker_records]:
        record["row"] = row_lookup[record["example_id"]]
    base_config = LargeComponentStudentConfig(
        device=config.device,
        require_device=config.require_device,
        feature_sets=("functional",),
        modes=("blind_support",),
        blind_selector="hybrid",
        blind_candidates_per_level=4000,
        blind_apply_top_k=6000,
        diagnostic_ablations=False,
        seed=2203,
    )
    student = joblib.load(prior_run / "component_student.joblib")
    baseline_ranker = joblib.load(prior_run / "cluster_ranker.joblib")
    state_path = run_dir / "student_inference_state.npz"
    if state_path.exists():
        with np.load(state_path, allow_pickle=False) as state:
            level_count = int(state["level_count"])
            reference_mean = [
                state[f"reference_mean_{level}"]
                for level in range(level_count)
            ]
            reference_std = [
                state[f"reference_std_{level}"]
                for level in range(level_count)
            ]
            target_scale = float(state["target_scale"])
        print("      Reused cached student inference state.", flush=True)
    else:
        reference_mean, reference_std = _attacked_reference_statistics(
            exp, model, detect, student_records
        )
        training_rows, _ = _build_training_rows(
            exp,
            model,
            detect,
            student_records,
            reference_mean,
            reference_std,
            base_config,
        )
        target_scale = _target_scale_from_training_rows(training_rows)
        state_payload = {
            "level_count": np.asarray(len(reference_mean)),
            "target_scale": np.asarray(target_scale),
        }
        for level, (mean, std) in enumerate(
            zip(reference_mean, reference_std, strict=True)
        ):
            state_payload[f"reference_mean_{level}"] = mean
            state_payload[f"reference_std_{level}"] = std
        np.savez_compressed(state_path, **state_payload)
    print("[2/5] Building expanded ranker features...", flush=True)
    ranker_frame_path = run_dir / "expanded_ranker_training_rows.csv"
    if ranker_frame_path.exists():
        ranker_frame = pd.read_csv(ranker_frame_path)
        print("      Reused cached expanded ranker features.", flush=True)
    else:
        ranker_frame = _expanded_ranker_frame(
            exp,
            model,
            detect,
            ranker_records,
            student,
            target_scale,
            reference_mean,
            reference_std,
            base_config,
            config,
        )
        ranker_frame.to_csv(ranker_frame_path, index=False)
    print("[3/5] Fitting general and no-IoU specialist rankers...", flush=True)
    general_ranker, general_pairs = _fit_pairwise_ranker(
        ranker_frame, config
    )
    specialist_frame = ranker_frame[
        ranker_frame.analysis_group.eq("hidden_no_iou_match")
    ].copy()
    specialist_ranker, specialist_pairs = _fit_pairwise_ranker(
        specialist_frame, config
    )
    print("[4/5] Evaluating the locked mechanisms on fresh scenes...", flush=True)
    evaluation = _evaluate(
        exp,
        model,
        detect,
        final_rows,
        student,
        baseline_ranker,
        general_ranker,
        specialist_ranker,
        target_scale,
        reference_mean,
        reference_std,
        base_config,
        config,
    )
    summary = _summary(evaluation)
    evaluation.to_csv(run_dir / "evaluation_rows.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    joblib.dump(general_ranker, run_dir / "general_ranker.joblib")
    joblib.dump(
        specialist_ranker, run_dir / "no_iou_specialist_ranker.joblib"
    )
    train_paths = set(
        pd.concat([student_rows, ranker_rows]).path.astype(str)
    )
    final_paths = set(final_rows.path.astype(str))
    previous_holdout_paths = set(previous_holdout.path.astype(str))
    audit = {
        "student_train_scenes": int(len(student_rows)),
        "ranker_train_scenes": int(len(ranker_rows)),
        "previous_validation_scenes": int(len(previous_holdout)),
        "fresh_final_scenes": int(len(final_rows)),
        "train_final_path_overlap": int(len(train_paths & final_paths)),
        "previous_validation_final_path_overlap": int(
            len(previous_holdout_paths & final_paths)
        ),
        "final_teacher_records_loaded": False,
        "final_teacher_values_used": False,
        "final_target_boxes_used_for_metrics_only": True,
        "inference_uses_analysis_group": False,
        "primary_condition_locked_before_final": (
            "expanded_specialist_setaware_top5_8k"
        ),
    }
    (run_dir / "leakage_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "elapsed_seconds": time.time() - started,
                "config": asdict(config),
                "general_training_pairs": int(general_pairs),
                "specialist_training_pairs": int(specialist_pairs),
                "primary_condition": (
                    "expanded_specialist_setaware_top5_8k"
                ),
                "audit": audit,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[5/5] Complete: {run_dir}", flush=True)
    return run_dir


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train observable multi-IoU spatial rankers and evaluate a "
            "pre-committed no-IoU defense on a fresh teacher-free final set."
        )
    )
    parser.add_argument("--prior-run", required=True)
    parser.add_argument("--pool-run", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = NoIouExpansionConfig(
        prior_run=args.prior_run,
        pool_run=args.pool_run,
        output_dir=args.output_dir,
        device=args.device,
        require_device=args.require_device,
    )
    if args.smoke:
        config.max_student_scenes = 4
        config.max_ranker_scenes = 8
        config.max_final_scenes = 3
        config.proposal_top_k = 300
        config.proposal_per_level_k = 100
        config.proposal_candidate_limit = 600
        config.max_hypotheses = 40
        config.candidate_budget = 200
        config.student_apply_top_k = 300
        config.pair_samples_per_scene = 40
        config.pairwise_max_iter = 12
    print(run_no_iou_expansion(config))


if __name__ == "__main__":
    main()
