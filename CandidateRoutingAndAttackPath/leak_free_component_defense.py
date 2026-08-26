from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _preprocess_pair
from .candidate_reserve import _cache_lookup, _evaluate_batch
from .causal_repair import _load_inputs
from .common import REPO_ROOT, load_experiment, release_accelerator_memory, stable_hash
from .component_aware_cluster_ranker import (
    FEATURE_COLUMNS,
    _clusters_for_levels,
    _fit_pairwise_ranker,
    _heuristic_indices,
    _pairwise_scores,
    _scene_features,
    _teacher_spatial_energy,
)
from .component_student import _capture_with_grad, _reset_detect_inference_cache
from .component_targeted_patch import _record_lookup
from .followup_common import ATTACK_PATH_DB, MANIFEST_CSV, TRACE_DB
from .large_component_student import (
    LargeComponentStudentConfig,
    _attacked_reference_statistics,
    _blind_candidate_indices,
    _build_training_rows,
    _fit_students,
    _parts_for_indices,
    _prediction_maps,
    _teacher_arrays,
)
from .learned_cluster_ranker import (
    _average_gradient_for_clusters,
    _cluster_energy_fraction,
)
from .mechanism_followup import _head_branches
from .self_counterfactual_defense import _all_class_nms, _detection_set_metrics


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "CandidateRoutingAndAttackPath" / "leak_free_defense_outputs"
)


@dataclass(slots=True)
class LeakFreeDefenseConfig:
    base_run: str
    pool_run: str | None = None
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "mps"
    require_device: bool = False
    student_feature_set: str = "functional"
    student_train_scenes: int = 150
    ranker_train_scenes: int = 50
    candidate_budget: int = 4000
    expanded_candidate_budget: int = 8000
    student_apply_top_k: int = 6000
    top_clusters: int = 5
    pair_samples_per_scene: int = 600
    pairwise_max_iter: int = 180
    pairwise_max_leaf_nodes: int = 31
    max_student_scenes: int | None = None
    max_ranker_scenes: int | None = None
    max_holdout_scenes: int | None = None
    seed: int = 2203
    method_version: int = 1


def _student_ranker_split(rows, student_n, ranker_n, seed):
    groups = sorted(rows.analysis_group.astype(str).unique())
    ranker_base, ranker_remainder = divmod(int(ranker_n), len(groups))
    student_parts = []
    ranker_parts = []
    for group_index, group in enumerate(groups):
        current = (
            rows[rows.analysis_group.eq(group)]
            .sample(frac=1.0, random_state=int(seed) + group_index)
            .reset_index(drop=True)
        )
        take_ranker = ranker_base + int(group_index < ranker_remainder)
        ranker_parts.append(current.iloc[:take_ranker])
        student_parts.append(current.iloc[take_ranker:])
    student = pd.concat(student_parts, ignore_index=True)
    ranker = pd.concat(ranker_parts, ignore_index=True)
    if len(student) != int(student_n) or len(ranker) != int(ranker_n):
        raise RuntimeError(
            "Requested student/ranker sizes do not exhaust the supplied rows "
            f"({len(student)} + {len(ranker)} != {student_n} + {ranker_n})."
        )
    if set(student.path.astype(str)) & set(ranker.path.astype(str)):
        raise AssertionError("Student/ranker path leakage.")
    return student, ranker


def _predict_maps(
    student,
    levels,
    gradients,
    candidates,
    reference_mean,
    reference_std,
    target_scale,
    apply_top_k,
):
    parts = _parts_for_indices(
        levels,
        gradients,
        candidates,
        reference_mean,
        reference_std,
    )
    return _prediction_maps(
        student,
        "functional",
        parts,
        candidates,
        levels,
        target_scale,
        apply_top_k=apply_top_k,
    )


def _without_seen(candidates, seen):
    output = []
    for current, previous in zip(candidates, seen, strict=True):
        previous_set = set(int(value) for value in previous)
        output.append(np.asarray(
            [int(value) for value in current if int(value) not in previous_set],
            dtype=np.int64,
        ))
    return output


def _student_condition(
    student,
    levels,
    gradients,
    candidates,
    reference_mean,
    reference_std,
    target_scale,
    apply_top_k,
):
    correction, metadata = _predict_maps(
        student,
        levels,
        gradients,
        candidates,
        reference_mean,
        reference_std,
        target_scale,
        apply_top_k,
    )
    return [
        level - delta
        for level, delta in zip(levels, correction, strict=True)
    ], metadata


def _ranker_training_frame(
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
        records, desc="leak-free ranker features", unit="scene"
    ):
        example = examples[record["example_id"]]
        _clean, patched, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, patched, patched)
        _reset_detect_inference_cache(detect)
        image = pair[:1].detach().requires_grad_(True)
        decoded, levels = _capture_with_grad(model, detect, image)
        clusters = _clusters_for_levels(detect, levels, base_config)
        features, _signal, _ = _scene_features(
            decoded,
            levels,
            clusters,
            int(pair.shape[-1]),
            student,
            target_scale,
            reference_mean,
            reference_std,
            config,
        )
        teacher_indices, teacher_components = _teacher_arrays(record, levels)
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


def _conditions_for_image(
    model,
    detect,
    image,
    image_size,
    student,
    ranker,
    target_scale,
    reference_mean,
    reference_std,
    base_config,
    config,
):
    _reset_detect_inference_cache(detect)
    image = image.detach().requires_grad_(True)
    decoded, levels = _capture_with_grad(model, detect, image)
    clusters = _clusters_for_levels(detect, levels, base_config)
    features, _signal, _ = _scene_features(
        decoded,
        levels,
        clusters,
        image_size,
        student,
        target_scale,
        reference_mean,
        reference_std,
        config,
    )
    pairwise_scores = _pairwise_scores(ranker, features)
    selections = {
        "student_heuristic_4k": _heuristic_indices(
            clusters, config.top_clusters
        ),
        "student_pairwise_4k": np.argsort(-pairwise_scores)[
            : config.top_clusters
        ].astype(int).tolist(),
    }
    conditions = {}
    metadata = {}
    candidate_sets = {}
    gradients_by_name = {}
    for name, indices in selections.items():
        gradients = _average_gradient_for_clusters(
            decoded, levels, [clusters[index] for index in indices]
        )
        candidates = _blind_candidate_indices(
            levels,
            gradients,
            reference_mean,
            reference_std,
            config.candidate_budget,
            "hybrid",
        )
        conditions[name], item = _student_condition(
            student,
            levels,
            gradients,
            candidates,
            reference_mean,
            reference_std,
            target_scale,
            config.student_apply_top_k,
        )
        metadata[name] = item
        candidate_sets[name] = candidates
        gradients_by_name[name] = gradients

    expanded = "student_pairwise_8k"
    expanded_candidates = _blind_candidate_indices(
        levels,
        gradients_by_name["student_pairwise_4k"],
        reference_mean,
        reference_std,
        config.expanded_candidate_budget,
        "hybrid",
    )
    conditions[expanded], metadata[expanded] = _student_condition(
        student,
        levels,
        gradients_by_name["student_pairwise_4k"],
        expanded_candidates,
        reference_mean,
        reference_std,
        target_scale,
        2 * config.student_apply_top_k,
    )

    intermediate = [
        value.detach().requires_grad_(True)
        for value in conditions["student_pairwise_4k"]
    ]
    from .mechanism_followup import _decode

    _box, _cls, raw = _head_branches(detect, intermediate)
    residual_decoded = _decode(detect, raw)
    residual_clusters = _clusters_for_levels(
        detect, intermediate, base_config
    )
    if residual_clusters:
        residual_indices = _heuristic_indices(
            residual_clusters, config.top_clusters
        )
        residual_gradients = _average_gradient_for_clusters(
            residual_decoded,
            intermediate,
            [residual_clusters[index] for index in residual_indices],
        )
        residual_candidates = _blind_candidate_indices(
            intermediate,
            residual_gradients,
            reference_mean,
            reference_std,
            config.candidate_budget,
            "hybrid",
        )
        residual_candidates = _without_seen(
            residual_candidates,
            candidate_sets["student_pairwise_4k"],
        )
        iterative, iterative_metadata = _student_condition(
            student,
            intermediate,
            residual_gradients,
            residual_candidates,
            reference_mean,
            reference_std,
            target_scale,
            config.student_apply_top_k,
        )
        conditions["student_iterative_pairwise_then_heuristic"] = iterative
        metadata[
            "student_iterative_pairwise_then_heuristic"
        ] = iterative_metadata
    return levels, conditions, metadata


def _evaluate_holdout(
    exp,
    model,
    detect,
    rows,
    student,
    ranker,
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
        desc="leak-free holdout",
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
                int(pair.shape[-1]),
                student,
                ranker,
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
            target_by_name = dict(zip(names, target_results, strict=True))
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
                    "baseline_target_conf": target_by_name["observed"][
                        "post_target_conf"
                    ],
                    "corrected_target_conf": target_by_name[name][
                        "post_target_conf"
                    ],
                    "full_detection_f1": full_set["detection_f1"],
                    **metadata[name],
                })
            _reset_detect_inference_cache(detect)
            release_accelerator_memory()
    return pd.DataFrame(output)


def _summarize(evaluation):
    rows = []
    for condition, group in evaluation.groupby("condition"):
        patched = group[group.input_kind.eq("patched")]
        clean = group[group.input_kind.eq("clean")]
        hidden = patched.baseline_target_detected.eq(0)
        rows.append({
            "condition": condition,
            "patched_n": int(len(patched)),
            "clean_n": int(len(clean)),
            "hidden_n": int(hidden.sum()),
            "hidden_recovered_n": int(
                (
                    hidden
                    & patched.corrected_target_detected.eq(1)
                ).sum()
            ),
            "baseline_lost_n": int(
                (
                    patched.baseline_target_detected.eq(1)
                    & patched.corrected_target_detected.eq(0)
                ).sum()
            ),
            "patched_confidence_gain": float(
                (
                    patched.corrected_target_conf
                    - patched.baseline_target_conf
                ).mean()
            ),
            "clean_target_change_n": int(
                (
                    clean.corrected_target_detected
                    != clean.baseline_target_detected
                ).sum()
            ),
            "clean_full_detection_f1": float(
                clean.full_detection_f1.mean()
            ),
        })
    return pd.DataFrame(rows)


def run_leak_free_defense(config: LeakFreeDefenseConfig) -> Path:
    started = time.time()
    base_run = Path(config.base_run).resolve()
    if config.pool_run is not None:
        pool_run = Path(config.pool_run).resolve()
        pool_metadata = json.loads(
            (pool_run / "run.json").read_text(encoding="utf-8")
        )
        if pool_metadata["status"] != "complete":
            raise RuntimeError(
                "The maximum balanced pool has no completed teacher cache."
            )
        student_rows = pd.read_csv(
            pool_run / "student_train_split.csv"
        )
        ranker_rows = pd.read_csv(
            pool_run / "ranker_train_split.csv"
        )
        holdout_rows = pd.read_csv(pool_run / "holdout_split.csv")
        teacher_cache_dir = Path(pool_metadata["teacher_cache_dir"])
    else:
        base_metadata = json.loads(
            (base_run / "run.json").read_text(encoding="utf-8")
        )
        original_train = pd.read_csv(base_run / "train_split.csv")
        original_test = pd.read_csv(base_run / "test_split.csv")
        student_rows, ranker_rows = _student_ranker_split(
            original_train,
            config.student_train_scenes,
            config.ranker_train_scenes,
            config.seed,
        )
        pool, _ = _load_inputs(
            Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None
        )
        previously_used_paths = set(
            pd.concat([original_train, original_test]).path.astype(str)
        )
        holdout_rows = pool[
            ~pool.path.astype(str).isin(previously_used_paths)
        ].copy().reset_index(drop=True)
        teacher_cache_dir = Path(base_metadata["teacher_cache_dir"])
    if config.max_student_scenes is not None:
        student_rows = student_rows.head(
            int(config.max_student_scenes)
        ).copy()
    if config.max_ranker_scenes is not None:
        ranker_rows = ranker_rows.head(
            int(config.max_ranker_scenes)
        ).copy()
    if config.max_holdout_scenes is not None:
        holdout_rows = holdout_rows.head(
            int(config.max_holdout_scenes)
        ).copy()
    splits = {
        "student_train": student_rows,
        "ranker_train": ranker_rows,
        "holdout": holdout_rows,
    }
    for left_name, left in splits.items():
        for right_name, right in splits.items():
            if left_name >= right_name:
                continue
            if set(left.path.astype(str)) & set(right.path.astype(str)):
                raise AssertionError(
                    f"Path leakage between {left_name} and {right_name}."
                )

    exp, _cache_path = load_experiment(
        prefer_device=config.device, require_device=config.require_device
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    manifest = json.loads(
        (teacher_cache_dir / "manifest.json").read_text(encoding="utf-8")
    )
    teacher_rows = pd.concat(
        [student_rows, ranker_rows], ignore_index=True
    )
    row_lookup = {
        str(row.example_id): row
        for row in teacher_rows.itertuples(index=False)
    }
    student_records = _record_lookup(exp, student_rows, manifest)
    ranker_records = _record_lookup(exp, ranker_rows, manifest)
    for record in [*student_records, *ranker_records]:
        record["row"] = row_lookup[record["example_id"]]

    student_config = LargeComponentStudentConfig(
        device=config.device,
        require_device=config.require_device,
        feature_sets=("functional",),
        modes=("blind_support",),
        blind_selector="hybrid",
        blind_candidates_per_level=config.candidate_budget,
        blind_apply_top_k=config.student_apply_top_k,
        diagnostic_ablations=False,
        seed=config.seed,
    )
    base_config = student_config
    reference_mean, reference_std = _attacked_reference_statistics(
        exp, model, detect, student_records
    )
    training_rows, _localization = _build_training_rows(
        exp,
        model,
        detect,
        student_records,
        reference_mean,
        reference_std,
        student_config,
    )
    students, fit_metadata = _fit_students(
        training_rows, student_config
    )
    student = students[("blind_support", "functional")]
    target_scale = float(
        fit_metadata["blind_support"]["target_scale"]
    )
    ranker_frame = _ranker_training_frame(
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
    ranker, training_pairs = _fit_pairwise_ranker(
        ranker_frame, config
    )
    evaluation = _evaluate_holdout(
        exp,
        model,
        detect,
        holdout_rows,
        student,
        ranker,
        target_scale,
        reference_mean,
        reference_std,
        base_config,
        config,
    )
    summary = _summarize(evaluation)
    payload = {
        **asdict(config),
        **{
            f"{name}_ids": frame.example_id.astype(str).tolist()
            for name, frame in splits.items()
        },
    }
    run_dir = (
        Path(config.output_dir)
        / f"leak_free_{stable_hash(payload)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(run_dir / f"{name}_split.csv", index=False)
    ranker_frame.to_csv(
        run_dir / "ranker_training_rows.csv", index=False
    )
    evaluation.to_csv(run_dir / "evaluation_rows.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    joblib.dump(student, run_dir / "component_student.joblib")
    joblib.dump(ranker, run_dir / "cluster_ranker.joblib")
    audit = {
        "student_ranker_path_overlap": 0,
        "student_holdout_path_overlap": 0,
        "ranker_holdout_path_overlap": 0,
        "holdout_teacher_records_loaded": False,
        "holdout_teacher_values_used_for_correction": False,
        "holdout_target_boxes_used_for_metrics_only": True,
        "student_train_scenes": len(student_rows),
        "ranker_train_scenes": len(ranker_rows),
        "holdout_scenes": len(holdout_rows),
    }
    (run_dir / "leakage_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "elapsed_seconds": time.time() - started,
                "training_pairs": training_pairs,
                "feature_columns": FEATURE_COLUMNS,
                "config": asdict(config),
                "audit": audit,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_dir


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a scene-disjoint, student-valued closed-loop "
            "component defense without loading holdout teacher components."
        )
    )
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--pool-run")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = LeakFreeDefenseConfig(
        base_run=args.base_run,
        pool_run=args.pool_run,
        output_dir=args.output_dir,
        device=args.device,
        require_device=args.require_device,
    )
    if args.smoke:
        config.max_student_scenes = 4
        config.max_ranker_scenes = 4
        config.max_holdout_scenes = 2
        config.pair_samples_per_scene = 40
        config.pairwise_max_iter = 12
    print(run_leak_free_defense(config))


if __name__ == "__main__":
    main()
