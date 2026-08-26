from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _preprocess_pair
from .balanced_target_path import _assign_groups
from .candidate_reserve import _cache_lookup, _evaluate_batch
from .common import REPO_ROOT, load_experiment, release_accelerator_memory, stable_hash
from .component_aware_cluster_ranker import (
    _pairwise_scores,
    _scene_features,
)
from .component_student import _capture_with_grad, _reset_detect_inference_cache
from .followup_common import TRACE_DB
from .large_component_student import (
    LargeComponentStudentConfig,
    _blind_candidate_indices,
)
from .learned_cluster_ranker import (
    _average_gradient_for_clusters,
    _clusters_for_levels,
)
from .leak_free_component_defense import _predict_maps
from .leak_free_no_iou_expansion import _balanced_limit
from .mechanism_followup import _head_branches
from .prepare_max_balanced_component_pool import LABELS_CSV
from .self_counterfactual_defense import _all_class_nms, _detection_set_metrics


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "CandidateRoutingAndAttackPath" / "level_ablation_outputs"
)
LEVEL_CONDITIONS = {
    "all_levels": (0, 1, 2),
    "levels_1_2": (1, 2),
    "level_2_only": (2,),
    "level_1_only": (1,),
}


@dataclass(slots=True)
class LevelAblationConfig:
    prior_run: str
    pool_run: str
    state_run: str
    eda_run: str
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "mps"
    require_device: bool = False
    student_feature_set: str = "functional"
    top_clusters: int = 5
    candidate_budget: int = 8000
    student_apply_top_k: int = 12000
    final_per_group: int = 100
    final_groups: tuple[str, ...] = (
        "hidden_low_conf_match",
        "hidden_no_iou_match",
        "visible_target_winner",
    )
    max_validation_scenes: int | None = None
    max_final_scenes: int | None = None
    seed: int = 3301
    method_version: int = 1


def _known_evaluation_paths() -> set[str]:
    roots_and_patterns = (
        (
            REPO_ROOT
            / "CandidateRoutingAndAttackPath"
            / "large_component_student_outputs",
            "large_student_*/test_split.csv",
        ),
        (
            REPO_ROOT
            / "CandidateRoutingAndAttackPath"
            / "leak_free_defense_outputs",
            "leak_free_*/holdout_split.csv",
        ),
        (
            REPO_ROOT
            / "CandidateRoutingAndAttackPath"
            / "no_iou_expansion_outputs",
            "no_iou_expansion_*/fresh_final_split.csv",
        ),
        (
            REPO_ROOT
            / "CandidateRoutingAndAttackPath"
            / "level_ablation_outputs",
            "level_ablation_*/fresh_final_split.csv",
        ),
    )
    paths: set[str] = set()
    for root, pattern in roots_and_patterns:
        for csv_path in root.glob(pattern):
            try:
                frame = pd.read_csv(csv_path, usecols=["path"])
            except (ValueError, OSError):
                continue
            paths.update(frame.path.astype(str))
    return paths


def _fresh_final_split(config: LevelAblationConfig) -> tuple[pd.DataFrame, int]:
    excluded_paths = _known_evaluation_paths()
    pool_run = Path(config.pool_run).resolve()
    for split_name in (
        "student_train_split.csv",
        "ranker_train_split.csv",
        "holdout_split.csv",
    ):
        excluded_paths.update(
            pd.read_csv(
                pool_run / split_name, usecols=["path"]
            ).path.astype(str)
        )
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
    available = trace.merge(
        labels,
        on="example_id",
        how="inner",
        validate="one_to_one",
    )
    available = available[
        ~available.path.astype(str).isin(excluded_paths)
    ].copy()
    parts = []
    for index, group in enumerate(config.final_groups):
        current = (
            available[available.analysis_group.eq(group)]
            .drop_duplicates("path")
        )
        if len(current) < int(config.final_per_group):
            raise RuntimeError(
                f"Only {len(current)} unseen scenes remain for {group!r}; "
                f"{config.final_per_group} requested."
            )
        parts.append(
            current.sample(
                n=int(config.final_per_group),
                random_state=int(config.seed) + index,
            )
        )
    final = pd.concat(parts, ignore_index=True)
    if set(final.path.astype(str)) & excluded_paths:
        raise AssertionError("Fresh final overlaps known evaluation paths.")
    return final, len(excluded_paths)


def _load_student_state(state_run: Path):
    state_path = state_run / "student_inference_state.npz"
    with np.load(state_path, allow_pickle=False) as state:
        count = int(state["level_count"])
        means = [
            state[f"reference_mean_{level}"]
            for level in range(count)
        ]
        stds = [
            state[f"reference_std_{level}"]
            for level in range(count)
        ]
        target_scale = float(state["target_scale"])
    return means, stds, target_scale


def _level_conditions_for_image(
    model,
    detect,
    image,
    student,
    ranker,
    reference_mean,
    reference_std,
    target_scale,
    base_config,
    config,
    condition_names,
):
    _reset_detect_inference_cache(detect)
    image = image.detach().requires_grad_(True)
    decoded, levels = _capture_with_grad(model, detect, image)
    clusters = _clusters_for_levels(detect, levels, base_config)
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
    scores = _pairwise_scores(ranker, features)
    selected = np.argsort(-scores)[
        : int(config.top_clusters)
    ].astype(int).tolist()
    gradients = _average_gradient_for_clusters(
        decoded, levels, [clusters[index] for index in selected]
    )
    candidates = _blind_candidate_indices(
        levels,
        gradients,
        reference_mean,
        reference_std,
        int(config.candidate_budget),
        "hybrid",
    )
    correction, metadata = _predict_maps(
        student,
        levels,
        gradients,
        candidates,
        reference_mean,
        reference_std,
        target_scale,
        int(config.student_apply_top_k),
    )
    conditions = {}
    condition_metadata = {}
    for name in condition_names:
        active = set(LEVEL_CONDITIONS[name])
        conditions[name] = [
            level - delta if level_index in active else level
            for level_index, (level, delta) in enumerate(
                zip(levels, correction, strict=True)
            )
        ]
        condition_metadata[name] = {
            **metadata,
            "active_levels": ",".join(
                str(level) for level in sorted(active)
            ),
            "active_level_count": len(active),
        }
    return levels, conditions, condition_metadata


def _evaluate(
    exp,
    model,
    detect,
    rows,
    student,
    ranker,
    reference_mean,
    reference_std,
    target_scale,
    base_config,
    config,
    condition_names,
    description,
):
    import torch

    examples = _cache_lookup(exp)
    output = []
    for row in tqdm(
        rows.itertuples(index=False),
        total=len(rows),
        desc=description,
        unit="scene",
    ):
        example = examples[str(row.example_id)]
        clean, patched, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean, patched)
        # Each branch is processed independently. The mechanism never compares
        # clean and patched tensors; clean is evaluated only as a safety check.
        for input_kind, image in (
            ("patched", pair[1:2]),
            ("clean", pair[0:1]),
        ):
            levels, conditions, metadata = _level_conditions_for_image(
                model,
                detect,
                image,
                student,
                ranker,
                reference_mean,
                reference_std,
                target_scale,
                base_config,
                config,
                condition_names,
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
    for condition, frame in evaluation.groupby("condition"):
        patched = frame[frame.input_kind.eq("patched")]
        clean = frame[frame.input_kind.eq("clean")]
        for analysis_group in (
            "all", *sorted(patched.analysis_group.unique())
        ):
            current = (
                patched
                if analysis_group == "all"
                else patched[
                    patched.analysis_group.eq(analysis_group)
                ]
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
                "confidence_gain": float(
                    (
                        current.corrected_target_conf
                        - current.baseline_target_conf
                    ).mean()
                ),
                "clean_target_change_n": (
                    int(
                        (
                            clean.corrected_target_detected
                            != clean.baseline_target_detected
                        ).sum()
                    )
                    if analysis_group == "all" else np.nan
                ),
                "clean_full_detection_f1": (
                    float(clean.full_detection_f1.mean())
                    if analysis_group == "all" else np.nan
                ),
            })
    return pd.DataFrame(rows)


def _select_condition(summary):
    overall = summary[summary.analysis_group.eq("all")].set_index(
        "condition"
    )
    no_iou = summary[
        summary.analysis_group.eq("hidden_no_iou_match")
    ].set_index("condition")
    baseline = overall.loc["all_levels"]
    baseline_no_iou = int(
        no_iou.loc["all_levels", "hidden_recovered_n"]
    )
    candidates = []
    for condition, row in overall.iterrows():
        safe = (
            int(row.baseline_lost_n)
            <= int(baseline.baseline_lost_n)
            and int(row.clean_target_change_n)
            <= int(baseline.clean_target_change_n)
            and float(row.clean_full_detection_f1)
            >= float(baseline.clean_full_detection_f1) - 0.005
        )
        no_iou_recovered = int(
            no_iou.loc[condition, "hidden_recovered_n"]
        )
        preserves_no_iou = no_iou_recovered >= baseline_no_iou - 1
        if safe and preserves_no_iou:
            candidates.append({
                "condition": condition,
                "hidden_recovered_n": int(row.hidden_recovered_n),
                "no_iou_recovered_n": no_iou_recovered,
                "active_level_count": len(
                    LEVEL_CONDITIONS[condition]
                ),
                "clean_full_detection_f1": float(
                    row.clean_full_detection_f1
                ),
            })
    if not candidates:
        raise RuntimeError("No level condition passed the safety gate.")
    selected = sorted(
        candidates,
        key=lambda item: (
            -item["hidden_recovered_n"],
            item["active_level_count"],
            -item["no_iou_recovered_n"],
            {
                "level_2_only": 0,
                "levels_1_2": 1,
                "all_levels": 2,
                "level_1_only": 3,
            }[item["condition"]],
            -item["clean_full_detection_f1"],
        ),
    )[0]
    return selected, candidates


def run_level_ablation(config: LevelAblationConfig) -> Path:
    started = time.time()
    prior_run = Path(config.prior_run).resolve()
    pool_run = Path(config.pool_run).resolve()
    state_run = Path(config.state_run).resolve()
    eda_run = Path(config.eda_run).resolve()
    pool_meta = json.loads(
        (pool_run / "run.json").read_text(encoding="utf-8")
    )
    prior_meta = json.loads(
        (prior_run / "run.json").read_text(encoding="utf-8")
    )
    state_meta = json.loads(
        (state_run / "run.json").read_text(encoding="utf-8")
    )
    eda_meta = json.loads(
        (eda_run / "run.json").read_text(encoding="utf-8")
    )
    if pool_meta["status"] != "complete":
        raise RuntimeError("Pool run is incomplete.")
    if Path(prior_meta["config"]["pool_run"]).resolve() != pool_run:
        raise RuntimeError("Prior defense and pool do not match.")
    if Path(state_meta["config"]["pool_run"]).resolve() != pool_run:
        raise RuntimeError("State run and pool do not match.")
    if Path(eda_meta["config"]["pool_run"]).resolve() != pool_run:
        raise RuntimeError("EDA run and pool do not match.")
    validation = pd.read_csv(pool_run / "holdout_split.csv")
    validation = _balanced_limit(
        validation, config.max_validation_scenes
    )
    final, known_eval_path_count = _fresh_final_split(config)
    final = _balanced_limit(final, config.max_final_scenes)
    payload = {
        **asdict(config),
        "validation_ids": validation.example_id.astype(str).tolist(),
        "final_ids": final.example_id.astype(str).tolist(),
        "conditions": LEVEL_CONDITIONS,
    }
    run_dir = (
        Path(config.output_dir)
        / f"level_ablation_{stable_hash(payload)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(run_dir / "validation_split.csv", index=False)
    final.to_csv(run_dir / "fresh_final_split.csv", index=False)

    print("[1/4] Loading the fixed student, ranker and inference state...", flush=True)
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
    student = joblib.load(prior_run / "component_student.joblib")
    ranker = joblib.load(prior_run / "cluster_ranker.joblib")
    reference_mean, reference_std, target_scale = _load_student_state(
        state_run
    )
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
    print("[2/4] Evaluating all level masks on observed validation...", flush=True)
    validation_rows = _evaluate(
        exp,
        model,
        detect,
        validation,
        student,
        ranker,
        reference_mean,
        reference_std,
        target_scale,
        base_config,
        config,
        list(LEVEL_CONDITIONS),
        "level-mask validation",
    )
    validation_summary = _summary(validation_rows)
    selected, eligible = _select_condition(validation_summary)
    selected_name = selected["condition"]
    print(
        f"      Locked final condition: {selected_name}",
        flush=True,
    )
    print("[3/4] Evaluating baseline and locked condition on fresh final...", flush=True)
    final_conditions = list(dict.fromkeys(
        ["all_levels", selected_name]
    ))
    final_rows = _evaluate(
        exp,
        model,
        detect,
        final,
        student,
        ranker,
        reference_mean,
        reference_std,
        target_scale,
        base_config,
        config,
        final_conditions,
        "fresh level-mask final",
    )
    final_summary = _summary(final_rows)
    validation_rows.to_csv(
        run_dir / "validation_evaluation_rows.csv", index=False
    )
    validation_summary.to_csv(
        run_dir / "validation_summary.csv", index=False
    )
    final_rows.to_csv(run_dir / "final_evaluation_rows.csv", index=False)
    final_summary.to_csv(run_dir / "final_summary.csv", index=False)
    selection = {
        "rule": (
            "no more visible losses or clean target changes than all-levels; "
            "clean F1 no more than 0.005 below all-levels; no-IoU recovery "
            "no more than one below all-levels; maximize total hidden "
            "recovery, then use fewer active levels; prefer level 2 over "
            "diagnostic level 1 on an exact tie"
        ),
        "selected": selected,
        "eligible": eligible,
    }
    (run_dir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    validation_paths = set(validation.path.astype(str))
    final_paths = set(final.path.astype(str))
    train_paths = set(
        pd.concat(
            [
                pd.read_csv(pool_run / "student_train_split.csv"),
                pd.read_csv(pool_run / "ranker_train_split.csv"),
            ],
            ignore_index=True,
        ).path.astype(str)
    )
    audit = {
        "mechanism_input": "one_image_only",
        "patched_clean_pair_compared_by_mechanism": False,
        "student_and_ranker_frozen": True,
        "validation_scenes": int(len(validation)),
        "fresh_final_scenes": int(len(final)),
        "known_prior_evaluation_paths_excluded": int(
            known_eval_path_count
        ),
        "train_final_path_overlap": int(len(train_paths & final_paths)),
        "validation_final_path_overlap": int(
            len(validation_paths & final_paths)
        ),
        "final_teacher_records_loaded": False,
        "final_target_boxes_used_for_metrics_only": True,
        "condition_selected_before_final_evaluation": selected_name,
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
                "level_conditions": LEVEL_CONDITIONS,
                "selection": selection,
                "audit": audit,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[4/4] Complete: {run_dir}", flush=True)
    return run_dir


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Mask a fixed attacked-image student correction by Detect feature "
            "level, select on prior validation, and evaluate once on fresh data."
        )
    )
    parser.add_argument("--prior-run", required=True)
    parser.add_argument("--pool-run", required=True)
    parser.add_argument("--state-run", required=True)
    parser.add_argument("--eda-run", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = LevelAblationConfig(
        prior_run=args.prior_run,
        pool_run=args.pool_run,
        state_run=args.state_run,
        eda_run=args.eda_run,
        output_dir=args.output_dir,
        device=args.device,
        require_device=args.require_device,
    )
    if args.smoke:
        config.max_validation_scenes = 8
        config.max_final_scenes = 6
        config.candidate_budget = 300
        config.student_apply_top_k = 450
    print(run_level_ablation(config))


if __name__ == "__main__":
    main()
