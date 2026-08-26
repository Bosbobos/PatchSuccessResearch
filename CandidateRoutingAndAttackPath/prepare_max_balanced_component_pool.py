from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .balanced_target_path import (
    BalancedSelectionConfig,
    build_balanced_target_selection,
)
from .common import REPO_ROOT, load_experiment, stable_hash
from .component_targeted_patch import (
    ComponentTargetedPatchConfig,
    build_teacher_cache,
)
from .followup_common import GROUP_ORDER, TRACE_DB


METRICS_DIR = (
    REPO_ROOT
    / "CandidateRoutingAndAttackPath"
    / "outputs"
    / "target_instance_all_metrics"
)
LABELS_CSV = (
    METRICS_DIR
    / "target_instance_bc9b647edb5cdec7"
    / "target_instance_labels.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "CandidateRoutingAndAttackPath"
    / "max_balanced_pool_outputs"
)


@dataclass(slots=True)
class MaxBalancedPoolConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "mps"
    require_device: bool = False
    student_per_group: int = 192
    ranker_per_group: int = 50
    holdout_per_group: int = 50
    teacher_path_steps: int = 3
    build_teacher_components: bool = True
    seed: int = 2203
    method_version: int = 1


def _split_per_group(rows, config):
    student = []
    ranker = []
    holdout = []
    required = (
        int(config.student_per_group)
        + int(config.ranker_per_group)
        + int(config.holdout_per_group)
    )
    for group_index, group in enumerate(GROUP_ORDER):
        current = (
            rows[rows.analysis_group.eq(group)]
            .sample(
                frac=1.0,
                random_state=int(config.seed) + group_index,
            )
            .reset_index(drop=True)
        )
        if len(current) != required:
            raise RuntimeError(
                f"Expected exactly {required} rows for {group!r}, "
                f"found {len(current)}."
            )
        student_end = int(config.student_per_group)
        ranker_end = student_end + int(config.ranker_per_group)
        student.append(current.iloc[:student_end])
        ranker.append(current.iloc[student_end:ranker_end])
        holdout.append(current.iloc[ranker_end:])
    return tuple(
        pd.concat(parts, ignore_index=True)
        for parts in (student, ranker, holdout)
    )


def run_max_balanced_pool(config: MaxBalancedPoolConfig) -> Path:
    started = time.time()
    print("[1/4] Building the maximum balanced manifest...", flush=True)
    selection = build_balanced_target_selection(
        LABELS_CSV,
        TRACE_DB,
        METRICS_DIR,
        BalancedSelectionConfig(
            n_per_group=10**9,
            seed=41,
            reference_group="visible_non_target_winner",
            method_version=2,
        ),
    )
    balanced_manifest = pd.read_csv(selection.manifest_path)
    with sqlite3.connect(TRACE_DB) as connection:
        trace = pd.read_sql_query(
            "SELECT * FROM examples WHERE error IS NULL", connection
        )
    rows = (
        balanced_manifest[
            ["example_id", "analysis_group", "match_set"]
        ]
        .merge(
            trace,
            on="example_id",
            how="inner",
            validate="one_to_one",
        )
    )
    student, ranker, holdout = _split_per_group(rows, config)
    teacher_rows = pd.concat([student, ranker], ignore_index=True)
    print(
        "[2/4] Split ready: "
        f"student={len(student)}, ranker={len(ranker)}, "
        f"holdout={len(holdout)}.",
        flush=True,
    )
    payload = {
        **asdict(config),
        "balanced_manifest": str(selection.manifest_path),
        "student_ids": student.example_id.astype(str).tolist(),
        "ranker_ids": ranker.example_id.astype(str).tolist(),
        "holdout_ids": holdout.example_id.astype(str).tolist(),
    }
    run_dir = (
        Path(config.output_dir)
        / f"max_balanced_{stable_hash(payload)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    balanced_manifest.to_csv(
        run_dir / "balanced_manifest.csv", index=False
    )
    student.to_csv(run_dir / "student_train_split.csv", index=False)
    ranker.to_csv(run_dir / "ranker_train_split.csv", index=False)
    holdout.to_csv(run_dir / "holdout_split.csv", index=False)

    teacher_cache_dir = None
    teacher_manifest = []
    if config.build_teacher_components:
        print(
            "[3/4] Building/resuming teacher cache for "
            f"{len(teacher_rows)} training scenes...",
            flush=True,
        )
        exp, attack_cache_path = load_experiment(
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
        teacher_config = ComponentTargetedPatchConfig(
            output_dir=str(
                REPO_ROOT
                / "CandidateRoutingAndAttackPath"
                / "component_patch_outputs"
            ),
            device=config.device,
            require_device=config.require_device,
            teacher_path_steps=config.teacher_path_steps,
            seed=config.seed,
            method_version=4,
        )
        teacher_cache_dir, teacher_manifest = build_teacher_cache(
            teacher_config,
            exp=exp,
            model=model,
            detect=detect,
            rows=teacher_rows,
            cache_path=attack_cache_path,
        )
    else:
        print("[3/4] Teacher cache skipped (manifest-only).", flush=True)
    teacher_ids = {
        str(item["example_id"]) for item in teacher_manifest
    }
    expected_teacher_ids = set(teacher_rows.example_id.astype(str))
    if config.build_teacher_components and teacher_ids != expected_teacher_ids:
        raise AssertionError("Teacher cache does not match training partitions.")
    holdout_ids = set(holdout.example_id.astype(str))
    if teacher_ids & holdout_ids:
        raise AssertionError("Holdout teacher component leakage.")
    audit = {
        "balanced_per_group": int(len(rows) / len(GROUP_ORDER)),
        "balanced_total": int(len(rows)),
        "student_train_scenes": int(len(student)),
        "ranker_train_scenes": int(len(ranker)),
        "holdout_scenes": int(len(holdout)),
        "teacher_scenes": int(len(teacher_manifest)),
        "holdout_teacher_scenes": 0,
        "student_ranker_path_overlap": int(
            len(set(student.path) & set(ranker.path))
        ),
        "student_holdout_path_overlap": int(
            len(set(student.path) & set(holdout.path))
        ),
        "ranker_holdout_path_overlap": int(
            len(set(ranker.path) & set(holdout.path))
        ),
    }
    (run_dir / "pool_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": (
                    "complete"
                    if config.build_teacher_components
                    else "manifest_only"
                ),
                "elapsed_seconds": time.time() - started,
                "teacher_cache_dir": (
                    str(teacher_cache_dir)
                    if teacher_cache_dir is not None else None
                ),
                "balanced_selection_dir": str(selection.run_dir),
                "config": asdict(config),
                "audit": audit,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[4/4] Pool complete: {run_dir}", flush=True)
    return run_dir


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build the maximum strictly balanced 292-per-group component "
            "pool and teacher-cache only its student/ranker partitions."
        )
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    print(run_max_balanced_pool(MaxBalancedPoolConfig(
        output_dir=args.output_dir,
        device=args.device,
        require_device=args.require_device,
        build_teacher_components=not args.manifest_only,
    )))


if __name__ == "__main__":
    main()
