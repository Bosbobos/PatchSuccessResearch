from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from . import AttackConfig, ExperimentConfig, PatchSuccessExperiment


def ensure_new_experiments_cwd() -> Path:
    """Run notebooks from the historical new_experiments cwd to reuse cache keys."""

    cwd = Path.cwd()
    if (cwd / "new_experiments" / "patch_success_analysis").exists():
        os.chdir(cwd / "new_experiments")
    elif not (cwd / "patch_success_analysis").exists():
        raise RuntimeError(f"Unexpected cwd={cwd}; run from repo root or new_experiments")
    current = Path.cwd()
    if str(current) not in sys.path:
        sys.path.insert(0, str(current))
    return current


def make_celeb_patch_success_experiment(
    *,
    pool_size: int = 7200,
    n_success: int = 3000,
    n_fail: int = 3000,
    target_layer: str = "model.22",
    n_steps: int = 8,
    alpha_batch_size: int = 8,
    metrics_batch_size: int = 1024,
    device: str | None = "mps",
) -> tuple[AttackConfig, ExperimentConfig, PatchSuccessExperiment]:
    attack_config = AttackConfig(
        dataset_path=["../datasets/celeb_fbi_640"],
        patch_path="../data/patch.png",
        model_path="yolo11s.pt",
        output_dir="new_experiments/outputs/patch_success_analysis",
        pool_size=int(pool_size),
        n_success=int(n_success),
        n_fail=int(n_fail),
        inference_batch_size=32,
        show_progress=True,
        success_thresh=0.30,
        seed=17,
        device=device,
        conf=0.01,
    )
    config = ExperimentConfig(
        attack=attack_config,
        target_layer=target_layer,
        n_steps=int(n_steps),
        alpha_batch_size=int(alpha_batch_size),
        metrics_batch_size=int(metrics_batch_size),
    )
    return attack_config, config, PatchSuccessExperiment(config)


def make_coco_people_patch_success_experiment(
    *,
    pool_size: int = 7200,
    n_success: int = 3000,
    n_fail: int = 3000,
    target_layer: str = "model.22",
    n_steps: int = 8,
    alpha_batch_size: int = 8,
    metrics_batch_size: int = 1024,
    device: str | None = "mps",
) -> tuple[AttackConfig, ExperimentConfig, PatchSuccessExperiment]:
    attack_config = AttackConfig(
        dataset_path=["../datasets/COCO_people"],
        patch_path="../data/patch.png",
        model_path="yolo11s.pt",
        output_dir="new_experiments/outputs/patch_success_analysis",
        pool_size=int(pool_size),
        n_success=int(n_success),
        n_fail=int(n_fail),
        inference_batch_size=32,
        show_progress=True,
        success_thresh=0.30,
        seed=17,
        device=device,
        conf=0.01,
    )
    config = ExperimentConfig(
        attack=attack_config,
        target_layer=target_layer,
        n_steps=int(n_steps),
        alpha_batch_size=int(alpha_batch_size),
        metrics_batch_size=int(metrics_batch_size),
    )
    return attack_config, config, PatchSuccessExperiment(config)


def load_segmentig_success_failure_rows(
    exp: PatchSuccessExperiment,
    *,
    force: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    sf = exp.run_segmentig_success_failure_metrics(force=force)
    rows_df = pd.DataFrame(sf["rows"])
    quality_df = pd.DataFrame(sf["quality"]).sort_values(["best_balanced_accuracy", "roc_auc", "best_accuracy"], ascending=False)
    forbidden_metric_prefixes = ("full" + "_ig_", "n" + "aa_")
    forbidden_columns = [col for col in rows_df.columns if col.startswith(forbidden_metric_prefixes)]
    if forbidden_columns:
        raise RuntimeError(f"Unexpected legacy columns in SegmentIG-only sf: {forbidden_columns[:10]}")
    return sf, rows_df, quality_df


def select_balanced_rows(rows_df: pd.DataFrame, *, per_class: int) -> pd.DataFrame:
    success_all = rows_df[rows_df["success"].astype(bool)]
    fail_all = rows_df[~rows_df["success"].astype(bool)]
    limit = min(int(per_class), len(success_all), len(fail_all))
    if limit <= 0:
        raise RuntimeError(f"Need at least one success/fail row, got {len(success_all)} / {len(fail_all)}")
    success_rows = success_all.head(limit)
    fail_rows = fail_all.head(limit)
    return pd.concat([success_rows, fail_rows], ignore_index=True)


def resolve_existing_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    candidates = [path, Path.cwd() / path]
    text = str(path)
    if text.startswith("new_experiments/outputs/"):
        candidates.append(Path("new_experiments") / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path_like)
