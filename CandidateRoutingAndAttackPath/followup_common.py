from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import REPO_ROOT


FOLLOWUP_DIR = REPO_ROOT / "CandidateRoutingAndAttackPath" / "followup_outputs"
TRACE_DB = REPO_ROOT / "CandidateRoutingAndAttackPath" / "outputs" / "candidate_trace_b3c25981c3ad62d5" / "candidate_tracing.sqlite"
ATTACK_PATH_DB = REPO_ROOT / "CandidateRoutingAndAttackPath" / "outputs" / "attack_path_0060cc9878ecdb6e" / "attack_path.sqlite"
MANIFEST_CSV = (
    REPO_ROOT
    / "CandidateRoutingAndAttackPath"
    / "outputs"
    / "target_instance_all_metrics"
    / "balanced_target_selection_5e479438583b28b0"
    / "balanced_target_manifest.csv"
)


GROUP_ORDER = (
    "visible_target_winner",
    "visible_non_target_winner",
    "hidden_low_conf_match",
    "hidden_no_iou_match",
)


def balanced_subset(frame: pd.DataFrame, n_per_group: int | None, *, seed: int) -> pd.DataFrame:
    if n_per_group is None:
        return frame.copy().reset_index(drop=True)
    parts = []
    for index, group in enumerate(GROUP_ORDER):
        subset = frame[frame.analysis_group == group]
        take = min(int(n_per_group), len(subset))
        parts.append(subset.sample(n=take, random_state=int(seed) + index))
    return pd.concat(parts, ignore_index=True)


def bootstrap_ci(values, *, seed: int, repeats: int = 2000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    means = np.asarray([
        rng.choice(array, size=len(array), replace=True).mean() for _ in range(int(repeats))
    ])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_summary(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return target


def box_iou_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-12)
