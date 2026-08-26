from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .followup_common import TRACE_DB


LABELS_CSV = (
    TRACE_DB.parents[1]
    / "target_instance_all_metrics"
    / "target_instance_bc9b647edb5cdec7"
    / "target_instance_labels.csv"
)
DEFAULT_OUTPUT_DIR = TRACE_DB.parents[1] / "ensemble_margin_v1"


@dataclass(slots=True)
class EnsembleMarginConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    target_iou: float = 0.50
    detection_conf: float = 0.25
    candidate_min_score: float = 0.01
    train_fraction: float = 0.60
    bootstrap_repeats: int = 500
    seed: int = 19


def _iou_to_target(frame: pd.DataFrame) -> np.ndarray:
    ix1 = np.maximum(frame.bbox_x1, frame.clean_target_x1)
    iy1 = np.maximum(frame.bbox_y1, frame.clean_target_y1)
    ix2 = np.minimum(frame.bbox_x2, frame.clean_target_x2)
    iy2 = np.minimum(frame.bbox_y2, frame.clean_target_y2)
    intersection = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_candidate = np.maximum(0, frame.bbox_x2 - frame.bbox_x1) * np.maximum(
        0, frame.bbox_y2 - frame.bbox_y1
    )
    area_target = np.maximum(0, frame.clean_target_x2 - frame.clean_target_x1) * np.maximum(
        0, frame.clean_target_y2 - frame.clean_target_y1
    )
    return intersection / np.maximum(area_candidate + area_target - intersection, 1e-12)


def _noisy_or(values) -> float:
    scores = np.clip(np.asarray(values, dtype=float), 0.0, 1.0 - 1e-9)
    return float(1.0 - np.prod(1.0 - scores)) if len(scores) else 0.0


def _logsumexp(values) -> float:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return -30.0
    maximum = float(array.max())
    return maximum + float(np.log(np.exp(array - maximum).sum()))


def _effective_count(values) -> float:
    scores = np.asarray(values, dtype=float)
    return float(scores.sum() ** 2 / max(float(np.square(scores).sum()), 1e-12)) if len(scores) else 0.0


def _split(example_id: str, fraction: float) -> str:
    value = int(hashlib.sha256(str(example_id).encode()).hexdigest()[:12], 16) % 10_000
    return "train" if value < int(fraction * 10_000) else "test"


def load_candidate_frames(config: EnsembleMarginConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(TRACE_DB) as connection:
        examples = pd.read_sql_query(
            "SELECT example_id,clean_target_x1,clean_target_y1,clean_target_x2,clean_target_y2,"
            "clean_target_flat,patched_tracked_score,patched_tracked_logit,patched_winner_score,error FROM examples",
            connection,
        )
        candidates = pd.read_sql_query(
            "SELECT example_id,variant,rank,flat_index,level_index,decoded_score,class_logit,"
            "bbox_x1,bbox_y1,bbox_x2,bbox_y2,nms_survived,nms_conf FROM candidates",
            connection,
        )
    labels = pd.read_csv(LABELS_CSV)
    labels = labels.loc[labels.target_eligible.eq(1), [
        "example_id", "target_detected", "target_hidden", "outcome", "patched_person_count"
    ]]
    examples = examples.loc[examples.error.isna()].drop(columns="error").merge(
        labels, on="example_id", validate="one_to_one"
    )
    candidates = candidates[candidates.example_id.isin(examples.example_id)].merge(
        examples[[
            "example_id", "clean_target_x1", "clean_target_y1",
            "clean_target_x2", "clean_target_y2", "clean_target_flat",
            "patched_tracked_score", "patched_tracked_logit",
        ]],
        on="example_id", validate="many_to_one",
    )
    candidates["target_iou"] = _iou_to_target(candidates)
    clean_members = candidates[
        (candidates.variant == "clean")
        & (candidates.target_iou >= config.target_iou)
        & (candidates.decoded_score >= config.candidate_min_score)
    ][["example_id", "flat_index"]].drop_duplicates()
    clean_members["in_clean_target_set"] = 1
    candidates = candidates.merge(clean_members, on=["example_id", "flat_index"], how="left")
    candidates["in_clean_target_set"] = candidates.in_clean_target_set.fillna(0).astype(bool)
    return examples, candidates


def _aggregate_one(item, config: EnsembleMarginConfig) -> dict:
    example_id, frame = item
    clean_set = frame[(frame.variant == "clean") & frame.in_clean_target_set]
    patched_set = frame[(frame.variant == "patched") & frame.in_clean_target_set]
    current = frame[(frame.variant == "patched") & (frame.target_iou >= config.target_iou)]
    non_target = frame[(frame.variant == "patched") & (frame.target_iou < 0.10)]
    post = current[(current.nms_survived == 1) & (current.nms_conf >= config.candidate_min_score)]
    clean_scores = clean_set.decoded_score.to_numpy(float)
    patched_scores = patched_set.decoded_score.to_numpy(float)
    patched_logits = patched_set.class_logit.to_numpy(float)
    # The trace stores only top-50 patched candidates. The tracked clean cell
    # can fall below that cutoff, but its patched score/logit are saved in the
    # examples table. It is a member of the clean target set by construction,
    # so add it back instead of letting truncation make clean_set_max < tracked.
    target_flat = int(frame.clean_target_flat.iloc[0])
    target_is_clean_member = bool((clean_set.flat_index == target_flat).any())
    target_is_saved_patched = bool((patched_set.flat_index == target_flat).any())
    if target_is_clean_member and not target_is_saved_patched:
        patched_scores = np.append(patched_scores, float(frame.patched_tracked_score.iloc[0]))
        patched_logits = np.append(patched_logits, float(frame.patched_tracked_logit.iloc[0]))
    current_scores = current.decoded_score.to_numpy(float)
    post_score = float(post.nms_conf.max()) if len(post) else 0.0
    post_winner = post.loc[post.nms_conf.idxmax()] if len(post) else None
    if post_score < config.detection_conf:
        lineage_outcome = "target_hidden"
    elif int(post_winner.flat_index) == target_flat:
        lineage_outcome = "original_tracked_candidate"
    elif bool(post_winner.in_clean_target_set):
        lineage_outcome = "other_preexisting_clean_candidate"
    else:
        lineage_outcome = "new_recruited_candidate"
    return {
        "example_id": example_id,
        "clean_set_size": len(clean_set),
        "clean_set_max_clean": float(clean_scores.max()) if len(clean_scores) else 0.0,
        "clean_set_max_patch": float(patched_scores.max()) if len(patched_scores) else 0.0,
        "clean_set_sum_clean": float(clean_scores.sum()),
        "clean_set_sum_patch": float(patched_scores.sum()),
        "clean_set_noisy_or_clean": _noisy_or(clean_scores),
        "clean_set_noisy_or_patch": _noisy_or(patched_scores),
        "clean_set_lse_clean": _logsumexp(clean_set.class_logit),
        "clean_set_lse_patch": _logsumexp(patched_logits),
        "clean_set_effective_clean": _effective_count(clean_scores),
        "clean_set_effective_patch": _effective_count(patched_scores),
        "clean_set_n025_patch": int((patched_scores >= config.detection_conf).sum()),
        "clean_set_n010_patch": int((patched_scores >= 0.10).sum()),
        "current_max_patch": float(current_scores.max()) if len(current_scores) else 0.0,
        "current_sum_patch": float(current_scores.sum()),
        "current_noisy_or_patch": _noisy_or(current_scores),
        "current_n025_patch": int((current_scores >= config.detection_conf).sum()),
        "current_n010_patch": int((current_scores >= 0.10).sum()),
        "non_target_max_patch": float(non_target.decoded_score.max()) if len(non_target) else 0.0,
        "post_target_score": post_score,
        "post_target_flat": int(post_winner.flat_index) if post_winner is not None else -1,
        "lineage_outcome": lineage_outcome,
        "trace_target_detected": int(post_score >= config.detection_conf),
    }


def build_features(
    examples: pd.DataFrame,
    candidates: pd.DataFrame,
    config: EnsembleMarginConfig,
) -> pd.DataFrame:
    groups = candidates.groupby("example_id", sort=False)
    rows = [
        _aggregate_one(item, config)
        for item in tqdm(groups, total=groups.ngroups, desc="ensemble margins", unit="image")
    ]
    features = examples.merge(pd.DataFrame(rows), on="example_id", validate="one_to_one")
    features["trace_target_hidden"] = 1 - features.trace_target_detected
    features["old_trace_agree"] = features.target_detected.astype(int).eq(features.trace_target_detected)
    features["tracked_margin"] = features.patched_tracked_score - config.detection_conf
    features["clean_set_max_margin"] = features.clean_set_max_patch - config.detection_conf
    features["current_max_margin"] = features.current_max_patch - config.detection_conf
    features["clean_set_max_drop"] = features.clean_set_max_clean - features.clean_set_max_patch
    features["clean_set_sum_ratio"] = features.clean_set_sum_patch / features.clean_set_sum_clean.clip(lower=1e-8)
    features["clean_set_lse_drop"] = features.clean_set_lse_clean - features.clean_set_lse_patch
    features["target_competitor_margin"] = features.current_max_patch - features.non_target_max_patch
    features["split"] = features.example_id.map(lambda value: _split(value, config.train_fraction))
    return features


MODEL_FEATURES = {
    "tracked_cell": ["patched_tracked_score"],
    "clean_set_max": ["clean_set_max_patch"],
    "clean_set_ensemble": [
        "clean_set_max_patch", "clean_set_sum_patch", "clean_set_noisy_or_patch",
        "clean_set_lse_patch", "clean_set_effective_patch", "clean_set_n025_patch",
        "clean_set_n010_patch", "clean_set_max_drop", "clean_set_sum_ratio", "clean_set_lse_drop",
    ],
    "current_target_ensemble": [
        "current_max_patch", "current_sum_patch", "current_noisy_or_patch",
        "current_n025_patch", "current_n010_patch",
    ],
}


def fit_models(features: pd.DataFrame, config: EnsembleMarginConfig):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score, precision_score,
        recall_score, roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train = features[features.split == "train"].copy()
    test = features[features.split == "test"].copy()
    y_train = train.trace_target_hidden.astype(int)
    y_test = test.trace_target_hidden.astype(int)
    models, predictions, metric_rows = {}, {}, []
    for name, columns in tqdm(MODEL_FEATURES.items(), desc="fit margin models"):
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.seed)),
        ])
        model.fit(train[columns], y_train)
        train_probability = model.predict_proba(train[columns])[:, 1]
        test_probability = model.predict_proba(test[columns])[:, 1]
        thresholds = np.linspace(0.01, 0.99, 199)
        threshold = max(
            thresholds,
            key=lambda value: balanced_accuracy_score(y_train, train_probability >= value),
        )
        predicted = test_probability >= threshold
        models[name] = model
        predictions[name] = test_probability
        metric_rows.append({
            "model": name, "n_train": len(train), "n_test": len(test), "threshold": threshold,
            "roc_auc": roc_auc_score(y_test, test_probability),
            "average_precision": average_precision_score(y_test, test_probability),
            "accuracy": accuracy_score(y_test, predicted),
            "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
            "precision": precision_score(y_test, predicted), "recall": recall_score(y_test, predicted),
        })
    rng = np.random.default_rng(config.seed)
    bootstrap_rows = []
    for name, probability in predictions.items():
        values = []
        for _ in tqdm(range(config.bootstrap_repeats), desc=f"bootstrap {name}", leave=False):
            indices = rng.integers(0, len(test), len(test))
            sampled = y_test.iloc[indices]
            if sampled.nunique() == 2:
                values.append(roc_auc_score(sampled, probability[indices]))
        bootstrap_rows.append({
            "model": name,
            "auc_ci_low": float(np.quantile(values, 0.025)),
            "auc_ci_high": float(np.quantile(values, 0.975)),
        })
    metrics = pd.DataFrame(metric_rows).merge(pd.DataFrame(bootstrap_rows), on="model")
    return metrics.sort_values("roc_auc", ascending=False), models, predictions, test


def run_ensemble_margin(config: EnsembleMarginConfig | None = None):
    config = config or EnsembleMarginConfig()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples, candidates = load_candidate_frames(config)
    features = build_features(examples, candidates, config)
    metrics, models, predictions, test = fit_models(features, config)
    features.to_csv(output_dir / "ensemble_margin_examples.csv", index=False)
    metrics.to_csv(output_dir / "ensemble_margin_model_metrics.csv", index=False)
    indexed = metrics.set_index("model")
    summary = {
        "status": "complete", "n_examples": len(features),
        "trace_hidden_rate": float(features.trace_target_hidden.mean()),
        "old_trace_agreement": float(features.old_trace_agree.mean()),
        "n_pipeline_disagreements": int((~features.old_trace_agree).sum()),
        "tracked_cell_test_auc": float(indexed.loc["tracked_cell", "roc_auc"]),
        "clean_set_ensemble_test_auc": float(indexed.loc["clean_set_ensemble", "roc_auc"]),
        "best_model": str(metrics.iloc[0].model), "best_test_auc": float(metrics.iloc[0].roc_auc),
        "lineage_counts": {str(key): int(value) for key, value in features.lineage_outcome.value_counts().items()},
        "limitations": [
            "Only the saved top-50 person candidates are available.",
            "Current-target ensemble is an endpoint proxy, not an independent mechanism.",
            "Predictive margin complements but does not replace causal interventions.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (output_dir / "analysis_digest.md").write_text(
        "# Ensemble margin\n\n"
        f"- examples: {len(features)}\n"
        f"- trace hidden rate: {summary['trace_hidden_rate']:.3f}\n"
        f"- pipeline disagreements: {summary['n_pipeline_disagreements']}\n"
        f"- tracked-cell test AUC: {summary['tracked_cell_test_auc']:.3f}\n"
        f"- clean-set ensemble test AUC: {summary['clean_set_ensemble_test_auc']:.3f}\n"
        f"- best model: {summary['best_model']} ({summary['best_test_auc']:.3f})\n"
    )
    return {
        "features": features, "candidates": candidates, "metrics": metrics,
        "models": models, "predictions": predictions, "test": test,
        "summary": summary, "output_dir": output_dir,
    }
