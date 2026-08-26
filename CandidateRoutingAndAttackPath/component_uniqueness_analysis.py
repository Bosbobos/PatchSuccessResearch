from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .followup_common import FOLLOWUP_DIR


RUN_GLOB = "autonomous_negative_repair_*"
PRIMARY_GATE_SCORE = "max_diffuse_negative_leverage"
GATE_QUANTILES = (0.95, 0.99)


def latest_complete_run(output_dir: Path = FOLLOWUP_DIR) -> Path:
    candidates = []
    for path in output_dir.glob(RUN_GLOB):
        summary_path = path / "summary.json"
        cluster_path = path / "autonomous_cluster_rows.csv"
        repair_path = path / "autonomous_repair_rows.csv"
        if not (summary_path.exists() and cluster_path.exists() and repair_path.exists()):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "complete":
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No complete run matching {RUN_GLOB!r}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _is_hidden(values: pd.Series) -> pd.Series:
    return values.astype(str).str.startswith("hidden")


def _safe_auc(labels: pd.Series | np.ndarray, values: pd.Series | np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    labels = labels[finite]
    values = values[finite]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, values))


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = (
        z
        * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return float(center - half), float(center + half)


def _image_scores(clusters: pd.DataFrame, input_kind: str) -> pd.DataFrame:
    current = clusters[clusters.input_kind.eq(input_kind)].copy()
    current["diffuse_negative_leverage"] = (
        current.total_available_negative_gain
        * (1.0 - current.top_negative_1000_gain_concentration)
        * current.top_negative_1000_gain_per_l2
    )
    scores = current.groupby("example_id", as_index=False).agg(
        max_diffuse_negative_leverage=("diffuse_negative_leverage", "max"),
        max_candidate_gain_min_k500=(
            "top_negative_500_candidate_gain_min",
            "max",
        ),
        max_gain_per_l2_k1000=("top_negative_1000_gain_per_l2", "max"),
        max_total_negative_gain=("total_available_negative_gain", "max"),
        max_reserve_tension=("reserve_tension", "max"),
        min_gain_concentration_k1000=(
            "top_negative_1000_gain_concentration",
            "min",
        ),
        max_predicted_gain=("predicted_gain", "max"),
    )
    scores["example_id"] = scores.example_id.astype(str)
    return scores


def analyze_component_uniqueness(run_dir: Path | None = None) -> Path:
    run_dir = Path(run_dir) if run_dir is not None else latest_complete_run()
    clusters = pd.read_csv(run_dir / "autonomous_cluster_rows.csv")
    repairs = pd.read_csv(run_dir / "autonomous_repair_rows.csv")
    clusters["example_id"] = clusters.example_id.astype(str)
    repairs["example_id"] = repairs.example_id.astype(str)
    clusters["diffuse_negative_leverage"] = (
        clusters.total_available_negative_gain
        * (1.0 - clusters.top_negative_1000_gain_concentration)
        * clusters.top_negative_1000_gain_per_l2
    )

    hidden_ids = set(
        repairs[
            repairs.input_kind.eq("patched")
            & repairs.condition.eq("observed")
            & repairs.source_hidden.astype(bool)
        ].example_id
    )
    hidden_clusters = clusters[
        clusters.input_kind.eq("patched") & clusters.example_id.isin(hidden_ids)
    ].copy()
    component_metrics = [
        "diffuse_negative_leverage",
        "reserve_tension",
        "predicted_gain",
        "n_members",
        "total_available_negative_gain",
    ]
    for k in (250, 500, 1000):
        component_metrics.extend([
            f"top_negative_{k}_gain_concentration",
            f"top_negative_{k}_gain_per_l2",
            f"top_negative_{k}_candidate_gain_min",
            f"top_negative_{k}_candidate_gain_cv",
        ])
    uniqueness_rows = []
    for metric in component_metrics:
        current = hidden_clusters[[metric, "is_target_cluster"]].dropna()
        raw_auc = _safe_auc(current.is_target_cluster, current[metric])
        if not np.isfinite(raw_auc):
            continue
        target_values = current.loc[current.is_target_cluster.astype(bool), metric]
        other_values = current.loc[~current.is_target_cluster.astype(bool), metric]
        uniqueness_rows.append({
            "metric": metric,
            "target_direction": "high" if raw_auc >= 0.5 else "low",
            "discrimination_auc": max(raw_auc, 1.0 - raw_auc),
            "raw_auc_target_is_high": raw_auc,
            "target_median": float(target_values.median()),
            "other_median": float(other_values.median()),
            "target_n": int(len(target_values)),
            "other_n": int(len(other_values)),
        })
    uniqueness = pd.DataFrame(uniqueness_rows).sort_values(
        "discrimination_auc", ascending=False
    )

    hidden_observed = repairs[
        repairs.input_kind.eq("patched")
        & repairs.condition.eq("observed")
        & repairs.source_hidden.astype(bool)
    ].drop_duplicates("example_id")
    localization = pd.DataFrame([{
        "ranking": str(hidden_observed.cluster_ranking.iloc[0]),
        "hidden_n": int(len(hidden_observed)),
        "target_cluster_top1_n": int(hidden_observed.chosen_is_target_cluster.sum()),
        "target_cluster_top1_rate": float(hidden_observed.chosen_is_target_cluster.mean()),
    }])

    clean_scores = _image_scores(clusters, "clean")
    patched_scores = _image_scores(clusters, "patched")
    patched_scores["hidden"] = patched_scores.example_id.isin(hidden_ids)

    ordered_ids = sorted(clean_scores.example_id.unique())
    calibration_ids = set(ordered_ids[::2])
    test_ids = set(ordered_ids[1::2])
    clean_calibration = clean_scores[
        clean_scores.example_id.isin(calibration_ids)
    ]
    clean_test = clean_scores[clean_scores.example_id.isin(test_ids)]
    score_columns = [
        column for column in clean_scores.columns if column != "example_id"
    ]
    low_direction_scores = {"min_gain_concentration_k1000"}
    gate_rows = []
    for score in score_columns:
        clean_vs_hidden_labels = np.r_[
            np.zeros(len(clean_scores), dtype=int),
            np.ones(int(patched_scores.hidden.sum()), dtype=int),
        ]
        clean_vs_hidden_values = np.r_[
            clean_scores[score].to_numpy(float),
            patched_scores.loc[patched_scores.hidden, score].to_numpy(float),
        ]
        raw_auc = _safe_auc(clean_vs_hidden_labels, clean_vs_hidden_values)
        for quantile in GATE_QUANTILES:
            low_direction = score in low_direction_scores
            calibration_quantile = 1.0 - quantile if low_direction else quantile
            threshold = float(
                clean_calibration[score].quantile(calibration_quantile)
            )
            compare = pd.Series.lt if low_direction else pd.Series.gt
            test_trigger = compare(clean_test[score], threshold)
            hidden_trigger = compare(
                patched_scores.loc[patched_scores.hidden, score], threshold
            )
            all_patched_trigger = compare(patched_scores[score], threshold)
            clean_low, clean_high = _wilson(
                int(test_trigger.sum()), len(test_trigger)
            )
            hidden_low, hidden_high = _wilson(
                int(hidden_trigger.sum()), len(hidden_trigger)
            )
            gate_rows.append({
                "score": score,
                "attack_direction": "low" if low_direction else "high",
                "threshold_quantile": quantile,
                "threshold_source": (
                    f"{'lower' if low_direction else 'upper'} {quantile:.1%} "
                    "tail of deterministic 50-image clean calibration split"
                ),
                "threshold": threshold,
                "clean_calibration_n": int(len(clean_calibration)),
                "clean_test_n": int(len(clean_test)),
                "clean_test_trigger_n": int(test_trigger.sum()),
                "clean_test_false_positive_rate": float(test_trigger.mean()),
                "clean_test_fpr_ci95_low": clean_low,
                "clean_test_fpr_ci95_high": clean_high,
                "hidden_n": int(len(hidden_trigger)),
                "hidden_trigger_n": int(hidden_trigger.sum()),
                "hidden_trigger_rate": float(hidden_trigger.mean()),
                "hidden_trigger_ci95_low": hidden_low,
                "hidden_trigger_ci95_high": hidden_high,
                "all_patched_trigger_rate": float(all_patched_trigger.mean()),
                "clean_vs_hidden_auc": (
                    1.0 - raw_auc if low_direction else raw_auc
                ),
            })
    gates = pd.DataFrame(gate_rows).sort_values(
        ["threshold_quantile", "hidden_trigger_rate",
         "clean_test_false_positive_rate"],
        ascending=[False, False, True],
    )

    observed_patched = repairs[
        repairs.input_kind.eq("patched") & repairs.condition.eq("observed")
    ].drop_duplicates("example_id").set_index("example_id")
    outcome_rows = []
    for gate_quantile in GATE_QUANTILES:
        primary_threshold = float(
            gates.loc[
                gates.score.eq(PRIMARY_GATE_SCORE)
                & gates.threshold_quantile.eq(gate_quantile),
                "threshold",
            ].iloc[0]
        )
        clean_gate = dict(zip(
            clean_scores.example_id,
            clean_scores[PRIMARY_GATE_SCORE].gt(primary_threshold),
            strict=True,
        ))
        patched_gate = dict(zip(
            patched_scores.example_id,
            patched_scores[PRIMARY_GATE_SCORE].gt(primary_threshold),
            strict=True,
        ))
        for condition in ("top1_k250", "top1_k500", "top1_k1000", "top2_k500"):
            clean_condition = repairs[
                repairs.input_kind.eq("clean")
                & repairs.condition.eq(condition)
                & repairs.example_id.isin(test_ids)
            ].drop_duplicates("example_id")
            clean_trigger = clean_condition.example_id.map(clean_gate).astype(bool)
            guarded_clean_f1 = np.where(
                clean_trigger,
                clean_condition.detection_detection_f1,
                1.0,
            )
            guarded_clean_target = np.where(
                clean_trigger,
                clean_condition.target_detected.astype(bool),
                True,
            )
            hidden_condition = repairs[
                repairs.input_kind.eq("patched")
                & repairs.condition.eq(condition)
                & repairs.source_hidden.astype(bool)
            ].drop_duplicates("example_id")
            hidden_trigger = hidden_condition.example_id.map(
                patched_gate
            ).astype(bool)
            hidden_recovered = (
                hidden_trigger & hidden_condition.target_detected.astype(bool)
            )
            patched_condition = repairs[
                repairs.input_kind.eq("patched")
                & repairs.condition.eq(condition)
            ].drop_duplicates("example_id")
            patched_trigger = patched_condition.example_id.map(
                patched_gate
            ).astype(bool)
            observed_f1 = patched_condition.example_id.map(
                observed_patched.detection_detection_f1
            )
            observed_target = patched_condition.example_id.map(
                observed_patched.target_detected
            ).astype(bool)
            guarded_patched_f1 = np.where(
                patched_trigger,
                patched_condition.detection_detection_f1,
                observed_f1,
            )
            guarded_patched_target = np.where(
                patched_trigger,
                patched_condition.target_detected.astype(bool),
                observed_target,
            )
            outcome_rows.append({
                "gate_score": PRIMARY_GATE_SCORE,
                "gate_quantile": gate_quantile,
                "gate_threshold": primary_threshold,
                "condition": condition,
                "clean_test_n": int(len(clean_condition)),
                "clean_test_gate_n": int(clean_trigger.sum()),
                "guarded_clean_target_detection_rate": float(
                    guarded_clean_target.mean()
                ),
                "guarded_clean_detection_f1": float(guarded_clean_f1.mean()),
                "hidden_n": int(len(hidden_condition)),
                "hidden_gate_n": int(hidden_trigger.sum()),
                "guarded_hidden_recovery_n": int(hidden_recovered.sum()),
                "guarded_hidden_recovery_rate": float(hidden_recovered.mean()),
                "ungated_hidden_recovery_rate": float(
                    hidden_condition.target_detected.mean()
                ),
                "guarded_all_patched_target_detection_rate": float(
                    guarded_patched_target.mean()
                ),
                "guarded_all_patched_detection_f1": float(
                    guarded_patched_f1.mean()
                ),
            })
    outcomes = pd.DataFrame(outcome_rows)

    uniqueness.to_csv(run_dir / "component_uniqueness.csv", index=False)
    localization.to_csv(run_dir / "component_localization.csv", index=False)
    gates.to_csv(run_dir / "component_gate_evaluation.csv", index=False)
    outcomes.to_csv(run_dir / "guarded_repair_evaluation.csv", index=False)

    best_uniqueness = uniqueness.iloc[0]
    conservative_gate = gates[
        gates.score.eq(PRIMARY_GATE_SCORE)
        & gates.threshold_quantile.eq(0.99)
    ].iloc[0]
    balanced_gate = gates[
        gates.score.eq(PRIMARY_GATE_SCORE)
        & gates.threshold_quantile.eq(0.95)
    ].iloc[0]
    conservative_outcome = outcomes[
        outcomes.condition.eq("top1_k1000")
        & outcomes.gate_quantile.eq(0.99)
    ].iloc[0]
    balanced_outcome = outcomes[
        outcomes.condition.eq("top1_k1000")
        & outcomes.gate_quantile.eq(0.95)
    ].iloc[0]
    report_lines = [
        "# What makes the suppressive component distinctive?",
        "",
        "This analysis uses no paired clean image, patch mask, patch location, or target box for selection.",
        "Clean images are an independent population reference and target annotations are used only after inference for evaluation.",
        "",
        "## Component versus other candidate clusters in successful hides",
        "",
        f"- Hidden inputs: {hidden_clusters.example_id.nunique()}; target-cluster rows: "
        f"{int(hidden_clusters.is_target_cluster.sum())}.",
        f"- Strongest single distinction: `{best_uniqueness.metric}`, "
        f"AUC={best_uniqueness.discrimination_auc:.3f}, target direction="
        f"{best_uniqueness.target_direction}.",
        f"- Reserve-tension ranking localizes the hidden target top-1 in "
        f"{int(localization.target_cluster_top1_n.iloc[0])}/"
        f"{int(localization.hidden_n.iloc[0])} inputs "
        f"({localization.target_cluster_top1_rate.iloc[0]:.1%}).",
        "",
        "The consistent signature is a large candidate reserve under a diffuse, high-leverage negative tail: "
        "large total negative mass, low top-k concentration, and negative coordinates that benefit many reserve members.",
        "The positive-coordinate fraction is nearly saturated and therefore does not distinguish the component.",
        "",
        "## Clean-calibrated gate",
        "",
        f"- Primary score: `{PRIMARY_GATE_SCORE}` = negative mass × "
        "(1 − concentration) × gain/L2.",
        f"- Clean versus hidden AUC: {conservative_gate.clean_vs_hidden_auc:.3f}.",
        f"- Conservative q99: held-out clean triggers "
        f"{int(conservative_gate.clean_test_trigger_n)}/"
        f"{int(conservative_gate.clean_test_n)}; successful-hide triggers "
        f"{int(conservative_gate.hidden_trigger_n)}/"
        f"{int(conservative_gate.hidden_n)}.",
        f"- Balanced q95: held-out clean triggers "
        f"{int(balanced_gate.clean_test_trigger_n)}/"
        f"{int(balanced_gate.clean_test_n)}; successful-hide triggers "
        f"{int(balanced_gate.hidden_trigger_n)}/"
        f"{int(balanced_gate.hidden_n)}.",
        "",
        "## Guarded intervention",
        "",
        f"- Conservative q99 + top-1 k=1000 recovers "
        f"{int(conservative_outcome.guarded_hidden_recovery_n)}/"
        f"{int(conservative_outcome.hidden_n)} hidden targets; clean target "
        f"detection={conservative_outcome.guarded_clean_target_detection_rate:.1%}, "
        f"clean full-output F1={conservative_outcome.guarded_clean_detection_f1:.3f}.",
        f"- Balanced q95 + top-1 k=1000 recovers "
        f"{int(balanced_outcome.guarded_hidden_recovery_n)}/"
        f"{int(balanced_outcome.hidden_n)} hidden targets; clean target "
        f"detection={balanced_outcome.guarded_clean_target_detection_rate:.1%}, "
        f"clean full-output F1={balanced_outcome.guarded_clean_detection_f1:.3f}.",
        "",
        "## Interpretation boundary",
        "",
        "This is evidence for a distinctive functional signature, not yet a universal attack detector. "
        "The cohort uses one detector, class, patch family, and placement protocol; the clean-test confidence interval is still wide.",
    ]
    (run_dir / "component_uniqueness_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return run_dir


if __name__ == "__main__":
    print(analyze_component_uniqueness())
