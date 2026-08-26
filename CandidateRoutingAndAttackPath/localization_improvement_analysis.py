from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .followup_common import FOLLOWUP_DIR


BASELINE_RUN = "improved_component_defense_90a24c45c5b058b6"
DIAGNOSTIC_RUN = "cluster_localization_diagnostic_e6dfe102870981d4"
EXPANDED_RUN = "improved_component_defense_cd7aa8f4006a7432"


def run_localization_improvement_analysis(
    output_root: str | Path = FOLLOWUP_DIR,
) -> Path:
    output_root = Path(output_root)
    baseline_dir = output_root / BASELINE_RUN
    diagnostic_dir = output_root / DIAGNOSTIC_RUN
    expanded_dir = output_root / EXPANDED_RUN

    baseline_rows = pd.read_csv(
        baseline_dir / "improved_repair_rows.csv"
    )
    expanded_rows = pd.read_csv(
        expanded_dir / "improved_repair_rows.csv"
    )
    baseline_loc = pd.read_csv(
        baseline_dir / "improved_localization_summary.csv"
    )
    expanded_loc = pd.read_csv(
        expanded_dir / "improved_localization_summary.csv"
    )
    ranking = pd.read_csv(diagnostic_dir / "ranking_summary.csv")
    availability = pd.read_csv(
        diagnostic_dir / "candidate_availability_summary.csv"
    )

    baseline_person = baseline_loc[
        baseline_loc.proposal_policy.eq("person")
        & baseline_loc.ranking.eq("reserve_tension")
    ].iloc[0]
    expanded_noisy = expanded_loc[
        expanded_loc.proposal_policy.eq("person")
        & expanded_loc.ranking.eq("noisy_or")
    ].iloc[0]
    crossfit = ranking[
        ranking.method.eq("crossfit_score_only")
    ].iloc[0]
    raw_top1000 = availability[
        availability.proposal_top_k.eq(1000)
    ].iloc[0]
    localization = pd.DataFrame([
        {
            "stage": "raw_target_geometry_top1000",
            "target_found_n": int(
                raw_top1000.target_candidate_available_n
            ),
            "hidden_n": int(raw_top1000.hidden_n),
            "status": "oracle diagnostic",
        },
        {
            "stage": "baseline_cluster_any",
            "target_found_n": int(
                baseline_person.target_in_any_cluster_n
            ),
            "hidden_n": int(baseline_person.hidden_n),
            "status": "old observable pipeline",
        },
        {
            "stage": "expanded_cluster_any",
            "target_found_n": int(
                expanded_noisy.target_in_any_cluster_n
            ),
            "hidden_n": int(expanded_noisy.hidden_n),
            "status": "new observable pipeline",
        },
        {
            "stage": "expanded_finalists",
            "target_found_n": int(
                expanded_noisy.target_in_finalists_n
            ),
            "hidden_n": int(expanded_noisy.hidden_n),
            "status": "new observable pipeline",
        },
        {
            "stage": "expanded_noisy_or_top1",
            "target_found_n": int(
                expanded_noisy.target_chosen_top1_n
            ),
            "hidden_n": int(expanded_noisy.hidden_n),
            "status": "unsupervised",
        },
        {
            "stage": "crossfit_score_ranker_top1",
            "target_found_n": int(crossfit.target_chosen_top1_n),
            "hidden_n": int(crossfit.hidden_n),
            "status": "exploratory cross-fitted",
        },
    ])
    localization.to_csv(
        expanded_dir / "localization_improvement_summary.csv", index=False
    )

    old_gate = baseline_rows[
        baseline_rows.condition.eq("person_coverage90_s1")
    ][[
        "example_id", "input_kind", "person_gate_score"
    ]].drop_duplicates(["example_id", "input_kind"])
    conditions = [
        value for value in expanded_rows.condition.unique()
        if value != "observed"
    ]
    modular_rows = []
    for condition in conditions:
        merged = expanded_rows[
            expanded_rows.condition.eq(condition)
        ].merge(
            old_gate,
            on=["example_id", "input_kind"],
            suffixes=("", "_compact"),
        )
        clean = merged[merged.input_kind.eq("clean")].copy()
        ordered_ids = sorted(clean.example_id.astype(str).unique())
        calibration_ids = set(ordered_ids[::2])
        test_ids = set(ordered_ids[1::2])
        calibration = clean[
            clean.example_id.astype(str).isin(calibration_ids)
        ]
        clean_test = clean[
            clean.example_id.astype(str).isin(test_ids)
        ]
        hidden = merged[
            merged.input_kind.eq("patched")
            & merged.source_hidden.astype(bool)
        ]
        for quantile in (0.80, 0.90, 0.95, 0.99):
            threshold = float(
                calibration.person_gate_score_compact.quantile(quantile)
            )
            clean_gate = clean_test.person_gate_score_compact.gt(threshold)
            hidden_gate = hidden.person_gate_score_compact.gt(threshold)
            recovered = hidden_gate & hidden.target_detected.astype(bool)
            modular_rows.append({
                "condition": condition,
                "gate_source": "compact_high_score_clusters",
                "localization_source": "expanded_low_score_clusters",
                "gate_quantile": quantile,
                "threshold": threshold,
                "clean_test_n": int(len(clean_test)),
                "clean_gate_n": int(clean_gate.sum()),
                "guarded_clean_target_detection_rate": float(
                    np.where(
                        clean_gate,
                        clean_test.target_detected.astype(bool),
                        True,
                    ).mean()
                ),
                "guarded_clean_detection_f1": float(
                    np.where(
                        clean_gate,
                        clean_test.detection_detection_f1,
                        1.0,
                    ).mean()
                ),
                "hidden_n": int(len(hidden)),
                "hidden_gate_n": int(hidden_gate.sum()),
                "guarded_hidden_recovery_n": int(recovered.sum()),
                "guarded_hidden_recovery_rate": float(recovered.mean()),
                "ungated_hidden_recovery_n": int(
                    hidden.target_detected.sum()
                ),
                "ungated_hidden_recovery_rate": float(
                    hidden.target_detected.mean()
                ),
            })
    modular = pd.DataFrame(modular_rows)
    modular.to_csv(
        expanded_dir / "modular_guarded_summary.csv", index=False
    )
    best = modular.sort_values(
        [
            "guarded_hidden_recovery_rate",
            "guarded_clean_detection_f1",
        ],
        ascending=False,
    ).iloc[0]
    (expanded_dir / "localization_analysis_digest.md").write_text(
        "\n".join([
            "# Improved cluster localization",
            "",
            "- Raw target geometry exists in 46/49 top-1000 proposal sets.",
            "- Lowering the proposal floor and expanding discovery membership "
            "raises target-in-any-cluster from 38/49 to 46/49.",
            "- The unsupervised noisy-or score selects 39/49 targets; a "
            "two-fold cross-fitted score-only ranker selects 41/49.",
            "- Expanding the intervention itself to 100 routes fails because "
            "the functional component is diluted by unrelated low-score routes.",
            "- Keeping a compact top-20 intervention recovers 38/49 targets.",
            f"- Best modular guarded result: `{best.condition}` at q"
            f"{int(best.gate_quantile * 100)} recovers "
            f"{int(best.guarded_hidden_recovery_n)}/{int(best.hidden_n)} "
            f"with clean target detection "
            f"{best.guarded_clean_target_detection_rate:.3f} and clean "
            f"full-output F1 {best.guarded_clean_detection_f1:.3f}.",
            "",
            "The compact gate and expanded localizer use only the observed "
            "image at inference. Clean pairs and target boxes are evaluation "
            "labels only.",
        ]) + "\n",
        encoding="utf-8",
    )
    return expanded_dir


if __name__ == "__main__":
    print(run_localization_improvement_analysis())
