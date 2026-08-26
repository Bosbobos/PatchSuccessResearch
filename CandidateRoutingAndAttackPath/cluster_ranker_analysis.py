from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .followup_common import FOLLOWUP_DIR


SCORE_FEATURES = [
    "log_n_members",
    "max_score",
    "mean_score",
    "median_score",
    "log_score_sum",
    "noisy_or",
    "reserve_tension",
    "n_levels",
]
SHAPE_FEATURES = [
    "log_box_area",
    "median_aspect_hw",
    "log_center_x_std",
    "log_center_y_std",
]


def _fold(example_id: str) -> int:
    digest = hashlib.sha256(str(example_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2


def _prepare(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["example_id"] = rows.example_id.astype(str)
    rows["fold"] = rows.example_id.map(_fold)
    rows["log_n_members"] = np.log1p(rows.n_members)
    rows["log_score_sum"] = np.log1p(rows.score_sum)
    rows["log_box_area"] = np.log1p(rows.median_box_area)
    rows["log_center_x_std"] = np.log1p(rows.center_x_std)
    rows["log_center_y_std"] = np.log1p(rows.center_y_std)
    return rows


def _top1(rows: pd.DataFrame, score: str, ascending: bool = False):
    return rows.sort_values(
        ["example_id", score], ascending=[True, ascending]
    ).drop_duplicates("example_id")


def _crossfit(rows: pd.DataFrame, features: list[str], name: str):
    rows = rows.copy()
    rows[name] = np.nan
    for fold in sorted(rows.fold.unique()):
        train = rows[rows.fold.ne(fold)]
        test = rows[rows.fold.eq(fold)]
        model = make_pipeline(
            SimpleImputer(),
            StandardScaler(),
            LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000
            ),
        )
        model.fit(train[features], train.is_target_cluster)
        rows.loc[test.index, name] = model.predict_proba(
            test[features]
        )[:, 1]
    return rows


def run_cluster_ranker_analysis(
    diagnostic_dir: str | Path | None = None,
) -> Path:
    if diagnostic_dir is None:
        candidates = sorted(
            Path(FOLLOWUP_DIR).glob("cluster_localization_diagnostic_*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        diagnostic_dir = next(
            path for path in candidates
            if (path / "cluster_details.csv").exists()
        )
    diagnostic_dir = Path(diagnostic_dir)
    details = _prepare(pd.read_csv(diagnostic_dir / "cluster_details.csv"))
    finalists = details[details.is_finalist.astype(bool)].copy()

    score_crossfit = _crossfit(
        finalists, SCORE_FEATURES, "score_ranker"
    )
    shape_crossfit = _crossfit(
        finalists, SCORE_FEATURES + SHAPE_FEATURES, "score_shape_ranker"
    )
    predictions = score_crossfit[[
        "example_id", "fold", "cluster_index", "is_target_cluster",
        "target_iou", "score_ranker",
    ]].merge(
        shape_crossfit[[
            "example_id", "cluster_index", "score_shape_ranker"
        ]],
        on=["example_id", "cluster_index"],
        how="left",
    )
    predictions.to_csv(
        diagnostic_dir / "crossfit_ranker_predictions.csv", index=False
    )

    methods = [
        ("reserve_tension", finalists, "reserve_tension"),
        ("noisy_or", finalists, "noisy_or"),
        ("score_sum", finalists, "score_sum"),
        ("crossfit_score_only", score_crossfit, "score_ranker"),
        (
            "crossfit_score_plus_shape",
            shape_crossfit,
            "score_shape_ranker",
        ),
    ]
    summary_rows = []
    oracle = finalists.groupby("example_id").is_target_cluster.max()
    for method, frame, column in methods:
        top = _top1(frame, column)
        summary_rows.append({
            "method": method,
            "hidden_n": int(top.example_id.nunique()),
            "target_available_in_finalists_n": int(oracle.sum()),
            "target_chosen_top1_n": int(top.is_target_cluster.sum()),
            "target_chosen_top1_rate": float(top.is_target_cluster.mean()),
            "fold0_top1_n": int(
                top[top.fold.eq(0)].is_target_cluster.sum()
            ),
            "fold0_n": int(top[top.fold.eq(0)].example_id.nunique()),
            "fold1_top1_n": int(
                top[top.fold.eq(1)].is_target_cluster.sum()
            ),
            "fold1_n": int(top[top.fold.eq(1)].example_id.nunique()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(diagnostic_dir / "ranking_summary.csv", index=False)

    candidate_summary = pd.read_csv(
        diagnostic_dir / "candidate_availability_summary.csv"
    )
    cluster_summary = pd.read_csv(
        diagnostic_dir / "cluster_discovery_summary.csv"
    )
    best = summary.sort_values(
        "target_chosen_top1_n", ascending=False
    ).iloc[0]
    (diagnostic_dir / "analysis_digest.md").write_text(
        "\n".join([
            "# Cluster localization diagnostic",
            "",
            f"- Raw target geometry in top-1000 candidates: "
            f"{int(candidate_summary.iloc[0].target_candidate_available_n)}/"
            f"{int(candidate_summary.iloc[0].hidden_n)}.",
            f"- Expanded cluster membership target availability: "
            f"{int(cluster_summary.iloc[0].target_in_any_cluster_n)}/"
            f"{int(cluster_summary.iloc[0].hidden_n)}.",
            f"- Target present among finalists: "
            f"{int(cluster_summary.iloc[0].target_in_finalists_n)}/"
            f"{int(cluster_summary.iloc[0].hidden_n)}.",
            f"- Best cross-fitted ranking: `{best.method}` selects "
            f"{int(best.target_chosen_top1_n)}/{int(best.hidden_n)} targets.",
            "",
            "The ranker never sees the clean image at inference. Target labels "
            "from paired images are used only to train/evaluate the cross-fitted "
            "ranking model. The 49-image cohort is exploratory.",
        ]) + "\n",
        encoding="utf-8",
    )
    return diagnostic_dir


if __name__ == "__main__":
    print(run_cluster_ranker_analysis())
