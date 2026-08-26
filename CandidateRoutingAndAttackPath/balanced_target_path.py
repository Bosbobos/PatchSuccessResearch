from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu

from .common import stable_hash, write_json, write_markdown


GROUP_ORDER = (
    "visible_target_winner",
    "visible_non_target_winner",
    "hidden_low_conf_match",
    "hidden_no_iou_match",
)

MATCH_COLUMNS = ("target_clean_conf", "log_target_area", "log_patch_area")


@dataclass(slots=True)
class BalancedSelectionConfig:
    n_per_group: int = 100
    seed: int = 41
    reference_group: str = "visible_non_target_winner"
    method_version: int = 1


@dataclass(slots=True)
class BalancedSelectionResult:
    run_dir: Path
    manifest_path: Path
    balance_path: Path
    summary_path: Path
    digest_path: Path


def _load_trace(trace_db: str | Path) -> pd.DataFrame:
    conn = sqlite3.connect(trace_db)
    try:
        return pd.read_sql_query("SELECT * FROM examples WHERE error IS NULL", conn)
    finally:
        conn.close()


def _assign_groups(frame: pd.DataFrame) -> pd.Series:
    group = pd.Series(pd.NA, index=frame.index, dtype="string")
    visible = ~frame["target_hidden"].astype(bool)
    winner_is_target = frame["patched_winner_is_target"].fillna(0).astype(bool)
    group.loc[visible & winner_is_target] = "visible_target_winner"
    group.loc[visible & ~winner_is_target] = "visible_non_target_winner"
    hidden = frame["target_hidden"].astype(bool)
    group.loc[hidden & frame["target_match_iou"].ge(frame["match_iou_threshold"])] = "hidden_low_conf_match"
    group.loc[hidden & frame["target_match_iou"].lt(frame["match_iou_threshold"])] = "hidden_no_iou_match"
    return group


def _standardize(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    active = [column for column in MATCH_COLUMNS if float(frame[column].std(ddof=0)) > 1e-10]
    values = frame[active].to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    return (values - mean) / scale, active


def _balance_table(frame: pd.DataFrame, reference_group: str) -> pd.DataFrame:
    rows = []
    reference = frame.loc[frame.analysis_group.eq(reference_group)]
    for group in GROUP_ORDER:
        subset = frame.loc[frame.analysis_group.eq(group)]
        for column in ("target_clean_conf", "target_area", "patch_area", "patch_target_area_ratio"):
            pooled = np.sqrt((reference[column].var(ddof=1) + subset[column].var(ddof=1)) / 2)
            smd = 0.0 if not np.isfinite(pooled) or pooled <= 1e-12 else (subset[column].mean() - reference[column].mean()) / pooled
            rows.append({
                "analysis_group": group,
                "covariate": column,
                "n": int(len(subset)),
                "mean": float(subset[column].mean()),
                "std": float(subset[column].std(ddof=1)),
                "reference_mean": float(reference[column].mean()),
                "standardized_mean_difference": float(smd),
            })
    return pd.DataFrame(rows)


def build_balanced_target_selection(
    labels_csv: str | Path,
    trace_db: str | Path,
    output_dir: str | Path,
    config: BalancedSelectionConfig | None = None,
) -> BalancedSelectionResult:
    config = config or BalancedSelectionConfig()
    labels_csv, trace_db, output_dir = Path(labels_csv), Path(trace_db), Path(output_dir)
    labels = pd.read_csv(labels_csv)
    trace = _load_trace(trace_db)
    frame = labels.merge(trace, on="example_id", how="inner", validate="one_to_one", suffixes=("_label", "_trace"))
    frame = frame.loc[frame["target_eligible"].astype(bool)].copy()
    frame["analysis_group"] = _assign_groups(frame)
    frame = frame.loc[frame.analysis_group.notna()].copy()
    frame["target_area"] = (
        (frame["clean_target_x2"] - frame["clean_target_x1"]).clip(lower=1e-6)
        * (frame["clean_target_y2"] - frame["clean_target_y1"]).clip(lower=1e-6)
    )
    frame["patch_area"] = (
        (frame["patch_x2"] - frame["patch_x1"]).clip(lower=1e-6)
        * (frame["patch_y2"] - frame["patch_y1"]).clip(lower=1e-6)
    )
    frame["patch_target_area_ratio"] = frame["patch_area"] / frame["target_area"]
    frame["log_target_area"] = np.log(frame["target_area"])
    frame["log_patch_area"] = np.log(frame["patch_area"])

    counts = frame.analysis_group.value_counts()
    missing = [group for group in GROUP_ORDER if group not in counts]
    if missing:
        raise ValueError(f"No eligible examples for groups: {missing}")
    n = min(int(config.n_per_group), *(int(counts[group]) for group in GROUP_ORDER))
    if n <= 0:
        raise ValueError("n_per_group must be positive")
    if config.reference_group not in GROUP_ORDER:
        raise ValueError(f"Unknown reference group: {config.reference_group}")

    z, active_columns = _standardize(frame)
    z_by_index = {index: z[position] for position, index in enumerate(frame.index)}
    reference = frame.loc[frame.analysis_group.eq(config.reference_group)].copy()
    other_groups = [group for group in GROUP_ORDER if group != config.reference_group]

    # Keep reference examples in common support: low nearest-neighbour distance to every other group.
    support = np.zeros(len(reference), dtype=float)
    ref_z = np.vstack([z_by_index[index] for index in reference.index])
    for group in other_groups:
        candidate = frame.loc[frame.analysis_group.eq(group)]
        candidate_z = np.vstack([z_by_index[index] for index in candidate.index])
        distance = np.linalg.norm(ref_z[:, None, :] - candidate_z[None, :, :], axis=2)
        support += distance.min(axis=1)
    rng = np.random.default_rng(int(config.seed))
    jitter = rng.uniform(0, 1e-9, size=len(reference))
    reference = reference.iloc[np.argsort(support + jitter)[:n]].copy()
    reference["match_set"] = np.arange(n, dtype=int)
    reference["distance_to_reference"] = 0.0
    selected = [reference]
    ref_z = np.vstack([z_by_index[index] for index in reference.index])

    for group in other_groups:
        candidate = frame.loc[frame.analysis_group.eq(group)].copy()
        candidate_z = np.vstack([z_by_index[index] for index in candidate.index])
        cost = np.linalg.norm(ref_z[:, None, :] - candidate_z[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        matched = candidate.iloc[col_ind].copy()
        matched["match_set"] = row_ind.astype(int)
        matched["distance_to_reference"] = cost[row_ind, col_ind]
        selected.append(matched)

    manifest = pd.concat(selected, ignore_index=True).sort_values(["match_set", "analysis_group"])
    keep = [
        "example_id", "path_label", "analysis_group", "target_hidden", "match_set",
        "distance_to_reference", "target_clean_conf", "target_patched_conf", "target_match_iou",
        "target_area", "patch_area", "patch_target_area_ratio", "legacy_success", "outcome",
    ]
    manifest = manifest[keep].rename(columns={"path_label": "path"}).reset_index(drop=True)

    payload = {
        "labels_csv": str(labels_csv.resolve()),
        "labels_size": labels_csv.stat().st_size,
        "trace_db": str(trace_db.resolve()),
        "trace_size": trace_db.stat().st_size,
        "config": asdict(config),
        "example_ids": manifest.example_id.tolist(),
    }
    run_dir = output_dir / f"balanced_target_selection_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "balanced_target_manifest.csv"
    balance_path = run_dir / "balanced_target_balance.csv"
    manifest.to_csv(manifest_path, index=False)
    balance = _balance_table(manifest, config.reference_group)
    balance.to_csv(balance_path, index=False)
    summary = {
        "status": "complete",
        "n_per_group": n,
        "n_total": int(len(manifest)),
        "group_counts_before_matching": {group: int(counts[group]) for group in GROUP_ORDER},
        "group_counts_after_matching": {group: int((manifest.analysis_group == group).sum()) for group in GROUP_ORDER},
        "reference_group": config.reference_group,
        "active_matching_covariates": active_columns,
        "constant_covariates": [column for column in MATCH_COLUMNS if column not in active_columns],
        "max_abs_standardized_mean_difference": float(balance.standardized_mean_difference.abs().max()),
        "manifest": str(manifest_path),
    }
    summary_path = write_json(run_dir / "summary.json", summary)
    digest_path = write_markdown(run_dir / "analysis_digest.md", [
        "# Balanced target attack-path selection", "",
        f"- selected: {summary['n_total']} ({n} per group)",
        f"- groups: {', '.join(GROUP_ORDER)}",
        f"- matching covariates: {', '.join(active_columns)}",
        f"- constant covariates retained as balance checks: {', '.join(summary['constant_covariates']) or 'none'}",
        f"- maximum post-match |SMD|: {summary['max_abs_standardized_mean_difference']:.3f}",
        "", "Read balanced_target_balance.csv before launching the causal-path run.",
    ])
    return BalancedSelectionResult(run_dir, manifest_path, balance_path, summary_path, digest_path)


def summarize_balanced_attack_path(attack_path_db: str | Path, output_dir: str | Path) -> dict[str, Path]:
    attack_path_db, output_dir = Path(attack_path_db), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(attack_path_db)
    try:
        examples = pd.read_sql_query("SELECT * FROM path_examples WHERE error IS NULL", conn)
    finally:
        conn.close()
    metrics = [
        "exact_score_delta", "first_order_sum", "first_order_residual",
        "total_abs_contribution", "positive_contribution", "negative_contribution",
        "top0p1_abs_fraction", "top1_abs_fraction", "top2_abs_fraction",
    ]
    groups = examples.groupby(["target_kind", "mechanism_mode"], dropna=False)
    rows = []
    for (target_kind, group), subset in groups:
        row = {"target_kind": target_kind, "analysis_group": group, "n": int(len(subset))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_median"] = float(subset[metric].median())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = output_dir / "balanced_attack_path_group_summary.csv"
    summary.to_csv(summary_path, index=False)

    pair_rows = []
    for target_kind, target_frame in examples.groupby("target_kind"):
        for left_index, left_group in enumerate(GROUP_ORDER):
            for right_group in GROUP_ORDER[left_index + 1:]:
                left = target_frame.loc[target_frame.mechanism_mode.eq(left_group)]
                right = target_frame.loc[target_frame.mechanism_mode.eq(right_group)]
                for metric in metrics:
                    pooled = np.sqrt((left[metric].var(ddof=1) + right[metric].var(ddof=1)) / 2)
                    effect = np.nan if not np.isfinite(pooled) or pooled <= 1e-12 else (right[metric].mean() - left[metric].mean()) / pooled
                    p_value = mannwhitneyu(left[metric], right[metric], alternative="two-sided").pvalue
                    pair_rows.append({
                        "target_kind": target_kind, "left_group": left_group, "right_group": right_group,
                        "metric": metric, "cohens_d_right_minus_left": effect, "mannwhitney_p": p_value,
                    })
    pairwise = pd.DataFrame(pair_rows)
    if not pairwise.empty:
        m = len(pairwise)
        order = np.argsort(pairwise.mannwhitney_p.to_numpy())
        ordered_p = pairwise.mannwhitney_p.to_numpy()[order]
        adjusted = np.minimum(1.0, np.maximum.accumulate(ordered_p * (m - np.arange(m))))
        pairwise["holm_p"] = np.nan
        pairwise.loc[pairwise.index[order], "holm_p"] = adjusted
    pairwise_path = output_dir / "balanced_attack_path_pairwise.csv"
    pairwise.to_csv(pairwise_path, index=False)

    digest = output_dir / "analysis_digest.md"
    write_markdown(digest, [
        "# Balanced target attack-path results", "",
        f"- completed target rows: {len(examples)}",
        f"- unique examples: {examples.example_id.nunique()}",
        f"- groups: {json.dumps(examples.mechanism_mode.value_counts().to_dict(), ensure_ascii=False)}",
        "", "Primary comparison: hidden_low_conf_match versus hidden_no_iou_match.",
        "Use the matched visible groups as controls; do not interpret winner_margin as causal.",
    ])
    return {"group_summary": summary_path, "pairwise": pairwise_path, "digest": digest}
