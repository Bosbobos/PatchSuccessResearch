from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .attack_path import AttackPathResult, load_attack_path_tables, load_spatial_maps
from .candidate_routing import CandidateTraceResult, load_example_candidates, load_trace_tables
from .common import connect_db, stable_hash


LEVEL_COLORS = {0: "#4c78a8", 1: "#f58518", 2: "#54a24b"}
SUCCESS_COLORS = {0: "#e45756", 1: "#4c78a8"}


def _example_from_cache(exp, example_id: str):
    for example in exp.get_cache().examples:
        key = stable_hash({"path": str(example.path), "drop": float(example.drop), "success": bool(example.success)})
        if key == str(example_id):
            return example
    raise KeyError(example_id)


def _draw_box(ax, values, *, color: str, label: str, linewidth: float = 2.0, linestyle: str = "-"):
    from matplotlib.patches import Rectangle

    if values is None or any(pd.isna(value) for value in values):
        return
    x1, y1, x2, y2 = [float(value) for value in values]
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=linewidth, linestyle=linestyle, label=label))


def plot_candidate_overview(
    examples: pd.DataFrame,
    *,
    outcome_names: dict[int, str] | None = None,
    confidence_drop_label: str = "global post-NMS confidence drop",
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = examples.copy()
    outcome_names = outcome_names or {0: "fail", 1: "success"}
    data["outcome"] = data["success"].map(outcome_names)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    counts = data.groupby(["mechanism_mode", "outcome"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=False, ax=axes[0, 0], color=[SUCCESS_COLORS[0], SUCCESS_COLORS[1]])
    axes[0, 0].set_title("Observable candidate-routing modes")
    axes[0, 0].set_ylabel("examples")
    axes[0, 0].tick_params(axis="x", rotation=35)

    sns.scatterplot(
        data=data, x="tracked_logit_delta", y="confidence_drop", hue="mechanism_mode",
        style="outcome", alpha=0.65, ax=axes[0, 1], legend=False,
    )
    axes[0, 1].axhline(0.3, color="black", linestyle="--", linewidth=1)
    axes[0, 1].axvline(0, color="black", linestyle=":", linewidth=1)
    axes[0, 1].set_title(f"Tracked clean candidate vs {confidence_drop_label}")
    axes[0, 1].set_ylabel(confidence_drop_label)

    transition = pd.crosstab(data["clean_target_level"], data["patched_winner_level"], dropna=False)
    transition.index = [f"P{int(v)+3}" if pd.notna(v) else "none" for v in transition.index]
    transition.columns = [f"P{int(v)+3}" if pd.notna(v) else "none" for v in transition.columns]
    sns.heatmap(transition, annot=True, fmt="g", cmap="Blues", ax=axes[1, 0])
    axes[1, 0].set_title("Clean target level → patched winner level")

    sns.boxplot(data=data, x="mechanism_mode", y="patched_winner_minus_tracked", hue="outcome", ax=axes[1, 1], showfliers=False)
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].set_title("Competition margin after patch")
    axes[1, 1].tick_params(axis="x", rotation=35)
    return fig


def plot_candidate_rank_profiles(candidates: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = candidates.copy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    sns.lineplot(data=data, x="rank", y="decoded_score", hue="variant", style="level_name", errorbar="sd", ax=axes[0])
    axes[0].set_title("Pre-NMS candidate score by rank")
    survival = data.groupby(["variant", "rank"])["nms_survived"].mean().reset_index()
    sns.lineplot(data=survival, x="rank", y="nms_survived", hue="variant", ax=axes[1])
    axes[1].set_title("NMS survival rate by pre-NMS rank")
    axes[1].set_ylabel("survival rate")
    return fig


def plot_candidate_example(exp, trace: CandidateTraceResult | str | Path, example_id: str, *, top_n: int = 12):
    import matplotlib.pyplot as plt

    db_path = trace.db_path if isinstance(trace, CandidateTraceResult) else Path(trace)
    conn = connect_db(db_path)
    try:
        row = pd.read_sql_query("SELECT * FROM examples WHERE example_id=?", conn, params=(str(example_id),)).iloc[0]
        lineage = pd.read_sql_query("SELECT * FROM lineage WHERE example_id=? ORDER BY clean_rank", conn, params=(str(example_id),))
    finally:
        conn.close()
    candidates = load_example_candidates(db_path, example_id)
    example = _example_from_cache(exp, example_id)
    clean_image, patched_image, _bbox = exp._images_for_example(example)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes[0, 0].imshow(clean_image)
    _draw_box(axes[0, 0], [row.clean_target_x1, row.clean_target_y1, row.clean_target_x2, row.clean_target_y2], color="#e45756", label="clean tracked")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].set_title(f"clean | conf={row.conf_clean_cached:.3f}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(patched_image)
    _draw_box(axes[0, 1], [row.patch_x1, row.patch_y1, row.patch_x2, row.patch_y2], color="#72b7b2", label="patch")
    _draw_box(axes[0, 1], [row.patched_tracked_x1, row.patched_tracked_y1, row.patched_tracked_x2, row.patched_tracked_y2], color="#f58518", label="same cell", linestyle="--")
    _draw_box(axes[0, 1], [row.patched_winner_x1, row.patched_winner_y1, row.patched_winner_x2, row.patched_winner_y2], color="#54a24b", label="patched winner")
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].set_title(f"patched | conf={row.conf_patch_cached:.3f} | {row.mechanism_mode}")
    axes[0, 1].axis("off")

    labels = ["clean\ntracked", "patched\nsame cell", "patched\nwinner", "clean\npost-NMS", "patched\npost-NMS"]
    values = [row.clean_target_score, row.patched_tracked_score, row.patched_winner_score, row.conf_clean_cached, row.conf_patch_cached]
    missing = [pd.isna(value) for value in values]
    plot_values = [0.0 if is_missing else float(value) for value, is_missing in zip(values, missing, strict=True)]
    colors = ["#e45756", "#f58518", "#54a24b", "#b279a2", "#72b7b2"]
    bars = axes[0, 2].bar(labels, plot_values, color=colors)
    for bar, is_missing in zip(bars, missing, strict=True):
        if is_missing:
            bar.set_facecolor("none")
            bar.set_edgecolor("#777777")
            bar.set_hatch("//")
    axes[0, 2].set_ylim(0, max(1.0, max(plot_values, default=0.0) * 1.12))
    axes[0, 2].set_title(f"Score story | drop={row.confidence_drop:.3f}")
    for idx, (value, is_missing) in enumerate(zip(plot_values, missing, strict=True)):
        label = "N/A" if is_missing else f"{value:.3f}"
        axes[0, 2].text(idx, value, label, ha="center", va="bottom", fontsize=9)

    for axis, variant in zip(axes[1, :2], ["clean", "patched"], strict=True):
        sub = candidates[candidates.variant == variant].head(int(top_n))
        colors = [LEVEL_COLORS.get(int(level), "gray") for level in sub.level_index]
        axis.bar(sub["rank"].astype(str), sub["decoded_score"], color=colors)
        for idx, survived in enumerate(sub["nms_survived"]):
            if survived:
                axis.text(idx, sub.iloc[idx].decoded_score, "●", ha="center", va="bottom", fontsize=8)
        axis.set_title(f"Top {len(sub)} {variant} candidates (● survived NMS)")
        axis.set_xlabel("pre-NMS rank")
        axis.set_ylabel("person score")

    if not lineage.empty:
        scatter = axes[1, 2].scatter(lineage.clean_score, lineage.patched_score, c=lineage.bbox_iou, cmap="viridis", s=60)
        lim = max(float(lineage.clean_score.max()), float(lineage.patched_score.max()), 0.01)
        axes[1, 2].plot([0, lim], [0, lim], "k--", linewidth=1)
        axes[1, 2].set_xlabel("clean candidate score")
        axes[1, 2].set_ylabel("matched patched score")
        axes[1, 2].set_title("Hungarian candidate lineage; color = bbox IoU")
        fig.colorbar(scatter, ax=axes[1, 2], label="IoU")
    return fig


def plot_attack_path_overview(examples: pd.DataFrame, levels: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    sns.scatterplot(data=examples, x="exact_score_delta", y="path_sum", hue="target_kind", style="success", ax=axes[0, 0])
    limits = [min(examples.exact_score_delta.min(), examples.path_sum.min()), max(examples.exact_score_delta.max(), examples.path_sum.max())]
    axes[0, 0].plot(limits, limits, "k--", linewidth=1)
    axes[0, 0].set_title("Completeness: exact head-score change vs path sum")

    sns.histplot(data=examples, x="relative_completeness_error", hue="target_kind", element="step", log_scale=(True, False), ax=axes[0, 1])
    axes[0, 1].set_title("Relative completeness error")

    sns.scatterplot(data=examples, x="first_order_sum", y="exact_score_delta", hue="mechanism_mode", style="success", legend=False, ax=axes[1, 0])
    limits = [min(examples.first_order_sum.min(), examples.exact_score_delta.min()), max(examples.first_order_sum.max(), examples.exact_score_delta.max())]
    axes[1, 0].plot(limits, limits, "k--", linewidth=1)
    axes[1, 0].set_title("Clean first-order approximation vs exact effect")

    merged = levels.merge(examples[["example_id", "target_kind", "success"]], on=["example_id", "target_kind"], how="left")
    sns.barplot(data=merged, x="level_name", y="abs_contribution_fraction", hue="target_kind", errorbar="sd", ax=axes[1, 1])
    axes[1, 1].set_title("Absolute contribution share by Detect input level")
    return fig


def plot_attack_path_by_mode(examples: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    sns.barplot(data=examples, x="mechanism_mode", y="path_minus_first_order", hue="target_kind", errorbar="sd", ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Nonlinear path correction over clean first-order")
    axes[0].tick_params(axis="x", rotation=35)
    concentration = examples.melt(
        id_vars=["mechanism_mode", "target_kind", "success"],
        value_vars=["top0p1_abs_fraction", "top1_abs_fraction", "top2_abs_fraction"],
        var_name="budget", value_name="absolute_contribution_fraction",
    )
    sns.barplot(data=concentration, x="budget", y="absolute_contribution_fraction", hue="target_kind", errorbar="sd", ax=axes[1])
    axes[1].set_title("Sparsity of signed causal contribution")
    return fig


def plot_attack_path_example(exp, result: AttackPathResult | str | Path, example_id: str, *, target_kind: str = "tracked_clean"):
    import matplotlib.pyplot as plt

    db_path = result.db_path if isinstance(result, AttackPathResult) else Path(result)
    conn = connect_db(db_path)
    try:
        row = pd.read_sql_query(
            "SELECT * FROM path_examples WHERE example_id=? AND target_kind=?",
            conn, params=(str(example_id), str(target_kind)),
        ).iloc[0]
        levels = pd.read_sql_query(
            "SELECT * FROM path_levels WHERE example_id=? AND target_kind=? ORDER BY level_index",
            conn, params=(str(example_id), str(target_kind)),
        )
        top = pd.read_sql_query(
            "SELECT * FROM top_contributions WHERE example_id=? AND target_kind=? ORDER BY global_rank",
            conn, params=(str(example_id), str(target_kind)),
        )
    finally:
        conn.close()
    if not row.maps_path or not Path(row.maps_path).exists():
        raise FileNotFoundError("No spatial maps were saved for this example; increase visual_map_examples and rerun.")
    maps = load_spatial_maps(row.maps_path)
    example = _example_from_cache(exp, example_id)
    clean_image, patched_image, _bbox = exp._images_for_example(example)

    fig, axes = plt.subplots(3, 4, figsize=(19, 14), constrained_layout=True)
    axes[0, 0].imshow(clean_image); axes[0, 0].set_title("clean"); axes[0, 0].axis("off")
    axes[0, 1].imshow(patched_image); axes[0, 1].set_title(f"patched | {row.mechanism_mode}"); axes[0, 1].axis("off")
    score_labels = ["clean", "patched", "exact Δ", "path Σ", "first-order"]
    score_values = [row.score_clean, row.score_patched, row.exact_score_delta, row.path_sum, row.first_order_sum]
    axes[0, 2].bar(score_labels, score_values, color=["#4c78a8", "#e45756", "#72b7b2", "#54a24b", "#f58518"])
    axes[0, 2].tick_params(axis="x", rotation=25)
    axes[0, 2].set_title(f"{target_kind} | completeness err={row.relative_completeness_error:.2e}")
    if not top.empty:
        cumulative = top.abs_contribution.cumsum() / max(float(row.total_abs_contribution), 1e-12)
        axes[0, 3].plot(top.global_rank, cumulative)
        axes[0, 3].set_xscale("log")
        axes[0, 3].set_ylim(0, 1.02)
        axes[0, 3].set_title("Cumulative |contribution| of top neurons")
        axes[0, 3].set_xlabel("rank")

    for col, level in enumerate(["P3", "P4", "P5"]):
        signed = maps[f"{level}_signed_contribution"]
        vmax = max(float(np.abs(signed).max()), 1e-12)
        image = axes[1, col].imshow(signed, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[1, col].set_title(f"{level} signed contribution")
        fig.colorbar(image, ax=axes[1, col], fraction=0.046)
        abs_map = maps[f"{level}_abs_contribution"]
        image = axes[2, col].imshow(abs_map, cmap="magma")
        axes[2, col].set_title(f"{level} absolute contribution")
        fig.colorbar(image, ax=axes[2, col], fraction=0.046)
    axes[1, 3].bar(levels.level_name, levels.signed_contribution, color=[LEVEL_COLORS.get(int(v), "gray") for v in levels.level_index])
    axes[1, 3].axhline(0, color="black", linewidth=1)
    axes[1, 3].set_title("Signed contribution by level")
    axes[2, 3].bar(levels.level_name, levels.abs_contribution_fraction, color=[LEVEL_COLORS.get(int(v), "gray") for v in levels.level_index])
    axes[2, 3].set_ylim(0, 1.02)
    axes[2, 3].set_title("Absolute contribution fraction")
    return fig


def select_visual_example_ids(examples: pd.DataFrame, *, n: int = 12) -> list[str]:
    if examples.empty:
        return []
    ordered = examples.sort_values(["mechanism_mode", "success", "confidence_drop"], ascending=[True, False, False])
    groups = [group for _, group in ordered.groupby(["mechanism_mode", "success"], dropna=False)]
    selected, cursor = [], 0
    while len(selected) < int(n):
        added = False
        for group in groups:
            if cursor < len(group) and len(selected) < int(n):
                selected.append(str(group.iloc[cursor].example_id))
                added = True
        if not added:
            break
        cursor += 1
    return selected
