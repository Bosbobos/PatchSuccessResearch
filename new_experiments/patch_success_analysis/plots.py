from __future__ import annotations


def ensure_output_dir(path):
    from pathlib import Path

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_ranked_attributions(method_scores: dict[str, object], *, sort_by: str, top_n: int, smoothed: bool, window: int = 15, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    from .metrics import moving_average

    sorter = np.argsort(-np.abs(np.asarray(method_scores[sort_by]).reshape(-1)))
    sorter = sorter[: min(int(top_n), sorter.size)]
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, scores in method_scores.items():
        raw = np.abs(np.asarray(scores).reshape(-1))
        y = raw[sorter] / max(float(np.nanmax(raw)), 1e-12)
        if smoothed:
            y = moving_average(y, window=window)
        ax.plot(np.arange(1, y.size + 1), y, label=name)
    ax.set_title(f"{'Smoothed' if smoothed else 'Raw'} top-{len(sorter)} neurons ranked by |{sort_by}|")
    ax.set_xlabel(f"Rank by |{sort_by}|")
    ax.set_ylabel("Attribution magnitude / method max")
    ax.grid(alpha=0.25)
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_overlap(rows, *, save_path=None):
    import matplotlib.pyplot as plt

    comparisons = sorted({r["comparison"] for r in rows})
    fig, ax = plt.subplots(figsize=(10, 5))
    for comp in comparisons:
        sub = [r for r in rows if r["comparison"] == comp]
        sub.sort(key=lambda r: r["top_percent"])
        ax.plot([r["top_percent"] for r in sub], [r["overlap_count"] for r in sub], marker="o", label=comp)
    perfect_by_percent = {}
    for row in rows:
        perfect_by_percent[float(row["top_percent"])] = int(row["top_k"])
    if perfect_by_percent:
        xs = sorted(perfect_by_percent)
        ys = [perfect_by_percent[x] for x in xs]
        ax.plot(xs, ys, linestyle="--", color="black", linewidth=1.5, label="perfect overlap")
    ax.set_title("Top-neuron overlap by attribution method")
    ax.set_xlabel("Top percent of neurons")
    ax.set_ylabel("Overlap count")
    ax.grid(alpha=0.25)
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_runtime(runtime_rows, *, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    methods = [r["method"] for r in runtime_rows]
    means = [float(r["mean_s"]) for r in runtime_rows]
    stds = [float(r.get("std_s", 0.0)) for r in runtime_rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(methods, means, yerr=stds, capsize=4)
    ax.set_title("Runtime benchmark on 100 images")
    ax.set_ylabel("Seconds per image")
    ax.grid(axis="y", alpha=0.25)
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_attack_cache_summary(cache_df, *, save_path=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    counts = cache_df["success"].value_counts().sort_index()
    axes[0].bar(["fail", "success"], [int(counts.get(False, 0)), int(counts.get(True, 0))])
    axes[0].set_title("Balanced attack cache labels")
    axes[0].set_ylabel("Examples")
    fail = cache_df.loc[cache_df["success"] == False, "drop"]
    succ = cache_df.loc[cache_df["success"] == True, "drop"]
    axes[1].hist(fail, bins=25, alpha=0.65, label="fail")
    axes[1].hist(succ, bins=25, alpha=0.65, label="success")
    axes[1].set_title("Confidence drop distribution")
    axes[1].set_xlabel("conf_clean - conf_patch")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_metric_distribution_and_roc(labels, values, *, metric_name: str, auc: float, best_accuracy: float, direction: int = 1, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    from .metrics import roc_curve_points

    y = np.asarray(labels, dtype=bool)
    v = np.asarray(values, dtype="float64")
    roc = roc_curve_points(y, v, direction=direction)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].hist(v[~y], bins=25, alpha=0.65, label="fail")
    axes[0].hist(v[y], bins=25, alpha=0.65, label="success")
    axes[0].set_title(metric_name)
    axes[0].set_xlabel("Metric value")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[1].plot(roc["fpr"], roc["tpr"], marker=".")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="0.5")
    axes[1].set_title(f"ROC-AUC={auc:.3f}, best acc={best_accuracy:.3f}")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].grid(alpha=0.25)
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_patch_propagation(example, clean_img, patched_img, layer_panels, *, save_path=None):
    import matplotlib.pyplot as plt

    n_layers = len(layer_panels)
    fig, axes = plt.subplots(n_layers, 6, figsize=(19, max(3.2, 3.1 * n_layers)), squeeze=False, constrained_layout=True)
    for row, panel in enumerate(layer_panels):
        axes[row, 0].imshow(clean_img)
        axes[row, 0].set_title("clean")
        axes[row, 1].imshow(patched_img)
        axes[row, 1].set_title("patched")
        axes[row, 2].imshow(panel["delta_abs"], cmap="magma")
        axes[row, 2].set_title(f"{panel['layer']} |delta|")
        axes[row, 3].imshow(panel["delta_signed"], cmap="coolwarm", vmin=-1, vmax=1)
        axes[row, 3].set_title("signed delta")
        axes[row, 4].imshow(panel["importance_clean"], cmap="viridis")
        metrics = panel.get("metrics", {})
        axes[row, 4].set_title(
            f"SegmentIG clean\nl2={metrics.get('delta_l2_rms', 0):.3g}, roi={metrics.get('roi_energy_frac', 0):.2f}"
        )
        axes[row, 5].imshow(panel["importance_patched"], cmap="viridis")
        axes[row, 5].set_title("SegmentIG patched")
        for col in range(6):
            axes[row, col].axis("off")
    fig.suptitle(
        f"{example.path} | success={example.success} | clean={example.conf_clean:.3f} "
        f"patched={example.conf_patch:.3f} drop={example.drop:.3f}"
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_all_layer_delta_strip(example_panels, *, save_path=None):
    import matplotlib.pyplot as plt

    if not example_panels:
        raise ValueError("example_panels is empty")
    layer_names = example_panels[0]["layers"]
    n_rows = len(example_panels) * 2
    n_cols = len(layer_names)
    fig_w = max(14, 1.25 * n_cols)
    fig_h = max(3.2, 1.9 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
    for item_idx, item in enumerate(example_panels):
        example = item["example"]
        maps = item["maps"]
        signed_maps = item["signed_maps"]
        energy_row = 2 * item_idx
        signed_row = energy_row + 1
        for col, layer_name in enumerate(layer_names):
            ax = axes[energy_row, col]
            ax.imshow(maps[layer_name], cmap="magma", vmin=0, vmax=1)
            if energy_row == 0:
                ax.set_title(layer_name, fontsize=8)
            if col == 0:
                ax.set_ylabel(
                    f"{'success' if example.success else 'fail'}\n"
                    f"|delta| drop={example.drop:.3f}",
                    fontsize=9,
                )
            ax.set_xticks([])
            ax.set_yticks([])

            ax_signed = axes[signed_row, col]
            ax_signed.imshow(signed_maps[layer_name], cmap="coolwarm", vmin=-1, vmax=1)
            if col == 0:
                ax_signed.set_ylabel("signed delta", fontsize=9)
            ax_signed.set_xticks([])
            ax_signed.set_yticks([])
    fig.suptitle("Расползание патча по всем слоям: |delta_layer| и signed mean delta")
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig


def plot_failure_diagnosis(diagnosis, *, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt

    groups = ["success", "fail"]
    map_cols = [
        ("delta_energy", "mean |delta|"),
        ("signed_delta", "mean signed delta"),
        ("delta_in_important", "|delta| in important neurons"),
        ("importance", "mean SegmentIG importance"),
        ("delta_outside_important", "|delta| outside important neurons"),
    ]
    fig, axes = plt.subplots(3, 5, figsize=(19, 10.5), constrained_layout=True)
    for row, group in enumerate(groups):
        maps = diagnosis["maps"][group]
        for col, (key, title) in enumerate(map_cols):
            ax = axes[row, col]
            if key == "signed_delta":
                ax.imshow(maps[key], cmap="coolwarm", vmin=-1, vmax=1)
            else:
                ax.imshow(maps[key], cmap="magma", vmin=0, vmax=1)
            ax.set_title(f"{group}: {title}")
            ax.set_xticks([])
            ax.set_yticks([])

    metric_names = [
        "delta_l2_rms",
        "topk_energy_frac",
        "segmentig_delta_energy_in_importance_top",
        "segmentig_align_cosine",
    ]
    labels = [
        "spread magnitude",
        "spread concentration",
        "energy in important",
        "delta/importance cosine",
    ]
    for idx, (metric, label) in enumerate(zip(metric_names, labels, strict=True)):
        ax = axes[2, idx]
        vals = [diagnosis["metric_means"][group].get(metric, float("nan")) for group in groups]
        ax.bar(groups, vals, color=["#4C78A8", "#F58518"])
        ax.set_title(label)
        ax.set_ylabel(metric)
        for x, val in enumerate(vals):
            if np.isfinite(val):
                ax.text(x, val, f"{val:.3g}", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    axes[2, 4].axis("off")

    fig.suptitle(
        "Диагностика неуспешных атак: патч не расползся или расползся не в важные фичи",
        fontsize=14,
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    return fig
