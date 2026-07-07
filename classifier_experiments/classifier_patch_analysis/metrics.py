from __future__ import annotations

from typing import Any


def roc_auc_score_manual(labels, scores):
    import numpy as np

    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype="float64")
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype="float64")
    ranks[order] = np.arange(1, s.size + 1, dtype="float64")
    for value in np.unique(s):
        tie = s == value
        if int(tie.sum()) > 1:
            ranks[tie] = ranks[tie].mean()
    rank_sum_pos = float(ranks[y].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion_counts(labels, predictions):
    import numpy as np

    y = np.asarray(labels, dtype=bool)
    p = np.asarray(predictions, dtype=bool)
    return {
        "tp": int((p & y).sum()),
        "fp": int((p & ~y).sum()),
        "tn": int((~p & ~y).sum()),
        "fn": int((~p & y).sum()),
    }


def _scores_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


def best_f1(labels, scores):
    import numpy as np

    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype="float64")
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if y.size == 0:
        return {
            "best_f1": float("nan"),
            "best_precision": float("nan"),
            "best_recall": float("nan"),
            "best_balanced_accuracy": float("nan"),
            "best_accuracy": float("nan"),
            "best_threshold": float("nan"),
            "best_direction": 1,
            "balanced_threshold": float("nan"),
            "balanced_direction": 1,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }
    candidates = np.unique(s)
    thresholds = np.concatenate(
        ([candidates[0] - 1e-12], (candidates[:-1] + candidates[1:]) / 2.0, [candidates[-1] + 1e-12])
    )
    best: dict[str, Any] | None = None
    best_balanced: dict[str, Any] | None = None
    for direction in (1, -1):
        for threshold in thresholds:
            pred = s >= threshold if direction == 1 else s <= threshold
            counts = confusion_counts(y, pred)
            vals = _scores_from_counts(counts)
            row = {
                "best_f1": float(vals["f1"]),
                "best_precision": float(vals["precision"]),
                "best_recall": float(vals["recall"]),
                "best_balanced_accuracy": float(vals["balanced_accuracy"]),
                "best_accuracy": float(vals["accuracy"]),
                "best_threshold": float(threshold),
                "best_direction": int(direction),
                **counts,
            }
            if best is None or (row["best_f1"], row["best_precision"], row["best_recall"], row["best_accuracy"]) > (
                best["best_f1"],
                best["best_precision"],
                best["best_recall"],
                best["best_accuracy"],
            ):
                best = row
            if best_balanced is None or (
                vals["balanced_accuracy"],
                vals["f1"],
                vals["precision"],
                vals["recall"],
                vals["accuracy"],
            ) > (
                best_balanced["best_balanced_accuracy"],
                best_balanced["balanced_f1"],
                best_balanced["balanced_precision"],
                best_balanced["balanced_recall"],
                best_balanced["balanced_accuracy_plain"],
            ):
                best_balanced = {
                    "best_balanced_accuracy": float(vals["balanced_accuracy"]),
                    "balanced_precision": float(vals["precision"]),
                    "balanced_recall": float(vals["recall"]),
                    "balanced_specificity": float(vals["specificity"]),
                    "balanced_f1": float(vals["f1"]),
                    "balanced_accuracy_plain": float(vals["accuracy"]),
                    "balanced_threshold": float(threshold),
                    "balanced_direction": int(direction),
                }
    result = best or {}
    if best_balanced:
        result.update(best_balanced)
    return result


def topk_indices(scores, k: int):
    import numpy as np

    arr = np.asarray(scores).reshape(-1)
    k = max(0, min(int(k), arr.size))
    if k == 0:
        return np.asarray([], dtype=int)
    idx = np.argpartition(-np.abs(arr), kth=k - 1)[:k]
    return idx[np.argsort(-np.abs(arr[idx]))]


def alignment_metrics(delta_flat, importance_flat, *, top_percent: float = 5.0):
    import numpy as np

    d = np.asarray(delta_flat, dtype="float64").reshape(-1)
    a = np.asarray(importance_flat, dtype="float64").reshape(-1)
    n = min(d.size, a.size)
    if n == 0:
        return {
            "align_cosine": float("nan"),
            "align_top_jaccard": float("nan"),
            "importance_energy_in_delta_top": float("nan"),
            "delta_energy_in_importance_top": float("nan"),
        }
    d, a = np.abs(d[:n]), np.abs(a[:n])
    denom = float(np.linalg.norm(d) * np.linalg.norm(a) + 1e-12)
    k = max(1, int(round(float(top_percent) / 100.0 * n)))
    d_top = set(int(i) for i in topk_indices(d, k))
    a_top = set(int(i) for i in topk_indices(a, k))
    inter = len(d_top & a_top)
    union = len(d_top | a_top)
    return {
        "align_cosine": float(np.dot(d, a) / denom),
        "align_top_jaccard": float(inter / max(1, union)),
        "importance_energy_in_delta_top": float(a[list(d_top)].sum() / (a.sum() + 1e-12)),
        "delta_energy_in_importance_top": float(d[list(a_top)].sum() / (d.sum() + 1e-12)),
    }


def metric_quality_rows(labels, metrics_by_name: dict[str, object]) -> list[dict[str, Any]]:
    import numpy as np

    rows = []
    for name, scores in metrics_by_name.items():
        s = np.asarray(scores, dtype="float64")
        best = best_f1(labels, s)
        auc = roc_auc_score_manual(labels, s)
        if int(best.get("best_direction", 1)) == -1 and np.isfinite(auc):
            auc = 1.0 - auc
        rows.append({"metric": name, "roc_auc": float(auc), **best})
    return rows
