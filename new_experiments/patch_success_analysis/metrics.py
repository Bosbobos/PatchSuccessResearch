from __future__ import annotations


def moving_average(values, window: int = 15):
    import numpy as np

    arr = np.asarray(values, dtype="float64")
    if int(window) <= 1 or arr.size == 0:
        return arr.copy()
    window = min(int(window), arr.size)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype="float64") / float(window)
    return np.convolve(padded, kernel, mode="valid")


def topk_indices(scores, k: int):
    import numpy as np

    arr = np.asarray(scores).reshape(-1)
    k = max(0, min(int(k), arr.size))
    if k == 0:
        return np.asarray([], dtype=int)
    idx = np.argpartition(-np.abs(arr), kth=k - 1)[:k]
    return idx[np.argsort(-np.abs(arr[idx]))]


def top_overlap_table(method_scores: dict[str, object], percentages=(1, 5, 10, 20, 50, 100)):
    import itertools
    import numpy as np

    names = list(method_scores)
    n = min(np.asarray(method_scores[name]).size for name in names)
    rows = []
    for pct in percentages:
        k = max(1, int(round(float(pct) / 100.0 * n)))
        sets = {name: set(int(i) for i in topk_indices(method_scores[name], k)) for name in names}
        all_inter = set.intersection(*(sets[name] for name in names)) if names else set()
        rows.append({"comparison": "all", "top_percent": float(pct), "top_k": k, "overlap_count": len(all_inter)})
        for a, b in itertools.combinations(names, 2):
            rows.append(
                {
                    "comparison": f"{a} vs {b}",
                    "top_percent": float(pct),
                    "top_k": k,
                    "overlap_count": len(sets[a] & sets[b]),
                }
            )
    return rows


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
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    rank_sum_pos = float(ranks[y].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def best_accuracy(labels, scores):
    import numpy as np

    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype="float64")
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if y.size == 0:
        return {"accuracy": float("nan"), "threshold": float("nan"), "direction": 1}
    candidates = np.unique(s)
    thresholds = np.concatenate(([candidates[0] - 1e-12], (candidates[:-1] + candidates[1:]) / 2.0, [candidates[-1] + 1e-12]))
    best = {"accuracy": -1.0, "threshold": float(thresholds[0]), "direction": 1}
    for direction in (1, -1):
        for thr in thresholds:
            pred = s >= thr if direction == 1 else s <= thr
            acc = float(np.mean(pred == y))
            if acc > best["accuracy"]:
                best = {"accuracy": acc, "threshold": float(thr), "direction": int(direction)}
    return best


def roc_curve_points(labels, scores, *, direction: int = 1):
    import numpy as np

    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype="float64") * (1 if int(direction) == 1 else -1)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    thresholds = np.r_[np.inf, np.unique(s)[::-1], -np.inf]
    tpr, fpr = [], []
    pos = max(1, int(y.sum()))
    neg = max(1, int((~y).sum()))
    for thr in thresholds:
        pred = s >= thr
        tpr.append(float((pred & y).sum() / pos))
        fpr.append(float((pred & ~y).sum() / neg))
    return {"fpr": fpr, "tpr": tpr, "thresholds": [float(v) for v in thresholds]}


def alignment_metrics(delta_flat, importance_flat, *, top_percent: float = 5.0):
    import numpy as np

    d = np.asarray(delta_flat, dtype="float64").reshape(-1)
    a = np.asarray(importance_flat, dtype="float64").reshape(-1)
    n = min(d.size, a.size)
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


def metric_quality_rows(labels, metrics_by_name: dict[str, object]):
    rows = []
    for name, values in metrics_by_name.items():
        auc = roc_auc_score_manual(labels, values)
        best = best_accuracy(labels, values)
        rows.append(
            {
                "metric": name,
                "roc_auc": auc,
                "best_accuracy": best["accuracy"],
                "best_threshold": best["threshold"],
                "best_direction": best["direction"],
            }
        )
    return rows
