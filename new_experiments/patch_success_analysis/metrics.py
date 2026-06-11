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


def importance_rank_bin_weights(importance_flat, *, cutoff_percent: int, family: str):
    import numpy as np

    a = np.asarray(importance_flat, dtype="float64").reshape(-1)
    cutoff_percent = int(cutoff_percent)
    if cutoff_percent <= 0 or cutoff_percent > 100:
        raise ValueError("cutoff_percent must be in 1..100")
    cutoff_bins = cutoff_percent
    n = int(a.size)
    weights = np.zeros(n, dtype="float64")
    if n == 0:
        return weights

    included_k = n if cutoff_percent == 100 else max(1, int(round(float(cutoff_percent) / 100.0 * n)))
    included_k = min(included_k, n)
    rank_order = np.argsort(-np.abs(a), kind="stable")
    bin_ids = np.floor(np.arange(included_k, dtype="float64") * cutoff_bins / included_k).astype(int)
    bin_ids = np.clip(bin_ids, 0, cutoff_bins - 1)

    if family == "linear":
        if cutoff_bins == 1:
            bin_weights = np.ones(1, dtype="float64")
        else:
            bin_weights = np.linspace(1.0, 1.0 / cutoff_bins, cutoff_bins, dtype="float64")
    elif family == "reciprocal":
        bin_weights = 1.0 / (np.arange(cutoff_bins, dtype="float64") + 1.0)
    elif family == "exp":
        half_life = cutoff_bins / 5.0
        bin_weights = np.power(0.5, np.arange(cutoff_bins, dtype="float64") / half_life)
    else:
        raise ValueError(f"Unknown weight family: {family!r}")

    weights[rank_order[:included_k]] = bin_weights[bin_ids]
    return weights


def segmentig_soft_alignment_metrics(delta_flat, importance_flat):
    import numpy as np

    d = np.asarray(delta_flat, dtype="float64").reshape(-1)
    a = np.asarray(importance_flat, dtype="float64").reshape(-1)
    n = min(d.size, a.size)
    d, a = d[:n], a[:n]
    if n == 0:
        metrics = {
            "delta_importance_product_signed": float("nan"),
            "delta_importance_product_unsigned": float("nan"),
        }
        for cutoff in (100, 50, 25, 10):
            for family in ("linear", "reciprocal", "exp"):
                metrics[f"delta_energy_importance_bins_top{cutoff}_{family}"] = float("nan")
        return metrics
    d_abs = np.abs(d)
    denom = float(d_abs.sum() + 1e-12)
    a_norm = a / float(np.max(np.abs(a)) + 1e-12)

    metrics = {
        "delta_importance_product_signed": float(np.sum((-a_norm) * d) / denom),
        "delta_importance_product_unsigned": float(np.sum(np.abs(a_norm) * d_abs) / denom),
    }
    for cutoff in (100, 50, 25, 10):
        for family in ("linear", "reciprocal", "exp"):
            weights = importance_rank_bin_weights(a, cutoff_percent=cutoff, family=family)
            metrics[f"delta_energy_importance_bins_top{cutoff}_{family}"] = float(np.sum(weights * d_abs) / denom)
    metrics.update(robust_importance_product_metrics(d, a))
    return metrics


def robust_importance_product_metrics(
    delta_flat,
    importance_flat,
    *,
    top_percents=(3, 5, 10),
    log_bases=(1.5, 2.0, 2.718281828459045, 10.0),
    power_alphas=(0.25, 0.5, 0.75),
    trim_quantiles=(99.0, 99.5),
):
    import math
    import numpy as np

    d = np.asarray(delta_flat, dtype="float64").reshape(-1)
    a = np.asarray(importance_flat, dtype="float64").reshape(-1)
    n = min(d.size, a.size)
    if n == 0:
        return {}
    d_abs = np.abs(d[:n])
    a_abs = np.abs(a[:n])
    denom = float(d_abs.sum() + 1e-12)
    a_scaled = a_abs / float(np.nanmax(a_abs) + 1e-12)
    order = np.argsort(-a_abs, kind="stable")
    metrics = {}

    for top_percent in top_percents:
        top_percent_int = int(top_percent)
        k = n if float(top_percent) >= 100 else max(1, int(round(float(top_percent) / 100.0 * n)))
        idx = order[: min(k, n)]
        d_top = d_abs[idx]
        a_top = a_scaled[idx]
        raw_product = a_top * d_top
        metrics[f"delta_importance_product_top{top_percent_int}_raw"] = float(np.sum(raw_product) / denom)

        for base in log_bases:
            base_float = float(base)
            if base_float <= 1.0:
                continue
            compressed = np.log1p((base_float - 1.0) * a_top) / math.log(base_float)
            base_label = "e" if abs(base_float - math.e) < 1e-9 else f"{base_float:g}".replace(".", "p")
            metrics[f"delta_importance_product_top{top_percent_int}_logbase_{base_label}"] = float(np.sum(compressed * d_top) / denom)

        for alpha in power_alphas:
            alpha_float = float(alpha)
            compressed = np.power(a_top, alpha_float)
            alpha_label = f"{alpha_float:g}".replace(".", "p")
            metrics[f"delta_importance_product_top{top_percent_int}_power_{alpha_label}"] = float(np.sum(compressed * d_top) / denom)

        for q in trim_quantiles:
            q_float = float(q)
            if raw_product.size:
                cap = float(np.nanpercentile(raw_product, q_float))
                trimmed = np.minimum(raw_product, cap)
            else:
                trimmed = raw_product
            q_label = f"{q_float:g}".replace(".", "p")
            metrics[f"delta_importance_product_top{top_percent_int}_trim_q{q_label}"] = float(np.sum(trimmed) / denom)

    return metrics


def importance_rank_bin_energy_fractions(delta_flat, importance_flat, *, n_bins: int = 100):
    import numpy as np

    d = np.asarray(delta_flat, dtype="float64").reshape(-1)
    a = np.asarray(importance_flat, dtype="float64").reshape(-1)
    n = min(d.size, a.size)
    d, a = np.abs(d[:n]), a[:n]
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if n == 0:
        return np.full(n_bins, np.nan, dtype="float64")

    rank_order = np.argsort(-np.abs(a), kind="stable")
    ranked_delta = d[rank_order]
    bin_ids = np.floor(np.arange(n, dtype="float64") * n_bins / n).astype(int)
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    fractions = np.bincount(bin_ids, weights=ranked_delta, minlength=n_bins).astype("float64")
    return fractions / float(d.sum() + 1e-12)


def _as_numpy(values):
    import numpy as np

    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype="float64")


def jpeg_zigzag_indices(h: int, w: int):
    import numpy as np

    h, w = int(h), int(w)
    if h <= 0 or w <= 0:
        raise ValueError("h and w must be positive")
    order: list[int] = []
    for diag in range(h + w - 1):
        r_min = max(0, diag - (w - 1))
        r_max = min(h - 1, diag)
        rows = range(r_min, r_max + 1)
        if diag % 2 == 0:
            rows = reversed(list(rows))
        for r in rows:
            c = diag - int(r)
            if 0 <= c < w:
                order.append(int(r) * w + int(c))
    return np.asarray(order, dtype=int)


def delta_zigzag_profile(delta_chw, *, mode: str = "mean_abs"):
    import numpy as np

    arr = _as_numpy(delta_chw)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected [C,H,W] or [1,C,H,W], got {arr.shape}")
    if mode == "mean_abs":
        spatial = np.mean(np.abs(arr), axis=0)
    elif mode == "signed_mean":
        spatial = np.mean(arr, axis=0)
    else:
        raise ValueError(f"Unsupported zig-zag profile mode: {mode!r}")
    order = jpeg_zigzag_indices(spatial.shape[0], spatial.shape[1])
    return spatial.reshape(-1)[order].astype("float64", copy=False)


def normalized_cumulative_curve(profile):
    import numpy as np

    mass = np.abs(np.asarray(profile, dtype="float64").reshape(-1))
    if mass.size == 0:
        return np.asarray([], dtype="float64"), np.asarray([], dtype="float64")
    total = float(mass.sum())
    x = np.linspace(1.0 / mass.size, 1.0, mass.size, dtype="float64")
    if total <= 0.0:
        return x, np.zeros_like(x)
    return x, np.cumsum(mass) / total


def normalized_cumulative_auc(profile):
    import numpy as np

    x, y = normalized_cumulative_curve(profile)
    if x.size == 0:
        return float("nan")
    return float(np.trapz(y, x))


def _safe_entropy(values):
    import numpy as np

    arr = np.asarray(values, dtype="float64").reshape(-1)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        return float("nan")
    p = arr / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(max(2, arr.size)))


def _center_of_mass(values_2d):
    import numpy as np

    arr = np.asarray(values_2d, dtype="float64")
    total = float(arr.sum())
    if total <= 0:
        return (float("nan"), float("nan"))
    yy, xx = np.indices(arr.shape)
    return (float((yy * arr).sum() / total), float((xx * arr).sum() / total))


def _top_set(values, k: int):
    import numpy as np

    arr = np.asarray(values, dtype="float64").reshape(-1)
    k = max(1, min(int(k), arr.size))
    idx = np.argpartition(-arr, kth=k - 1)[:k]
    return set(int(i) for i in idx)


def handcrafted_delta_importance_features(
    delta_chw,
    importance_chw,
    *,
    patch_bbox_xyxy=None,
    imgsz: int | None = None,
    rank_bins: int = 100,
):
    import numpy as np

    d_chw = _as_numpy(delta_chw)
    a_chw = _as_numpy(importance_chw)
    if d_chw.ndim == 4:
        d_chw = d_chw[0]
    if a_chw.ndim == 4:
        a_chw = a_chw[0]
    if d_chw.ndim != 3 or a_chw.ndim != 3:
        raise ValueError(f"Expected [C,H,W] arrays, got delta={d_chw.shape}, importance={a_chw.shape}")
    c = min(d_chw.shape[0], a_chw.shape[0])
    h = min(d_chw.shape[1], a_chw.shape[1])
    w = min(d_chw.shape[2], a_chw.shape[2])
    d_chw = d_chw[:c, :h, :w]
    a_chw = a_chw[:c, :h, :w]

    d = d_chw.reshape(-1)
    a = a_chw.reshape(-1)
    d_abs = np.abs(d)
    a_abs = np.abs(a)
    d_sum = float(d_abs.sum() + 1e-12)
    a_sum = float(a_abs.sum() + 1e-12)
    n = d_abs.size
    out: dict[str, float] = {}

    rank_fracs = importance_rank_bin_energy_fractions(d_abs, a_abs, n_bins=int(rank_bins))
    prefix = "hand_rank"
    for k in (1, 2, 3, 5, 6, 10, 25):
        out[f"{prefix}_top{k}_delta_frac"] = float(rank_fracs[:k].sum())
    out[f"{prefix}_ratio_top1_top5"] = float(rank_fracs[:1].sum() / (rank_fracs[:5].sum() + 1e-12))
    out[f"{prefix}_ratio_top3_top10"] = float(rank_fracs[:3].sum() / (rank_fracs[:10].sum() + 1e-12))
    out[f"{prefix}_ratio_top5_top25"] = float(rank_fracs[:5].sum() / (rank_fracs[:25].sum() + 1e-12))
    out[f"{prefix}_entropy"] = _safe_entropy(rank_fracs)
    out[f"{prefix}_peak_bin"] = float(int(np.nanargmax(rank_fracs)) + 1 if np.isfinite(rank_fracs).any() else np.nan)

    denom = float(np.linalg.norm(d_abs) * np.linalg.norm(a_abs) + 1e-12)
    out["hand_flat_abs_cosine"] = float(np.dot(d_abs, a_abs) / denom)
    for pct in (1, 3, 5, 10, 25):
        k = max(1, int(round(float(pct) / 100.0 * n)))
        a_top = _top_set(a_abs, k)
        d_top = _top_set(d_abs, k)
        union = len(a_top | d_top)
        out[f"hand_overlap_top{pct}_jaccard"] = float(len(a_top & d_top) / max(1, union))
        out[f"hand_overlap_top{pct}_delta_in_topA"] = float(d_abs[list(a_top)].sum() / d_sum)
        out[f"hand_overlap_top{pct}_importance_in_topDelta"] = float(a_abs[list(d_top)].sum() / a_sum)

    a_norm = a / float(np.max(a_abs) + 1e-12)
    signed = -a_norm * d
    unsigned = np.abs(a_norm) * d_abs
    correct = signed > 0
    correct_mass = float(unsigned[correct].sum())
    wrong_mass = float(unsigned[~correct].sum())
    total_mass = correct_mass + wrong_mass + 1e-12
    out["hand_signed_correct_mass_ratio"] = float(correct_mass / total_mass)
    out["hand_signed_correct_minus_wrong"] = float((correct_mass - wrong_mass) / total_mass)
    out["hand_signed_sum_norm_delta"] = float(signed.sum() / d_sum)
    pos = a > 0
    neg = a < 0
    out["hand_signed_pos_down_mass"] = float(unsigned[pos & (d < 0)].sum() / (unsigned[pos].sum() + 1e-12)) if np.any(pos) else float("nan")
    out["hand_signed_neg_up_mass"] = float(unsigned[neg & (d > 0)].sum() / (unsigned[neg].sum() + 1e-12)) if np.any(neg) else float("nan")

    d_sp = np.sqrt(np.mean(d_chw * d_chw, axis=0).clip(min=0.0))
    a_sp = np.sqrt(np.mean(a_chw * a_chw, axis=0).clip(min=0.0))
    sp_d = d_sp.reshape(-1)
    sp_a = a_sp.reshape(-1)
    sp_d_sum = float(sp_d.sum() + 1e-12)
    sp_a_sum = float(sp_a.sum() + 1e-12)
    out["hand_spatial_cosine"] = float(np.dot(sp_d, sp_a) / (np.linalg.norm(sp_d) * np.linalg.norm(sp_a) + 1e-12))
    out["hand_spatial_delta_entropy"] = _safe_entropy(sp_d)
    out["hand_spatial_importance_entropy"] = _safe_entropy(sp_a)
    cy_d, cx_d = _center_of_mass(d_sp)
    cy_a, cx_a = _center_of_mass(a_sp)
    out["hand_spatial_center_distance"] = float(np.sqrt((cy_d - cy_a) ** 2 + (cx_d - cx_a) ** 2) / max(1.0, np.sqrt(h * h + w * w)))
    for pct in (5, 10, 25):
        k_sp = max(1, int(round(float(pct) / 100.0 * sp_d.size)))
        a_top = _top_set(sp_a, k_sp)
        d_top = _top_set(sp_d, k_sp)
        out[f"hand_spatial_top{pct}_jaccard"] = float(len(a_top & d_top) / max(1, len(a_top | d_top)))
        out[f"hand_spatial_top{pct}_delta_in_topA"] = float(sp_d[list(a_top)].sum() / sp_d_sum)
    if patch_bbox_xyxy is not None and imgsz is not None:
        x1, y1, x2, y2 = [float(v) for v in patch_bbox_xyxy]
        gx1 = max(0, min(w, int(np.floor(x1 / float(imgsz) * w))))
        gx2 = max(0, min(w, int(np.ceil(x2 / float(imgsz) * w))))
        gy1 = max(0, min(h, int(np.floor(y1 / float(imgsz) * h))))
        gy2 = max(0, min(h, int(np.ceil(y2 / float(imgsz) * h))))
        mask = np.zeros((h, w), dtype=bool)
        if gx2 > gx1 and gy2 > gy1:
            mask[gy1:gy2, gx1:gx2] = True
        roi_energy = float(d_sp[mask].sum()) if mask.any() else 0.0
        out["hand_spatial_roi_delta_frac"] = float(roi_energy / sp_d_sum)
        out["hand_spatial_outside_roi_delta_frac"] = float(1.0 - roi_energy / sp_d_sum)

    d_ch = d_abs.reshape(c, h * w).sum(axis=1)
    a_ch = a_abs.reshape(c, h * w).sum(axis=1)
    ch_d_sum = float(d_ch.sum() + 1e-12)
    out["hand_channel_cosine"] = float(np.dot(d_ch, a_ch) / (np.linalg.norm(d_ch) * np.linalg.norm(a_ch) + 1e-12))
    out["hand_channel_delta_entropy"] = _safe_entropy(d_ch)
    out["hand_channel_importance_entropy"] = _safe_entropy(a_ch)
    out["hand_channel_max_delta_frac"] = float(np.max(d_ch) / ch_d_sum)
    for pct in (5, 10, 25):
        k_ch = max(1, int(round(float(pct) / 100.0 * c)))
        a_top = _top_set(a_ch, k_ch)
        d_top = _top_set(d_ch, k_ch)
        out[f"hand_channel_top{pct}_jaccard"] = float(len(a_top & d_top) / max(1, len(a_top | d_top)))
        out[f"hand_channel_top{pct}_delta_in_topA"] = float(d_ch[list(a_top)].sum() / ch_d_sum)
    return out


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
