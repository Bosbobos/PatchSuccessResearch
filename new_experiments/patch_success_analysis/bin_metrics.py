from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


def _rows_cache_key(rows_df, *, columns=None, params: dict[str, Any] | None = None) -> str:
    import numpy as np
    import pandas as pd

    if columns is None:
        columns = list(rows_df.columns)
    columns = [col for col in columns if col in rows_df.columns]
    frame = rows_df.loc[:, columns].copy()
    for col in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = frame[col].astype("float64")
    row_hash = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64", copy=False)
    payload = {
        "columns": list(columns),
        "params": params or {},
        "shape": list(frame.shape),
        "hash": hashlib.sha256(row_hash.tobytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]


def _cache_path(exp, rows_df, *, analysis_name: str, columns=None, params: dict[str, Any] | None = None) -> Path:
    key = _rows_cache_key(rows_df, columns=columns, params=params)
    return exp.derived_cache_dir / f"{analysis_name}_{key}.pkl"


def _save_pickle(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)
    payload["cache_path"] = str(path)
    payload["loaded_from_cache"] = False
    return payload


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    payload["cache_path"] = str(path)
    payload["loaded_from_cache"] = True
    return payload


def compute_or_load_elasticnet_bin_logreg(
    exp,
    rows_df,
    *,
    random_state: int = 17,
    cv_splits: int = 5,
    cv_repeats: int = 20,
    force: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    feature_cols = [f"segmentig_delta_energy_importance_binfrac_{idx:03d}" for idx in range(1, 101)]
    missing = [col for col in feature_cols if col not in rows_df.columns]
    if missing:
        raise RuntimeError(f"Missing bin-fraction features: {missing[:5]}")
    params = {
        "version": 1,
        "random_state": int(random_state),
        "cv_splits": int(cv_splits),
        "cv_repeats": int(cv_repeats),
        "feature_cols": feature_cols,
    }
    cache_path = _cache_path(
        exp,
        rows_df,
        analysis_name="binmetrics_elasticnet_bin_logreg_v1",
        columns=["success", *feature_cols],
        params=params,
    )
    if cache_path.exists() and not force:
        return _load_pickle(cache_path)

    x_df = rows_df[feature_cols].astype("float64")
    y = rows_df["success"].astype(int).to_numpy()
    finite_mask = np.isfinite(x_df.to_numpy()).all(axis=1)
    x = x_df.loc[finite_mask].to_numpy()
    y_fit = y[finite_mask]
    if len(np.unique(y_fit)) != 2:
        raise RuntimeError("Need both success/fail classes for logistic regression.")

    cv = RepeatedStratifiedKFold(n_splits=int(cv_splits), n_repeats=int(cv_repeats), random_state=int(random_state))
    logreg = LogisticRegressionCV(
        Cs=np.logspace(-4, 2, 20),
        penalty="elasticnet",
        solver="saga",
        scoring="roc_auc",
        class_weight="balanced",
        l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
        cv=cv,
        max_iter=20000,
        n_jobs=-1,
        refit=True,
        random_state=int(random_state),
    )
    logreg.fit(x, y_fit)

    score_key = sorted(logreg.scores_.keys())[-1]
    cv_scores = logreg.scores_[score_key]
    cv_mean = cv_scores.mean(axis=0)
    train_score = logreg.decision_function(x)
    coef = logreg.coef_[0].copy()
    coef_df = pd.DataFrame(
        {
            "bin": np.arange(1, 101),
            "rank_percent_from": np.arange(0, 100),
            "rank_percent_to": np.arange(1, 101),
            "coef": coef,
        }
    )
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df["nonzero"] = coef_df["coef"].abs() > 1e-10

    score_series = pd.Series(np.nan, index=rows_df.index, dtype="float64")
    score_series.loc[finite_mask] = train_score
    out = {
        "feature_cols": feature_cols,
        "coef_df": coef_df,
        "score_series": score_series,
        "metrics": {
            "n_examples": int(len(y_fit)),
            "n_features": int(x.shape[1]),
            "best_cv_auc": float(np.nanmax(cv_mean)),
            "train_auc": float(roc_auc_score(y_fit, train_score)),
            "best_C": float(logreg.C_[0]),
            "best_l1_ratio": float(logreg.l1_ratio_[0]),
            "intercept": float(logreg.intercept_[0]),
            "nonzero_coefficients": int(coef_df["nonzero"].sum()),
        },
    }
    return _save_pickle(cache_path, out)


def resolve_layer_map_path(cache_path):
    path = Path(str(cache_path))
    if path.exists():
        return path
    alt = Path("new_experiments") / path
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Layer map cache not found: {cache_path}")


def manual_weight_vector(n_bins, family):
    import numpy as np

    n_bins = int(n_bins)
    j = np.arange(n_bins, dtype="float64")
    if family == "flat":
        weights = np.ones(n_bins, dtype="float64")
    elif family == "reciprocal":
        weights = 1.0 / (j + 1.0)
    elif family == "sqrt_reciprocal":
        weights = 1.0 / np.sqrt(j + 1.0)
    elif family == "linear":
        weights = np.linspace(1.0, 1.0 / float(n_bins), n_bins)
    elif family == "exp":
        half_life = max(1.0, n_bins / 5.0)
        weights = 0.5 ** (j / half_life)
    else:
        raise ValueError(f"Unknown weight family: {family}")
    return weights / max(float(weights[0]), 1e-12)


def format_weight_label(family, n_bins):
    if family == "flat":
        return "[1, 1, ..., 1]"
    if family == "reciprocal":
        return f"[1, 1/2, ..., 1/{int(n_bins)}]"
    if family == "sqrt_reciprocal":
        return f"[1, 1/sqrt(2), ..., 1/sqrt({int(n_bins)})]"
    weights = manual_weight_vector(n_bins, family)
    return f"[{weights[0]:.3g}, {weights[1]:.3g}, ..., {weights[-1]:.3g}]"


def compute_or_load_bin_granularity_analysis(
    exp,
    rows_df,
    *,
    selection_random_state: int = 31,
    train_per_class: int = 100,
    val_per_class: int = 100,
    force: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .metrics import best_accuracy, importance_rank_bin_energy_fractions

    one_percent_cols = [f"segmentig_delta_energy_importance_binfrac_{idx:03d}" for idx in range(1, 101)]
    required_cols = ["success", "layer_maps_cache_path", "segmentig_delta_energy_importance_bins_top10_reciprocal", *one_percent_cols]
    missing = [col for col in required_cols if col not in rows_df.columns]
    if missing:
        raise RuntimeError(f"Missing bin granularity inputs: {missing[:5]}")
    params = {
        "version": 1,
        "selection_random_state": int(selection_random_state),
        "train_per_class": int(train_per_class),
        "val_per_class": int(val_per_class),
    }
    cache_path = _cache_path(
        exp,
        rows_df,
        analysis_name="binmetrics_bin_granularity_v1",
        columns=required_cols,
        params=params,
    )
    if cache_path.exists() and not force:
        return _load_pickle(cache_path)

    rng = np.random.default_rng(int(selection_random_state))
    y = rows_df["success"].astype(bool).to_numpy().astype(int)
    success_idx = np.flatnonzero(y == 1)
    fail_idx = np.flatnonzero(y == 0)
    rng.shuffle(success_idx)
    rng.shuffle(fail_idx)
    needed_per_class = int(train_per_class) + int(val_per_class)
    if len(success_idx) < needed_per_class or len(fail_idx) < needed_per_class:
        raise RuntimeError(
            f"Need {needed_per_class} success and {needed_per_class} fail for split, "
            f"got success={len(success_idx)}, fail={len(fail_idx)}"
        )
    train_idx = np.concatenate([success_idx[:train_per_class], fail_idx[:train_per_class]])
    val_idx = np.concatenate(
        [
            success_idx[train_per_class:needed_per_class],
            fail_idx[train_per_class:needed_per_class],
        ]
    )
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    def eval_scores(train_scores, train_y, eval_scores, eval_y):
        best = best_accuracy(train_y.astype(bool), train_scores)
        direction = int(best["direction"])
        threshold = float(best["threshold"])
        pred = eval_scores >= threshold if direction == 1 else eval_scores <= threshold
        return {
            "accuracy": float(accuracy_score(eval_y, pred)),
            "roc_auc": float(roc_auc_score(eval_y, eval_scores)),
            "direction": direction,
            "threshold": threshold,
            "train_best_accuracy": float(best["accuracy"]),
        }

    def fit_logreg_variant(name, x, *, model_kind="cv_l2", fixed_c=0.03):
        x = np.asarray(x, dtype="float64")
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        if model_kind == "fixed_l2":
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=float(fixed_c), penalty="l2", class_weight="balanced", max_iter=20000, solver="lbfgs"),
            )
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(
                    Cs=np.logspace(-5, 1, 25),
                    cv=5,
                    penalty="l2",
                    scoring="roc_auc",
                    class_weight="balanced",
                    max_iter=20000,
                    solver="lbfgs",
                    refit=True,
                ),
            )
        model.fit(x_train, y_train)
        train_scores = model.decision_function(x_train)
        val_scores = model.decision_function(x_val)
        all_scores = model.decision_function(x)
        train_eval = eval_scores(train_scores, y_train, train_scores, y_train)
        val_eval = eval_scores(train_scores, y_train, val_scores, y_val)
        all_best = best_accuracy(y.astype(bool), all_scores)
        lr = model.named_steps.get("logisticregressioncv") or model.named_steps.get("logisticregression")
        return {
            "variant": name,
            "model_kind": model_kind,
            "n_features": int(x.shape[1]),
            "C": float(lr.C_[0]) if hasattr(lr, "C_") else float(lr.C),
            "train_acc": train_eval["accuracy"],
            "train_auc": float(roc_auc_score(y_train, train_scores)),
            "val_acc": val_eval["accuracy"],
            "val_auc": val_eval["roc_auc"],
            "all_best_acc": float(all_best["accuracy"]),
            "all_auc": float(roc_auc_score(y, all_scores)),
            "coef": lr.coef_[0].copy(),
        }

    x_1pct_5 = rows_df[one_percent_cols[:5]].astype("float64").to_numpy()
    x_1pct_10 = rows_df[one_percent_cols[:10]].astype("float64").to_numpy()

    def load_01pct_features(row):
        cache_path_value = row.get("layer_maps_cache_path")
        if not isinstance(cache_path_value, str) or not cache_path_value:
            raise RuntimeError("rows_df has no layer_maps_cache_path; cannot compute cached 0.1% bins.")
        with np.load(resolve_layer_map_path(cache_path_value), allow_pickle=False) as data:
            delta = data["delta_chw"].astype("float32", copy=False).reshape(-1)
            segmentig = data["segmentig_chw"].astype("float32", copy=False).reshape(-1)
        return importance_rank_bin_energy_fractions(delta, segmentig, n_bins=1000)

    binfrac_01 = np.vstack([load_01pct_features(row) for _, row in rows_df.iterrows()])
    x_01pct_1 = binfrac_01[:, :10]
    x_01pct_5 = binfrac_01[:, :50]
    x_01pct_10 = binfrac_01[:, :100]

    feature_sets = [
        ("1% bins: top 5%", x_1pct_5),
        ("1% bins: top 10%", x_1pct_10),
        ("0.1% bins: top 1%", x_01pct_1),
        ("0.1% bins: top 5%", x_01pct_5),
        ("0.1% bins: top 10%", x_01pct_10),
    ]
    strong_cs = [0.03, 0.01, 0.003, 0.001]
    runs = []
    for name, x in feature_sets:
        runs.append(fit_logreg_variant(name + " / L2 CV", x, model_kind="cv_l2"))
        for c_value in strong_cs:
            runs.append(fit_logreg_variant(name + f" / L2 C={c_value:g}", x, model_kind="fixed_l2", fixed_c=c_value))
    bin_logreg_df = pd.DataFrame([{k: v for k, v in row.items() if k != "coef"} for row in runs])

    def simple_metric_eval(name, scores, *, kind="manual", weights=""):
        scores = np.asarray(scores, dtype="float64")
        train_eval = eval_scores(scores[train_idx], y[train_idx], scores[train_idx], y[train_idx])
        val_eval = eval_scores(scores[train_idx], y[train_idx], scores[val_idx], y[val_idx])
        return {
            "method": name,
            "weights": weights,
            "train_acc": train_eval["accuracy"],
            "val_acc": val_eval["accuracy"],
            "val_auc": val_eval["roc_auc"],
            "kind": kind,
        }

    manual_01_rows = []
    for cutoff_percent, n_bins in [(5, 50), (10, 100)]:
        for family in ["flat", "linear", "reciprocal", "sqrt_reciprocal", "exp"]:
            weights = manual_weight_vector(n_bins, family)
            scores = binfrac_01[:, :n_bins] @ weights
            name = f"0.1% top{cutoff_percent} {family}"
            eval_row = simple_metric_eval(name, scores)
            eval_row["cutoff_percent"] = cutoff_percent
            eval_row["weight_family"] = family
            eval_row["n_bins"] = n_bins
            eval_row["weights"] = format_weight_label(family, n_bins)
            manual_01_rows.append(eval_row)
    manual_01_df = pd.DataFrame(manual_01_rows).sort_values(["val_acc", "val_auc"], ascending=False).reset_index(drop=True)

    manual_compare_rows = [
        simple_metric_eval("top1% flat", rows_df[one_percent_cols[:1]].sum(axis=1).to_numpy(), weights="[1]"),
        simple_metric_eval("top5% flat", rows_df[one_percent_cols[:5]].sum(axis=1).to_numpy(), weights="[1, 1, ..., 1]"),
        simple_metric_eval("top10% flat", rows_df[one_percent_cols[:10]].sum(axis=1).to_numpy(), weights="[1, 1, ..., 1]"),
        simple_metric_eval(
            "top10% reciprocal",
            rows_df["segmentig_delta_energy_importance_bins_top10_reciprocal"].to_numpy(),
            weights="[1, 1/2, ..., 1/10]",
        ),
    ]
    for _, row in manual_01_df.iterrows():
        manual_compare_rows.append(
            {
                "method": row["method"],
                "weights": row["weights"],
                "train_acc": float(row["train_acc"]),
                "val_acc": float(row["val_acc"]),
                "val_auc": float(row["val_auc"]),
                "kind": "manual_0.1%",
            }
        )
    best_ml_row = bin_logreg_df.sort_values(["val_acc", "val_auc"], ascending=False).iloc[0]
    manual_compare_rows.append(
        {
            "method": "best ML\n" + str(best_ml_row["variant"]).replace(" / ", "\n"),
            "weights": "learned",
            "train_acc": float(best_ml_row["train_acc"]),
            "val_acc": float(best_ml_row["val_acc"]),
            "val_auc": float(best_ml_row["val_auc"]),
            "kind": "learned",
        }
    )
    method_compare_df = pd.DataFrame(manual_compare_rows).sort_values(["val_acc", "val_auc"], ascending=False).reset_index(drop=True)

    overall_rows = []
    exclude_overall_cols = {
        "success",
        "path",
        "conf_clean",
        "conf_patch",
        "drop",
        "layer_maps_loaded_from_cache",
        "segmentig_logreg_bin_score",
    }
    for col in rows_df.columns:
        if col in exclude_overall_cols:
            continue
        if not pd.api.types.is_numeric_dtype(rows_df[col]):
            continue
        scores = rows_df[col].astype("float64").to_numpy()
        if not np.isfinite(scores).all() or np.nanstd(scores) <= 1e-12:
            continue
        eval_row = simple_metric_eval(col, scores, kind="precomputed", weights="")
        eval_row["source"] = "precomputed"
        overall_rows.append(eval_row)
    for _, row in manual_01_df.iterrows():
        overall_rows.append(
            {
                "method": row["method"],
                "weights": row["weights"],
                "train_acc": float(row["train_acc"]),
                "val_acc": float(row["val_acc"]),
                "val_auc": float(row["val_auc"]),
                "kind": "manual_0.1%",
                "source": "manual_0.1%",
            }
        )
    for _, row in bin_logreg_df.iterrows():
        overall_rows.append(
            {
                "method": "ML: " + str(row["variant"]),
                "weights": "learned",
                "train_acc": float(row["train_acc"]),
                "val_acc": float(row["val_acc"]),
                "val_auc": float(row["val_auc"]),
                "kind": "learned",
                "source": "ML",
            }
        )
    overall_val_df = pd.DataFrame(overall_rows).sort_values(["val_acc", "val_auc"], ascending=False).reset_index(drop=True)
    out = {
        "bin_logreg_df": bin_logreg_df,
        "manual_01_df": manual_01_df,
        "method_compare_df": method_compare_df,
        "overall_val_df": overall_val_df,
        "runs": runs,
        "feature_set_names": [name for name, _ in feature_sets],
        "strong_cs": strong_cs,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }
    return _save_pickle(cache_path, out)


def compute_or_load_unregularized_logreg_capacity(
    exp,
    rows_df,
    *,
    cutoffs=(1, 5, 10, 25, 50, 100),
    granularities=("0.1%", "1%"),
    selection_random_state: int = 31,
    train_per_class: int = 100,
    val_per_class: int = 100,
    force: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .metrics import best_accuracy, importance_rank_bin_energy_fractions

    cutoffs = tuple(float(value) for value in cutoffs)
    granularities = tuple(str(value) for value in granularities)
    one_percent_cols = [f"segmentig_delta_energy_importance_binfrac_{idx:03d}" for idx in range(1, 101)]
    required_cols = ["success", "layer_maps_cache_path", *one_percent_cols]
    missing = [col for col in required_cols if col not in rows_df.columns]
    if missing:
        raise RuntimeError(f"Missing unregularized logreg inputs: {missing[:5]}")
    params = {
        "version": 1,
        "cutoffs": cutoffs,
        "granularities": granularities,
        "selection_random_state": int(selection_random_state),
        "train_per_class": int(train_per_class),
        "val_per_class": int(val_per_class),
    }
    cache_path = _cache_path(
        exp,
        rows_df,
        analysis_name="binmetrics_unregularized_logreg_capacity_v1",
        columns=required_cols,
        params=params,
    )
    if cache_path.exists() and not force:
        return _load_pickle(cache_path)

    rng = np.random.default_rng(int(selection_random_state))
    y = rows_df["success"].astype(bool).to_numpy().astype(int)
    success_idx = np.flatnonzero(y == 1)
    fail_idx = np.flatnonzero(y == 0)
    rng.shuffle(success_idx)
    rng.shuffle(fail_idx)
    needed_per_class = int(train_per_class) + int(val_per_class)
    if len(success_idx) < needed_per_class or len(fail_idx) < needed_per_class:
        raise RuntimeError(
            f"Need {needed_per_class} success and {needed_per_class} fail for split, "
            f"got success={len(success_idx)}, fail={len(fail_idx)}"
        )
    train_idx = np.concatenate([success_idx[:train_per_class], fail_idx[:train_per_class]])
    val_idx = np.concatenate(
        [
            success_idx[train_per_class:needed_per_class],
            fail_idx[train_per_class:needed_per_class],
        ]
    )
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    x_1pct_all = rows_df[one_percent_cols].astype("float64").to_numpy()

    def load_01pct_features(row):
        cache_path_value = row.get("layer_maps_cache_path")
        if not isinstance(cache_path_value, str) or not cache_path_value:
            raise RuntimeError("rows_df has no layer_maps_cache_path; cannot compute cached 0.1% bins.")
        with np.load(resolve_layer_map_path(cache_path_value), allow_pickle=False) as data:
            delta = data["delta_chw"].astype("float32", copy=False).reshape(-1)
            segmentig = data["segmentig_chw"].astype("float32", copy=False).reshape(-1)
        return importance_rank_bin_energy_fractions(delta, segmentig, n_bins=1000)

    x_01pct_all = np.vstack([load_01pct_features(row) for _, row in rows_df.iterrows()])

    def fit_unregularized(x):
        def make_model(penalty):
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty=penalty,
                    class_weight="balanced",
                    max_iter=50000,
                    solver="lbfgs",
                ),
            )

        model = make_model(None)
        try:
            model.fit(x[train_idx], y[train_idx])
        except Exception as exc:  # noqa: BLE001 - sklearn changed the no-penalty spelling across versions.
            if "penalty" not in str(exc):
                raise
            model = make_model("none")
            model.fit(x[train_idx], y[train_idx])
        train_scores = model.decision_function(x[train_idx])
        val_scores = model.decision_function(x[val_idx])
        best = best_accuracy(y[train_idx].astype(bool), train_scores)
        direction = int(best["direction"])
        threshold = float(best["threshold"])
        train_pred = train_scores >= threshold if direction == 1 else train_scores <= threshold
        val_pred = val_scores >= threshold if direction == 1 else val_scores <= threshold
        lr = model.named_steps["logisticregression"]
        return {
            "train_acc": float(accuracy_score(y[train_idx], train_pred)),
            "val_acc": float(accuracy_score(y[val_idx], val_pred)),
            "train_auc": float(roc_auc_score(y[train_idx], train_scores)),
            "val_auc": float(roc_auc_score(y[val_idx], val_scores)),
            "threshold": threshold,
            "direction": direction,
            "coef_l2": float(np.linalg.norm(lr.coef_[0])),
            "n_iter": int(np.max(lr.n_iter_)),
            "converged": bool(np.max(lr.n_iter_) < lr.max_iter),
        }

    rows = []
    for granularity in granularities:
        x_all = x_01pct_all if granularity == "0.1%" else x_1pct_all
        bins_per_percent = 10 if granularity == "0.1%" else 1
        for cutoff in cutoffs:
            n_bins = min(x_all.shape[1], max(1, int(round(cutoff * bins_per_percent))))
            x = x_all[:, :n_bins]
            metrics = fit_unregularized(x)
            rows.append(
                {
                    "granularity": granularity,
                    "cutoff_percent": float(cutoff),
                    "n_bins": int(n_bins),
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    **metrics,
                }
            )
    results_df = pd.DataFrame(rows).sort_values(["granularity", "cutoff_percent"]).reset_index(drop=True)
    out = {
        "results_df": results_df,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "cutoffs": cutoffs,
        "granularities": granularities,
    }
    return _save_pickle(cache_path, out)


def compute_or_load_robust_product_compression_comparison(
    exp,
    rows_df,
    *,
    top_percents=(3, 5, 10),
    log_bases=(1.5, 2.0, 2.718281828459045, 10.0),
    power_alphas=(0.25, 0.5, 0.75),
    trim_quantiles=(99.0, 99.5),
    force: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from .metrics import metric_quality_rows, robust_importance_product_metrics

    required_cols = ["success", "layer_maps_cache_path"]
    missing = [col for col in required_cols if col not in rows_df.columns]
    if missing:
        raise RuntimeError(f"Missing robust product inputs: {missing}")
    params = {
        "version": 1,
        "top_percents": tuple(float(v) for v in top_percents),
        "log_bases": tuple(float(v) for v in log_bases),
        "power_alphas": tuple(float(v) for v in power_alphas),
        "trim_quantiles": tuple(float(v) for v in trim_quantiles),
    }
    cache_path = _cache_path(
        exp,
        rows_df,
        analysis_name="binmetrics_robust_product_compression_v1",
        columns=required_cols,
        params=params,
    )
    if cache_path.exists() and not force:
        return _load_pickle(cache_path)

    metric_rows = []
    skipped_rows = []
    for _, row in rows_df.iterrows():
        cache_path_value = row.get("layer_maps_cache_path")
        if not isinstance(cache_path_value, str) or not cache_path_value:
            skipped_rows.append({"path": row.get("path", ""), "reason": "missing layer_maps_cache_path"})
            continue
        try:
            with np.load(resolve_layer_map_path(cache_path_value), allow_pickle=False) as data:
                delta = data["delta_chw"].astype("float32", copy=False).reshape(-1)
                segmentig = data["segmentig_chw"].astype("float32", copy=False).reshape(-1)
            metrics = robust_importance_product_metrics(
                delta,
                segmentig,
                top_percents=top_percents,
                log_bases=log_bases,
                power_alphas=power_alphas,
                trim_quantiles=trim_quantiles,
            )
        except Exception as exc:  # noqa: BLE001 - one bad cached map should not stop comparison.
            skipped_rows.append({"path": row.get("path", ""), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        metric_rows.append(
            {
                "path": row.get("path", ""),
                "success": bool(row["success"]),
                **{f"segmentig_{name}": float(value) for name, value in metrics.items()},
            }
        )

    if not metric_rows:
        raise RuntimeError(f"No robust product rows computed; skipped={len(skipped_rows)}")
    metrics_df = pd.DataFrame(metric_rows)
    labels = metrics_df["success"].astype(bool).tolist()
    metric_cols = [col for col in metrics_df.columns if col.startswith("segmentig_delta_importance_product_top")]
    quality_df = pd.DataFrame(
        metric_quality_rows(labels, {name: metrics_df[name].to_numpy(dtype="float64") for name in metric_cols})
    )

    def parse_family(metric: str) -> str:
        if "_logbase_" in metric:
            return "log importance"
        if "_power_" in metric:
            return "power importance"
        if "_trim_" in metric:
            return "trim product"
        return "raw"

    def parse_top(metric: str) -> int:
        part = metric.split("_top", 1)[1].split("_", 1)[0]
        return int(part)

    quality_df["family"] = quality_df["metric"].map(parse_family)
    quality_df["top_percent"] = quality_df["metric"].map(parse_top)
    quality_df = quality_df.sort_values(["best_accuracy", "roc_auc"], ascending=False).reset_index(drop=True)
    out = {
        "metrics_df": metrics_df,
        "quality_df": quality_df,
        "skipped": skipped_rows,
        "n_skipped": len(skipped_rows),
    }
    return _save_pickle(cache_path, out)


def compute_or_load_handcrafted_selection(
    exp,
    rows_df,
    *,
    selection_random_state: int = 31,
    selection_examples_per_class: int = 1000,
    selection_cv_splits: int = 5,
    selection_cv_repeats: int = 20,
    force: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import RepeatedStratifiedKFold

    from .metrics import best_accuracy, roc_auc_score_manual

    y = rows_df["success"].astype(bool).to_numpy()
    exclude_cols = {"success", "path", "conf_clean", "conf_patch", "drop"}
    feature_cols = []
    for col in rows_df.columns:
        if col in exclude_cols:
            continue
        if not pd.api.types.is_numeric_dtype(rows_df[col]):
            continue
        values = rows_df[col].astype("float64").to_numpy()
        if np.isfinite(values).all() and np.nanstd(values) > 1e-12:
            feature_cols.append(col)
    params = {
        "version": 1,
        "selection_random_state": int(selection_random_state),
        "selection_examples_per_class": int(selection_examples_per_class),
        "selection_cv_splits": int(selection_cv_splits),
        "selection_cv_repeats": int(selection_cv_repeats),
        "feature_cols": feature_cols,
    }
    cache_path = _cache_path(
        exp,
        rows_df,
        analysis_name="binmetrics_handcrafted_selection_v1",
        columns=["success", *feature_cols],
        params=params,
    )
    if cache_path.exists() and not force:
        return _load_pickle(cache_path)

    rng = np.random.default_rng(int(selection_random_state))
    success_idx = np.flatnonzero(y)
    fail_idx = np.flatnonzero(~y)
    rng.shuffle(success_idx)
    rng.shuffle(fail_idx)
    needed_per_class = 2 * int(selection_examples_per_class)
    if success_idx.size < needed_per_class or fail_idx.size < needed_per_class:
        raise RuntimeError(
            f"Need at least {needed_per_class} success and {needed_per_class} fail examples in sf rows; "
            f"got success={success_idx.size}, fail={fail_idx.size}."
        )
    train_idx = np.concatenate([success_idx[:selection_examples_per_class], fail_idx[:selection_examples_per_class]])
    val_idx = np.concatenate(
        [
            success_idx[selection_examples_per_class:needed_per_class],
            fail_idx[selection_examples_per_class:needed_per_class],
        ]
    )
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    def eval_with_train_threshold(train_scores, train_y, eval_scores, eval_y):
        best = best_accuracy(train_y, train_scores)
        direction = int(best["direction"])
        threshold = float(best["threshold"])
        pred = eval_scores >= threshold if direction == 1 else eval_scores <= threshold
        return {
            "accuracy": float(np.mean(pred == eval_y)),
            "roc_auc": float(roc_auc_score_manual(eval_y, eval_scores)),
            "direction": direction,
            "threshold": threshold,
            "train_best_accuracy": float(best["accuracy"]),
        }

    def inner_cv(score):
        cv = RepeatedStratifiedKFold(
            n_splits=int(selection_cv_splits),
            n_repeats=int(selection_cv_repeats),
            random_state=17,
        )
        train_scores = score[train_idx]
        train_y = y[train_idx]
        accs, aucs = [], []
        for tr, te in cv.split(train_scores.reshape(-1, 1), train_y):
            res = eval_with_train_threshold(train_scores[tr], train_y[tr], train_scores[te], train_y[te])
            accs.append(res["accuracy"])
            aucs.append(res["roc_auc"])
        return float(np.mean(accs)), float(np.std(accs)), float(np.mean(aucs))

    candidates = []
    for col in feature_cols:
        score = rows_df[col].astype("float64").to_numpy()
        cv_acc, cv_acc_std, cv_auc = inner_cv(score)
        train_eval = eval_with_train_threshold(score[train_idx], y[train_idx], score[train_idx], y[train_idx])
        val_eval = eval_with_train_threshold(score[train_idx], y[train_idx], score[val_idx], y[val_idx])
        candidates.append(
            {
                "name": col,
                "family": "single",
                "formula": col,
                "cv_acc": cv_acc,
                "cv_acc_std": cv_acc_std,
                "cv_auc": cv_auc,
                "train_best_acc": train_eval["train_best_accuracy"],
                "val_acc": val_eval["accuracy"],
                "val_auc": val_eval["roc_auc"],
                "direction": val_eval["direction"],
                "threshold": val_eval["threshold"],
            }
        )
    selection_df = pd.DataFrame(candidates).sort_values(["cv_acc", "cv_auc"], ascending=False).reset_index(drop=True)
    out = {
        "selection_df": selection_df,
        "feature_cols": feature_cols,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "summary": {
            "examples": int(len(rows_df)),
            "features": int(len(feature_cols)),
            "train": int(len(train_idx)),
            "train_success": int(y[train_idx].sum()),
            "train_fail": int((~y[train_idx]).sum()),
            "val": int(len(val_idx)),
            "val_success": int(y[val_idx].sum()),
            "val_fail": int((~y[val_idx]).sum()),
        },
    }
    return _save_pickle(cache_path, out)
