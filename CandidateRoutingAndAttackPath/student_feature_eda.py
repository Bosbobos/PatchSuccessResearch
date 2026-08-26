from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _preprocess_pair
from .candidate_reserve import _cache_lookup
from .candidate_routing import _box_iou, _xywh_to_xyxy
from .common import REPO_ROOT, load_experiment, release_accelerator_memory, stable_hash
from .component_student import (
    _capture_with_grad,
    _coordinate_arrays,
    _reset_detect_inference_cache,
)
from .component_targeted_patch import _record_lookup
from .large_component_student import (
    LargeComponentStudentConfig,
    _attacked_reference_statistics,
    _blind_candidate_indices,
    _blind_targets,
    _labels_on_candidates,
    _teacher_arrays,
)
from .mechanism_aware_patch import dynamic_score_geometry_loss


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "CandidateRoutingAndAttackPath" / "student_feature_eda_outputs"
)

FEATURE_GROUPS = {
    "activation": ("z",),
    "identity": ("level", "channel"),
    "position": ("y", "x"),
    "functional": ("gradient", "z_gradient"),
    "local": (
        "local_mean_z",
        "local_std_z",
        "center_minus_local",
    ),
    "branch": (
        "class_gradient",
        "geometry_gradient",
        "z_class_gradient",
        "z_geometry_gradient",
    ),
    "consensus": (
        "gradient_std",
        "gradient_max_abs",
        "gradient_sign_agreement",
    ),
    "relative": ("gradient_abs_rank", "z_abs_rank"),
}
ALL_FEATURES = tuple(
    feature
    for group in FEATURE_GROUPS.values()
    for feature in group
)


@dataclass(slots=True)
class StudentFeatureEdaConfig:
    pool_run: str
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "mps"
    require_device: bool = False
    positive_per_scene: int = 512
    negative_per_scene: int = 512
    candidates_per_level: int = 4000
    blind_top_clusters: int = 3
    cv_folds: int = 3
    probe_max_train_rows: int = 250_000
    probe_max_eval_rows: int = 120_000
    probe_max_iter: int = 55
    probe_max_leaf_nodes: int = 31
    univariate_max_rows: int = 80_000
    min_greedy_relative_improvement: float = 0.005
    max_scenes: int | None = None
    seed: int = 3107
    method_version: int = 1


def _smoothmax(values, temperature: float = 0.35):
    import torch

    return temperature * (
        torch.logsumexp(values / temperature, dim=0)
        - np.log(max(int(values.numel()), 1))
    )


def _branch_objectives(decoded, target_box, class_id):
    import torch

    boxes = _xywh_to_xyxy(decoded[0, :4].transpose(0, 1))
    iou = _box_iou(boxes, target_box).reshape(-1)
    score = decoded[0, 4 + int(class_id), :].clamp(1e-6, 1.0 - 1e-6)
    membership = torch.sigmoid((iou - 0.50) / 0.07).clamp_min(1e-6)
    class_risk = torch.logit(score) + 4.0 * torch.log(
        membership.detach()
    )
    geometry_risk = torch.logit(score.detach()) + 4.0 * torch.log(
        membership
    )
    return _smoothmax(class_risk), _smoothmax(geometry_risk)


def _observable_gradient_families(model, detect, image, config):
    import torch

    _reset_detect_inference_cache(detect)
    image = image.detach().requires_grad_(True)
    decoded, levels = _capture_with_grad(model, detect, image)
    base_config = LargeComponentStudentConfig(
        blind_top_clusters=int(config.blind_top_clusters)
    )
    targets, _cluster_count = _blind_targets(
        detect, levels, base_config
    )
    if not targets:
        objective = decoded[:, 4:, :].sigmoid().amax()
        gradient = torch.autograd.grad(objective, levels)
        zeros = [torch.zeros_like(value) for value in gradient]
        return (
            [value.detach() for value in levels],
            [value.detach() for value in gradient],
            [value.detach() for value in gradient],
            zeros,
            zeros,
            [value.detach().abs() for value in gradient],
            [torch.ones_like(value) for value in gradient],
        )

    combined_by_target = [[] for _ in levels]
    class_by_target = [[] for _ in levels]
    geometry_by_target = [[] for _ in levels]
    for target in targets:
        target_box = torch.as_tensor(
            [target["box"]],
            device=levels[0].device,
            dtype=torch.float32,
        )
        class_id = int(target["class_id"])
        combined, _ = dynamic_score_geometry_loss(
            decoded,
            target_box,
            torch.as_tensor(
                [class_id], device=levels[0].device, dtype=torch.long
            ),
            match_iou=0.50,
            iou_temperature=0.07,
            iou_weight=4.0,
            smoothmax_temperature=0.35,
        )
        class_objective, geometry_objective = _branch_objectives(
            decoded, target_box, class_id
        )
        combined_gradients = torch.autograd.grad(
            combined, levels, retain_graph=True
        )
        class_gradients = torch.autograd.grad(
            class_objective, levels, retain_graph=True
        )
        geometry_gradients = torch.autograd.grad(
            geometry_objective, levels, retain_graph=True
        )
        for level, value in enumerate(combined_gradients):
            combined_by_target[level].append(value.detach())
        for level, value in enumerate(class_gradients):
            class_by_target[level].append(value.detach())
        for level, value in enumerate(geometry_gradients):
            geometry_by_target[level].append(value.detach())

    combined_mean = []
    class_mean = []
    geometry_mean = []
    gradient_std = []
    gradient_max_abs = []
    gradient_sign_agreement = []
    for combined, class_values, geometry in zip(
        combined_by_target,
        class_by_target,
        geometry_by_target,
        strict=True,
    ):
        stack = torch.stack(combined)
        combined_mean.append(stack.mean(dim=0))
        class_mean.append(torch.stack(class_values).mean(dim=0))
        geometry_mean.append(torch.stack(geometry).mean(dim=0))
        gradient_std.append(stack.std(dim=0, unbiased=False))
        gradient_max_abs.append(stack.abs().amax(dim=0))
        gradient_sign_agreement.append(
            stack.sign().float().mean(dim=0).abs()
        )
    return (
        [value.detach() for value in levels],
        combined_mean,
        class_mean,
        geometry_mean,
        gradient_std,
        gradient_max_abs,
        gradient_sign_agreement,
    )


def _selected_values(value, indices):
    import torch

    index = torch.as_tensor(
        indices, device=value.device, dtype=torch.long
    )
    return value[0].reshape(-1)[index].detach().float().cpu().numpy()


def _normalize(values):
    values = np.asarray(values, dtype=np.float32)
    scale = max(float(np.sqrt(np.mean(np.square(values)))), 1e-8)
    return values / scale


def _rank01(values):
    values = np.asarray(values)
    if not len(values):
        return np.empty(0, dtype=np.float32)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks / max(len(values) - 1, 1)


def _sample_rows(frame, config, scene_index):
    rng = np.random.default_rng(int(config.seed) + 7919 * scene_index)
    target = frame.teacher_component.to_numpy(float)
    positive = np.flatnonzero(np.abs(target) > 1e-8)
    negative = np.flatnonzero(np.abs(target) <= 1e-8)

    def choose(indices, count, priority):
        if len(indices) <= count:
            return indices
        top_n = count // 2
        ordered = indices[np.argsort(-priority[indices], kind="stable")]
        top = ordered[:top_n]
        remaining = np.setdiff1d(indices, top, assume_unique=False)
        random = rng.choice(
            remaining, size=count - len(top), replace=False
        )
        return np.concatenate([top, random])

    positive_keep = choose(
        positive,
        int(config.positive_per_scene),
        np.abs(target),
    )
    hard_negative = (
        np.abs(frame.gradient.to_numpy(float))
        * (1.0 + np.abs(frame.z.to_numpy(float)))
    )
    negative_keep = choose(
        negative,
        int(config.negative_per_scene),
        hard_negative,
    )
    keep = np.concatenate([positive_keep, negative_keep])
    return frame.iloc[keep].sample(
        frac=1.0, random_state=int(config.seed) + scene_index
    ).reset_index(drop=True)


def _scene_feature_frame(
    record,
    scene_index,
    exp,
    model,
    detect,
    reference_mean,
    reference_std,
    config,
):
    examples = _cache_lookup(exp)
    example = examples[record["example_id"]]
    _clean, patched, _ = exp._images_for_example(example)
    image = _preprocess_pair(exp, patched, patched)[:1]
    (
        levels,
        combined,
        class_gradient,
        geometry_gradient,
        gradient_std,
        gradient_max_abs,
        gradient_sign_agreement,
    ) = _observable_gradient_families(
        model, detect, image, config
    )
    candidates = _blind_candidate_indices(
        levels,
        combined,
        reference_mean,
        reference_std,
        int(config.candidates_per_level),
        "hybrid",
    )
    teacher_indices, teacher_components = _teacher_arrays(record, levels)
    labels = _labels_on_candidates(
        candidates, teacher_indices, teacher_components
    )
    frames = []
    for level_index, (
        level,
        gradient,
        class_value,
        geometry_value,
        std_value,
        max_value,
        agreement_value,
        indices,
        target,
    ) in enumerate(zip(
        levels,
        combined,
        class_gradient,
        geometry_gradient,
        gradient_std,
        gradient_max_abs,
        gradient_sign_agreement,
        candidates,
        labels,
        strict=True,
    )):
        arrays = _coordinate_arrays(
            level[0],
            indices,
            reference_mean[level_index],
            reference_std[level_index],
            gradient[0],
        )
        arrays["level"].fill(
            level_index / max(len(levels) - 1, 1)
        )
        class_selected = _normalize(_selected_values(class_value, indices))
        geometry_selected = _normalize(
            _selected_values(geometry_value, indices)
        )
        combined_selected = arrays["gradient"]
        arrays.update({
            "class_gradient": class_selected.astype(np.float32),
            "geometry_gradient": geometry_selected.astype(np.float32),
            "z_class_gradient": (
                arrays["z"] * class_selected
            ).astype(np.float32),
            "z_geometry_gradient": (
                arrays["z"] * geometry_selected
            ).astype(np.float32),
            "gradient_std": _normalize(
                _selected_values(std_value, indices)
            ).astype(np.float32),
            "gradient_max_abs": _normalize(
                _selected_values(max_value, indices)
            ).astype(np.float32),
            "gradient_sign_agreement": _selected_values(
                agreement_value, indices
            ).astype(np.float32),
            "gradient_abs_rank": _rank01(
                np.abs(combined_selected)
            ),
            "z_abs_rank": _rank01(np.abs(arrays["z"])),
            "teacher_component": target.astype(np.float32),
            "feature_level": np.full(
                len(indices), level_index, dtype=np.int8
            ),
        })
        frames.append(pd.DataFrame(arrays))
    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, "analysis_group", record["analysis_group"])
    frame.insert(0, "example_id", record["example_id"])
    return _sample_rows(frame, config, scene_index)


def _subsample(frame, maximum, seed):
    if len(frame) <= int(maximum):
        return frame
    return frame.sample(n=int(maximum), random_state=int(seed))


def _weights(target):
    target = np.asarray(target, dtype=np.float64)
    nonzero = np.abs(target) > 1e-8
    scale = max(
        float(np.std(target[nonzero] if nonzero.any() else target)),
        1e-8,
    )
    return 1.0 + 8.0 * nonzero + 2.0 * np.minimum(
        np.abs(target) / scale, 5.0
    )


def _probe_metrics(target, prediction):
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score

    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    weight = _weights(target)
    denominator = max(
        float(np.average(np.abs(target), weights=weight)), 1e-8
    )
    nonzero = np.abs(target) > 1e-8
    support_ap = average_precision_score(
        nonzero.astype(int), np.abs(prediction)
    )
    if nonzero.sum() >= 3:
        sign_accuracy = float(
            (
                np.sign(prediction[nonzero])
                == np.sign(target[nonzero])
            ).mean()
        )
        magnitude_spearman = float(
            spearmanr(
                np.abs(target[nonzero]),
                np.abs(prediction[nonzero]),
            ).statistic
        )
    else:
        sign_accuracy = np.nan
        magnitude_spearman = np.nan
    return {
        "normalized_weighted_mae": float(
            np.average(np.abs(target - prediction), weights=weight)
            / denominator
        ),
        "support_average_precision": float(support_ap),
        "sign_accuracy": sign_accuracy,
        "magnitude_spearman": magnitude_spearman,
    }


def _cv_probe(frame, features, config):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    groups = frame.example_id.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=int(config.cv_folds))
    fold_rows = []
    for fold, (train_index, eval_index) in enumerate(
        splitter.split(frame, groups=groups)
    ):
        train = _subsample(
            frame.iloc[train_index],
            config.probe_max_train_rows,
            config.seed + fold,
        )
        evaluation = _subsample(
            frame.iloc[eval_index],
            config.probe_max_eval_rows,
            config.seed + 100 + fold,
        )
        x_train = train[list(features)].to_numpy(np.float32)
        y_train = train.teacher_component.to_numpy(np.float32)
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.08,
            max_iter=int(config.probe_max_iter),
            max_leaf_nodes=int(config.probe_max_leaf_nodes),
            l2_regularization=1.0,
            random_state=int(config.seed) + fold,
        )
        model.fit(x_train, y_train, sample_weight=_weights(y_train))
        prediction = model.predict(
            evaluation[list(features)].to_numpy(np.float32)
        )
        fold_rows.append({
            "fold": fold,
            **_probe_metrics(
                evaluation.teacher_component.to_numpy(float),
                prediction,
            ),
        })
    metrics = pd.DataFrame(fold_rows).mean(numeric_only=True).to_dict()
    return metrics, fold_rows


def _univariate_screen(frame, config):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.feature_selection import (
        mutual_info_classif,
        mutual_info_regression,
    )
    from sklearn.metrics import average_precision_score, roc_auc_score

    sample = _subsample(
        frame, config.univariate_max_rows, config.seed
    )
    target = sample.teacher_component.to_numpy(float)
    support = np.abs(target) > 1e-8

    def safe_correlation(function, left, right):
        if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
            return np.nan
        return float(function(left, right).statistic)

    rows = []
    for feature in ALL_FEATURES:
        values = sample[feature].to_numpy(float)
        candidates = (values, -values, np.abs(values))
        support_ap = max(
            average_precision_score(support.astype(int), current)
            for current in candidates
        )
        sign_auc = np.nan
        magnitude_spearman = np.nan
        magnitude_mi = np.nan
        if support.sum() >= 10:
            sign = (target[support] > 0).astype(int)
            if len(np.unique(sign)) == 2:
                auc = roc_auc_score(sign, values[support])
                sign_auc = max(float(auc), float(1.0 - auc))
            magnitude_spearman = safe_correlation(
                spearmanr,
                np.abs(values[support]),
                np.abs(target[support]),
            )
            magnitude_mi = float(
                mutual_info_regression(
                    values[support].reshape(-1, 1),
                    np.log1p(np.abs(target[support])),
                    random_state=int(config.seed),
                )[0]
            )
        rows.append({
            "feature": feature,
            "group": next(
                group for group, members in FEATURE_GROUPS.items()
                if feature in members
            ),
            "signed_pearson": safe_correlation(
                pearsonr, values, target
            ),
            "absolute_spearman": safe_correlation(
                spearmanr, np.abs(values), np.abs(target)
            ),
            "support_average_precision": float(support_ap),
            "support_mutual_information": float(
                mutual_info_classif(
                    values.reshape(-1, 1),
                    support.astype(int),
                    random_state=int(config.seed),
                )[0]
            ),
            "sign_auc": sign_auc,
            "magnitude_spearman_nonzero": magnitude_spearman,
            "magnitude_mutual_information": magnitude_mi,
        })
    return pd.DataFrame(rows)


def _scene_univariate(frame):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for (example_id, analysis_group), scene in frame.groupby(
        ["example_id", "analysis_group"], sort=False
    ):
        target = scene.teacher_component.to_numpy(float)
        support = np.abs(target) > 1e-8
        for feature in ALL_FEATURES:
            values = scene[feature].to_numpy(float)
            signed_pearson = (
                float(pearsonr(values, target).statistic)
                if np.std(values) > 1e-12 and np.std(target) > 1e-12
                else np.nan
            )
            absolute_spearman = (
                float(
                    spearmanr(
                        np.abs(values), np.abs(target)
                    ).statistic
                )
                if np.std(np.abs(values)) > 1e-12
                and np.std(np.abs(target)) > 1e-12
                else np.nan
            )
            support_ap = (
                max(
                    average_precision_score(
                        support.astype(int), current
                    )
                    for current in (
                        values, -values, np.abs(values)
                    )
                )
                if 0 < support.sum() < len(support)
                else np.nan
            )
            sign_auc = np.nan
            if support.sum() >= 4:
                sign = (target[support] > 0).astype(int)
                if len(np.unique(sign)) == 2:
                    auc = roc_auc_score(sign, values[support])
                    sign_auc = max(float(auc), float(1.0 - auc))
            rows.append({
                "example_id": example_id,
                "analysis_group": analysis_group,
                "feature": feature,
                "signed_pearson": signed_pearson,
                "absolute_spearman": absolute_spearman,
                "support_average_precision": support_ap,
                "sign_auc": sign_auc,
            })
    return pd.DataFrame(rows)


def _redundancy(frame, config):
    sample = _subsample(
        frame[list(ALL_FEATURES)],
        min(config.univariate_max_rows, 100_000),
        config.seed + 1,
    )
    return sample.corr(method="spearman")


def _feature_selection(frame, config):
    cache = {}
    records = []

    def evaluate(name, groups):
        features = tuple(
            feature
            for group in groups
            for feature in FEATURE_GROUPS[group]
        )
        key = tuple(sorted(features))
        if key not in cache:
            metrics, folds = _cv_probe(frame, features, config)
            cache[key] = metrics
            records.extend({
                "feature_set": name,
                "groups": ",".join(groups),
                "n_features": len(features),
                **row,
            } for row in folds)
        return cache[key]

    group_names = list(FEATURE_GROUPS)
    full_metrics = evaluate("full", group_names)
    drop_rows = []
    for group in group_names:
        kept = [name for name in group_names if name != group]
        metrics = evaluate(f"drop_{group}", kept)
        drop_rows.append({
            "dropped_group": group,
            **metrics,
            "mae_change_vs_full": (
                metrics["normalized_weighted_mae"]
                - full_metrics["normalized_weighted_mae"]
            ),
        })

    baselines = {
        "current_functional": (
            "activation", "identity", "position", "functional"
        ),
        "compact_no_xy": (
            "activation", "identity", "functional"
        ),
        "combined_legacy": (
            "activation", "identity", "position", "functional", "local"
        ),
    }
    baseline_rows = []
    for name, groups in baselines.items():
        baseline_rows.append({
            "feature_set": name,
            "groups": ",".join(groups),
            **evaluate(name, list(groups)),
        })

    selected = ["activation", "identity"]
    remaining = [
        group for group in group_names if group not in selected
    ]
    current = evaluate("greedy_core", selected)
    greedy_rows = [{
        "step": 0,
        "added_group": "core",
        "selected_groups": ",".join(selected),
        **current,
    }]
    step = 1
    while remaining:
        candidates = []
        for group in remaining:
            groups = [*selected, group]
            metrics = evaluate(
                f"greedy_{step}_{group}", groups
            )
            candidates.append((group, metrics))
        winner, winner_metrics = min(
            candidates,
            key=lambda item: item[1]["normalized_weighted_mae"],
        )
        relative = (
            current["normalized_weighted_mae"]
            - winner_metrics["normalized_weighted_mae"]
        ) / max(current["normalized_weighted_mae"], 1e-8)
        greedy_rows.append({
            "step": step,
            "added_group": winner,
            "selected_groups": ",".join([*selected, winner]),
            "relative_mae_improvement": relative,
            **winner_metrics,
        })
        if relative < float(config.min_greedy_relative_improvement):
            break
        selected.append(winner)
        remaining.remove(winner)
        current = winner_metrics
        step += 1
    return (
        pd.DataFrame(records),
        pd.DataFrame(drop_rows),
        pd.DataFrame(baseline_rows),
        pd.DataFrame(greedy_rows),
        selected,
    )


def run_student_feature_eda(config: StudentFeatureEdaConfig) -> Path:
    started = time.time()
    pool_run = Path(config.pool_run).resolve()
    pool_meta = json.loads(
        (pool_run / "run.json").read_text(encoding="utf-8")
    )
    if pool_meta["status"] != "complete":
        raise RuntimeError("A complete balanced pool is required.")
    student_rows = pd.read_csv(pool_run / "student_train_split.csv")
    if config.max_scenes is not None:
        parts = []
        groups = sorted(student_rows.analysis_group.unique())
        per_group = max(1, int(config.max_scenes) // len(groups))
        for group in groups:
            parts.append(
                student_rows[
                    student_rows.analysis_group.eq(group)
                ].head(per_group)
            )
        student_rows = pd.concat(parts, ignore_index=True).head(
            int(config.max_scenes)
        )
    payload = {
        **asdict(config),
        "student_ids": student_rows.example_id.astype(str).tolist(),
        "feature_groups": FEATURE_GROUPS,
    }
    run_dir = (
        Path(config.output_dir)
        / f"student_feature_eda_{stable_hash(payload)}"
    )
    scene_dir = run_dir / "scene_tables"
    scene_dir.mkdir(parents=True, exist_ok=True)
    student_rows.to_csv(run_dir / "student_train_split.csv", index=False)

    print("[1/4] Loading model and training-only teacher records...", flush=True)
    exp, _cache_path = load_experiment(
        prefer_device=config.device,
        require_device=config.require_device,
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    teacher_dir = Path(pool_meta["teacher_cache_dir"])
    manifest = json.loads(
        (teacher_dir / "manifest.json").read_text(encoding="utf-8")
    )
    records = _record_lookup(exp, student_rows, manifest)
    row_lookup = {
        str(row.example_id): row
        for row in student_rows.itertuples(index=False)
    }
    for record in records:
        record["row"] = row_lookup[record["example_id"]]
    state_path = run_dir / "reference_statistics.npz"
    if state_path.exists():
        with np.load(state_path, allow_pickle=False) as state:
            count = int(state["level_count"])
            reference_mean = [
                state[f"mean_{level}"] for level in range(count)
            ]
            reference_std = [
                state[f"std_{level}"] for level in range(count)
            ]
    else:
        reference_mean, reference_std = _attacked_reference_statistics(
            exp, model, detect, records
        )
        state = {"level_count": np.asarray(len(reference_mean))}
        for level, (mean, std) in enumerate(
            zip(reference_mean, reference_std, strict=True)
        ):
            state[f"mean_{level}"] = mean
            state[f"std_{level}"] = std
        np.savez_compressed(state_path, **state)

    print("[2/4] Extracting/resuming per-scene feature tables...", flush=True)
    for scene_index, record in enumerate(
        tqdm(records, desc="student feature EDA", unit="scene")
    ):
        target = scene_dir / f"{record['example_id']}.parquet"
        if not target.exists():
            frame = _scene_feature_frame(
                record,
                scene_index,
                exp,
                model,
                detect,
                reference_mean,
                reference_std,
                config,
            )
            frame.to_parquet(target, index=False)
        release_accelerator_memory()
    table_path = run_dir / "coordinate_features.parquet"
    if table_path.exists():
        coordinate_table = pd.read_parquet(table_path)
    else:
        coordinate_table = pd.concat(
            [
                pd.read_parquet(
                    scene_dir / f"{record['example_id']}.parquet"
                )
                for record in records
            ],
            ignore_index=True,
        )
        coordinate_table.to_parquet(table_path, index=False)

    print("[3/4] Running univariate and redundancy EDA...", flush=True)
    univariate = _univariate_screen(coordinate_table, config)
    scene_univariate = _scene_univariate(coordinate_table)
    scene_summary = (
        scene_univariate.groupby("feature")
        .agg(
            scene_signed_pearson_median=("signed_pearson", "median"),
            scene_absolute_spearman_median=(
                "absolute_spearman", "median"
            ),
            scene_absolute_spearman_q25=(
                "absolute_spearman", lambda value: value.quantile(0.25)
            ),
            scene_absolute_spearman_q75=(
                "absolute_spearman", lambda value: value.quantile(0.75)
            ),
            scene_support_ap_median=(
                "support_average_precision", "median"
            ),
            scene_sign_auc_median=("sign_auc", "median"),
            scene_count=("example_id", "nunique"),
        )
        .reset_index()
    )
    univariate = univariate.merge(
        scene_summary, on="feature", how="left", validate="one_to_one"
    )
    scene_group_summary = (
        scene_univariate.groupby(["analysis_group", "feature"])
        .agg(
            scene_absolute_spearman_median=(
                "absolute_spearman", "median"
            ),
            scene_support_ap_median=(
                "support_average_precision", "median"
            ),
            scene_sign_auc_median=("sign_auc", "median"),
            scene_count=("example_id", "nunique"),
        )
        .reset_index()
    )
    redundancy = _redundancy(coordinate_table, config)
    univariate.to_csv(run_dir / "univariate_screen.csv", index=False)
    scene_univariate.to_csv(
        run_dir / "scene_univariate.csv", index=False
    )
    scene_group_summary.to_csv(
        run_dir / "scene_univariate_by_group.csv", index=False
    )
    redundancy.to_csv(run_dir / "spearman_redundancy.csv")

    print("[4/4] Running grouped drop-one and greedy probes...", flush=True)
    (
        cv_folds,
        drop_one,
        baselines,
        greedy,
        selected,
    ) = _feature_selection(coordinate_table, config)
    cv_folds.to_csv(run_dir / "cv_fold_metrics.csv", index=False)
    drop_one.to_csv(run_dir / "drop_one_groups.csv", index=False)
    baselines.to_csv(run_dir / "baseline_feature_sets.csv", index=False)
    greedy.to_csv(run_dir / "greedy_selection.csv", index=False)
    selected_features = [
        feature for group in selected for feature in FEATURE_GROUPS[group]
    ]
    recommendation = {
        "selected_groups": selected,
        "selected_features": selected_features,
        "selection_metric": "scene-grouped normalized weighted MAE",
        "position_shortcut_warning": (
            "Absolute x/y may encode a fixed patch-location prior; treat "
            "compact_no_xy as a required downstream control."
        ),
        "coordinate_rows": int(len(coordinate_table)),
        "scenes": int(coordinate_table.example_id.nunique()),
    }
    (run_dir / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2), encoding="utf-8"
    )
    audit = {
        "source_partition": "student_train_only",
        "student_scenes": int(len(student_rows)),
        "ranker_scenes_loaded": 0,
        "previous_holdout_scenes_loaded": 0,
        "fresh_final_scenes_loaded": 0,
        "teacher_records_loaded": int(len(records)),
        "teacher_record_ids_equal_student_ids": (
            {record["example_id"] for record in records}
            == set(student_rows.example_id.astype(str))
        ),
        "cv_split_unit": "example_id",
        "inference_unavailable_fields_in_features": [],
    }
    (run_dir / "leakage_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "elapsed_seconds": time.time() - started,
                "config": asdict(config),
                "feature_groups": FEATURE_GROUPS,
                "recommendation": recommendation,
                "audit": audit,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Complete: {run_dir}", flush=True)
    return run_dir


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract a training-only student coordinate table once, then run "
            "scene-grouped feature EDA and group-wise selection."
        )
    )
    parser.add_argument("--pool-run", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = StudentFeatureEdaConfig(
        pool_run=args.pool_run,
        output_dir=args.output_dir,
        device=args.device,
        require_device=args.require_device,
    )
    if args.smoke:
        config.max_scenes = 8
        config.positive_per_scene = 32
        config.negative_per_scene = 32
        config.candidates_per_level = 200
        config.cv_folds = 2
        config.probe_max_train_rows = 400
        config.probe_max_eval_rows = 400
        config.probe_max_iter = 5
        config.univariate_max_rows = 400
    print(run_student_feature_eda(config))


if __name__ == "__main__":
    main()
