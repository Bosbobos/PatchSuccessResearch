from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .autonomous_negative_repair import _clusters_from_frame, _target_overlap
from .candidate_reserve import _cache_lookup
from .common import REPO_ROOT, load_experiment, release_accelerator_memory, stable_hash
from .component_student import _capture_with_grad, _reset_detect_inference_cache
from .component_targeted_patch import _record_lookup
from .improved_component_defense import _proposal_frame
from .large_component_student import (
    _blind_candidate_indices,
    _blind_cluster_config,
    _direct_maps,
    _labels_on_candidates,
    _support_metrics,
    _teacher_arrays,
)
from .localization_mechanism_sweep import (
    _evaluate_conditions,
    _load_base_config,
)
from .mechanism_aware_patch import dynamic_score_geometry_loss
from .mechanism_followup import _head_branches


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "CandidateRoutingAndAttackPath" / "cluster_ranker_outputs"
)
RANK_METRICS = (
    "max_score",
    "max_proposal_score",
    "noisy_or",
    "reserve_tension",
    "object_suppression_tension",
)
FEATURE_COLUMNS = (
    "n_members",
    "max_score",
    "max_proposal_score",
    "noisy_or",
    "reserve_tension",
    "object_suppression_tension",
    "score_mean",
    "score_std",
    "score_q25",
    "score_q50",
    "score_q75",
    "proposal_mean",
    "proposal_std",
    "box_area_mean",
    "box_area_std",
    "box_aspect_mean",
    "box_aspect_std",
    "center_x_mean",
    "center_y_mean",
    "center_x_std",
    "center_y_std",
    "level_mean",
    "level_std",
    "level_unique",
    "rank_max_score",
    "rank_max_proposal_score",
    "rank_noisy_or",
    "rank_reserve_tension",
    "rank_object_suppression_tension",
)


@dataclass(slots=True)
class LearnedClusterRankerConfig:
    base_run: str
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "mps"
    require_device: bool = False
    top_clusters: int = 5
    candidate_budget: int = 4000
    pool_per_strategy: int = 12
    pool_oracle_energy: int = 8
    pool_random: int = 16
    cluster_label_batch_size: int = 12
    ranker_max_iter: int = 250
    ranker_max_leaf_nodes: int = 31
    max_train_scenes: int | None = None
    max_test_scenes: int | None = None
    seed: int = 1409
    method_version: int = 1


def _clusters_for_levels(detect, levels, base_config):
    import torch

    with torch.no_grad():
        _box, _cls, raw = _head_branches(detect, levels)
        frame = _proposal_frame(
            detect,
            raw,
            class_id=0,
            policy="hybrid",
            config=_blind_cluster_config(base_config),
        )
    return _clusters_from_frame(frame, _blind_cluster_config(base_config))


def _cluster_features(clusters, image_size: int) -> pd.DataFrame:
    records = []
    scale2 = float(max(image_size * image_size, 1))
    for cluster_index, cluster in enumerate(clusters):
        selection = cluster["selection"]
        scores = selection.score.to_numpy(float)
        proposal = (
            selection.proposal_score.to_numpy(float)
            if "proposal_score" in selection
            else scores
        )
        boxes = selection[["x1", "y1", "x2", "y2"]].to_numpy(float)
        widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1e-6)
        heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1e-6)
        centers_x = (boxes[:, 0] + boxes[:, 2]) / (2.0 * image_size)
        centers_y = (boxes[:, 1] + boxes[:, 3]) / (2.0 * image_size)
        levels = selection.level_index.to_numpy(float)
        records.append({
            "cluster_index": cluster_index,
            "n_members": int(len(selection)),
            **{key: float(cluster[key]) for key in RANK_METRICS},
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_q25": float(np.quantile(scores, 0.25)),
            "score_q50": float(np.quantile(scores, 0.50)),
            "score_q75": float(np.quantile(scores, 0.75)),
            "proposal_mean": float(proposal.mean()),
            "proposal_std": float(proposal.std()),
            "box_area_mean": float((widths * heights).mean() / scale2),
            "box_area_std": float((widths * heights).std() / scale2),
            "box_aspect_mean": float((widths / heights).mean()),
            "box_aspect_std": float((widths / heights).std()),
            "center_x_mean": float(centers_x.mean()),
            "center_y_mean": float(centers_y.mean()),
            "center_x_std": float(centers_x.std()),
            "center_y_std": float(centers_y.std()),
            "level_mean": float(levels.mean() / 2.0),
            "level_std": float(levels.std() / 2.0),
            "level_unique": int(np.unique(levels).size),
        })
    frame = pd.DataFrame(records)
    for metric in RANK_METRICS:
        frame[f"rank_{metric}"] = (
            frame[metric].rank(method="average", ascending=False)
            / max(len(frame), 1)
        )
    return frame


def _teacher_spatial_energy(teacher_indices, teacher_components, levels):
    energy = []
    total = 0.0
    for indices, component, level in zip(
        teacher_indices, teacher_components, levels, strict=True
    ):
        height, width = int(level.shape[-2]), int(level.shape[-1])
        current = np.zeros(height * width, dtype=np.float64)
        spatial = indices % (height * width)
        np.add.at(current, spatial, np.square(component, dtype=np.float64))
        energy.append(current.reshape(height, width))
        total += float(current.sum())
    return energy, max(total, 1e-12)


def _cluster_mask(cluster, levels, radius: int = 2):
    masks = [
        np.zeros((int(level.shape[-2]), int(level.shape[-1])), dtype=bool)
        for level in levels
    ]
    for item in cluster["selection"].itertuples(index=False):
        level = int(item.level_index)
        y, x = int(item.y_index), int(item.x_index)
        height, width = masks[level].shape
        masks[level][
            max(0, y - radius):min(height, y + radius + 1),
            max(0, x - radius):min(width, x + radius + 1),
        ] = True
    return masks


def _cluster_teacher_support(
    cluster,
    teacher_indices,
    teacher_components,
    levels,
):
    masks = _cluster_mask(cluster, levels)
    selected_indices = []
    selected_values = []
    for indices, component, level, mask in zip(
        teacher_indices, teacher_components, levels, masks, strict=True
    ):
        height, width = int(level.shape[-2]), int(level.shape[-1])
        spatial = indices % (height * width)
        keep = mask.reshape(-1)[spatial]
        selected_indices.append(indices[keep])
        selected_values.append(component[keep])
    return selected_indices, selected_values


def _cluster_energy_fraction(cluster, energy_maps, total_energy, levels):
    masks = _cluster_mask(cluster, levels)
    return float(
        sum(float(energy[mask].sum()) for energy, mask in zip(
            energy_maps, masks, strict=True
        ))
        / total_energy
    )


def _training_pool(features, energy, config, rng):
    chosen = set()
    for metric in RANK_METRICS:
        chosen.update(
            features.nlargest(
                min(int(config.pool_per_strategy), len(features)), metric
            ).cluster_index.astype(int)
        )
    energy_order = np.argsort(-np.asarray(energy))
    chosen.update(
        int(value)
        for value in energy_order[: int(config.pool_oracle_energy)]
    )
    remaining = np.asarray(
        sorted(set(range(len(features))) - chosen), dtype=np.int64
    )
    if len(remaining):
        count = min(int(config.pool_random), len(remaining))
        chosen.update(
            int(value)
            for value in rng.choice(remaining, size=count, replace=False)
        )
    return sorted(chosen)


def _label_training_clusters(
    detect,
    levels,
    clusters,
    pool,
    teacher_indices,
    teacher_components,
    row,
    base_config,
    batch_size,
):
    conditions = {}
    metadata = {}
    for cluster_index in pool:
        indices, values = _cluster_teacher_support(
            clusters[cluster_index],
            teacher_indices,
            teacher_components,
            levels,
        )
        correction, _ = _direct_maps(values, indices, levels)
        name = f"cluster_{cluster_index}"
        conditions[name] = [
            level - delta
            for level, delta in zip(levels, correction, strict=True)
        ]
        metrics = _support_metrics(
            indices, teacher_indices, teacher_components
        )
        metadata[name] = {"cluster_index": cluster_index, **metrics}
    evaluated = _evaluate_conditions(
        detect, levels, conditions, row, base_config, batch_size
    )
    baseline = evaluated["observed"]
    labels = []
    for name, item in metadata.items():
        corrected = evaluated[name]
        recovered = int(
            not baseline["target_detected"] and corrected["target_detected"]
        )
        confidence_gain = (
            corrected["post_target_conf"] - baseline["post_target_conf"]
        )
        labels.append({
            **item,
            "baseline_target_detected": baseline["target_detected"],
            "corrected_target_detected": corrected["target_detected"],
            "oracle_confidence_gain": confidence_gain,
            "oracle_recovered": recovered,
            "rank_target": float(
                confidence_gain
                + 0.75 * recovered
                + 0.15 * np.sqrt(max(item["support_energy_recall"], 0.0))
            ),
        })
    return labels


def _representative(cluster):
    selection = cluster["selection"]
    score_column = (
        "proposal_score" if "proposal_score" in selection else "score"
    )
    return selection.iloc[int(selection[score_column].argmax())]


def _average_gradient_for_clusters(decoded, levels, clusters):
    import torch

    losses = []
    for cluster in clusters:
        representative = _representative(cluster)
        target_box = torch.as_tensor(
            [[
                float(representative.x1),
                float(representative.y1),
                float(representative.x2),
                float(representative.y2),
            ]],
            device=levels[0].device,
            dtype=torch.float32,
        )
        class_id = torch.zeros(
            1, device=levels[0].device, dtype=torch.long
        )
        loss, _ = dynamic_score_geometry_loss(
            decoded,
            target_box,
            class_id,
            match_iou=0.50,
            iou_temperature=0.07,
            iou_weight=4.0,
            smoothmax_temperature=0.35,
        )
        losses.append(loss)
    objective = torch.stack(losses).mean()
    return [
        value.detach()
        for value in torch.autograd.grad(
            objective, levels, retain_graph=True
        )
    ]


def _fit_ranker(training_rows, config):
    from sklearn.ensemble import HistGradientBoostingRegressor

    frame = pd.DataFrame(training_rows)
    x = frame[list(FEATURE_COLUMNS)].to_numpy(np.float32)
    y = frame.rank_target.to_numpy(np.float32)
    weights = (
        1.0
        + 3.0 * frame.baseline_target_detected.eq(0).to_numpy(float)
        + 8.0 * frame.oracle_recovered.to_numpy(float)
        + 4.0 * frame.support_energy_recall.to_numpy(float)
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=int(config.ranker_max_iter),
        max_leaf_nodes=int(config.ranker_max_leaf_nodes),
        l2_regularization=2.0,
        random_state=int(config.seed),
    )
    model.fit(x, y, sample_weight=weights)
    return model, frame


def _selected_cluster_sets(clusters, features, ranker, energy, record, top_k):
    heuristic = sorted(
        range(len(clusters)),
        key=lambda index: (
            clusters[index]["object_suppression_tension"],
            clusters[index]["reserve_tension"],
        ),
        reverse=True,
    )[:top_k]
    predictions = ranker.predict(
        features[list(FEATURE_COLUMNS)].to_numpy(np.float32)
    )
    learned = np.argsort(-predictions)[:top_k].astype(int).tolist()
    oracle_energy = np.argsort(-np.asarray(energy))[:top_k].astype(int).tolist()
    target_overlap = [
        _target_overlap(cluster["selection"], record["row"])
        for cluster in clusters
    ]
    target_oracle = np.argsort(-np.asarray(target_overlap))[
        :top_k
    ].astype(int).tolist()
    return {
        "heuristic_top5": heuristic,
        "learned_top5": learned,
        "oracle_energy_top5": oracle_energy,
        "target_oracle_top5": target_oracle,
    }, predictions, target_overlap


def run_learned_cluster_ranker(
    config: LearnedClusterRankerConfig,
) -> Path:
    started = time.time()
    base_run = Path(config.base_run).resolve()
    metadata = json.loads((base_run / "run.json").read_text(encoding="utf-8"))
    base_config = _load_base_config(metadata, config)
    train_rows = pd.read_csv(base_run / "train_split.csv")
    test_rows = pd.read_csv(base_run / "test_split.csv")
    if config.max_train_scenes is not None:
        train_rows = train_rows.head(int(config.max_train_scenes)).copy()
    if config.max_test_scenes is not None:
        hidden = test_rows[
            test_rows.analysis_group.astype(str).str.startswith("hidden")
        ]
        test_rows = hidden.head(int(config.max_test_scenes)).copy()
    stats = np.load(base_run / "attacked_reference_statistics.npz")
    level_count = len([key for key in stats.files if key.startswith("mean_")])
    reference_mean = [stats[f"mean_{index}"] for index in range(level_count)]
    reference_std = [stats[f"std_{index}"] for index in range(level_count)]
    manifest = json.loads(
        (
            Path(metadata["teacher_cache_dir"]) / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    exp, _cache_path = load_experiment(
        prefer_device=config.device, require_device=config.require_device
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_records = _record_lookup(exp, train_rows, manifest)
    test_records = _record_lookup(exp, test_rows, manifest)
    all_rows = pd.concat([train_rows, test_rows], ignore_index=True)
    row_lookup = {
        str(row.example_id): row
        for row in all_rows.itertuples(index=False)
    }
    for record in [*train_records, *test_records]:
        record["row"] = row_lookup[record["example_id"]]
    examples = _cache_lookup(exp)
    rng = np.random.default_rng(int(config.seed))
    training_labels = []
    for record in tqdm(
        train_records, desc="cluster ranker labels", unit="scene"
    ):
        example = examples[record["example_id"]]
        _clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, patched_image, patched_image)
        levels = _capture_detect_inputs(model, detect, pair[:1])
        clusters = _clusters_for_levels(detect, levels, base_config)
        features = _cluster_features(
            clusters, int(pair.shape[-1])
        )
        teacher_indices, teacher_components = _teacher_arrays(record, levels)
        energy_maps, total_energy = _teacher_spatial_energy(
            teacher_indices, teacher_components, levels
        )
        energy = [
            _cluster_energy_fraction(
                cluster, energy_maps, total_energy, levels
            )
            for cluster in clusters
        ]
        pool = _training_pool(features, energy, config, rng)
        labels = _label_training_clusters(
            detect,
            levels,
            clusters,
            pool,
            teacher_indices,
            teacher_components,
            record["row"],
            base_config,
            config.cluster_label_batch_size,
        )
        feature_lookup = features.set_index("cluster_index")
        for item in labels:
            cluster_index = int(item["cluster_index"])
            training_labels.append({
                "example_id": record["example_id"],
                "analysis_group": record["analysis_group"],
                **feature_lookup.loc[cluster_index].to_dict(),
                "teacher_energy_fraction": energy[cluster_index],
                "target_overlap": _target_overlap(
                    clusters[cluster_index]["selection"], record["row"]
                ),
                **item,
            })
        _reset_detect_inference_cache(detect)
        release_accelerator_memory()
    ranker, training_frame = _fit_ranker(training_labels, config)
    evaluation_rows = []
    ranking_rows = []
    for record in tqdm(
        test_records, desc="cluster ranker evaluation", unit="scene"
    ):
        example = examples[record["example_id"]]
        _clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, patched_image, patched_image)
        _reset_detect_inference_cache(detect)
        image = pair[:1].detach().requires_grad_(True)
        decoded, levels = _capture_with_grad(model, detect, image)
        clusters = _clusters_for_levels(detect, levels, base_config)
        features = _cluster_features(clusters, int(pair.shape[-1]))
        teacher_indices, teacher_components = _teacher_arrays(record, levels)
        energy_maps, total_energy = _teacher_spatial_energy(
            teacher_indices, teacher_components, levels
        )
        energy = [
            _cluster_energy_fraction(
                cluster, energy_maps, total_energy, levels
            )
            for cluster in clusters
        ]
        selected, predictions, target_overlap = _selected_cluster_sets(
            clusters,
            features,
            ranker,
            energy,
            record,
            int(config.top_clusters),
        )
        conditions = {}
        condition_metadata = {}
        for condition, indices in selected.items():
            chosen = [clusters[index] for index in indices]
            gradients = _average_gradient_for_clusters(
                decoded, levels, chosen
            )
            candidates = _blind_candidate_indices(
                levels,
                gradients,
                reference_mean,
                reference_std,
                int(config.candidate_budget),
                "hybrid",
            )
            values = _labels_on_candidates(
                candidates, teacher_indices, teacher_components
            )
            correction, _ = _direct_maps(values, candidates, levels)
            conditions[condition] = [
                level - delta
                for level, delta in zip(levels, correction, strict=True)
            ]
            condition_metadata[condition] = {
                **_support_metrics(
                    candidates, teacher_indices, teacher_components
                ),
                "selected_max_target_overlap": float(
                    max(target_overlap[index] for index in indices)
                ),
                "selected_teacher_energy": float(
                    max(energy[index] for index in indices)
                ),
            }
            for rank, cluster_index in enumerate(indices, start=1):
                ranking_rows.append({
                    "example_id": record["example_id"],
                    "analysis_group": record["analysis_group"],
                    "condition": condition,
                    "rank": rank,
                    "cluster_index": cluster_index,
                    "predicted_rank_target": float(
                        predictions[cluster_index]
                    ),
                    "teacher_energy_fraction": energy[cluster_index],
                    "target_overlap": target_overlap[cluster_index],
                })
        exact_maps, _ = _direct_maps(
            teacher_components, teacher_indices, levels
        )
        conditions["exact_teacher"] = [
            level - delta
            for level, delta in zip(levels, exact_maps, strict=True)
        ]
        condition_metadata["exact_teacher"] = {
            "support_recall": 1.0,
            "support_energy_recall": 1.0,
            "selected_max_target_overlap": 1.0,
            "selected_teacher_energy": 1.0,
        }
        evaluated = _evaluate_conditions(
            detect,
            levels,
            conditions,
            record["row"],
            base_config,
            config.cluster_label_batch_size,
        )
        baseline = evaluated["observed"]
        for condition, item in condition_metadata.items():
            corrected = evaluated[condition]
            evaluation_rows.append({
                "example_id": record["example_id"],
                "analysis_group": record["analysis_group"],
                "condition": condition,
                "baseline_target_detected": baseline["target_detected"],
                "corrected_target_detected": corrected["target_detected"],
                "baseline_target_conf": baseline["post_target_conf"],
                "corrected_target_conf": corrected["post_target_conf"],
                **item,
            })
        _reset_detect_inference_cache(detect)
        release_accelerator_memory()
    evaluation = pd.DataFrame(evaluation_rows)
    summary_rows = []
    group_rows = []
    for condition, current in evaluation.groupby("condition"):
        hidden = current.baseline_target_detected.eq(0)
        summary_rows.append({
            "condition": condition,
            "n": int(len(current)),
            "baseline_target_rate": float(
                current.baseline_target_detected.mean()
            ),
            "corrected_target_rate": float(
                current.corrected_target_detected.mean()
            ),
            "hidden_n": int(hidden.sum()),
            "hidden_recovered_n": int(
                (hidden & current.corrected_target_detected.eq(1)).sum()
            ),
            "baseline_lost_n": int(
                (
                    current.baseline_target_detected.eq(1)
                    & current.corrected_target_detected.eq(0)
                ).sum()
            ),
            "confidence_gain": float(
                (
                    current.corrected_target_conf
                    - current.baseline_target_conf
                ).mean()
            ),
            "support_energy_recall": float(
                current.support_energy_recall.mean()
            ),
            "selected_target_overlap": float(
                current.selected_max_target_overlap.mean()
            ),
            "selected_teacher_energy": float(
                current.selected_teacher_energy.mean()
            ),
        })
        for analysis_group, group in current.groupby("analysis_group"):
            group_hidden = group.baseline_target_detected.eq(0)
            group_rows.append({
                "condition": condition,
                "analysis_group": analysis_group,
                "n": int(len(group)),
                "corrected_target_rate": float(
                    group.corrected_target_detected.mean()
                ),
                "hidden_recovered_n": int(
                    (
                        group_hidden
                        & group.corrected_target_detected.eq(1)
                    ).sum()
                ),
            })
    summary = pd.DataFrame(summary_rows)
    group_summary = pd.DataFrame(group_rows)
    payload = {
        **asdict(config),
        "base_run": str(base_run),
        "train_ids": train_rows.example_id.astype(str).tolist(),
        "test_ids": test_rows.example_id.astype(str).tolist(),
    }
    run_dir = (
        Path(config.output_dir)
        / f"cluster_ranker_{stable_hash(payload)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    training_frame.to_csv(run_dir / "ranker_training_rows.csv", index=False)
    evaluation.to_csv(run_dir / "ranker_evaluation_rows.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(
        run_dir / "selected_cluster_rows.csv", index=False
    )
    summary.to_csv(run_dir / "ranker_summary.csv", index=False)
    group_summary.to_csv(run_dir / "ranker_group_summary.csv", index=False)
    joblib.dump(ranker, run_dir / "cluster_ranker.joblib")
    best = summary[
        ~summary.condition.eq("exact_teacher")
    ].sort_values("hidden_recovered_n", ascending=False).iloc[0]
    heuristic = summary[
        summary.condition.eq("heuristic_top5")
    ].iloc[0]
    (run_dir / "recommendation.md").write_text(
        "# Learned cluster ranker\n\n"
        f"- heuristic top-5: {int(heuristic.hidden_recovered_n)}/"
        f"{int(heuristic.hidden_n)} recovered\n"
        f"- best: {best.condition}, {int(best.hidden_recovered_n)}/"
        f"{int(best.hidden_n)} recovered\n"
        f"- gain: {int(best.hidden_recovered_n - heuristic.hidden_recovered_n)}\n",
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "elapsed_seconds": time.time() - started,
                "base_run": str(base_run),
                "train_scenes": len(train_records),
                "test_scenes": len(test_records),
                "feature_columns": FEATURE_COLUMNS,
                "config": asdict(config),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_dir


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train an attacked-endpoint cluster ranker from offline oracle "
            "repair-gain labels and evaluate its localization ceiling."
        )
    )
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--top-clusters", type=int, default=5)
    parser.add_argument("--candidate-budget", type=int, default=4000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = LearnedClusterRankerConfig(
        base_run=args.base_run,
        output_dir=args.output_dir,
        device=args.device,
        require_device=args.require_device,
        top_clusters=args.top_clusters,
        candidate_budget=args.candidate_budget,
    )
    if args.smoke:
        config.max_train_scenes = 4
        config.max_test_scenes = 2
        config.pool_per_strategy = 2
        config.pool_oracle_energy = 2
        config.pool_random = 2
        config.cluster_label_batch_size = 4
        config.ranker_max_iter = 10
        config.top_clusters = 2
        config.candidate_budget = 300
    print(run_learned_cluster_ranker(config))


if __name__ == "__main__":
    main()
