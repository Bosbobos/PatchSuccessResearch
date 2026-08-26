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
from .common import REPO_ROOT, load_experiment, release_accelerator_memory, stable_hash
from .component_student import _capture_with_grad, _reset_detect_inference_cache
from .component_targeted_patch import _record_lookup
from .large_component_student import (
    _blind_candidate_indices,
    _blind_targets,
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


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "CandidateRoutingAndAttackPath" / "localization_ceiling_outputs"
)


@dataclass(slots=True)
class LocalizationCeilingConfig:
    base_run: str
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "mps"
    require_device: bool = False
    candidate_budget: int = 4000
    averaged_hypotheses: int = 5
    separate_hypotheses: tuple[int, ...] = (5, 10)
    geometry_fraction: float = 0.30
    condition_batch_size: int = 6
    max_test_scenes: int | None = None
    method_version: int = 1


def _normalized_max(gradient_sets):
    import torch

    output = []
    for level_index in range(len(gradient_sets[0])):
        normalized = []
        for gradients in gradient_sets:
            value = gradients[level_index].float()
            value = value / torch.sqrt(value.square().mean()).clamp_min(1e-8)
            normalized.append(value.abs())
        output.append(torch.stack(normalized).amax(dim=0))
    return output


def _mean_gradients(gradient_sets):
    import torch

    return [
        torch.stack([item[level] for item in gradient_sets]).mean(dim=0)
        for level in range(len(gradient_sets[0]))
    ]


def _multi_hypothesis_gradients(model, detect, image, base_config, top_k: int):
    """One detector forward followed by independent head-level VJPs."""

    import torch

    _reset_detect_inference_cache(detect)
    image = image.detach().requires_grad_(True)
    decoded, levels = _capture_with_grad(model, detect, image)
    local_config = type(base_config)(**asdict(base_config))
    local_config.blind_top_clusters = int(top_k)
    targets, cluster_count = _blind_targets(detect, levels, local_config)
    joint_gradients = []
    geometry_gradients = []
    image_scale = float(max(int(image.shape[-2]), int(image.shape[-1]), 1))
    for target in targets:
        box = torch.as_tensor(
            [target["box"]], device=image.device, dtype=torch.float32
        )
        class_id = torch.as_tensor(
            [target["class_id"]], device=image.device, dtype=torch.long
        )
        joint, _ = dynamic_score_geometry_loss(
            decoded,
            box,
            class_id,
            match_iou=0.50,
            iou_temperature=0.07,
            iou_weight=4.0,
            smoothmax_temperature=0.35,
        )
        joint_gradients.append(
            tuple(
                value.detach()
                for value in torch.autograd.grad(
                    joint, levels, retain_graph=True
                )
            )
        )
        flat = int(target["flat_index"])
        xywh = decoded[0, :4, flat]
        for geometry_scalar in (
            (xywh[0] + xywh[1]) / image_scale,
            (xywh[2] + xywh[3]) / image_scale,
        ):
            geometry_gradients.append(
                tuple(
                    value.detach()
                    for value in torch.autograd.grad(
                        geometry_scalar, levels, retain_graph=True
                    )
                )
            )
    return (
        [item.detach() for item in levels],
        joint_gradients,
        geometry_gradients,
        {
            "available_cluster_count": int(cluster_count),
            "used_hypothesis_count": int(len(targets)),
        },
    )


def _ordered_budget_union(primary, secondary, fallback, budget: int):
    output = []
    for primary_level, secondary_level, fallback_level in zip(
        primary, secondary, fallback, strict=True
    ):
        ordered = []
        seen = set()
        for index in np.concatenate(
            (primary_level, secondary_level, fallback_level)
        ):
            value = int(index)
            if value not in seen:
                ordered.append(value)
                seen.add(value)
            if len(ordered) >= int(budget):
                break
        output.append(np.asarray(ordered, dtype=np.int64))
    return output


def _candidate_conditions(
    levels,
    joint_gradients,
    geometry_gradients,
    reference_mean,
    reference_std,
    config,
):
    conditions = {}
    average_n = min(int(config.averaged_hypotheses), len(joint_gradients))
    average_gradient = _mean_gradients(joint_gradients[:average_n])
    conditions[f"averaged_top{average_n}"] = _blind_candidate_indices(
        levels,
        average_gradient,
        reference_mean,
        reference_std,
        int(config.candidate_budget),
        "hybrid",
    )
    for count in config.separate_hypotheses:
        count = min(int(count), len(joint_gradients))
        max_gradient = _normalized_max(joint_gradients[:count])
        conditions[f"separate_top{count}"] = _blind_candidate_indices(
            levels,
            max_gradient,
            reference_mean,
            reference_std,
            int(config.candidate_budget),
            "hybrid",
        )
    count = min(max(config.separate_hypotheses), len(joint_gradients))
    joint_max = _normalized_max(joint_gradients[:count])
    geometry_count = min(2 * count, len(geometry_gradients))
    geometry_max = _normalized_max(geometry_gradients[:geometry_count])
    geometry_budget = max(
        1, int(round(config.geometry_fraction * config.candidate_budget))
    )
    joint_budget = max(1, int(config.candidate_budget) - geometry_budget)
    joint_candidates = _blind_candidate_indices(
        levels,
        joint_max,
        reference_mean,
        reference_std,
        joint_budget,
        "hybrid",
    )
    geometry_candidates = _blind_candidate_indices(
        levels,
        geometry_max,
        reference_mean,
        reference_std,
        geometry_budget,
        "hybrid",
    )
    fallback = _blind_candidate_indices(
        levels,
        joint_max,
        reference_mean,
        reference_std,
        int(config.candidate_budget),
        "hybrid",
    )
    conditions[f"separate_top{count}_score_geometry"] = _ordered_budget_union(
        joint_candidates,
        geometry_candidates,
        fallback,
        int(config.candidate_budget),
    )
    return conditions


def _summaries(rows: pd.DataFrame):
    summary = []
    group_summary = []
    for condition, current in rows.groupby("condition"):
        hidden = current.baseline_target_detected.eq(0)
        base = {
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
            "support_recall": float(current.support_recall.mean()),
            "support_energy_recall": float(
                current.support_energy_recall.mean()
            ),
        }
        summary.append(base)
        for analysis_group, group in current.groupby("analysis_group"):
            group_hidden = group.baseline_target_detected.eq(0)
            group_summary.append({
                "condition": condition,
                "analysis_group": analysis_group,
                "n": int(len(group)),
                "baseline_target_rate": float(
                    group.baseline_target_detected.mean()
                ),
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
    return pd.DataFrame(summary), pd.DataFrame(group_summary)


def _recommend(summary: pd.DataFrame) -> str:
    candidates = summary[~summary.condition.eq("exact_teacher")].sort_values(
        ["hidden_recovered_n", "support_energy_recall"],
        ascending=False,
    )
    best = candidates.iloc[0]
    baseline = summary[
        summary.condition.str.startswith("averaged_top")
    ].iloc[0]
    exact = summary[summary.condition.eq("exact_teacher")].iloc[0]
    return (
        "# Localization ceiling sweep\n\n"
        "All supports use oracle values; this isolates localization.\n\n"
        f"- baseline {baseline.condition}: "
        f"{int(baseline.hidden_recovered_n)}/{int(baseline.hidden_n)} recovered\n"
        f"- best {best.condition}: "
        f"{int(best.hidden_recovered_n)}/{int(best.hidden_n)} recovered\n"
        f"- exact teacher: "
        f"{int(exact.hidden_recovered_n)}/{int(exact.hidden_n)} recovered\n"
        f"- improvement over averaged baseline: "
        f"{int(best.hidden_recovered_n - baseline.hidden_recovered_n)} cases\n"
    )


def run_localization_ceiling(config: LocalizationCeilingConfig) -> Path:
    started = time.time()
    base_run = Path(config.base_run).resolve()
    metadata = json.loads((base_run / "run.json").read_text(encoding="utf-8"))
    base_config = _load_base_config(metadata, config)
    test_rows = pd.read_csv(base_run / "test_split.csv")
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
    records = _record_lookup(exp, test_rows, manifest)
    row_lookup = {
        str(row.example_id): row
        for row in test_rows.itertuples(index=False)
    }
    for record in records:
        record["row"] = row_lookup[record["example_id"]]
    examples = _cache_lookup(exp)
    rows = []
    top_k = max(
        int(config.averaged_hypotheses),
        max(int(value) for value in config.separate_hypotheses),
    )
    for record in tqdm(records, desc="ceiling sweep", unit="scene"):
        example = examples[record["example_id"]]
        _clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, patched_image, patched_image)
        levels, joint_gradients, geometry_gradients, endpoint_metadata = (
            _multi_hypothesis_gradients(
                model, detect, pair[:1], base_config, top_k
            )
        )
        teacher_indices, teacher_components = _teacher_arrays(record, levels)
        candidate_sets = _candidate_conditions(
            levels,
            joint_gradients,
            geometry_gradients,
            reference_mean,
            reference_std,
            config,
        )
        conditions = {}
        condition_metadata = {}
        for condition, candidates in candidate_sets.items():
            values = _labels_on_candidates(
                candidates, teacher_indices, teacher_components
            )
            correction, _ = _direct_maps(values, candidates, levels)
            conditions[condition] = [
                level - delta
                for level, delta in zip(levels, correction, strict=True)
            ]
            condition_metadata[condition] = {
                **endpoint_metadata,
                **_support_metrics(
                    candidates, teacher_indices, teacher_components
                ),
            }
        exact_maps, _ = _direct_maps(
            teacher_components, teacher_indices, levels
        )
        conditions["exact_teacher"] = [
            level - delta
            for level, delta in zip(levels, exact_maps, strict=True)
        ]
        condition_metadata["exact_teacher"] = {
            **endpoint_metadata,
            "support_recall": 1.0,
            "support_energy_recall": 1.0,
        }
        evaluated = _evaluate_conditions(
            detect,
            levels,
            conditions,
            record["row"],
            base_config,
            config.condition_batch_size,
        )
        baseline = evaluated["observed"]
        for condition, item in condition_metadata.items():
            corrected = evaluated[condition]
            rows.append({
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
    rows = pd.DataFrame(rows)
    summary, group_summary = _summaries(rows)
    payload = {
        **asdict(config),
        "base_run": str(base_run),
        "test_ids": test_rows.example_id.astype(str).tolist(),
    }
    run_dir = (
        Path(config.output_dir)
        / f"localization_ceiling_{stable_hash(payload)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(run_dir / "ceiling_rows.csv", index=False)
    summary.to_csv(run_dir / "ceiling_summary.csv", index=False)
    group_summary.to_csv(run_dir / "ceiling_group_summary.csv", index=False)
    (run_dir / "recommendation.md").write_text(
        _recommend(summary), encoding="utf-8"
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "elapsed_seconds": time.time() - started,
                "base_run": str(base_run),
                "n_test_scenes": len(records),
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
            "Measure whether separate proposal gradients and geometry leverage "
            "raise the blind localization ceiling."
        )
    )
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument("--candidate-budget", type=int, default=4000)
    parser.add_argument("--condition-batch-size", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = LocalizationCeilingConfig(
        base_run=args.base_run,
        output_dir=args.output_dir,
        device=args.device,
        require_device=args.require_device,
        candidate_budget=args.candidate_budget,
        condition_batch_size=args.condition_batch_size,
    )
    if args.smoke:
        config.candidate_budget = 300
        config.averaged_hypotheses = 2
        config.separate_hypotheses = (2, 3)
        config.max_test_scenes = 2
        config.condition_batch_size = 4
    print(run_localization_ceiling(config))


if __name__ == "__main__":
    main()
