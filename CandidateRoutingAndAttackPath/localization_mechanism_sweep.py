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
from .candidate_reserve import _cache_lookup, _evaluate_batch
from .common import REPO_ROOT, load_experiment, release_accelerator_memory, stable_hash
from .component_student import _capture_with_grad, _reset_detect_inference_cache
from .component_targeted_patch import _record_lookup
from .large_component_student import (
    LargeComponentStudentConfig,
    _blind_candidate_indices,
    _blind_endpoint,
    _direct_maps,
    _labels_on_candidates,
    _support_metrics,
    _teacher_arrays,
)
from .mechanism_aware_patch import dynamic_score_geometry_loss
from .mechanism_followup import _head_branches


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "CandidateRoutingAndAttackPath" / "localization_sweep_outputs"
)


@dataclass(slots=True)
class LocalizationSweepConfig:
    base_run: str
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    device: str = "mps"
    require_device: bool = False
    candidate_budgets: tuple[int, ...] = (2000, 4000, 8000)
    blind_cluster_counts: tuple[int, ...] = (1, 3, 5)
    condition_batch_size: int = 8
    max_test_scenes: int | None = None
    seed: int = 1201
    method_version: int = 2


def _load_base_config(metadata: dict, config: LocalizationSweepConfig):
    base = LargeComponentStudentConfig()
    for key, value in metadata["config"].items():
        if hasattr(base, key):
            current = getattr(base, key)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(base, key, value)
    base.device = config.device
    base.require_device = config.require_device
    return base


def _oracle_target_endpoint(model, detect, image, record):
    import torch

    _reset_detect_inference_cache(detect)
    image = image.detach().requires_grad_(True)
    decoded, levels = _capture_with_grad(model, detect, image)
    target = torch.as_tensor(
        [record["target_box"]], device=image.device, dtype=torch.float32
    )
    class_id = torch.as_tensor(
        [record["class_id"]], device=image.device, dtype=torch.long
    )
    loss, _ = dynamic_score_geometry_loss(
        decoded,
        target,
        class_id,
        match_iou=0.50,
        iou_temperature=0.07,
        iou_weight=4.0,
        smoothmax_temperature=0.35,
    )
    gradients = torch.autograd.grad(loss, levels)
    return [item.detach() for item in levels], [item.detach() for item in gradients]


def _experiment_specs(config: LocalizationSweepConfig) -> list[dict]:
    budgets = tuple(sorted(set(int(value) for value in config.candidate_budgets)))
    clusters = tuple(
        sorted(set(int(value) for value in config.blind_cluster_counts))
    )
    middle_budget = min(budgets, key=lambda value: abs(value - 4000))
    middle_clusters = min(clusters, key=lambda value: abs(value - 3))
    specs = []
    # Budget sweep for the strongest previous selector and its hybrid extension.
    for selector in ("coordinate", "hybrid"):
        for budget in budgets:
            specs.append({
                "target_source": "blind",
                "selector": selector,
                "candidate_budget": budget,
                "cluster_count": middle_clusters,
            })
    # Direct selector comparison at the current operating budget.
    specs.append({
        "target_source": "blind",
        "selector": "spatial",
        "candidate_budget": middle_budget,
        "cluster_count": middle_clusters,
    })
    # Multi-hypothesis sweep for the hybrid mechanism.
    for cluster_count in clusters:
        specs.append({
            "target_source": "blind",
            "selector": "hybrid",
            "candidate_budget": middle_budget,
            "cluster_count": cluster_count,
        })
    # With the true target functional objective, only support search remains.
    for selector in ("coordinate", "spatial", "hybrid"):
        specs.append({
            "target_source": "oracle_target",
            "selector": selector,
            "candidate_budget": middle_budget,
            "cluster_count": 0,
        })
    unique = {}
    for spec in specs:
        key = (
            spec["target_source"],
            spec["selector"],
            spec["candidate_budget"],
            spec["cluster_count"],
        )
        unique[key] = spec
    return list(unique.values())


def _condition_key(spec: dict) -> str:
    return (
        f"{spec['target_source']}__{spec['selector']}"
        f"__b{int(spec['candidate_budget'])}"
        f"__c{int(spec['cluster_count'])}"
    )


def _evaluate_conditions(
    detect,
    observed,
    conditions,
    row,
    evaluator_config,
    condition_batch_size: int,
):
    import torch

    result = {}
    names = list(conditions)
    for start in range(0, len(names), int(condition_batch_size)):
        chunk = names[start:start + int(condition_batch_size)]
        plan = ["observed", *chunk]
        levels = {
            "observed": observed,
            **{name: conditions[name] for name in chunk},
        }
        with torch.no_grad():
            batched = [
                torch.cat([levels[name][level] for name in plan], dim=0)
                for level in range(len(observed))
            ]
            _box, _cls, raw = _head_branches(detect, batched)
            evaluated = _evaluate_batch(detect, raw, row, evaluator_config)
        by_name = dict(zip(plan, evaluated, strict=True))
        result.setdefault("observed", by_name["observed"])
        result.update({name: by_name[name] for name in chunk})
    return result


def _summarize(rows: pd.DataFrame):
    group_columns = [
        "target_source",
        "selector",
        "candidate_budget",
        "cluster_count",
    ]
    summary_rows = []
    group_rows = []
    for keys, current in rows.groupby(group_columns):
        target_source, selector, budget, cluster_count = keys
        hidden = current.baseline_target_detected.eq(0)
        item = {
            "target_source": target_source,
            "selector": selector,
            "candidate_budget": int(budget),
            "cluster_count": int(cluster_count),
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
        total_budget = 3 * int(budget) if int(budget) > 0 else np.nan
        item["recovered_per_10k_coordinates"] = float(
            10000 * item["hidden_recovered_n"] / total_budget
            if np.isfinite(total_budget) else np.nan
        )
        summary_rows.append(item)
        for analysis_group, group in current.groupby("analysis_group"):
            group_hidden = group.baseline_target_detected.eq(0)
            group_rows.append({
                **{
                    key: item[key]
                    for key in group_columns
                },
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
    return pd.DataFrame(summary_rows), pd.DataFrame(group_rows)


def _recommendation(summary: pd.DataFrame) -> str:
    blind = summary[
        summary.target_source.eq("blind")
        & summary.selector.ne("exact")
    ].copy()
    best = blind.sort_values(
        [
            "hidden_recovered_n",
            "baseline_lost_n",
            "candidate_budget",
        ],
        ascending=[False, True, True],
    ).iloc[0]
    efficient = blind.sort_values(
        ["recovered_per_10k_coordinates", "hidden_recovered_n"],
        ascending=[False, False],
    ).iloc[0]
    oracle = summary[summary.target_source.eq("oracle_target")].sort_values(
        "hidden_recovered_n", ascending=False
    )
    exact = summary[summary.target_source.eq("teacher")]
    lines = [
        "# Localization mechanism sweep",
        "",
        "All candidate mechanisms use oracle component values on the support "
        "they found. Differences therefore measure localization only.",
        "",
        "## Best blind mechanism",
        "",
        (
            f"- selector={best.selector}, budget/level={int(best.candidate_budget)}, "
            f"clusters={int(best.cluster_count)}"
        ),
        (
            f"- recovered {int(best.hidden_recovered_n)}/"
            f"{int(best.hidden_n)} hidden targets"
        ),
        (
            f"- support energy recall={float(best.support_energy_recall):.3f}"
        ),
        "",
        "## Most budget-efficient blind mechanism",
        "",
        (
            f"- selector={efficient.selector}, "
            f"budget/level={int(efficient.candidate_budget)}, "
            f"clusters={int(efficient.cluster_count)}"
        ),
        (
            f"- recovered per 10k coordinates="
            f"{float(efficient.recovered_per_10k_coordinates):.2f}"
        ),
    ]
    if len(oracle):
        item = oracle.iloc[0]
        lines.extend([
            "",
            "## Oracle-target diagnostic",
            "",
            (
                f"Best oracle-target selector recovered "
                f"{int(item.hidden_recovered_n)}/{int(item.hidden_n)}. "
                "The gap to the best blind condition estimates proposal-target "
                "localization loss."
            ),
        ])
    if len(exact):
        item = exact.iloc[0]
        lines.extend([
            "",
            "## Exact component ceiling",
            "",
            (
                f"Recovered {int(item.hidden_recovered_n)}/"
                f"{int(item.hidden_n)} hidden targets."
            ),
        ])
    return "\n".join(lines) + "\n"


def run_localization_sweep(
    config: LocalizationSweepConfig,
) -> Path:
    started = time.time()
    base_run = Path(config.base_run).resolve()
    metadata = json.loads((base_run / "run.json").read_text(encoding="utf-8"))
    base_config = _load_base_config(metadata, config)
    test_rows = pd.read_csv(base_run / "test_split.csv")
    if config.max_test_scenes is not None:
        test_rows = test_rows.head(int(config.max_test_scenes)).copy()
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
    specs = _experiment_specs(config)
    blind_counts = sorted({
        int(spec["cluster_count"])
        for spec in specs
        if spec["target_source"] == "blind"
    })
    examples = _cache_lookup(exp)
    rows = []
    for record in tqdm(records, desc="localization sweep", unit="scene"):
        example = examples[record["example_id"]]
        _clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, patched_image, patched_image)
        endpoints = {}
        for cluster_count in blind_counts:
            current_config = LargeComponentStudentConfig(
                **asdict(base_config)
            )
            current_config.blind_top_clusters = int(cluster_count)
            levels, gradients, endpoint_metadata = _blind_endpoint(
                model, detect, pair[:1], current_config
            )
            endpoints[("blind", cluster_count)] = (
                levels, gradients, endpoint_metadata
            )
        oracle_levels, oracle_gradients = _oracle_target_endpoint(
            model, detect, pair[:1], record
        )
        endpoints[("oracle_target", 0)] = (
            oracle_levels, oracle_gradients, {}
        )
        observed = endpoints[("blind", blind_counts[0])][0]
        teacher_indices, teacher_components = _teacher_arrays(record, observed)
        conditions = {}
        condition_metadata = {}
        for spec in specs:
            levels, gradients, endpoint_metadata = endpoints[
                (spec["target_source"], int(spec["cluster_count"]))
            ]
            candidates = _blind_candidate_indices(
                levels,
                gradients,
                reference_mean,
                reference_std,
                int(spec["candidate_budget"]),
                str(spec["selector"]),
            )
            values = _labels_on_candidates(
                candidates, teacher_indices, teacher_components
            )
            correction, _ = _direct_maps(values, candidates, levels)
            name = _condition_key(spec)
            conditions[name] = [
                level - delta
                for level, delta in zip(levels, correction, strict=True)
            ]
            condition_metadata[name] = {
                **spec,
                **endpoint_metadata,
                **_support_metrics(
                    candidates, teacher_indices, teacher_components
                ),
            }
        exact_maps, _ = _direct_maps(
            teacher_components, teacher_indices, observed
        )
        exact_name = "teacher__exact__b0__c0"
        conditions[exact_name] = [
            level - delta
            for level, delta in zip(observed, exact_maps, strict=True)
        ]
        condition_metadata[exact_name] = {
            "target_source": "teacher",
            "selector": "exact",
            "candidate_budget": 0,
            "cluster_count": 0,
            "support_recall": 1.0,
            "support_energy_recall": 1.0,
        }
        evaluated = _evaluate_conditions(
            detect,
            observed,
            conditions,
            record["row"],
            base_config,
            config.condition_batch_size,
        )
        baseline = evaluated["observed"]
        for name, item in condition_metadata.items():
            corrected = evaluated[name]
            rows.append({
                "example_id": record["example_id"],
                "analysis_group": record["analysis_group"],
                "baseline_target_detected": baseline["target_detected"],
                "corrected_target_detected": corrected["target_detected"],
                "baseline_target_conf": baseline["post_target_conf"],
                "corrected_target_conf": corrected["post_target_conf"],
                **item,
            })
        _reset_detect_inference_cache(detect)
        release_accelerator_memory()
    rows = pd.DataFrame(rows)
    summary, group_summary = _summarize(rows)
    payload = {
        **asdict(config),
        "base_run": str(base_run),
        "test_ids": test_rows.example_id.astype(str).tolist(),
        "specs": specs,
    }
    run_dir = (
        Path(config.output_dir)
        / f"localization_sweep_{stable_hash(payload)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(run_dir / "localization_rows.csv", index=False)
    summary.to_csv(run_dir / "localization_summary.csv", index=False)
    group_summary.to_csv(
        run_dir / "localization_group_summary.csv", index=False
    )
    (run_dir / "recommendation.md").write_text(
        _recommendation(summary), encoding="utf-8"
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "elapsed_seconds": time.time() - started,
                "config": asdict(config),
                "base_run": str(base_run),
                "n_test_scenes": len(records),
                "specs": specs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return run_dir


def _csv_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare blind localization mechanisms using oracle correction "
            "values, without retraining the student."
        )
    )
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--require-device", action="store_true")
    parser.add_argument(
        "--candidate-budgets", type=_csv_int_tuple, default=(2000, 4000, 8000)
    )
    parser.add_argument(
        "--blind-cluster-counts", type=_csv_int_tuple, default=(1, 3, 5)
    )
    parser.add_argument("--condition-batch-size", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = LocalizationSweepConfig(
        base_run=args.base_run,
        output_dir=args.output_dir,
        device=args.device,
        require_device=args.require_device,
        candidate_budgets=args.candidate_budgets,
        blind_cluster_counts=args.blind_cluster_counts,
        condition_batch_size=args.condition_batch_size,
    )
    if args.smoke:
        config.candidate_budgets = (200, 400)
        config.blind_cluster_counts = (1, 3)
        config.max_test_scenes = 2
        config.condition_batch_size = 4
    print(run_localization_sweep(config))


if __name__ == "__main__":
    main()
