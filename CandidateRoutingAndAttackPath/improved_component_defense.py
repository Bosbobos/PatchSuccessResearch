from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .autonomous_negative_repair import (
    AutonomousNegativeRepairConfig,
    _clusters_from_frame,
    _target_overlap,
)
from .candidate_reserve import _cache_lookup, _evaluate_batch
from .candidate_routing import _flat_location, _level_slices, _xywh_to_xyxy
from .causal_repair import _load_inputs
from .common import StorageBudget, load_experiment, release_accelerator_memory, stable_hash
from .followup_common import (
    ATTACK_PATH_DB,
    FOLLOWUP_DIR,
    MANIFEST_CSV,
    TRACE_DB,
    balanced_subset,
    write_summary,
)
from .mechanism_followup import _decode, _head_branches
from .self_counterfactual_defense import _all_class_nms, _detection_set_metrics
from .single_forward_component import (
    _collect_clean_channel_moments,
    _fast_aggregate_top_negative_maps,
    _intervention_raw,
    _split_reference_evaluation,
)


@dataclass(slots=True)
class ImprovedComponentDefenseConfig(AutonomousNegativeRepairConfig):
    person_top_k: int = 1000
    class_agnostic_top_k: int = 1000
    class_agnostic_per_level_k: int = 300
    cluster_candidate_limit: int = 2200
    prefilter_per_strategy: int = 2
    coverage_fractions: tuple[float, ...] = (0.75, 0.90)
    repair_scales: tuple[float, ...] = (1.0, 1.25, 1.50)
    clean_evaluation_examples: int = 100
    proposal_policies: tuple[str, ...] = ("person", "hybrid")
    method_version: int = 1


def _proposal_frame(
    detect,
    raw,
    class_id: int,
    policy: str,
    config: ImprovedComponentDefenseConfig,
) -> pd.DataFrame:
    import torch

    decoded = _decode(detect, raw)
    boxes = _xywh_to_xyxy(decoded[0, :4].transpose(0, 1))
    class_scores = decoded[0, 4:4 + int(detect.nc)].transpose(0, 1)
    person_scores = class_scores[:, int(class_id)]
    proposal_scores, predicted_classes = class_scores.max(dim=1)
    total = int(person_scores.numel())
    person_k = min(total, int(config.person_top_k))
    indices = set(person_scores.topk(person_k).indices.detach().cpu().tolist())
    if policy == "hybrid":
        class_k = min(total, int(config.class_agnostic_top_k))
        indices.update(
            proposal_scores.topk(class_k).indices.detach().cpu().tolist()
        )
        slices = _level_slices(raw)
        for item in slices:
            start, stop = int(item["start"]), int(item["end"])
            count = min(stop - start, int(config.class_agnostic_per_level_k))
            local = proposal_scores[start:stop].topk(count).indices + start
            indices.update(local.detach().cpu().tolist())
    elif policy != "person":
        raise ValueError(f"Unknown proposal policy: {policy}")
    ordered = sorted(
        indices,
        key=lambda index: float(
            person_scores[index] if policy == "person" else proposal_scores[index]
        ),
        reverse=True,
    )
    selected = torch.as_tensor(
        ordered, dtype=torch.long, device=person_scores.device
    )
    flat_indices = selected.detach().cpu().numpy().astype(np.int64)
    selected_boxes = boxes[selected].detach().float().cpu().numpy()
    selected_person = (
        person_scores[selected].detach().float().cpu().numpy()
    )
    selected_proposal = (
        (
            person_scores[selected]
            if policy == "person"
            else proposal_scores[selected]
        ).detach().float().cpu().numpy()
    )
    selected_classes = (
        predicted_classes[selected].detach().cpu().numpy().astype(np.int64)
    )
    slices = _level_slices(raw)
    levels = np.empty(len(flat_indices), dtype=np.int64)
    ys = np.empty(len(flat_indices), dtype=np.int64)
    xs = np.empty(len(flat_indices), dtype=np.int64)
    for item in slices:
        start, stop = int(item["start"]), int(item["end"])
        level = int(item["level"])
        mask = (flat_indices >= start) & (flat_indices < stop)
        local = flat_indices[mask] - start
        width = int(raw[level].shape[-1])
        levels[mask] = level
        ys[mask] = local // width
        xs[mask] = local % width
    return pd.DataFrame({
        "flat_index": flat_indices,
        "level_index": levels,
        "y_index": ys,
        "x_index": xs,
        "score": selected_person,
        "proposal_score": selected_proposal,
        "predicted_class": selected_classes,
        "x1": selected_boxes[:, 0],
        "y1": selected_boxes[:, 1],
        "x2": selected_boxes[:, 2],
        "y2": selected_boxes[:, 3],
    })


def _prefilter(
    clusters: list[dict],
    policy: str,
    config: ImprovedComponentDefenseConfig,
) -> list[dict]:
    metrics = ["noisy_or", "reserve_tension"]
    if policy == "hybrid":
        metrics.extend(["object_suppression_tension", "max_proposal_score"])
    chosen = []
    seen = set()
    for metric in metrics:
        ordered = sorted(clusters, key=lambda item: item[metric], reverse=True)
        for item in ordered[: int(config.prefilter_per_strategy)]:
            key = tuple(sorted(item["selection"].flat_index.astype(int)))
            if key not in seen:
                chosen.append(item)
                seen.add(key)
    return chosen


def _scaled(maps, scale: float):
    return [value * float(scale) for value in maps]


def _condition_plan(scored_by_policy: dict[str, list[dict]]) -> dict[str, dict]:
    person_reserve = sorted(
        scored_by_policy["person"],
        key=lambda item: item["cluster"]["reserve_tension"],
        reverse=True,
    )[0]
    person_noisy_or = sorted(
        scored_by_policy["person"],
        key=lambda item: item["cluster"]["noisy_or"],
        reverse=True,
    )[0]
    person_diffuse = sorted(
        scored_by_policy["person"],
        key=lambda item: item["metadata"]["fixed_k1000"][
            "diffuse_negative_leverage"
        ],
        reverse=True,
    )[0]
    plan = {
        "person_fixed_k1000_s1": {
            "policy": "person", "ranking": "reserve_tension",
            "source": person_reserve, "map": "fixed_k1000", "scale": 1.0,
        },
        "person_coverage90_s1": {
            "policy": "person", "ranking": "reserve_tension",
            "source": person_reserve, "map": "coverage_90", "scale": 1.0,
        },
        "person_noisy_or_coverage90_s1": {
            "policy": "person", "ranking": "noisy_or",
            "source": person_noisy_or, "map": "coverage_90", "scale": 1.0,
        },
        "person_diffuse_coverage90_s1": {
            "policy": "person", "ranking": "diffuse_negative_leverage",
            "source": person_diffuse, "map": "coverage_90", "scale": 1.0,
        },
    }
    if "hybrid" not in scored_by_policy:
        return plan
    hybrid_reserve = sorted(
        scored_by_policy["hybrid"],
        key=lambda item: item["cluster"]["reserve_tension"],
        reverse=True,
    )[0]
    hybrid_object = sorted(
        scored_by_policy["hybrid"],
        key=lambda item: item["cluster"]["object_suppression_tension"],
        reverse=True,
    )[0]
    plan.update({
        "hybrid_reserve_fixed_k1000_s1": {
            "policy": "hybrid", "ranking": "reserve_tension",
            "source": hybrid_reserve, "map": "fixed_k1000", "scale": 1.0,
        },
        "hybrid_reserve_coverage75_s1": {
            "policy": "hybrid", "ranking": "reserve_tension",
            "source": hybrid_reserve, "map": "coverage_75", "scale": 1.0,
        },
        "hybrid_reserve_coverage90_s1": {
            "policy": "hybrid", "ranking": "reserve_tension",
            "source": hybrid_reserve, "map": "coverage_90", "scale": 1.0,
        },
        "hybrid_reserve_coverage90_s1p25": {
            "policy": "hybrid", "ranking": "reserve_tension",
            "source": hybrid_reserve, "map": "coverage_90", "scale": 1.25,
        },
        "hybrid_reserve_coverage90_s1p5": {
            "policy": "hybrid", "ranking": "reserve_tension",
            "source": hybrid_reserve, "map": "coverage_90", "scale": 1.50,
        },
        "hybrid_object_coverage90_s1": {
            "policy": "hybrid", "ranking": "object_suppression_tension",
            "source": hybrid_object, "map": "coverage_90", "scale": 1.0,
        },
    })
    return plan


def _analyze(run_dir: Path) -> None:
    clusters = pd.read_csv(run_dir / "improved_cluster_rows.csv")
    rows = pd.read_csv(run_dir / "improved_repair_rows.csv")
    rows["example_id"] = rows.example_id.astype(str)
    clusters["example_id"] = clusters.example_id.astype(str)
    hidden_ids = set(
        rows[
            rows.input_kind.eq("patched")
            & rows.condition.eq("observed")
            & rows.source_hidden.astype(bool)
        ].example_id
    )
    hidden_clusters = clusters[
        clusters.input_kind.eq("patched")
        & clusters.example_id.isin(hidden_ids)
    ]
    localization_rows = []
    for policy in sorted(hidden_clusters.proposal_policy.unique()):
        current = hidden_clusters[hidden_clusters.proposal_policy.eq(policy)]
        for ranking in (
            ["reserve_tension", "noisy_or"]
            if policy == "person"
            else ["reserve_tension", "object_suppression_tension"]
        ):
            finalists = current[current.is_finalist.astype(bool)]
            chosen = finalists.sort_values(
                ["example_id", ranking], ascending=[True, False]
            ).drop_duplicates("example_id")
            target_any = current.groupby("example_id").is_target_cluster.max()
            target_finalist = finalists.groupby(
                "example_id"
            ).is_target_cluster.max()
            localization_rows.append({
                "proposal_policy": policy,
                "ranking": ranking,
                "hidden_n": int(len(hidden_ids)),
                "target_in_any_cluster_n": int(target_any.sum()),
                "target_in_finalists_n": int(target_finalist.sum()),
                "target_chosen_top1_n": int(
                    chosen.is_target_cluster.sum()
                ),
            })
    localization = pd.DataFrame(localization_rows)
    localization.to_csv(run_dir / "improved_localization_summary.csv", index=False)

    hidden = rows[
        rows.input_kind.eq("patched") & rows.source_hidden.astype(bool)
    ]
    raw_summary = rows.groupby(
        ["input_kind", "condition"], as_index=False
    ).agg(
        n=("example_id", "nunique"),
        target_detection_rate=("target_detected", "mean"),
        mean_detection_f1=("detection_detection_f1", "mean"),
    )
    hidden_summary = hidden.groupby("condition", as_index=False).agg(
        hidden_n=("example_id", "nunique"),
        hidden_recovery_n=("target_detected", "sum"),
        hidden_recovery_rate=("target_detected", "mean"),
        chosen_target_rate=("chosen_is_target_cluster", "mean"),
        mean_actual_k=("actual_k", "mean"),
    )
    raw_summary = raw_summary.merge(hidden_summary, on="condition", how="left")
    raw_summary.to_csv(run_dir / "improved_raw_summary.csv", index=False)

    clean_observed = rows[
        rows.input_kind.eq("clean") & rows.condition.eq("observed")
    ].drop_duplicates("example_id")
    ordered_ids = sorted(clean_observed.example_id.unique())
    calibration_ids = set(ordered_ids[::2])
    test_ids = set(ordered_ids[1::2])
    gate_rows = []
    guarded_rows = []
    observed_patched = rows[
        rows.input_kind.eq("patched") & rows.condition.eq("observed")
    ].drop_duplicates("example_id").set_index("example_id")
    for condition in sorted(
        value for value in rows.condition.unique() if value != "observed"
    ):
        condition_rows = rows[
            rows.input_kind.eq("clean") & rows.condition.eq(condition)
        ].drop_duplicates("example_id")
        calibration = condition_rows[
            condition_rows.example_id.isin(calibration_ids)
        ]
        clean_test = condition_rows[
            condition_rows.example_id.isin(test_ids)
        ]
        patched = rows[
            rows.input_kind.eq("patched") & rows.condition.eq(condition)
        ].drop_duplicates("example_id")
        patched_hidden = patched[patched.source_hidden.astype(bool)]
        for quantile in (0.80, 0.90, 0.95, 0.99):
            threshold = float(calibration.gate_score.quantile(quantile))
            clean_gate = clean_test.gate_score.gt(threshold)
            hidden_gate = patched_hidden.gate_score.gt(threshold)
            gate_rows.append({
                "condition": condition,
                "gate_quantile": quantile,
                "threshold": threshold,
                "clean_test_n": int(len(clean_test)),
                "clean_gate_n": int(clean_gate.sum()),
                "hidden_n": int(len(patched_hidden)),
                "hidden_gate_n": int(hidden_gate.sum()),
            })
            clean_f1 = np.where(
                clean_gate, clean_test.detection_detection_f1, 1.0
            )
            clean_target = np.where(
                clean_gate, clean_test.target_detected.astype(bool), True
            )
            hidden_recovered = (
                hidden_gate & patched_hidden.target_detected.astype(bool)
            )
            patched_gate = patched.gate_score.gt(threshold)
            observed_f1 = patched.example_id.map(
                observed_patched.detection_detection_f1
            )
            observed_target = patched.example_id.map(
                observed_patched.target_detected
            ).astype(bool)
            guarded_rows.append({
                "condition": condition,
                "gate_quantile": quantile,
                "clean_test_n": int(len(clean_test)),
                "clean_gate_n": int(clean_gate.sum()),
                "guarded_clean_target_detection_rate": float(
                    clean_target.mean()
                ),
                "guarded_clean_detection_f1": float(clean_f1.mean()),
                "hidden_n": int(len(patched_hidden)),
                "hidden_gate_n": int(hidden_gate.sum()),
                "guarded_hidden_recovery_n": int(hidden_recovered.sum()),
                "guarded_hidden_recovery_rate": float(
                    hidden_recovered.mean()
                ),
                "ungated_hidden_recovery_n": int(
                    patched_hidden.target_detected.sum()
                ),
                "ungated_hidden_recovery_rate": float(
                    patched_hidden.target_detected.mean()
                ),
                "guarded_all_patched_target_detection_rate": float(
                    np.where(
                        patched_gate,
                        patched.target_detected.astype(bool),
                        observed_target,
                    ).mean()
                ),
                "guarded_all_patched_detection_f1": float(
                    np.where(
                        patched_gate,
                        patched.detection_detection_f1,
                        observed_f1,
                    ).mean()
                ),
            })
    pd.DataFrame(gate_rows).to_csv(
        run_dir / "improved_gate_summary.csv", index=False
    )
    guarded = pd.DataFrame(guarded_rows)
    guarded.to_csv(run_dir / "improved_guarded_summary.csv", index=False)
    best = guarded.sort_values(
        [
            "guarded_hidden_recovery_rate",
            "guarded_clean_detection_f1",
        ],
        ascending=False,
    ).iloc[0]
    (run_dir / "analysis_digest.md").write_text(
        "\n".join([
            "# Improved component defense",
            "",
            f"- hidden examples: {len(hidden_ids)}",
            f"- best guarded condition: `{best.condition}` at q"
            f"{int(best.gate_quantile * 100)}",
            f"- guarded recovery: {int(best.guarded_hidden_recovery_n)}/"
            f"{int(best.hidden_n)}",
            f"- clean target detection: "
            f"{best.guarded_clean_target_detection_rate:.3f}",
            f"- clean full-output F1: {best.guarded_clean_detection_f1:.3f}",
            "",
            "Read improved_localization_summary.csv, improved_raw_summary.csv, "
            "and improved_guarded_summary.csv.",
        ]) + "\n",
        encoding="utf-8",
    )


def run_improved_component_defense(
    config: ImprovedComponentDefenseConfig | None = None,
) -> Path:
    config = config or ImprovedComponentDefenseConfig()
    started = time.time()
    StorageBudget(config.output_dir, config.max_output_gb).check()
    selected, _ = _load_inputs(
        Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None
    )
    selected = balanced_subset(
        selected, int(config.examples_per_group), seed=int(config.seed)
    )
    reference, evaluation = _split_reference_evaluation(selected, config)
    exp, cache_path = load_experiment(
        prefer_device=config.device, require_device=bool(config.require_device)
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    cache = _cache_lookup(exp)
    means, _stds = _collect_clean_channel_moments(
        exp, model, detect, cache, reference
    )
    result_rows = []
    cluster_rows = []
    for row in tqdm(
        evaluation.itertuples(index=False),
        total=len(evaluation),
        desc="improved component defense",
        unit="image",
    ):
        example_id = str(row.example_id)
        example = cache[example_id]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        clean_inputs = _capture_detect_inputs(model, detect, pair[0:1])
        import torch

        with torch.inference_mode():
            _clean_box, _clean_cls, clean_raw = _head_branches(
                detect, clean_inputs
            )
            clean_reference = _all_class_nms(detect, clean_raw, config)[0]
        del clean_inputs, clean_raw
        for input_kind, image in (("clean", pair[0:1]), ("patched", pair[1:2])):
            endpoint = _capture_detect_inputs(model, detect, image)
            with torch.inference_mode():
                _box, _cls, raw = _head_branches(detect, endpoint)
            scored_by_policy = {}
            gate_by_policy = {}
            for policy in config.proposal_policies:
                frame = _proposal_frame(
                    detect, raw, int(row.class_id), policy, config
                )
                all_clusters = _clusters_from_frame(frame, config)
                finalists = _prefilter(all_clusters, policy, config)
                finalist_keys = {
                    tuple(sorted(item["selection"].flat_index.astype(int)))
                    for item in finalists
                }
                for cluster_index, cluster in enumerate(all_clusters):
                    target_iou = _target_overlap(cluster["selection"], row)
                    key = tuple(
                        sorted(cluster["selection"].flat_index.astype(int))
                    )
                    cluster_rows.append({
                        "example_id": example_id,
                        "analysis_group": str(row.analysis_group),
                        "input_kind": input_kind,
                        "proposal_policy": policy,
                        "cluster_index": int(cluster_index),
                        "is_finalist": int(key in finalist_keys),
                        "is_target_cluster": int(
                            target_iou >= float(config.target_iou)
                        ),
                        "target_iou": target_iou,
                        "n_members": int(cluster["n_members"]),
                        "max_score": float(cluster["max_score"]),
                        "max_proposal_score": float(
                            cluster["max_proposal_score"]
                        ),
                        "noisy_or": float(cluster["noisy_or"]),
                        "reserve_tension": float(
                            cluster["reserve_tension"]
                        ),
                        "object_suppression_tension": float(
                            cluster["object_suppression_tension"]
                        ),
                    })
                scored = []
                for cluster in finalists:
                    maps, metadata = _fast_aggregate_top_negative_maps(
                        detect,
                        endpoint,
                        cluster["selection"],
                        int(row.class_id),
                        means,
                        config,
                        fixed_ks=(1000,),
                        coverage_fractions=config.coverage_fractions,
                    )
                    target_iou = _target_overlap(cluster["selection"], row)
                    scored.append({
                        "cluster": cluster,
                        "maps": maps,
                        "metadata": metadata,
                        "target_iou": target_iou,
                    })
                if not scored:
                    raise RuntimeError(
                        f"No finalist clusters for {example_id}/{policy}"
                    )
                scored_by_policy[policy] = scored
                gate_by_policy[policy] = max(
                    item["metadata"]["fixed_k1000"][
                        "diffuse_negative_leverage"
                    ]
                    for item in scored
                )
            plan = _condition_plan(scored_by_policy)
            intervention_maps = {
                name: _scaled(
                    spec["source"]["maps"][spec["map"]], spec["scale"]
                )
                for name, spec in plan.items()
            }
            names, intervention_raw = _intervention_raw(
                detect, endpoint, intervention_maps
            )
            with torch.inference_mode():
                evaluated = _evaluate_batch(
                    detect, intervention_raw, row, config
                )
                detections = _all_class_nms(
                    detect, intervention_raw, config
                )
            source_hidden = int(evaluated[0]["target_hidden"])
            for name, result, detection in zip(
                names, evaluated, detections, strict=True
            ):
                if name == "observed":
                    spec = None
                    metadata = {}
                    chosen_target_iou = np.nan
                    gate_score = max(gate_by_policy.values())
                    proposal_policy = "none"
                    ranking = "none"
                    map_name = "none"
                    scale = 0.0
                else:
                    spec = plan[name]
                    metadata = spec["source"]["metadata"][spec["map"]]
                    chosen_target_iou = float(
                        spec["source"]["target_iou"]
                    )
                    proposal_policy = str(spec["policy"])
                    gate_score = float(gate_by_policy[proposal_policy])
                    ranking = str(spec["ranking"])
                    map_name = str(spec["map"])
                    scale = float(spec["scale"])
                result.update({
                    "example_id": example_id,
                    "analysis_group": str(row.analysis_group),
                    "input_kind": input_kind,
                    "condition": name,
                    "source_hidden": source_hidden,
                    "proposal_policy": proposal_policy,
                    "ranking": ranking,
                    "map_name": map_name,
                    "repair_scale": scale,
                    "chosen_target_iou": chosen_target_iou,
                    "chosen_is_target_cluster": int(
                        np.isfinite(chosen_target_iou)
                        and chosen_target_iou >= float(config.target_iou)
                    ),
                    "gate_score": gate_score,
                    "person_gate_score": float(gate_by_policy["person"]),
                    "hybrid_gate_score": float(
                        gate_by_policy.get("hybrid", np.nan)
                    ),
                    **metadata,
                })
                for key, value in _detection_set_metrics(
                    clean_reference, detection, config.target_iou
                ).items():
                    result[f"detection_{key}"] = value
                result_rows.append(result)
            del endpoint, raw, intervention_raw
            release_accelerator_memory()

    payload = {
        **asdict(config),
        "reference_ids": reference.example_id.astype(str).tolist(),
        "evaluation_ids": evaluation.example_id.astype(str).tolist(),
    }
    run_dir = Path(config.output_dir) / (
        f"improved_component_defense_{stable_hash(payload)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(
        run_dir / "improved_repair_rows.csv", index=False
    )
    pd.DataFrame(cluster_rows).to_csv(
        run_dir / "improved_cluster_rows.csv", index=False
    )
    elapsed = time.time() - started
    write_summary(run_dir / "summary.json", {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "reference_examples": int(reference.example_id.nunique()),
        "evaluation_examples": int(evaluation.example_id.nunique()),
        "cache_path": str(cache_path),
        "config": asdict(config),
        "limitations": [
            "One detector, class, patch family, and placement protocol.",
            "Clean paired images and target boxes are used only for evaluation.",
            "The method uses white-box Detect-input activations and gradients.",
            "Operating points are exploratory until frozen and validated on a new cohort.",
        ],
    })
    _analyze(run_dir)
    StorageBudget(config.output_dir, config.max_output_gb).check()
    return run_dir


if __name__ == "__main__":
    print(run_improved_component_defense())
