from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .autonomous_negative_repair import (
    AutonomousNegativeRepairConfig,
    _clusters_from_frame,
    _prefilter_clusters,
)
from .candidate_reserve import _cache_lookup
from .causal_repair import _load_inputs
from .common import ensure_import_paths, load_experiment
from .followup_common import (
    ATTACK_PATH_DB,
    FOLLOWUP_DIR,
    MANIFEST_CSV,
    TRACE_DB,
    balanced_subset,
)
from .mechanism_followup import _head_branches
from .self_counterfactual_defense import _all_class_nms, _decoded_frame
from .single_forward_component import (
    _collect_clean_channel_moments,
    _fast_aggregate_top_negative_map,
    _intervention_raw,
    _single_forward_maps,
    _split_reference_evaluation,
)


def _sync() -> None:
    import torch

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def _milliseconds(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float) * 1000.0
    return {
        "median_ms": float(np.median(array)),
        "mean_ms": float(np.mean(array)),
        "p10_ms": float(np.quantile(array, 0.10)),
        "p90_ms": float(np.quantile(array, 0.90)),
        "stdev_ms": float(statistics.pstdev(array)),
        "n": int(len(array)),
    }


def run_benchmark(repeats: int = 7, warmups: int = 2) -> Path:
    import torch

    ensure_import_paths()
    from segmentig_detector.yolo_utils import get_detect_module, safe_model_forward

    config = AutonomousNegativeRepairConfig(
        device="mps",
        require_device=True,
        examples_per_group=50,
        reference_examples_per_group=25,
        clean_evaluation_examples=100,
        top_negative_k=(250, 500, 1000),
        cluster_selection_k=500,
        repair_top_clusters=(1,),
        cluster_ranking="reserve_tension",
        method_version=5,
    )
    selected, _ = _load_inputs(
        Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None
    )
    selected = balanced_subset(
        selected, int(config.examples_per_group), seed=int(config.seed)
    )
    reference, evaluation = _split_reference_evaluation(selected, config)
    exp, _cache_path = load_experiment(prefer_device="mps", require_device=True)
    yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    cache = _cache_lookup(exp)
    means, _stds = _collect_clean_channel_moments(
        exp, model, detect, cache, reference
    )
    hidden = evaluation[
        evaluation.analysis_group.astype(str).str.startswith("hidden")
    ]
    row = next(hidden.itertuples(index=False))
    example = cache[str(row.example_id)]
    _clean_image, patched_image, _ = exp._images_for_example(example)
    # Preprocessing and the one-time population-baseline estimation are not
    # included in either model or defense latency.
    image = _preprocess_pair(exp, patched_image, patched_image)[0:1]

    def model_forward() -> None:
        with torch.inference_mode():
            safe_model_forward(model, image)

    def observed_detection() -> None:
        endpoint = _capture_detect_inputs(model, detect, image)
        with torch.inference_mode():
            _box, _cls, raw = _head_branches(detect, endpoint)
            _all_class_nms(detect, raw, config)

    def defense_once() -> dict[str, float | int]:
        _sync()
        total_start = time.perf_counter()
        endpoint = _capture_detect_inputs(model, detect, image)
        with torch.inference_mode():
            _box, _cls, raw = _head_branches(detect, endpoint)
        frame = _decoded_frame(detect, raw, int(row.class_id))
        all_clusters = _clusters_from_frame(frame, config)
        finalists = _prefilter_clusters(all_clusters, config)
        _sync()
        attribution_start = time.perf_counter()
        scored = []
        for cluster in finalists:
            maps, metadata = _single_forward_maps(
                detect,
                endpoint,
                cluster["selection"],
                int(row.class_id),
                means,
                config,
                oracle_delta_inputs=None,
            )
            scored.append({
                "cluster": cluster,
                "maps": maps,
                "metadata": metadata,
            })
        _sync()
        attribution_end = time.perf_counter()
        scored.sort(
            key=lambda item: float(item["cluster"]["reserve_tension"]),
            reverse=True,
        )
        chosen = scored[0]
        names, intervention_raw = _intervention_raw(
            detect,
            endpoint,
            {"top_negative_1000": chosen["maps"]["top_negative_1000"]},
        )
        with torch.inference_mode():
            _all_class_nms(detect, intervention_raw, config)
        _sync()
        total_end = time.perf_counter()
        return {
            "total_seconds": total_end - total_start,
            "attribution_seconds": attribution_end - attribution_start,
            "other_seconds": (
                total_end - total_start - attribution_end + attribution_start
            ),
            "n_all_clusters": int(len(all_clusters)),
            "n_finalists": int(len(finalists)),
            "intervention_conditions": int(len(names)),
        }

    def fast_defense_once() -> dict[str, float | int]:
        _sync()
        total_start = time.perf_counter()
        endpoint = _capture_detect_inputs(model, detect, image)
        with torch.inference_mode():
            _box, _cls, raw = _head_branches(detect, endpoint)
        frame = _decoded_frame(detect, raw, int(row.class_id))
        all_clusters = _clusters_from_frame(frame, config)
        finalists = _prefilter_clusters(all_clusters, config)
        _sync()
        attribution_start = time.perf_counter()
        scored = []
        for cluster in finalists:
            intervention_map, metadata = _fast_aggregate_top_negative_map(
                detect,
                endpoint,
                cluster["selection"],
                int(row.class_id),
                means,
                config,
                k=1000,
            )
            scored.append({
                "cluster": cluster,
                "map": intervention_map,
                "metadata": metadata,
            })
        _sync()
        attribution_end = time.perf_counter()
        scored.sort(
            key=lambda item: float(item["cluster"]["reserve_tension"]),
            reverse=True,
        )
        chosen = scored[0]
        names, intervention_raw = _intervention_raw(
            detect,
            endpoint,
            {"top_negative_1000": chosen["map"]},
        )
        with torch.inference_mode():
            _all_class_nms(detect, intervention_raw, config)
        _sync()
        total_end = time.perf_counter()
        return {
            "total_seconds": total_end - total_start,
            "attribution_seconds": attribution_end - attribution_start,
            "other_seconds": (
                total_end - total_start - attribution_end + attribution_start
            ),
            "n_all_clusters": int(len(all_clusters)),
            "n_finalists": int(len(finalists)),
            "intervention_conditions": int(len(names)),
            "max_gate_score": float(max(
                item["metadata"]["diffuse_negative_leverage"]
                for item in scored
            )),
        }

    for _ in range(int(warmups)):
        model_forward()
        observed_detection()
        defense_once()
        fast_defense_once()
    _sync()

    forward_times = []
    observed_times = []
    defense_profiles = []
    fast_defense_profiles = []
    for _ in range(int(repeats)):
        _sync()
        started = time.perf_counter()
        model_forward()
        _sync()
        forward_times.append(time.perf_counter() - started)

        _sync()
        started = time.perf_counter()
        observed_detection()
        _sync()
        observed_times.append(time.perf_counter() - started)

        defense_profiles.append(defense_once())
        fast_defense_profiles.append(fast_defense_once())

    defense_times = [
        float(item["total_seconds"]) for item in defense_profiles
    ]
    attribution_times = [
        float(item["attribution_seconds"]) for item in defense_profiles
    ]
    other_times = [
        float(item["other_seconds"]) for item in defense_profiles
    ]
    fast_defense_times = [
        float(item["total_seconds"]) for item in fast_defense_profiles
    ]
    fast_attribution_times = [
        float(item["attribution_seconds"]) for item in fast_defense_profiles
    ]
    fast_other_times = [
        float(item["other_seconds"]) for item in fast_defense_profiles
    ]
    output = {
        "status": "complete",
        "device": str(next(model.parameters()).device),
        "example_id": str(row.example_id),
        "analysis_group": str(row.analysis_group),
        "image_shape": [int(value) for value in image.shape],
        "warmups": int(warmups),
        "population_reference_examples": int(len(reference)),
        "population_reference_is_offline": True,
        "model_forward_only": _milliseconds(forward_times),
        "observed_detection_instrumented": _milliseconds(observed_times),
        "current_defense_total": _milliseconds(defense_times),
        "current_defense_attribution": _milliseconds(attribution_times),
        "current_defense_other": _milliseconds(other_times),
        "fast_exact_defense_total": _milliseconds(fast_defense_times),
        "fast_exact_defense_attribution": _milliseconds(
            fast_attribution_times
        ),
        "fast_exact_defense_other": _milliseconds(fast_other_times),
        "median_ratios": {
            "defense_over_forward_only": float(
                np.median(defense_times) / np.median(forward_times)
            ),
            "defense_over_instrumented_detection": float(
                np.median(defense_times) / np.median(observed_times)
            ),
            "fast_defense_over_forward_only": float(
                np.median(fast_defense_times) / np.median(forward_times)
            ),
            "fast_defense_over_instrumented_detection": float(
                np.median(fast_defense_times) / np.median(observed_times)
            ),
            "fast_over_current_defense": float(
                np.median(fast_defense_times) / np.median(defense_times)
            ),
        },
        "n_all_clusters": int(defense_profiles[-1]["n_all_clusters"]),
        "n_finalists": int(defense_profiles[-1]["n_finalists"]),
        "intervention_conditions": int(
            defense_profiles[-1]["intervention_conditions"]
        ),
        "scope": [
            "Preprocessing is excluded from both timings.",
            "Population channel means are estimated offline and excluded.",
            "Instrumented detection includes one full model forward, one repeated Detect-head forward, decoding, and NMS.",
            "Current defense retains diagnostic SVD and all configured top-k maps; a production implementation can remove them.",
            "Fast exact defense replaces per-candidate batched VJP by one backward from the sum of candidate logits; its aggregate contribution and top-negative map are mathematically unchanged.",
        ],
    }
    target = FOLLOWUP_DIR / "component_defense_latency_mps.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(run_benchmark())
