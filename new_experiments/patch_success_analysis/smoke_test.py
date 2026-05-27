from __future__ import annotations

from dataclasses import replace

from .data import AttackConfig
from .experiments import ExperimentConfig, PatchSuccessExperiment


def main() -> None:
    attack = replace(
        AttackConfig(),
        pool_size=20,
        n_success=2,
        n_fail=2,
    )
    exp = PatchSuccessExperiment(
        ExperimentConfig(
            attack=attack,
            n_steps=4,
            alpha_batch_size=2,
            ranking_top_ns=(500,),
            runtime_n_images=4,
        )
    )
    cache = exp.build_or_load_cache()
    if len(cache.successes) == 0 or len(cache.failures) == 0:
        raise RuntimeError("Smoke cache must contain at least one success and one failure.")
    subset = cache.successes[:1] + cache.failures[:1]
    comparison = exp.run_attribution_comparison(examples=subset, max_examples=2)
    propagation = exp.run_patch_propagation(examples=subset, max_examples=2, layer_names=[exp.config.target_layer])
    metrics = exp.run_success_failure_metrics(max_examples=2)
    if not comparison["figure_paths"]:
        raise RuntimeError("Attribution comparison did not produce figures.")
    if not propagation:
        raise RuntimeError("Patch propagation did not produce rows.")
    if not metrics["quality"]:
        raise RuntimeError("Success/failure metrics did not produce quality rows.")


if __name__ == "__main__":
    main()
