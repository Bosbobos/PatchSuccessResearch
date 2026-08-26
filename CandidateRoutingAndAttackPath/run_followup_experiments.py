from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .attack_direction import AttackOracleConfig, run_attack_oracle
from .candidate_reserve import CandidateReserveConfig, run_candidate_reserve
from .defense_direction import RoutePoolConfig, run_route_pool
from .mechanism_followup import MechanismFollowupConfig, run_mechanism_followups
from .shared_candidate_mechanism import (
    SharedCandidateMechanismConfig,
    run_shared_candidate_mechanism,
)
from .score_functional_subspace import (
    ScoreFunctionalSubspaceConfig,
    run_score_functional_subspace,
)
from .full_success_closure import FullSuccessClosureConfig, run_full_success_closure
from .self_counterfactual_defense import (
    SelfCounterfactualDefenseConfig,
    run_self_counterfactual_defense,
)
from .single_forward_component import (
    SingleForwardComponentConfig,
    run_single_forward_component,
)
from .autonomous_negative_repair import (
    AutonomousNegativeRepairConfig,
    run_autonomous_negative_repair,
)


def _device_check(device: str) -> None:
    if device != "mps":
        return
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available in this Python process. Run the script from the IAD environment "
            "in a normal Terminal session and verify `torch.backends.mps.is_available()` first."
        )


def run_mechanism(device: str) -> Path:
    _device_check(device)
    return run_mechanism_followups(MechanismFollowupConfig(
        device=device,
        require_device=(device == "mps"),
        branch_examples_per_group=100,
        path_examples_per_group=100,
        alphas=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
        method_version=3,
    ))


def run_attack(device: str) -> Path:
    _device_check(device)
    return run_attack_oracle(AttackOracleConfig(
        device=device,
        require_device=(device == "mps"),
        examples_per_group=25,
        steps=20,
        checkpoints=tuple(range(0, 21)),
        step_fraction=0.02,
        epsilon_actual_l2_fraction=1.0,
        method_version=2,
    ))


def run_defense() -> Path:
    return run_route_pool(RoutePoolConfig(
        balanced_only=False,
        cluster_iou=0.70,
        train_fraction=0.60,
        method_version=2,
    ))


def run_reserve(device: str) -> Path:
    _device_check(device)
    return run_candidate_reserve(CandidateReserveConfig(
        device=device,
        require_device=(device == "mps"),
        examples_per_group=100,
        budgets=(1, 2, 4, 6, 8, 9, 10, 12),
        method_version=2,
    ))


def run_shared(device: str) -> Path:
    _device_check(device)
    return run_shared_candidate_mechanism(SharedCandidateMechanismConfig(
        device=device,
        require_device=(device == "mps"),
        examples_per_group=100,
        window_radius=2,
        ranks=(1, 2, 4),
        random_energy_controls=3,
        method_version=1,
    ))


def run_score_subspace(device: str) -> Path:
    _device_check(device)
    return run_score_functional_subspace(ScoreFunctionalSubspaceConfig(
        device=device,
        require_device=(device == "mps"),
        examples_per_group=25,
        path_steps=5,
        window_radius=2,
        ranks=(1, 2, 4),
        random_energy_controls=3,
        method_version=1,
    ))


def run_full_success(device: str) -> Path:
    _device_check(device)
    return run_full_success_closure(FullSuccessClosureConfig(
        device=device,
        require_device=(device == "mps"),
        closure_examples_per_group=100,
        functional_examples_per_group=25,
        path_steps=5,
        radii=(0, 1, 2, 4, 8, 16),
        method_version=1,
    ))


def run_full_success_expanded(device: str) -> Path:
    """Repeat the joint functional experiment on the full balanced cohort.

    The original run computed exact/spatial closure for 400 examples but the
    expensive score+geometry row-space intervention for only 25 examples per
    group (100 total, 45 trace-consistent hidden endpoints).  This configuration
    keeps the method fixed and expands that functional subset to all 400
    examples, so the key repair/transplant claim is evaluated on roughly four
    times as many hidden endpoints.
    """

    _device_check(device)
    return run_full_success_closure(FullSuccessClosureConfig(
        device=device,
        require_device=(device == "mps"),
        closure_examples_per_group=100,
        functional_examples_per_group=100,
        path_steps=5,
        radii=(0, 1, 2, 4, 8, 16),
        method_version=3,
    ))


def run_self_counterfactual(device: str) -> Path:
    _device_check(device)
    return run_self_counterfactual_defense(SelfCounterfactualDefenseConfig(
        device=device,
        require_device=(device == "mps"),
        examples_per_group=25,
        path_steps=5,
        method_version=4,
    ))


def run_self_counterfactual_weak(device: str) -> Path:
    _device_check(device)
    return run_self_counterfactual_defense(SelfCounterfactualDefenseConfig(
        device=device,
        require_device=(device == "mps"),
        examples_per_group=25,
        path_steps=5,
        proxy_strength=0.50,
        repair_scales=(1.0, 2.0),
        method_version=3,
    ))


def run_self_counterfactual_blind(device: str) -> Path:
    _device_check(device)
    return run_self_counterfactual_defense(SelfCounterfactualDefenseConfig(
        device=device,
        require_device=(device == "mps"),
        examples_per_group=25,
        path_steps=5,
        blind_search=True,
        blind_coarse_size=192,
        blind_coarse_stride=160,
        blind_top_coarse=2,
        blind_top_refined=5,
        blind_refine_sizes=(128, 160, 192),
        blind_scan_batch_size=24,
        method_version=5,
    ))


def run_single_forward(device: str) -> Path:
    _device_check(device)
    return run_single_forward_component(SingleForwardComponentConfig(
        device=device,
        require_device=(device == "mps"),
        examples_per_group=50,
        reference_examples_per_group=25,
        method_version=1,
    ))


def run_autonomous_repair(device: str) -> Path:
    _device_check(device)
    return run_autonomous_negative_repair(AutonomousNegativeRepairConfig(
        device=device,
        require_device=(device == "mps"),
        examples_per_group=50,
        reference_examples_per_group=25,
        clean_evaluation_examples=100,
        top_negative_k=(250, 500, 1000),
        cluster_selection_k=500,
        repair_top_clusters=(1, 2),
        cluster_ranking="reserve_tension",
        method_version=5,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run expanded follow-up experiments and save compact tables for result notebooks."
    )
    parser.add_argument(
        "experiment",
        choices=(
            "mechanism", "attack", "defense", "reserve", "shared", "score_subspace",
            "full_success", "full_success_expanded", "self_counterfactual",
            "self_counterfactual_weak", "self_counterfactual_blind",
            "single_forward_component", "autonomous_negative_repair", "all",
        ),
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    started = time.time()
    outputs: dict[str, str] = {}
    if args.experiment in ("mechanism", "all"):
        outputs["mechanism"] = str(run_mechanism(args.device))
    if args.experiment in ("attack", "all"):
        outputs["attack"] = str(run_attack(args.device))
    if args.experiment in ("defense", "all"):
        outputs["defense"] = str(run_defense())
    if args.experiment in ("reserve", "all"):
        outputs["reserve"] = str(run_reserve(args.device))
    if args.experiment in ("shared", "all"):
        outputs["shared"] = str(run_shared(args.device))
    if args.experiment in ("score_subspace", "all"):
        outputs["score_subspace"] = str(run_score_subspace(args.device))
    if args.experiment in ("full_success", "all"):
        outputs["full_success"] = str(run_full_success(args.device))
    if args.experiment == "full_success_expanded":
        outputs["full_success_expanded"] = str(run_full_success_expanded(args.device))
    if args.experiment == "self_counterfactual":
        outputs["self_counterfactual"] = str(run_self_counterfactual(args.device))
    if args.experiment == "self_counterfactual_weak":
        outputs["self_counterfactual_weak"] = str(run_self_counterfactual_weak(args.device))
    if args.experiment == "self_counterfactual_blind":
        outputs["self_counterfactual_blind"] = str(run_self_counterfactual_blind(args.device))
    if args.experiment == "single_forward_component":
        outputs["single_forward_component"] = str(run_single_forward(args.device))
    if args.experiment == "autonomous_negative_repair":
        outputs["autonomous_negative_repair"] = str(run_autonomous_repair(args.device))
    print(json.dumps({
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "outputs": outputs,
    }, indent=2))


if __name__ == "__main__":
    main()
