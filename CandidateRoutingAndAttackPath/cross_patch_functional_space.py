from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .common import StorageBudget, stable_hash
from .followup_common import write_summary
from .shared_functional_space import (
    ROOT,
    SharedFunctionalSpaceConfig,
    _analytic_rows,
    _analytic_summary,
    _build_bases,
    _causal_rows,
    _causal_summary,
    _collect_records,
)


DEFAULT_TARGET_PATCH = (
    ROOT
    / "server_artifacts"
    / "protocol_500x1000_160_tl_seed7"
    / "final_surfaces"
    / "general_adv.png"
)
OUTPUT_ROOT = ROOT / "cross_patch_functional_space_outputs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mps_audit(requested_device: str, model) -> dict:
    import torch

    model_device = str(next(model.parameters()).device)
    audit = {
        "requested_device": str(requested_device),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "model_parameter_device": model_device,
    }
    if requested_device == "mps":
        if not audit["mps_available"]:
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False")
        if not model_device.startswith("mps"):
            raise RuntimeError(f"MPS was requested but model parameters are on {model_device}")
    return audit


def run_cross_patch(
    target_patch: str | Path = DEFAULT_TARGET_PATCH,
    *,
    device: str = "mps",
    examples_per_group: int = 100,
    path_steps: int = 3,
    train_fraction: float = 0.60,
    smoke: bool = False,
) -> Path:
    target_patch = Path(target_patch).expanduser().resolve()
    if not target_patch.is_file():
        raise FileNotFoundError(target_patch)
    if not smoke and device != "mps":
        raise RuntimeError("A full cross-patch run must use --device mps; CPU is allowed only with --smoke")

    config = SharedFunctionalSpaceConfig(
        output_dir=str(OUTPUT_ROOT),
        device=str(device),
        require_device=(device == "mps"),
        examples_per_group=2 if smoke else int(examples_per_group),
        train_fraction=float(train_fraction),
        path_steps=1 if smoke else int(path_steps),
        ranks=(1, 2, 4) if smoke else (1, 2, 4, 8, 16, 32),
        causal_ranks=(1, 4) if smoke else (1, 4, 16, 32),
        method_version=9001 if smoke else 1,
    )
    started = time.time()
    StorageBudget(config.output_dir, config.max_output_gb).check()

    (
        selected,
        frames,
        records,
        exp,
        model,
        detect,
        cache,
        cache_path,
    ) = _collect_records(config, test_patch_path=target_patch)
    device_audit = _mps_audit(device, model)
    bases = _build_bases(records, config)
    analytic_rows = _analytic_rows(records, bases, config)
    analytic_summary = _analytic_summary(analytic_rows)
    causal_rows = _causal_rows(
        selected,
        records,
        frames,
        bases,
        exp,
        model,
        detect,
        cache,
        config,
        test_patch_path=target_patch,
    )
    causal_summary = _causal_summary(causal_rows)

    source_patch = Path(exp.config.attack.patch_path).expanduser().resolve()
    payload = {
        **asdict(config),
        "source_patch_sha256": _sha256(source_patch),
        "target_patch_sha256": _sha256(target_patch),
        "example_ids": selected.example_id.tolist(),
    }
    run_dir = Path(config.output_dir) / f"cross_patch_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(run_dir / "split.csv", index=False)
    analytic_rows.to_csv(run_dir / "analytic_rows.csv", index=False)
    analytic_summary.to_csv(run_dir / "analytic_summary.csv", index=False)
    causal_rows.to_csv(run_dir / "causal_rows.csv", index=False)
    causal_summary.to_csv(run_dir / "causal_summary.csv", index=False)

    baseline = causal_summary[
        causal_summary.family.eq("none") & causal_summary.setting.eq("none")
    ].iloc[0]
    pooled = causal_summary[
        causal_summary.family.eq("attack")
        & causal_summary.setting.eq("pooled_all")
        & causal_summary["rank"].eq(max(config.causal_ranks))
    ].iloc[0]
    pooled_sensitivity = causal_summary[
        causal_summary.family.eq("sensitivity")
        & causal_summary.setting.eq("pooled_all")
        & causal_summary["rank"].eq(max(config.causal_ranks))
    ].iloc[0]
    analytic_pooled = analytic_summary[
        analytic_summary.setting.eq("pooled_all")
        & analytic_summary["rank"].eq(max(config.ranks))
    ]
    summary = {
        "status": "complete",
        "elapsed_seconds": time.time() - started,
        "n_examples": int(selected.example_id.nunique()),
        "n_train_source_patch": int(selected.split.eq("train").sum()),
        "n_test_target_patch": int(selected.split.eq("test").sum()),
        "source_patch": {
            "path": str(source_patch),
            "sha256": _sha256(source_patch),
            "role": "basis training on train images",
        },
        "target_patch": {
            "path": str(target_patch),
            "sha256": _sha256(target_patch),
            "role": "unseen-patch evaluation on holdout images",
        },
        "cache_path": str(cache_path),
        "device_audit": device_audit,
        "config": asdict(config),
        "target_patch_baseline_hidden": int(baseline.n_hidden_baseline),
        "target_patch_baseline_hidden_rate": float(baseline.target_hidden_rate),
        "source_attack_basis_rank_max": pooled.to_dict(),
        "source_sensitivity_basis_rank_max": pooled_sensitivity.to_dict(),
        "pooled_rank_max_analytic": analytic_pooled.to_dict("records"),
        "validity_rule": (
            "Cross-patch transfer is supported when a basis fitted only with the source patch "
            "captures target-patch holdout Jacobians/effects and causally repairs target-patch "
            "hidden endpoints above the same-rank random basis."
        ),
        "important_scope": (
            "The source and target surfaces share the 160x160 top-left protocol and detector. "
            "This tests cross-patch, not cross-model or cross-position transfer."
        ),
    }
    write_summary(run_dir / "summary.json", summary)
    write_summary(run_dir / "device_audit.json", device_audit)
    (Path(config.output_dir) / "LATEST.txt").write_text(
        str(run_dir.resolve()) + "\n", encoding="utf-8"
    )
    StorageBudget(config.output_dir, config.max_output_gb).check()
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a functional basis on the original patch and test it on an unseen "
            "fixed-corner surface."
        )
    )
    parser.add_argument("--target-patch", type=Path, default=DEFAULT_TARGET_PATCH)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--examples-per-group", type=int, default=100)
    parser.add_argument("--path-steps", type=int, default=3)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(
        run_cross_patch(
            args.target_patch,
            device=args.device,
            examples_per_group=args.examples_per_group,
            path_steps=args.path_steps,
            train_fraction=args.train_fraction,
            smoke=args.smoke,
        )
    )
