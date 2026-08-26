from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path


LOCAL_TO_REMOTE_VARIANT = {
    "depatch": "surface_dropout",
    "robust_dpatch": "surface_transform",
    "adversarial_patch": "surface_baseline",
}


def gpu_state() -> list[dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows: list[dict[str, int]] = []
    for raw in result.stdout.splitlines():
        index, utilization, memory_free = [int(value.strip()) for value in raw.split(",")]
        rows.append(
            {
                "index": index,
                "utilization": utilization,
                "memory_free_mib": memory_free,
            }
        )
    return rows


def command_for(
    args: argparse.Namespace,
    *,
    variant: str,
    gpu: int,
) -> list[str]:
    return [
        str(args.python),
        str(args.training_script),
        "--variant",
        LOCAL_TO_REMOTE_VARIANT[variant],
        "--dataset-dir",
        str(args.dataset_dir),
        "--split-root",
        str(args.split_root),
        "--output-root",
        str(args.output_root),
        "--weights",
        args.weights,
        "--device",
        "cuda:0",
        "--steps",
        str(args.steps),
        "--train-images",
        str(args.train_images),
        "--eval-images",
        str(args.eval_images),
        "--patch-size",
        "160",
        "--batch-size",
        str(args.batch_size),
        "--duty-cycle",
        str(args.duty_cycle),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resource-aware scheduler for surface optimisation.")
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--weights", default="yolo11s.pt")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--train-images", type=int, default=500)
    parser.add_argument("--eval-images", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--duty-cycle", type=float, default=0.20)
    parser.add_argument(
        "--max-existing-utilization",
        type=int,
        default=80,
        help="Do not place a new process above this pre-launch GPU utilization.",
    )
    parser.add_argument("--min-free-memory-mib", type=int, default=20000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    variants = list(LOCAL_TO_REMOTE_VARIANT)
    pending = [
        variant
        for variant in variants
        if not (
            args.output_root
            / LOCAL_TO_REMOTE_VARIANT[variant]
            / "completed.json"
        ).exists()
    ]
    running: dict[str, dict] = {}
    events_path = args.log_root / "scheduler_events.jsonl"

    def event(kind: str, **payload: object) -> None:
        row = {"time": time.time(), "event": kind, **payload}
        with events_path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    event("started", pending=pending)
    while pending or running:
        for variant, record in list(running.items()):
            returncode = record["process"].poll()
            if returncode is None:
                continue
            record["log"].close()
            event(
                "finished",
                variant=LOCAL_TO_REMOTE_VARIANT[variant],
                gpu=record["gpu"],
                returncode=returncode,
            )
            del running[variant]
            if returncode != 0:
                pending.append(variant)

        occupied = {record["gpu"] for record in running.values()}
        available = [
            row
            for row in gpu_state()
            if row["index"] not in occupied
            and row["utilization"] <= args.max_existing_utilization
            and row["memory_free_mib"] >= args.min_free_memory_mib
        ]
        available.sort(key=lambda row: (row["utilization"], -row["memory_free_mib"]))
        while pending and available:
            variant = pending.pop(0)
            gpu = available.pop(0)["index"]
            log_path = args.log_root / f"{LOCAL_TO_REMOTE_VARIANT[variant]}.log"
            log_handle = log_path.open("a")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = command_for(args, variant=variant, gpu=gpu)
            process = subprocess.Popen(
                command,
                cwd=args.training_script.parent,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[variant] = {
                "process": process,
                "gpu": gpu,
                "log": log_handle,
                "log_path": str(log_path),
            }
            event(
                "launched",
                variant=LOCAL_TO_REMOTE_VARIANT[variant],
                gpu=gpu,
                pid=process.pid,
                command=command,
                duty_cycle=args.duty_cycle,
            )

        if pending or running:
            time.sleep(args.poll_seconds)
    event("complete")


if __name__ == "__main__":
    main()
