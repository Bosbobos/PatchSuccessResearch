from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


LOCAL_TO_REMOTE = {
    "person_adv_patch": "class0_score",
    "general_adv_patch": "allclass_score",
    "person_dpatch": "class0_joint",
    "general_dpatch": "allclass_joint",
}


def gpu_state() -> list[dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, utilization, memory_free = [
            int(value.strip()) for value in line.split(",")
        ]
        rows.append(
            {
                "index": index,
                "utilization": utilization,
                "memory_free_mib": memory_free,
            }
        )
    return rows


def command_for(args: argparse.Namespace, variant: str) -> list[str]:
    return [
        str(args.python),
        str(args.training_script),
        "--variant",
        LOCAL_TO_REMOTE[variant],
        "--dataset-dir",
        str(args.dataset_dir),
        "--split-root",
        str(args.split_root),
        "--output-root",
        str(args.output_root),
        "--weights",
        str(args.weights),
        "--device",
        "cuda:0",
        "--steps",
        "1000",
        "--train-images",
        "500",
        "--eval-images",
        str(args.eval_images),
        "--patch-size",
        "160",
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--topk",
        str(args.topk),
        "--box-weight",
        str(args.box_weight),
        "--duty-cycle",
        str(args.duty_cycle),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resource-aware four-view scheduler.")
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--training-script", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--eval-images", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--box-weight", type=float, default=1.0)
    parser.add_argument("--duty-cycle", type=float, default=0.20)
    parser.add_argument("--max-existing-utilization", type=int, default=80)
    parser.add_argument("--min-free-memory-mib", type=int, default=20000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    variants = list(LOCAL_TO_REMOTE)
    pending = [
        variant
        for variant in variants
        if not (
            args.output_root / LOCAL_TO_REMOTE[variant] / "completed.json"
        ).exists()
    ]
    running: dict[str, dict] = {}
    events_path = args.log_root / "scheduler_events.jsonl"

    def event(kind: str, **payload: object) -> None:
        row = {"time": time.time(), "event": kind, **payload}
        with events_path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    event("started", pending=[LOCAL_TO_REMOTE[value] for value in pending])
    while pending or running:
        for variant, record in list(running.items()):
            returncode = record["process"].poll()
            if returncode is None:
                continue
            record["log"].close()
            event(
                "finished",
                variant=LOCAL_TO_REMOTE[variant],
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
        available.sort(
            key=lambda row: (
                row["utilization"],
                -row["memory_free_mib"],
            )
        )
        while pending and available:
            variant = pending.pop(0)
            gpu = available.pop(0)["index"]
            remote_name = LOCAL_TO_REMOTE[variant]
            log_path = args.log_root / f"{remote_name}.log"
            log_handle = log_path.open("a")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command_for(args, variant),
                cwd=args.training_script.parent,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[variant] = {
                "process": process,
                "gpu": gpu,
                "log": log_handle,
            }
            event(
                "launched",
                variant=remote_name,
                gpu=gpu,
                pid=process.pid,
                duty_cycle=args.duty_cycle,
            )

        if pending or running:
            time.sleep(args.poll_seconds)
    event("complete")


if __name__ == "__main__":
    main()
