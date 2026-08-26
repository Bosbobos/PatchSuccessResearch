from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "CandidateRoutingAndAttackPath" / "outputs"
# The existing patch-success cache is about 11 GiB. Keeping this experiment under
# 8 GiB keeps the combined analysis artifacts below the user's 20 GiB ceiling.
DEFAULT_MAX_OUTPUT_GB = 8.0


def ensure_import_paths(repo_root: str | Path = REPO_ROOT) -> Path:
    root = Path(repo_root).resolve()
    for candidate in (root, root / "new_experiments"):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)
    return root


def stable_hash(payload: Any, *, length: int = 16) -> str:
    if is_dataclass(payload):
        payload = asdict(payload)
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[: int(length)]


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for item in root.rglob("*"):
        try:
            if item.is_file():
                total += int(item.stat().st_size)
        except FileNotFoundError:
            continue
    return total


def output_size_gb(path: str | Path = DEFAULT_OUTPUT_DIR) -> float:
    return directory_size_bytes(path) / float(1024**3)


class StorageBudget:
    def __init__(self, root: str | Path, max_gb: float = DEFAULT_MAX_OUTPUT_GB):
        self.root = Path(root)
        self.max_bytes = int(float(max_gb) * 1024**3)
        if self.max_bytes <= 0:
            raise ValueError("max_gb must be positive")

    def check(self, *, extra_bytes: int = 0) -> None:
        used = directory_size_bytes(self.root)
        projected = used + max(0, int(extra_bytes))
        if projected > self.max_bytes:
            raise RuntimeError(
                "Experiment output budget exceeded: "
                f"used={used / 1024**3:.3f} GiB, "
                f"projected={projected / 1024**3:.3f} GiB, "
                f"limit={self.max_bytes / 1024**3:.3f} GiB."
            )


def connect_db(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def upsert_metadata(conn: sqlite3.Connection, values: dict[str, Any]) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)"
    )
    conn.executemany(
        "INSERT OR REPLACE INTO metadata(key, value_json) VALUES (?, ?)",
        [(str(key), json.dumps(value, ensure_ascii=False, default=str)) for key, value in values.items()],
    )
    conn.commit()


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return target


def write_markdown(path: str | Path, lines: Iterable[str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(str(line) for line in lines).rstrip() + "\n", encoding="utf-8")
    return target


def load_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    prefer_dataset: str = "COCO_people",
    prefer_device: str = "mps",
    require_device: bool = False,
):
    """Reuse the largest existing patch-success cache without copying it."""

    root = ensure_import_paths(repo_root)
    from CausalTracingViaPatching.causal_patching import load_existing_experiment

    return load_existing_experiment(
        repo_root=root,
        prefer_dataset=prefer_dataset,
        prefer_device=prefer_device,
        require_device=require_device,
    )


def release_accelerator_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "patch_success_matplotlib"))
