from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: str | Path, *, n_hex: int = 16) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[: int(n_hex)]


def stable_hash(payload: dict[str, Any], *, n_hex: int = 16) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[: int(n_hex)]


def list_image_paths(root: str | Path | Iterable[str | Path]) -> list[Path]:
    if not isinstance(root, (str, Path)):
        out: list[Path] = []
        seen: set[str] = set()
        for item in root:
            for path in list_image_paths(item):
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    out.append(path)
        return sorted(out)
    root = Path(root).expanduser()
    if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
        return [root]
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def dataset_manifest(paths: list[Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        stat = path.stat()
        rows.append({"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return {"count": len(rows), "files": rows}


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
