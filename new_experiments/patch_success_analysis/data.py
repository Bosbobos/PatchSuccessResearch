from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class AttackConfig:
    dataset_path: str | list[str] | tuple[str, ...] = "datasets/inria_test"
    patch_path: str = "data/patch.png"
    model_path: str = "yolo11s.pt"
    output_dir: str = "new_experiments/outputs/patch_success_analysis"
    imgsz: int = 640
    device: str | None = None
    target_class_name: str | None = "person"
    conf: float = 0.001
    success_thresh: float = 0.30
    seed: int = 17
    pool_size: int = 500
    n_success: int = 100
    n_fail: int = 100
    inference_batch_size: int = 16
    show_progress: bool = True
    apply_patch_after_letterbox: bool = True
    patch_xy: tuple[int, int] = (0, 0)

    def cache_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("inference_batch_size", None)
        payload.pop("show_progress", None)
        payload["dataset_path"] = normalize_dataset_paths(self.dataset_path)
        payload["patch_path"] = str(Path(self.patch_path).expanduser())
        payload["model_path"] = str(Path(self.model_path).expanduser())
        return payload

    def cache_key(self) -> str:
        raw = json.dumps(self.cache_payload(), sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    @property
    def cache_dir(self) -> Path:
        return Path(self.output_dir) / "cache"

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / f"attack_pool_{self.cache_key()}.pkl"

    @property
    def summary_path(self) -> Path:
        return self.cache_dir / f"attack_pool_{self.cache_key()}_summary.json"


@dataclass(slots=True)
class AttackExample:
    path: str
    conf_clean: float
    conf_patch: float
    drop: float
    success: bool
    patch_bbox_lb: tuple[float, float, float, float] | None
    target_class_id: int | None
    clean_detection: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AttackCache:
    config: dict[str, Any]
    cache_key: str
    examples: list[AttackExample]
    pool_summary: dict[str, Any]

    @property
    def successes(self) -> list[AttackExample]:
        return [item for item in self.examples if item.success]

    @property
    def failures(self) -> list[AttackExample]:
        return [item for item in self.examples if not item.success]

    def to_summary(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "config": self.config,
            "balanced_total": len(self.examples),
            "balanced_success": len(self.successes),
            "balanced_fail": len(self.failures),
            **self.pool_summary,
        }


def normalize_dataset_paths(roots: str | Path | Iterable[str | Path]) -> list[str]:
    if isinstance(roots, (str, Path)):
        items = [roots]
    else:
        items = list(roots)
    return [str(Path(item).expanduser()) for item in items]


def list_image_paths(root: str | Path | Iterable[str | Path]) -> list[Path]:
    if not isinstance(root, (str, Path)):
        paths: list[Path] = []
        seen: set[str] = set()
        for item in root:
            for path in list_image_paths(item):
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    paths.append(path)
        return sorted(paths)
    root = Path(root).expanduser()
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def sample_image_paths(paths: Iterable[str | Path], *, n: int, seed: int) -> list[Path]:
    import numpy as np

    items = [Path(p) for p in paths]
    if not items:
        raise RuntimeError("No image paths were provided.")
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(items))
    rng.shuffle(order)
    keep = order[: min(int(n), len(items))]
    return [items[int(i)] for i in keep]


def save_attack_cache(cache: AttackCache, config: AttackConfig) -> None:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    with config.cache_path.open("wb") as f:
        pickle.dump(cache, f)
    config.summary_path.write_text(json.dumps(cache.to_summary(), indent=2, ensure_ascii=False))


def load_attack_cache(config: AttackConfig) -> AttackCache | None:
    if not config.cache_path.exists():
        return None
    with config.cache_path.open("rb") as f:
        cache = pickle.load(f)
    if not isinstance(cache, AttackCache):
        raise TypeError(f"Unexpected cache object in {config.cache_path}: {type(cache)}")
    if cache.cache_key != config.cache_key():
        return None
    return cache


def build_attack_cache(config: AttackConfig, *, force: bool = False) -> AttackCache:
    cached = None if force else load_attack_cache(config)
    if cached is not None:
        return cached

    import numpy as np
    from PIL import Image

    from .patching import build_clean_and_patched_letterboxed
    from .yolo import detection_dict_from_result, get_class_id, load_yolo, yolo_predict_conf_batch

    yolo = load_yolo(config.model_path, device=config.device)
    names = getattr(yolo, "names", getattr(getattr(yolo, "model", None), "names", {}))
    target_class_id = None
    if config.target_class_name is not None:
        target_class_id = get_class_id(names, config.target_class_name)

    image_paths = sample_image_paths(
        list_image_paths(config.dataset_path),
        n=int(config.pool_size),
        seed=int(config.seed),
    )
    patch_pil = Image.open(config.patch_path).convert("RGB")

    pool: list[AttackExample] = []
    failed_paths: list[str] = []
    batch_size = max(1, int(getattr(config, "inference_batch_size", 16)))
    show_progress = bool(getattr(config, "show_progress", True))
    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(total=len(image_paths), desc="Patch success/fail", unit="img")
        except Exception:
            progress = None
    success_count = 0
    fail_count = 0

    def predict_records(records: list[tuple[Path, Any, Any, Any]], effective_batch_size: int):
        try:
            clean_confs, clean_results = yolo_predict_conf_batch(
                yolo,
                [record[1] for record in records],
                imgsz=int(config.imgsz),
                target_class_id=target_class_id,
                conf=float(config.conf),
                device=config.device,
                batch_size=effective_batch_size,
            )
            patch_confs, _patch_results = yolo_predict_conf_batch(
                yolo,
                [record[2] for record in records],
                imgsz=int(config.imgsz),
                target_class_id=target_class_id,
                conf=float(config.conf),
                device=config.device,
                batch_size=effective_batch_size,
            )
            return list(zip(records, clean_confs, patch_confs, clean_results, strict=True))
        except Exception as exc:  # noqa: BLE001 - retry smaller batches before dropping images.
            if len(records) <= 1:
                failed_paths.append(f"{records[0][0]}: {type(exc).__name__}: {exc}")
                return []
            mid = len(records) // 2
            next_batch_size = max(1, min(effective_batch_size // 2, mid))
            return predict_records(records[:mid], next_batch_size) + predict_records(records[mid:], next_batch_size)

    try:
        for start in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[start : start + batch_size]
            chunk_records: list[tuple[Path, Any, Any, Any]] = []
            for path in chunk_paths:
                try:
                    with Image.open(path) as image:
                        base_pil = image.convert("RGB")
                    clean_lb, patched_lb, patch_bbox = build_clean_and_patched_letterboxed(base_pil, patch_pil, config)
                    chunk_records.append((path, clean_lb, patched_lb, patch_bbox))
                except Exception as exc:  # noqa: BLE001 - cache build should continue over bad images.
                    failed_paths.append(f"{path}: {type(exc).__name__}: {exc}")
            if not chunk_records:
                if progress is not None:
                    progress.update(len(chunk_paths))
                continue

            for (path, _clean_lb, _patched_lb, patch_bbox), conf_clean, conf_patch, res_clean in predict_records(
                chunk_records,
                batch_size,
            ):
                clean_detection = detection_dict_from_result(res_clean, target_class_id=target_class_id)
                if clean_detection is None:
                    failed_paths.append(f"{path}: no clean detection")
                    continue
                drop = float(conf_clean - conf_patch)
                success = bool(drop > float(config.success_thresh))
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                pool.append(
                    AttackExample(
                        path=str(path),
                        conf_clean=float(conf_clean),
                        conf_patch=float(conf_patch),
                        drop=drop,
                        success=success,
                        patch_bbox_lb=tuple(float(v) for v in patch_bbox) if patch_bbox is not None else None,
                        target_class_id=target_class_id,
                        clean_detection=clean_detection,
                    )
                )
            if progress is not None:
                progress.update(len(chunk_paths))
                progress.set_postfix(
                    success=success_count,
                    fail=fail_count,
                    bad=len(failed_paths),
                )
    finally:
        if progress is not None:
            progress.close()

    rng = np.random.default_rng(int(config.seed))
    successes = [item for item in pool if item.success]
    failures = [item for item in pool if not item.success]
    rng.shuffle(successes)
    rng.shuffle(failures)

    balanced = successes[: int(config.n_success)] + failures[: int(config.n_fail)]
    rng.shuffle(balanced)
    cache = AttackCache(
        config=config.cache_payload(),
        cache_key=config.cache_key(),
        examples=balanced,
        pool_summary={
            "pool_total": len(pool),
            "pool_success": len(successes),
            "pool_fail": len(failures),
            "pool_failed_images": len(failed_paths),
            "failed_paths": failed_paths[:50],
        },
    )
    save_attack_cache(cache, config)
    return cache
