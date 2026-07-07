from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .utils import dataset_manifest, list_image_paths, sha256_file, stable_hash


@dataclass(slots=True)
class ClassifierAttackConfig:
    dataset_path: str = "datasets/COCO_people_224"
    patch_path: str = "data/cls_patch.png"
    checkpoint_path: str = "data/yolo11-cls-person.pt"
    base_weights_path: str = "data/yolo11s-cls.pt"
    output_dir: str = "classifier_experiments/outputs/classifier_patch_analysis"
    img_size: int = 224
    device: str | None = "auto"
    success_drop_threshold: float = 0.5
    inference_batch_size: int = 64
    show_progress: bool = True
    seed: int = 17
    patch_xy: tuple[int, int] = (0, 0)
    overlay_policy: str = "fixed_top_left"
    method_version: int = 1

    @property
    def cache_dir(self) -> Path:
        out = Path(self.output_dir) / "cache"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def cache_payload(self) -> dict[str, Any]:
        paths = list_image_paths(self.dataset_path)
        payload = asdict(self)
        payload.pop("inference_batch_size", None)
        payload.pop("show_progress", None)
        payload["dataset_manifest"] = dataset_manifest(paths)
        for key in ("patch_path", "checkpoint_path", "base_weights_path"):
            path = Path(payload[key])
            payload[f"{key}_sha256"] = sha256_file(path) if path.exists() else "MISSING"
            payload[key] = str(path)
        return payload

    def cache_key(self) -> str:
        return stable_hash(self.cache_payload())

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / f"classifier_attack_cache_{self.cache_key()}.pkl"

    @property
    def summary_path(self) -> Path:
        return self.cache_dir / f"classifier_attack_cache_{self.cache_key()}_summary.json"


@dataclass(slots=True)
class ClassifierAttackExample:
    path: str
    clean_logit: float
    patched_logit: float
    conf_clean: float
    conf_patch: float
    drop: float
    success: bool
    patch_bbox: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClassifierAttackCache:
    config: dict[str, Any]
    cache_key: str
    examples: list[ClassifierAttackExample]
    summary: dict[str, Any]

    @property
    def successes(self) -> list[ClassifierAttackExample]:
        return [item for item in self.examples if item.success]

    @property
    def failures(self) -> list[ClassifierAttackExample]:
        return [item for item in self.examples if not item.success]

    def to_summary(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "n_examples": len(self.examples),
            "n_success": len(self.successes),
            "n_fail": len(self.failures),
            "success_rate": len(self.successes) / max(1, len(self.examples)),
            **self.summary,
        }


def load_attack_cache(config: ClassifierAttackConfig) -> ClassifierAttackCache | None:
    path = config.cache_path
    if not path.exists():
        return None
    with path.open("rb") as fh:
        cache = pickle.load(fh)
    if not isinstance(cache, ClassifierAttackCache):
        raise TypeError(f"Unexpected cache object in {path}: {type(cache)}")
    return cache if cache.cache_key == config.cache_key() else None


def save_attack_cache(cache: ClassifierAttackCache, config: ClassifierAttackConfig) -> None:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    with config.cache_path.open("wb") as fh:
        pickle.dump(cache, fh)
    config.summary_path.write_text(json.dumps(cache.to_summary(), indent=2, ensure_ascii=False))


def build_or_load_attack_cache(config: ClassifierAttackConfig, *, force: bool = False) -> ClassifierAttackCache:
    cached = None if force else load_attack_cache(config)
    if cached is not None:
        return cached

    import torch
    from PIL import Image

    from .modeling import forward_logits, load_yolo_cls_model, preprocess_pil_batch
    from .patching import load_rgb_image, overlay_top_left_patch

    image_paths = list_image_paths(config.dataset_path)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {config.dataset_path}. Expected flat or recursive COCO_people_224 image folder."
        )
    patch_path = Path(config.patch_path)
    if not patch_path.exists():
        raise FileNotFoundError(f"Missing patch image: {patch_path}")

    model = load_yolo_cls_model(
        config.checkpoint_path,
        base_weights_path=config.base_weights_path,
        device=config.device,
    )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    patch = Image.open(patch_path).convert("RGB")
    batch_size = max(1, int(config.inference_batch_size))

    progress = None
    if config.show_progress:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(total=len(image_paths), desc="classifier attack cache", unit="img")
        except Exception:
            progress = None

    examples: list[ClassifierAttackExample] = []
    failed: list[str] = []
    for start in range(0, len(image_paths), batch_size):
        chunk = image_paths[start : start + batch_size]
        clean_images = []
        patched_images = []
        records = []
        for path in chunk:
            try:
                clean = load_rgb_image(path, img_size=int(config.img_size))
                patched, bbox = overlay_top_left_patch(clean, patch, xy=config.patch_xy)
                clean_images.append(clean)
                patched_images.append(patched)
                records.append((path, bbox))
            except Exception as exc:  # noqa: BLE001 - keep bad files from killing the whole cache build.
                failed.append(f"{path}: {type(exc).__name__}: {exc}")
        if records:
            clean_x = preprocess_pil_batch(clean_images, img_size=int(config.img_size), device=device, dtype=dtype)
            patched_x = preprocess_pil_batch(patched_images, img_size=int(config.img_size), device=device, dtype=dtype)
            with torch.no_grad():
                clean_logits = forward_logits(model, clean_x).detach().cpu()
                patched_logits = forward_logits(model, patched_x).detach().cpu()
                clean_conf = torch.sigmoid(clean_logits)
                patched_conf = torch.sigmoid(patched_logits)
            for idx, (path, bbox) in enumerate(records):
                conf_clean = float(clean_conf[idx])
                conf_patch = float(patched_conf[idx])
                drop = float(conf_clean - conf_patch)
                success = bool(drop > float(config.success_drop_threshold))
                examples.append(
                    ClassifierAttackExample(
                        path=str(path),
                        clean_logit=float(clean_logits[idx]),
                        patched_logit=float(patched_logits[idx]),
                        conf_clean=conf_clean,
                        conf_patch=conf_patch,
                        drop=drop,
                        success=success,
                        patch_bbox=tuple(int(v) for v in bbox),
                    )
                )
        if progress is not None:
            progress.update(len(chunk))
    if progress is not None:
        progress.close()

    if not examples:
        raise RuntimeError(f"Classifier attack cache is empty; failed files: {failed[:10]}")
    cache = ClassifierAttackCache(
        config=config.cache_payload(),
        cache_key=config.cache_key(),
        examples=examples,
        summary={"n_failed": len(failed), "failed_head": failed[:20], "unbalanced": True},
    )
    save_attack_cache(cache, config)
    return cache
