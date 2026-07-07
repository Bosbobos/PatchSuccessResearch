from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import (
    ClassifierAttackConfig,
    ClassifierAttackExample,
    build_or_load_attack_cache as _build_or_load_attack_cache,
    load_attack_cache,
)
from .utils import stable_hash


@dataclass(slots=True)
class ExperimentConfig:
    attack: ClassifierAttackConfig = field(default_factory=ClassifierAttackConfig)
    target_layer: str = "model.9"
    display_layers: tuple[str, ...] = tuple(f"model.{idx}" for idx in range(10))
    top_percent: float = 5.0
    method_version: int = 2


class ClassifierPatchExperiment:
    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config or ExperimentConfig()
        self.cache = None
        self.model = None

    @property
    def output_dir(self) -> Path:
        out = Path(self.config.attack.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out

    @property
    def figures_dir(self) -> Path:
        out = self.output_dir / "figures"
        out.mkdir(parents=True, exist_ok=True)
        return out

    @property
    def derived_cache_dir(self) -> Path:
        out = self.output_dir / "cache"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def build_or_load_cache(self, *, force: bool = False):
        self.cache = _build_or_load_attack_cache(self.config.attack, force=force)
        return self.cache

    def get_cache(self):
        if self.cache is None:
            self.cache = load_attack_cache(self.config.attack)
        if self.cache is None:
            self.cache = _build_or_load_attack_cache(self.config.attack)
        return self.cache

    def load_model(self):
        if self.model is None:
            from .modeling import load_yolo_cls_model

            self.model = load_yolo_cls_model(
                self.config.attack.checkpoint_path,
                base_weights_path=self.config.attack.base_weights_path,
                device=self.config.attack.device,
            )
        return self.model

    def all_display_layer_names(self) -> list[str]:
        return list(self.config.display_layers)

    def _images_for_example(self, example: ClassifierAttackExample):
        from PIL import Image

        from .patching import load_rgb_image, overlay_top_left_patch

        clean = load_rgb_image(example.path, img_size=int(self.config.attack.img_size))
        patch = Image.open(self.config.attack.patch_path).convert("RGB")
        patched, bbox = overlay_top_left_patch(clean, patch, xy=self.config.attack.patch_xy)
        return clean, patched, bbox

    def _preprocess(self, image):
        from .modeling import preprocess_pil_batch

        model = self.load_model()
        param = next(model.parameters())
        return preprocess_pil_batch([image], img_size=int(self.config.attack.img_size), device=param.device, dtype=param.dtype)

    def _layer_map_cache_path(self, example: ClassifierAttackExample, *, layer_name: str) -> Path:
        payload = {
            "attack_cache_key": self.get_cache().cache_key,
            "path": example.path,
            "drop": float(example.drop),
            "success": bool(example.success),
            "layer_name": layer_name,
            "img_size": int(self.config.attack.img_size),
            "patch_bbox": list(example.patch_bbox),
            "target": "clean_one_logit_person_score",
            "method_version": int(self.config.method_version),
        }
        key = stable_hash(payload)
        out = self.derived_cache_dir / "layer_maps" / layer_name.replace(".", "_")
        out.mkdir(parents=True, exist_ok=True)
        return out / f"layer_maps_{key}.npz"

    def compute_layer_map(self, example: ClassifierAttackExample, *, layer_name: str | None = None, force: bool = False):
        import torch

        from .activations import capture_activation, gradient_x_activation_importance

        layer_name = layer_name or self.config.target_layer
        path = self._layer_map_cache_path(example, layer_name=layer_name)
        if path.exists() and not force:
            cached = self._load_layer_map_cache(example, layer_name=layer_name)
            if cached is not None:
                return cached
            path.unlink(missing_ok=True)

        model = self.load_model()
        clean_img, patched_img, patch_bbox = self._images_for_example(example)
        clean_x = self._preprocess(clean_img)
        patched_x = self._preprocess(patched_img)
        with torch.no_grad():
            clean_act = capture_activation(model, clean_x, layer_name)
            patched_act = capture_activation(model, patched_x, layer_name)
        importance = gradient_x_activation_importance(model, clean_x, layer_name)
        delta = patched_act - clean_act
        metadata = {
            "path": example.path,
            "drop": float(example.drop),
            "success": bool(example.success),
            "conf_clean": float(example.conf_clean),
            "conf_patch": float(example.conf_patch),
            "layer": layer_name,
            "patch_bbox": tuple(int(v) for v in patch_bbox),
            "cache_key": self.get_cache().cache_key,
            "importance_target": "clean_one_logit_person_score",
        }
        np.savez_compressed(
            path,
            clean_activation_chw=clean_act[0].detach().cpu().numpy().astype("float32", copy=False),
            patched_activation_chw=patched_act[0].detach().cpu().numpy().astype("float32", copy=False),
            delta_chw=delta[0].detach().cpu().numpy().astype("float32", copy=False),
            importance_chw=importance[0].detach().cpu().numpy().astype("float32", copy=False),
            metadata=json.dumps(metadata, ensure_ascii=False),
        )
        return self._load_layer_map_cache(example, layer_name=layer_name)

    def _load_layer_map_cache(self, example: ClassifierAttackExample, *, layer_name: str):
        path = self._layer_map_cache_path(example, layer_name=layer_name)
        if not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=False)
            metadata = json.loads(str(data["metadata"]))
        except Exception:  # noqa: BLE001 - invalid/corrupt cache should be treated as missing.
            return None
        return {
            "clean_activation_chw": data["clean_activation_chw"],
            "patched_activation_chw": data["patched_activation_chw"],
            "delta_chw": data["delta_chw"],
            "importance_chw": data["importance_chw"],
            "metadata": metadata,
            "cache_path": str(path),
            "loaded_from_cache": True,
        }


def build_or_load_attack_cache(config: ClassifierAttackConfig, *, force: bool = False):
    return _build_or_load_attack_cache(config, force=force)
