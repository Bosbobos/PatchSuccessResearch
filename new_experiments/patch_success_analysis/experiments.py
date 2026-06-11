from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .data import AttackCache, AttackConfig, AttackExample, build_attack_cache, load_attack_cache


@dataclass(slots=True)
class ExperimentConfig:
    attack: AttackConfig = field(default_factory=AttackConfig)
    target_layer: str = "model.22"
    detect_layer: str = "model.23"
    target_mode: str = "class_only"
    n_steps: int = 64
    alpha_batch_size: int = 4
    smoothing_window: int = 15
    ranking_top_ns: tuple[int, ...] = (500, 2000, 8000, 50000)
    overlap_percentages: tuple[int, ...] = (1, 5, 10, 20, 50, 100)
    runtime_n_images: int = 100
    metrics_batch_size: int = 64


class PatchSuccessExperiment:
    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config or ExperimentConfig()
        self.cache: AttackCache | None = None
        self.yolo = None
        self.model = None

    @property
    def output_dir(self) -> Path:
        return Path(self.config.attack.output_dir)

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

    def build_or_load_cache(self, *, force: bool = False) -> AttackCache:
        self.cache = build_attack_cache(self.config.attack, force=force)
        return self.cache

    def get_cache(self) -> AttackCache:
        if self.cache is None:
            self.cache = load_attack_cache(self.config.attack)
        if self.cache is None:
            self.cache = build_attack_cache(self.config.attack)
        return self.cache

    def load_model(self):
        if self.yolo is None or self.model is None:
            from .yolo import get_torch_model, load_yolo

            self.yolo = load_yolo(self.config.attack.model_path, device=self.config.attack.device)
            self.model = get_torch_model(self.yolo)
            if self.config.attack.device is not None:
                self.model.to(self.config.attack.device)
            self.model.eval()
        return self.yolo, self.model

    @staticmethod
    def _release_batch_memory() -> None:
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

    def representative_layers(self, *, include_target: bool = True, max_layers: int = 8) -> list[str]:
        _yolo, model = self.load_model()
        names = self.all_display_layer_names()
        keep = names[-int(max_layers) :]
        if include_target and self.config.target_layer not in keep and self.config.target_layer != self.config.detect_layer:
            keep.append(self.config.target_layer)
        return keep

    def all_display_layer_names(self) -> list[str]:
        _yolo, model = self.load_model()
        names = [
            name
            for name, _module in model.named_modules()
            if name.startswith("model.")
            and "." not in name[len("model.") :]
            and name != self.config.detect_layer
        ]
        names = sorted(names, key=lambda item: int(item.split(".")[1]) if item.split(".")[1].isdigit() else 999)
        return names

    def _images_for_example(self, example: AttackExample):
        from PIL import Image

        from .patching import build_clean_and_patched_letterboxed

        base = Image.open(example.path).convert("RGB")
        patch = Image.open(self.config.attack.patch_path).convert("RGB")
        return build_clean_and_patched_letterboxed(base, patch, self.config.attack)

    def _preprocess(self, image):
        from segmentig_detector.yolo_utils import preprocess_pil

        yolo, model = self.load_model()
        pack = preprocess_pil(
            yolo,
            image,
            imgsz=int(self.config.attack.imgsz),
            conf=float(self.config.attack.conf),
            device=self.config.attack.device,
        )
        param = next(model.parameters())
        pack["im"] = pack["im"].to(device=param.device, dtype=param.dtype)
        return pack

    def _context_for_example(self, example: AttackExample, *, image_variant: str = "clean"):
        import torch

        from segmentig_detector.targets import select_fixed_detector_target

        from .attributions import detector_target_fn
        from .yolo import detection_choice_from_dict, detection_dict_from_result, yolo_predict_conf_scalar

        yolo, model = self.load_model()
        clean_lb, patched_lb, _patch_bbox = self._images_for_example(example)
        image = patched_lb if image_variant == "patched" else clean_lb
        pack = self._preprocess(image)
        inputs = pack["im"]
        baselines = torch.zeros_like(inputs)

        detection_dict = example.clean_detection
        if detection_dict is None:
            _conf, result = yolo_predict_conf_scalar(
                yolo,
                clean_lb,
                imgsz=int(self.config.attack.imgsz),
                target_class_id=example.target_class_id,
                conf=float(self.config.attack.conf),
                device=self.config.attack.device,
            )
            detection_dict = detection_dict_from_result(result, target_class_id=example.target_class_id)
        if detection_dict is None:
            raise RuntimeError(f"No clean detection is available for {example.path}")
        detection = detection_choice_from_dict(example.path, detection_dict)
        fixed_target = select_fixed_detector_target(
            model,
            inputs,
            detection,
            detect_name=self.config.detect_layer,
            orig_hw=pack["orig_hw"],
        )
        target_fn = detector_target_fn(
            fixed_target,
            imgsz=int(self.config.attack.imgsz),
            detect_name=self.config.detect_layer,
            mode=self.config.target_mode,
        )
        return {
            "example": example,
            "clean_lb": clean_lb,
            "patched_lb": patched_lb,
            "inputs": inputs,
            "baselines": baselines,
            "fixed_target": fixed_target,
            "target_fn": target_fn,
        }

    def _selected_examples(self, examples: list[AttackExample] | None, max_examples: int | None) -> list[AttackExample]:
        cache = self.get_cache()
        selected = list(examples or cache.examples)
        if max_examples is not None:
            selected = selected[: int(max_examples)]
        return selected

    def _attribution_comparison_cache_path(self, selected: list[AttackExample]) -> Path:
        payload = {
            "attack_cache_key": self.get_cache().cache_key,
            "paths": [item.path for item in selected],
            "target_layer": self.config.target_layer,
            "detect_layer": self.config.detect_layer,
            "target_mode": self.config.target_mode,
            "n_steps": int(self.config.n_steps),
            "alpha_batch_size": int(self.config.alpha_batch_size),
            "overlap_percentages": list(self.config.overlap_percentages),
            "method_version": 2,
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
        return self.derived_cache_dir / f"attribution_comparison_{key}.pkl"

    def _success_failure_metrics_cache_path(self, selected: list[AttackExample], *, layer_name: str, top_percent: float) -> Path:
        payload = {
            "attack_cache_key": self.get_cache().cache_key,
            "paths": [item.path for item in selected],
            "target_layer": layer_name,
            "detect_layer": self.config.detect_layer,
            "target_mode": self.config.target_mode,
            "n_steps": int(self.config.n_steps),
            "alpha_batch_size": int(self.config.alpha_batch_size),
            "top_percent": float(top_percent),
            "method_version": 6,
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
        return self.derived_cache_dir / f"success_failure_metrics_{key}.pkl"

    def _segmentig_success_failure_metrics_cache_path(
        self,
        selected: list[AttackExample],
        *,
        layer_name: str,
        top_percent: float,
    ) -> Path:
        payload = {
            "attack_cache_key": self.get_cache().cache_key,
            "paths": [item.path for item in selected],
            "target_layer": layer_name,
            "detect_layer": self.config.detect_layer,
            "target_mode": self.config.target_mode,
            "n_steps": int(self.config.n_steps),
            "alpha_batch_size": int(self.config.alpha_batch_size),
            "top_percent": float(top_percent),
            "method_version": 2,
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
        return self.derived_cache_dir / f"segmentig_success_failure_metrics_{key}.pkl"

    def _layer_map_cache_path(self, example: AttackExample, *, layer_name: str) -> Path:
        payload = {
            "attack_cache_key": self.get_cache().cache_key,
            "path": example.path,
            "drop": float(example.drop),
            "success": bool(example.success),
            "target_layer": layer_name,
            "detect_layer": self.config.detect_layer,
            "target_mode": self.config.target_mode,
            "n_steps": int(self.config.n_steps),
            "alpha_batch_size": int(self.config.alpha_batch_size),
            "imgsz": int(self.config.attack.imgsz),
            "method_version": 1,
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:16]
        out = self.derived_cache_dir / "layer_maps"
        out.mkdir(parents=True, exist_ok=True)
        return out / f"layer_maps_{key}.npz"

    def _load_layer_map_cache(self, example: AttackExample, *, layer_name: str, include_clean_activation: bool = True):
        import numpy as np

        cache_path = self._layer_map_cache_path(example, layer_name=layer_name)
        if not cache_path.exists():
            return None
        with np.load(cache_path, allow_pickle=False) as data:
            required = {"delta_chw", "segmentig_chw", "activation_shape"}
            if include_clean_activation:
                required.add("clean_activation_chw")
            if not required.issubset(set(data.files)):
                return None
            out = {
                "delta_chw": data["delta_chw"].astype("float16", copy=False),
                "segmentig_chw": data["segmentig_chw"].astype("float16", copy=False),
                "activation_shape": tuple(int(v) for v in data["activation_shape"].tolist()),
                "cache_path": str(cache_path),
                "loaded_from_cache": True,
            }
            if include_clean_activation:
                out["clean_activation_chw"] = data["clean_activation_chw"].astype("float16", copy=False)
            return out

    def _save_layer_map_cache(self, example: AttackExample, *, layer_name: str, maps: dict[str, Any]):
        import numpy as np

        cache_path = self._layer_map_cache_path(example, layer_name=layer_name)
        delta_chw = np.asarray(maps["delta_chw"], dtype="float16")
        segmentig_chw = np.asarray(maps["segmentig_chw"], dtype="float16")
        clean_activation_chw = np.asarray(maps["clean_activation_chw"], dtype="float16")
        np.savez_compressed(
            cache_path,
            delta_chw=delta_chw,
            segmentig_chw=segmentig_chw,
            clean_activation_chw=clean_activation_chw,
            activation_shape=np.asarray(maps["activation_shape"], dtype="int64"),
        )
        maps = dict(maps)
        maps["delta_chw"] = delta_chw
        maps["segmentig_chw"] = segmentig_chw
        maps["clean_activation_chw"] = clean_activation_chw
        maps["cache_path"] = str(cache_path)
        maps["loaded_from_cache"] = False
        return maps

    def _compute_or_load_segmentig_layer_maps(
        self,
        example: AttackExample,
        ctx: dict[str, Any],
        *,
        model,
        layer,
        layer_name: str,
        force: bool = False,
        include_clean_activation: bool = True,
    ):
        import numpy as np

        from .activations import capture_activations, compute_layer_deltas
        from .attributions import compute_layer_ig_attribution

        if not force:
            cached = self._load_layer_map_cache(
                example,
                layer_name=layer_name,
                include_clean_activation=include_clean_activation,
            )
            if cached is not None:
                return cached

        clean_pack = self._preprocess(ctx["clean_lb"])
        patched_pack = self._preprocess(ctx["patched_lb"])
        deltas = compute_layer_deltas(model, clean_pack["im"], patched_pack["im"], [layer_name])
        if layer_name not in deltas:
            raise RuntimeError(f"Layer {layer_name!r} was not captured for {example.path}")
        clean_acts = capture_activations(model, clean_pack["im"], [layer_name])
        if layer_name not in clean_acts:
            raise RuntimeError(f"Clean activation for layer {layer_name!r} was not captured for {example.path}")
        segmentig = compute_layer_ig_attribution(
            model,
            ctx["inputs"],
            ctx["baselines"],
            target_fn=ctx["target_fn"],
            layer=layer,
            layer_name=layer_name,
            method="SegmentIG[0;0.1]",
            n_steps=int(self.config.n_steps),
            alpha_batch_size=int(self.config.alpha_batch_size),
            segment_start=0.0,
            segment_end=0.1,
        )

        def chw_array(tensor):
            arr = tensor.detach().cpu().numpy()
            if arr.ndim == 4:
                arr = arr[0]
            return arr.astype("float32", copy=False)

        maps = {
            "delta_chw": chw_array(deltas[layer_name].delta),
            "segmentig_chw": chw_array(segmentig.chw()),
            "clean_activation_chw": chw_array(clean_acts[layer_name]),
            "activation_shape": np.asarray(segmentig.activation_shape, dtype="int64"),
        }
        saved = self._save_layer_map_cache(example, layer_name=layer_name, maps=maps)
        if not include_clean_activation:
            saved.pop("clean_activation_chw", None)
        return saved

    def _plot_attribution_comparison(self, result: dict[str, Any]) -> dict[str, Any]:
        from .plots import plot_overlap, plot_ranked_attributions, plot_runtime

        figure_paths: list[str] = []
        for top_n in self.config.ranking_top_ns:
            for smoothed in (False, True):
                path = self.figures_dir / f"ranked_attributions_top{top_n}_{'smooth' if smoothed else 'raw'}.png"
                fig = plot_ranked_attributions(
                    result["scores"],
                    sort_by="SegmentIG[0;0.1]",
                    top_n=int(top_n),
                    smoothed=smoothed,
                    window=int(self.config.smoothing_window),
                    save_path=path,
                )
                import matplotlib.pyplot as plt

                plt.close(fig)
                figure_paths.append(str(path))
        overlap_path = self.figures_dir / "top_overlap.png"
        fig = plot_overlap(result["overlap"], save_path=overlap_path)
        import matplotlib.pyplot as plt

        plt.close(fig)
        figure_paths.append(str(overlap_path))
        runtime_path = self.figures_dir / "runtime_benchmark.png"
        fig = plot_runtime(result["runtime"], save_path=runtime_path)
        plt.close(fig)
        figure_paths.append(str(runtime_path))
        result = dict(result)
        result["figure_paths"] = figure_paths
        return result

    def run_attribution_comparison(
        self,
        *,
        examples: list[AttackExample] | None = None,
        max_examples: int | None = None,
        force: bool = False,
    ):
        import numpy as np

        from .attributions import aggregate_flat_abs, compute_layer_ig_attribution, compute_naa_attribution, mean_ia_gradient
        from .metrics import top_overlap_table
        from .yolo import get_module_by_name

        selected = self._selected_examples(examples, max_examples)
        cache_path = self._attribution_comparison_cache_path(selected)
        if cache_path.exists() and not force:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            cached["loaded_from_cache"] = True
            cached["cache_path"] = str(cache_path)
            return self._plot_attribution_comparison(cached)

        _yolo, model = self.load_model()
        layer = get_module_by_name(model, self.config.target_layer)
        contexts = [self._context_for_example(item, image_variant="clean") for item in selected]

        ia_contexts = [(ctx["inputs"], ctx["target_fn"]) for ctx in contexts]
        ia_gradient = mean_ia_gradient(ia_contexts, model=model, layer=layer)

        method_results: dict[str, list[Any]] = {"Full IG": [], "SegmentIG[0;0.1]": [], "NAA": []}
        runtime_raw: dict[str, list[float]] = {name: [] for name in method_results}
        for ctx in contexts:
            full = compute_layer_ig_attribution(
                model,
                ctx["inputs"],
                ctx["baselines"],
                target_fn=ctx["target_fn"],
                layer=layer,
                layer_name=self.config.target_layer,
                method="Full IG",
                n_steps=int(self.config.n_steps),
                alpha_batch_size=int(self.config.alpha_batch_size),
                segment_start=0.0,
                segment_end=1.0,
            )
            segment = compute_layer_ig_attribution(
                model,
                ctx["inputs"],
                ctx["baselines"],
                target_fn=ctx["target_fn"],
                layer=layer,
                layer_name=self.config.target_layer,
                method="SegmentIG[0;0.1]",
                n_steps=int(self.config.n_steps),
                alpha_batch_size=int(self.config.alpha_batch_size),
                segment_start=0.0,
                segment_end=0.1,
            )
            naa = compute_naa_attribution(
                model,
                ctx["inputs"],
                ctx["baselines"],
                target_fn=ctx["target_fn"],
                layer=layer,
                layer_name=self.config.target_layer,
                ia_gradient=ia_gradient,
            )
            for result in (full, segment, naa):
                method_results[result.method].append(result)
                runtime_raw[result.method].append(float(result.elapsed_s))

        scores = {name: aggregate_flat_abs(results).numpy() for name, results in method_results.items()}
        overlap = top_overlap_table(scores, percentages=self.config.overlap_percentages)
        runtime = [
            {"method": name, "mean_s": float(np.mean(values)), "std_s": float(np.std(values))}
            for name, values in runtime_raw.items()
        ]

        result = {
            "scores": scores,
            "overlap": overlap,
            "runtime": runtime,
            "cache_path": str(cache_path),
            "loaded_from_cache": False,
        }
        with cache_path.open("wb") as f:
            pickle.dump(result, f)
        return self._plot_attribution_comparison(result)

    def run_patch_propagation(self, *, examples: list[AttackExample] | None = None, layer_names: list[str] | None = None, max_examples: int = 4):
        import matplotlib.pyplot as plt

        from .activations import compute_layer_deltas, reduce_chw_to_hw, robust_normalize, delta_spread_metrics
        from .attributions import attribution_spatial_map, compute_layer_ig_attribution
        from .plots import plot_patch_propagation
        from .yolo import get_module_by_name

        cache = self.get_cache()
        selected = list(examples or cache.examples)[: int(max_examples)]
        layer_names = [name for name in (layer_names or self.representative_layers(max_layers=4)) if name != self.config.detect_layer]
        _yolo, model = self.load_model()
        rows = []
        for example in selected:
            ctx = self._context_for_example(example, image_variant="clean")
            ctx_patched = self._context_for_example(example, image_variant="patched")
            clean_pack = self._preprocess(ctx["clean_lb"])
            patched_pack = self._preprocess(ctx["patched_lb"])
            deltas = compute_layer_deltas(model, clean_pack["im"], patched_pack["im"], layer_names)
            panels = []
            for layer_name, delta_obj in deltas.items():
                layer = get_module_by_name(model, layer_name)
                seg_clean = compute_layer_ig_attribution(
                    model,
                    ctx["inputs"],
                    ctx["baselines"],
                    target_fn=ctx["target_fn"],
                    layer=layer,
                    layer_name=layer_name,
                    method="SegmentIG[0;0.1]",
                    n_steps=int(self.config.n_steps),
                    alpha_batch_size=int(self.config.alpha_batch_size),
                    segment_start=0.0,
                    segment_end=0.1,
                )
                seg_patched = compute_layer_ig_attribution(
                    model,
                    ctx_patched["inputs"],
                    ctx_patched["baselines"],
                    target_fn=ctx_patched["target_fn"],
                    layer=layer,
                    layer_name=layer_name,
                    method="SegmentIG[0;0.1]",
                    n_steps=int(self.config.n_steps),
                    alpha_batch_size=int(self.config.alpha_batch_size),
                    segment_start=0.0,
                    segment_end=0.1,
                )
                metrics = delta_spread_metrics(delta_obj.delta, patch_bbox_xyxy=example.patch_bbox_lb, imgsz=int(self.config.attack.imgsz))
                panels.append(
                    {
                        "layer": layer_name,
                        "delta_abs": robust_normalize(reduce_chw_to_hw(delta_obj.delta, mode="l2")),
                        "delta_signed": robust_normalize(reduce_chw_to_hw(delta_obj.delta, mode="signed_mean"), signed=True),
                        "importance_clean": attribution_spatial_map(seg_clean.chw(), reduction="l2"),
                        "importance_patched": attribution_spatial_map(seg_patched.chw(), reduction="l2"),
                        "metrics": metrics,
                    }
                )
                rows.append({"path": example.path, "success": example.success, "layer": layer_name, **metrics})
            save_path = self.figures_dir / f"patch_propagation_{Path(example.path).stem}.png"
            fig = plot_patch_propagation(example, ctx["clean_lb"], ctx["patched_lb"], panels, save_path=save_path)
            plt.close(fig)
        return rows

    def run_all_layer_patch_spread_table(self, *, examples: list[AttackExample] | None = None, max_per_class: int = 1):
        import matplotlib.pyplot as plt

        from .activations import compute_layer_deltas, delta_spread_metrics, reduce_chw_to_hw, robust_normalize
        from .plots import plot_all_layer_delta_strip

        cache = self.get_cache()
        if examples is None:
            selected = cache.successes[: int(max_per_class)] + cache.failures[: int(max_per_class)]
        else:
            selected = list(examples)
        layer_names = self.all_display_layer_names()
        _yolo, model = self.load_model()
        figure_panels = []
        metric_rows = []
        for example in selected:
            ctx = self._context_for_example(example, image_variant="clean")
            clean_pack = self._preprocess(ctx["clean_lb"])
            patched_pack = self._preprocess(ctx["patched_lb"])
            deltas = compute_layer_deltas(model, clean_pack["im"], patched_pack["im"], layer_names)
            maps = {}
            signed_maps = {}
            for layer_name in layer_names:
                if layer_name not in deltas:
                    continue
                delta = deltas[layer_name].delta
                maps[layer_name] = robust_normalize(reduce_chw_to_hw(delta, mode="l2"))
                signed_maps[layer_name] = robust_normalize(reduce_chw_to_hw(delta, mode="signed_mean"), signed=True)
                metric_rows.append(
                    {
                        "path": example.path,
                        "success": bool(example.success),
                        "layer": layer_name,
                        "conf_clean": float(example.conf_clean),
                        "conf_patch": float(example.conf_patch),
                        "drop": float(example.drop),
                        **delta_spread_metrics(delta, patch_bbox_xyxy=example.patch_bbox_lb, imgsz=int(self.config.attack.imgsz)),
                    }
                )
            figure_panels.append(
                {
                    "example": example,
                    "layers": [name for name in layer_names if name in maps],
                    "maps": maps,
                    "signed_maps": signed_maps,
                }
            )
        save_path = self.figures_dir / "all_layer_patch_spread_success_fail.png"
        fig = plot_all_layer_delta_strip(figure_panels, save_path=save_path)
        plt.close(fig)
        return {"rows": metric_rows, "figure_path": str(save_path), "layers": layer_names}

    def run_success_failure_metrics(
        self,
        *,
        layer_name: str | None = None,
        max_examples: int | None = None,
        top_percent: float = 5.0,
        force: bool = False,
    ):
        import matplotlib.pyplot as plt

        from .activations import delta_spread_metrics
        from .attributions import compute_layer_ig_attribution, compute_naa_attribution, mean_ia_gradient
        from .metrics import (
            alignment_metrics,
            handcrafted_delta_importance_features,
            importance_rank_bin_energy_fractions,
            metric_quality_rows,
            segmentig_soft_alignment_metrics,
        )
        from .plots import plot_metric_distribution_and_roc
        from .yolo import get_module_by_name

        cache = self.get_cache()
        selected = list(cache.examples)
        if max_examples is not None:
            selected = selected[: int(max_examples)]
        layer_name = layer_name or self.config.target_layer
        cache_path = self._success_failure_metrics_cache_path(selected, layer_name=layer_name, top_percent=float(top_percent))
        if cache_path.exists() and not force:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            cached["loaded_from_cache"] = True
            cached["cache_path"] = str(cache_path)
            return cached

        _yolo, model = self.load_model()
        layer = get_module_by_name(model, layer_name)
        contexts = [(example, self._context_for_example(example, image_variant="clean")) for example in selected]
        ia_gradient = mean_ia_gradient(
            [(ctx["inputs"], ctx["target_fn"]) for _example, ctx in contexts],
            model=model,
            layer=layer,
        )
        metric_rows = []
        metrics_batch_size = max(1, int(getattr(self.config, "metrics_batch_size", 64)))
        for example_idx, (example, ctx) in enumerate(contexts, start=1):
            layer_maps = self._compute_or_load_segmentig_layer_maps(
                example,
                ctx,
                model=model,
                layer=layer,
                layer_name=layer_name,
                include_clean_activation=False,
            )
            delta_chw = layer_maps["delta_chw"]
            segmentig_chw = layer_maps["segmentig_chw"]
            row = {
                "path": example.path,
                "success": bool(example.success),
                "conf_clean": example.conf_clean,
                "conf_patch": example.conf_patch,
                "drop": example.drop,
                "layer_maps_cache_path": layer_maps["cache_path"],
                "layer_maps_loaded_from_cache": bool(layer_maps["loaded_from_cache"]),
                **delta_spread_metrics(delta_chw, patch_bbox_xyxy=example.patch_bbox_lb, imgsz=int(self.config.attack.imgsz)),
            }
            attribution_results = {
                "full_ig": compute_layer_ig_attribution(
                    model,
                    ctx["inputs"],
                    ctx["baselines"],
                    target_fn=ctx["target_fn"],
                    layer=layer,
                    layer_name=layer_name,
                    method="Full IG",
                    n_steps=int(self.config.n_steps),
                    alpha_batch_size=int(self.config.alpha_batch_size),
                    segment_start=0.0,
                    segment_end=1.0,
                ),
                "segmentig": compute_layer_ig_attribution(
                    model,
                    ctx["inputs"],
                    ctx["baselines"],
                    target_fn=ctx["target_fn"],
                    layer=layer,
                    layer_name=layer_name,
                    method="SegmentIG[0;0.1]",
                    n_steps=int(self.config.n_steps),
                    alpha_batch_size=int(self.config.alpha_batch_size),
                    segment_start=0.0,
                    segment_end=0.1,
                ),
                "naa": compute_naa_attribution(
                    model,
                    ctx["inputs"],
                    ctx["baselines"],
                    target_fn=ctx["target_fn"],
                    layer=layer,
                    layer_name=layer_name,
                    ia_gradient=ia_gradient,
                ),
            }
            delta_flat = delta_chw.reshape(-1)
            for method_key, attribution in attribution_results.items():
                attribution_flat = attribution.chw().detach().cpu().reshape(-1).numpy()
                method_metrics = alignment_metrics(
                    delta_flat,
                    attribution_flat,
                    top_percent=float(top_percent),
                )
                if method_key == "segmentig":
                    method_metrics.update(segmentig_soft_alignment_metrics(delta_flat, attribution_flat))
                    method_metrics.update(
                        handcrafted_delta_importance_features(
                            delta_chw,
                            segmentig_chw,
                            patch_bbox_xyxy=example.patch_bbox_lb,
                            imgsz=int(self.config.attack.imgsz),
                        )
                    )
                    bin_fractions = importance_rank_bin_energy_fractions(delta_flat, segmentig_chw.reshape(-1), n_bins=100)
                    method_metrics.update(
                        {
                            f"delta_energy_importance_binfrac_{idx:03d}": float(value)
                            for idx, value in enumerate(bin_fractions, start=1)
                        }
                    )
                for metric_name, value in method_metrics.items():
                    row[f"{method_key}_{metric_name}"] = value
            # Backward-compatible aliases for the selected attribution method.
            for metric_name in ("align_cosine", "align_top_jaccard", "importance_energy_in_delta_top", "delta_energy_in_importance_top"):
                row[metric_name] = row[f"segmentig_{metric_name}"]
            metric_rows.append(row)
            del row, attribution_results, delta_flat, delta_chw, segmentig_chw, layer_maps
            if example_idx % metrics_batch_size == 0:
                self._release_batch_memory()
        self._release_batch_memory()
        labels = [r["success"] for r in metric_rows]
        metric_names = [
            k
            for k in metric_rows[0].keys()
            if (
                k.startswith("delta_")
                or k.endswith("_frac")
                or k.startswith("align_")
                or k.endswith("_top")
                or k.startswith("segmentig_delta_importance_product_")
                or k.startswith("segmentig_delta_energy_importance_bins_")
                or k.startswith("segmentig_hand_")
            )
        ] if metric_rows else []
        quality = metric_quality_rows(labels, {name: [r[name] for r in metric_rows] for name in metric_names})
        for item in quality:
            name = item["metric"]
            path = self.figures_dir / f"metric_{name}.png"
            fig = plot_metric_distribution_and_roc(
                labels,
                [r[name] for r in metric_rows],
                metric_name=name,
                auc=float(item["roc_auc"]),
                best_accuracy=float(item["best_accuracy"]),
                direction=int(item["best_direction"]),
                save_path=path,
            )
            plt.close(fig)
            item["figure_path"] = str(path)
        result = {"rows": metric_rows, "quality": quality, "cache_path": str(cache_path), "loaded_from_cache": False}
        with cache_path.open("wb") as f:
            pickle.dump(result, f)
        return result

    def run_segmentig_success_failure_metrics(
        self,
        *,
        layer_name: str | None = None,
        max_examples: int | None = None,
        top_percent: float = 5.0,
        force: bool = False,
    ):
        import matplotlib.pyplot as plt

        from .activations import delta_spread_metrics
        from .metrics import (
            alignment_metrics,
            handcrafted_delta_importance_features,
            importance_rank_bin_energy_fractions,
            metric_quality_rows,
            segmentig_soft_alignment_metrics,
        )
        from .plots import plot_metric_distribution_and_roc
        from .yolo import get_module_by_name

        cache = self.get_cache()
        selected = list(cache.examples)
        if max_examples is not None:
            selected = selected[: int(max_examples)]
        layer_name = layer_name or self.config.target_layer
        cache_path = self._segmentig_success_failure_metrics_cache_path(
            selected,
            layer_name=layer_name,
            top_percent=float(top_percent),
        )
        if cache_path.exists() and not force:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            cached["loaded_from_cache"] = True
            cached["cache_path"] = str(cache_path)
            return cached

        model = None
        layer = None
        metric_rows = []
        skipped_rows = []
        metrics_batch_size = max(1, int(getattr(self.config, "metrics_batch_size", 64)))
        for example_idx, example in enumerate(selected, start=1):
            try:
                layer_maps = self._load_layer_map_cache(
                    example,
                    layer_name=layer_name,
                    include_clean_activation=False,
                )
                if layer_maps is None:
                    if model is None or layer is None:
                        _yolo, model = self.load_model()
                        layer = get_module_by_name(model, layer_name)
                    ctx = self._context_for_example(example, image_variant="clean")
                    layer_maps = self._compute_or_load_segmentig_layer_maps(
                        example,
                        ctx,
                        model=model,
                        layer=layer,
                        layer_name=layer_name,
                        include_clean_activation=False,
                    )
            except Exception as exc:  # noqa: BLE001 - one unusable example should not stop cache building.
                skipped_rows.append(
                    {
                        "path": example.path,
                        "success": bool(example.success),
                        "drop": float(example.drop),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                if example_idx % metrics_batch_size == 0:
                    self._release_batch_memory()
                continue
            delta_chw = layer_maps["delta_chw"]
            segmentig_chw = layer_maps["segmentig_chw"]
            delta_flat = delta_chw.reshape(-1)
            segmentig_flat = segmentig_chw.reshape(-1)

            method_metrics = alignment_metrics(
                delta_flat,
                segmentig_flat,
                top_percent=float(top_percent),
            )
            method_metrics.update(segmentig_soft_alignment_metrics(delta_flat, segmentig_flat))
            method_metrics.update(
                handcrafted_delta_importance_features(
                    delta_chw,
                    segmentig_chw,
                    patch_bbox_xyxy=example.patch_bbox_lb,
                    imgsz=int(self.config.attack.imgsz),
                )
            )
            bin_fractions = importance_rank_bin_energy_fractions(delta_flat, segmentig_flat, n_bins=100)
            method_metrics.update(
                {
                    f"delta_energy_importance_binfrac_{idx:03d}": float(value)
                    for idx, value in enumerate(bin_fractions, start=1)
                }
            )

            row = {
                "path": example.path,
                "success": bool(example.success),
                "conf_clean": example.conf_clean,
                "conf_patch": example.conf_patch,
                "drop": example.drop,
                "layer_maps_cache_path": layer_maps["cache_path"],
                "layer_maps_loaded_from_cache": bool(layer_maps["loaded_from_cache"]),
                **delta_spread_metrics(delta_chw, patch_bbox_xyxy=example.patch_bbox_lb, imgsz=int(self.config.attack.imgsz)),
            }
            for metric_name, value in method_metrics.items():
                row[f"segmentig_{metric_name}"] = value
            metric_rows.append(row)
            del row, method_metrics, delta_flat, segmentig_flat, delta_chw, segmentig_chw, layer_maps
            if example_idx % metrics_batch_size == 0:
                self._release_batch_memory()
        self._release_batch_memory()

        if not metric_rows:
            raise RuntimeError(f"No valid SegmentIG success/failure rows; skipped={len(skipped_rows)}")

        labels = [r["success"] for r in metric_rows]
        metric_names = [
            k
            for k in metric_rows[0].keys()
            if (
                k.startswith("delta_")
                or k.startswith("segmentig_align_")
                or k.startswith("segmentig_importance_")
                or k.startswith("segmentig_delta_energy_in_importance_top")
                or k.startswith("segmentig_delta_importance_product_")
                or k.startswith("segmentig_delta_energy_importance_bins_")
                or k.startswith("segmentig_delta_energy_importance_binfrac_")
                or k.startswith("segmentig_hand_")
            )
        ] if metric_rows else []
        quality = metric_quality_rows(labels, {name: [r[name] for r in metric_rows] for name in metric_names})
        for item in quality:
            name = item["metric"]
            path = self.figures_dir / f"metric_{name}.png"
            fig = plot_metric_distribution_and_roc(
                labels,
                [r[name] for r in metric_rows],
                metric_name=name,
                auc=float(item["roc_auc"]),
                best_accuracy=float(item["best_accuracy"]),
                direction=int(item["best_direction"]),
                save_path=path,
            )
            plt.close(fig)
            item["figure_path"] = str(path)
        result = {
            "rows": metric_rows,
            "quality": quality,
            "skipped": skipped_rows,
            "cache_path": str(cache_path),
            "loaded_from_cache": False,
        }
        with cache_path.open("wb") as f:
            pickle.dump(result, f)
        return result

    def run_failure_diagnosis(self, *, layer_name: str | None = None, max_examples: int | None = None, top_percent: float = 5.0):
        import matplotlib.pyplot as plt
        import numpy as np

        from .activations import compute_layer_deltas, reduce_chw_to_hw, robust_normalize
        from .attributions import compute_layer_ig_attribution
        from .metrics import alignment_metrics
        from .plots import plot_failure_diagnosis
        from .yolo import get_module_by_name

        cache = self.get_cache()
        selected = list(cache.examples)
        if max_examples is not None:
            selected = selected[: int(max_examples)]
        layer_name = layer_name or self.config.target_layer
        _yolo, model = self.load_model()
        layer = get_module_by_name(model, layer_name)
        groups = {
            "success": {
                "delta_energy": [],
                "signed_delta": [],
                "delta_in_important": [],
                "importance": [],
                "delta_outside_important": [],
                "rows": [],
            },
            "fail": {
                "delta_energy": [],
                "signed_delta": [],
                "delta_in_important": [],
                "importance": [],
                "delta_outside_important": [],
                "rows": [],
            },
        }

        for example in selected:
            ctx = self._context_for_example(example, image_variant="clean")
            clean_pack = self._preprocess(ctx["clean_lb"])
            patched_pack = self._preprocess(ctx["patched_lb"])
            deltas = compute_layer_deltas(model, clean_pack["im"], patched_pack["im"], [layer_name])
            if layer_name not in deltas:
                continue
            delta = deltas[layer_name].delta.detach()
            seg = compute_layer_ig_attribution(
                model,
                ctx["inputs"],
                ctx["baselines"],
                target_fn=ctx["target_fn"],
                layer=layer,
                layer_name=layer_name,
                method="SegmentIG[0;0.1]",
                n_steps=int(self.config.n_steps),
                alpha_batch_size=int(self.config.alpha_batch_size),
                segment_start=0.0,
                segment_end=0.1,
            )
            delta_abs = delta.abs()
            importance_abs = seg.chw().detach().abs().to(device=delta.device, dtype=delta.dtype)
            flat_importance = importance_abs.reshape(-1)
            k = max(1, int(round(float(top_percent) / 100.0 * flat_importance.numel())))
            top_idx = flat_importance.topk(k=min(k, flat_importance.numel())).indices
            important_mask = importance_abs.reshape(-1).new_zeros(flat_importance.shape, dtype=delta.dtype)
            important_mask[top_idx] = 1.0
            important_mask = important_mask.reshape_as(importance_abs)

            delta_energy = reduce_chw_to_hw(delta, mode="l2")
            signed_delta = reduce_chw_to_hw(delta, mode="signed_mean")
            delta_in_important = reduce_chw_to_hw(delta_abs * important_mask, mode="l2")
            delta_outside_important = reduce_chw_to_hw(delta_abs * (1.0 - important_mask), mode="l2")
            importance_map = reduce_chw_to_hw(importance_abs, mode="mean_abs")
            group = "success" if example.success else "fail"
            groups[group]["delta_energy"].append(delta_energy.detach().cpu().numpy())
            groups[group]["signed_delta"].append(signed_delta.detach().cpu().numpy())
            groups[group]["delta_in_important"].append(delta_in_important.detach().cpu().numpy())
            groups[group]["delta_outside_important"].append(delta_outside_important.detach().cpu().numpy())
            groups[group]["importance"].append(importance_map.detach().cpu().numpy())

            row = {
                "path": example.path,
                "success": bool(example.success),
                "drop": float(example.drop),
                "delta_l2_rms": float(np.sqrt(np.mean(delta_energy.detach().cpu().numpy().reshape(-1) ** 2))),
            }
            delta_hw_flat = delta_energy.detach().cpu().numpy().reshape(-1)
            spatial_k = max(1, int(round(float(top_percent) / 100.0 * delta_hw_flat.size)))
            row["topk_energy_frac"] = float(np.sort(delta_hw_flat)[-spatial_k:].sum() / (delta_hw_flat.sum() + 1e-12))
            align = alignment_metrics(
                delta.detach().cpu().reshape(-1).numpy(),
                seg.chw().detach().cpu().reshape(-1).numpy(),
                top_percent=float(top_percent),
            )
            row.update({f"segmentig_{name}": value for name, value in align.items()})
            groups[group]["rows"].append(row)

        def mean_map(items, *, signed: bool = False):
            if not items:
                return np.zeros((1, 1), dtype="float32")
            return robust_normalize(np.mean(np.stack(items, axis=0), axis=0), signed=signed)

        maps = {
            group: {
                key: mean_map(groups[group][key], signed=(key == "signed_delta"))
                for key in ("delta_energy", "signed_delta", "delta_in_important", "importance", "delta_outside_important")
            }
            for group in groups
        }
        metric_names = [
            "delta_l2_rms",
            "topk_energy_frac",
            "segmentig_delta_energy_in_importance_top",
            "segmentig_align_cosine",
        ]
        metric_means = {}
        for group in groups:
            rows = groups[group]["rows"]
            metric_means[group] = {
                name: float(np.nanmean([row.get(name, np.nan) for row in rows])) if rows else float("nan")
                for name in metric_names
            }
        diagnosis = {
            "maps": maps,
            "metric_means": metric_means,
            "rows": groups["success"]["rows"] + groups["fail"]["rows"],
            "layer": layer_name,
            "top_percent": float(top_percent),
        }
        save_path = self.figures_dir / "failure_diagnosis_spread_vs_importance.png"
        fig = plot_failure_diagnosis(diagnosis, save_path=save_path)
        plt.close(fig)
        diagnosis["figure_path"] = str(save_path)
        return diagnosis

    def with_attack_config(self, **updates):
        return PatchSuccessExperiment(replace(self.config, attack=replace(self.config.attack, **updates)))
