from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultiPatchASRConfig:
    dataset_dir: str
    patch_paths: tuple[str, ...]
    model_path: str
    output_root: str
    imgsz: int = 640
    conf: float = 0.01
    success_drop: float = 0.30
    patch_xy: tuple[int, int] = (0, 0)
    patch_size: tuple[int, int] | None = None
    batch_size: int = 32
    device: str | None = "mps"
    max_images: int | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_hash(paths: Iterable[Path]) -> str:
    names = "\n".join(str(path.resolve()) for path in paths)
    return hashlib.sha256(names.encode("utf-8")).hexdigest()


def _run_key(config: MultiPatchASRConfig, image_paths: list[Path]) -> str:
    patch_meta = [
        {"path": str(Path(path).resolve()), "sha256": _sha256(Path(path))}
        for path in config.patch_paths
    ]
    payload = {
        **config.payload(),
        "dataset_dir": str(Path(config.dataset_dir).resolve()),
        "model_path": str(Path(config.model_path).resolve()),
        "patches": patch_meta,
        "manifest_sha256": _manifest_hash(image_paths),
        "method_version": 1,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _clean_key(config: MultiPatchASRConfig, image_paths: list[Path]) -> str:
    model_path = Path(config.model_path).resolve()
    payload = {
        "dataset_dir": str(Path(config.dataset_dir).resolve()),
        "manifest_sha256": _manifest_hash(image_paths),
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "imgsz": int(config.imgsz),
        "conf": float(config.conf),
        "max_images": config.max_images,
        "method_version": 1,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = float(successes) / float(total)
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return center - half, center + half


def _person_record(result, person_class_id: int) -> dict[str, float | bool]:
    if result.boxes is None or len(result.boxes) == 0:
        return {
            "detected": False,
            "confidence": 0.0,
            "bbox_x1": float("nan"),
            "bbox_y1": float("nan"),
            "bbox_x2": float("nan"),
            "bbox_y2": float("nan"),
        }
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    indices = np.flatnonzero(classes == int(person_class_id))
    if not len(indices):
        return {
            "detected": False,
            "confidence": 0.0,
            "bbox_x1": float("nan"),
            "bbox_y1": float("nan"),
            "bbox_x2": float("nan"),
            "bbox_y2": float("nan"),
        }
    confidences = result.boxes.conf.detach().cpu().numpy()
    index = int(indices[int(np.argmax(confidences[indices]))])
    box = result.boxes.xyxy.detach().cpu().numpy()[index]
    return {
        "detected": True,
        "confidence": float(confidences[index]),
        "bbox_x1": float(box[0]),
        "bbox_y1": float(box[1]),
        "bbox_x2": float(box[2]),
        "bbox_y2": float(box[3]),
    }


def _predict_with_retry(
    yolo,
    images: list[Any],
    records: list[dict[str, Any]],
    *,
    imgsz: int,
    conf: float,
    device: str | None,
    batch_size: int,
) -> tuple[list[tuple[dict[str, Any], Any]], list[dict[str, str]]]:
    if not records:
        return [], []
    try:
        from new_experiments.patch_success_analysis.yolo import pil_to_np_bgr

        results = yolo.predict(
            source=[pil_to_np_bgr(image) for image in images],
            imgsz=int(imgsz),
            conf=float(conf),
            device=device,
            batch=max(1, int(batch_size)),
            verbose=False,
        )
        return list(zip(records, results, strict=True)), []
    except Exception as exc:  # retry smaller groups before marking an image bad
        if len(records) == 1:
            return [], [{
                "path": str(records[0]["path"]),
                "error": f"{type(exc).__name__}: {exc}",
            }]
        middle = len(records) // 2
        left, left_bad = _predict_with_retry(
            yolo,
            images[:middle],
            records[:middle],
            imgsz=imgsz,
            conf=conf,
            device=device,
            batch_size=max(1, min(batch_size // 2, middle)),
        )
        right, right_bad = _predict_with_retry(
            yolo,
            images[middle:],
            records[middle:],
            imgsz=imgsz,
            conf=conf,
            device=device,
            batch_size=max(1, min(batch_size // 2, len(records) - middle)),
        )
        return left + right, left_bad + right_bad


def _load_letterboxed(paths: list[Path], imgsz: int) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, str]]]:
    from PIL import Image
    from new_experiments.patch_success_analysis.patching import letterbox_pil

    images: list[Any] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                images.append(letterbox_pil(image.convert("RGB"), imgsz=int(imgsz)))
            records.append({"path": str(path.resolve()), "filename": path.name})
        except Exception as exc:
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return images, records, failures


def _clean_baseline(
    yolo,
    image_paths: list[Path],
    *,
    person_class_id: int,
    config: MultiPatchASRConfig,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    from tqdm.auto import tqdm

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    progress = tqdm(total=len(image_paths), desc="Clean baseline", unit="img")
    for start in range(0, len(image_paths), int(config.batch_size)):
        paths = image_paths[start : start + int(config.batch_size)]
        images, records, load_bad = _load_letterboxed(paths, config.imgsz)
        failures.extend(load_bad)
        predicted, predict_bad = _predict_with_retry(
            yolo,
            images,
            records,
            imgsz=config.imgsz,
            conf=config.conf,
            device=config.device,
            batch_size=config.batch_size,
        )
        failures.extend(predict_bad)
        for record, result in predicted:
            detection = _person_record(result, person_class_id)
            width = float(detection["bbox_x2"]) - float(detection["bbox_x1"])
            height = float(detection["bbox_y2"]) - float(detection["bbox_y1"])
            rows.append({
                **record,
                "clean_detected": bool(detection["detected"]),
                "conf_clean": float(detection["confidence"]),
                "clean_bbox_x1": detection["bbox_x1"],
                "clean_bbox_y1": detection["bbox_y1"],
                "clean_bbox_x2": detection["bbox_x2"],
                "clean_bbox_y2": detection["bbox_y2"],
                "clean_bbox_area_frac": (
                    max(width, 0.0) * max(height, 0.0) / float(config.imgsz**2)
                    if bool(detection["detected"])
                    else float("nan")
                ),
            })
        progress.update(len(paths))
        progress.set_postfix(valid=sum(bool(row["clean_detected"]) for row in rows), bad=len(failures))
    progress.close()
    return pd.DataFrame(rows), failures


def _patch_rows(
    yolo,
    clean_valid: pd.DataFrame,
    patch_path: Path,
    *,
    patch_name: str,
    person_class_id: int,
    config: MultiPatchASRConfig,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    from PIL import Image
    from tqdm.auto import tqdm
    from new_experiments.patch_success_analysis.patching import apply_patch_to_image, letterbox_pil

    patch = Image.open(patch_path).convert("RGB")
    source_patch_width, source_patch_height = patch.size
    if config.patch_size is not None:
        applied_patch_width, applied_patch_height = map(int, config.patch_size)
        if applied_patch_width <= 0 or applied_patch_height <= 0:
            raise ValueError(f"patch_size must be positive, got {config.patch_size}")
        patch = patch.resize(
            (applied_patch_width, applied_patch_height),
            resample=Image.Resampling.LANCZOS,
        )
    else:
        applied_patch_width, applied_patch_height = patch.size
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    source = clean_valid.to_dict("records")
    progress = tqdm(total=len(source), desc=f"Patch: {patch_name}", unit="img")
    for start in range(0, len(source), int(config.batch_size)):
        chunk = source[start : start + int(config.batch_size)]
        images: list[Any] = []
        records: list[dict[str, Any]] = []
        for record in chunk:
            path = Path(record["path"])
            try:
                with Image.open(path) as image:
                    clean_lb = letterbox_pil(image.convert("RGB"), imgsz=int(config.imgsz))
                patched, patch_bbox, patch_area = apply_patch_to_image(
                    clean_lb, patch, position_xy=config.patch_xy
                )
                images.append(patched)
                records.append({
                    "path": str(path.resolve()),
                    "filename": path.name,
                    "patch": patch_name,
                    "patch_path": str(patch_path.resolve()),
                    "source_patch_width": int(source_patch_width),
                    "source_patch_height": int(source_patch_height),
                    "patch_width": int(applied_patch_width),
                    "patch_height": int(applied_patch_height),
                    "patch_area_frac": float(patch_area) / float(config.imgsz**2),
                    "patch_bbox": patch_bbox,
                })
            except Exception as exc:
                failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
        predicted, predict_bad = _predict_with_retry(
            yolo,
            images,
            records,
            imgsz=config.imgsz,
            conf=config.conf,
            device=config.device,
            batch_size=config.batch_size,
        )
        failures.extend(predict_bad)
        for record, result in predicted:
            detection = _person_record(result, person_class_id)
            rows.append({
                **record,
                "conf_patch": float(detection["confidence"]),
                "patch_detected": bool(detection["detected"]),
            })
        progress.update(len(chunk))
        progress.set_postfix(bad=len(failures))
    progress.close()
    frame = pd.DataFrame(rows).merge(
        clean_valid[
            [
                "path", "conf_clean", "clean_bbox_area_frac",
                "clean_bbox_x1", "clean_bbox_y1", "clean_bbox_x2", "clean_bbox_y2",
            ]
        ],
        on="path",
        how="left",
        validate="one_to_one",
    )
    frame["drop"] = frame["conf_clean"] - frame["conf_patch"]
    frame["success"] = frame["drop"] >= float(config.success_drop)
    frame["complete_hide"] = ~frame["patch_detected"].astype(bool)
    return frame, failures


def summarize_results(details: pd.DataFrame, config: MultiPatchASRConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for patch, group in details.groupby("patch", sort=False):
        successes = int(group["success"].sum())
        ci_low, ci_high = wilson_interval(successes, len(group))
        rows.append({
            "patch": patch,
            "n": int(len(group)),
            "successes": successes,
            "asr": float(group["success"].mean()),
            "asr_ci_low": ci_low,
            "asr_ci_high": ci_high,
            "complete_hide_rate": float(group["complete_hide"].mean()),
            "mean_conf_clean": float(group["conf_clean"].mean()),
            "mean_conf_patch": float(group["conf_patch"].mean()),
            "mean_drop": float(group["drop"].mean()),
            "median_drop": float(group["drop"].median()),
            "patch_width": int(group["patch_width"].iloc[0]),
            "patch_height": int(group["patch_height"].iloc[0]),
            "patch_area_frac": float(group["patch_area_frac"].iloc[0]),
            "success_drop": float(config.success_drop),
        })
    return pd.DataFrame(rows).sort_values("asr", ascending=False).reset_index(drop=True)


def evaluate_multi_patch_asr(
    config: MultiPatchASRConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    from new_experiments.patch_success_analysis.data import list_image_paths
    from new_experiments.patch_success_analysis.yolo import get_class_id, load_yolo

    dataset_dir = Path(config.dataset_dir).expanduser().resolve()
    model_path = Path(config.model_path).expanduser().resolve()
    patch_paths = [Path(path).expanduser().resolve() for path in config.patch_paths]
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"COCO_people was not found at {dataset_dir}. "
            "Set DATASET_DIR in the notebook to the directory containing the images."
        )
    if not model_path.is_file():
        raise FileNotFoundError(f"Model was not found: {model_path}")
    missing = [str(path) for path in patch_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Patch files were not found: {missing}")

    image_paths = list_image_paths(dataset_dir)
    if config.max_images is not None:
        image_paths = image_paths[: int(config.max_images)]
    if not image_paths:
        raise RuntimeError(f"No images found in {dataset_dir}")

    run_key = _run_key(config, image_paths)
    output_dir = Path(config.output_root).expanduser().resolve() / f"multi_patch_asr_{run_key}"
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "details.csv"
    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / "metadata.json"
    if not force and details_path.exists() and summary_path.exists() and metadata_path.exists():
        return {
            "output_dir": output_dir,
            "details": pd.read_csv(details_path),
            "summary": pd.read_csv(summary_path),
            "metadata": json.loads(metadata_path.read_text()),
            "loaded_from_cache": True,
        }

    started = time.perf_counter()
    yolo = load_yolo(model_path, device=config.device)
    names = getattr(yolo, "names", getattr(getattr(yolo, "model", None), "names", {}))
    person_class_id = get_class_id(names, "person")
    if person_class_id is None:
        raise RuntimeError("The model has no class named 'person'.")

    clean_cache_dir = Path(config.output_root).expanduser().resolve() / "_clean_baselines"
    clean_cache_dir.mkdir(parents=True, exist_ok=True)
    clean_key = _clean_key(config, image_paths)
    clean_cache_path = clean_cache_dir / f"clean_{clean_key}.csv"
    clean_failure_path = clean_cache_dir / f"clean_{clean_key}_failures.json"
    # `force` invalidates patch results, not the identical clean baseline shared
    # by native-size and controlled-size runs.
    if clean_cache_path.exists():
        clean = pd.read_csv(clean_cache_path)
        clean_failures = (
            json.loads(clean_failure_path.read_text())
            if clean_failure_path.exists()
            else []
        )
        clean_loaded_from_cache = True
    else:
        clean, clean_failures = _clean_baseline(
            yolo,
            image_paths,
            person_class_id=person_class_id,
            config=config,
        )
        clean.to_csv(clean_cache_path, index=False)
        clean_failure_path.write_text(
            json.dumps(clean_failures, indent=2, ensure_ascii=False)
        )
        clean_loaded_from_cache = False
    clean_valid = clean[clean["clean_detected"].astype(bool)].copy()
    clean.to_csv(output_dir / "clean_baseline.csv", index=False)
    if clean_valid.empty:
        raise RuntimeError("The detector found no person on any clean image.")

    frames: list[pd.DataFrame] = []
    failure_payload: dict[str, list[dict[str, str]]] = {"clean": clean_failures}
    for patch_path in patch_paths:
        frame, failures = _patch_rows(
            yolo,
            clean_valid,
            patch_path,
            patch_name=patch_path.stem,
            person_class_id=person_class_id,
            config=config,
        )
        frames.append(frame)
        failure_payload[patch_path.stem] = failures

    details = pd.concat(frames, ignore_index=True)
    summary = summarize_results(details, config)
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    (output_dir / "failures.json").write_text(
        json.dumps(failure_payload, indent=2, ensure_ascii=False)
    )
    metadata = {
        "method_version": 1,
        "metric_semantics": "image_level_max_person_confidence_drop",
        "success_rule": f"conf_clean - conf_patch >= {config.success_drop}",
        "validity_rule": f"clean person detection at inference conf >= {config.conf}",
        "paired_design": True,
        "patches_use_native_pixel_sizes": config.patch_size is None,
        "applied_patch_size": config.patch_size,
        "apply_patch_after_letterbox": True,
        "clean_loaded_from_cache": clean_loaded_from_cache,
        "config": config.payload(),
        "dataset_images_total": len(image_paths),
        "clean_images_evaluated": len(clean),
        "clean_images_valid": len(clean_valid),
        "details_rows": len(details),
        "elapsed_seconds": time.perf_counter() - started,
        "output_dir": str(output_dir),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    del yolo
    try:
        import gc
        import torch

        gc.collect()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass
    return {
        "output_dir": output_dir,
        "details": details,
        "summary": summary,
        "metadata": metadata,
        "loaded_from_cache": False,
    }


def plot_multi_patch_results(
    details: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    success_drop: float = 0.30,
):
    import matplotlib.pyplot as plt

    order = summary.sort_values("asr", ascending=False)["patch"].tolist()
    colors = plt.cm.Set2(np.linspace(0.05, 0.95, len(order)))
    color_map = dict(zip(order, colors, strict=True))
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    ordered_summary = summary.set_index("patch").loc[order].reset_index()
    lower = ordered_summary["asr"] - ordered_summary["asr_ci_low"]
    upper = ordered_summary["asr_ci_high"] - ordered_summary["asr"]
    axes[0, 0].bar(
        ordered_summary["patch"],
        ordered_summary["asr"],
        color=[color_map[name] for name in order],
        yerr=np.vstack([lower, upper]),
        capsize=4,
    )
    axes[0, 0].set(title="Attack success rate (95% Wilson CI)", ylabel="ASR", ylim=(0, 1))
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].grid(axis="y", alpha=0.25)

    values = [details.loc[details["patch"].eq(name), "drop"].to_numpy() for name in order]
    violin = axes[0, 1].violinplot(values, showmedians=True, showextrema=False)
    for body, name in zip(violin["bodies"], order, strict=True):
        body.set_facecolor(color_map[name])
        body.set_alpha(0.75)
    axes[0, 1].axhline(success_drop, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set(
        title="Confidence-drop distribution",
        ylabel="clean confidence − patched confidence",
        xticks=np.arange(1, len(order) + 1),
        xticklabels=order,
    )
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].grid(axis="y", alpha=0.25)

    for name in order:
        values = np.sort(details.loc[details["patch"].eq(name), "drop"].to_numpy())
        y = np.arange(1, len(values) + 1) / max(1, len(values))
        axes[1, 0].plot(values, y, label=name, color=color_map[name], linewidth=2)
    axes[1, 0].axvline(success_drop, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set(
        title="Empirical CDF of confidence drop",
        xlabel="clean confidence − patched confidence",
        ylabel="fraction of images",
    )
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    image_sizes = details[["path", "clean_bbox_area_frac"]].drop_duplicates("path").copy()
    image_sizes["object_size_bin"] = pd.qcut(
        image_sizes["clean_bbox_area_frac"].rank(method="first"),
        q=4,
        labels=["smallest", "small", "large", "largest"],
    )
    size_frame = details.merge(
        image_sizes[["path", "object_size_bin"]],
        on="path",
        how="left",
        validate="many_to_one",
    )
    size_asr = (
        size_frame.groupby(["object_size_bin", "patch"], observed=True)["success"]
        .mean()
        .unstack("patch")
        .reindex(columns=order)
    )
    for name in order:
        axes[1, 1].plot(
            size_asr.index.astype(str),
            size_asr[name],
            marker="o",
            label=name,
            color=color_map[name],
            linewidth=2,
        )
    axes[1, 1].set(
        title="ASR by clean target-box area quartile",
        xlabel="person bbox size",
        ylabel="ASR",
        ylim=(0, 1),
    )
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    return fig


def plot_success_overlap(details: pd.DataFrame):
    import matplotlib.pyplot as plt

    success = details.pivot(index="path", columns="patch", values="success").astype(float)
    names = success.columns.tolist()
    jaccard = np.zeros((len(names), len(names)), dtype=float)
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            left_values = success[left].to_numpy(dtype=float)
            right_values = success[right].to_numpy(dtype=float)
            valid = np.isfinite(left_values) & np.isfinite(right_values)
            left_bool = left_values[valid].astype(bool)
            right_bool = right_values[valid].astype(bool)
            union = int(np.logical_or(left_bool, right_bool).sum())
            intersection = int(np.logical_and(left_bool, right_bool).sum())
            jaccard[i, j] = (
                float(intersection / union) if union > 0 else float("nan")
            )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    correlation = success.corr()
    matrices = [
        (correlation.to_numpy(), "Phi correlation of attack success", -1.0, 1.0, "coolwarm"),
        (jaccard, "Jaccard overlap", 0.0, 1.0, "viridis"),
    ]
    for axis, (matrix, title, vmin, vmax, cmap) in zip(axes, matrices, strict=True):
        image = axis.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
        axis.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names, title=title)
        axis.tick_params(axis="x", rotation=25)
        for i in range(len(names)):
            for j in range(len(names)):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046)
    return fig
