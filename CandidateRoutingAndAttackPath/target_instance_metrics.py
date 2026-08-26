from __future__ import annotations

import json
import pickle
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .common import connect_db, stable_hash, write_json, write_markdown


@dataclass(slots=True)
class TargetInstanceConfig:
    output_dir: str
    match_iou: float = 0.50
    detection_conf: float = 0.25
    inference_conf: float = 0.01
    target_suppression_delta: float = 0.30
    batch_size: int = 16
    max_examples: int | None = None
    progress: bool = True
    method_version: int = 1


@dataclass(slots=True)
class TargetInstanceResult:
    run_dir: Path
    db_path: Path
    labels_path: Path
    summary_path: Path
    digest_path: Path
    config: TargetInstanceConfig


LABEL_COLUMNS = (
    "image_key", "path", "example_id", "legacy_success", "legacy_drop",
    "target_policy", "target_class_id", "target_clean_conf",
    "target_clean_x1", "target_clean_y1", "target_clean_x2", "target_clean_y2",
    "target_eligible", "match_iou_threshold", "detection_conf_threshold",
    "target_match_iou", "target_patched_conf", "target_conf_drop",
    "target_detected", "target_hidden", "target_suppressed_0p3",
    "matched_x1", "matched_y1", "matched_x2", "matched_y2",
    "patched_person_count", "patched_global_conf", "patched_winner_iou_target",
    "patched_winner_is_target", "patched_winner_x1", "patched_winner_y1",
    "patched_winner_x2", "patched_winner_y2", "outcome", "error",
)


def image_key(path: str | Path) -> str:
    return Path(str(path)).name


def _example_id(example: Any) -> str:
    return stable_hash({"path": str(example.path), "drop": float(example.drop), "success": bool(example.success)})


def _box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float32).reshape(4)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not len(boxes):
        return np.zeros(0, dtype=np.float32)
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    wh = np.maximum(0.0, rb - lt)
    inter = wh[:, 0] * wh[:, 1]
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(1e-9, area_a + area_b - inter)


def _create_label_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS labels (
            image_key TEXT PRIMARY KEY, path TEXT, example_id TEXT,
            legacy_success INTEGER, legacy_drop REAL, target_policy TEXT,
            target_class_id INTEGER, target_clean_conf REAL,
            target_clean_x1 REAL, target_clean_y1 REAL, target_clean_x2 REAL, target_clean_y2 REAL,
            target_eligible INTEGER, match_iou_threshold REAL, detection_conf_threshold REAL,
            target_match_iou REAL, target_patched_conf REAL, target_conf_drop REAL,
            target_detected INTEGER, target_hidden INTEGER, target_suppressed_0p3 INTEGER,
            matched_x1 REAL, matched_y1 REAL, matched_x2 REAL, matched_y2 REAL,
            patched_person_count INTEGER, patched_global_conf REAL, patched_winner_iou_target REAL,
            patched_winner_is_target INTEGER, patched_winner_x1 REAL, patched_winner_y1 REAL,
            patched_winner_x2 REAL, patched_winner_y2 REAL, outcome TEXT, error TEXT
        )
        """
    )
    conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL)")
    conn.commit()


def _upsert_label_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    sql = f"INSERT OR REPLACE INTO labels ({','.join(LABEL_COLUMNS)}) VALUES ({','.join('?' for _ in LABEL_COLUMNS)})"
    conn.executemany(sql, [[row.get(col) for col in LABEL_COLUMNS] for row in rows])


def _result_paths(config: TargetInstanceConfig, cache_key: str, n_examples: int) -> TargetInstanceResult:
    payload = {"attack_cache_key": cache_key, "n_examples": n_examples, **asdict(config)}
    run_dir = Path(config.output_dir) / f"target_instance_{stable_hash(payload)}"
    return TargetInstanceResult(
        run_dir=run_dir,
        db_path=run_dir / "target_instance.sqlite",
        labels_path=run_dir / "target_instance_labels.csv",
        summary_path=run_dir / "summary.json",
        digest_path=run_dir / "analysis_digest.md",
        config=config,
    )


def _result_row(example: Any, result: Any, config: TargetInstanceConfig) -> dict[str, Any]:
    target = example.clean_detection or {}
    target_box = np.asarray(target.get("bbox_xyxy_orig", []), dtype=np.float32)
    if target_box.size != 4:
        raise ValueError("clean_detection does not contain a four-coordinate target bbox")
    target_class = int(example.target_class_id if example.target_class_id is not None else target.get("class_id", 0))

    if result.boxes is None or len(result.boxes) == 0:
        boxes = np.zeros((0, 4), dtype=np.float32)
        confs = np.zeros(0, dtype=np.float32)
        classes = np.zeros(0, dtype=np.int64)
    else:
        boxes = result.boxes.xyxy.detach().float().cpu().numpy()
        confs = result.boxes.conf.detach().float().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    keep = classes == target_class
    boxes = boxes[keep]
    confs = confs[keep]
    ious = _box_iou_one_to_many(target_box, boxes)

    matched_idx = None
    valid_match = np.flatnonzero(ious >= float(config.match_iou))
    if len(valid_match):
        matched_idx = int(valid_match[int(np.argmax(confs[valid_match]))])
    matched_conf = float(confs[matched_idx]) if matched_idx is not None else 0.0
    matched_iou = float(ious[matched_idx]) if matched_idx is not None else float(ious.max(initial=0.0))
    matched_box = boxes[matched_idx].tolist() if matched_idx is not None else [None] * 4

    winner_idx = int(np.argmax(confs)) if len(confs) else None
    winner_conf = float(confs[winner_idx]) if winner_idx is not None else 0.0
    winner_iou = float(ious[winner_idx]) if winner_idx is not None else 0.0
    winner_box = boxes[winner_idx].tolist() if winner_idx is not None else [None] * 4

    clean_conf = float(example.conf_clean)
    eligible = clean_conf >= float(config.detection_conf)
    detected = matched_idx is not None and matched_conf >= float(config.detection_conf)
    hidden = bool(eligible and not detected)
    conf_drop = clean_conf - matched_conf
    suppressed = bool(eligible and conf_drop > float(config.target_suppression_delta))
    winner_is_target = bool(winner_idx is not None and winner_iou >= float(config.match_iou))
    if not eligible:
        outcome = "baseline_ineligible"
    elif hidden and winner_idx is None:
        outcome = "target_hidden_no_person"
    elif hidden:
        outcome = "target_hidden_non_target_winner"
    elif winner_is_target:
        outcome = "target_visible_target_winner"
    else:
        outcome = "target_visible_non_target_winner"

    row = {col: None for col in LABEL_COLUMNS}
    row.update({
        "image_key": image_key(example.path), "path": str(example.path), "example_id": _example_id(example),
        "legacy_success": int(bool(example.success)), "legacy_drop": float(example.drop),
        "target_policy": "clean_top_person_pre_attack", "target_class_id": target_class,
        "target_clean_conf": clean_conf,
        "target_clean_x1": float(target_box[0]), "target_clean_y1": float(target_box[1]),
        "target_clean_x2": float(target_box[2]), "target_clean_y2": float(target_box[3]),
        "target_eligible": int(eligible), "match_iou_threshold": float(config.match_iou),
        "detection_conf_threshold": float(config.detection_conf), "target_match_iou": matched_iou,
        "target_patched_conf": matched_conf, "target_conf_drop": conf_drop,
        "target_detected": int(detected), "target_hidden": int(hidden) if eligible else None,
        "target_suppressed_0p3": int(suppressed) if eligible else None,
        "matched_x1": matched_box[0], "matched_y1": matched_box[1],
        "matched_x2": matched_box[2], "matched_y2": matched_box[3],
        "patched_person_count": int(len(confs)), "patched_global_conf": winner_conf,
        "patched_winner_iou_target": winner_iou, "patched_winner_is_target": int(winner_is_target),
        "patched_winner_x1": winner_box[0], "patched_winner_y1": winner_box[1],
        "patched_winner_x2": winner_box[2], "patched_winner_y2": winner_box[3],
        "outcome": outcome, "error": None,
    })
    return row


def _write_label_summary(result: TargetInstanceResult, conn: sqlite3.Connection) -> None:
    labels = pd.read_sql_query("SELECT * FROM labels WHERE error IS NULL", conn)
    labels.to_csv(result.labels_path, index=False)
    eligible = labels[labels.target_eligible.eq(1)].copy()
    payload = {
        "status": "complete",
        "n_rows": int(len(labels)),
        "n_eligible": int(len(eligible)),
        "n_target_hidden": int(eligible.target_hidden.fillna(0).sum()),
        "target_hidden_rate": float(eligible.target_hidden.mean()) if len(eligible) else None,
        "legacy_success_rate": float(eligible.legacy_success.mean()) if len(eligible) else None,
        "legacy_target_agreement": float((eligible.legacy_success == eligible.target_hidden).mean()) if len(eligible) else None,
        "outcome_counts": {str(k): int(v) for k, v in eligible.outcome.value_counts().items()},
        "config": asdict(result.config),
        "labels_csv": str(result.labels_path),
    }
    write_json(result.summary_path, payload)
    write_markdown(result.digest_path, [
        "# Target-instance labels", "",
        f"- rows: {payload['n_rows']}",
        f"- baseline-eligible targets: {payload['n_eligible']}",
        f"- primary target-hidden rate: {payload['target_hidden_rate']:.3f}" if payload["target_hidden_rate"] is not None else "- primary target-hidden rate: n/a",
        f"- legacy max-confidence success rate: {payload['legacy_success_rate']:.3f}" if payload["legacy_success_rate"] is not None else "- legacy success rate: n/a",
        f"- legacy/target label agreement: {payload['legacy_target_agreement']:.3f}" if payload["legacy_target_agreement"] is not None else "- label agreement: n/a",
        "", "Primary success is target_hidden. legacy_success is retained only for comparison.",
    ])


def build_target_instance_labels(exp: Any, config: TargetInstanceConfig, *, force: bool = False) -> TargetInstanceResult:
    from new_experiments.patch_success_analysis.yolo import pil_to_np_bgr

    cache = exp.get_cache()
    selected = list(cache.examples)
    if config.max_examples is not None:
        selected = selected[: int(config.max_examples)]
    result = _result_paths(config, str(cache.cache_key), len(selected))
    result.run_dir.mkdir(parents=True, exist_ok=True)
    if force and result.db_path.exists():
        result.db_path.unlink()
    conn = connect_db(result.db_path)
    _create_label_schema(conn)
    conn.execute("INSERT OR REPLACE INTO metadata(key,value_json) VALUES (?,?)", ("config", json.dumps(asdict(config))))
    conn.commit()
    done = {row[0] for row in conn.execute("SELECT image_key FROM labels WHERE error IS NULL")}
    pending = [example for example in selected if image_key(example.path) not in done]

    yolo, _model = exp.load_model()
    progress = None
    if config.progress:
        try:
            from tqdm.auto import tqdm
            progress = tqdm(total=len(pending), desc="target-instance labeling", unit="img")
        except Exception:
            progress = None
    try:
        for start in range(0, len(pending), max(1, int(config.batch_size))):
            batch_examples = pending[start : start + max(1, int(config.batch_size))]
            valid_examples, patched_images, rows = [], [], []
            for example in batch_examples:
                try:
                    _clean, patched, _bbox = exp._images_for_example(example)
                    valid_examples.append(example)
                    patched_images.append(patched)
                except Exception as exc:  # noqa: BLE001
                    row = {col: None for col in LABEL_COLUMNS}
                    row.update({"image_key": image_key(example.path), "path": str(example.path), "example_id": _example_id(example), "error": f"{type(exc).__name__}: {exc}"})
                    rows.append(row)
            if valid_examples:
                results = yolo.predict(
                    source=[pil_to_np_bgr(image) for image in patched_images],
                    imgsz=int(exp.config.attack.imgsz), conf=float(config.inference_conf),
                    device=exp.config.attack.device, batch=max(1, int(config.batch_size)), verbose=False,
                )
                for example, prediction in zip(valid_examples, results, strict=True):
                    try:
                        rows.append(_result_row(example, prediction, config))
                    except Exception as exc:  # noqa: BLE001
                        row = {col: None for col in LABEL_COLUMNS}
                        row.update({"image_key": image_key(example.path), "path": str(example.path), "example_id": _example_id(example), "error": f"{type(exc).__name__}: {exc}"})
                        rows.append(row)
            _upsert_label_rows(conn, rows)
            conn.commit()
            if progress is not None:
                progress.update(len(batch_examples))
    finally:
        if progress is not None:
            progress.close()
        _write_label_summary(result, conn)
        conn.close()
    return result


def load_target_labels(result_or_path: TargetInstanceResult | str | Path) -> pd.DataFrame:
    path = result_or_path.labels_path if isinstance(result_or_path, TargetInstanceResult) else Path(result_or_path)
    labels = pd.read_csv(path)
    for col in ("legacy_success", "target_eligible", "target_detected", "target_hidden", "target_suppressed_0p3", "patched_winner_is_target"):
        if col in labels.columns:
            labels[col] = pd.to_numeric(labels[col], errors="coerce").astype("Int64")
    return labels


def _context_subset(rows: pd.DataFrame, quality_row: pd.Series) -> pd.DataFrame:
    out = rows
    for col in ("cov_method", "sigma", "cov_group", "top_percent", "aggregate"):
        if col not in out.columns or col not in quality_row.index or pd.isna(quality_row[col]):
            continue
        value = quality_row[col]
        if pd.api.types.is_numeric_dtype(out[col]):
            out = out[np.isclose(pd.to_numeric(out[col], errors="coerce"), float(value), equal_nan=False)]
        else:
            out = out[out[col].astype(str).eq(str(value))]
    return out


def _correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame.x.nunique() < 2 or frame.y.nunique() < 2:
        return {"pearson": np.nan, "spearman": np.nan}
    return {"pearson": float(frame.x.corr(frame.y, method="pearson")), "spearman": float(frame.x.corr(frame.y, method="spearman"))}


def reevaluate_historical_detector_metrics(
    labels: pd.DataFrame,
    *,
    legacy_quality_path: str | Path,
    output_dir: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    from ClassifierDetectorExperiments.side_by_side_analysis import best_f1_threshold, roc_auc_score_manual

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_path = output_dir / "target_instance_all_metrics_quality.csv"
    regression_path = output_dir / "target_instance_all_metrics_regression.csv"
    coverage_path = output_dir / "target_instance_all_metrics_coverage.csv"
    if not force and quality_path.exists() and regression_path.exists() and coverage_path.exists():
        return {"quality": pd.read_csv(quality_path), "regression": pd.read_csv(regression_path), "coverage": pd.read_csv(coverage_path), "loaded_from_cache": True}

    label_cols = ["image_key", "target_hidden", "target_conf_drop", "target_eligible", "legacy_success"]
    label_table = labels[label_cols].copy()
    label_table = label_table[label_table.target_eligible.eq(1) & label_table.target_hidden.notna()]
    legacy = pd.read_csv(legacy_quality_path, low_memory=False)
    legacy = legacy[legacy.source.astype(str).eq("detector")].copy()
    quality_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for cache_file, quality_spec in legacy.groupby("cache_file", dropna=False, sort=True):
        cache_path = Path(str(cache_file))
        if not cache_path.exists() or cache_path.suffix != ".pkl":
            coverage_rows.append({"cache_file": str(cache_file), "status": "missing_or_unsupported"})
            continue
        try:
            with cache_path.open("rb") as fh:
                payload = pickle.load(fh)
            rows = pd.DataFrame(payload.get("rows", [])) if isinstance(payload, dict) else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            coverage_rows.append({"cache_file": str(cache_file), "status": f"load_error:{type(exc).__name__}"})
            continue
        if rows.empty or "path" not in rows.columns:
            coverage_rows.append({"cache_file": str(cache_file), "status": "no_rows"})
            continue
        rows = rows.copy()
        source_n = int(len(rows))
        rows["image_key"] = rows.path.map(image_key)
        rows = rows.merge(label_table, on="image_key", how="inner", validate="many_to_one")
        coverage_rows.append({
            "cache_file": str(cache_file), "status": "ok", "source_rows": source_n,
            "matched_rows": int(len(rows)), "matched_unique_images": int(rows.image_key.nunique()),
        })
        for _, spec in quality_spec.iterrows():
            metric = str(spec.metric)
            if metric not in rows.columns:
                continue
            sub = _context_subset(rows, spec)
            values = pd.to_numeric(sub[metric], errors="coerce")
            valid = values.notna() & sub.target_hidden.notna()
            if int(valid.sum()) < 20:
                continue
            y = sub.loc[valid, "target_hidden"].astype(bool).to_numpy()
            x = values.loc[valid].to_numpy(dtype=float)
            if int(y.sum()) == 0 or int((~y).sum()) == 0:
                continue
            fit = best_f1_threshold(y, x)
            record = {
                "source": "detector_target_instance", "experiment": spec.experiment,
                "metric": metric, "n": int(len(x)), "n_success": int(y.sum()), "n_failure": int((~y).sum()),
                "roc_auc": roc_auc_score_manual(y, x), "cache_file": str(cache_file),
                "legacy_roc_auc": spec.get("roc_auc", np.nan),
            }
            for col in ("family", "top_percent", "aggregate", "cov_method", "sigma", "cov_group"):
                record[col] = spec.get(col, np.nan)
            record.update(fit)
            quality_rows.append(record)
            corr = _correlations(x, sub.loc[valid, "target_conf_drop"].to_numpy(dtype=float))
            regression_rows.append({
                "source": "detector_target_instance", "experiment": spec.experiment, "metric": metric,
                "n": int(len(x)), "cache_file": str(cache_file), **{col: spec.get(col, np.nan) for col in ("family", "top_percent", "aggregate", "cov_method", "sigma", "cov_group")},
                **corr, "abs_pearson": abs(corr["pearson"]), "abs_spearman": abs(corr["spearman"]),
                "main_score": abs(corr["spearman"]),
            })

    quality = pd.DataFrame(quality_rows).sort_values(["best_balanced_accuracy", "roc_auc"], ascending=False, na_position="last") if quality_rows else pd.DataFrame()
    regression = pd.DataFrame(regression_rows).sort_values("main_score", ascending=False, na_position="last") if regression_rows else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    quality.to_csv(quality_path, index=False)
    regression.to_csv(regression_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    return {"quality": quality, "regression": regression, "coverage": coverage, "loaded_from_cache": False}


def evaluate_candidate_routing_metrics(
    labels: pd.DataFrame,
    *,
    trace_db: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    from ClassifierDetectorExperiments.side_by_side_analysis import best_f1_threshold, roc_auc_score_manual

    conn = connect_db(Path(trace_db))
    try:
        rows = pd.read_sql_query("SELECT * FROM examples WHERE error IS NULL", conn)
    finally:
        conn.close()
    rows["image_key"] = rows.path.map(image_key)
    rows = rows.merge(labels[["image_key", "target_hidden", "target_eligible"]], on="image_key", how="inner")
    rows = rows[rows.target_eligible.eq(1) & rows.target_hidden.notna()].copy()
    excluded = {
        "success", "confidence_drop", "conf_clean_cached", "conf_patch_cached", "conf_clean_recomputed",
        "conf_patch_recomputed", "clean_conf_abs_error", "patch_conf_abs_error", "target_hidden", "target_eligible",
        "class_id", "patch_x1", "patch_y1", "patch_x2", "patch_y2", "error",
    }
    metric_cols = [
        col for col in rows.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(rows[col]) and rows[col].notna().sum() >= 20
    ]
    y = rows.target_hidden.astype(bool).to_numpy()
    out = []
    for metric in metric_cols:
        values = pd.to_numeric(rows[metric], errors="coerce")
        valid = values.notna()
        yy, xx = y[valid.to_numpy()], values[valid].to_numpy(dtype=float)
        if len(xx) < 20 or yy.sum() == 0 or (~yy).sum() == 0 or float(np.nanstd(xx)) == 0.0:
            continue
        out.append({"source": "candidate_routing", "experiment": "candidate_routing", "metric": metric, "n": len(xx), "roc_auc": roc_auc_score_manual(yy, xx), **best_f1_threshold(yy, xx)})
    result = pd.DataFrame(out).sort_values("best_balanced_accuracy", ascending=False, na_position="last") if out else pd.DataFrame()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "target_instance_candidate_routing_quality.csv", index=False)
    return result


def evaluate_attack_path_metrics(
    labels: pd.DataFrame,
    *,
    attack_path_db: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    from ClassifierDetectorExperiments.side_by_side_analysis import best_f1_threshold, roc_auc_score_manual

    conn = connect_db(Path(attack_path_db))
    try:
        rows = pd.read_sql_query("SELECT * FROM path_examples WHERE error IS NULL", conn)
    finally:
        conn.close()
    rows["image_key"] = rows.path.map(image_key)
    rows = rows.merge(labels[["image_key", "target_hidden", "target_eligible"]], on="image_key", how="inner")
    rows = rows[rows.target_eligible.eq(1) & rows.target_hidden.notna()].copy()
    excluded = {
        "success", "confidence_drop", "target_hidden", "target_eligible", "n_steps", "error",
    }
    out = []
    for target_kind, sub in rows.groupby("target_kind", sort=True):
        y = sub.target_hidden.astype(bool).to_numpy()
        metric_cols = [
            col for col in sub.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(sub[col]) and sub[col].notna().sum() >= 20
        ]
        for metric in metric_cols:
            values = pd.to_numeric(sub[metric], errors="coerce")
            valid = values.notna()
            yy, xx = y[valid.to_numpy()], values[valid].to_numpy(dtype=float)
            if len(xx) < 20 or yy.sum() == 0 or (~yy).sum() == 0 or float(np.nanstd(xx)) == 0.0:
                continue
            out.append({
                "source": "attack_path", "experiment": f"attack_path_{target_kind}",
                "metric": metric, "n": len(xx), "roc_auc": roc_auc_score_manual(yy, xx),
                **best_f1_threshold(yy, xx),
            })
    result = pd.DataFrame(out).sort_values("best_balanced_accuracy", ascending=False, na_position="last") if out else pd.DataFrame()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "target_instance_attack_path_quality.csv", index=False)
    return result
