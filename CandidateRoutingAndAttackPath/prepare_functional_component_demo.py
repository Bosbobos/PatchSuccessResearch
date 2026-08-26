from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .candidate_reserve import _cache_lookup
from .candidate_routing import _box_iou, _xywh_to_xyxy
from .causal_repair import _load_inputs
from .common import load_experiment
from .followup_common import ATTACK_PATH_DB, MANIFEST_CSV, TRACE_DB
from .full_success_closure import (
    FullSuccessClosureConfig,
    _candidate_closure,
    _local_indices,
    _path_jacobians,
)
from .mechanism_followup import _decode, _head_branches, _raw_from_branches


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "demo_artifacts" / "functional_component"
DEFAULT_EXAMPLE_ID = "2241a17c14b865cd"


def prepare_demo(example_id: str = DEFAULT_EXAMPLE_ID, *, path_steps: int = 3) -> Path:
    import torch

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected, _ = _load_inputs(Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None)
    matches = selected[selected.example_id.astype(str).eq(str(example_id))]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for {example_id}, got {len(matches)}")
    row = matches.iloc[0]

    exp, cache_path = load_experiment(prefer_device="cpu", require_device=False)
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    example = _cache_lookup(exp)[str(example_id)]
    clean_image, patched_image, _ = exp._images_for_example(example)
    clean_image.save(OUTPUT_DIR / "clean.png")
    patched_image.save(OUTPUT_DIR / "patched.png")

    pair = _preprocess_pair(exp, clean_image, patched_image)
    captured = _capture_detect_inputs(model, detect, pair)
    clean_inputs = [item[0:1] for item in captured]
    patched_inputs = [item[1:2] for item in captured]
    with torch.no_grad():
        clean_box, clean_cls, _ = _head_branches(detect, clean_inputs)
        patched_box, patched_cls, _ = _head_branches(detect, patched_inputs)
        clean_raw = _raw_from_branches(clean_box, clean_cls)
        patched_raw = _raw_from_branches(patched_box, patched_cls)
        clean_decoded = _decode(detect, clean_raw)
        patched_decoded = _decode(detect, patched_raw)

    config = FullSuccessClosureConfig(
        device="cpu",
        path_steps=int(path_steps),
        nms_max_time_img=1.0,
    )
    selection = _candidate_closure(detect, clean_raw, patched_raw, row, config)
    if selection.empty:
        raise RuntimeError("Candidate reserve is empty")
    level = int(selection.groupby("level_index").size().idxmax())
    subset = selection[selection.level_index.astype(int).eq(level)].reset_index(drop=True)
    target_box = torch.as_tensor(
        [[row.clean_target_x1, row.clean_target_y1, row.clean_target_x2, row.clean_target_y2]],
        device=clean_inputs[level].device,
        dtype=torch.float32,
    )
    score_j_full, geometry_j_full = _path_jacobians(
        detect,
        level,
        clean_inputs[level],
        patched_inputs[level],
        subset,
        int(row.class_id),
        target_box,
        clean_box,
        clean_cls,
        int(path_steps),
    )
    indices = _local_indices(clean_inputs[level][0].shape, subset, radius=2)
    score_j = score_j_full.cpu().numpy().reshape(len(subset), -1)[:, indices].astype(np.float64)
    geometry_j = (
        geometry_j_full.cpu().numpy().reshape(len(subset), -1)[:, indices].astype(np.float64)
    )
    jacobian = np.concatenate([score_j, geometry_j], axis=0)
    u, singular, vh = np.linalg.svd(jacobian, full_matrices=False)
    tolerance = max(jacobian.shape) * np.finfo(np.float64).eps * max(
        float(singular.max()), 1.0
    )
    rank = int((singular > tolerance).sum())
    delta_full = (
        patched_inputs[level][0] - clean_inputs[level][0]
    ).detach().float().cpu().numpy().reshape(-1)
    delta = delta_full[indices].astype(np.float64)
    functional = vh[:rank].T @ (vh[:rank] @ delta)
    residual = delta - functional

    flats = subset.flat_index.astype(int).to_numpy()
    clean_boxes = _xywh_to_xyxy(clean_decoded[0, :4, flats].transpose(0, 1))
    patched_boxes = _xywh_to_xyxy(patched_decoded[0, :4, flats].transpose(0, 1))
    clean_scores = clean_decoded[0, 4 + int(row.class_id), flats]
    patched_scores = patched_decoded[0, 4 + int(row.class_id), flats]
    clean_ious = _box_iou(clean_boxes, target_box).reshape(-1)
    patched_ious = _box_iou(patched_boxes, target_box).reshape(-1)
    ys = subset.y_index.astype(int).to_numpy()
    xs = subset.x_index.astype(int).to_numpy()
    clean_logits = clean_cls[level][0, int(row.class_id), ys, xs]
    patched_logits = patched_cls[level][0, int(row.class_id), ys, xs]

    candidates = subset[
        ["flat_index", "level_index", "y_index", "x_index"]
    ].copy()
    for prefix, boxes, scores, ious, logits in (
        ("clean", clean_boxes, clean_scores, clean_ious, clean_logits),
        ("patched", patched_boxes, patched_scores, patched_ious, patched_logits),
    ):
        boxes_np = boxes.detach().float().cpu().numpy()
        candidates[f"{prefix}_x1"] = boxes_np[:, 0]
        candidates[f"{prefix}_y1"] = boxes_np[:, 1]
        candidates[f"{prefix}_x2"] = boxes_np[:, 2]
        candidates[f"{prefix}_y2"] = boxes_np[:, 3]
        candidates[f"{prefix}_score"] = scores.detach().float().cpu().numpy()
        candidates[f"{prefix}_iou"] = ious.detach().float().cpu().numpy()
        candidates[f"{prefix}_logit"] = logits.detach().float().cpu().numpy()
    candidates.to_csv(OUTPUT_DIR / "candidates.csv", index=False)

    f_clean = np.concatenate(
        [candidates.clean_logit.to_numpy(float), candidates.clean_iou.to_numpy(float)]
    )
    f_patched = np.concatenate(
        [candidates.patched_logit.to_numpy(float), candidates.patched_iou.to_numpy(float)]
    )
    np.savez_compressed(
        OUTPUT_DIR / "svd_demo.npz",
        jacobian=jacobian.astype(np.float32),
        U=u.astype(np.float32),
        singular=singular.astype(np.float32),
        Vt=vh.astype(np.float32),
        delta=delta.astype(np.float32),
        functional=functional.astype(np.float32),
        residual=residual.astype(np.float32),
        f_clean=f_clean.astype(np.float32),
        f_patched=f_patched.astype(np.float32),
        selected_flat_indices=indices.astype(np.int64),
    )
    total_energy = sum(
        float(torch.square((patched - clean).detach().float()).sum().cpu())
        for clean, patched in zip(clean_inputs, patched_inputs, strict=True)
    )
    metadata = {
        "example_id": str(example_id),
        "analysis_group": str(row.analysis_group),
        "cache_path": str(cache_path),
        "level_index": level,
        "level_name": f"P{level + 3}",
        "n_candidates": int(len(subset)),
        "local_dimension": int(len(delta)),
        "jacobian_shape": list(jacobian.shape),
        "U_shape": list(u.shape),
        "Sigma_shape": [int(len(singular)), int(len(singular))],
        "Vt_shape": list(vh.shape),
        "numerical_rank": rank,
        "path_steps": int(path_steps),
        "target_box": [
            float(row.clean_target_x1),
            float(row.clean_target_y1),
            float(row.clean_target_x2),
            float(row.clean_target_y2),
        ],
        "local_delta_energy": float(np.dot(delta, delta)),
        "functional_energy": float(np.dot(functional, functional)),
        "residual_energy": float(np.dot(residual, residual)),
        "functional_fraction_of_local": float(
            np.dot(functional, functional) / max(np.dot(delta, delta), 1e-12)
        ),
        "functional_fraction_of_full": float(
            np.dot(functional, functional) / max(total_energy, 1e-12)
        ),
        "linear_completeness_relative_error": float(
            np.linalg.norm(jacobian @ delta - (f_patched - f_clean))
            / max(np.linalg.norm(f_patched - f_clean), 1e-12)
        ),
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return OUTPUT_DIR


if __name__ == "__main__":
    print(prepare_demo())
