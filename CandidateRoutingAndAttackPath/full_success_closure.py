from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .attack_path import _capture_detect_inputs, _preprocess_pair
from .candidate_reserve import _cache_lookup, _evaluate_batch
from .candidate_routing import _box_iou, _flat_location, _level_slices, _xywh_to_xyxy
from .causal_repair import _load_inputs
from .common import StorageBudget, load_experiment, release_accelerator_memory, stable_hash
from .followup_common import (
    ATTACK_PATH_DB,
    FOLLOWUP_DIR,
    GROUP_ORDER,
    MANIFEST_CSV,
    TRACE_DB,
    balanced_subset,
    write_summary,
)
from .mechanism_followup import _decode, _head_branches, _raw_from_branches


@dataclass(slots=True)
class FullSuccessClosureConfig:
    output_dir: str = str(FOLLOWUP_DIR)
    device: str = "cpu"
    require_device: bool = False
    closure_examples_per_group: int = 100
    functional_examples_per_group: int = 25
    path_steps: int = 3
    radii: tuple[int, ...] = (0, 1, 2, 4, 8, 16)
    target_iou: float = 0.50
    detection_conf: float = 0.25
    candidate_min_score: float = 0.01
    nms_conf: float = 0.01
    nms_iou: float = 0.70
    nms_max_det: int = 300
    nms_max_time_img: float = 1.0
    seed: int = 509
    max_output_gb: float = 1.0
    method_version: int = 1


def _candidate_closure(detect, clean_raw, patched_raw, row, config) -> pd.DataFrame:
    """All clean target candidates plus endpoint candidates needed for closure.

    The clean half is logically sufficient for clean->patched hiding: every cell
    that can currently detect the fixed target is included.  Patched endpoint
    cells are added to make repair and mixed-branch interventions symmetric.
    """

    import torch

    class_id = int(row.class_id)
    target_box = torch.as_tensor(
        [[row.clean_target_x1, row.clean_target_y1, row.clean_target_x2, row.clean_target_y2]],
        device=clean_raw[0].device,
        dtype=torch.float32,
    )
    clean_decoded = _decode(detect, clean_raw)
    patched_decoded = _decode(detect, patched_raw)
    clean_boxes = _xywh_to_xyxy(clean_decoded[0, :4].transpose(0, 1))
    patched_boxes = _xywh_to_xyxy(patched_decoded[0, :4].transpose(0, 1))
    clean_scores = clean_decoded[0, 4 + class_id]
    patched_scores = patched_decoded[0, 4 + class_id]
    clean_iou = _box_iou(clean_boxes, target_box).reshape(-1)
    patched_iou = _box_iou(patched_boxes, target_box).reshape(-1)
    relevant = (
        ((clean_iou >= float(config.target_iou)) & (clean_scores >= float(config.candidate_min_score)))
        | ((patched_iou >= float(config.target_iou)) & (patched_scores >= float(config.candidate_min_score)))
    )
    tracked = int(row.clean_target_flat)
    relevant[tracked] = True
    slices = _level_slices(clean_raw)
    records = []
    for flat in torch.nonzero(relevant, as_tuple=False).reshape(-1).tolist():
        level, y, x = _flat_location(int(flat), slices)
        records.append({
            "flat_index": int(flat),
            "level_index": int(level),
            "y_index": int(y),
            "x_index": int(x),
            "clean_score": float(clean_scores[flat].detach().cpu()),
            "patched_score": float(patched_scores[flat].detach().cpu()),
            "clean_iou": float(clean_iou[flat].detach().cpu()),
            "patched_iou": float(patched_iou[flat].detach().cpu()),
        })
    return pd.DataFrame(records).sort_values(
        ["clean_score", "patched_score"], ascending=False
    ).reset_index(drop=True)


def _replace_selected(base, source, selection: pd.DataFrame):
    output = [item.clone() for item in base]
    for item in selection.itertuples(index=False):
        level, y, x = int(item.level_index), int(item.y_index), int(item.x_index)
        output[level][0, :, y, x] = source[level][0, :, y, x]
    return output


def _exact_output_conditions(clean_box, clean_cls, patched_box, patched_cls, selection):
    candidate_box = _replace_selected(clean_box, patched_box, selection)
    candidate_cls = _replace_selected(clean_cls, patched_cls, selection)
    return {
        "clean": _raw_from_branches(clean_box, clean_cls),
        "candidate_score": _raw_from_branches(clean_box, candidate_cls),
        "candidate_geometry": _raw_from_branches(candidate_box, clean_cls),
        "candidate_both": _raw_from_branches(candidate_box, candidate_cls),
        "global_score": _raw_from_branches(clean_box, patched_cls),
        "global_geometry": _raw_from_branches(patched_box, clean_cls),
        "global_both": _raw_from_branches(patched_box, patched_cls),
    }


def _stack_raw(conditions: dict[str, list]):
    names = list(conditions)
    levels = [
        __import__("torch").cat([conditions[name][level] for name in names], dim=0)
        for level in range(len(next(iter(conditions.values()))))
    ]
    return names, levels


def _mechanism_label(frame: pd.DataFrame) -> str:
    values = frame.set_index("condition").target_hidden.astype(bool)
    if not bool(values.get("global_both", False)):
        return "endpoint_not_hidden"
    if not bool(values.get("candidate_both", False)):
        return "candidate_closure_failure"
    score = bool(values.get("candidate_score", False))
    geometry = bool(values.get("candidate_geometry", False))
    if score and geometry:
        label = "redundant_score_or_geometry"
    elif score:
        label = "score_only"
    elif geometry:
        label = "geometry_only"
    else:
        label = "cooperative_score_and_geometry"
    candidate_both = frame[frame.condition.eq("candidate_both")].iloc[0]
    if bool(candidate_both.nms_only_hidden):
        return f"{label}+nms"
    return label


def _spatial_components(clean_inputs, patched_inputs, selection, config):
    import torch

    names = [f"radius_{int(radius)}" for radius in config.radii] + ["full_maps"]
    maps = {
        name: [torch.zeros_like(item[0], dtype=torch.float32, device="cpu") for item in clean_inputs]
        for name in names
    }
    total_energy = sum(
        float(torch.square((patched - clean).detach().float()).sum().cpu())
        for clean, patched in zip(clean_inputs, patched_inputs, strict=True)
    )
    energy = {}
    coverage = {}
    for radius in config.radii:
        name = f"radius_{int(radius)}"
        selected_coordinates = selection.groupby("level_index")
        used = 0
        total = 0
        component_energy = 0.0
        for level, (clean, patched) in enumerate(zip(clean_inputs, patched_inputs, strict=True)):
            delta = (patched[0] - clean[0]).detach().float().cpu()
            mask = torch.zeros(delta.shape[-2:], dtype=torch.bool)
            if level in selected_coordinates.groups:
                rows = selection.loc[selected_coordinates.groups[level]]
                for item in rows.itertuples(index=False):
                    y, x = int(item.y_index), int(item.x_index)
                    y1, y2 = max(0, y - radius), min(mask.shape[0], y + radius + 1)
                    x1, x2 = max(0, x - radius), min(mask.shape[1], x + radius + 1)
                    mask[y1:y2, x1:x2] = True
            component = delta * mask.unsqueeze(0)
            maps[name][level] = component
            component_energy += float(torch.square(component).sum())
            used += int(mask.sum()) * int(delta.shape[0])
            total += int(delta.numel())
        energy[name] = component_energy / max(total_energy, 1e-12)
        coverage[name] = used / max(total, 1)
    for level, (clean, patched) in enumerate(zip(clean_inputs, patched_inputs, strict=True)):
        maps["full_maps"][level] = (patched[0] - clean[0]).detach().float().cpu()
    energy["full_maps"] = 1.0
    coverage["full_maps"] = 1.0
    return maps, energy, coverage


def _batched_input_interventions(clean_inputs, patched_inputs, maps, direction):
    import torch

    names = ["none", *maps]
    levels = []
    for level in range(len(clean_inputs)):
        values = []
        for name in names:
            base = patched_inputs[level] if direction == "repair_patched" else clean_inputs[level]
            if name == "none":
                values.append(base)
            else:
                component = maps[name][level].to(base.device, base.dtype).unsqueeze(0)
                values.append(base - component if direction == "repair_patched" else base + component)
        levels.append(torch.cat(values, dim=0))
    return names, levels


def _local_indices(shape, selection: pd.DataFrame, radius: int = 2) -> np.ndarray:
    channels, height, width = (int(value) for value in shape)
    spatial = np.zeros((height, width), dtype=bool)
    for item in selection.itertuples(index=False):
        y, x = int(item.y_index), int(item.x_index)
        spatial[
            max(0, y - radius):min(height, y + radius + 1),
            max(0, x - radius):min(width, x + radius + 1),
        ] = True
    return np.flatnonzero(np.broadcast_to(spatial, (channels, height, width)).reshape(-1))


def _path_jacobians(
    detect,
    level: int,
    clean,
    patched,
    selection: pd.DataFrame,
    class_id: int,
    target_box,
    fixed_box,
    fixed_cls,
    steps: int,
):
    """Path-integrated score-logit and decoded-IoU Jacobians."""

    import torch

    local_flats = selection.flat_index.astype(int).to_numpy()
    ys = torch.as_tensor(selection.y_index.astype(int).to_numpy(), device=clean.device)
    xs = torch.as_tensor(selection.x_index.astype(int).to_numpy(), device=clean.device)
    score_accumulated = None
    iou_accumulated = None
    for step in range(int(steps)):
        alpha = (step + 0.5) / float(steps)
        value = (clean + alpha * (patched - clean)).detach().requires_grad_(True)
        box_level = detect.cv2[level](value)
        cls_level = detect.cv3[level](value)
        logits = cls_level[0, int(class_id), ys, xs]
        raw = [
            __import__("torch").cat((
                box_level if index == level else fixed_box[index],
                cls_level if index == level else fixed_cls[index],
            ), dim=1)
            for index in range(len(fixed_box))
        ]
        decoded = detect._inference(raw)
        boxes = _xywh_to_xyxy(decoded[0, :4, local_flats].transpose(0, 1))
        ious = _box_iou(boxes, target_box).reshape(-1)
        score_basis = torch.eye(len(logits), device=value.device, dtype=value.dtype)
        iou_basis = torch.eye(len(ious), device=value.device, dtype=value.dtype)
        score_gradient = torch.autograd.grad(
            logits, value, grad_outputs=score_basis, is_grads_batched=True,
            retain_graph=True, create_graph=False,
        )[0][:, 0].detach().float()
        iou_gradient = torch.autograd.grad(
            ious, value, grad_outputs=iou_basis, is_grads_batched=True,
            retain_graph=False, create_graph=False,
        )[0][:, 0].detach().float()
        score_accumulated = score_gradient if score_accumulated is None else score_accumulated + score_gradient
        iou_accumulated = iou_gradient if iou_accumulated is None else iou_accumulated + iou_gradient
    return score_accumulated / float(steps), iou_accumulated / float(steps)


def _rowspace_projection(jacobian: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, int]:
    if not jacobian.size:
        return np.zeros_like(delta, dtype=np.float32), 0
    _u, singular, vh = np.linalg.svd(jacobian.astype(np.float64), full_matrices=False)
    tolerance = max(jacobian.shape) * np.finfo(np.float64).eps * max(float(singular.max()), 1.0)
    rank = int((singular > tolerance).sum())
    basis = vh[:rank]
    projection = basis.T @ (basis @ delta.astype(np.float64))
    return projection.astype(np.float32), rank


def _functional_components(
    detect,
    clean_inputs,
    patched_inputs,
    clean_box,
    clean_cls,
    patched_box,
    patched_cls,
    selection,
    row,
    config,
):
    import torch

    names = [
        "score_rowspace", "geometry_rowspace", "joint_rowspace",
        "joint_null", "full_candidate_windows", "full_maps",
    ]
    maps = {
        name: [torch.zeros_like(item[0], dtype=torch.float32, device="cpu") for item in clean_inputs]
        for name in names
    }
    target_box = torch.as_tensor(
        [[row.clean_target_x1, row.clean_target_y1, row.clean_target_x2, row.clean_target_y2]],
        device=clean_inputs[0].device, dtype=torch.float32,
    )
    total_energy = sum(
        float(torch.square((patched - clean).detach().float()).sum().cpu())
        for clean, patched in zip(clean_inputs, patched_inputs, strict=True)
    )
    local_energy = 0.0
    score_errors = []
    geometry_errors = []
    score_cosines = []
    geometry_cosines = []
    ranks = {"score": 0, "geometry": 0, "joint": 0}
    for level in range(len(clean_inputs)):
        subset = selection[selection.level_index.astype(int).eq(level)].reset_index(drop=True)
        if subset.empty:
            continue
        indices = _local_indices(clean_inputs[level][0].shape, subset, radius=2)
        delta_full = (
            patched_inputs[level][0] - clean_inputs[level][0]
        ).detach().float().cpu().numpy().reshape(-1)
        delta = delta_full[indices].astype(np.float32)
        score_j_full, geometry_j_full = _path_jacobians(
            detect, level, clean_inputs[level], patched_inputs[level], subset,
            int(row.class_id), target_box, clean_box, clean_cls, int(config.path_steps),
        )
        score_j = score_j_full.cpu().numpy().reshape(len(subset), -1)[:, indices]
        geometry_j = geometry_j_full.cpu().numpy().reshape(len(subset), -1)[:, indices]
        score_projection, score_rank = _rowspace_projection(score_j, delta)
        geometry_projection, geometry_rank = _rowspace_projection(geometry_j, delta)
        joint_projection, joint_rank = _rowspace_projection(
            np.concatenate([score_j, geometry_j], axis=0), delta
        )
        ranks["score"] += score_rank
        ranks["geometry"] += geometry_rank
        ranks["joint"] += joint_rank
        with torch.no_grad():
            ys = subset.y_index.astype(int).to_numpy()
            xs = subset.x_index.astype(int).to_numpy()
            exact_score = (
                patched_cls[level][0, int(row.class_id), ys, xs]
                - clean_cls[level][0, int(row.class_id), ys, xs]
            ).float().cpu().numpy()
            clean_raw = _raw_from_branches(clean_box, clean_cls)
            patched_raw = _raw_from_branches(patched_box, patched_cls)
            clean_decoded = _decode(detect, clean_raw)
            patched_decoded = _decode(detect, patched_raw)
            flats = subset.flat_index.astype(int).to_numpy()
            clean_boxes = _xywh_to_xyxy(clean_decoded[0, :4, flats].transpose(0, 1))
            patched_boxes = _xywh_to_xyxy(patched_decoded[0, :4, flats].transpose(0, 1))
            exact_geometry = (
                _box_iou(patched_boxes, target_box) - _box_iou(clean_boxes, target_box)
            ).reshape(-1).float().cpu().numpy()
        for jacobian, exact, errors, cosines in (
            (score_j, exact_score, score_errors, score_cosines),
            (geometry_j, exact_geometry, geometry_errors, geometry_cosines),
        ):
            predicted = jacobian.astype(np.float64) @ delta.astype(np.float64)
            errors.append(
                float(np.linalg.norm(predicted - exact) / max(np.linalg.norm(exact), 1e-12))
            )
            cosines.append(
                float(np.dot(predicted, exact)
                      / max(np.linalg.norm(predicted) * np.linalg.norm(exact), 1e-12))
            )
        components = {
            "score_rowspace": score_projection,
            "geometry_rowspace": geometry_projection,
            "joint_rowspace": joint_projection,
            "joint_null": delta - joint_projection,
            "full_candidate_windows": delta,
        }
        local_energy += float(np.square(delta.astype(np.float64)).sum())
        for name, values in components.items():
            full = np.zeros_like(delta_full, dtype=np.float32)
            full[indices] = values
            maps[name][level] = torch.from_numpy(
                full.reshape(tuple(int(value) for value in clean_inputs[level][0].shape))
            )
    for level, (clean, patched) in enumerate(zip(clean_inputs, patched_inputs, strict=True)):
        maps["full_maps"][level] = (patched[0] - clean[0]).detach().float().cpu()
    energy = {
        name: float(sum(np.square(item.numpy().astype(np.float64)).sum() for item in level_maps))
        / max(total_energy, 1e-12)
        for name, level_maps in maps.items()
    }
    metadata = {
        "n_candidates": int(len(selection)),
        "local_energy_fraction": float(local_energy / max(total_energy, 1e-12)),
        "score_rowspace_rank": int(ranks["score"]),
        "geometry_rowspace_rank": int(ranks["geometry"]),
        "joint_rowspace_rank": int(ranks["joint"]),
        "score_completeness_error": float(np.mean(score_errors)) if score_errors else np.nan,
        "geometry_completeness_error": float(np.mean(geometry_errors)) if geometry_errors else np.nan,
        "score_completeness_cosine": float(np.mean(score_cosines)) if score_cosines else np.nan,
        "geometry_completeness_cosine": float(np.mean(geometry_cosines)) if geometry_cosines else np.nan,
    }
    return maps, energy, metadata


def _summarize(rows: pd.DataFrame, kind: str) -> pd.DataFrame:
    records = []
    keys = ["analysis_group", "direction", "condition"] if "direction" in rows else [
        "analysis_group", "condition"
    ]
    groups = [(key, frame) for key, frame in rows.groupby(keys, sort=False)]
    all_keys = ["direction", "condition"] if "direction" in rows else ["condition"]
    groups += [(("all", *key) if isinstance(key, tuple) else ("all", key), frame)
               for key, frame in rows.groupby(all_keys, sort=False)]
    for key, frame in groups:
        if not isinstance(key, tuple):
            key = (key,)
        labels = ["analysis_group", *all_keys] if len(key) == len(all_keys) + 1 else keys
        record = dict(zip(labels, key, strict=True))
        baseline_hidden = frame.baseline_hidden.astype(bool) if "baseline_hidden" in frame else None
        source_hidden = frame.source_hidden.astype(bool) if "source_hidden" in frame else baseline_hidden
        record.update({
            "n": int(frame.example_id.nunique()),
            "target_hidden_rate": float(frame.target_hidden.mean()),
            "pre_hidden_rate": float(1.0 - frame.pre_target_detected.mean()),
            "nms_only_hidden_rate": float(frame.nms_only_hidden.mean()),
            "mean_post_target_conf": float(frame.post_target_conf.mean()),
            "mean_post_target_iou": float(frame.post_target_iou.mean()),
        })
        if "component_energy_fraction" in frame:
            record["mean_component_energy_fraction"] = float(frame.component_energy_fraction.mean())
        if "spatial_coverage_fraction" in frame:
            record["mean_spatial_coverage_fraction"] = float(frame.spatial_coverage_fraction.mean())
        if baseline_hidden is not None:
            eligible = source_hidden if source_hidden is not None else baseline_hidden
            record["recovery_rate"] = (
                float(frame.loc[eligible, "target_detected"].mean())
                if eligible.any() else np.nan
            )
            record["reproduced_hiding_rate"] = (
                float(frame.loc[eligible, "target_hidden"].mean())
                if eligible.any() else np.nan
            )
            record["visible_source_false_hiding_rate"] = (
                float(frame.loc[~eligible, "target_hidden"].mean())
                if (~eligible).any() else np.nan
            )
        records.append(record)
    output = pd.DataFrame(records)
    output.insert(0, "experiment_kind", kind)
    return output


def _activation_mechanisms(functional: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    transplant = functional[
        functional.direction.eq("transplant_clean") & functional.source_hidden.astype(bool)
    ]
    pivot = transplant.pivot(
        index=["example_id", "analysis_group"],
        columns="condition",
        values="target_hidden",
    ).reset_index()

    def label(row) -> str:
        joint = bool(row.joint_rowspace)
        local = bool(row.full_candidate_windows)
        if joint and local:
            return "direct_joint_sufficiency"
        if joint and not local:
            return "joint_sufficient_but_local_residual_antagonizes"
        if local:
            return "joint_plus_local_nonlinear_synergy"
        return "outer_context_required"

    pivot["activation_mechanism"] = pivot.apply(label, axis=1)
    summary = pivot.groupby(
        ["analysis_group", "activation_mechanism"], as_index=False
    ).agg(n=("example_id", "nunique"))
    overall = pivot.groupby("activation_mechanism", as_index=False).agg(
        n=("example_id", "nunique")
    )
    overall.insert(0, "analysis_group", "all")
    summary = pd.concat([summary, overall], ignore_index=True)
    summary["fraction"] = summary.n / summary.groupby("analysis_group").n.transform("sum")
    return pivot, summary


def refresh_full_success_artifacts(run_dir: str | Path) -> Path:
    """Rebuild all compact summaries without another model run."""

    run_dir = Path(run_dir)
    exact = pd.read_csv(run_dir / "exact_output_rows.csv")
    spatial = pd.read_csv(run_dir / "spatial_cone_rows.csv")
    functional = pd.read_csv(run_dir / "joint_functional_rows.csv")
    metadata = pd.read_csv(run_dir / "joint_functional_metadata.csv")
    mechanisms = pd.read_csv(run_dir / "mechanism_rows.csv")
    exact_summary = _summarize(exact, "exact_output")
    spatial_summary = _summarize(spatial, "spatial_cone")
    functional_summary = _summarize(functional, "joint_functional")
    mechanism_summary = mechanisms.groupby(
        ["analysis_group", "mechanism_mode"], as_index=False
    ).agg(n=("example_id", "nunique"))
    mechanism_summary["fraction"] = mechanism_summary.n / mechanism_summary.groupby(
        "analysis_group"
    ).n.transform("sum")
    activation_rows, activation_summary = _activation_mechanisms(functional)
    exact_summary.to_csv(run_dir / "exact_output_summary.csv", index=False)
    spatial_summary.to_csv(run_dir / "spatial_cone_summary.csv", index=False)
    functional_summary.to_csv(run_dir / "joint_functional_summary.csv", index=False)
    mechanism_summary.to_csv(run_dir / "mechanism_summary.csv", index=False)
    activation_rows.to_csv(run_dir / "activation_mechanism_rows.csv", index=False)
    activation_summary.to_csv(run_dir / "activation_mechanism_summary.csv", index=False)

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_hidden = exact[
        exact.condition.eq("global_both") & exact.target_hidden.astype(bool)
    ].example_id.nunique()
    candidate_closed = exact[
        exact.condition.eq("candidate_both")
        & exact.baseline_hidden.astype(bool)
        & exact.target_hidden.astype(bool)
    ].example_id.nunique()
    spatial_index = spatial_summary.set_index(
        ["analysis_group", "direction", "condition"]
    )
    functional_index = functional_summary.set_index(
        ["analysis_group", "direction", "condition"]
    )
    summary.update({
        "endpoint_hidden_examples": int(source_hidden),
        "candidate_output_closure_rate": float(candidate_closed / max(source_hidden, 1)),
        "radius4_transplant_closure_rate": float(
            spatial_index.loc[
                ("all", "transplant_clean", "radius_4"), "reproduced_hiding_rate"
            ]
        ),
        "radius4_mean_energy_fraction": float(
            spatial_index.loc[
                ("all", "transplant_clean", "radius_4"), "mean_component_energy_fraction"
            ]
        ),
        "radius4_mean_spatial_coverage_fraction": float(
            spatial_index.loc[
                ("all", "transplant_clean", "radius_4"), "mean_spatial_coverage_fraction"
            ]
        ),
        "joint_rowspace_repair_rate": float(
            functional_index.loc[
                ("all", "repair_patched", "joint_rowspace"), "recovery_rate"
            ]
        ),
        "joint_rowspace_transplant_rate": float(
            functional_index.loc[
                ("all", "transplant_clean", "joint_rowspace"), "reproduced_hiding_rate"
            ]
        ),
        "local_window_transplant_rate": float(
            functional_index.loc[
                ("all", "transplant_clean", "full_candidate_windows"),
                "reproduced_hiding_rate",
            ]
        ),
        "mean_joint_rowspace_energy_fraction": float(
            functional_index.loc[
                ("all", "transplant_clean", "joint_rowspace"),
                "mean_component_energy_fraction",
            ]
        ),
        "mean_score_completeness_error": float(metadata.score_completeness_error.mean()),
        "mean_geometry_completeness_error": float(metadata.geometry_completeness_error.mean()),
    })
    write_summary(summary_path, summary)
    digest = [
        "# Full-success causal closure", "",
        f"- elapsed: {float(summary['elapsed_seconds']):.1f} s",
        f"- closure examples: {int(summary['closure_examples'])}",
        f"- functional examples: {int(summary['functional_examples'])}",
        f"- endpoint-hidden examples: {int(summary['endpoint_hidden_examples'])}",
        f"- candidate-output closure: {summary['candidate_output_closure_rate']:.3f}",
        f"- radius-4 transplant closure: {summary['radius4_transplant_closure_rate']:.3f}",
        f"- radius-4 coordinate coverage: {summary['radius4_mean_spatial_coverage_fraction']:.3f}",
        f"- joint row-space repair: {summary['joint_rowspace_repair_rate']:.3f}",
        f"- joint row-space transplant: {summary['joint_rowspace_transplant_rate']:.3f}",
        f"- local-window transplant: {summary['local_window_transplant_rate']:.3f}",
        f"- joint row-space full-map energy: {summary['mean_joint_rowspace_energy_fraction']:.5f}",
        f"- score path-completeness error: {summary['mean_score_completeness_error']:.3f}",
        f"- geometry path-completeness error: {summary['mean_geometry_completeness_error']:.3f}",
    ]
    (run_dir / "analysis_digest.md").write_text("\n".join(digest) + "\n", encoding="utf-8")
    return run_dir


def run_full_success_closure(config: FullSuccessClosureConfig | None = None) -> Path:
    config = config or FullSuccessClosureConfig()
    started = time.time()
    StorageBudget(config.output_dir, config.max_output_gb).check()
    selected, _ = _load_inputs(Path(ATTACK_PATH_DB), Path(TRACE_DB), Path(MANIFEST_CSV), None)
    closure_selected = balanced_subset(
        selected, config.closure_examples_per_group, seed=config.seed
    )
    functional_selected = balanced_subset(
        closure_selected, config.functional_examples_per_group, seed=config.seed + 17
    )
    functional_ids = set(functional_selected.example_id.astype(str))
    exp, cache_path = load_experiment(
        prefer_device=config.device, require_device=bool(config.require_device)
    )
    from segmentig_detector.yolo_utils import get_detect_module

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    cache = _cache_lookup(exp)
    exact_rows = []
    spatial_rows = []
    functional_rows = []
    metadata_rows = []
    progress = tqdm(
        closure_selected.itertuples(index=False), total=len(closure_selected),
        desc="full success closure", unit="image",
    )
    for row in progress:
        example_id = str(row.example_id)
        example = cache[example_id]
        clean_image, patched_image, _ = exp._images_for_example(example)
        pair = _preprocess_pair(exp, clean_image, patched_image)
        captured = _capture_detect_inputs(model, detect, pair)
        clean_inputs = [item[0:1] for item in captured]
        patched_inputs = [item[1:2] for item in captured]
        import torch

        with torch.inference_mode():
            clean_box, clean_cls, clean_raw = _head_branches(detect, clean_inputs)
            patched_box, patched_cls, patched_raw = _head_branches(detect, patched_inputs)
            selection = _candidate_closure(detect, clean_raw, patched_raw, row, config)
            exact_conditions = _exact_output_conditions(
                clean_box, clean_cls, patched_box, patched_cls, selection
            )
            exact_names, exact_batch = _stack_raw(exact_conditions)
            for condition, result in zip(
                exact_names, _evaluate_batch(detect, exact_batch, row, config), strict=True
            ):
                result.update({
                    "example_id": example_id,
                    "analysis_group": str(row.analysis_group),
                    "condition": condition,
                    "candidate_closure_size": int(len(selection)),
                })
                exact_rows.append(result)
            spatial_maps, spatial_energy, spatial_coverage = _spatial_components(
                clean_inputs, patched_inputs, selection, config
            )
            for direction in ("repair_patched", "transplant_clean"):
                names, inputs = _batched_input_interventions(
                    clean_inputs, patched_inputs, spatial_maps, direction
                )
                _box, _cls, raw = _head_branches(detect, inputs)
                for condition, result in zip(
                    names, _evaluate_batch(detect, raw, row, config), strict=True
                ):
                    result.update({
                        "example_id": example_id,
                        "analysis_group": str(row.analysis_group),
                        "direction": direction,
                        "condition": condition,
                        "component_energy_fraction": 0.0 if condition == "none"
                        else spatial_energy[condition],
                        "spatial_coverage_fraction": 0.0 if condition == "none"
                        else spatial_coverage[condition],
                    })
                    spatial_rows.append(result)
        if example_id in functional_ids:
            maps, energy, metadata = _functional_components(
                detect, clean_inputs, patched_inputs, clean_box, clean_cls,
                patched_box, patched_cls, selection, row, config,
            )
            metadata.update({
                "example_id": example_id,
                "analysis_group": str(row.analysis_group),
            })
            metadata_rows.append(metadata)
            with torch.inference_mode():
                for direction in ("repair_patched", "transplant_clean"):
                    names, inputs = _batched_input_interventions(
                        clean_inputs, patched_inputs, maps, direction
                    )
                    _box, _cls, raw = _head_branches(detect, inputs)
                    for condition, result in zip(
                        names, _evaluate_batch(detect, raw, row, config), strict=True
                    ):
                        result.update({
                            "example_id": example_id,
                            "analysis_group": str(row.analysis_group),
                            "direction": direction,
                            "condition": condition,
                            "component_energy_fraction": 0.0 if condition == "none"
                            else energy[condition],
                        })
                        functional_rows.append(result)
        release_accelerator_memory()

    exact = pd.DataFrame(exact_rows)
    exact_baseline = exact[exact.condition.eq("global_both")][
        ["example_id", "target_hidden"]
    ].rename(columns={"target_hidden": "baseline_hidden"})
    exact = exact.merge(exact_baseline, on="example_id", validate="many_to_one")
    mechanisms = exact.groupby("example_id", sort=False).apply(
        _mechanism_label, include_groups=False
    ).rename("mechanism_mode").reset_index()
    mechanisms = mechanisms.merge(
        closure_selected[["example_id", "analysis_group"]], on="example_id", validate="one_to_one"
    )

    spatial = pd.DataFrame(spatial_rows)
    spatial_baseline = spatial[spatial.condition.eq("none")][
        ["example_id", "direction", "target_hidden"]
    ].rename(columns={"target_hidden": "baseline_hidden"})
    spatial = spatial.merge(
        spatial_baseline, on=["example_id", "direction"], validate="many_to_one"
    )
    spatial = spatial.merge(
        exact_baseline.rename(columns={"baseline_hidden": "source_hidden"}),
        on="example_id", validate="many_to_one",
    )
    functional = pd.DataFrame(functional_rows)
    functional_baseline = functional[functional.condition.eq("none")][
        ["example_id", "direction", "target_hidden"]
    ].rename(columns={"target_hidden": "baseline_hidden"})
    functional = functional.merge(
        functional_baseline, on=["example_id", "direction"], validate="many_to_one"
    )
    functional = functional.merge(
        exact_baseline.rename(columns={"baseline_hidden": "source_hidden"}),
        on="example_id", validate="many_to_one",
    )
    metadata = pd.DataFrame(metadata_rows)

    exact_summary = _summarize(exact, "exact_output")
    spatial_summary = _summarize(spatial, "spatial_cone")
    functional_summary = _summarize(functional, "joint_functional")
    activation_rows, activation_summary = _activation_mechanisms(functional)
    mechanism_summary = mechanisms.groupby(
        ["analysis_group", "mechanism_mode"], as_index=False
    ).agg(n=("example_id", "nunique"))
    mechanism_summary["fraction"] = mechanism_summary.n / mechanism_summary.groupby(
        "analysis_group"
    ).n.transform("sum")

    payload = {
        **asdict(config),
        "closure_ids": closure_selected.example_id.tolist(),
        "functional_ids": functional_selected.example_id.tolist(),
    }
    run_dir = Path(config.output_dir) / f"full_success_{stable_hash(payload)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    exact.to_csv(run_dir / "exact_output_rows.csv", index=False)
    exact_summary.to_csv(run_dir / "exact_output_summary.csv", index=False)
    mechanisms.to_csv(run_dir / "mechanism_rows.csv", index=False)
    mechanism_summary.to_csv(run_dir / "mechanism_summary.csv", index=False)
    spatial.to_csv(run_dir / "spatial_cone_rows.csv", index=False)
    spatial_summary.to_csv(run_dir / "spatial_cone_summary.csv", index=False)
    functional.to_csv(run_dir / "joint_functional_rows.csv", index=False)
    functional_summary.to_csv(run_dir / "joint_functional_summary.csv", index=False)
    metadata.to_csv(run_dir / "joint_functional_metadata.csv", index=False)
    activation_rows.to_csv(run_dir / "activation_mechanism_rows.csv", index=False)
    activation_summary.to_csv(run_dir / "activation_mechanism_summary.csv", index=False)

    elapsed = time.time() - started
    summary = {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "closure_examples": int(exact.example_id.nunique()),
        "functional_examples": int(functional.example_id.nunique()),
        "cache_path": str(cache_path),
        "config": asdict(config),
        "endpoint_hidden_examples": int(exact_baseline.baseline_hidden.sum()),
        "candidate_output_closure_rate": float(
            exact.loc[
                exact.condition.eq("candidate_both") & exact.baseline_hidden.astype(bool),
                "target_hidden",
            ].mean()
        ),
        "full_map_transplant_closure_rate": float(
            spatial.loc[
                spatial.direction.eq("transplant_clean")
                & spatial.condition.eq("full_maps")
                & spatial.source_hidden.astype(bool),
                "target_hidden",
            ].mean()
        ),
        "mean_score_completeness_error": float(metadata.score_completeness_error.mean()),
        "mean_geometry_completeness_error": float(metadata.geometry_completeness_error.mean()),
        "limitations": [
            "Causal closure is established at the Detect-input/head boundary for one detector and patch.",
            "The functional basis is an image-specific oracle constructed from clean-patched pairs.",
            "Decoded IoU is piecewise differentiable; path completeness is reported explicitly.",
            "Candidate closure uses a 0.01 endpoint score floor, with exact full-map intervention as a sanity control.",
        ],
    }
    write_summary(run_dir / "summary.json", summary)
    digest = [
        "# Full-success causal closure",
        "",
        f"- elapsed: {elapsed:.1f} s",
        f"- closure examples: {summary['closure_examples']}",
        f"- functional examples: {summary['functional_examples']}",
        f"- endpoint-hidden examples: {summary['endpoint_hidden_examples']}",
        f"- candidate-output closure: {summary['candidate_output_closure_rate']:.3f}",
        f"- full-map transplant closure: {summary['full_map_transplant_closure_rate']:.3f}",
        f"- score path-completeness error: {summary['mean_score_completeness_error']:.3f}",
        f"- geometry path-completeness error: {summary['mean_geometry_completeness_error']:.3f}",
    ]
    (run_dir / "analysis_digest.md").write_text("\n".join(digest) + "\n", encoding="utf-8")
    StorageBudget(config.output_dir, config.max_output_gb).check()
    return refresh_full_success_artifacts(run_dir)


if __name__ == "__main__":
    print(run_full_success_closure())
