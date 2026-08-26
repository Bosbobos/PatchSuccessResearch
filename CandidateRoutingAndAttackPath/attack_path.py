from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .candidate_routing import CandidateTraceResult
from .common import (
    DEFAULT_MAX_OUTPUT_GB,
    DEFAULT_OUTPUT_DIR,
    StorageBudget,
    connect_db,
    release_accelerator_memory,
    stable_hash,
    upsert_metadata,
    write_json,
    write_markdown,
)


@dataclass(slots=True)
class AttackPathConfig:
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    max_output_gb: float = DEFAULT_MAX_OUTPUT_GB
    max_examples: int | None = 100
    n_steps: int = 16
    alpha_batch_size: int = 4
    target_kinds: tuple[str, ...] = ("tracked_clean", "winner_margin")
    top_contributions: int = 1000
    visual_map_examples: int = 40
    selection_seed: int = 17
    selection_manifest: str | None = None
    progress: bool = True
    method_version: int = 4


@dataclass(slots=True)
class AttackPathResult:
    run_dir: Path
    db_path: Path
    summary_path: Path
    digest_path: Path
    maps_dir: Path
    config: AttackPathConfig


@dataclass(frozen=True, slots=True)
class TargetTerm:
    level: int
    y: int
    x: int
    coefficient: float = 1.0


PATH_EXAMPLE_COLUMNS = (
    "example_id", "target_kind", "path", "success", "confidence_drop", "mechanism_mode",
    "target_terms_json", "score_clean", "score_patched", "exact_score_delta",
    "path_sum", "completeness_error", "relative_completeness_error", "first_order_sum",
    "first_order_residual", "path_minus_first_order", "total_abs_contribution",
    "positive_contribution", "negative_contribution", "top0p1_abs_fraction",
    "top1_abs_fraction", "top2_abs_fraction", "maps_path", "n_steps", "error",
)

PATH_LEVEL_COLUMNS = (
    "example_id", "target_kind", "level_index", "level_name", "channels", "height", "width",
    "signed_contribution", "abs_contribution", "positive_contribution", "negative_contribution",
    "abs_contribution_fraction", "delta_l1", "delta_l2", "clean_activation_l2",
    "patched_activation_l2", "first_order_contribution",
)

TOP_COLUMNS = (
    "example_id", "target_kind", "global_rank", "level_index", "level_name", "channel",
    "y_index", "x_index", "contribution", "abs_contribution", "delta_activation",
    "average_gradient", "sign",
)


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS path_examples (
            example_id TEXT, target_kind TEXT, path TEXT, success INTEGER, confidence_drop REAL,
            mechanism_mode TEXT, target_terms_json TEXT, score_clean REAL, score_patched REAL,
            exact_score_delta REAL, path_sum REAL, completeness_error REAL,
            relative_completeness_error REAL, first_order_sum REAL, first_order_residual REAL,
            path_minus_first_order REAL, total_abs_contribution REAL, positive_contribution REAL,
            negative_contribution REAL, top0p1_abs_fraction REAL, top1_abs_fraction REAL,
            top2_abs_fraction REAL, maps_path TEXT, n_steps INTEGER, error TEXT,
            PRIMARY KEY(example_id, target_kind)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS path_levels (
            example_id TEXT, target_kind TEXT, level_index INTEGER, level_name TEXT,
            channels INTEGER, height INTEGER, width INTEGER, signed_contribution REAL,
            abs_contribution REAL, positive_contribution REAL, negative_contribution REAL,
            abs_contribution_fraction REAL, delta_l1 REAL, delta_l2 REAL,
            clean_activation_l2 REAL, patched_activation_l2 REAL, first_order_contribution REAL,
            PRIMARY KEY(example_id, target_kind, level_index)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS top_contributions (
            example_id TEXT, target_kind TEXT, global_rank INTEGER, level_index INTEGER,
            level_name TEXT, channel INTEGER, y_index INTEGER, x_index INTEGER,
            contribution REAL, abs_contribution REAL, delta_activation REAL,
            average_gradient REAL, sign INTEGER,
            PRIMARY KEY(example_id, target_kind, global_rank)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_path_mode ON path_examples(target_kind, mechanism_mode, success)")
    conn.commit()


def _insert_rows(conn: sqlite3.Connection, table: str, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    placeholders = ",".join("?" for _ in columns)
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    conn.executemany(sql, [[row.get(column) for column in columns] for row in rows])


def _trace_db_path(trace: CandidateTraceResult | str | Path) -> Path:
    return trace.db_path if isinstance(trace, CandidateTraceResult) else Path(trace)


def _select_trace_examples(trace_db: Path, config: AttackPathConfig) -> pd.DataFrame:
    conn = connect_db(trace_db)
    try:
        rows = pd.read_sql_query("SELECT * FROM examples WHERE error IS NULL", conn)
    finally:
        conn.close()
    if config.selection_manifest is not None:
        manifest_path = Path(config.selection_manifest)
        manifest = pd.read_csv(manifest_path)
        required = {"example_id", "analysis_group", "target_hidden"}
        missing = sorted(required.difference(manifest.columns))
        if missing:
            raise ValueError(f"Selection manifest is missing columns: {missing}")
        if manifest["example_id"].duplicated().any():
            raise ValueError("Selection manifest contains duplicate example_id values")
        selected = manifest.merge(rows, on="example_id", how="left", validate="one_to_one", suffixes=("_selection", ""))
        missing_trace = selected.loc[selected["path"].isna(), "example_id"].astype(str).tolist()
        if missing_trace:
            raise ValueError(f"Selection manifest contains {len(missing_trace)} examples absent from trace DB")
        selected["routing_mode"] = selected["mechanism_mode"]
        selected["mechanism_mode"] = selected["analysis_group"].astype(str)
        selected["success"] = selected["target_hidden"].astype(int)
        if config.max_examples is not None:
            selected = selected.head(int(config.max_examples))
        return selected.reset_index(drop=True)
    if rows.empty or config.max_examples is None or len(rows) <= int(config.max_examples):
        return rows.reset_index(drop=True)

    rng = np.random.default_rng(int(config.selection_seed))
    groups = []
    for _key, group in rows.groupby(["success", "mechanism_mode"], dropna=False, sort=True):
        order = rng.permutation(len(group))
        groups.append(group.iloc[order].reset_index(drop=True))
    selected = []
    cursor = 0
    while len(selected) < int(config.max_examples):
        added = False
        for group in groups:
            if cursor < len(group) and len(selected) < int(config.max_examples):
                selected.append(group.iloc[cursor])
                added = True
        if not added:
            break
        cursor += 1
    return pd.DataFrame(selected).reset_index(drop=True)


def _preprocess_pair(exp, clean_image, patched_image):
    from new_experiments.patch_success_analysis.yolo import pil_to_np_bgr
    from segmentig_detector.yolo_utils import ensure_predictor

    yolo, model = exp.load_model()
    predictor = ensure_predictor(
        yolo, imgsz=int(exp.config.attack.imgsz), conf=float(exp.config.attack.conf),
        device=exp.config.attack.device,
    )
    # Initialising the Ultralytics predictor may move the original torch model
    # back to CPU even when the experiment requested MPS.  This is the model
    # used by all activation hooks below, so restore its requested device after
    # predictor setup rather than trusting the predictor's separate backend.
    if exp.config.attack.device is not None:
        model.to(exp.config.attack.device)
    inputs = predictor.preprocess([pil_to_np_bgr(clean_image), pil_to_np_bgr(patched_image)])
    parameter = next(model.parameters())
    return inputs.to(device=parameter.device, dtype=parameter.dtype)


def _capture_detect_inputs(model, detect, inputs) -> list[Any]:
    from segmentig_detector.yolo_utils import safe_model_forward
    import torch

    captured: dict[str, Any] = {}

    def hook(_module, args):
        value = args[0]
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Expected Detect input list, got {type(value)}")
        captured["levels"] = [item.detach().clone() for item in value]

    handle = detect.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            safe_model_forward(model, inputs)
    finally:
        handle.remove()
    if "levels" not in captured:
        raise RuntimeError("Detect pre-hook did not capture P3/P4/P5 inputs.")
    return captured["levels"]


def _raw_head_levels(detect, inputs: list[Any]) -> list[Any]:
    import torch

    out = detect(list(inputs))
    raw = out
    if isinstance(out, (list, tuple)) and len(out) >= 2 and isinstance(out[0], torch.Tensor):
        raw = out[1]
    levels = [item for item in raw if isinstance(item, torch.Tensor) and item.ndim == 4]
    if not levels:
        raise RuntimeError(f"Detect head returned no raw levels: {type(out)}")
    return levels


def _target_vector(raw_levels, terms: list[TargetTerm], class_channel: int):
    score = None
    for term in terms:
        value = raw_levels[int(term.level)][:, int(class_channel), int(term.y), int(term.x)] * float(term.coefficient)
        score = value if score is None else score + value
    if score is None:
        raise ValueError("Target must contain at least one term.")
    return score


def _target_terms(row: pd.Series, target_kind: str) -> list[TargetTerm] | None:
    tracked = TargetTerm(
        level=int(row["clean_target_level"]), y=int(row["clean_target_y"]),
        x=int(row["clean_target_x"]), coefficient=1.0,
    )
    if target_kind == "tracked_clean":
        return [tracked]
    if pd.isna(row.get("patched_winner_level")):
        return None
    winner = TargetTerm(
        level=int(row["patched_winner_level"]), y=int(row["patched_winner_y"]),
        x=int(row["patched_winner_x"]), coefficient=1.0,
    )
    if target_kind == "patched_winner":
        return [winner]
    if target_kind == "winner_margin":
        if (
            tracked.level == winner.level and tracked.y == winner.y and tracked.x == winner.x
        ):
            return None
        return [tracked, TargetTerm(winner.level, winner.y, winner.x, -1.0)]
    raise ValueError(f"Unsupported target_kind={target_kind!r}")


def _endpoint_score(detect, inputs: list[Any], terms: list[TargetTerm], class_channel: int) -> float:
    import torch

    with torch.no_grad():
        value = _target_vector(_raw_head_levels(detect, [item.clone() for item in inputs]), terms, class_channel)
    return float(value.detach().float().cpu().reshape(-1)[0])


def _first_order(detect, clean_inputs, deltas, terms, class_channel):
    import torch

    variables = [item.detach().clone().requires_grad_(True) for item in clean_inputs]
    score = _target_vector(_raw_head_levels(detect, variables), terms, class_channel).sum()
    grads = torch.autograd.grad(score, variables, allow_unused=True)
    contributions = []
    for grad, delta in zip(grads, deltas, strict=True):
        if grad is None:
            contributions.append(torch.zeros_like(delta, dtype=torch.float32))
        else:
            contributions.append((grad.detach() * delta).float())
    return contributions


def _path_integral(detect, clean_inputs, deltas, terms, class_channel, *, n_steps: int, alpha_batch_size: int):
    import torch

    alphas = (torch.arange(int(n_steps), device=clean_inputs[0].device, dtype=torch.float32) + 0.5) / float(n_steps)
    accumulated = [torch.zeros_like(delta, dtype=torch.float32) for delta in deltas]
    for start in range(0, int(n_steps), max(1, int(alpha_batch_size))):
        alpha = alphas[start : start + max(1, int(alpha_batch_size))]
        variables = []
        for clean, delta in zip(clean_inputs, deltas, strict=True):
            view = alpha.to(dtype=clean.dtype).reshape(-1, 1, 1, 1)
            value = (clean + view * delta).detach().requires_grad_(True)
            variables.append(value)
        scores = _target_vector(_raw_head_levels(detect, variables), terms, class_channel)
        grads = torch.autograd.grad(scores.sum(), variables, allow_unused=True)
        for idx, (grad, delta) in enumerate(zip(grads, deltas, strict=True)):
            if grad is not None:
                per_alpha = grad.detach().float() * delta.float()
                accumulated[idx] += per_alpha.sum(dim=0, keepdim=True)
    return [item / float(n_steps) for item in accumulated]


def _fraction_top(abs_arrays: list[np.ndarray], fraction: float) -> float:
    values = np.concatenate([item.reshape(-1) for item in abs_arrays])
    total = float(values.sum())
    if total <= 1e-12:
        return 0.0
    k = max(1, int(np.ceil(len(values) * float(fraction))))
    if k >= len(values):
        return 1.0
    threshold = len(values) - k
    top = np.partition(values, threshold)[threshold:]
    return float(top.sum() / total)


def _top_rows(example_id: str, target_kind: str, contributions, deltas, top_n: int) -> list[dict[str, Any]]:
    flattened = [np.abs(item).reshape(-1) for item in contributions]
    sizes = [len(item) for item in flattened]
    all_abs = np.concatenate(flattened)
    k = min(max(0, int(top_n)), len(all_abs))
    if k == 0:
        return []
    indices = np.argpartition(all_abs, len(all_abs) - k)[len(all_abs) - k :]
    indices = indices[np.argsort(all_abs[indices])[::-1]]
    offsets = np.cumsum([0] + sizes)
    rows = []
    for rank, global_index in enumerate(indices.tolist(), start=1):
        level = int(np.searchsorted(offsets[1:], global_index, side="right"))
        local = int(global_index - offsets[level])
        _batch, channels, height, width = contributions[level].shape
        channel = local // (height * width)
        spatial = local % (height * width)
        y, x = spatial // width, spatial % width
        contribution = float(contributions[level].reshape(-1)[local])
        delta = float(deltas[level].detach().float().cpu().numpy().reshape(-1)[local])
        avg_gradient = contribution / delta if abs(delta) > 1e-12 else 0.0
        rows.append({
            "example_id": example_id, "target_kind": target_kind, "global_rank": rank,
            "level_index": level, "level_name": f"P{level + 3}", "channel": int(channel),
            "y_index": int(y), "x_index": int(x), "contribution": contribution,
            "abs_contribution": abs(contribution), "delta_activation": delta,
            "average_gradient": avg_gradient, "sign": int(np.sign(contribution)),
        })
    return rows


def _save_spatial_maps(path: Path, contributions, deltas, first_order) -> Path:
    payload: dict[str, Any] = {}
    for level, (contribution, delta, first) in enumerate(zip(contributions, deltas, first_order, strict=True)):
        c = contribution[0]
        d = delta.detach().float().cpu().numpy()[0]
        f = first[0]
        prefix = f"P{level + 3}"
        payload[f"{prefix}_signed_contribution"] = c.sum(axis=0).astype(np.float32)
        payload[f"{prefix}_abs_contribution"] = np.abs(c).sum(axis=0).astype(np.float32)
        payload[f"{prefix}_delta_l2"] = np.sqrt(np.square(d).sum(axis=0)).astype(np.float32)
        payload[f"{prefix}_first_order"] = f.sum(axis=0).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def _analyse_target(
    *, detect, head_inputs, row: pd.Series, target_kind: str, terms: list[TargetTerm],
    class_channel: int, config: AttackPathConfig, maps_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    clean_inputs = [item[0:1].detach() for item in head_inputs]
    patched_inputs = [item[1:2].detach() for item in head_inputs]
    deltas = [patched - clean for clean, patched in zip(clean_inputs, patched_inputs, strict=True)]
    clean_score = _endpoint_score(detect, clean_inputs, terms, class_channel)
    patched_score = _endpoint_score(detect, patched_inputs, terms, class_channel)
    exact_delta = patched_score - clean_score
    first_tensors = _first_order(detect, clean_inputs, deltas, terms, class_channel)
    path_tensors_torch = _path_integral(
        detect, clean_inputs, deltas, terms, class_channel,
        n_steps=config.n_steps, alpha_batch_size=config.alpha_batch_size,
    )
    contributions = [item.detach().float().cpu().numpy() for item in path_tensors_torch]
    first_order = [item.detach().float().cpu().numpy() for item in first_tensors]
    path_sum = float(sum(item.sum() for item in contributions))
    first_sum = float(sum(item.sum() for item in first_order))
    completeness_error = path_sum - exact_delta
    relative_error = abs(completeness_error) / max(abs(exact_delta), 1e-8)
    total_abs = float(sum(np.abs(item).sum() for item in contributions))
    positive = float(sum(item[item > 0].sum() for item in contributions))
    negative = float(sum(item[item < 0].sum() for item in contributions))
    abs_arrays = [np.abs(item) for item in contributions]

    saved_maps = None
    if maps_path is not None:
        saved_maps = str(_save_spatial_maps(maps_path, contributions, deltas, first_order))

    target_json = [asdict(term) for term in terms]
    example_row = {column: None for column in PATH_EXAMPLE_COLUMNS}
    example_row.update({
        "example_id": str(row["example_id"]), "target_kind": target_kind, "path": str(row["path"]),
        "success": int(row["success"]), "confidence_drop": float(row["confidence_drop"]),
        "mechanism_mode": str(row["mechanism_mode"]), "target_terms_json": json_dumps(target_json),
        "score_clean": clean_score, "score_patched": patched_score, "exact_score_delta": exact_delta,
        "path_sum": path_sum, "completeness_error": completeness_error,
        "relative_completeness_error": relative_error, "first_order_sum": first_sum,
        "first_order_residual": exact_delta - first_sum,
        "path_minus_first_order": path_sum - first_sum,
        "total_abs_contribution": total_abs, "positive_contribution": positive,
        "negative_contribution": negative,
        "top0p1_abs_fraction": _fraction_top(abs_arrays, 0.001),
        "top1_abs_fraction": _fraction_top(abs_arrays, 0.01),
        "top2_abs_fraction": _fraction_top(abs_arrays, 0.02),
        "maps_path": saved_maps, "n_steps": int(config.n_steps), "error": None,
    })

    level_rows = []
    for level, (contribution, delta, clean, patched, first) in enumerate(
        zip(contributions, deltas, clean_inputs, patched_inputs, first_order, strict=True)
    ):
        abs_sum = float(np.abs(contribution).sum())
        level_rows.append({
            "example_id": str(row["example_id"]), "target_kind": target_kind,
            "level_index": level, "level_name": f"P{level + 3}",
            "channels": int(contribution.shape[1]), "height": int(contribution.shape[2]),
            "width": int(contribution.shape[3]), "signed_contribution": float(contribution.sum()),
            "abs_contribution": abs_sum,
            "positive_contribution": float(contribution[contribution > 0].sum()),
            "negative_contribution": float(contribution[contribution < 0].sum()),
            "abs_contribution_fraction": abs_sum / max(total_abs, 1e-12),
            "delta_l1": float(delta.detach().float().abs().sum().cpu()),
            "delta_l2": float(delta.detach().float().square().sum().sqrt().cpu()),
            "clean_activation_l2": float(clean.float().square().sum().sqrt().cpu()),
            "patched_activation_l2": float(patched.float().square().sum().sqrt().cpu()),
            "first_order_contribution": float(first.sum()),
        })
    top_rows = _top_rows(str(row["example_id"]), target_kind, contributions, deltas, config.top_contributions)
    return example_row, level_rows, top_rows


def json_dumps(value: Any) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _summary_outputs(conn: sqlite3.Connection, run_dir: Path, config: AttackPathConfig) -> tuple[Path, Path]:
    examples = pd.read_sql_query("SELECT * FROM path_examples WHERE error IS NULL", conn)
    levels = pd.read_sql_query("SELECT * FROM path_levels", conn)
    if examples.empty:
        return (
            write_json(run_dir / "summary.json", {"status": "empty", "n_rows": 0}),
            write_markdown(run_dir / "analysis_digest.md", ["# Attack-path digest", "", "No completed targets."]),
        )
    group_summary = (
        examples.groupby(["target_kind", "success", "mechanism_mode"], dropna=False)
        .agg(
            n=("example_id", "size"), mean_exact_delta=("exact_score_delta", "mean"),
            mean_path_sum=("path_sum", "mean"),
            median_relative_completeness_error=("relative_completeness_error", "median"),
            mean_abs_first_order_residual=("first_order_residual", lambda x: float(np.abs(x).mean())),
            mean_top0p1_abs_fraction=("top0p1_abs_fraction", "mean"),
            mean_top1_abs_fraction=("top1_abs_fraction", "mean"),
            mean_top2_abs_fraction=("top2_abs_fraction", "mean"),
        ).reset_index()
    )
    group_summary.to_csv(run_dir / "group_summary.csv", index=False)
    level_summary = (
        levels.merge(examples[["example_id", "target_kind", "success", "mechanism_mode"]], on=["example_id", "target_kind"], how="left")
        .groupby(["target_kind", "success", "mechanism_mode", "level_name"], dropna=False)
        .agg(
            n=("example_id", "size"), mean_signed_contribution=("signed_contribution", "mean"),
            mean_abs_contribution_fraction=("abs_contribution_fraction", "mean"),
            mean_delta_l2=("delta_l2", "mean"),
        ).reset_index()
    )
    level_summary.to_csv(run_dir / "level_summary.csv", index=False)
    payload = {
        "status": "complete", "n_target_rows": int(len(examples)),
        "n_examples": int(examples["example_id"].nunique()),
        "target_kinds": {str(k): int(v) for k, v in examples["target_kind"].value_counts().items()},
        "median_relative_completeness_error": float(examples["relative_completeness_error"].median()),
        "mean_abs_first_order_residual": float(examples["first_order_residual"].abs().mean()),
        "mean_top_abs_fractions": {
            "0.1%": float(examples["top0p1_abs_fraction"].mean()),
            "1%": float(examples["top1_abs_fraction"].mean()),
            "2%": float(examples["top2_abs_fraction"].mean()),
        },
        "database": str(run_dir / "attack_path.sqlite"), "config": asdict(config),
    }
    summary_path = write_json(run_dir / "summary.json", payload)
    digest_path = write_markdown(run_dir / "analysis_digest.md", [
        "# Attack-path digest", "",
        f"- unique examples: {payload['n_examples']}",
        f"- target rows: {payload['n_target_rows']}",
        f"- median relative completeness error: {payload['median_relative_completeness_error']:.6f}",
        f"- mean |exact delta - clean first-order|: {payload['mean_abs_first_order_residual']:.6f}",
        f"- mean absolute contribution in top 0.1% / 1% / 2%: {payload['mean_top_abs_fractions']['0.1%']:.3f} / {payload['mean_top_abs_fractions']['1%']:.3f} / {payload['mean_top_abs_fractions']['2%']:.3f}",
        "", "Read summary.json, group_summary.csv, and level_summary.csv before querying SQLite.",
    ])
    return summary_path, digest_path


def run_attack_path_decomposition(
    exp,
    candidate_trace: CandidateTraceResult | str | Path,
    config: AttackPathConfig | None = None,
    *,
    force: bool = False,
) -> AttackPathResult:
    import torch
    from segmentig_detector.yolo_utils import get_detect_module

    config = config or AttackPathConfig()
    if int(config.n_steps) <= 0:
        raise ValueError("n_steps must be positive")
    trace_db = _trace_db_path(candidate_trace)
    selected = _select_trace_examples(trace_db, config)
    payload = {
        "trace_db": str(trace_db.resolve()), "trace_size": int(trace_db.stat().st_size),
        "selection_rows": selected[["example_id", "success", "mechanism_mode"]].to_dict("records"),
        **asdict(config),
    }
    run_dir = Path(config.output_dir) / f"attack_path_{stable_hash(payload)}"
    maps_dir = run_dir / "spatial_maps"
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = StorageBudget(Path(config.output_dir), config.max_output_gb)
    budget.check()
    db_path = run_dir / "attack_path.sqlite"
    if force and db_path.exists():
        db_path.unlink()
    conn = connect_db(db_path)
    _create_schema(conn)
    upsert_metadata(conn, {"config": asdict(config), "payload": payload, "trace_db": str(trace_db)})
    completed = {
        (row[0], row[1]) for row in conn.execute("SELECT example_id, target_kind FROM path_examples WHERE error IS NULL")
    }

    _yolo, model = exp.load_model()
    model.eval()
    detect = get_detect_module(model, exp.config.detect_layer)
    detect.eval()
    reg_max = int(getattr(detect, "reg_max", 16))
    visual_ids = set(selected["example_id"].head(max(0, int(config.visual_map_examples))).astype(str))
    cache_examples = {stable_hash({"path": str(item.path), "drop": float(item.drop), "success": bool(item.success)}): item for item in exp.get_cache().examples}

    progress = None
    if config.progress:
        try:
            from tqdm.auto import tqdm
            progress = tqdm(total=len(selected), desc="attack-path decomposition", unit="img")
        except Exception:
            progress = None
    try:
        for _, row in selected.iterrows():
            example_id = str(row["example_id"])
            example = cache_examples.get(example_id)
            if example is None:
                error_rows = []
                for target_kind in config.target_kinds:
                    item = {column: None for column in PATH_EXAMPLE_COLUMNS}
                    item.update({"example_id": example_id, "target_kind": target_kind, "path": str(row["path"]), "success": int(row["success"]), "confidence_drop": float(row["confidence_drop"]), "mechanism_mode": str(row["mechanism_mode"]), "error": "AttackExample not found in cache"})
                    error_rows.append(item)
                _insert_rows(conn, "path_examples", PATH_EXAMPLE_COLUMNS, error_rows)
                conn.commit()
                continue
            pending_kinds = [kind for kind in config.target_kinds if (example_id, kind) not in completed]
            if not pending_kinds:
                if progress is not None:
                    progress.update(1)
                continue
            try:
                clean_image, patched_image, _bbox = exp._images_for_example(example)
                pair_inputs = _preprocess_pair(exp, clean_image, patched_image)
                head_inputs = _capture_detect_inputs(model, detect, pair_inputs)
                class_channel = 4 * reg_max + int(row["class_id"])
                example_rows, level_rows, top_rows = [], [], []
                for target_kind in pending_kinds:
                    terms = _target_terms(row, target_kind)
                    if terms is None:
                        continue
                    map_path = None
                    if example_id in visual_ids:
                        map_path = maps_dir / f"{example_id}_{target_kind}.npz"
                    result_row, result_levels, result_top = _analyse_target(
                        detect=detect, head_inputs=head_inputs, row=row, target_kind=target_kind,
                        terms=terms, class_channel=class_channel, config=config, maps_path=map_path,
                    )
                    example_rows.append(result_row)
                    level_rows.extend(result_levels)
                    top_rows.extend(result_top)
                _insert_rows(conn, "path_examples", PATH_EXAMPLE_COLUMNS, example_rows)
                _insert_rows(conn, "path_levels", PATH_LEVEL_COLUMNS, level_rows)
                _insert_rows(conn, "top_contributions", TOP_COLUMNS, top_rows)
            except Exception as exc:  # noqa: BLE001
                error_rows = []
                for target_kind in pending_kinds:
                    item = {column: None for column in PATH_EXAMPLE_COLUMNS}
                    item.update({
                        "example_id": example_id, "target_kind": target_kind, "path": str(row["path"]),
                        "success": int(row["success"]), "confidence_drop": float(row["confidence_drop"]),
                        "mechanism_mode": str(row["mechanism_mode"]), "n_steps": int(config.n_steps),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    error_rows.append(item)
                _insert_rows(conn, "path_examples", PATH_EXAMPLE_COLUMNS, error_rows)
            conn.commit()
            budget.check(extra_bytes=100 * 1024**2)
            if progress is not None:
                progress.update(1)
            release_accelerator_memory()
    finally:
        if progress is not None:
            progress.close()
        summary_path, digest_path = _summary_outputs(conn, run_dir, config)
        conn.close()
    return AttackPathResult(
        run_dir=run_dir, db_path=db_path, summary_path=summary_path,
        digest_path=digest_path, maps_dir=maps_dir, config=config,
    )


def load_attack_path_tables(result_or_db: AttackPathResult | str | Path, *, top: bool = False):
    db_path = result_or_db.db_path if isinstance(result_or_db, AttackPathResult) else Path(result_or_db)
    conn = connect_db(db_path)
    try:
        out = {
            "examples": pd.read_sql_query("SELECT * FROM path_examples WHERE error IS NULL", conn),
            "levels": pd.read_sql_query("SELECT * FROM path_levels", conn),
        }
        if top:
            out["top"] = pd.read_sql_query("SELECT * FROM top_contributions", conn)
        return out
    finally:
        conn.close()


def load_spatial_maps(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}
