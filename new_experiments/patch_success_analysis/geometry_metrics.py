from __future__ import annotations

from typing import Any


def _bbox_xyxy(detection: dict[str, Any] | None):
    if not detection:
        return None
    bbox = detection.get("bbox_xyxy_orig")
    if bbox is None:
        return None
    if len(bbox) != 4:
        return None
    return tuple(float(v) for v in bbox)


def _rect_center(rect):
    x1, y1, x2, y2 = [float(v) for v in rect]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _rect_area(rect) -> float:
    x1, y1, x2, y2 = [float(v) for v in rect]
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _rect_intersection_area(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _rect_intersection(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _rect_union_area(rects) -> float:
    rects = [tuple(float(v) for v in rect) for rect in rects if rect is not None and _rect_area(rect) > 0]
    if not rects:
        return 0.0
    xs = sorted({v for rect in rects for v in (rect[0], rect[2])})
    ys = sorted({v for rect in rects for v in (rect[1], rect[3])})
    total = 0.0
    for x1, x2 in zip(xs[:-1], xs[1:], strict=True):
        if x2 <= x1:
            continue
        for y1, y2 in zip(ys[:-1], ys[1:], strict=True):
            if y2 <= y1:
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if any(rect[0] <= cx <= rect[2] and rect[1] <= cy <= rect[3] for rect in rects):
                total += (x2 - x1) * (y2 - y1)
    return float(total)


def _point_distance(a, b) -> float:
    import math

    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def rectangle_distance(a, b) -> float:
    import math

    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return float(math.hypot(dx, dy))


def farthest_patch_center_to_bbox_corner_distance(patch_bbox, object_bbox) -> float:
    patch_center = _rect_center(patch_bbox)
    x1, y1, x2, y2 = [float(v) for v in object_bbox]
    corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    return float(max(_point_distance(patch_center, corner) for corner in corners))


def letterbox_padding_for_image(path: str, *, imgsz: int = 640):
    from pathlib import Path

    from PIL import Image

    with Image.open(Path(path)) as image:
        w, h = image.size
    scale = min(float(imgsz) / float(h), float(imgsz) / float(w))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pad_left = (int(imgsz) - new_w) // 2
    pad_top = (int(imgsz) - new_h) // 2
    pad_right = int(imgsz) - new_w - pad_left
    pad_bottom = int(imgsz) - new_h - pad_top
    rects = []
    if pad_left > 0:
        rects.append((0.0, 0.0, float(pad_left), float(imgsz)))
    if pad_right > 0:
        rects.append((float(imgsz - pad_right), 0.0, float(imgsz), float(imgsz)))
    if pad_top > 0:
        rects.append((0.0, 0.0, float(imgsz), float(pad_top)))
    if pad_bottom > 0:
        rects.append((0.0, float(imgsz - pad_bottom), float(imgsz), float(imgsz)))
    return {
        "orig_width": int(w),
        "orig_height": int(h),
        "letterbox_scale": float(scale),
        "content_width": int(new_w),
        "content_height": int(new_h),
        "pad_left": int(pad_left),
        "pad_right": int(pad_right),
        "pad_top": int(pad_top),
        "pad_bottom": int(pad_bottom),
        "padding_rects": rects,
    }


def detected_gray_padding_for_image(
    path: str,
    *,
    imgsz: int = 640,
    gray_value: int = 114,
    tolerance: int = 16,
    min_fraction: float = 0.95,
):
    from pathlib import Path

    import numpy as np
    from PIL import Image

    with Image.open(Path(path)) as image:
        arr = np.asarray(image.convert("RGB").resize((int(imgsz), int(imgsz))))
    target = np.full(3, int(gray_value), dtype=np.int16)
    close = np.max(np.abs(arr.astype(np.int16) - target), axis=2) <= int(tolerance)
    h, w = close.shape

    def is_col(idx):
        return float(close[:, int(idx)].mean()) >= float(min_fraction)

    def is_row(idx):
        return float(close[int(idx), :].mean()) >= float(min_fraction)

    left = 0
    while left < w and is_col(left):
        left += 1
    right = 0
    while right < w - left and is_col(w - 1 - right):
        right += 1
    top = 0
    while top < h and is_row(top):
        top += 1
    bottom = 0
    while bottom < h - top and is_row(h - 1 - bottom):
        bottom += 1

    rects = []
    if left > 0:
        rects.append((0.0, 0.0, float(left), float(h)))
    if right > 0:
        rects.append((float(w - right), 0.0, float(w), float(h)))
    if top > 0:
        rects.append((0.0, 0.0, float(w), float(top)))
    if bottom > 0:
        rects.append((0.0, float(h - bottom), float(w), float(h)))
    return {
        "gray_pad_left": int(left),
        "gray_pad_right": int(right),
        "gray_pad_top": int(top),
        "gray_pad_bottom": int(bottom),
        "gray_padding_rects": rects,
        "gray_padding_area": _rect_union_area(rects),
    }


def detected_gray_padding_right_symmetric_for_image(
    path: str,
    *,
    imgsz: int = 640,
    gray_value: int = 114,
    tolerance: int = 16,
    min_fraction: float = 0.95,
):
    detected = detected_gray_padding_for_image(
        path,
        imgsz=int(imgsz),
        gray_value=int(gray_value),
        tolerance=int(tolerance),
        min_fraction=float(min_fraction),
    )
    width = int(detected["gray_pad_right"])
    top = int(detected["gray_pad_top"])
    bottom = int(detected["gray_pad_bottom"])
    rects = []
    if width > 0:
        rects.append((0.0, 0.0, float(width), float(imgsz)))
        rects.append((float(imgsz - width), 0.0, float(imgsz), float(imgsz)))
    if top > 0:
        rects.append((0.0, 0.0, float(imgsz), float(top)))
    if bottom > 0:
        rects.append((0.0, float(imgsz - bottom), float(imgsz), float(imgsz)))
    return {
        "right_gray_pad_left": int(width),
        "right_gray_pad_right": int(width),
        "right_gray_pad_top": int(top),
        "right_gray_pad_bottom": int(bottom),
        "right_gray_padding_rects": rects,
        "right_gray_padding_area": _rect_union_area(rects),
    }


def _padding_overlap_area(rect, padding_rects) -> float:
    return _rect_union_area([_rect_intersection(rect, pad_rect) for pad_rect in padding_rects])


def _distance_to_padding(rect, padding_rects) -> float:
    if not padding_rects:
        return 0.0
    if _padding_overlap_area(rect, padding_rects) > 0:
        return 0.0
    return float(min(rectangle_distance(rect, pad_rect) for pad_rect in padding_rects))


def geometry_row(example, *, imgsz: int = 640):
    bbox = _bbox_xyxy(example.clean_detection)
    patch_bbox = example.patch_bbox_lb
    if bbox is None or patch_bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    patch_area = _rect_area(patch_bbox)
    patch_center = _rect_center(patch_bbox)
    object_center = _rect_center(bbox)
    padding = letterbox_padding_for_image(example.path, imgsz=int(imgsz))
    padding_rects = padding["padding_rects"]
    gray_padding = detected_gray_padding_for_image(example.path, imgsz=int(imgsz))
    gray_padding_rects = gray_padding["gray_padding_rects"]
    right_gray_padding = detected_gray_padding_right_symmetric_for_image(example.path, imgsz=int(imgsz))
    right_gray_padding_rects = right_gray_padding["right_gray_padding_rects"]
    padding_area = float(
        int(imgsz) * int(imgsz) - padding["content_width"] * padding["content_height"]
    )
    gray_padding_area = float(gray_padding["gray_padding_area"])
    right_gray_padding_area = float(right_gray_padding["right_gray_padding_area"])
    patch_padding_overlap = _padding_overlap_area(patch_bbox, padding_rects)
    object_padding_overlap = _padding_overlap_area(bbox, padding_rects)
    patch_gray_padding_overlap = _padding_overlap_area(patch_bbox, gray_padding_rects)
    object_gray_padding_overlap = _padding_overlap_area(bbox, gray_padding_rects)
    patch_right_gray_padding_overlap = _padding_overlap_area(patch_bbox, right_gray_padding_rects)
    object_right_gray_padding_overlap = _padding_overlap_area(bbox, right_gray_padding_rects)
    return {
        "path": example.path,
        "success": bool(example.success),
        "drop": float(example.drop),
        "conf_clean": float(example.conf_clean),
        "conf_patch": float(example.conf_patch),
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": area,
        "bbox_area_frac": area / float(int(imgsz) * int(imgsz)),
        "bbox_sqrt_area": area ** 0.5,
        "bbox_sqrt_area_frac": (area ** 0.5) / float(imgsz),
        "orig_width": padding["orig_width"],
        "orig_height": padding["orig_height"],
        "letterbox_scale": padding["letterbox_scale"],
        "content_width": padding["content_width"],
        "content_height": padding["content_height"],
        "pad_left": padding["pad_left"],
        "pad_right": padding["pad_right"],
        "pad_top": padding["pad_top"],
        "pad_bottom": padding["pad_bottom"],
        "pad_x_total": padding["pad_left"] + padding["pad_right"],
        "pad_y_total": padding["pad_top"] + padding["pad_bottom"],
        "padding_area": padding_area,
        "padding_area_frac": padding_area / float(int(imgsz) * int(imgsz)),
        "patch_padding_overlap_area": patch_padding_overlap,
        "patch_padding_overlap_frac": patch_padding_overlap / max(1e-12, patch_area),
        "object_padding_overlap_area": object_padding_overlap,
        "object_padding_overlap_frac": object_padding_overlap / max(1e-12, area),
        "object_distance_to_padding": _distance_to_padding(bbox, padding_rects),
        "object_distance_to_padding_frac": _distance_to_padding(bbox, padding_rects) / float(imgsz),
        "gray_pad_left": gray_padding["gray_pad_left"],
        "gray_pad_right": gray_padding["gray_pad_right"],
        "gray_pad_top": gray_padding["gray_pad_top"],
        "gray_pad_bottom": gray_padding["gray_pad_bottom"],
        "gray_pad_x_total": gray_padding["gray_pad_left"] + gray_padding["gray_pad_right"],
        "gray_pad_y_total": gray_padding["gray_pad_top"] + gray_padding["gray_pad_bottom"],
        "gray_padding_area": gray_padding_area,
        "gray_padding_area_frac": gray_padding_area / float(int(imgsz) * int(imgsz)),
        "patch_gray_padding_overlap_area": patch_gray_padding_overlap,
        "patch_gray_padding_overlap_frac": patch_gray_padding_overlap / max(1e-12, patch_area),
        "object_gray_padding_overlap_area": object_gray_padding_overlap,
        "object_gray_padding_overlap_frac": object_gray_padding_overlap / max(1e-12, area),
        "object_distance_to_gray_padding": _distance_to_padding(bbox, gray_padding_rects),
        "object_distance_to_gray_padding_frac": _distance_to_padding(bbox, gray_padding_rects) / float(imgsz),
        "right_gray_pad_left": right_gray_padding["right_gray_pad_left"],
        "right_gray_pad_right": right_gray_padding["right_gray_pad_right"],
        "right_gray_pad_top": right_gray_padding["right_gray_pad_top"],
        "right_gray_pad_bottom": right_gray_padding["right_gray_pad_bottom"],
        "right_gray_pad_x_total": right_gray_padding["right_gray_pad_left"] + right_gray_padding["right_gray_pad_right"],
        "right_gray_pad_y_total": right_gray_padding["right_gray_pad_top"] + right_gray_padding["right_gray_pad_bottom"],
        "right_gray_padding_area": right_gray_padding_area,
        "right_gray_padding_area_frac": right_gray_padding_area / float(int(imgsz) * int(imgsz)),
        "patch_right_gray_padding_overlap_area": patch_right_gray_padding_overlap,
        "patch_right_gray_padding_overlap_frac": patch_right_gray_padding_overlap / max(1e-12, patch_area),
        "object_right_gray_padding_overlap_area": object_right_gray_padding_overlap,
        "object_right_gray_padding_overlap_frac": object_right_gray_padding_overlap / max(1e-12, area),
        "object_distance_to_right_gray_padding": _distance_to_padding(bbox, right_gray_padding_rects),
        "object_distance_to_right_gray_padding_frac": _distance_to_padding(bbox, right_gray_padding_rects) / float(imgsz),
        "patch_x1": float(patch_bbox[0]),
        "patch_y1": float(patch_bbox[1]),
        "patch_x2": float(patch_bbox[2]),
        "patch_y2": float(patch_bbox[3]),
        "distance_to_object_center": _point_distance(patch_center, object_center),
        "distance_to_object_center_frac": _point_distance(patch_center, object_center) / float(imgsz),
        "distance_to_nearest_object_edge": rectangle_distance(patch_bbox, bbox),
        "distance_to_nearest_object_edge_frac": rectangle_distance(patch_bbox, bbox) / float(imgsz),
        "distance_to_farthest_object_edge": farthest_patch_center_to_bbox_corner_distance(patch_bbox, bbox),
        "distance_to_farthest_object_edge_frac": farthest_patch_center_to_bbox_corner_distance(patch_bbox, bbox) / float(imgsz),
    }


def geometry_frame(examples, *, imgsz: int = 640):
    import pandas as pd

    rows = [geometry_row(example, imgsz=int(imgsz)) for example in examples]
    return pd.DataFrame([row for row in rows if row is not None])


def select_balanced_examples(cache, *, max_per_class: int | None = None):
    if max_per_class is None:
        return list(cache.examples)
    return list(cache.successes[: int(max_per_class)]) + list(cache.failures[: int(max_per_class)])


def geometry_summary(df):
    metrics = [
        "bbox_area_frac",
        "bbox_sqrt_area_frac",
        "bbox_width",
        "bbox_height",
        "padding_area_frac",
        "patch_padding_overlap_frac",
        "object_padding_overlap_frac",
        "object_distance_to_padding",
        "gray_padding_area_frac",
        "patch_gray_padding_overlap_frac",
        "object_gray_padding_overlap_frac",
        "object_distance_to_gray_padding",
        "right_gray_padding_area_frac",
        "patch_right_gray_padding_overlap_frac",
        "object_right_gray_padding_overlap_frac",
        "object_distance_to_right_gray_padding",
        "distance_to_object_center",
        "distance_to_nearest_object_edge",
        "distance_to_farthest_object_edge",
    ]
    return df.groupby("success")[metrics].agg(["count", "mean", "std", "median"])


def plot_object_size_summary(df, *, metric: str = "bbox_area_frac"):
    import matplotlib.pyplot as plt
    import numpy as np

    grouped = df.groupby("success")[metric].agg(["mean", "std", "count"]).reindex([False, True])
    labels = ["fail", "success"]
    colors = ["#F58518", "#4C78A8"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    x = np.arange(2)
    axes[0].bar(x, grouped["mean"], yerr=grouped["std"], capsize=5, color=colors, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel(metric)
    axes[0].set_title("Object size mean +/- std")
    axes[0].grid(axis="y", alpha=0.25)

    jitter = np.random.default_rng(17)
    for idx, success in enumerate([False, True]):
        values = df.loc[df["success"] == success, metric].astype("float64").to_numpy()
        sample = values if values.size <= 1200 else jitter.choice(values, size=1200, replace=False)
        xs = np.full(sample.size, idx, dtype="float64") + jitter.normal(0.0, 0.045, size=sample.size)
        axes[1].scatter(xs, sample, s=6, alpha=0.22, color=colors[idx], label=labels[idx])
        axes[1].boxplot(values, positions=[idx], widths=0.28, showfliers=False)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel(metric)
    axes[1].set_title("All object sizes (sampled scatter + box)")
    axes[1].grid(axis="y", alpha=0.25)
    return fig


def plot_geometry_distance_summary(df):
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        ("distance_to_object_center", "center"),
        ("distance_to_nearest_object_edge", "nearest edge"),
        ("distance_to_farthest_object_edge", "farthest edge"),
    ]
    labels = ["fail", "success"]
    colors = ["#F58518", "#4C78A8"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0), constrained_layout=True)

    width = 0.35
    x = np.arange(len(metrics), dtype="float64")
    for offset, success, label, color in [(-width / 2, False, "fail", colors[0]), (width / 2, True, "success", colors[1])]:
        sub = df[df["success"] == success]
        means = [float(sub[metric].mean()) for metric, _name in metrics]
        stds = [float(sub[metric].std()) for metric, _name in metrics]
        axes[0].bar(x + offset, means, width=width, yerr=stds, capsize=4, color=color, label=label, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([name for _metric, name in metrics])
    axes[0].set_ylabel("pixels")
    axes[0].set_title("Patch-to-object distances mean +/- std")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    for metric, name in metrics:
        fail = df.loc[df["success"] == False, metric].astype("float64")
        success = df.loc[df["success"] == True, metric].astype("float64")
        axes[1].hist(fail, bins=50, alpha=0.35, density=True, color=colors[0], label=f"fail {name}")
        axes[1].hist(success, bins=50, alpha=0.35, density=True, color=colors[1], label=f"success {name}")
    axes[1].set_xlabel("pixels")
    axes[1].set_ylabel("density")
    axes[1].set_title("Distance distributions")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    return fig


def padding_metric_quality(df):
    from .metrics import metric_quality_rows

    metric_names = [
        "padding_area_frac",
        "pad_x_total",
        "pad_y_total",
        "patch_padding_overlap_frac",
        "object_padding_overlap_frac",
        "object_distance_to_padding",
        "object_distance_to_padding_frac",
        "gray_padding_area_frac",
        "gray_pad_x_total",
        "gray_pad_y_total",
        "patch_gray_padding_overlap_frac",
        "object_gray_padding_overlap_frac",
        "object_distance_to_gray_padding",
        "object_distance_to_gray_padding_frac",
        "right_gray_padding_area_frac",
        "right_gray_pad_x_total",
        "right_gray_pad_y_total",
        "patch_right_gray_padding_overlap_frac",
        "object_right_gray_padding_overlap_frac",
        "object_distance_to_right_gray_padding",
        "object_distance_to_right_gray_padding_frac",
    ]
    return metric_quality_rows(
        df["success"].astype(bool).tolist(),
        {name: df[name].astype("float64").to_numpy() for name in metric_names if name in df.columns},
    )


def plot_padding_summary(df):
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        ("right_gray_padding_area_frac", "right-symmetric gray padding area"),
        ("patch_right_gray_padding_overlap_frac", "patch on right-symmetric gray padding"),
        ("object_right_gray_padding_overlap_frac", "object on right-symmetric gray padding"),
        ("object_distance_to_right_gray_padding", "object-to-right-symmetric-padding px"),
    ]
    labels = ["fail", "success"]
    colors = ["#F58518", "#4C78A8"]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5), constrained_layout=True)
    axes = axes.reshape(-1)
    width = 0.35
    x = np.arange(2, dtype="float64")
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        grouped = df.groupby("success")[metric].agg(["mean", "std"]).reindex([False, True])
        ax.bar(x, grouped["mean"], yerr=grouped["std"], capsize=4, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Right-detected symmetric gray padding metrics by attack outcome")
    return fig


def plot_padding_distributions(df):
    import matplotlib.pyplot as plt

    metrics = [
        ("right_gray_padding_area_frac", "right-symmetric gray padding area fraction"),
        ("patch_right_gray_padding_overlap_frac", "patch overlap with right-symmetric padding"),
        ("object_distance_to_right_gray_padding", "object distance to right-symmetric padding"),
    ]
    colors = {False: "#F58518", True: "#4C78A8"}
    labels = {False: "fail", True: "success"}
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for success in (False, True):
            values = df.loc[df["success"] == success, metric].astype("float64")
            ax.hist(values, bins=50, density=True, alpha=0.4, color=colors[success], label=labels[success])
        ax.set_title(title)
        ax.set_xlabel(metric)
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Right-detected symmetric gray padding distributions")
    return fig


def plot_padding_metric_quality(quality_df, *, metric_prefix: str = "right_"):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    q = pd.DataFrame(quality_df).copy()
    if metric_prefix:
        q = q[q["metric"].astype(str).str.startswith(str(metric_prefix))]
    q = q.sort_values(["best_accuracy", "roc_auc"], ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, max(4.5, 0.42 * max(1, len(q)))), constrained_layout=True)
    y = np.arange(len(q), dtype=int)
    axes[0].barh(y, q["roc_auc"].astype("float64"), color="#72B7B2")
    axes[0].axvline(0.5, color="0.35", linestyle="--", linewidth=1.0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(q["metric"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("ROC-AUC")
    axes[0].set_title("Padding metric ROC-AUC")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(y, q["best_accuracy"].astype("float64"), color="#54A24B")
    axes[1].axvline(0.5, color="0.35", linestyle="--", linewidth=1.0)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("best accuracy")
    axes[1].set_title("Padding metric best accuracy")
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("Right-detected symmetric padding metric quality")
    return fig


def gray_padding_edge_summary(df, *, imgsz: int = 640):
    import pandas as pd

    out = df.copy()
    out["gray_left_inner_edge_x"] = out["gray_pad_left"].astype("float64")
    out["gray_right_inner_edge_x"] = float(imgsz) - out["gray_pad_right"].astype("float64")
    out["gray_left_right_width_diff"] = out["gray_pad_left"].astype("float64") - out["gray_pad_right"].astype("float64")
    columns = [
        "gray_pad_left",
        "gray_pad_right",
        "gray_left_inner_edge_x",
        "gray_right_inner_edge_x",
        "gray_left_right_width_diff",
    ]
    return pd.concat(
        {
            "overall": out[columns].agg(["mean", "std", "median"]),
            "fail": out.loc[out["success"] == False, columns].agg(["mean", "std", "median"]),
            "success": out.loc[out["success"] == True, columns].agg(["mean", "std", "median"]),
        },
        axis=0,
    )


def plot_gray_padding_edges_summary(df, *, imgsz: int = 640):
    import matplotlib.pyplot as plt
    import numpy as np

    groups = [
        ("overall", df, "#222222"),
        ("fail", df[df["success"] == False], "#F58518"),
        ("success", df[df["success"] == True], "#4C78A8"),
    ]
    fig, axes = plt.subplots(len(groups), 1, figsize=(12.5, 6.8), constrained_layout=True)
    if len(groups) == 1:
        axes = [axes]
    for ax, (label, sub, color) in zip(axes, groups, strict=True):
        left_edge = sub["gray_pad_left"].astype("float64")
        right_edge = float(imgsz) - sub["gray_pad_right"].astype("float64")
        left_mean, left_std = float(left_edge.mean()), float(left_edge.std())
        right_mean, right_std = float(right_edge.mean()), float(right_edge.std())

        canvas = np.ones((96, int(imgsz), 3), dtype="float64")
        ax.imshow(canvas, extent=(0, int(imgsz), 0, 1), aspect="auto")
        ax.axvspan(0, left_mean, color="0.65", alpha=0.35)
        ax.axvspan(right_mean, int(imgsz), color="0.65", alpha=0.35)
        ax.axvspan(max(0.0, left_mean - left_std), min(float(imgsz), left_mean + left_std), color=color, alpha=0.18)
        ax.axvspan(max(0.0, right_mean - right_std), min(float(imgsz), right_mean + right_std), color=color, alpha=0.18)
        ax.axvline(left_mean, color=color, linewidth=2.2, label=f"left edge mean={left_mean:.1f} +/- {left_std:.1f}")
        ax.axvline(right_mean, color=color, linewidth=2.2, linestyle="--", label=f"right edge mean={right_mean:.1f} +/- {right_std:.1f}")
        ax.set_xlim(0, int(imgsz))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("x coordinate in 640x640 image")
        ax.set_title(f"{label}: detected vertical gray-padding inner edges")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="upper center", ncol=2)
    fig.suptitle("Average detected gray padding edges with +/- std bands")
    return fig
