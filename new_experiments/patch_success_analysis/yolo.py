from __future__ import annotations

from pathlib import Path
from typing import Any


def pil_to_np_bgr(pil):
    import numpy as np

    rgb = np.asarray(pil.convert("RGB"))
    return rgb[..., ::-1].copy()


def load_yolo(model_path: str | Path, *, device: str | None = None):
    from ultralytics import YOLO

    yolo = YOLO(str(model_path))
    if device is not None:
        try:
            yolo.to(device)
        except Exception:
            pass
    return yolo


def get_torch_model(yolo):
    import torch

    model = getattr(yolo, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("Ultralytics YOLO object does not expose a torch nn.Module at .model")
    return model


def get_class_id(names: Any, class_name: str = "person") -> int | None:
    wanted = str(class_name).lower().strip()
    if isinstance(names, dict):
        for key, value in names.items():
            if str(value).lower().strip() == wanted:
                return int(key)
    if isinstance(names, (list, tuple)):
        for idx, value in enumerate(names):
            if str(value).lower().strip() == wanted:
                return int(idx)
    return None


def class_name_from_id(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id))
    if isinstance(names, (list, tuple)) and 0 <= int(class_id) < len(names):
        return str(names[int(class_id)])
    return str(class_id)


def yolo_predict_conf_scalar(
    yolo_model,
    pil_img,
    *,
    imgsz: int,
    target_class_id: int | None,
    conf: float = 0.001,
    device: str | None = None,
):
    import numpy as np

    result = yolo_model.predict(
        source=pil_to_np_bgr(pil_img),
        imgsz=int(imgsz),
        conf=float(conf),
        device=device,
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return 0.0, result
    confs = result.boxes.conf.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy().astype(int)
    if target_class_id is None:
        return float(confs.max(initial=0.0)), result
    mask = clss == int(target_class_id)
    if not np.any(mask):
        return 0.0, result
    return float(confs[mask].max(initial=0.0)), result


def detection_dict_from_result(res, *, target_class_id: int | None) -> dict[str, Any] | None:
    import numpy as np

    if res is None or getattr(res, "boxes", None) is None or len(res.boxes) == 0:
        return None
    xyxy = res.boxes.xyxy.detach().cpu().numpy()
    conf = res.boxes.conf.detach().cpu().numpy()
    cls = res.boxes.cls.detach().cpu().numpy().astype(int)
    if target_class_id is None:
        idx = int(np.argmax(conf))
    else:
        mask = cls == int(target_class_id)
        if not np.any(mask):
            return None
        candidates = np.flatnonzero(mask)
        idx = int(candidates[int(np.argmax(conf[candidates]))])
    names = getattr(res, "names", {})
    return {
        "class_id": int(cls[idx]),
        "class_name": class_name_from_id(names, int(cls[idx])),
        "confidence": float(conf[idx]),
        "bbox_xyxy_orig": [float(v) for v in xyxy[idx].tolist()],
        "result_index": idx,
    }


def detection_choice_from_dict(path: str | Path, data: dict[str, Any]):
    from segmentig_detector.yolo_utils import DetectionChoice

    return DetectionChoice(
        image_path=Path(path),
        class_id=int(data["class_id"]),
        class_name=str(data.get("class_name", data["class_id"])),
        confidence=float(data["confidence"]),
        bbox_xyxy_orig=tuple(float(v) for v in data["bbox_xyxy_orig"]),
    )


def get_module_by_name(model, name: str):
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Layer {name!r} was not found.")
    return modules[name]
