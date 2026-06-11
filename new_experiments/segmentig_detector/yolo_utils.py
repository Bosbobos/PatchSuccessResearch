from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class DetectionChoice:
    image_path: Path
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy_orig: tuple[float, float, float, float]


def list_image_paths(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_np_bgr(pil: Image.Image) -> np.ndarray:
    rgb = np.asarray(pil.convert("RGB"))
    return rgb[..., ::-1].copy()


def load_yolo(weights: str | Path):
    from ultralytics import YOLO

    return YOLO(str(weights))


def get_torch_model(yolo) -> torch.nn.Module:
    model = getattr(yolo, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("Ultralytics YOLO object does not expose a torch nn.Module at .model")
    return model


def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Layer {name!r} was not found in model.named_modules().")
    return modules[name]


def get_detect_module(model: torch.nn.Module, name: str = "model.23") -> torch.nn.Module:
    try:
        return get_module_by_name(model, name)
    except KeyError:
        for module in model.modules():
            if module.__class__.__name__ == "Detect":
                return module
    raise KeyError("Could not locate Ultralytics Detect head.")


def get_class_id(names: Any, class_name: str = "person", fallback: int = 0) -> int:
    wanted = class_name.lower().strip()
    if isinstance(names, dict):
        for key, value in names.items():
            if str(value).lower().strip() == wanted:
                return int(key)
    if isinstance(names, (list, tuple)):
        for idx, value in enumerate(names):
            if str(value).lower().strip() == wanted:
                return int(idx)
    return int(fallback)


def class_name_from_id(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id))
    if isinstance(names, (list, tuple)) and 0 <= int(class_id) < len(names):
        return str(names[int(class_id)])
    return str(class_id)


def ensure_predictor(
    yolo,
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
):
    if getattr(yolo, "predictor", None) is None:
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        _ = yolo.predict(source=dummy, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)

    yolo.predictor.args.imgsz = imgsz
    yolo.predictor.args.conf = conf
    yolo.predictor.args.iou = iou
    if device is not None:
        yolo.predictor.device = torch.device(device)
        yolo.predictor.model.to(device)
    return yolo.predictor


def preprocess_pil(
    yolo,
    pil: Image.Image,
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
) -> dict[str, Any]:
    predictor = ensure_predictor(yolo, imgsz=imgsz, conf=conf, iou=iou, device=device)
    orig_bgr = pil_to_np_bgr(pil)
    im = predictor.preprocess([orig_bgr])
    return {
        "predictor": predictor,
        "orig_bgr": orig_bgr,
        "im": im,
        "im_hw": (int(im.shape[2]), int(im.shape[3])),
        "orig_hw": (int(orig_bgr.shape[0]), int(orig_bgr.shape[1])),
    }


@torch.no_grad()
def predict_best_person_detection(
    yolo,
    image_path: str | Path,
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
    person_class_id: int | None = None,
) -> DetectionChoice | None:
    pil = load_rgb(image_path)
    bgr = pil_to_np_bgr(pil)
    result = yolo.predict(source=bgr, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
    names = getattr(yolo, "names", getattr(getattr(yolo, "model", None), "names", {}))
    person_class_id = get_class_id(names, "person", 0) if person_class_id is None else int(person_class_id)

    if result.boxes is None or len(result.boxes) == 0:
        return None
    cls = result.boxes.cls.detach().cpu().long()
    mask = cls == int(person_class_id)
    if not bool(mask.any()):
        return None

    confs = result.boxes.conf.detach().cpu()
    person_indices = torch.nonzero(mask, as_tuple=False).flatten()
    best_local = torch.argmax(confs[person_indices])
    best_idx = int(person_indices[int(best_local)].item())
    bbox = result.boxes.xyxy[best_idx].detach().cpu().to(torch.float32).tolist()
    return DetectionChoice(
        image_path=Path(image_path),
        class_id=int(person_class_id),
        class_name=class_name_from_id(names, int(person_class_id)),
        confidence=float(confs[best_idx].item()),
        bbox_xyxy_orig=tuple(float(v) for v in bbox),
    )


def select_images_with_person(
    yolo,
    image_paths: Iterable[str | Path],
    *,
    count: int = 3,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
    person_class_id: int | None = None,
) -> list[DetectionChoice]:
    choices: list[DetectionChoice] = []
    for image_path in image_paths:
        choice = predict_best_person_detection(
            yolo,
            image_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            person_class_id=person_class_id,
        )
        if choice is not None:
            choices.append(choice)
            if len(choices) >= int(count):
                break
    if len(choices) < int(count):
        raise RuntimeError(f"Found only {len(choices)} images with person detections; need {count}.")
    return choices


def safe_model_forward(model: torch.nn.Module, x: torch.Tensor) -> Any:
    try:
        return model(x, augment=False, visualize=False)
    except TypeError:
        return model(x)


def detect_train_levels(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    detect_name: str = "model.23",
) -> tuple[list[torch.Tensor], torch.nn.Module]:
    detect = get_detect_module(model, detect_name)
    was_training = bool(detect.training)
    detect.train()
    try:
        out = safe_model_forward(model, x)
    finally:
        if not was_training:
            detect.eval()

    pred_levels: Any = out
    if isinstance(out, (list, tuple)) and len(out) == 2 and isinstance(out[0], torch.Tensor):
        pred_levels = out[1]
    if not isinstance(pred_levels, (list, tuple)):
        raise RuntimeError(f"Unexpected Detect train output type: {type(pred_levels)}")
    levels = [t for t in pred_levels if isinstance(t, torch.Tensor) and t.ndim == 4]
    if not levels:
        raise RuntimeError("Detect train output did not contain 4D prediction levels.")
    return levels, detect


def raw_inference_tensor(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = safe_model_forward(model, x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise RuntimeError(f"Unexpected model inference output type: {type(out)}")
    if out.ndim != 3:
        raise RuntimeError(f"Expected raw prediction tensor [B,C,N], got shape {tuple(out.shape)}")
    return out


def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    return torch.stack((x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0), dim=1)


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.to(dtype=torch.float32)
    b = b.to(dtype=torch.float32, device=a.device)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))[None, :]
    return inter / (area_a + area_b - inter).clamp(min=1e-9)
