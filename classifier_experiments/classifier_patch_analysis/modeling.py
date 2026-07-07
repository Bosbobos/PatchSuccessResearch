from __future__ import annotations

from pathlib import Path
from typing import Any


def select_device(device: str | None = "auto"):
    import torch

    if device and str(device).lower() != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _replace_cls_head_with_one_logit(model):
    import torch.nn as nn

    head = getattr(model, "model", None)
    if head is None or len(head) == 0:
        raise RuntimeError("Unexpected YOLO classification model structure: missing model list.")
    cls = head[-1]
    linear = getattr(cls, "linear", None)
    if not isinstance(linear, nn.Linear):
        raise RuntimeError("Unexpected YOLO classification head: missing .linear nn.Linear.")
    cls.linear = nn.Linear(linear.in_features, 1, bias=linear.bias is not None)
    return model


def load_yolo_cls_model(
    checkpoint_path: str | Path,
    *,
    base_weights_path: str | Path = "data/yolo11s-cls.pt",
    device: str | None = "auto",
):
    import torch
    from ultralytics import YOLO

    base_weights_path = Path(base_weights_path)
    checkpoint_path = Path(checkpoint_path)
    if not base_weights_path.exists():
        raise FileNotFoundError(
            f"Missing YOLO11-cls skeleton weights: {base_weights_path}. "
            "Put yolo11s-cls.pt there explicitly; this code does not download weights."
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing one-logit classifier checkpoint: {checkpoint_path}")

    yolo = YOLO(str(base_weights_path))
    model = _replace_cls_head_with_one_logit(yolo.model)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid classifier checkpoint {checkpoint_path}: expected a dict with key 'model_state_dict'."
        )
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading classifier checkpoint: {unexpected[:20]}")
    if missing:
        raise RuntimeError(f"Missing keys while loading classifier checkpoint: {missing[:20]}")

    dev = select_device(device)
    model.to(dev)
    model.eval()
    return model


def forward_logits(model, x):
    import torch

    out = model(x)
    if isinstance(out, torch.Tensor):
        logits = out
    elif isinstance(out, (list, tuple)):
        tensors = [item for item in out if isinstance(item, torch.Tensor)]
        if not tensors:
            raise RuntimeError(f"Model returned no tensor logits: {type(out)}")
        logits = next((item for item in reversed(tensors) if item.ndim == 2), tensors[-1])
    elif isinstance(out, dict):
        logits = next((out[key] for key in ("logits", "pred", "output") if key in out), None)
        if logits is None:
            raise RuntimeError(f"Model returned dict without logits-like keys: {list(out)}")
    else:
        raise RuntimeError(f"Unsupported model output type: {type(out)}")
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    raise RuntimeError(f"Expected one-logit classifier output [B] or [B,1], got {tuple(logits.shape)}")


def predict_person_confidence(model, batch):
    import torch

    with torch.no_grad():
        return torch.sigmoid(forward_logits(model, batch)).detach().cpu()


def preprocess_pil_batch(images: list[Any], *, img_size: int, device, dtype=None):
    import numpy as np
    import torch

    tensors = []
    for image in images:
        arr = np.asarray(image.convert("RGB").resize((int(img_size), int(img_size))), dtype="float32") / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    x = torch.stack(tensors, dim=0).to(device=device)
    if dtype is not None:
        x = x.to(dtype=dtype)
    return x
