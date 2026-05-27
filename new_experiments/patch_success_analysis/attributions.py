from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TargetFn = Callable[[object, object], object]


@dataclass(slots=True)
class AttributionResult:
    method: str
    layer_name: str
    attribution: object
    activation_shape: tuple[int, ...]
    elapsed_s: float

    def flat(self):
        return self.attribution.detach().reshape(self.attribution.shape[0], -1)

    def chw(self):
        return self.attribution.detach().reshape((self.attribution.shape[0],) + self.activation_shape)


def detector_target_fn(fixed_target, *, imgsz: int = 640, detect_name: str = "model.23", mode: str = "class_only"):
    from segmentig_detector.targets import detector_target_scalar

    def target(model, x):
        return detector_target_scalar(
            model,
            x,
            fixed_target,
            mode=mode,
            imgsz=int(imgsz),
            detect_name=detect_name,
        )

    return target


def compute_layer_ig_attribution(
    model,
    inputs,
    baselines,
    *,
    target_fn: TargetFn,
    layer,
    layer_name: str,
    method: str,
    n_steps: int = 64,
    alpha_batch_size: int = 4,
    segment_start: float = 0.0,
    segment_end: float = 1.0,
) -> AttributionResult:
    import time

    from segmentig_detector.layer_segmentig import compute_layer_segmentig_scalar

    t0 = time.perf_counter()
    result = compute_layer_segmentig_scalar(
        model,
        inputs,
        baselines,
        target_fn=target_fn,
        layer=layer,
        n_steps=int(n_steps),
        alpha_batch_size=int(alpha_batch_size),
        segment_start=float(segment_start),
        segment_end=float(segment_end),
    )
    attr = result.raw_attribution_bchw().detach()
    return AttributionResult(
        method=method,
        layer_name=layer_name,
        attribution=attr,
        activation_shape=tuple(int(v) for v in result.activation_shape),
        elapsed_s=float(time.perf_counter() - t0),
    )


class _ActivationHook:
    def __init__(self, layer):
        self.activation = None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        import torch

        if isinstance(output, torch.Tensor):
            self.activation = output
            return
        if isinstance(output, (list, tuple)):
            self.activation = next((item for item in output if isinstance(item, torch.Tensor)), None)

    def clear(self):
        self.activation = None

    def get(self):
        if self.activation is None:
            raise RuntimeError("Layer hook did not capture an activation.")
        return self.activation

    def remove(self):
        self.handle.remove()


def _activation_and_gradient(model, inputs, *, target_fn: TargetFn, layer):
    import torch

    hook = _ActivationHook(layer)
    try:
        hook.clear()
        x = inputs.detach().requires_grad_(True)
        score = target_fn(model, x)
        if score.ndim != 0:
            score = score.sum()
        activation = hook.get()
        grad = torch.autograd.grad(
            score,
            activation,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]
        if grad is None:
            grad = torch.zeros_like(activation)
        return activation.detach(), grad.detach()
    finally:
        hook.remove()


def _activation_only(model, inputs, *, layer):
    from segmentig_detector.yolo_utils import safe_model_forward

    hook = _ActivationHook(layer)
    try:
        hook.clear()
        safe_model_forward(model, inputs)
        return hook.get().detach()
    finally:
        hook.remove()


def compute_naa_attribution(
    model,
    inputs,
    baselines,
    *,
    target_fn: TargetFn,
    layer,
    layer_name: str,
    ia_gradient=None,
) -> AttributionResult:
    import time

    t0 = time.perf_counter()
    act_input, grad = _activation_and_gradient(model, inputs, target_fn=target_fn, layer=layer)
    act_base = _activation_only(model, baselines, layer=layer)
    ia = grad if ia_gradient is None else ia_gradient.to(device=act_input.device, dtype=act_input.dtype)
    attr = (act_input - act_base) * ia
    return AttributionResult(
        method="NAA",
        layer_name=layer_name,
        attribution=attr.detach(),
        activation_shape=tuple(int(v) for v in attr.shape[1:]),
        elapsed_s=float(time.perf_counter() - t0),
    )


def mean_ia_gradient(contexts: list[tuple[object, object]], *, model, layer):
    import torch

    grads = []
    for inputs, target_fn in contexts:
        _act, grad = _activation_and_gradient(model, inputs, target_fn=target_fn, layer=layer)
        grads.append(grad.detach())
    if not grads:
        raise ValueError("Cannot compute NAA IA gradient from an empty context list.")
    return torch.stack(grads, dim=0).mean(dim=0)


def attribution_spatial_map(attribution_bchw, *, reduction: str = "l2"):
    from .activations import reduce_chw_to_hw, robust_normalize

    hw = reduce_chw_to_hw(attribution_bchw, mode=reduction)
    return robust_normalize(hw, signed=(reduction == "signed_mean"))


def aggregate_flat_abs(results: list[AttributionResult]):
    import torch

    if not results:
        raise ValueError("No attribution results to aggregate.")
    flats = [item.flat().abs().detach().cpu() for item in results]
    return torch.cat(flats, dim=0).mean(dim=0)
