from __future__ import annotations

"""Layer SegmentIG for arbitrary scalar model targets.

This is the detector-oriented sibling of SegmentIGImpl/portable_layer_ig.py.
The key difference is that callers provide target_fn(model, x) -> scalar,
instead of classification logits plus target class ids.
"""

from dataclasses import dataclass
from typing import Callable

import torch
from torch.func import jvp


SelectionMode = str
TargetFn = Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]


@dataclass(slots=True)
class LayerScalarIGResult:
    attribution: torch.Tensor
    raw_attribution: torch.Tensor
    activation_shape: tuple[int, ...]
    alpha_values: torch.Tensor
    selected_mask: torch.Tensor | None
    selected_positive_counts: list[int] | None
    selected_negative_counts: list[int] | None

    def attribution_bchw(self) -> torch.Tensor:
        """Return attribution reshaped as [B, *activation_shape]."""
        return self.attribution.reshape((self.attribution.shape[0],) + self.activation_shape)

    def raw_attribution_bchw(self) -> torch.Tensor:
        """Return dense attribution reshaped as [B, *activation_shape]."""
        return self.raw_attribution.reshape((self.raw_attribution.shape[0],) + self.activation_shape)


class LayerActivationHook:
    def __init__(self, layer: torch.nn.Module):
        self.activation: torch.Tensor | None = None
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output) -> None:
        if isinstance(output, torch.Tensor):
            self.activation = output
            return
        if isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    self.activation = item
                    return
        self.activation = None

    def clear(self) -> None:
        self.activation = None

    def get(self) -> torch.Tensor:
        if self.activation is None:
            raise RuntimeError("Layer hook did not capture a tensor activation.")
        return self.activation

    def remove(self) -> None:
        self._handle.remove()


def midpoint_alphas(
    n_steps: int,
    *,
    segment_start: float,
    segment_end: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    width = float(segment_end - segment_start)
    return segment_start + width * ((torch.arange(n_steps, device=device, dtype=dtype) + 0.5) / float(n_steps))


def _validate_inputs(
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    *,
    n_steps: int,
    alpha_batch_size: int,
    segment_start: float,
    segment_end: float,
    top_k: int | None,
    selection_mode: str,
) -> None:
    if inputs.shape != baselines.shape:
        raise ValueError(f"inputs and baselines must have same shape, got {inputs.shape} and {baselines.shape}")
    if inputs.ndim < 2:
        raise ValueError(f"inputs must include a batch dimension, got {inputs.shape}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if alpha_batch_size <= 0:
        raise ValueError(f"alpha_batch_size must be positive, got {alpha_batch_size}")
    if not 0.0 <= segment_start < segment_end <= 1.0:
        raise ValueError(f"segment must satisfy 0 <= start < end <= 1, got [{segment_start}, {segment_end}]")
    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k must be positive or None, got {top_k}")
    if selection_mode not in {"signed", "positive", "unsigned"}:
        raise ValueError(f"Unsupported selection_mode={selection_mode!r}")


def _topk_mask(
    scores: torch.Tensor,
    *,
    top_k: int,
    selection_mode: str,
) -> tuple[torch.Tensor, list[int], list[int]]:
    batch_size, width = scores.shape
    positive_mask = torch.zeros((batch_size, width), device=scores.device, dtype=torch.bool)
    negative_mask = torch.zeros((batch_size, width), device=scores.device, dtype=torch.bool)

    def select(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        k = min(int(top_k), int(values.shape[1]))
        masked = values.masked_fill(~valid, float("-inf"))
        top_values, top_indices = torch.topk(masked, k=k, dim=1)
        chosen = torch.zeros_like(valid)
        chosen.scatter_(1, top_indices, torch.isfinite(top_values) & (top_values > 0.0))
        return chosen & valid

    if selection_mode in {"signed", "positive"}:
        positive_mask = select(scores, scores > 0.0)
    if selection_mode == "signed":
        negative_mask = select(scores.abs(), scores < 0.0)
    elif selection_mode == "unsigned":
        values, indices = torch.topk(scores.abs(), k=min(int(top_k), width), dim=1)
        selected = torch.zeros_like(positive_mask)
        selected.scatter_(1, indices, values > 0.0)
        positive_mask = selected & (scores >= 0.0)
        negative_mask = selected & (scores < 0.0)

    selected_mask = positive_mask | negative_mask
    return (
        selected_mask,
        [int(v) for v in positive_mask.sum(dim=1).detach().cpu().tolist()],
        [int(v) for v in negative_mask.sum(dim=1).detach().cpu().tolist()],
    )


def _as_scalar(score: torch.Tensor) -> torch.Tensor:
    if not isinstance(score, torch.Tensor):
        raise TypeError(f"target_fn must return a torch.Tensor, got {type(score)}")
    if score.ndim == 0:
        return score
    return score.sum()


def compute_layer_segmentig_scalar(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    *,
    target_fn: TargetFn,
    layer: torch.nn.Module,
    n_steps: int = 64,
    alpha_batch_size: int = 4,
    segment_start: float = 0.0,
    segment_end: float = 0.1,
    top_k: int | None = None,
    selection_mode: SelectionMode = "signed",
) -> LayerScalarIGResult:
    """Compute layer SegmentIG/conductance for an arbitrary scalar target.

    target_fn is called as target_fn(model, x_batch). For detector targets it
    should return a scalar sum over the x_batch rows, using any fixed metadata
    selected before this call.
    """

    _validate_inputs(
        inputs,
        baselines,
        n_steps=n_steps,
        alpha_batch_size=alpha_batch_size,
        segment_start=segment_start,
        segment_end=segment_end,
        top_k=top_k,
        selection_mode=selection_mode,
    )

    alphas = midpoint_alphas(
        n_steps,
        segment_start=segment_start,
        segment_end=segment_end,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    delta_x = (inputs - baselines).contiguous()
    hook = LayerActivationHook(layer)

    def forward_with_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hook.clear()
        score = _as_scalar(target_fn(model, x))
        activation = hook.get()
        return score, activation

    def flat_activation(x: torch.Tensor) -> torch.Tensor:
        _, activation = forward_with_activation(x)
        return activation.reshape(activation.shape[0], -1)

    try:
        with torch.no_grad():
            _, clean_activation = forward_with_activation(inputs)
            activation_shape = tuple(clean_activation.shape[1:])

        total_attr: torch.Tensor | None = None
        batch_size = int(inputs.shape[0])
        for start in range(0, int(alphas.numel()), int(alpha_batch_size)):
            alpha_batch = alphas[start : start + int(alpha_batch_size)]
            alpha_count = int(alpha_batch.numel())

            view_shape = (alpha_count,) + (1,) * inputs.ndim
            x_alpha = baselines.unsqueeze(0) + alpha_batch.reshape(view_shape) * delta_x.unsqueeze(0)
            x_alpha = x_alpha.reshape((alpha_count * batch_size,) + tuple(inputs.shape[1:]))
            x_alpha = x_alpha.contiguous().detach().requires_grad_(True)

            score, activation = forward_with_activation(x_alpha)
            if not bool(score.requires_grad):
                raise RuntimeError("target_fn returned a scalar that does not require grad.")
            grad_activation = torch.autograd.grad(
                score,
                activation,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad_activation is None:
                grad_activation = torch.zeros_like(activation)

            delta_batch = delta_x.unsqueeze(0).expand(alpha_count, -1, *([-1] * (inputs.ndim - 1)))
            delta_batch = delta_batch.reshape_as(x_alpha).contiguous()
            _, activation_direction = jvp(flat_activation, (x_alpha,), (delta_batch,))

            grad_flat = grad_activation.detach().reshape(alpha_count, batch_size, -1)
            dir_flat = activation_direction.detach().reshape(alpha_count, batch_size, -1)
            contribution = (grad_flat * dir_flat).sum(dim=0)
            total_attr = contribution if total_attr is None else total_attr + contribution

            del x_alpha, score, activation, grad_activation
            del delta_batch, activation_direction, grad_flat, dir_flat, contribution

        if total_attr is None:
            raise RuntimeError("No alpha samples were evaluated.")

        raw_attribution = (total_attr / float(alphas.numel())).detach()
        selected_mask = None
        selected_positive_counts = None
        selected_negative_counts = None
        attribution = raw_attribution
        if top_k is not None:
            selected_mask, selected_positive_counts, selected_negative_counts = _topk_mask(
                raw_attribution,
                top_k=int(top_k),
                selection_mode=selection_mode,
            )
            attribution = raw_attribution * selected_mask.to(dtype=raw_attribution.dtype)

        return LayerScalarIGResult(
            attribution=attribution.detach(),
            raw_attribution=raw_attribution.detach(),
            activation_shape=activation_shape,
            alpha_values=alphas.detach(),
            selected_mask=selected_mask.detach() if selected_mask is not None else None,
            selected_positive_counts=selected_positive_counts,
            selected_negative_counts=selected_negative_counts,
        )
    finally:
        hook.remove()
