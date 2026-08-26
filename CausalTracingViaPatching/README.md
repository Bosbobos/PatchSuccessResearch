# CausalTracingViaPatching

This folder contains a first causal tracing experiment for YOLO adversarial
patch failures.

The core idea is activation repair. For a clean/patched image pair, we capture
an intermediate activation tensor `A_clean[L]` and `A_patch[L]` at a detector
layer. During a patched-image forward pass, we replace selected entries of
`A_patch[L]` with their clean values and measure whether the fixed clean
detection target recovers.

The first notebook compares three families of repair masks:

- **Spatial**: restore the patch region, object region, and their complements.
- **Importance-guided**: restore `[C,H,W]` neurons ranked by absolute SegmentIG
  importance, both descending and ascending.
- **Delta-guided**: restore neurons ranked by `abs(A_patch - A_clean)`, both
  descending and ascending.

Random and delta-matched random controls are included to separate true ranking
quality from the amount of counter-delta inserted.

The primary score is the fixed pre-NMS detector target class logit. Recovery is
normalized as:

```text
recovery = (score_intervened - score_patched) / max(eps, score_clean - score_patched)
```

The fair comparison axis is counter-delta budget, especially
`counterdelta_l2_frac`: the fraction of the total activation delta L2 norm that
was replaced by clean values.
