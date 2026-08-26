# Mechanism-aware pixel patch

- train/eval: 64/64 (disjoint)
- clean-visible eval targets: 63
- initial patch hidden: 26/63 (0.413)
- hidden after patch: 55/63 (0.873)
- absolute hiding-rate gain: +0.460
- elapsed: 157.3 s

Objective: differentiable smooth maximum over the dynamic clean-target candidate reserve, including candidate geometry, plus TV regularization.
