# Defensive mechanism stress suite

- train/eval: 16/16 (disjoint)
- clean-visible eval: 16
- baseline failure rate: 0.438
- elapsed: 372.7 s

| mechanism | failure_reproduction_rate | component_coefficient_after | component_cosine_after | final_max_target_iou | final_max_target_score | patch_tv |
| --- | --- | --- | --- | --- | --- | --- |
| candidate_handoff | 0.750 | 1.050 | 0.156 | 0.853 | 0.250 | 0.674 |
| clean_signature_matched | 0.750 | 1.050 | 0.156 | 0.853 | 0.250 | 0.674 |
| component_minimal | 0.750 | 1.049 | 0.156 | 0.851 | 0.248 | 0.674 |
| cross_scale | 0.750 | 1.050 | 0.156 | 0.851 | 0.252 | 0.674 |
| naturalistic | 0.750 | 1.050 | 0.156 | 0.853 | 0.251 | 0.674 |
| nonlinear_residual | 0.750 | 1.050 | 0.156 | 0.853 | 0.250 | 0.674 |
| reserve_score | 0.750 | 1.050 | 0.156 | 0.854 | 0.251 | 0.674 |
| tail_compact | 0.750 | 1.050 | 0.156 | 0.854 | 0.251 | 0.674 |
| tail_diffuse | 0.750 | 1.050 | 0.156 | 0.854 | 0.251 | 0.674 |
| dormant_geometry | 0.688 | 0.965 | 0.145 | 0.868 | 0.191 | 0.674 |
| nms_only | 0.312 | 0.990 | 0.157 | 0.916 | 0.716 | 0.672 |
| geometry_only | 0.000 | 0.142 | 0.035 | 0.954 | 0.844 | 0.673 |
