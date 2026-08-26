# Candidate routing and attack-path decomposition

This folder implements the first two follow-up stages of the adversarial-patch
mechanism study for YOLO11s.

## Stage 1: candidate tracing

For every clean/patched pair the experiment records:

- top pre-NMS person candidates on P3/P4/P5;
- decoded score, pre-sigmoid class logit, DFL geometry and bounding box;
- whether each candidate survived NMS;
- the clean target at the same cell after patching;
- the patched global winner;
- Hungarian geometric lineage between the top clean and patched candidates;
- an observable routing mode:
  `same_pre_nms_candidate`, same-object rerouting, competing candidate,
  geometry/object shift, replacement/fabrication, or disappearance.

The labels are intentionally descriptive. They do not claim a causal mechanism
before the attack-path and repair experiments validate one.

## Stage 2: signed clean-to-patched attack path

The experiment captures all three inputs to the Detect head: P3, P4 and P5.
For a fixed target score it computes

```text
C_i = (A_patch_i - A_clean_i)
      * integral_0^1 d score(A_clean + alpha * deltaA) / d A_i d alpha
```

The sum of all `C_i` approximates the exact pre-sigmoid score change. Two target
types are enabled by default:

- `tracked_clean`: the clean target cell;
- `winner_margin`: clean target score minus patched-winner score, using fixed
  endpoint-selected cells.

The clean first-order approximation is saved as a control. Its residual against
the exact score change measures how much a single local gradient misses.

## Storage contract

- Images and attack caches are reused by path; they are never copied.
- Resumable tables are stored in SQLite.
- Full P3/P4/P5 tensors are never written to disk.
- Only aggregate rows, configurable top neuron contributions, and small
  channel-reduced spatial maps are saved.
- The default hard limit for this folder is **8 GiB**. The existing historical
  patch-success outputs occupy about 11 GiB, so this conservative limit keeps
  their combined analysis footprint below 20 GiB.
- Every batch checks the budget and aborts before the folder can exceed it.

Typical full candidate tracing with `top_k=50` should be far below 1 GiB. The
default 100-example path experiment should normally be tens to hundreds of MiB.

## Token-efficient result files

Read these before opening notebooks or SQLite databases:

- `summary.json`;
- `analysis_digest.md`;
- Stage 1: `mode_summary.csv`, `level_transition_summary.csv`,
  `score_summary.csv`;
- Stage 2: `group_summary.csv`, `level_summary.csv`.

They are deliberately small and contain the main numerical conclusions.

## Running

Use the repository's `IAD` conda environment and open
`CandidateRoutingAndAttackPath.ipynb`. Its current configuration runs candidate
tracing over all 6000 cached examples and a stratified attack-path decomposition
over 200 examples. This scale was enabled after the corrected smoke run matched
clean pre-NMS targets to post-NMS detections exactly and achieved median relative
completeness error below `0.001`.

SQLite makes both stages resumable: rerunning the same configuration skips
completed examples. Changing a scientifically relevant configuration produces a
new hash-named output directory.

## Canonical success definition

`01_TargetDefinitionAndMetrics.ipynb` is the canonical entry point for
success/failure analyses.
It fixes the target before the attack as the highest-confidence clean `person`,
matches patched detections to that target by IoU, and uses `target_hidden` at the
configured deployment confidence threshold as the primary label. The older
image-level maximum-confidence drop is retained only as `legacy_success`.

## Balanced target-aware causal path

`02_BalancedCausalPath.ipynb` creates a matched, mutually exclusive four-group
sample:

- `visible_target_winner`;
- `visible_non_target_winner`;
- `hidden_low_conf_match`;
- `hidden_no_iou_match`.

It matches 100 examples per group on clean-target confidence and log target
area. Patch area is retained as a balance check; it is constant at 160 x 160 in
the current cache and is therefore excluded automatically from the distance.
The causal-path run uses only the pre-attack `tracked_clean` target. Selection,
balance diagnostics, group summaries, pairwise effect sizes, corrected tests,
and a short digest are written to compact files. The run is resumable through
its SQLite database.

## Sign-selective causal repair

`03_CausalRepair.ipynb` tests whether the signed path components are
necessary mediators rather than only correlates. On patched Detect inputs it
restores selected coordinates to their clean values and measures the fixed-cell
logit, decoded box, post-NMS target confidence, maximum target IoU, and target
rescue at the canonical thresholds.

The primary `top_negative` repair is compared with positive-sign, unsigned
attribution, activation-magnitude, and delta-matched random controls. Two clean
head oracles provide an upper bound. Results are resumable in SQLite and the
main dose-response table is also written as `repair_group_summary.csv`.

## Signed causal transplant

`04_CausalTransplant.ipynb` tests sufficiency in the reverse direction. It
starts from clean Detect inputs and replaces selected coordinates with their
patched values. The same signed, unsigned, magnitude, and random strategies are
compared at equal coordinate budgets. The primary endpoint is reproduction of
target hiding; logit, confidence, maximum IoU, and fixed-cell box IoU losses are
saved as continuous endpoints.

The run stores only numeric rows in SQLite and writes compact
`transplant_group_summary.csv` and `transplant_pairwise.csv` files. No activation
tensors are persisted.

## Target candidate-set redundancy

`05_TargetCandidateSet.ipynb` replaces the single fixed-cell objective with a
smooth maximum over clean candidates whose boxes match the target. It records
the exact flat index that supplies the post-intervention target detection, so a
handoff to a neighboring candidate is measured directly. Fixed-cell and
candidate-set signed coordinates are then transplanted at equal budgets.

The compact outputs are `candidate_set_group_summary.csv`,
`candidate_handoff_summary.csv`, and `candidate_set_pairwise.csv`. Full
activation tensors are not saved.

## Notebook map

1. `01_TargetDefinitionAndMetrics.ipynb`: target label, label audit, historical
   metric reevaluation, and candidate-routing leaderboard.
2. `02_BalancedCausalPath.ipynb`: matched selection, path decomposition, paired
   statistics, and path visualizations.
3. `03_CausalRepair.ipynb`: sign-selective repair, controls, and dose-response.
4. `04_CausalTransplant.ipynb`: sufficiency test by transplanting the selected
   attack-associated changes into clean Detect inputs.
5. `05_TargetCandidateSet.ipynb`: direct candidate-handoff audit and
   multi-candidate causal sufficiency test.

## Follow-up direction-selection notebooks

The next stage is intentionally split into three independent result notebooks.
Heavy computation is run from `run_followup_experiments.py`; the notebooks only
load compact CSV/JSON tables and render figures. Mechanism and attack runs use
MPS, while the defense run operates on saved candidate tables and stays on CPU.

6. `06_MechanismFollowups.ipynb`: a 2x2 clean/patched classification-versus-box
   branch factorial, explicit pre-NMS/post-NMS separation, and validation of
   the straight activation path against a real image-opacity path.
7. `07_AttackDirection.ipynb`: a controlled activation-space feasibility
   oracle comparing fixed-cell, static candidate-set, and differentiable
   dynamic score-plus-geometry objectives. It does not synthesize a pixel or
   physical patch.
8. `08_DefenseDirection.ipynb`: a post-hoc RoutePool pilot that clusters and
   aggregates saved pre-NMS candidates. Thresholds are calibrated on a train
   split to match clean post-NMS output volume and evaluated on a disjoint test
   split.
9. `09_EnsembleMargin.ipynb`: a full 5,985-example score-only test of whether
   the clean target-candidate ensemble predicts target hiding better than the
   single tracked cell. It includes a trace-versus-legacy-label pipeline audit,
   deterministic train/test split, bootstrap intervals, calibration, and
   per-example candidate bar charts.
10. `10_CandidateReserveCausal.ipynb`: target-specific necessity/sufficiency
    interventions on actual patch-induced person logits. It compares tracked,
    top-1/2/4/6/8/9/10/12, and all clean target candidates against score/level-matched
    random controls while holding geometry at the patched or clean endpoint.
11. `11_SharedCandidateMechanism.ipynb`: SVD of `5x5` Detect-input windows
    around the fixed clean target reserve, followed by rank-k repair, reverse
    transplant, translated-layout controls, and activation-energy-matched
    controls. The 400-example run rejects a single rank-1 bottleneck but
    localizes a multidimensional target-local mediator: full-window transplant
    reproduces 178/200 hidden-source and 0/200 visible-source outcomes.
12. `12_ScoreFunctionalSubspace.ipynb`: a score-first, path-integrated vector
    Jacobian of the fixed clean candidate reserve. Classification inputs are
    intervened while box outputs stay at their clean/patched endpoint. The
    score-functional row space contains about 0.64% of local activation energy
    but nearly reaches the full class-window repair oracle; equal-energy random
    and magnitude controls have almost no effect.
13. `13_FullSuccessCausalClosure.ipynb`: exact class/box/NMS decomposition plus
    a nested Detect-input closure test. On 193 reproducibly hidden endpoints,
    a radius-4 neighborhood around target-candidate routes reproduces 193/193
    attacks and 0 genuinely visible controls while covering about 5.2% of
    Detect-input coordinates. A joint person-logit/decoded-IoU row space contains
    about 0.29% of full feature-delta energy; removing it repairs every hidden
    target in the functional subset, while complete sufficiency additionally
    requires local nonlinear context in roughly half the examples.
14. `14_SelfCounterfactualDefense.ipynb`: a no-paired-clean gray-box pilot.
    Three counterfactual views of the known patch box define a pseudo-target and
    a proxy-to-observed functional direction. Full masking and the target-only
    joint correction recover 47/47 hidden targets, but the latter restores the
    full clean detection set poorly (`F1=0.334` on hidden cases versus `0.953`
    for masking). On genuinely clean inputs the ordering reverses
    (`F1=0.996` versus `0.938`). This motivates a global, all-candidate
    proxy-functional correction rather than a target-only one.
15. `15_BlindSelfCounterfactualDefense.ipynb`: removes the patch-location
    oracle. A coarse-to-fine multi-window search uses only agreement among
    three counterfactual views of the observed image. Without paired clean or
    patch coordinates it recovers 45/47 hidden targets with the target-only
    component (46/47 with direct proxy masking). A clean-only multivariate
    consistency gate recovers 36/47 while activating on 2/50 held-out clean
    inputs; this is exploratory and requires a larger, varied-location
    confirmatory cohort.
16. `16_SingleForwardNegativeComponent.ipynb`: removes pixel masking entirely.
    A disjoint clean reference estimates Detect-input channel means; sparse
    negative contributions relative to that population baseline are repaired
    using one unmodified forward. With an oracle target reserve, `k=500`
    recovers 23/49 hidden targets and preserves 100/100 clean targets. Fully
    autonomous candidate-cluster ranking plus `k=1000` recovers 22/49, preserves
    25/25 evaluated clean targets, and yields clean full-output `F1=0.953`.
    Global row-space and negative-mode estimators fail, showing that the useful
    signature is coordinate-sparse and sign-selective rather than a simple
    low-rank anomaly.
17. `17_ComponentUniqueness.ipynb`: asks what distinguishes that component
    from ordinary single-image signal. On 49 successful hides, the target
    cluster is characterized by a diffuse negative tail (top-1000
    concentration AUC 0.926), large negative mass, and high functional gain per
    removed L2. Their pre-specified composite reaches clean-vs-hidden
    `AUC=0.962`. A q99 threshold calibrated on 50 clean images triggers on
    27/49 hides and 0/50 held-out clean images; guarded top-1 `k=1000` repair
    recovers 13/49 while preserving clean full-output `F1=1.000`. The q95
    operating point recovers 21/49 with clean `F1=0.994`.
18. `18_DefenseMechanismPresentation.ipynb`: a supervisor-facing, inference-free
    narrative that keeps only the experiments needed to support the main causal
    claim. It starts from the target-specific success definition, follows the
    candidate reserve through score/geometry and local functional closure, and
    ends with the adaptive one-image defense and its improved cluster
    localizer. Lowering the discovery-only score floor exposes target geometry
    in 46/49 cases; noisy-or chooses 39/49 and compact top-20 repair recovers
    38/49. A separate compact q80 anomaly gate retains all 38 repairs at clean
    full-output `F1=0.980` and 100% clean-target preservation. Expanding the
    intervention itself to 100 routes is retained as a negative ablation: it
    dilutes the functional component and recovers only 28/49. Every figure is
    followed by a result interpretation and the reason for the next
    experiment. Rebuild it with
    `build_presentation_notebook.py`; running the notebook only reads compact
    CSV/JSON outputs and is nearly instantaneous.

## Mechanism-aware pixel-patch experiment

`mechanism_aware_patch.py` turns the strongest controlled activation-space
objective into a differentiable pixel-space experiment. It minimizes a smooth
maximum over every current candidate whose decoded geometry can still represent
the fixed clean target. This follows the `07_AttackDirection.ipynb` result that
`dynamic_score_geometry` hid 100/100 targets at substantially lower feature
norm than a fixed-cell objective. Training and evaluation use disjoint cached
images, and evaluation uses the canonical target-aware post-NMS definition.

Run a quick end-to-end check with:

```bash
conda run -n IAD python -m CandidateRoutingAndAttackPath.mechanism_aware_patch --smoke
```

The output directory contains `patch.png`, `training_history.csv`,
`evaluation.csv`, `summary.json`, and a compact `analysis_digest.md`.

### Component-targeted defensive challenge patch

`component_targeted_patch.py` distills the image-specific signed
`joint_rowspace` component from `13_FullSuccessCausalClosure.ipynb` into a
pixel patch. The detector remains frozen. An offline cache stores only local
clean Detect-input values and the sparse joint component; patch training then
aligns the new feature delta with that signed teacher direction. The default
`hybrid` objective retains the dynamic score-plus-geometry loss because the
0.29%-energy joint component was necessary but not sufficient on every
example.

```bash
conda run -n IAD python -m CandidateRoutingAndAttackPath.component_targeted_patch --smoke
```

Available ablations are `dynamic`, `component`, `hybrid`, and `hybrid_null`.

### Defensive mechanism stress suite

`defensive_stress_suite.py` trains a controlled library of mechanism-specific
robustness stressors on one frozen detector, split, initialization, and compute
budget. It includes reserve-wide score change, geometry-only, NMS-only,
candidate handoff, cross-scale desynchronization, nonlinear residual,
component-minimal, compact/diffuse functional tails, dormant geometry,
clean-signature-matched, and naturalistic controls. Outputs are evaluated by
the same target-aware post-NMS metric and summarized in
`mechanism_comparison.csv`.

```bash
conda run -n IAD python -m CandidateRoutingAndAttackPath.defensive_stress_suite --smoke
```

### Single-endpoint component student

`component_student.py` uses the paired-clean joint component only as an offline
teacher. At inference the student consumes detector-internal coordinate
features from one endpoint and predicts a sparse correction. Four feature sets
are compared: activation residual, local context, functional gradient, and
their combination. Clean endpoints are included with a zero target, and
evaluation reports both target recovery and clean full-output preservation.
The current pilot is conditional on the component support stored by the
offline teacher cache. It therefore tests component regression and correction,
not autonomous support localization yet. A deployment version should replace
that oracle support with the existing candidate-cluster localizer. Neither
stage needs to inspect image pixels for a patch-shaped template.

On the 16/16 train/evaluation run, all four students changed target recovery
from 56.25% to 87.5% (5 of 7 initially missed targets recovered, none lost).
Functional and combined features produced the largest mean confidence gain
(about +0.42 versus +0.15 for activation/local). All clean evaluation outputs
were preserved exactly at the detection-set level (F1 1.0). The clean
component prediction itself was not numerically zero, so the observed behavior
is best interpreted as an output-invariant correction rather than a perfect
zero-on-clean classifier.

```bash
conda run -n IAD python -m CandidateRoutingAndAttackPath.component_student --smoke
```

### Large attacked-only component student

`large_component_student.py` is the scalable follow-up. It creates an exact
scene-disjoint split of 200 attacked training endpoints and 150 held-out
scenes. Every held-out scene is evaluated once as attacked and once as clean,
giving 150 attacked plus 150 clean test endpoints. Clean endpoints are never
student-training examples, and normalization statistics are fitted on attacked
training endpoints only.

Two inference modes are evaluated:

- `known_support` receives the teacher support coordinates but only the current
  endpoint values;
- `blind_support` receives only the current image. It obtains target hypotheses
  from detector proposal clusters, builds a functional objective, searches
  internal coordinates, and predicts a correction without teacher coordinates,
  a ground-truth box/class, or a clean reference.

The paired clean endpoint remains necessary offline to construct teacher
targets. It is not passed through the student training-feature pipeline or
either inference path. The teacher cache is resumable at the per-scene `.npz`
level.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n IAD python \
  -m CandidateRoutingAndAttackPath.large_component_student \
  --device mps --require-device
```

Run a short integration check first:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n IAD python \
  -m CandidateRoutingAndAttackPath.large_component_student \
  --smoke --device mps --require-device
```

The default run compares all four feature sets and both support modes. For a
faster first full run, retain the two strongest feature families:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n IAD python \
  -m CandidateRoutingAndAttackPath.large_component_student \
  --device mps --require-device \
  --feature-sets functional,combined
```

The minimal follow-up experiment uses the strongest functional student,
spatial-first blind support, a correction-scale sweep, and two patched-only
diagnostic ceilings:

- `oracle_component`: teacher support and teacher values;
- `blind_oracle_values`: blind support with teacher values on its intersection.

Together with `known_support` and `blind_support`, these conditions separate
support-localization error from value-prediction error. The oracle conditions
are diagnostics, not deployable inference modes.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n IAD python -u \
  -m CandidateRoutingAndAttackPath.large_component_student \
  --device mps --require-device \
  --feature-sets functional \
  --blind-selector spatial \
  --correction-scales 0.25,0.5,0.75,1.0 \
  --diagnostic-ablations
```

After that run, `localization_mechanism_sweep.py` selects the next localization
mechanism without retraining a student. Every candidate support receives the
teacher values on its intersection, so differences measure localization rather
than regression. The compact grid covers coordinate/spatial/hybrid selectors,
2k/4k/8k per-level budgets, top-1/3/5 blind proposal hypotheses, a true-target
functional-objective diagnostic, and the exact teacher-component ceiling.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n IAD python -u \
  -m CandidateRoutingAndAttackPath.localization_mechanism_sweep \
  --base-run /absolute/path/to/large_student_run \
  --device mps --require-device
```

The main outputs are `localization_summary.csv`,
`localization_group_summary.csv`, and an automatically generated
`recommendation.md`.

`localization_ceiling_sweep.py` then tests whether the chosen 4k hybrid support
can raise its oracle-value ceiling without increasing its coordinate budget.
It compares the current averaged top-5 gradient, independent top-5 and top-10
gradients aggregated by coordinate-wise maximum, and a top-10 score/geometry
union. The geometry branch uses separate decoded center and extent VJPs.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n IAD python -u \
  -m CandidateRoutingAndAttackPath.localization_ceiling_sweep \
  --base-run /absolute/path/to/large_student_run \
  --device mps --require-device
```

All conditions use exactly 4k coordinates per level and oracle values on the
selected support, so any recovery difference is attributable to localization.

`learned_cluster_ranker.py` keeps the winning averaged-gradient/hybrid-4k
mechanism and replaces only its heuristic proposal-cluster ordering. On 200
patched training scenes, candidate clusters receive offline labels from the
confidence gain and recovery produced by their local teacher-component
intersection. A histogram gradient boosting ranker then uses only current
endpoint proposal statistics. Evaluation on the disjoint 150 patched scenes
compares heuristic, learned, teacher-energy-oracle, target-oracle, and exact
teacher conditions.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n IAD python -u \
  -m CandidateRoutingAndAttackPath.learned_cluster_ranker \
  --base-run /absolute/path/to/large_student_run \
  --device mps --require-device
```

The oracle conditions are diagnostics. `learned_top5` is the only new
deployable cluster selection in this experiment.

`component_aware_cluster_ranker.py` follows up the weak endpoint-only learned
gain. It reuses the saved blind component student and scores every proposal
cluster with student-component energy, activation anomaly, normalized
functional gradient, and gradient/anomaly leverage. A symmetric within-scene
pairwise model learns teacher-energy ordering on the 200 patched training
scenes. The deployable comparison includes direct student/fused energy,
pairwise top-5, and a set-aware top-5 that trades predicted rank against
marginal spatial coverage. Teacher-energy and teacher-union selectors remain
diagnostic ceilings only.

Method version 2 additionally evaluates three ways to move past the static
top-5 ceiling: detector-level merging of the complementary heuristic and
pairwise branches, fixed-budget versus expanded unions of their candidate
supports, and closed-loop residual localization. The closed-loop condition
applies the first pairwise support, rebuilds proposals from that intermediate
endpoint, and uses a second heuristic or pairwise pass to locate residual
component support. It still uses no target coordinates.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n IAD python -u \
  -m CandidateRoutingAndAttackPath.component_aware_cluster_ranker \
  --base-run /absolute/path/to/large_student_run \
  --ranker-run /absolute/path/to/completed_component_ranker_run \
  --device mps --require-device
```

This run does not retrain the coordinate-level component student. It trains
only the lightweight cluster pairwise model and evaluates all cluster selectors
with the same averaged-gradient/hybrid-4k downstream reconstruction. Omit
`--ranker-run` to train that model; provide a compatible completed run to reuse
its checkpoint and skip the unchanged 200-scene feature pass.

`leak_free_component_defense.py` is the end-to-end follow-up. It does not use
teacher values for test corrections. The previous 200 training scenes are
split by path into 150 student-only and 50 ranker-only scenes, which prevents
the ranker from seeing in-sample student predictions. Evaluation uses the 50
remaining paths outside both the previous 200-train and 150-test splits.
Holdout teacher component records are never loaded: target boxes are passed
only to the metric evaluator. Both stages of the closed loop subtract component
values predicted by the student, and the run evaluates the same mechanism on
patched and clean holdout images.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n IAD python -u \
  -m CandidateRoutingAndAttackPath.leak_free_component_defense \
  --base-run /absolute/path/to/large_student_run \
  --device mps --require-device
```

Each output contains the three split CSVs and `leakage_audit.json`; the audit
records zero path overlap and that no holdout teacher values were loaded or
used.

`prepare_max_balanced_component_pool.py` replaces the earlier 100-per-group
selection with the largest possible strict four-way balance supported by the
current trace. The limiting group has 292 scenes, so the resulting pool has
1168 scenes: 768 student-train (192 per group), 200 ranker-train (50 per
group), and 200 holdout (50 per group). Teacher components are cached only for
the 968 training scenes. The 200 holdout scenes are never teacher-cached.

Build the pool first:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n IAD python -u \
  -m CandidateRoutingAndAttackPath.prepare_max_balanced_component_pool \
  --device mps --require-device
```

The command prints its `max_balanced_<hash>` output directory. Pass that exact
directory to the leak-free experiment:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n IAD python -u \
  -m CandidateRoutingAndAttackPath.leak_free_component_defense \
  --base-run /absolute/path/to/large_student_run \
  --pool-run /absolute/path/to/max_balanced_<hash> \
  --device mps --require-device
```

Pool construction is resumable: rerunning the same command reuses already
written per-scene teacher records. The output `pool_audit.json` records the
group sizes, all three partition sizes, path overlaps, and the absence of
holdout teacher records.

`leak_free_no_iou_expansion.py` targets the remaining localization failures
without using patch coordinates or target geometry at inference. It builds a
larger observable proposal graph at IoU scales 0.30, 0.50, and 0.70, caps it
with four endpoint-only rankings, and trains both a general ranker and a
`hidden_no_iou` specialist on the existing ranker-train partition. The
specialist is applied to every image; the analysis-group label is not an
inference input.

The old 200-scene holdout is treated as already observed validation. Final
metrics are computed on 300 newly sampled scenes (100 `hidden_low_conf`, 100
`hidden_no_iou`, and 100 visible-target scenes) whose paths occur in none of
the previous 1168 pool scenes. No teacher component is built or loaded for
this fresh final set. The primary condition
`expanded_specialist_setaware_top5_8k` is recorded before final evaluation.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n IAD python -u \
  -m CandidateRoutingAndAttackPath.leak_free_no_iou_expansion \
  --prior-run /absolute/path/to/leak_free_593effe4b13ad110 \
  --pool-run /absolute/path/to/max_balanced_b597061f5a82eb58 \
  --device mps --require-device
```

The reconstructed student normalization/scale and expanded ranker feature
table are cached inside the output directory, so restarting the identical
command does not repeat completed preparation stages.

`student_feature_eda.py` performs training-only feature discovery without a
combinatorial downstream sweep. It extracts a resumable per-scene coordinate
table from the 768 student scenes, including the current functional features,
local context, separate class/geometry gradients, multi-hypothesis gradient
consensus, and within-level ranks. It then runs:

- pooled and per-scene univariate support/sign/magnitude screening;
- a Spearman redundancy matrix;
- scene-grouped cross-validated probes;
- drop-one tests for each semantic feature group;
- greedy forward selection with an early stopping threshold.

The ranker split and every previous/fresh holdout are not loaded. Absolute
`x/y` are explicitly reported as a possible fixed-placement shortcut, and the
coordinate-free compact set is always retained as a downstream control.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n IAD python -u \
  -m CandidateRoutingAndAttackPath.student_feature_eda \
  --pool-run /absolute/path/to/max_balanced_b597061f5a82eb58 \
  --device mps --require-device
```

Heavy feature extraction is saved per scene under `scene_tables/`; rerunning
the identical command resumes missing scenes. CPU EDA starts only after the
combined `coordinate_features.parquet` table is complete.

`leak_free_level_ablation.py` is the minimal downstream follow-up to the
feature EDA. It keeps the trained student, its seven selected features, and the
ranker fixed. For each image the student predicts one sparse correction, then
the experiment applies that same prediction as `all_levels`, `levels_1_2`,
`level_2_only`, or diagnostic `level_1_only`. Thus differences isolate where
the attacked-image component must be removed; they do not reflect retraining
or different localization.

The four masks are evaluated on the already observed 200-scene pool holdout.
A safety-constrained rule locks one mask before final evaluation. The final
split contains 300 newly sampled scenes and excludes the entire 1168-scene
pool plus every test/final path recorded by the large-student, leak-free, and
no-IoU experiments. Only `all_levels` and the locked mask are evaluated on
that fresh set. The mechanism processes one image at a time; clean images are
separate safety inputs, never references for patched-image correction.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run --no-capture-output -n IAD python -u \
  -m CandidateRoutingAndAttackPath.leak_free_level_ablation \
  --prior-run /absolute/path/to/leak_free_593effe4b13ad110 \
  --pool-run /absolute/path/to/max_balanced_b597061f5a82eb58 \
  --state-run /absolute/path/to/no_iou_expansion_ebb0720a5fc75739 \
  --eda-run /absolute/path/to/student_feature_eda_dfeaa6a5ea2cc46c \
  --device mps --require-device
```

`ResearchPath.ipynb` is the unnumbered, inference-free reading path across all
18 notebooks. It preserves the smallest complete chain of decisive
experiments, including the negative results that motivated every transition,
and links back to each detailed notebook. Rebuild it with
`build_research_path_notebook.py`.

`Method_Mathematical.md` is the complete mathematical specification of the
mechanistic result and the resulting one-image defense: target/reserve
definitions, path-integrated Jacobian, SVD row-space projection,
repair/transplant causality, nonlinear synergy, observable surrogate, adaptive
intervention, gate, results, and claim boundaries.

Compact outputs are written under `followup_outputs/`. Runner modules include
`mechanism_followup.py`, `attack_direction.py`, `defense_direction.py`,
`candidate_reserve.py`, `shared_candidate_mechanism.py`,
`score_functional_subspace.py`, `full_success_closure.py`,
`cluster_localization_diagnostic.py`, `cluster_ranker_analysis.py`, and
`localization_improvement_analysis.py`.
Expanded runs are launched from the repository root:

```bash
conda run -n IAD python -m CandidateRoutingAndAttackPath.run_followup_experiments mechanism --device mps
conda run -n IAD python -m CandidateRoutingAndAttackPath.run_followup_experiments attack --device mps
conda run -n IAD python -m CandidateRoutingAndAttackPath.run_followup_experiments defense
conda run -n IAD python -m CandidateRoutingAndAttackPath.run_followup_experiments reserve --device mps
conda run -n IAD python -m CandidateRoutingAndAttackPath.run_followup_experiments shared --device mps
conda run -n IAD python -m CandidateRoutingAndAttackPath.run_followup_experiments score_subspace --device mps
conda run -n IAD python -m CandidateRoutingAndAttackPath.run_followup_experiments full_success --device cpu
conda run -n IAD python -m CandidateRoutingAndAttackPath.run_followup_experiments full_success_expanded --device mps
conda run -n IAD python -m CandidateRoutingAndAttackPath.run_followup_experiments self_counterfactual --device mps
conda run -n IAD python -m CandidateRoutingAndAttackPath.run_followup_experiments self_counterfactual_weak --device mps
conda run -n IAD python -m CandidateRoutingAndAttackPath.run_followup_experiments self_counterfactual_blind --device mps
conda run -n IAD python -m CandidateRoutingAndAttackPath.run_followup_experiments single_forward_component --device mps
conda run -n IAD python -m CandidateRoutingAndAttackPath.run_followup_experiments autonomous_negative_repair --device mps
```

`full_success_expanded` leaves the original run intact and repeats the expensive
joint score+geometry functional intervention on all 400 balanced examples
(100 per analysis group), rather than the original 100-example functional
subset. The experiment still writes only compact CSV/JSON outputs and enforces
the 1 GiB output budget.

Every notebook begins with previously rejected or impractical directions and
the numerical reason they were rejected; after a full run, the result-specific
Markdown interpretation should be updated rather than deleting negative
evidence.

`AllMetrics.ipynb` is now a small navigation page. The executed monolithic
version is preserved as `AllMetrics_legacy.ipynb` only for historical outputs.
