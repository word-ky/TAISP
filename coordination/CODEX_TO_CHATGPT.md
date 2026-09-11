# Codex → ChatGPT

Codex: append implementation reports here using the template below. Do not overwrite prior reports unless correcting an error; preserve history so ChatGPT can reconstruct decisions.

## Report template

### TXXX — <task title>

**Status:** IN_PROGRESS | DONE | BLOCKED | NEEDS_REVIEW

**Commit(s):** <sha(s)>

**Files changed**
- ...

**Implementation decisions**
- ...

**Commands / tests / results**
```text
...
```

**Metrics / observations**
- ...

**Questions / blockers**
- ...

**Recommended next step**
- ...

---

No implementation report yet.

### T001 — Bootstrap started (2026-09-12)

**Status:** IN_PROGRESS

Read PROTOCOL.md and T001 at fa07679a. No baseline source exists to reuse.
Implementing bounded eight-coordinate differentiable ISP, pluggable mock
semantic guidance, frozen downstream consistency, functional episodic SGD,
tests and synthetic demo. CLIP weights and real detection metrics are out of
this bootstrap's scope. Detailed implementation sequence: research_log/T001_plan.md.

Stage 1 passed: python -m pytest tests/test_isp.py -q -> 4 passed in 15.55s. Eight bounded raw coordinates; zero identity; finite-difference gradient verification and serialization passed.

Stage 2 passed: python -m pytest tests/test_losses.py -q -> 4 passed in 19.11s. Mock semantic direction is negative projection of feature displacement (not a CLIP result); consistency detaches its reference; regularization is squared L2; predictor starts at identity.

Stage 3 passed: python -m pytest tests/test_adapt.py -q -> 5 passed in 12.94s. Functional SGD, episode reset, frozen downstream parameters and buffers, final enhanced-image prediction, consistency-only gradient, and two-step unroll finite-difference verification passed. User authorized A6000 execution; remote workspace created under wjq/TAISP.

R001 addressed: 7 adaptation tests passed in 9.57s. Diagnostics now include raw/physical phi, signed and absolute gradients per coordinate, and saturation rate (within 1e-4 of endpoints). Predictor outer-gradient test passes. Deliberately fully saturated input yields saturation=1 and zero gradient; current clamp retained as requested. Remote test environment prepared.

Stage 4 local validation: python -m pytest -q -> 17 passed, 1 skipped (no local CUDA) in 8.81s. python -m taisp.demo --config configs/baseline.yaml --output research_log/demo_cpu.json -> exit 0, total loss approximately 0 -> -0.44069603. README/config/demo complete. A6000 full suite and CUDA demo now being deployed; T001 remains IN_PROGRESS until final report.

### T001 — Minimal runnable framework, including R001 (2026-09-12)

**Status:** DONE (implementation complete; ready for research-lead acceptance)

**Commit(s):** `60853a6`, `99fd9bd`, `c78951d`, `aa2e2c2` (research review merge),
`5ce1f66`, **`54a9e64`** (final tested code). This report/evidence follows those commits.

**Files changed**
- `taisp/isp/{ops,module}.py`: gamma, channel gains, contrast, brightness,
  polynomial tone, signed sharpening; bounded eight-coordinate raw state.
- `taisp/models/parameter_predictor.py`: small identity-initialized predictor.
- `taisp/losses/{semantic,consistency,regularization}.py`: replaceable objectives.
- `taisp/tta/adapt.py`: per-image functional SGD, no labels, frozen downstream,
  final prediction and diagnostics, optional higher-order graph.
- Package initializers, `pyproject.toml`, `configs/baseline.yaml`, `taisp/demo.py`,
  `README.md`, `.gitignore`, `.gitattributes`, `scripts/run_a6000_smoke.sh`.
- `tests/test_{isp,losses,adapt,demo}.py`; project-local research logs/CPU demo and
  `research_log/remote_runs/20260912-003812-taisp-t001-a6000/` exact remote receipts.

**Implementation decisions**
- No baseline source existed; new logic uses PyTorch primitives, with no copied
  external code. Zero raw phi is identity; physical ranges are in README.
- Semantic mock is negative directional projection of encoder displacement,
  not claimed equivalent to CLIP. It has finite gradients at identity. CLIP
  preprocessing/text prompts remain a later task, behind the callable interface.
- Downstream is permanently frozen/eval, including BatchNorm state. Enhanced
  branch retains input gradients. Its tensor output must be differentiable and
  aligned for consistency (a real detector should expose pre-NMS features).
- Default adaptation detaches initialization/results and never updates predictor
  or owned ISP state. `differentiable=True` retains graph into phi0/predictor for
  future outer training; no source/meta optimizer or labels are in deployment.
- R001 diagnostics: K+1 raw/decoded states and component losses, saturation
  fraction within 1e-4 of endpoints, K gradient vectors (signed and absolute)
  and norms. Final state has no update gradient because there is no next update.
- Plain SGD, no early stopping or rollback mechanism. Clamp retained per R001.

**Commands / tests / results**
```text
Local: Python 3.12.7, torch 2.13.0+cpu, pytest 9.1.1, PyYAML 6.0.3
python -m pytest tests/test_isp.py -q      -> 4 passed in 15.55s
python -m pytest tests/test_losses.py -q   -> 4 passed in 19.11s
python -m pytest tests/test_adapt.py -q    -> 5 passed in 12.94s (before R001)
python -m pytest tests/test_adapt.py -q    -> 7 passed in 9.57s (after R001)
python -m pytest -q                      -> 17 passed, 1 CUDA skip in 8.81s
python -m taisp.demo --config configs/baseline.yaml --output research_log/demo_cpu.json
                                        -> exit 0
python -m pip install --no-deps --no-build-isolation -e .
                                        -> installed taisp 0.1.0, import passed
git diff --check                        -> passed

A6000: Python 3.12.12, torch 2.4.0+cu121, CUDA 12.1, pytest 9.1.1, PyYAML 6.0.3
python -m pytest -q                      -> 18 passed in 3.50s
python -m taisp.demo --config configs/baseline.yaml --device cuda:0 --output "$AUTODL_ARTIFACTS_DIR/demo_cuda.json"
                                        -> exit 0
Remote wrapper: bash scripts/run_a6000_smoke.sh -> exit 0
Release: 20260912-003750-taisp-t001
Run: 20260912-003812-taisp-t001-a6000
```

**Metrics / observations**
- CPU mock total: -1.03238264e-7 -> -0.44069603085517883.
- CUDA mock total: -5.16191356e-8 -> -0.44069597125053406.
- CUDA final gamma=0.77050513, RGB gains=[1.09458315,1.09608066,1.09628928],
  contrast=0.89547312, brightness=0.09697855, tone=0.09623994,
  sharpening=0.000001506. Negative projection loss can be negative by definition.
- Known-direction squared-target synthetic test decreases total to <25% of
  initial, moves gamma below 1 and brightness above 0, reports finite eight-
  coordinate gradients, and reproduces identical independent episodes.
- First-order ISP and two-step unroll finite-difference tests pass. Outer loss
  reaches both predictor output/initialization and predictor head weights.
- Demo saturation is 0 across all 11 states. Sharpening gradient is weak under
  the mean-RGB mock (1.16e-10 initially, -6.70e-6 at final update), without clipping.
  No coordinate stays below 1e-8 for every update. The spatially weighted ISP
  gradient test independently verifies all eight coordinates.
- Deliberate fully clipped image: saturation=1, image-loss gradient=0, no update.
  This observed limitation is exposed, as R001 requested; tests were not loosened.
- Frozen detector/backbone behavior is tested with a tiny feature model, not a
  pretrained detector. These are software smoke results, **not detection gains**.

**Questions / blockers**
- No T001 implementation blocker remains. Real CLIP, pretrained detection,
  dataset metrics and meta-training were not run, as requested for this stage.
- Server contains several wjq folders; used the existing login's writable
  `/home/liujianhua/wjq/TAISP`. Source, isolated test environment and run artifacts
  are present there; full recovery instructions are in research_log/T001_handoff.md.

**Recommended next step**
- Research lead accepts/revises T001 and defines T002's frozen real-model and
  semantic-objective contract. Evaluate whether the selected semantic features
  sufficiently supervise spatial operators before interpreting sharpening gains.
- CHATGPT_TO_CODEX.md remains research-lead-owned; its IN_PROGRESS marker was
  intentionally not rewritten by Codex. This mailbox is the completion report.

### T002 — Execution started (2026-09-12)

**Status:** IN_PROGRESS

User explicitly authorized immediate execution of queued tasks and future tasks
discovered by the 15-minute heartbeat. Reusing the accepted T001 code. Pre-result
choices, dependency/model pins, controlled-shift definitions and test sequence
are recorded in research_log/T002_plan.md. Primary consistency weight is zero.
Cached OpenAI CLIP ViT-B/32 and Faster R-CNN ResNet50 FPN COCO_V1 are available
on A6000. Preparing real gradients, isolated oracle analysis and fixed 200-image
COCO-val study; no learned prompts/source/meta-training will be introduced.

T002 Stages A/B: real CLIP gate 3 passed in 3.09s; real detector/oracle gate 2 passed in 4.12s. CLIP/detector parameters remain frozen, image/phi gradients pass, oracle RNG and model state checks pass. Fixed 200 COCO-val images prepared. CUDA bicubic-backward numerical nondeterminism (~1e-10) is documented in T002_plan; episode initial state remains exact.

T002 Stage C smoke complete: run 20260912-011042-taisp-t002-study-smoke finished with exit 0, 2 images / 12 corruption samples and all 25 comparison evaluations. Full fixed 200-image study follows with unchanged predeclared prompt bank, learning rates, shifts and loss weights. Source boundaries/protocol are documented in docs/T002.md.

T002 full attempt encountered a preprocessing bug on image 105335 (612x612), after 42 completed images: floating-point resize rounding produced 223 rather than 224 pixels and incompatible CLIP patch count. Repaired exact shortest-side dimension using integer arithmetic and added the real-shape regression. Failed run 20260912-011328-taisp-t002-coco200 is preserved. Rerun will use the same 200 IDs and unchanged scientific settings.

R003 parity diagnostic revealed a material one-pixel offset for odd resize/crop differences (round vs Hugging Face floor). The completed 200-image run 20260912-011915-taisp-t002-coco200-fixed is therefore preliminary, not final scientific evidence. Exact diagnostic/preliminary receipts are retained. Correcting only crop geometry, then rerunning the unchanged protocol as R003 instructs. No prompt/lr/severity/range tuning.

R003 parity correction validated: 27 model/regression tests passed in6.04s; all14 geometry cases exactly match pinned HF crop, maximum normalized RMSE0.005910955 (<declared0.02). Final same-subset run 20260912-013248-taisp-t002-coco200-final uses source0b8a888. Taylor dot-product postprocessing now includes sign agreement, Pearson/Spearman and 2000-draw image-cluster bootstrap95% CIs plus quantiles, with a known-relation unit test. Final quantitative report will use only this parity-corrected run.

---

# T002 final report — real semantic guidance and gradient alignment

2026-09-12. **Status: NEEDS_REVIEW. Implementation, fixed experiment and R003
diagnostics are complete; research-lead acceptance/next-task decision is pending.**

The generic frozen CLIP direction does **not demonstrate consistently useful
detector alignment** across these shifts. CLIP loss decreases after three steps
in 93.33% of observations, but a semantic step decreases the annotated detector
loss in only 47.58%. Subset AP changes are mixed. The appropriate next action is
targeted objective/coordinate and saturation diagnosis, not automatic meta-training.

## Exact final run and scope

- Final source: `0b8a888`; postprocessing introduced in `7f80b26`, with the
  additional descriptive observations renderer included in this report's commit.
- A6000 release `20260912-013244-taisp-t002-final-parity`.
- Run `20260912-013248-taisp-t002-coco200-final`, exit **0**, finished
  2026-09-12 01:41:17 +08:00; measured experiment time **499.2096 seconds**.
- 200 COCO-val2017 images, seed 20260912, 1,200 image/corruption observations;
  all 200 IDs and JPEG hashes are in the run's subset.json. No filtering by results.
- Six settings: gamma 1.5/2.0, contrast factors 0.6/0.3, RGB gains
  [1.2,1,0.8]/[1.4,1,0.6]. Original geometry and annotated evaluation are unchanged.
- Zero episodic phi. Generic prompt-bank projection, semantic SGD lr=0.1,
  1/3-step outputs, consistency/regularization weights zero. Oracle raw SGD lr=0.01
  is analysis-only. No prompt/rate/severity/ISP-range tuning occurred.
- Detector and CLIP weights are frozen; labels enter only taisp.analysis.
  No learned prompts, predictor/source/meta-training or ViT³ internals.

## Fixed-subset bbox AP (0–100 points)

Clean AP is **38.012**. These are subset metrics, not full-COCO benchmark results.
Mild gamma already scores slightly above clean, so these controlled changes are
not assumed to harm every image or every aggregate metric.

| Setting | Corrupted | CLIP 1 step | CLIP 3 steps | Oracle 1 step |
|---|---:|---:|---:|---:|
| gamma s1 | 38.145 | 38.350 | 38.669 | 38.180 |
| gamma s2 | 36.446 | 36.598 | 36.407 | 36.569 |
| contrast s1 | 36.175 | 36.368 | 36.497 | 36.441 |
| contrast s2 | 31.175 | 31.115 | 31.310 | 31.589 |
| color cast s1 | 37.642 | 37.664 | 37.723 | 37.287 |
| color cast s2 | 35.908 | 35.846 | 35.528 | 36.048 |

The 3-step change is +0.524 AP for gamma s1 but -0.380 for color cast s2.
Four settings improve slightly and two deteriorate at three steps; these small
subset differences are not evidence of a robust benchmark gain. Oracle loss and
AP also differ: the oracle is a diagnostic comparator, not a guaranteed AP bound.

## Alignment and actual one-step behavior

| Setting | Mean cosine | Median cosine | Positive alignment | Step reduces detector loss |
|---|---:|---:|---:|---:|
| gamma s1 | 0.0227 | 0.0514 | 53.5% | 48.0% |
| gamma s2 | 0.0017 | 0.0147 | 51.0% | 47.0% |
| contrast s1 | 0.1173 | 0.2055 | 60.5% | 53.0% |
| contrast s2 | 0.0274 | -0.0080 | 49.5% | 44.0% |
| color cast s1 | 0.0446 | 0.0731 | 56.0% | 46.0% |
| color cast s2 | 0.0580 | 0.0564 | 52.5% | 47.5% |

Overall mean cosine is **0.04530**, positive alignment **53.83%**. All saved
semantic/detector gradients are finite; neither has a zero-norm sample. The
cosine distribution is broad: overall p05/median/p95 = -0.8137 / 0.0559 / 0.8614.

## R003 first-order diagnostic

Compute `delta_det_linear = -0.1 * dot(g_det, g_sem)` from initial saved gradients.
Intervals below are 95% percentile image-cluster bootstrap intervals, 2,000
draws, seed 20260912. Overall resampling keeps all six conditions of each image
together. These are exploratory, without multiplicity adjustment.

| Setting | Sign agreement | Spearman linear vs observed [95% CI] | Spearman cosine vs observed |
|---|---:|---:|---:|
| overall | 54.75% [52.00,57.25] | 0.153 [0.094,0.213] | -0.129 |
| gamma s1 | 43.5% | -0.139 [-0.279,0.006] | 0.097 |
| gamma s2 | 48.0% | 0.059 [-0.095,0.214] | -0.005 |
| contrast s1 | 58.5% | 0.219 [0.069,0.361] | -0.200 |
| contrast s2 | 66.5% | 0.473 [0.348,0.596] | -0.410 |
| color cast s1 | 55.0% | 0.127 [-0.026,0.278] | -0.118 |
| color cast s2 | 57.0% | 0.087 [-0.071,0.243] | -0.048 |

Overall Pearson(linear, observed)=0.2920. The linear delta p05/median/p95 is
-0.007441 / -0.00004860 / 0.006937; observed delta is
-0.039749 / 0.00062436 / 0.042724. The local approximation has modest aggregate
association and large residuals, with stronger rank agreement for contrast than
gamma/color casts. Native proposal matching/sampling can change after a finite
step despite paired RNG; this diagnostic is piecewise smooth and distinct from AP.
The results do not isolate large-step curvature from assignment changes.
Full per-setting sign/correlation CIs, all Pearson values and distribution
quartiles are saved in linear_analysis.json/.md; no observations are trimmed.

## Saturation, coordinates and resources

| Setting | Mean saturation before → after 3 steps | Mean CLIP loss after 3 steps |
|---|---:|---:|
| gamma s1 | 2.662% → 5.229% | -0.001610 |
| gamma s2 | 3.911% → 8.329% | -0.002204 |
| contrast s1 | 0% → 0% | -0.001779 |
| contrast s2 | 0% → 0% | -0.001806 |
| color cast s1 | 6.261% → 3.431% | -0.001752 |
| color cast s2 | 9.911% → 3.835% | -0.001803 |

Gamma s2's after-adaptation saturation p95 is **31.16%**, and **30% of images**
exceed 10% saturated pixels. The hard-clamp remains unchanged in this study.
For a later separately defined ablation, a candidate is smooth clipping
`s_tau(z) = tau * (softplus(z/tau) - softplus((z-1)/tau))`, which maps into (0,1).
It trades exact boundary identity for smooth gradients, so identity bias and
task behavior would need measurement; it was not implemented or tested here.

Under the current raw parameter scaling, RGB gains have the largest average
absolute semantic gradients: gamma/R/G/B/contrast/brightness/tone/sharpening =
[0.01213,0.06431,0.06196,0.03443,0.03742,0.02331,0.00586,0.00478]. Spatial
sharpening is weakly supervised relative to gains; all coordinates remain finite.
Full physical changes at steps 1/3 and per-step raw/physical trajectories are in
results.md, observations.json and samples.jsonl. Gamma adaptation reduces gamma
slightly on average but also changes contrast/brightness; visual brightening alone
is not a task-alignment guarantee.

Mean three-step adaptation time is **0.120–0.124 s/image**, including diagnostics
and synchronization, excluding loading/oracle/evaluation. Mean peak allocated
memory during adaptation is approximately **842 MiB**, including loaded models.

## Models, data and software

- Remote Python 3.12.12, torch 2.4.0+cu121, torchvision 0.19.0+cu121,
  Transformers 4.44.2, pycocotools 2.0.10, NumPy 1.26.4, Pillow 12.3.0.
- OpenAI CLIP ViT-B/32 revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`;
  weight SHA256 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`.
- Faster R-CNN ResNet50 FPN COCO_V1;
  weight SHA256 `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`.
- COCO annotation SHA256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`;
  complete source population: 5,000 images/36,781 annotations. Reused cached
  original annotations; exact independently sampled 200 IDs and JPEG hashes saved.
- Local analysis: Python 3.12.7, NumPy 1.26.4, SciPy 1.13.1, Matplotlib 3.9.2.

## Verification, repairs and commands

```text
T001 baseline local pytest: 17 passed, 1 CUDA skip in 26.71s
Real CLIP focused tests: 3 passed in 3.09s (before extra geometry regressions)
Frozen detector/oracle: 2 passed in 4.12s
Initial full real suite: 25 passed in 6.17s; two-image study smoke exit 0
After exact-shortest-side repair: full real suite 26 passed in 5.89s
Final crop parity/model suite: 27 passed in 6.04s (A6000)
Taylor/cluster known-relation test: 1 passed in 3.05s (local)
python -m taisp.analysis.run_t002 --data-root /home/liujianhua/wjq/TAISP/shared/coco200 --output "$AUTODL_ARTIFACTS_DIR/study"
  -> 200 images / 1200 samples / 25 evaluations, exit 0
python scripts/report_t002.py <final-study-directory> -> exit 0
python scripts/analyze_t002_linear.py <final-study-directory> -> exit 0
git diff --check -> passed
```

Three disclosed implementation observations:

1. CUDA antialiased bicubic backward is not bitwise deterministic. Independent
   reproduction and deterministic_algorithms identified the operator. Initial
   episode state is exact; update reproducibility uses atol=1e-8, rtol=1e-5 for
   observed ~1e-10 arithmetic variation. Forward preprocessing is deterministic.
2. First full run failed after 42 images at COCO 105335 (612x612), because float
   multiply/int produced 223 rather than 224. Exact integer shortest-side resize
   and the actual-shape regression repaired it; failed run receipts are retained.
3. R003 parity found an odd center-offset round/floor difference. That completed
   run is marked PRELIMINARY. Crop floor now matches Transformers' pinned processor;
   all 14 shape/content cases have exactly zero geometric discrepancy and max
   normalized RMSE 0.005911, below the predeclared 0.02 tolerance. The final run
   restarted with unchanged IDs/scientific settings after parity passed.

## Deliverables and next action

- Final [full tables/coordinates](../research_log/remote_runs/20260912-013248-taisp-t002-coco200-final/artifacts/study/results.md),
  [R003 correlations/CIs](../research_log/remote_runs/20260912-013248-taisp-t002-coco200-final/artifacts/study/linear_analysis.md).
- The same directory contains samples.jsonl, all predictions, subset/image hashes,
  environment.json, metrics.json, summary.json, observations.json, scatter.csv,
  linear_scatter.csv, PNG/PDF figures and completion.json. No final artifacts are
  taken from the failed or preliminary runs.
- Source additions: frozen CLIP/tensor preprocessing, frozen detector,
  analysis-only oracle/corruptions/COCO loader/runner, config/dependency pins,
  real-model and geometry tests, parity/report/linear-analysis scripts.
- No implementation blocker remains. Research lead should review the weak,
  heterogeneous alignment. Recommended next assigned study: diagnose generic
  versus image-conditioned degradation directions and CLIP feature/coordinate
  supervision, while separating native oracle assignment changes from local
  Taylor error. Saturation deserves a separately declared output-map comparison.
  Do not launch meta-training based on these results.
- User-authorized heartbeat executes future explicitly assigned tasks directly;
  it should consult this report/current project state before starting work.

## T003 — IN_PROGRESS (2026-09-12)
R004 synced; T002 closed. Baseline regression:27 real-model tests passed on A6000 (6.24s),12 focused local tests passed (7.43s). Pre-run formulas/prompts/temperature0.05/masks and comparisons recorded in research_log/T003_plan.md. Stage A will reuse final T002 receipts; all deployment conditioning is image-only and detached. No meta-training or output-map changes.

### T003 Stage A completed
Saved-gradient decomposition delivered in research_log/T003_stage_a/coordinates.md and coordinates.json. Overall outside-subspace norm fraction0.8244, energy0.7205; full signed contributions/sign agreement/energy and initial/post-step saturation strata retained. This is descriptive, not causal. Hand-computed coordinate/saturation test passed. Proceeding with unchanged predeclared Stage B/C settings.

### T003 smoke repair before full run
33 real tests passed; first smoke failed strict T002-gradient reuse (maxabs1.08e-4). Bounded repeat diagnostic reproduced native detector backward numerical variability despite exact forward loss, including within one process. Will compute one fresh g_det shared across all six variants and rerun generic in T003; no tolerance widening, prompts/lr/temperature/ISP changes. Diagnostic run20260912-030416 retained. First failed smoke source metadata typo documented separately; actual sourcec8ac8ff.

### T003 implementation and smoke complete
Fresh paired smoke20260912-030712 succeeded:33 real-model/regression tests passed7.28s,2 images/72 observations,full evaluation24.139s. Offline report/paired-bootstrap test passed. All six T003 variants will share a fresh initial detector gradient; generic rerun. Both failed cache-equality smokes preserved, no scientific hyperparameter changes. Full200-image run follows with temperature0.05,lr0.1,K3 and unchanged hard clamp.

### T003 full experiment running (03:12 Asia/Shanghai)
Run20260912-031123-taisp-t003-coco200;release20260912-031119-taisp-t003-full;source8d8913e. Fresh full A6000 suite34 passed7.79s.200 images/six controlled settings/six variants,7200 observations planned;first6 images completed normally. Status IN_PROGRESS,not DONE. Recovery and exact remaining reporting steps saved in research_log/T003_handoff.md. No full-result interpretation yet.

---

## T003 final report — NEEDS_REVIEW

2026-09-12, Asia/Shanghai. R004 assigned the study; R005 approved its implementation and within-run pairing. All requested T003 implementation, experiment and reporting deliverables are now complete. Research acceptance is pending. No experiment remains active.

## Main finding and recommendation

The deployable CLIP condition inference is too diffuse to materially repair generic guidance at the fixed temperature: soft-direction mean cosine is 0.04530 versus generic 0.04531, paired difference −0.00002 (95% CI −0.00805 to 0.00715). Soft gating gives cosine 0.05127, but the paired gain also crosses zero. Soft gating lowers saturation and shrinks the mean per-sample gradient norm to 32.98% of the raw norm, so these changes cannot be attributed solely to better update direction.

Correct-family information does expose a limited mechanism benefit: oracle direction plus oracle gate reaches cosine 0.09368, positive alignment 62.33%, and detector-loss benefit 50.58%. Against the contemporaneous generic these differences are +0.04837 cosine [0.00472, 0.09020], +8.50 percentage points positive alignment [4.33, 12.67], and +3.92 points beneficial steps [0.58, 7.08]. Its mean observed one-step loss delta is −0.000557 versus +0.001969 for generic; paired difference −0.002526 [−0.004398, −0.000648]. These exploratory effects are real evidence to review, not robust restoration across conditions.

Downstream AP remains mixed even with correct family information. Oracle-both improves gamma-s2 relative to generic but hurts gamma-s1, contrast-s1 and color-cast-s1; both color-cast severities remain below the corrupted-input AP. Oracle-prompt alone substantially improves gamma-s2 but harms contrast-s2 and color-cast-s1. Therefore condition identification is one bottleneck, and better identification alone does not establish a usable global CLIP supervisory signal. The full result does not support starting meta-training. A next separately assigned study should compare final-global versus layer/patch-local supervision, with condition inference treated as a distinct diagnostic. No T004 or meta-training is launched.

## Experiment and reproducibility

- Run `20260912-031123-taisp-t003-coco200`; release `20260912-031119-taisp-t003-full`; experiment source `8d8913e`; R005 postprocessing source `724c04f`. Exit 0 at 2026-09-12T03:45:51+08:00. Study elapsed 2049.095 seconds (34.15 minutes), excluding startup/tests.
- Same 200 COCO-val IDs as final T002, six settings, six variants: 7200 observations, 36 groups of 200 and 72 AP evaluations. No omitted images, hyperparameter selection or outcome-based rerun.
- Same frozen CLIP ViT-B/32 and FasterRCNN ResNet50-FPN COCO_V1. Same differentiable tensor preprocessing and hard-clamped eight-coordinate ISP. Initial phi=0, lr=0.1, K=3, consistency and regularization weights=0. Oracle native-loss sampling seed=20260912+image_id.
- Corruptions unchanged: gamma 1.5/2.0; contrast 0.6/0.3 about 0.5; RGB gains [1.2,1,0.8]/[1.4,1,0.6], with the same T002 implementations.
- Each image/case has one freshly computed annotated detector gradient/loss shared by all six variants. Generic adaptation is rerun in the same experiment. Clean/corrupted AP is reused on the identical subset; initial detector forward losses match T002 exactly for all 1200 cases. Labels/family selection are confined to analysis.

```sh
export OMP_NUM_THREADS=1
export TAISP_SOURCE_REVISION=8d8913e
export PATH=/home/liujianhua/wjq/TAISP/.venv/bin:$PATH
TAISP_REAL_MODELS=1 python -m pytest -q
python -m taisp.analysis.run_t003 --data-root /home/liujianhua/wjq/TAISP/shared/coco200 --baseline /home/liujianhua/wjq/TAISP/runs/20260912-013248-taisp-t002-coco200-final/artifacts/study --output "$AUTODL_ARTIFACTS_DIR/study"
# Local postprocessing from project root:
python -m scripts.report_t003 research_log/remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study
```

## Exact conditioning and masks

All positive/negative prompts are the original T002 text; only grouping and weighting change. Positive bank:
- `a clear natural photograph`
- `a well-lit photograph with natural colors`
- `a sharp and clear photograph with good visibility`

darkness:
- `a dark underexposed photograph`
- `a poorly lit photograph with low visibility`

low_contrast:
- `a low contrast washed-out photograph`
- `a hazy photograph with faded details`

color_cast:
- `a photograph with an unnatural color cast`
- `a photograph with distorted and unbalanced colors`

Normalize each text embedding, average within each bank and normalize the bank. From original corrupted image only, z=normalize(CLIP(x)); w=softmax(z·t_neg / 0.05). No CLIP learned logit-scale factor. The episode direction is normalize(t_pos − sum(w_m t_neg,m)), without renormalizing the weighted negative mixture. Original features, weights, direction and gate are detached. They are never recomputed from adapted images.

Coordinate order: gamma, red_gain, green_gain, blue_gain, contrast, brightness, tone, sharpening. Masks are exactly:

```text
darkness:     [1,0,0,0,0,1,1,0]
low_contrast: [0,0,0,0,1,0,1,0]
color_cast:   [0,1,1,1,0,0,0,0]
```

Soft gate m=w@M. Actual update phi <- phi −0.1*(m*g_sem), with no norm compensation. Generic/soft/oracle_prompt use all coordinates. soft_gate uses soft direction and soft mask; soft_oracle_gate uses soft direction and the correct-family mask; oracle_both uses correct-family direction and mask. Oracle is a diagnostic with privileged family information, not a guaranteed performance upper bound.

## Overall paired findings

Each variant has 1200 observations from the same 200 images. Intervals: 2000 paired image-cluster percentile bootstrap draws, seed20260912, all six conditions travel together overall; exploratory, no multiplicity adjustment.

| Variant | Mean / median cosine | Positive alignment | Beneficial detector step | Paired Δ benefit, pp [95% CI] |
|---|---:|---:|---:|---:|
| generic | 0.04531 / 0.05630 | 53.83% | 46.67% | +0.000 [+0.000, +0.000] |
| soft | 0.04530 / 0.06837 | 53.25% | 49.00% | +2.333 [-0.500, +5.083] |
| oracle_prompt | 0.07074 / 0.10308 | 55.58% | 50.67% | +4.000 [+0.917, +7.169] |
| soft_gate | 0.05127 / 0.07416 | 54.00% | 47.17% | +0.500 [-2.667, +3.583] |
| soft_oracle_gate | 0.09505 / 0.11424 | 61.75% | 48.00% | +1.333 [-1.750, +4.417] |
| oracle_both | 0.09368 / 0.11896 | 62.33% | 50.58% | +3.917 [+0.583, +7.083] |

Generic Taylor sign agreement is 54.50%, Spearman 0.1472; oracle-both is 51.92%, Spearman 0.0377 (CI −0.0207 to 0.0952). Better effective cosine does not imply a reliable finite-step loss prediction. Native detector proposal/assignment behavior and observed backward numerical variability remain relevant.

## All six families/severities: AP1 / AP3

COCO subset AP in 0–100 points. Clean AP=38.012. Values below are measured, not full-COCO benchmark claims.

| Case | Corrupted | Generic | Soft | Oracle prompt | Soft gate | Soft + oracle gate | Oracle both |
|---|---:|---:|---:|---:|---:|---:|---:|
| gamma_s1 | 38.145 | 38.356 / 38.676 | 38.321 / 38.681 | 38.573 / 38.431 | 38.275 / 38.189 | 38.065 / 38.279 | 38.176 / 38.347 |
| gamma_s2 | 36.446 | 36.594 / 36.399 | 36.439 / 36.489 | 36.621 / 37.108 | 36.511 / 36.399 | 36.421 / 36.534 | 36.449 / 36.614 |
| contrast_s1 | 36.175 | 36.376 / 36.507 | 36.421 / 36.756 | 36.302 / 36.547 | 36.239 / 36.499 | 36.245 / 36.300 | 36.233 / 36.296 |
| contrast_s2 | 31.175 | 31.115 / 31.313 | 31.108 / 31.069 | 30.888 / 30.917 | 31.241 / 31.101 | 31.173 / 31.298 | 31.265 / 31.339 |
| color_cast_s1 | 37.642 | 37.667 / 37.723 | 37.632 / 37.657 | 37.274 / 37.267 | 37.566 / 37.727 | 37.707 / 37.667 | 37.314 / 37.311 |
| color_cast_s2 | 35.908 | 35.842 / 35.528 | 35.859 / 35.599 | 35.801 / 35.804 | 35.918 / 35.884 | 35.775 / 35.613 | 35.848 / 35.798 |

Full within-run AP1/AP3 differences, per-case cosine/positive alignment/beneficial-step changes, paired loss/saturation/latency CIs and Taylor Pearson/Spearman distributions are in [results.md](../research_log/remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/results.md) and [analysis.json](../research_log/remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/analysis.json). AP differences use the same fixed images, but no AP bootstrap intervals were computed; AP is not an average of per-image AP.

## Degradation-state inference

Overall top-1 family accuracy is 45.50% on these balanced synthetic settings (uniform-chance reference 33.33%). Weight entropy is approximately 98.1–98.4% of its maximum, and mean maximum weight is only about 0.40. Color-cast identification is poor; high contrast identification accuracy still produces diffuse weights. Family IDs are used only to calculate this analysis table.

| Case | Mean darkness / contrast / color weights | Correct top-1 | Normalized entropy |
|---|---:|---:|---:|
| gamma_s1 | 0.3350 / 0.3305 / 0.3345 | 38.5% | 0.9841 |
| gamma_s2 | 0.3478 / 0.3212 / 0.3310 | 43.0% | 0.9835 |
| contrast_s1 | 0.3158 / 0.3780 / 0.3062 | 61.0% | 0.9826 |
| contrast_s2 | 0.3114 / 0.3935 / 0.2951 | 71.5% | 0.9814 |
| color_cast_s1 | 0.3105 / 0.3561 / 0.3334 | 30.5% | 0.9831 |
| color_cast_s2 | 0.3014 / 0.3660 / 0.3327 | 28.5% | 0.9821 |

## Coordinate mechanism, saturation and runtime

T002 offline Stage A is complete in [coordinates.md](../research_log/T003_stage_a/coordinates.md): mean outside-subspace norm fraction0.8244, energy0.7205. Darkness and contrast have especially large cross-talk; contrast has essentially zero saturation. These are descriptive subspace diagnostics, not a causal decomposition of failure.

In T003, raw soft-direction outside-subspace energy is 72.41%; the soft gate still leaves 70.14% of effective energy outside the correct analysis subspace. Correct-family hard gates make that energy zero by construction, which is not itself evidence of task alignment.

| Variant | Raw / effective outside energy | Effective/raw norm ratio | Mean saturation3 | Own CLIP loss decreases at3 | Mean seconds3 |
|---|---:|---:|---:|---:|---:|
| generic | 72.05% / 72.05% | 1.0000 | 3.470% | 93.33% | 0.1179 |
| soft | 72.41% / 72.41% | 1.0000 | 3.463% | 93.50% | 0.1232 |
| oracle_prompt | 69.10% / 69.10% | 1.0000 | 3.250% | 94.75% | 0.1261 |
| soft_gate | 72.41% / 70.14% | 0.3298 | 2.651% | 97.58% | 0.1248 |
| soft_oracle_gate | 72.41% / 0.00% | 0.4413 | 2.438% | 94.67% | 0.1252 |
| oracle_both | 69.10% / 0.00% | 0.4672 | 2.414% | 93.92% | 0.1264 |

Gamma-s2 saturation after3 steps is 8.33% generic, 5.55% soft-gate, 3.20% oracle-both; p95 is 31.16%,20.35%,15.91% respectively. Yet beneficial one-step frequencies are45.5%,45.5%,52.5%, and the paired oracle-both improvement CI still includes zero within gamma-s2. Its post1 saturation buckets show harmful fractions47.5%,59.1%,28.6%,40.0% (N=120,44,21,15), without a monotonic pattern. Gamma-s1 has some high-saturation small groups with more harm, but these do not justify a broad causal claim. Both contrast settings remain at zero saturation across variants while still showing mixed detector/AP effects. A clamp ablation should remain a separate research decision.

Runtime includes condition inference/setup and three-step adaptation with diagnostics/GPU synchronization, excluding annotated loss/AP evaluation, image loading and common-CLIP diagnostics. Mean extra time versus generic is about5–9ms. Mean peak allocated GPU memory is about839MiB, with model residency included. Oracle timing includes the common conditioner invocation and is implementation timing, not a minimal theoretical oracle cost. Per-case latency/saturation paired intervals and complete raw/physical trajectories are retained.

## Validation, environment and observed failures

- Final A6000 full suite:34 passed in7.79s, immediately before full study. Relevant tests cover detached condition equations, masks/SGD, frozen parameter/buffer state, phi gradients, episode reset, detector oracle isolation and baseline preprocessing/evaluation.
- R005 postprocessing: paired-bootstrap and coordinate known-example tests2 passed in1.18s; full smoke renderer passed. Final receipt audit verified7200 rows,36 groups×200,200 IDs, all-zero fresh phi0, identical initial detector gradient/loss across variants, and effective-gradient=gate×raw-gradient.
- Two strict cached-equality smokes failed and are retained:20260912-030241 (native detector gradients max discrepancy1.08e-4) and20260912-030559 (three-step generic phi discrepancy6.2841e-7). Repeat diagnostic20260912-030416 reproduced same-process detector gradient variation despite exact forward loss. Protocol repair uses fresh paired gradients and fresh generic; equality tolerances were not widened.
- First failed smoke command mislabeled its revision35d074b; actual c8ac8ff was verified by local/remote SHA and documented in SOURCE_CORRECTION.md. A local fetch/report ordering mistake was fixed by waiting for SCP completion; no model/code change.
- Final cross-run audit: initial detector forward loss max difference0; detector gradient max difference0.0015283 versus T002; generic AP differences versus T002 are below0.01 points. Primary conclusions use within-T003 pairing, never cached T002 gradients.
- CUDA backward nondeterminism and this one-subset, one-seed exploratory study limit generalization. Confidence intervals reflect image sampling, not between-run numerical variance or multiple comparisons. No temperature/lr/prompt tuning was performed.

Remote environment: Python3.12.12, torch2.4.0, torchvision0.19.0+cu121, Transformers4.44.2, pycocotools2.0.10, NumPy1.26.4, Pillow12.3.0, CUDA12.1, RTX A6000. Local postprocessing uses Python3.12.7, NumPy1.26.4, SciPy1.13.1, Matplotlib3.9.2.

CLIP revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; weight SHA256 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`. Detector weight SHA256 `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`. Dataset annotation SHA256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`. The subset manifest retains exact IDs/image hashes.

## Artifacts and next action

- [Full paired tables](../research_log/remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/results.md), [all statistics](../research_log/remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/analysis.json), [AP figure](../research_log/remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/ap_change.pdf), [environment](../research_log/remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/environment.json), [completion](../research_log/remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/completion.json).
- Raw7200 rows exceed GitHub single-file size; lossless [samples.jsonl.gz](../research_log/remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/samples.jsonl.gz) is tracked instead, with [SHA256/round-trip receipt](../research_log/remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/samples_archive.json) and [decompression instructions](../research_log/remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/ARCHIVE.md). Raw JSONL remains local and on A6000; no receipt was deleted. All72 prediction files are retained.
- Remote experiment artifacts remain in /home/liujianhua/wjq/TAISP/runs/20260912-031123-taisp-t003-coco200/artifacts/study. Reports and recovery notes are mirrored to the remote project research_log.
- Source changes: conditioned_clip.py, optional gate in adapt.py, run_t003.py/config, coordinate/report scripts and focused tests. All existing model/data/ISP/oracle/evaluator owners were reused. No learned prompts, predictor/source/meta-training, detector updates, smooth clamp or ViT³ structures.
- T003 status NEEDS_REVIEW. Await research lead acceptance/next explicit task; heartbeat executes newly assigned work directly and avoids rerunning this completed experiment.

## T004 — IN_PROGRESS (R006 accepted)
T003 closed. Baseline local9 passed10.77s/A6000 real suite34 passed7.69s. Pre-run contract research_log/T004_plan.md fixes last-layer postnorm/projected7x7 CLIP tokens, detector score>=0.5/top20 overlap weights/uniform empty fallback, generic/oracle directions, no masks, lr0.1/K3, original fixed200 IDs, and norm-matched one-step diagnostics. No intermediate layer search or training.

### T004 implementation/smoke complete
Patch feature gate15 passed5.23s on A6000;full suite42 passed8.52s. Smoke20260912-050701 completed2 images/72 variant observations/all AP evaluations in27.5378s,exit0. Exact geometry,postnorm/projection,frozen-state/gradient/reset,original-only detector weights,fallback and norm-match/partition tests passed. Report pipeline succeeded. Same predeclared settings proceed to full200;no training or tuning.

### T004 full fixed-subset run active
Run20260912-051214-taisp-t004-coco200;release20260912-051210-taisp-t004-full;source4817825. Fresh42 real-model/regression tests passed8.17s;first4/200 images processed normally.7200 primary observations/72 AP evaluations plus norm-matched loss and patch-partition diagnostics planned. IN_PROGRESS; no full results yet. Peak memory is adaptation-phase allocation, not total detector-setup peak. Recovery/next reporting steps in research_log/T004_handoff.md.

---

## T004 final report — NEEDS_REVIEW

2026-09-12, Asia/Shanghai. R006 cfdec6e assigned this study after accepting T003. All T004 implementation, fixed-subset experiment and report deliverables are complete. No active experiment remains; research acceptance is pending.

## Finding and recommendation

Last-layer patch-token supervision does not provide a supported overall improvement in deployable semantic-gradient alignment. Global-generic mean cosine is0.04531, versus0.01529 for uniform patches and0.02003 for region-weighted patches; paired differences are−0.03002 [−0.08513,0.02399] and−0.02528 [−0.07905,0.02981]. Primary beneficial detector-loss steps rise slightly from47.17% to48.58%/48.75%, but both paired intervals cross zero. These results do not support global pooling being the main bottleneck under the tested readout.

The positive oracle result must be separated from the representation effect. Norm-matched region-oracle updates benefit51.50% of samples, +4.33 percentage points [1.17,7.58] versus global-generic. Contrast-s2 contributes a+11.0-point improvement [2.0,19.5]. However, holding the oracle text choice fixed, the same norm-matched region-oracle improvement over norm-matched global-oracle is only+1.58 points [−1.92,5.08]. Region-oracle cosine is lower than global-oracle by−0.06353 [−0.11187,−0.01523]; uniform-oracle is lower by−0.05464 [−0.10160,−0.00666]. Thus the oracle-only positive frequency signal does not establish a patch-representation advantage.

Region selection reduces effective patches from49 to24.09 on average, but it does not materially beat uniform patches in the overall paired mechanism metrics. With generic text, region-minus-uniform cosine is+0.00474 [−0.02486,0.03413], beneficial-step gain+0.17pp [−3.00,3.50]. With oracle text the corresponding differences are−0.00889 [−0.03941,0.02089] and+0.83pp [−2.17,3.58].

AP remains heterogeneous. Region-oracle AP3 improves over corrupted input in five of six settings, including+0.353 on contrast-s2 and+0.039 on color-cast-s2, but gamma-s1 drops0.039. Against the stronger contemporaneous global-generic, region-oracle gains0.224 AP on gamma-s2,0.261 on contrast-s2 and0.418 on color-cast-s2, while losing0.574 on gamma-s1 and also losing on contrast-s1/color-cast-s1. Preserve these positive and negative cases; aggregate cosine alone is not a performance predictor.

The evidence remains weak/mixed even with oracle family information and norm matching. Under the R006 decision rule, recommend that the next explicitly assigned task compare a different self-supervised signal (for example frozen detector consistency or self-distillation), rather than meta-learning this CLIP directional loss. No T005, meta-training or spatially varying ISP is launched. This conclusion concerns the tested final-layer projected patch readout and fixed region weighting, not every possible CLIP feature/signal.

## Experiment receipt

- Run `20260912-051214-taisp-t004-coco200`; release `20260912-051210-taisp-t004-full`; experiment source `4817825`. Final report helper source `8cd0eb0` adds matched-text controls without changing experiment data/settings.
- Exit0 at2026-09-12T05:56:05+08:00; study elapsed2612.53855 seconds (43.54 minutes), excluding test/model startup.
- Same200 COCO-val IDs and six corruptions as T002/T003; six primary variants;7200 observations,36 groups×200,72 AP evaluations, plus7200 norm-matched one-step diagnostics. No sample exclusion or tuning.
- Same frozen CLIP ViT-B/32 and FasterRCNN ResNet50-FPN COCO_V1; same hard-clamped global8D ISP. phi0=0,lr0.1,K3,consistency/reg0,seed20260912. Annotated native-loss sampling seed20260912+image_id.
- A fresh g_det/loss is computed once per image/case and shared across all variants. No cached cross-run detector gradients are used. Generic and oracle global baselines are rerun in T004. Clean/corrupted AP uses identical T002 subset receipts.

```sh
export OMP_NUM_THREADS=1
export TAISP_SOURCE_REVISION=4817825
export PATH=/home/liujianhua/wjq/TAISP/.venv/bin:$PATH
TAISP_REAL_MODELS=1 python -m pytest -q
python -m taisp.analysis.run_t004 --data-root /home/liujianhua/wjq/TAISP/shared/coco200 --baseline /home/liujianhua/wjq/TAISP/runs/20260912-013248-taisp-t002-coco200-final/artifacts/study --output "$AUTODL_ARTIFACTS_DIR/study"
# Local report from project root:
python -m scripts.report_t004 research_log/remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study
```

## Exact representations and weighting

Global baseline uses the existing final normalized CLS embedding. Local features use the parity-correct differentiable shortest-side224 resize/floor center crop; take last_hidden_state[:,1:], apply frozen vision post_layernorm, then the same visual_projection768→512, then per-token L2 normalization. This yields7×7 patch features. No intermediate block was selected. Pinned embedding code confirms row-major flatten(2).transpose(1,2) after CLS. Patch-wise text alignment is diagnostic and was not directly guaranteed by global CLIP pretraining; patch tokens retain global self-attention context.

Uniform loss is the average negative projected enhanced-minus-original token displacement. Region loss is the weighted sum of the same per-token scores. The original image reference is detached and re-encoded under no_grad within each loss call, as in the global baseline. CLIP weights/buffers remain frozen; gradients reach phi through enhanced pixels.

Detector-region weights use exactly one frozen eval/no_grad detector inference on original corrupted input: score>=0.5,descending top20. Map original xyxy boxes with the exact integer resized width/height and floor crop offsets, clip to224 crop, intersect each32×32 patch. Raw weight is sum(score×intersection_area/1024), then normalize over49 patches. If no visible valid region exists, use uniform1/49. Boxes/scores/support/weights are fixed and detached for the episode; no annotation, corruption ID, detector gradient or adapted-image prediction determines them.

All six primary variants update all eight ISP coordinates with the same fixed lr; there are no oracle masks. The extra diagnostic rescales each initial semantic gradient to the contemporaneous global-generic gradient norm and takes one step. Labels do not determine scaling. No zero gradients occurred. Maximum norm-target absolute discrepancy is1.1921e-7 from floating arithmetic. Primary fixed-lr AP remains the deployment comparison.

Positive text bank (unchanged):
- `a clear natural photograph`
- `a well-lit photograph with natural colors`
- `a sharp and clear photograph with good visibility`

darkness:
- `a dark underexposed photograph`
- `a poorly lit photograph with low visibility`

low_contrast:
- `a low contrast washed-out photograph`
- `a hazy photograph with faded details`

color_cast:
- `a photograph with an unnatural color cast`
- `a photograph with distorted and unbalanced colors`

Normalize text embeddings, average each bank and normalize its mean. Generic direction remains normalize(t_pos−normalized aggregate negative mean); oracle direction uses the correct analysis-family negative bank. T004 reuses stored bank construction but never runs the T003 condition classifier prepare() method. No learned prompts.

## Overall primary and norm-matched results

Each variant contains1200 observations. Intervals below use2000 paired image-cluster percentile bootstrap draws,seed20260912; all six conditions follow each sampled image. Exploratory95% intervals, no multiplicity adjustment.

| Variant | Mean gradient norm | Mean / median cosine | Positive alignment | Primary loss benefit | Norm-matched benefit | Norm-matched Δbenefit pp vs global-generic [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| global_generic | 0.12575 | 0.04531 / 0.05628 | 53.83% | 47.17% | 47.17% | +0.000 [+0.000, +0.000] |
| global_oracle | 0.12457 | 0.07074 / 0.10302 | 55.58% | 50.83% | 49.92% | +2.750 [-0.250, +5.833] |
| patch_generic | 0.03470 | 0.01529 / 0.04155 | 51.92% | 48.58% | 48.33% | +1.167 [-2.333, +4.252] |
| patch_oracle | 0.03510 | 0.01610 / 0.02802 | 51.67% | 48.58% | 48.00% | +0.833 [-2.417, +4.085] |
| region_generic | 0.04038 | 0.02003 / 0.04505 | 52.83% | 48.75% | 48.00% | +0.833 [-2.417, +4.000] |
| region_oracle | 0.04179 | 0.00721 / 0.03496 | 51.58% | 49.42% | 51.50% | +4.333 [+1.167, +7.583] |

Mean raw local-gradient norms are0.0347–0.0418 compared with0.12575 global-generic. Norm matching is therefore essential to interpreting the apparent stability. Even matched region-oracle mean loss delta remains+0.000276; its paired mean-loss improvement versus global-generic−0.001484 has interval[−0.003298,+0.000568], crossing zero. Beneficial-step frequency and mean-loss change answer different questions.

### Visual effect with text choice held fixed

Both sides of the norm-matched comparison use each image/case global-generic target norm.

| Local variant | Global control | Δcosine [95% CI] | Δprimary benefit pp [95% CI] | Δmatched benefit pp [95% CI] |
|---|---|---:|---:|---:|
| patch_generic | global_generic | -0.030 [-0.085, +0.024] | +1.417 [-2.167, +4.919] | +1.167 [-2.333, +4.252] |
| patch_oracle | global_oracle | -0.055 [-0.102, -0.007] | -2.250 [-5.419, +1.000] | -1.917 [-5.250, +1.500] |
| region_generic | global_generic | -0.025 [-0.079, +0.030] | +1.583 [-1.917, +5.083] | +0.833 [-2.417, +4.000] |
| region_oracle | global_oracle | -0.064 [-0.112, -0.015] | -1.417 [-4.585, +1.917] | +1.583 [-1.917, +5.083] |

## All family/severity AP1 / AP3

COCO subset AP in0–100 points; clean AP38.012. No full-COCO benchmark claim, no per-image AP averaging or AP confidence intervals.

| Case | Corrupted | Global generic | Global oracle | Uniform generic | Uniform oracle | Region generic | Region oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| gamma_s1 | 38.145 | 38.351 / 38.680 | 38.568 / 38.437 | 38.177 / 38.253 | 38.152 / 38.481 | 38.211 / 38.112 | 38.184 / 38.106 |
| gamma_s2 | 36.446 | 36.573 / 36.388 | 36.619 / 37.113 | 36.488 / 36.486 | 36.490 / 36.558 | 36.492 / 36.604 | 36.492 / 36.612 |
| contrast_s1 | 36.175 | 36.369 / 36.508 | 36.301 / 36.553 | 36.154 / 36.136 | 36.212 / 36.191 | 36.172 / 36.138 | 36.315 / 36.434 |
| contrast_s2 | 31.175 | 31.115 / 31.267 | 30.888 / 31.016 | 31.389 / 31.267 | 31.320 / 31.277 | 31.388 / 31.369 | 31.363 / 31.528 |
| color_cast_s1 | 37.642 | 37.668 / 37.724 | 37.274 / 37.266 | 37.399 / 37.718 | 37.349 / 37.324 | 37.738 / 37.804 | 37.688 / 37.664 |
| color_cast_s2 | 35.908 | 35.841 / 35.528 | 35.801 / 35.784 | 35.828 / 35.813 | 35.839 / 35.795 | 35.891 / 35.784 | 35.963 / 35.947 |

All per-case gradient statistics, paired deltas versus global-generic, same-text global controls, region-versus-uniform intervals, norm-matched/Taylor analyses, AP differences, saturation and runtime are in [results.md](../research_log/remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/results.md) and [analysis.json](../research_log/remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/analysis.json). No negative family is omitted.

## Patch evidence and coordinate mechanism

Original detector predictions cover62.12% of the crop patch grid on average. Mean selected boxes7.16, visible boxes6.24. Exactly11/1200 image/cases use uniform fallback (0.92%). Region weights affect63.03% of patches including fallback and have effective count24.09 versus49 for uniform weighting. Weights are based on predictions, not GT object masks.

| Local variant | Effective patches | Object / background initial gradient norm | Object / background weighted loss3 | Partition-gradient residual max |
|---|---:|---:|---:|---:|
| patch_generic | 49.00 | 0.02134 / 0.01943 | -0.000158 / -0.000152 | 9.17e-05 |
| patch_oracle | 49.00 | 0.02183 / 0.01942 | -0.000162 / -0.000161 | 0.000117 |
| region_generic | 24.09 | 0.03958 / 0.00080 | -0.000407 / -0.000013 | 8.87e-08 |
| region_oracle | 24.09 | 0.04097 / 0.00083 | -0.000426 / -0.000014 | 6.8e-08 |

Region background weighted loss is zero by construction when object support exists. The small nonzero overall background contribution comes from the11 uniform fallback cases. Uniform-patch object and background gradient norms are similar despite different support sizes; selecting object tokens does not guarantee task-aligned gradients. The raw records retain object/background gradients, all49 weighted/unweighted patch scores at1/3 steps, boxes, scores, weights and support.

Gradient-energy percentages below normalize squared coordinates per observation before averaging. RGB gains still dominate local objectives (approximately73–74% combined), so token readout has not removed coordinate cross-talk.

| Variant | Gamma | Red | Green | Blue | Contrast | Brightness | Tone | Sharpen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global_generic | 3.41 | 29.00 | 26.18 | 12.30 | 19.14 | 8.55 | 0.82 | 0.60 |
| global_oracle | 5.20 | 28.21 | 27.61 | 12.40 | 16.19 | 8.49 | 1.27 | 0.63 |
| patch_generic | 4.53 | 32.29 | 27.66 | 13.21 | 13.47 | 7.69 | 0.94 | 0.21 |
| patch_oracle | 4.20 | 32.35 | 28.40 | 12.86 | 13.32 | 7.81 | 0.87 | 0.19 |
| region_generic | 5.46 | 31.89 | 27.82 | 14.07 | 12.43 | 6.93 | 1.16 | 0.22 |
| region_oracle | 4.96 | 31.34 | 28.76 | 13.43 | 13.10 | 7.12 | 1.05 | 0.24 |

## Semantic progress, saturation and costs

Local own semantic losses decrease after3 steps in96.67–98.08% of observations, yet primary detector-loss benefit remains48.58–49.42%. Mean saturation3 drops from3.47% global-generic to2.75% uniform-generic and2.37% region-generic. These reductions accompany substantially smaller raw gradients and do not establish improved semantic direction. Both contrast settings remain at essentially zero saturation.

| Variant | Own semantic loss decreases at3 | Mean saturation3 | Adapt seconds3 | Deployment seconds3 | Adaptation peak MiB |
|---|---:|---:|---:|---:|---:|
| global_generic | 93.33% | 3.470% | 0.1106 | 0.1106 | 841.8 |
| global_oracle | 94.75% | 3.250% | 0.1122 | 0.1122 | 841.9 |
| patch_generic | 96.67% | 2.753% | 0.1143 | 0.1143 | 842.0 |
| patch_oracle | 98.08% | 2.386% | 0.1096 | 0.1096 | 842.0 |
| region_generic | 97.08% | 2.372% | 0.1088 | 0.1386 | 841.9 |
| region_oracle | 97.58% | 2.335% | 0.1091 | 0.1390 | 842.0 |

Region deployment time includes the original detector inference/map, averaging29.86ms extra setup; that inference is diagnostic-only and excluded for the global/uniform paths. Adaptation timing includes its diagnostics and GPU synchronization but excludes annotated-loss/AP/norm-match/partition computation. Peak memory is adaptation-phase allocated memory only; detector-setup peak was not separately recorded. Variant execution order was fixed, so millisecond differences are descriptive and not a hardware-optimized speed benchmark.

## Validation and reproducibility

- Baseline:9 local tests passed10.77s;34 A6000 tests passed7.69s. New feature/preprocessing/adapt gate15 passed5.23s.
- Smoke run20260912-050701-taisp-t004-study-smoke:42 real-model tests passed8.52s;2 images/72 observations/all AP evaluations,27.53785s,exit0. No smoke-driven setting changes.
- Full run starts with42 real-model/regression tests passed8.17s. Tests cover odd/rectangular/square crop geometry, direct post-LN/projection equivalence, frozen model state, finite phi gradients, repeated fresh episodes, region overlap/score/top-k/fallback, original-only detector invocation, norm matching and partition gradients.
- Final analysis-related tests3 passed6.85s. Complete receipt audit passed:7200 rows,36 groups×200,200 IDs, all-zero fresh phi0, exact shared g_det/loss across variants, patch weight normalization and norm-match target checks. Figure visually inspected.
- No T004 implementation or experiment failures. Two documentation trailing-CR lines were normalized after git diff --check identified them; no scientific change.
- Recomputed uniform patch partition gradients differ from the saved total by at most1.1682e-4; region partition residuals are below9e-8. These discrepancies are retained rather than assuming bitwise CUDA backward reproducibility. CPU partition sum tests pass. All scientific comparisons use fresh within-run detector gradients.
- One fixed200-image subset and one seed; intervals reflect image resampling, not run-to-run numerical uncertainty or multiplicity control. Oracle directions are analysis-only. No meta-training, learned prompts, intermediate-layer search, predictor/source training, smooth clamp, detector update or ViT³ mechanism was introduced.

Remote environment: Python3.12.12,torch2.4.0,torchvision0.19.0+cu121,Transformers4.44.2,pycocotools2.0.10,NumPy1.26.4,Pillow12.3.0,CUDA12.1,RTX A6000. Local reports use Python3.12.7,NumPy1.26.4,SciPy1.13.1,Matplotlib3.9.2.

CLIP revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; SHA256 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`. Detector SHA256 `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`. Annotation SHA256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`. The subset manifest retains exact image IDs/hashes.

## Delivery and next action

- [Full paired tables](../research_log/remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/results.md), [all analysis](../research_log/remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/analysis.json), [AP figure](../research_log/remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/ap_change.pdf), [raw7200 observations](../research_log/remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/samples.jsonl), [receipt audit](../research_log/remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/receipt_audit.json), [environment](../research_log/remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/environment.json), [completion](../research_log/remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/completion.json).
- Raw samples.jsonl is96,709,726 bytes, below GitHub single-file limit; retained unchanged without compression. SHA25681a75284aed074427f82e6b743d62192ef5cf93882f4f7ab2f31ad34d92c56ca. All72 prediction files and raw logs are retained.
- Remote artifacts: /home/liujianhua/wjq/TAISP/runs/20260912-051214-taisp-t004-coco200/artifacts/study; reports/recovery notes mirrored under project research_log.
- Source: shared CLIP geometry/patch extension, spatial loss, region mapper, spatial diagnostics/runner/config/tests and report script. Existing ISP, adaptation, detector, COCO and oracle/evaluator owners were reused.
- T004 NEEDS_REVIEW. Await research lead acceptance/next explicit task. Heartbeat executes newly assigned work directly; do not rerun this completed study or start meta-training automatically.

## T005 IN_PROGRESS — R007 acknowledged

T004 accepted/closed. T005 pre-run definitions in research_log/T005_plan.md;
fixed-ROI signal and detached stable support implemented. Baseline42real tests
pass; ROI5tests pass. Native CE/JS/fallback CPU unit tests pass. GPU repeated
update discrepancy1.49682e-5 retained; validating reset on real CPU with unchanged
1e-6 tolerance and GPU nonzero gradients/frozen model checks. No full run yet.
No meta-training, spatial ISP, annotation inputs or tuned hyperparameters added.

### T005 implementation checkpoint — ready for fixed full study

50real tests passed87.80s; smoke20260912-064017 completed2images/56rows/all63AP
evaluations27.37618s,exit0. CPU repeated episode same1e-6 tolerance passes; GPU
freeze/nonzero-gradient checks pass. Four analysis tests passed11.02s. Exact
sharedgdet/zero phi0/weight sums/normtargets audited. Smoke only checks software;
no result-driven change. Sourcef64151a, report helper and raw smoke receipts
committed next. Starting unchanged200-image study plus clean controls directly.
