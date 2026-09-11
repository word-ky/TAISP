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
