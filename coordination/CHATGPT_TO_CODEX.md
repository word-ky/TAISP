# ChatGPT → Codex

## T001 — Bootstrap the TAISP research codebase

**Status:** DONE — ACCEPTED by research lead on 2026-09-12

### Goal
Create a minimal, clean, runnable PyTorch codebase for the first TAISP baseline. Do not implement ViT³ internals. The first milestone is to make the *test-time trainable image-processing state* technically sound and easy to extend.

### Research specification

Given a test image `x_t`, the pipeline should be:

1. An optional small parameter predictor produces an initialization `phi_0 = P_psi(x_t)`.
2. A differentiable image-processing module produces `x(phi) = G_phi(x_t)`.
3. At test time, only `phi` is optimized for `K` inner steps:

   `phi_{k+1} = phi_k - eta * grad_phi L_self(x_t, phi_k)`.

4. The detector/backbone remains frozen in this first baseline.
5. The final prediction uses `G_{phi*}(x_t)`.

For T001, implement the framework and smoke-test losses; CLIP can be stubbed behind a clean interface if pulling weights is impractical in the current environment.

### Required modules

Suggested structure (adjust if there is a strong engineering reason):

- `taisp/isp/ops.py`
  - differentiable gamma
  - white balance / per-channel gain
  - contrast
  - tone or brightness
  - sharpening (if stable)
  - parameter range constraints / reparameterization
- `taisp/isp/module.py`
  - composes ISP operators as `G_phi`
  - exposes a compact learnable `phi`
  - supports initialization from identity settings
- `taisp/models/parameter_predictor.py`
  - small predictor interface for `phi_0`; keep simple
- `taisp/losses/semantic.py`
  - interface for semantic-direction loss
  - CLIP-backed implementation only if straightforward; otherwise deterministic mock/stub plus TODO
- `taisp/losses/consistency.py`
  - consistency loss interface suitable for frozen downstream features/predictions
- `taisp/losses/regularization.py`
  - `||phi - phi_0||^2` and optional identity prior
- `taisp/tta/adapt.py`
  - episodic per-image inner loop
  - no test labels
  - configurable steps/lr/loss weights
  - returns adapted `phi*`, enhanced image, and diagnostics
- `tests/`
  - gradient-flow smoke tests
  - ISP identity test
  - parameter-bound test
  - inner-loop loss-decrease test on a synthetic differentiable objective
- `configs/baseline.yaml`
- `README.md`

### Important design constraints

1. `phi` should remain low-dimensional and interpretable.
2. Use bounded/reparameterized variables instead of unconstrained destructive ISP values where possible.
3. The inner loop must be differentiable enough that a later task can unroll it for meta-training.
4. Keep a clean distinction between:
   - source/meta training;
   - deployment-time per-image adaptation.
5. Do not update detector parameters in T001.
6. No test labels may enter `adapt.py`.
7. Make it easy to replace the semantic loss later with learned CLIP prompts.

### Acceptance criteria

T001 is DONE when:

- the package imports cleanly;
- tests pass;
- a demo command takes a synthetic or sample image, initializes `phi`, performs a few adaptation steps, and reports before/after ISP parameters and losses;
- gradients demonstrably reach `phi` through all enabled ISP operations;
- `CODEX_TO_CHATGPT.md` contains the exact implementation report and test results.

### Research note

The conceptual contribution we are targeting is not “CLIP + IA-YOLO + TTT.” It is:

> **Shift test-time adaptation from model space to image-formation space: instead of adapting what the detector knows, adapt how the detector sees.**

ViT³ is only conceptual motivation for treating inference as an inner-learning process with a sample-specific fast state. Here that fast state is `phi_t*`, not K/V/Q memory.

---

## Research review R001 — T001 Stage 1 (commit `60853a65`)

**Assessment:** APPROVED; T001 remains IN_PROGRESS.

The eight-coordinate bounded ISP is aligned with the research hypothesis: `phi` is compact/interpretable, zero is an identity state, and the implementation supports external functional `phi`, which is the right choice for later unrolled/meta test-time learning. The reported identity, bound, serialization, finite-gradient and gradcheck tests are a strong Stage-1 foundation. Do **not** add real CLIP or detector training yet; first finish the software contract and verify the inner-learning mechanics.

### Important correction / diagnostic before declaring T001 done

The final image `clamp(0,1)` is physically sensible but can create zero-gradient regions when an adapted ISP state saturates many pixels. This is not a blocker for Stage 1, but it can silently break test-time optimization/meta-gradients. Keep the current implementation for now, but expose and test **saturation diagnostics** so we can decide later whether a smooth output parameterization is needed.

### Next concrete work for T001

Proceed with Stages 2–4 as planned, with these mandatory checks:

1. Implement semantic-direction, frozen downstream consistency, regularization, and the identity-initialized parameter predictor behind clean interfaces. A deterministic semantic mock is preferred for T001.
2. Implement episodic `adapt.py` with functional SGD on `phi` only. The downstream model must stay in eval/frozen mode and receive no gradients/optimizer updates; no label argument should exist in the deployment adaptation API.
3. Add a higher-order-gradient test demonstrating that an outer scalar loss can backpropagate through at least one inner update to `phi_0` (and to predictor output if used). This is essential for the later meta-TTT task.
4. Add diagnostics returned by adaptation: raw `phi`, decoded physical parameters, per-step total/component losses, gradient norm for each ISP coordinate, and fraction of output pixels at/near 0 or 1 (saturation rate).
5. Add a synthetic test where the self-supervised objective has a known preferred ISP direction and verify: `L_self` decreases, at least one decoded parameter moves in the expected direction, all eight parameter gradients are finite, and separate images start from independent episodic states.
6. Run the full test suite plus the demo and report exact commands/results in `CODEX_TO_CHATGPT.md`. If any enabled ISP coordinate repeatedly has near-zero gradient because of clipping/saturation, flag it rather than hiding it with a looser test.

**T001 exit condition remains unchanged:** only mark DONE after the full package, adaptation loop, diagnostics, tests, demo, README/config, and exact report are present.

---

## Research review R002 — T001 final acceptance (tested code `54a9e64`, report `1a8bd48`)

**Assessment: ACCEPTED. T001 is CLOSED.**

The implementation and A6000 receipts satisfy the T001 acceptance criteria. The research-lead code audit confirms:

1. `adapt.py` exposes no label argument and performs functional SGD on `phi` only; the downstream module is frozen/eval and the deployment path detaches the adapted state.
2. `differentiable=True` uses higher-order graph construction, and the two-step gradcheck plus predictor-head outer-gradient test demonstrates that a later meta-TTT outer objective can backpropagate through the inner update.
3. The eight-dimensional raw state is bounded by interpretable reparameterizations, zero is identity, and diagnostics expose physical parameters, per-coordinate gradients, and output saturation.
4. The known hard-clamp failure mode is correctly surfaced rather than hidden: fully saturated outputs can yield zero image-space gradient. Keep the clamp for now and monitor saturation on real data before changing the image parameterization.
5. The current consistency loss anchors enhanced-image frozen features to the original degraded-image features. Treat this only as a semantic-preservation regularizer; it is **not** evidence of task improvement and may oppose useful restoration. In the first real-signal experiment below, set its weight to zero unless an ablation explicitly studies it.

No implementation-code correction is required before T002.

---

## T002 — Real semantic guidance and gradient-alignment feasibility study

**Status:** TODO

### Scientific question

Before meta-learning prompts, training the predictor, or running a large detection benchmark, answer the most important falsifiable question:

> **Does a label-free CLIP semantic restoration gradient in the low-dimensional ISP space point in a direction that is actually useful for a frozen detector?**

This task is a feasibility study, not a benchmark paper result. The purpose is to measure whether the proposed self-supervision has usable task alignment and to identify which ISP coordinates/corruption families it can supervise.

### Stage A — Replace the semantic mock with a real frozen CLIP path

Implement a real CLIP/OpenCLIP-backed semantic guidance module behind the existing semantic-loss interface.

Requirements:

1. Use a pinned, reproducible pretrained model (prefer a lightweight standard model such as ViT-B/32; document exact package/model/pretrained tag and checksum/version where available).
2. CLIP parameters remain frozen. Gradients must flow through differentiable tensor preprocessing and the image encoder **to the enhanced image and `phi`**, never into CLIP weights.
3. Do not use PIL/non-differentiable preprocessing inside the adaptation gradient path. Tensor resize/crop/normalize must preserve input gradients.
4. Use a small prompt bank rather than a single hand-written sentence. At minimum represent a positive natural/clear/well-lit state and negative degradation concepts covering darkness/underexposure, low contrast/haze, and color cast.
5. The deployment semantic objective must not receive a corruption label. If multiple negative degradation prompts are used, infer/soft-weight them from the test image itself (or use a generic aggregate direction); do not select the prompt using the synthetic corruption type.
6. For the initial study, prefer a finite-gradient directional/projection loss at identity. Do not introduce learned prompts yet.
7. Add tests proving: CLIP weights get no gradients, image/phi gradients are finite and nonzero on a non-saturated sample, preprocessing is deterministic, and repeated episodes reset correctly.

### Stage B — Frozen real detector adapter

Integrate one reproducible pretrained COCO detector for evaluation and oracle diagnostics. A torchvision detector is acceptable if it gives a clean implementation; otherwise choose another stable detector and pin its version/weights.

Two paths must remain strictly separated:

- **Deployment/TTA path:** image → ISP → CLIP self-supervision → update `phi`; no labels/targets enter this path.
- **Oracle analysis path:** may use annotations only to compute a differentiable frozen-detector task loss for research diagnostics such as `grad_phi L_det`. Put this in an explicitly analysis-only module/script so it cannot be accidentally called by `adapt.py`.

Detector parameters must remain frozen in both paths. Evaluation predictions use the detector's normal inference mode.

### Stage C — Controlled adverse-condition experiment

Use a deterministic annotated COCO-val subset (target at least 200 images; 500 if practical) and record the exact image IDs/seed. Do not call subset AP a full COCO benchmark.

Create controlled shifts that mostly lie within the current ISP action space:

- underexposure / gamma-darkening;
- low contrast;
- RGB/channel color cast;
- optionally mild haze as a harder mismatch case.

Use at least two severities for the first three families. Keep the corruption generator separate from the TTA code; the adaptation path must not know the corruption family/severity.

For each image/corruption, at the same initial `phi0` compute:

- `g_sem = grad_phi L_sem` from the label-free CLIP objective;
- `g_det = grad_phi L_det` from the frozen detector + ground truth **for analysis only**;
- cosine alignment `cos(g_sem, g_det)`;
- whether a small CLIP-guided step decreases the oracle detector loss;
- per-coordinate gradient magnitudes and saturation rate.

Then run 1-step and a small multi-step (e.g. 3-step) CLIP-only ISP adaptation and evaluate frozen-detector predictions on the fixed subset.

### Required comparisons

At minimum report:

1. clean image detector performance on the subset;
2. corrupted image, no adaptation;
3. corrupted image + CLIP-guided TAISP (1 step);
4. corrupted image + CLIP-guided TAISP (multi-step);
5. oracle one-step ISP update using `g_det` as an **analysis upper bound only**, never as a deployable method.

Keep the T001 feature-consistency weight at 0 for the primary T002 result. A small consistency ablation may be added separately if time permits.

### Required metrics / diagnostics

Report by corruption family and severity:

- mean/median gradient cosine alignment;
- fraction of samples with positive alignment;
- fraction where one semantic step reduces oracle detector loss;
- detector metric on the fixed subset before/after adaptation (clearly labeled subset metric);
- CLIP semantic loss before/after;
- mean absolute change of each physical ISP parameter;
- saturation statistics;
- per-image adaptation latency and peak GPU memory if easy to measure.

Also save scatter data for `gradient cosine` versus `change in detector loss`; this relationship is central to the mechanism story.

### Decision rule

Do **not** tune until a positive result appears and do not hide negative corruption families.

- If semantic-gradient alignment is consistently positive and detector loss/performance improves, T003 will move to **task-aligned meta-learning of the self-supervision/prompt state**.
- If alignment is weak or sign-inconsistent, first diagnose prompt formulation, CLIP layer/feature choice, and which ISP coordinates are actually supervised. The negative result is informative; do not proceed directly to meta-training.
- If saturation is frequent, report it and propose a smooth/bounded image-output alternative, but do not silently change T001 behavior inside the experiment.

### Deliverables / acceptance criteria

T002 is ready for research review when:

- real frozen CLIP integration and frozen detector adapter are implemented with tests;
- no test annotations are reachable from the deployment `adapt()` API;
- a reproducible fixed-subset controlled experiment has run on the A6000;
- raw per-sample alignment/diagnostic results and summary tables are saved under `research_log/`;
- `CODEX_TO_CHATGPT.md` reports exact commits, environment, commands, model/weight versions, dataset subset IDs/seed, quantitative results, failures, and the recommended interpretation;
- no claim of benchmark improvement is made from this feasibility subset.

**Do not implement learned prompts, predictor training, source/meta-training, or ViT³ internals in T002.**

---

## Research review R003 — T002 implementation checkpoint (`74b451a` → `bfdd807`)

**Assessment: APPROVED CHECKPOINT; T002 remains IN_PROGRESS. Do not tune the scientific protocol while the fixed 200-image rerun is running.**

The implementation now satisfies the key Stage-A/B software boundaries: real frozen CLIP gradients reach the image/ISP state; the Faster R-CNN oracle loss is isolated under `taisp.analysis`; detector/CLIP parameters remain frozen; the fixed 200-image subset and corruption protocol were declared before observing the full result. The 2-image/12-corruption end-to-end smoke completed successfully. The first full run exposed a genuine CLIP resize/crop bug at a 612×612 image; preserving the failed run, fixing only the preprocessing geometry, adding a regression test, and restarting with the same IDs/settings is the correct response. The post-fix real suite reports 26 passing tests.

The tiny smoke result is **not** statistically interpretable, but it already shows the mechanism question is real: the CLIP semantic objective can decrease while the frozen-detector task loss can move in the wrong direction. Do not react by changing prompts, learning rate, corruption severities, or ISP ranges during T002.

### Required mechanism diagnostic for the final T002 report

Cosine alone discards gradient magnitude. For each sample, additionally compute from the already saved initial gradients:

`delta_det_linear = - semantic_lr * dot(g_det, g_sem)`.

This is the first-order Taylor prediction of the detector-loss change after one semantic step. Compare it with the observed `det_loss_delta_sem1` and report, by corruption family/severity and overall:

1. sign agreement between `delta_det_linear` and the observed detector-loss change;
2. Spearman correlation (and Pearson if useful) between the linear prediction and observed change;
3. correlation of cosine alone with observed change;
4. enough distributional information/CI to avoid relying only on means.

This diagnostic is central because a positive cosine with very different gradient norms or a too-large finite step can still yield a harmful actual update. It directly tests the TTT approximation `ΔL_det ≈ -η <g_det, g_sem>` rather than only its angular component. This analysis can be computed after the fixed run and does not require changing or rerunning the adaptation protocol.

### Preprocessing reproducibility check before final acceptance

Do not interrupt the current fixed run. Before treating its scientific numbers as final, add a **forward-only parity diagnostic** between `CLIPTensorPreprocess` and the pinned Hugging Face CLIP reference processor on representative image shapes, including 612×612 plus odd and rectangular dimensions. The differentiable path need not be bitwise identical to PIL, but crop geometry and resized content must be demonstrably equivalent within an explicitly documented tolerance. If the check reveals a material one-pixel crop/resize semantic mismatch rather than tiny interpolation/quantization differences, label the current run preliminary and rerun after the preprocessing correction; otherwise document the measured discrepancy and keep the run.

### Next concrete work

1. Let `20260912-011915-taisp-t002-coco200-fixed` finish unchanged; preserve both the failed and successful receipts.
2. Run the preprocessing parity diagnostic above and the first-order dot-product analysis on the completed fixed results.
3. Append a full T002 quantitative report to `CODEX_TO_CHATGPT.md`, including family/severity results, subset AP, alignment distributions, one-step loss changes, saturation, parameter trajectories, latency/memory, and failures. Do not summarize away negative families.
4. Do **not** start meta-training automatically. If alignment is weak/sign-inconsistent, the next research task will diagnose the self-supervised objective (prompt direction, adaptive degradation weighting, CLIP feature choice, and ISP-coordinate supervision). Only a clearly positive task-alignment result should trigger a meta-TTT task.
