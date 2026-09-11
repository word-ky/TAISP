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

---

## Research review R004 — T002 final acceptance (`0b8a888`, `7f80b26`, report `f6cdb80`)

**Assessment: ACCEPTED AS A NEGATIVE/DIAGNOSTIC RESULT. T002 is CLOSED. Do not start meta-training.**

The final parity-corrected 200-image/1,200-observation study satisfies the T002 protocol and acceptance criteria. The preprocessing repair was handled correctly: the earlier 200-image run was explicitly demoted to preliminary after the one-pixel center-crop mismatch was found, the pinned Hugging Face geometry was matched, 27 model/regression tests passed, and the final experiment was rerun with the same image IDs and scientific settings. The deployment/oracle boundary remains consistent with the protocol: CLIP and detector are frozen, labels are confined to `taisp.analysis`, and no learned prompts, predictor training, source/meta-training, or ViT³ internals were introduced.

The scientific conclusion is clear enough to reject the naive next step. The generic CLIP restoration direction is **not task-aligned enough** to justify meta-learning yet: mean gradient cosine is only 0.0453, positive alignment is 53.83%, and a one-step semantic update lowers annotated detector loss in only 47.58% of observations, even though the CLIP objective decreases after three steps in 93.33%. Fixed-subset AP is mixed rather than robustly improved: four settings improve slightly and two deteriorate, with gamma-s1 +0.524 AP and color-cast-s2 -0.380 AP after three steps. This is exactly the failure mode T002 was designed to expose: `L_sem ↓` does not imply `L_det ↓`.

R003 also changes the interpretation of the gradient story. The first-order prediction `-eta <g_det,g_sem>` has only 54.75% sign agreement and overall Spearman 0.153 with the observed one-step detector-loss change. Thus cosine alone is insufficient, but finite-step/non-smooth detector behavior is not the whole problem either: the self-supervised direction itself is weak and heterogeneous. Contrast-s2 shows stronger Taylor rank agreement while still having poor beneficial-step frequency, which is evidence that we must distinguish **whether the local approximation is accurate** from **whether the semantic direction is desirable**.

Saturation is a real secondary issue, especially for gamma-s2 (after-adaptation p95 saturation 31.16%; 30% of images exceed 10% saturated pixels), but it is not a complete explanation because the contrast conditions have essentially zero saturation and still show weak/mixed task alignment. Likewise, the semantic gradient is heavily concentrated on RGB gains while tone/sharpening are weak, suggesting substantial ISP-coordinate cross-talk from the generic global CLIP direction.

### T003 — Diagnose and repair condition-awareness and ISP-coordinate cross-talk

**Status: TODO. This is still a diagnosis/feasibility task, not meta-training.**

#### Scientific question

Determine whether T002 failed mainly because (a) one generic CLIP direction does not identify the current degradation, or (b) the semantic objective excites the wrong ISP coordinates even when the degradation concept is appropriate.

The key hypothesis is:

> **A test image should first infer a soft degradation state, then use that state to choose both the semantic restoration direction and the low-dimensional ISP subspace allowed to adapt.**

No test labels or synthetic corruption IDs may enter the deployable path.

#### Stage A — Offline decomposition using existing T002 receipts

Before launching new GPU experiments, use the saved initial gradients from the final T002 run to produce a per-coordinate mechanism report. No rerun is needed for this stage.

For each ISP coordinate `j`, compute and report by corruption family/severity:

- signed contribution `c_j = g_det[j] * g_sem[j]` to the local dot product;
- sign agreement between `g_det[j]` and `g_sem[j]`;
- relative semantic-gradient energy per coordinate;
- the fraction of semantic-gradient norm outside a plausible corruption subspace.

Use the following predeclared diagnostic groups only for analysis, not as corruption labels in deployment:

- darkness/underexposure subspace: `{gamma, brightness, tone}`;
- low-contrast subspace: `{contrast, tone}`;
- color-cast subspace: `{red_gain, green_gain, blue_gain}`.

Also stratify harmful/beneficial one-step outcomes by initial and post-step saturation buckets. This should answer whether saturation explains a substantial fraction of gamma failures and whether coordinate cross-talk remains after controlling for saturation.

#### Stage B — Image-conditioned CLIP degradation direction

Implement a deployable, frozen-CLIP condition-aware direction without learned prompts.

1. Keep the same positive natural/clear prompt bank as T002.
2. Maintain separate negative concept banks for at least darkness/underexposure, low contrast/haze, and color cast.
3. For the **original corrupted test image only**, compute frozen CLIP similarity to the degradation concepts and convert them to soft weights `w_m` with a predeclared temperature. Detach these weights from the adaptation graph so the ISP update cannot game the degradation classifier.
4. Form an image-conditioned text direction such as `d(x) = t_pos - sum_m w_m t_neg,m` and use it in the same directional semantic loss interface.
5. No corruption family/severity is available to this deployable weighting rule.

Run the same fixed 200-image subset and same six controlled settings. Compare at minimum:

- T002 generic aggregate direction;
- image-conditioned soft direction;
- **oracle family-selected prompt direction**, clearly analysis-only, as an upper bound on degradation identification.

Do not tune prompts/lr on the reported 200 images. Any prompt templates/temperature must be recorded before the full run; use a tiny implementation smoke only to catch software errors.

#### Stage C — Interpretable soft coordinate gating

Test whether broad ISP updates are the main source of task conflict.

Define fixed concept-to-coordinate masks before seeing the full results:

- darkness → `{gamma, brightness, tone}`;
- low contrast → `{contrast, tone}`;
- color cast → `{red_gain, green_gain, blue_gain}`.

Construct a deployable soft gate `m(x) = sum_m w_m m_m` from the same detached degradation weights and update with `g_sem_gated = m(x) ⊙ g_sem`. Do not use `g_det` to choose or modify this mask.

Compare:

1. image-conditioned direction without gating;
2. image-conditioned direction + soft coordinate gate;
3. oracle-family coordinate gate, analysis-only.

For each comparison report mean/median cosine, positive-alignment rate, one-step detector-loss improvement rate, first-order dot-product prediction, 1-step/3-step subset AP, CLIP-loss change, parameter trajectories, saturation, and latency. Preserve all negative families.

#### Stage D — Decision logic

Do not add a smooth clamp or learned prompts in the primary T003 run. First determine whether objective conditioning/gating fixes the failure; saturation is secondary and should remain measured under the unchanged hard clamp.

Interpret outcomes as follows:

- If the **oracle prompt/gate** substantially improves alignment but the deployable soft weighting does not, degradation identification is the bottleneck; T004 should improve degradation-state inference.
- If both oracle and deployable condition-aware/gated variants improve alignment and downstream behavior, T004 may proceed to task-aligned meta-learning of the weighting/prompt initialization.
- If even oracle-family prompt/gating remains weak, the global final CLIP direction is likely the wrong supervisory signal; do **not** meta-learn it. T004 should instead test CLIP layer/patch-local features or a different self-supervised/VLM signal.
- If saturation strongly predicts harm after conditioning/gating, then schedule a separate, predeclared output-parameterization ablation; do not silently change `G_phi` inside T003.

#### T003 acceptance criteria

T003 is ready for research review when:

- the offline coordinate/saturation decomposition is saved from the existing final T002 receipts;
- condition-aware weighting is label-free in the deployment path and covered by frozen-gradient/reset tests;
- generic vs adaptive vs oracle prompt-direction results are reported on the same fixed subset;
- ungated vs soft-gated vs oracle-gated results are reported with the same mechanism metrics;
- exact prompt banks, temperature, masks, commands, environment, run IDs and failures are appended to `CODEX_TO_CHATGPT.md`;
- no learned prompts, predictor training, source/meta-training, detector updates, or ViT³ internals are introduced.

---

## Research review R005 — T003 Stage A/B/C implementation checkpoint (`37e564e` → `0481fde`)

**Assessment: APPROVED CHECKPOINT; T003 remains IN_PROGRESS. Do not launch a duplicate run and do not tune while the fixed 200-image experiment is active.**

Stage A provides strong evidence that the T002 failure is genuinely dominated by coordinate cross-talk rather than saturation alone. Across the 1,200 saved T002 observations, the mean fraction of semantic-gradient norm outside the predeclared plausible corruption subspace is 0.8244 and the corresponding energy fraction is 0.7205, while the harmful-step rate remains 52.4%. The saturation strata do not show a monotonic harm pattern strong enough to explain this. This justifies the T003 subspace-gating experiment without yet changing the hard clamp.

The Stage B/C implementation is consistent with R004. `CLIPConditioner.prepare()` computes degradation weights only from the original test image under `no_grad`, the weights/direction/gate are detached for the episode, and the deployable soft path receives neither synthetic family nor annotation. The masks exactly implement the predeclared darkness `{gamma, brightness, tone}`, contrast `{contrast, tone}`, and color-cast `{RGB gains}` subspaces. Oracle family choices remain confined to the analysis driver. The coordinate gate multiplies only the inner update gradient and preserves the raw gradient in diagnostics, which is the correct implementation for separating objective quality from update-subspace effects.

The observed CUDA backward variability is handled correctly by the revised pairing protocol. Do not compare T003 variants against cached T002 gradients or require cross-run bitwise equality. For each image/corruption, compute one fresh `g_det` and share it across all six T003 variants, rerun the generic direction in the same experiment, and make the scientific comparisons paired within that run. Preserve the failed strict-cache smokes and their measured discrepancies as reproducibility evidence rather than widening tolerances. The successful 2-image smoke and 33-test gate are sufficient to proceed, but are not scientific evidence.

### Required completion/reporting for T003

1. Let `20260912-031123-taisp-t003-coco200` finish with the predeclared temperature 0.05, semantic lr 0.1, K=3, fixed 200 IDs, six corruption settings, six variants, and unchanged hard clamp. Do not restart it merely because this review arrives.
2. Report **paired deltas versus the within-run generic variant** for cosine, positive-alignment rate, one-step detector-loss change, 1-step/3-step AP, saturation, and latency. Use image-cluster paired bootstrap intervals where already implemented; do not infer success from separate variant means alone.
3. Report the degradation-state inference itself as an analysis diagnostic: mean/entropy of the three soft weights, family-conditioned mean weights, and top-1 family identification rate against the known synthetic family. This synthetic family is permitted only for analysis. The key distinction is whether oracle prompt/gate gains are unavailable because the CLIP condition classifier fails, or because the underlying global CLIP direction remains poor even with correct family information.
4. For every gated comparison, report both the **raw semantic gradient** and the **effective gated update gradient**. Alignment/Taylor claims for the optimization step must use the effective gradient, while raw-gradient energy/cross-talk analysis should remain separately labeled.
5. Keep the decision rule unchanged: do not start T004/meta-training until the full paired report is committed. If oracle-family prompt/gating is also weak, the next task should move away from final global CLIP supervision rather than learning its parameters.

No implementation-code change is required by this review. T003 is not accepted until the full 200-image report and receipts are committed to `CODEX_TO_CHATGPT.md`.

---

## Research review R006 — T003 final acceptance (`8d8913e`, `724c04f`, report `e99ef82`)

**Assessment: ACCEPTED AS A DIAGNOSTIC RESULT. T003 is CLOSED. Do not start meta-training.**

The completed fixed-subset experiment satisfies R004/R005: 200 identical COCO-val images, six controlled settings, six variants, 7,200 paired observations, fresh within-run detector gradients shared across variants, unchanged hard clamp/lr/K, and no learned prompts, predictor training, detector updates or ViT³ internals. The reporting now separates raw semantic gradients from effective gated update gradients and uses paired within-run comparisons, so no implementation correction is required before the next research task.

The deployable condition-aware hypothesis is not supported at the current representation level. The three-way degradation classifier is too diffuse: top-1 family accuracy is 45.50% on the balanced synthetic settings, normalized entropy stays around 98% of maximum, and the image-conditioned soft direction gives essentially no mean-cosine improvement over generic (`0.04530` vs `0.04531`, paired delta approximately zero). Soft coordinate gating raises mean cosine only to `0.05127`, with a paired CI crossing zero; it also shrinks the update norm to about 33% of the raw gradient and reduces saturation, so any apparent stability cannot be interpreted as a better semantic direction.

Correct family information does reveal a **limited but real mechanism signal**. Oracle direction + oracle gate reaches mean cosine `0.09368`, positive alignment `62.33%`, and beneficial one-step detector-loss updates `50.58%`. Relative to within-run generic, the paired mean-cosine gain is `+0.04837` with a positive 95% interval, positive-alignment rate improves by `+8.50` percentage points, beneficial-step rate by `+3.92` points, and mean observed one-step detector-loss change improves by about `-0.00253`. This confirms that degradation identity and ISP subspace matter, but the effect is modest rather than sufficient.

Most importantly, downstream AP remains heterogeneous even with privileged family information. Oracle-both helps some cases (for example gamma-s2) while hurting others (gamma-s1, contrast-s1, color-cast-s1), and both color-cast severities remain problematic. Therefore the T003 decision rule resolves to: **better degradation identification alone is not enough; final global CLIP supervision remains too weak/heterogeneous to justify meta-learning it.** The Taylor diagnostic also remains weak after oracle conditioning, so gradient cosine should continue to be treated as a mechanism diagnostic rather than a reliable finite-step performance predictor.

Saturation remains secondary. Gating clearly reduces gamma saturation, but contrast conditions have zero saturation and still show mixed detector/AP behavior. Do not change the hard clamp yet; first test whether the semantic representation itself is the bottleneck.

---

## T004 — Spatial / region-aware semantic supervision feasibility study

**Status: TODO. This is a representation-diagnosis task, not meta-training and not spatially varying ISP yet.**

### Scientific question

T002/T003 used the final global CLIP image embedding. That pooling may discard exactly the local object evidence a detector needs. Test the following hypothesis before learning prompts or making the ISP spatially varying:

> **A global ISP state can receive better test-time gradients if semantic restoration is measured on spatial/region-aware CLIP features rather than only on the final global image embedding.**

Keep the eight-dimensional global `phi` unchanged in T004. This isolates the supervisory representation. If spatial supervision succeeds, a later task may consider spatially varying ISP states; do not conflate those two changes now.

### Stage A — Frozen spatial CLIP feature interface

Extend the existing pinned CLIP ViT-B/32 wrapper to expose differentiable patch-token features while keeping every CLIP parameter frozen.

1. Use the existing parity-correct differentiable preprocessing.
2. At minimum expose last-layer patch tokens after the model's final normalization; project each token with the same visual projection used by the global embedding so the token dimensionality matches the text space. Clearly document that patch-wise text alignment is a diagnostic use of the frozen representation, not a pretrained guarantee.
3. Optionally expose one predeclared intermediate transformer block if straightforward, but do not search layers on the 200-image evaluation subset. If an intermediate layer is used, declare it before the full run and apply a fixed projection/normalization rule.
4. Gradients must flow from every local semantic loss through the enhanced image to `phi`; CLIP weights/buffers remain unchanged.
5. Add tests for shape/patch-grid geometry, frozen model state, finite nonzero `phi` gradients, deterministic repeated episodes, and odd/rectangular image preprocessing.

### Stage B — Predeclare three self-supervised objectives

Use the same generic positive/negative prompt banks from T002 so only the visual representation changes.

1. **Global-final baseline:** the accepted T003 generic objective, rerun within T004.
2. **Uniform patch-direction objective:** for patch token `p`, compute the enhanced-minus-original token displacement projected onto the same text restoration direction, then average over patches. The original-image patch features are detached references.
3. **Detector-region-weighted patch objective:** run the frozen detector once on the original corrupted image in inference mode, detach its predicted boxes/scores, map them to the CLIP patch grid, and use them only as fixed nonnegative patch weights. No annotation, corruption ID, detector gradient, or adapted-image prediction may determine these weights. Predeclare score threshold/top-k and a uniform fallback when no valid region exists.

Keep feature consistency weight zero. Do not introduce learned prompts, a predictor, source/meta-training, smooth clamp, or detector parameter updates.

### Stage C — Separate representation quality from condition-identification quality

The T003 soft degradation classifier is known to be diffuse, so do not make it the main T004 variable.

For each visual objective above, run:

- a deployable **generic text direction**;
- an **oracle family-selected text direction** as analysis-only diagnostic.

Do not use oracle coordinate masks in the primary representation comparison; the goal is to ask whether local visual features improve the semantic gradient itself. A small oracle-mask diagnostic may be reported separately if already inexpensive.

At the initial `phi0`, compute the same frozen-detector oracle gradient for analysis and report alignment of each semantic gradient with it. In addition to the fixed-lr one-step result, add a **direction-only norm-matched diagnostic**: rescale each local semantic gradient to the norm of the contemporaneous global-baseline semantic gradient before taking one analysis step. This uses no labels and isolates whether a result is caused by direction versus gradient magnitude. Keep the ordinary fixed-lr path as the primary deployable comparison.

### Stage D — Fixed-subset experiment

Use exactly the same 200 image IDs, six corruption settings, CLIP/detector weights, `phi0=0`, hard clamp, and primary semantic lr/K as T003 unless a software stability issue forces a separately documented pre-run change. No outcome-driven tuning.

Report for every variant and family/severity:

- raw semantic-gradient norm and per-coordinate energy;
- cosine / positive alignment versus fresh shared `g_det`;
- one-step detector-loss benefit rate and observed loss delta;
- norm-matched one-step diagnostic;
- 1-step / 3-step fixed-subset AP;
- semantic-loss decrease, saturation, parameter trajectories, latency/memory;
- for patch variants, object-weighted versus background patch contribution statistics and the effective number/fraction of patches receiving weight.

Use fresh within-run `g_det` and paired comparisons exactly as in R005/T003; do not compare cached detector gradients across runs.

### T004 decision rule

- If **patch-local supervision improves even with generic direction**, global pooling was a major bottleneck and T005 can study stronger local weighting and/or a spatially varying ISP state.
- If local supervision improves **only with oracle family direction**, the visual representation is useful but degradation-state inference remains a separate bottleneck; T005 should improve condition inference before meta-learning.
- If detector-region weighting materially beats uniform patches, task-relevant spatial selection is important and should become part of the final method story.
- If even **oracle-family local supervision** remains weak/mixed, stop investing in CLIP directional restoration. The next task should compare a different self-supervised signal (e.g. frozen self-distillation/MAE-style or detector-consistency signal) rather than meta-learning the current CLIP loss.
- Do not start meta-training or spatially varying ISP automatically from a smoke result; require the full paired fixed-subset evidence.

### T004 acceptance criteria

T004 is ready for review when the spatial feature path and region-weight path are tested/frozen/label-free, the exact predeclared variant definitions are recorded before the full run, the same 200-image paired study is complete, raw receipts and paired statistics are saved, and `CODEX_TO_CHATGPT.md` reports all positive and negative families plus exact commits/run IDs/environment/failures.

---

## Research review R007 — T004 final acceptance (`4817825`, `8cd0eb0`, report `36823868`)

**Assessment: ACCEPTED AS A NEGATIVE/DIAGNOSTIC RESULT. T004 is CLOSED. Stop investing in CLIP directional restoration for now; do not start meta-training or spatially varying ISP.**

The fixed 200-image/six-corruption study satisfies R006/T004: six primary variants, 7,200 paired observations, fresh within-run `g_det`, frozen CLIP/detector, unchanged global 8D ISP/hard clamp/lr/K, label-free detached region selection, norm-matched diagnostics, and all positive/negative families reported. The A6000 gate passed 42 real-model/regression tests before the full run, and no sample/result-driven tuning or filtering was introduced.

The representation hypothesis is not supported. Global-generic mean cosine is `0.04531`; uniform-patch and detector-region generic supervision fall to `0.01529` and `0.02003`. Their paired cosine changes versus global are negative with intervals crossing zero, and norm-matched beneficial-step gains are only `+1.17` and `+0.83` percentage points with intervals crossing zero. Region weighting also does not materially beat uniform patches: generic region-minus-uniform cosine is `+0.00474` and beneficial-step gain `+0.17` points, both inconclusive. Thus final global pooling is not the dominant bottleneck under this tested CLIP readout.

The apparent oracle signal must not be misread as a spatial-feature success. Norm-matched region-oracle reaches a `+4.33` point beneficial-step gain versus **global-generic**, but when text choice is held fixed the same region-oracle versus global-oracle gain is only `+1.58` points with a confidence interval crossing zero, while its cosine is actually lower by `-0.0635` with a negative interval. This indicates that the useful part came mainly from privileged family text selection, not from last-layer patch/region representation. AP remains strongly family-dependent and mixed. The detector regions cover meaningful support (about 24 effective patches on average with <1% uniform fallback), so the negative result cannot be dismissed as a trivial no-region failure.

**Research conclusion:** T002–T004 now jointly indicate that CLIP's directional “make the image look clearer/natural” objective is the bottleneck, not merely prompt aggregation, ISP coordinate breadth, global pooling, or simple detector-region weighting. Do not meta-learn this loss and do not introduce spatially varying ISP yet. The next falsifiable step is to replace the self-supervised signal while keeping the image-formation action space fixed.

---

## T005 — Detector-native self-supervision feasibility study

**Status: TODO. Signal-screening task only; no meta-training, no learned ISP predictor, no spatially varying ISP.**

### Scientific question

Test whether a label-free objective built from the **frozen downstream detector's own stable predictions** produces a more task-aligned ISP gradient than CLIP directional restoration:

> **Instead of asking a VLM whether the image looks “clear,” can the test image adapt its ISP state so that the frozen detector becomes confident and view-consistent on its own stable object hypotheses?**

Keep the same global 8D `phi`, identity initialization, hard clamp, detector weights, COCO-200 IDs and six controlled corruptions. CLIP remains only as the within-run historical baseline; it must not contribute to the detector-native objective.

### Stage A — Frozen differentiable detector-signal adapter

Use the same Faster R-CNN as T002–T004 and keep every parameter/buffer frozen.

1. At episode start, run **no-grad inference on the original corrupted image** and its horizontal flip. Use score threshold `0.5`, descending top-20 per view.
2. Map flipped boxes back to original coordinates and form a detached stable set by matching detections with the same predicted foreground class and IoU `>=0.5`. Keep deterministic one-to-one highest-IoU matching and record the exact rule before the full run.
3. Stable boxes/classes/scores are fixed for the entire episode. No adapted-image prediction, annotation, corruption ID, or oracle gradient may change the support.
4. Expose a differentiable path that evaluates Faster R-CNN ROI class logits on **fixed boxes** for the enhanced base view and the enhanced horizontal-flip view. Discrete proposal selection/NMS must stay outside the gradient path; gradients must reach pixels/`phi`, never detector weights.
5. If no valid support exists, use a safe **no-update fallback** rather than inventing pseudo-labels. Report fallback rate; do not silently drop those images.

Add tests for flip-box geometry, deterministic matching, frozen detector state, no annotation argument in the deployable signal, finite/nonzero `phi` gradients when support exists, exact no-update fallback, and episode reset.

### Stage B — Predeclare detector-native objectives

Compare these objectives without tuning on the 200-image evaluation subset:

1. **CLIP-global generic baseline:** contemporaneous T004 global-final objective, rerun within T005 for pairing.
2. **Detector pseudo-confidence (`det_pseudo`):** from the original base-view inference only, take score>=0.5/top-20 detached foreground boxes/classes. On the enhanced base view, minimize score-weighted cross-entropy to each detached pseudo-class at its fixed box.
3. **Stable pseudo-confidence (`det_stable`):** use only the base/flip stable matched set and minimize the equally weighted average of base-view and flip-view cross-entropy to the detached consensus class.
4. **Stable confidence + view consistency (`det_stable_js`):** `det_stable` plus Jensen-Shannon divergence between the base/flip class distributions for each matched fixed box. Set the JS coefficient to `1.0` before the smoke and do not tune it on the reported subset.

Use the detector's full ROI class logits including background for softmax/CE/JS, while pseudo targets are foreground classes selected before adaptation. Weight objects only by detached original confidence; normalize weights to sum to one per image. Do not add box-regression pseudo-loss in T005.

### Stage C — Mechanism screen and scale control

For every image/corruption, compute one fresh annotated oracle `g_det` **for analysis only** and share it across all objectives. Report initial self-gradient norm, cosine/positive alignment, per-coordinate energy, first-order dot-product prediction, and observed one-step detector-loss change.

Because detector-native loss scales may differ greatly from CLIP, report both:

- the ordinary fixed raw-phi step with the existing `lr=0.1`;
- a **norm-matched one-step diagnostic** rescaling each detector-native gradient to the contemporaneous CLIP-global gradient norm before the step.

Norm matching is analysis-only and uses no labels. Do not infer superiority from a smaller gradient norm or lower saturation alone. Preserve hard-clamp saturation diagnostics.

### Stage D — Fixed-subset behavior and safety controls

Run the same 200 IDs and six corruption settings with `phi0=0`, K=1/3 and the unchanged primary raw-phi lr. Report paired image-cluster bootstrap intervals versus CLIP-global for cosine, positive alignment, beneficial detector-loss step rate and mean loss delta, plus fixed-subset AP1/AP3 for every family/severity.

Also run the same objectives on the **clean 200-image subset** as a safety control and report clean AP before/after plus `||delta phi||`; a useful test-time signal should not require aggressive changes on already clean inputs.

For detector-native variants report support diagnostics: number of pseudo/stable objects, confidence distribution, base/flip match rate, no-update fallback rate, and outcomes stratified by support size. Do not exclude fallback cases from aggregate results.

### Decision rule

- If `det_stable`/`det_stable_js` materially improves paired alignment **and** beneficial-step/AP behavior over CLIP without damaging clean images, T006 may refine this detector-native objective and then revisit meta-learning or a learned initialization.
- If `det_pseudo` helps but stable-view filtering/JS does not, the key mechanism is detector confidence rather than invariance; simplify rather than adding more consistency machinery.
- If alignment improves but AP does not, treat it as detector-loss gaming/confirmation bias; do **not** meta-learn it. A later task should test cross-detector transfer or another self-supervised representation.
- If all detector-native signals remain weak/mixed, stop iterating on hand-designed self-losses and test a genuinely independent self-supervised restoration/representation signal (e.g. frozen DINO/MAE-style) before any meta-TTT.

### T005 acceptance criteria

T005 is ready for review when the fixed-ROI differentiable detector signal is tested and label-free, the exact support/matching/objective rules are recorded before the full run, the same paired corrupted study plus clean safety control is complete, norm-matched diagnostics and support/fallback statistics are saved, and `CODEX_TO_CHATGPT.md` reports exact commits/run IDs/environment/failures and all positive/negative families. Do not start T006 automatically from a smoke result.
