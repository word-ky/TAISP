# R036 / T022-A2 — attribute runtime repeatability blocker before any AP

## Research review R036 — T022-A1 (`8948d4c` → `c713150` → `7001d0c` → `86c1d59`)

**Assessment: ACCEPT the R035 execution as protocol-compliant. T022-A remains scientifically OPEN and pre-AP BLOCKED. The parity repair is now fully validated; the remaining issue is runtime repeatability, not evidence that spatial dose helps or hurts AP. Do not relax the committed repeatability check and do not run the 200-image AP study yet.**

Codex followed `coordination/PROTOCOL.md` and R035 correctly. The only authorized parity-harness repair was applied; current Ours, `taisp/tta/spatial_dose.py`, `taisp/isp/spatial.py`, detector/CLIP/ISP behavior, `rho=0.5`, masks, support rules, K/LR, dtype, tolerances and the frozen cohort were not changed. The exact 28-record A6000 parity rerun passed all records and all CLIP isolation/hash checks. The maximum relative-L2 errors remained inside the predeclared `1e-5` envelope: pseudo `4.772744e-6`, CLIP `4.281781e-7`, and mean update `4.365902e-6`; focused/full regression also passed.

The subsequent zero-AP K=3 smoke correctly stopped on the precommitted exact-repeat edge check. Repeating the same empty-mask/nonempty-support episode produced final-state max-abs difference `1.4580879e-4` and relative L2 `4.51558e-3`; the saved pseudo-gradient relative difference grew from `8.69e-8` at step 0 to `1.86e-2` at step 3. The full-mask episode differed only by `2.79e-9` max-abs (`7.93e-8` relative L2), while empty-support was exactly repeatable. All other reset, inactive-region, finite-state, support, dose-bound, model-freeze and state-hash checks passed. No AP was evaluated.

This does **not** justify calling the candidate unstable or failed yet. The empty/full masks with nonempty supports are deliberate degenerate diagnostics rather than normal support-derived masks, and the fixed-ROI pseudo gradient passes through the frozen Faster R-CNN backbone and ROI pooling on CUDA. Repeated GPU backward may therefore contribute small numerical variation, but that root cause has not been demonstrated. Conversely, the larger empty-mask divergence could also reveal amplification inside the spatial path. We need to localize the first non-repeatable quantity before changing any reproducibility criterion.

---

## T022-A2 — Bounded repeatability attribution audit

**Status: TODO. Analysis-only, target roughly one hour. No scientific tuning and zero AP evaluations.**

### A. Freeze provenance and do not change the method

Reuse the exact R035 release/state, image `160585`, condition `gamma_s1`, original frozen supports, source/CLIP weights, ISP, `rho=0.5`, K=3/LR=0.1, and the saved T022-A1 edge trajectories. Add only an analysis script/tests/receipts needed to measure repeatability. Do **not** modify:

- `taisp/tta/spatial_dose.py` or `taisp/isp/spatial.py`;
- current Ours or its loss;
- detector/CLIP/ISP modules or preprocessing;
- support/mask construction, `rho`, K/LR, CLIP magnitude rule, dtype, tolerances or cohort;
- the existing failed exact-repeat receipt/check.

Pin all relevant hashes before the new diagnostic. No new image cohort is needed.

### B. Locate whether non-repeatability first appears in the image cotangents

Use fixed enhanced tensors, not an updating episode. At minimum test identity plus the saved empty-mask trajectory states for steps 1–3. For each fixed tensor, independently recompute **five times**:

1. the detector pseudo loss value and image cotangent `cp = dL_pseudo/dy`;
2. the CLIP loss value and image cotangent `cc = dL_clip/dy`.

For every repeat pair save exact equality, max-absolute difference, relative L2 and cosine for the scalar loss and cotangent. Keep detector/CLIP frozen/eval/hash checks active. Do not average the cotangents and do not change the runtime from these observations.

In a **separate diagnostic process only**, enable `torch.use_deterministic_algorithms(True, warn_only=False)` before one representative detector/CLIP cotangent computation. Record whether PyTorch completes or raises a nondeterministic-operation error and preserve the exact operator/error text. This is attribution only; deterministic mode is not authorized for the method or later AP run in T022-A2.

### C. Test the spatial JVP/reduction independently of detector/CLIP backward

Take one saved `cp` and one saved `cc` as fixed tensors. At the same saved states, repeat `regional_gradients(...)` five times for:

- the real support-derived mask;
- the all-zero mask;
- the all-one mask.

Save all 8-D object/background pseudo and CLIP vectors and the same exact/max-abs/relative-L2/cosine comparisons. This stage must make clear whether the common-Jacobian/JVP plus float64 masked reductions add any variability when their inputs are literally fixed.

Then freeze one `(go, gb, co, cb)` tuple and call the pure `dose_step(...)` at least 20 times. Verify/report exact equality of `c`, both multipliers, `u`, `v`, and the two 8-D deltas. A failure here is a direct implementation blocker; do not repair it inside this task.

### D. Establish the inherited runtime repeatability reference

Without AP or ground truth, run five independent K=3 repeats from zero state using the same image/supports for:

1. **current Ours**;
2. `spatial_dose_ours` with the **real support-derived mask**;
3. `spatial_dose_ours` with the all-zero mask and the same nonempty supports;
4. `spatial_dose_ours` with the all-one mask and the same nonempty supports;
5. empty-support spatial dose as the exact-zero control.

For each method/case report pairwise dispersion of every step's pseudo/CLIP image cotangent, common 8-D gradients, final state, and final processed image: exact equality, max absolute error, relative L2 and cosine where defined. For current Ours retain its own 8-D state/image repeatability as the **baseline reference**; do not invent a post-outcome pass tolerance from it in this task. For the real spatial mask also save the mask hash/area and verify it is unchanged across repeats.

Do not evaluate detector AP, labels, or any official metric. This audit is about computational repeatability only.

### E. Interpretation and stop rule

Use the following attribution language without promoting or closing the method:

- If independently recomputed `cp`/`cc` vary while fixed-cotangent `regional_gradients` and fixed-input `dose_step` are exact/repeatable, attribute the first observed variability to the upstream CUDA forward/backward path. Report whether current Ours exhibits the same phenomenon and its scale; do **not** yet relax the spatial exact-repeat gate.
- If fixed-cotangent `regional_gradients` or fixed-input `dose_step` varies, classify this as a spatial implementation/numerical blocker and stop. Do not patch it in T022-A2.
- If variability is strongly amplified only by the synthetic empty/full-mask episodes while the real support-derived mask tracks the current-Ours repeatability scale, report an **edge-case amplification** diagnosis. This still requires research review before any gate change or AP run.
- If the real support-derived spatial episode shows materially larger divergence than current Ours even though the lower-level JVP/dose pieces are repeatable, report that trajectory amplification explicitly and remain BLOCKED.
- Empty support must remain exactly zero/repeatable; otherwise report a hard reset/control failure.

Do not define a new tolerance after seeing these outcomes. The purpose is to obtain the evidence needed for the next research decision: either preserve the blocker, or precommit a baseline-referenced reproducibility criterion in a later task before resuming T022-A.

### F. Tests and deliverables

Add only analysis/tests/reporting code. Run focused tests and the full regression suite. Save raw five-repeat tensors/metrics, deterministic-algorithm diagnostic, model/state hashes and commands. Append the exact result to `coordination/CODEX_TO_CHATGPT.md`, commit/push receipts, and stop for research review.

**Do not run the 200-image formal T022-A experiment, official COCO AP, a new cohort, `rho`/mask/threshold/K/LR tuning, deterministic-mode performance runs, independent regional directions, objective redesign, source/meta-training, FCOS/SSD evaluation, or deployment redesign during T022-A2.**