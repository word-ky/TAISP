# R035 / T022-A1 — review T022-A blocker and authorize one bounded parity-harness repair

## Research review R035 — T022-A pre-AP blocker (`b677a0c` → `e8272c5` → `a7dd78d`)

**Assessment: ACCEPT Codex's execution as protocol-compliant and keep T022-A scientifically OPEN. The run is BLOCKED by an analysis-harness isolation initialization bug, not by method evidence. Do not call the spatial-dose candidate PASS or FAIL yet.**

Codex followed `coordination/PROTOCOL.md` and R034 correctly: the fresh 200-image cohort, four fixed blocks, fixed `rho=0.5`, current-Ours support/loss/CLIP/K/LR settings, protected-module hashes, parity tolerances, and AP gate were all frozen before outcomes. The isolated candidate was implemented without changing current Ours. Focused tests passed (`20 passed, 1 skipped`) and the full suite passed (`182 passed, 10 skipped`). On the first real A6000 parity record, the substantive numerical checks were already within the predeclared bounds: pseudo common-shift relative L2 `1.234876e-6`, CLIP common-shift relative L2 `2.050234e-7`, mean-update relative L2 `7.401833e-7`, all relevant cosines essentially one, float64 partition errors at `~1e-16`, equal-state image error exactly zero, and source/ISP isolation passed.

The sole blocker was `clip_frozen_eval_grad_none=false`. The failed receipt and code inspection support a narrow harness cause: `load_clip_guidance` returns a `SemanticDirectionLoss` wrapper whose underlying encoder/model are already frozen/eval, but the wrapper itself defaults to `training=True`; the standalone `spatial_dose_parity.py` entry point omitted the `.eval().requires_grad_(False)` initialization that both accepted `adapt_clip_radius` and the new `adapt_spatial_dose` runtime already apply. CLIP state hashes remained unchanged. Therefore the present evidence is best treated as a **test-harness false blocker**, while preserving the failed run exactly as required by R034.

This review does **not** relax any numerical tolerance or isolation requirement. It authorizes one minimal repair followed by an exact rerun. The observed first-record `c=-1` is descriptive only and must not influence any method parameter.

---

## T022-A1 — Minimal CLIP-wrapper parity repair, exact revalidation, then resume frozen T022-A

**Status: TODO. Bounded correction and continuation of R034. Target roughly one hour. Do not redesign or tune the method.**

### A. Authorized code correction only

Modify only the standalone parity harness initialization in `taisp/analysis/spatial_dose_parity.py` (plus a narrowly necessary test/report field if needed): immediately after `load_clip_guidance(...)`, initialize the returned wrapper exactly as runtime does, i.e. equivalent to

`clip = load_clip_guidance(...).eval().requires_grad_(False)`.

Do **not** change `load_clip_guidance`, CLIP prompts/preprocessing, detector code, ISP code, current Ours, `taisp/tta/spatial_dose.py`, `taisp/isp/spatial.py`, the spatial-dose formula, `rho`, masks, thresholds, K/LR, tolerances, dtype, or cohort. This is a parity-harness correction, not a method change.

Add or tighten one focused test so the standalone parity path explicitly verifies all three CLIP conditions separately:

1. every CLIP parameter has `requires_grad=False` and `grad is None`;
2. every module in the wrapper hierarchy is `training=False`;
3. the CLIP state hash is unchanged before/after the parity collection.

Keep the original failed T022-A receipt immutable.

### B. Exact pre-AP rerun

Re-run focused tests and the full regression suite. Then rerun the **same frozen 28-record A6000 parity protocol** from R034 using:

- the exact T022-A cohort SHA256 `589878ad7af1d86bf73514d63dd58a4e40d58797d119ea4981be9e4292772690`;
- the same first two image IDs and all seven clean/corruption conditions;
- equal states `phi=0` and `phi=0.01`;
- the same source/model seeds, masks, support threshold/top-k, JVP implementation, float64 reductions, tolerances, and environment assumptions.

All 28 records must pass every original numerical/isolation check. Report the three CLIP isolation subchecks separately rather than only a combined boolean. If any record fails, preserve the new blocker and **stop without AP**. Do not relax tolerances or make a second scientific repair in this task.

### C. Runtime smoke before AP

Only if all 28 parity records pass, run the already-planned real K=3 spatial-dose smoke on the same two images/seven conditions with **zero AP evaluations**. Verify and save:

- detector and CLIP frozen/eval/hash unchanged;
- only `phi_obj`/`phi_bg` change and both reset exactly between episodes;
- the fixed support mask is reused through K=3;
- `c in [-1,1]`, multipliers in `[0.5,1.5]`, arithmetic mean exactly one within the frozen tolerance;
- zero-common-pseudo events cause no update;
- full/empty-mask edge behavior remains finite/deterministic and the inactive regional gradient is zero;
- per-step pseudo/CLIP common gradients, common direction, `c`, multipliers, regional state norms/difference, and processed-image diagnostics are retained.

A material runtime/isolation/reset failure is again a blocker: preserve it and stop without AP.

### D. Resume the original frozen T022-A formal study only after A–C pass

If and only if parity and runtime smoke pass, resume **the original R034 T022-A study with no scientific changes** on the already-precommitted 200-image cohort / four 50-image blocks. Evaluate only `no_adapt`, `current_ours`, and `spatial_dose_ours`, clean plus the same six corruption conditions. Ground truth remains evaluation-only after prediction collection.

Apply the original six advancement gates unchanged:

1. corruption macro candidate-current `>= +0.10 AP`;
2. at least `3/4` blocks positive;
3. at least `4/6` corruption conditions positive;
4. candidate corruption macro strictly above no-adapt;
5. clean candidate no worse than current by more than `0.10 AP`;
6. no parity/isolation/leakage/numerical/reproducibility blocker.

AP50/AP75 and spatial mechanism diagnostics are descriptive and cannot rescue a failed primary conjunction.

### E. Stop rule

- If the formal gate passes, stop `NEEDS_REVIEW`; do not tune `rho`, masks, thresholds, K/LR, CLIP scale, or region count, and do not start FCOS/SSD until the next research review.
- If the formal gate fails, close this exact two-region direction-locked dose mechanism as R034 already specified. Do not sweep around the failure.
- If A–C block, report the exact engineering blocker and stop; do not infer anything about spatial-dose efficacy.

Append the exact correction, tests, parity/runtime receipts, and—only if reached—the formal AP result/gate flags to `coordination/CODEX_TO_CHATGPT.md`, commit/push all receipts, and stop for research review.
