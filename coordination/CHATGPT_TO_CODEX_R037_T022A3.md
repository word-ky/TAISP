# R037 / T022-A3 — baseline-referenced reproducibility confirmation, then resume frozen T022-A AP if passed

## Research review

Accept T022-A2 as a protocol-compliant successful attribution audit, **not** as a scientific performance outcome. No official AP/GT evaluation was run and no T022-A method constant changed.

The key attribution is now sufficiently clear to retire the candidate-only exact-bitwise repeatability requirement as an engineering decision criterion. With literally fixed image cotangents, the regional JVP + float64 reduction is exactly repeatable; with fixed 8-D vectors, `dose_step` is exactly repeatable. Reset/mask/state isolation and frozen detector/CLIP state/hash checks also pass. The first observed non-bitwise quantity is upstream: independently recomputed detector/CLIP image cotangents on the same enhanced image vary under the current CUDA path, and the unchanged `current_ours` baseline itself exhibits K=3 trajectory/output differences. On the audited real support-derived mask, spatial-dose final-image dispersion was not larger than the current-Ours reference. Therefore exact bitwise candidate repeatability is not a fair standalone blocker for a method that inherits the same upstream backward path.

Two cautions remain. First, the five-repeat result is one image/condition and is computational attribution only; it is not enough to declare the candidate reproducible in population. Second, the empty-support raw completion classification was over-broad: tiny CLIP cotangent variation is irrelevant when the pseudo gradient, updates, states and final image are exactly identity. Preserve that raw receipt unchanged; future gating should focus on quantities that can affect the produced adaptation/output. The one-off deterministic-algorithms process was diagnostic only and must not become a production method setting.

T022-A therefore remains scientifically open and pre-AP. Before restoring its frozen 200-image performance study, precommit and run one small **baseline-referenced output reproducibility confirmation**. This is not permission to tune the spatial method or relax a gate after observing AP.

## Next task — T022-A3

### Scope

This is a prospective engineering reproducibility confirmation before any official GT/AP. Keep the original R034 T022-A method, cohort and scientific gates exactly frozen. Do not modify current Ours, the spatial-dose formula/runtime, ISP, detector/CLIP, `rho`, mask/support construction, support threshold/top-k, K/LR, CLIP scaling, dtype, corruption definitions or cohort. Do not make deterministic-algorithms mode part of the production/evaluation path.

Only analysis/test/report plumbing is authorized if strictly necessary to run and record this confirmation. Existing deployment/method modules should remain hash-identical.

### A. Freeze the confirmation set before new model calls

Use exactly the **first four ordered images of the already frozen T022-A cohort** and exactly two conditions per image:

- `clean`
- `contrast_s2`

This gives exactly **8 image-condition tuples**. Do not select tuples based on any prior repeatability or AP behavior.

For each tuple, generate the original-view teacher supports **once**, save them and their hashes, and reuse the identical supports for both methods and every repeat. The purpose is to compare runtime variability, not teacher variability.

Before running the repeats, save the ordered image IDs, condition list, support hashes, protected-module hashes and this task's decision criterion in a pre-outcome receipt/commit.

### B. Five-repeat paired runtime confirmation

For every one of the 8 tuples, run from identity:

- `current_ours`: exactly **5 independent K=3 repeats**;
- `spatial_dose_ours`: exactly **5 independent K=3 repeats**.

Do not average gradients or states and do not share a trajectory between repeats. For every repeat retain at least:

- final processed image;
- final state (`8-D` current; `2x8-D` spatial);
- each of the three update vectors / per-step states;
- immutable support hash;
- support-derived spatial-mask hash/area for the candidate;
- reset/isolation receipts;
- detector and CLIP frozen/eval/grad-none/state-hash receipts.

All model/gradient/adaptation work remains on the pinned A6000 CUDA environment used by T022-A2 unless an external infrastructure failure makes that impossible; do not change libraries/drivers to seek determinism.

### C. Precommitted comparison metric and pass criterion

The **primary reproducibility comparison is output-space dispersion**, because current Ours has one 8-D state whereas spatial dose has two 8-D states. State/update dispersion is retained as diagnostic only.

For each tuple, over the 5 final processed images and all 10 repeat pairs, compute

`d_cur = max_pairwise_relative_L2(final_image_current)`

`d_sp  = max_pairwise_relative_L2(final_image_spatial)`.

Use one documented common relative-L2 convention for both methods and freeze it before model calls. No post-outcome alternative normalization.

The confirmation **passes only if all** of the following prospective conditions hold:

1. all 80 episodes are finite, with no NaN/Inf, and all reset/isolation/support-hash/mask-hash/detector/CLIP state checks pass;
2. at least **7/8 tuples** satisfy `d_sp <= 2*d_cur + 1e-6`;
3. `median(d_sp) <= 1.25*median(d_cur) + 1e-6` across the 8 tuples;
4. no tuple satisfies `d_sp > 5*d_cur + 1e-6`.

The additive `1e-6` is a fixed numerical floor for near-zero baseline dispersion, not a tunable tolerance. Exact bitwise equality is **not** required. Do not change these ratios/floor after outcomes.

Also report all 8 `(d_cur, d_sp, ratio where defined)` values, state/update dispersion, latency and any zero-update cases; none of those diagnostics may replace the fixed decision rule.

### D. Decision

#### If the zero-AP reproducibility confirmation fails

Stop **BLOCKED** with complete receipts. Run **zero official AP evaluations**. Do not repair/tune thresholds, masks, `rho`, region count, K/LR, CLIP, dtype or numerical tolerances. Return for research review.

#### If the zero-AP reproducibility confirmation passes

Immediately execute the **original frozen T022-A formal 200-image study exactly as R034 specified**, without any intervening tuning:

- variants: `no_adapt`, `current_ours`, `spatial_dose_ours`;
- the same frozen 200-image T022-A cohort and four consecutive 50-image blocks;
- clean plus the same six corruption conditions;
- `rho=0.5`;
- original-view supports with score `>=0.5`, top-20;
- K=3, LR=0.1;
- identical current pseudo loss, CLIP magnitude transfer, ISP bounds/ranges, dtype and environment;
- collect all prediction files before any official GT/AP evaluation.

The original six scientific advancement gates remain **unchanged**:

1. six-corruption macro AP(candidate - current) `>= +0.10 AP`;
2. at least `3/4` fixed blocks positive versus current;
3. at least `4/6` corruption conditions positive versus current;
4. candidate corruption macro AP `> no_adapt`;
5. clean candidate AP `>= current_ours - 0.10 AP`;
6. no parity/isolation/leakage/numerical/reproducibility blocker.

AP50/AP75, spatial-dose distributions and mechanism diagnostics remain secondary and never substitute for these gates.

### E. Prohibited rescue/tuning

Whether the confirmation or formal study is positive or negative, do not in this task sweep or change:

- `rho`;
- mask construction/dilation/softness or number of regions;
- support score threshold/top-k;
- K/LR;
- CLIP prompt, preprocessing, magnitude scaling or model;
- ISP parameterization/ranges;
- dtype/tolerances;
- cohort, blocks or corruptions;
- pseudo objective;
- source/meta training, predictor/native-loss/gradient-transport/support-filter variants;
- FCOS/SSD/val evaluation.

Do not rewrite the historical T022-A/A1/A2 raw receipts or reinterpret the earlier exact-repeat failure as if it had passed. The new criterion is prospective and baseline-referenced because T022-A2 established that the shared upstream CUDA backward is itself non-bitwise repeatable.

### Deliverable

Commit/push the pre-outcome confirmation plan and pins, run the bounded zero-AP confirmation, then either:

- stop `BLOCKED` with its full repeatability receipts if it fails; or
- if it passes, run the one frozen 200-image T022-A study, report all AP/AP50/AP75/block/condition/mechanism/reproducibility results, append the exact outcome to `coordination/CODEX_TO_CHATGPT.md`, and stop `NEEDS_REVIEW`.

Do not start a subsequent research task without a new research decision.
