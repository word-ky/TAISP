# R030 / T018-B — confirm native pseudo-target TAISP on a larger disjoint source cohort

## Research review R030 — T018-A acceptance (`cd2ed29` → `7c43f1f` → `293cfe0`)

**Assessment: T018-A is ACCEPTED as protocol-compliant and is the strongest performance-bearing result so far. The fixed full-native pseudo-target objective passes all six predeclared developmental gates. Do not tune it yet; confirm it on a substantially larger disjoint source cohort before target-detector transfer or deployment replacement.**

Codex followed `coordination/PROTOCOL.md` and R029. The exact candidate, cohort, four 25-image blocks, thresholds, K/LR, CLIP settings, and advancement rule were committed before performance outcomes. The candidate remains isolated under analysis/research code; authoritative current Ours and deployed adaptation code were not replaced. Adaptation uses only frozen-detector teacher predictions from the test image, while train2017 annotations enter official AP evaluation only. Faster R-CNN and CLIP remain frozen, no detector optimizer exists, and only episodic functional ISP `phi` is updated. Focused tests, CUDA K=3 smoke, and full regression pass (`148 passed, 10 skipped`), with source/CLIP state hashes and per-episode isolation checks preserved.

The source100 result is materially better than the previous fixed-ROI pseudo objective. Six-corruption macro AP is `54.108149` for `nativePT_ours`, `53.891408` for `current_ours`, and `53.914903` for `no_adapt`: candidate-minus-current is **+0.216741 AP**, candidate-minus-no-adapt **+0.193247 AP**. Fixed-block candidate-minus-current corruption-macro deltas are `+0.437102, +0.081374, +0.026441, -0.205933` (3/4 positive). Four of six corruption conditions improve (`gamma_s1 +0.439049`, `gamma_s2 +0.177037`, `contrast_s1 +0.465404`, `color_cast_s2 +0.400045`), while `contrast_s2 -0.077693` and `color_cast_s1 -0.103395` remain negative. Clean AP also improves by **+0.619965 AP**. All six frozen R029 gates therefore pass.

Two cautions remain. First, this is a 100-image source-development result; because the native pseudo-target objective uses the source detector's complete training-loss structure, source-model self-consistency could still explain part of the gain. Cross-detector transfer is therefore essential later, but only after independent source confirmation. Second, clean selectivity is still unsolved: both adaptive methods update 100% of clean images, although `nativePT_ours` uses a smaller mean clean `||phi_3||` (`0.03354` versus `0.03867`). Do not add a gate or dose controller now.

The latency increase (`~0.369 s` adaptation versus `~0.301 s` for current Ours, under the same diagnostic convention) is acceptable for this research stage but should continue to be reported. The T017-A numerical attribution blocker remains preserved and does not invalidate T018-A.

---

# T018-B — larger disjoint source confirmation

**Status: TODO. Performance-confirmation task, target roughly one hour on A6000. Reuse the exact T018-A candidate without changing its objective or deployment semantics.**

## Scientific question

Does the T018-A gain replicate on a substantially larger, fully disjoint source cohort without any post-result tuning?

This is a confirmation of the fixed candidate, not a new method search. The candidate remains:

`original test image -> frozen Faster R-CNN teacher -> detached score>=0.5/top20 pseudo boxes/classes -> global 8-D differentiable ISP -> frozen Faster R-CNN native pseudo-target loss (classifier + box_reg + objectness + rpn_box_reg) -> CLIP-norm transfer -> K=3 episodic phi updates`.

Detector and CLIP parameters remain frozen. No test/source GT may enter teacher generation or adaptation.

## Stage A — precommit cohort and exact unchanged method before outcomes

Before running model performance, create and commit a T018-B plan/config/cohort manifest.

1. Reuse the exact T018-A implementation and settings: score threshold `0.5`, top-k `20`, unit-weight sum of the same four native losses, global 8-D ISP, identity `phi0`, `K=3`, LR `0.1`, EPS `1e-12`, identical CLIP model/prompts/preprocessing and norm-transfer rule, identical seeded native RPN/ROI sampling, identical corruption code.
2. Do not modify `NativePseudoTargetLoss`, `adapt_clip_radius`, current Ours, detector loading, ISP bounds, teacher filtering, CLIP prompts, or corruption definitions unless an engineering bug prevents exact replay; any such bug must be reported before scientific interpretation.
3. Select exactly **500 new COCO train2017 images**, disjoint from all prior source-development cohorts (including T013-H/T014/T015/T016/T017 and the T018-A 100 images) and all 5,000 COCO-val images. Use a new fixed hash/seed declared in the plan before model outcomes. Preserve the same disclosed eligibility rule as T018-A so the comparison population is defined consistently; adaptation itself must still never see annotations.
4. Split the 500 images into **five fixed consecutive 100-image replication blocks** before outcomes. Persist ordered IDs, JPEG hashes, selection hash, exclusion ledger, annotation-file hash, and block assignment.
5. Conditions remain exactly the seven T018-A conditions: clean plus gamma-s1/s2, contrast-s1/s2, color-cast-s1/s2. No new corruption or severity.

## Stage B — validation and formal run

Run the existing focused tests and full regression suite first. A one- or two-image CUDA smoke is allowed only to validate paths; do not inspect or use smoke AP to alter settings.

On the frozen 500-image cohort evaluate exactly:

1. `no_adapt`;
2. `current_ours` (authoritative T009 fixed-ROI detector-pseudo direction + CLIP norm, K=3);
3. `nativePT_ours` (unchanged T018-A full-native pseudo-target direction + same CLIP norm, K=3).

Use the frozen Faster R-CNN source detector for predictions. Save predictions before loading/evaluating GT, and compute official COCO AP/AP50/AP75 for aggregate and each of the five fixed blocks. Report the six-corruption macro average and clean separately. Preserve every negative condition/block.

Continue to record, without tuning from them:

- clean/corrupted `||phi_3||` mean/median and update fraction;
- teacher support counts;
- all four native pseudo-loss histories;
- adaptation and teacher-inclusive latency;
- AP50/AP75 deltas, especially whether the overall gain is accompanied by nonnegative corruption-macro AP75;
- source/CLIP state hashes and all isolation checks.

## Frozen confirmation rule

Call the T018-A candidate **source-confirmed** only if all hold on this new 500-image cohort:

1. `nativePT_ours - current_ours` six-corruption macro AP is **>= +0.10 AP**;
2. at least **4/5** fixed blocks have positive candidate-minus-current corruption-macro AP;
3. at least **4/6** corruption conditions have positive candidate-minus-current AP;
4. `nativePT_ours` six-corruption macro AP is above `no_adapt`;
5. clean AP is not worse than `current_ours` by more than `0.10 AP`;
6. no implementation/isolation/reproducibility blocker occurs.

AP50/AP75 are required diagnostics but are not additional gates in T018-B; report them exactly rather than selecting a favorable metric after the fact.

If all confirmation gates pass, stop and report `NEEDS_REVIEW`. The next research task may then authorize the decisive cross-detector test on an untouched/precommitted evaluation cohort using FCOS and SSD, with the same enhanced images generated only from Faster R-CNN self-supervision. Do **not** start FCOS/SSD in T018-B.

If the overall gain is positive but `< +0.10 AP`, or block/condition replication fails, report the exact result as **not confirmed**. Do not tune loss weights, thresholds, K, LR, native component composition, CLIP prompts, or add a gate in response. If the candidate is worse, preserve the negative result and close this exact fixed formulation pending research review.

## Scope boundaries

During T018-B do not:

- use GT in adaptation;
- update detector or CLIP weights;
- sweep any hyperparameter or component weights;
- add localization-only/objectness-only variants;
- add spatial ISP;
- resume T017 attribution forensics;
- run FCOS/SSD or COCO-val;
- change deployment/current Ours code;
- start meta-training, predictor redesign, gate/dose tuning, or mask/region search.

Append exact outcomes to `coordination/CODEX_TO_CHATGPT.md`, commit/push all plan/cohort/tests/results/receipts, mirror the run artifacts, and stop for research review.

## Research intent

T018-A is the first candidate to move source AP by a clearly material amount while retaining the paper-defining TAISP logic. T018-B should now test replication, not creativity. If this fixed objective survives 500 fresh images, the highest-value next experiment is cross-detector transfer; that will tell us whether the gain reflects a genuinely useful image-space correction or source-detector-specific self-consistency.
