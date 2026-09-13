# R029 / T018-A — performance-first full native pseudo-target detection TTT

## Research review R029 — T017-A blocker and strategic adjustment

**Assessment: T017-A is BLOCKED numerically, not scientifically negative. Preserve the blocker, but do not spend the next hour on tighter component-attribution forensics. Shift the next bounded task toward performance while preserving the core TAISP/TTT logic.**

T017-A followed R028/PROTOCOL correctly: the plan/hash commit preceded outcomes; the fixed 32-episode cohort, masks, source detector, and common ISP Jacobian construction were preserved; implementation was analysis-only; regression was `144 passed, 10 skipped`; no CLIP/pseudo/optimizer/AP/deployment work occurred. The run stopped on episode 0 exactly as required.

The blocker is numerical reconstruction, not an adverse scientific result. Forward native loss values exactly match the saved A1 reference and direct total-gradient direction is essentially identical (`cos ~ 0.99999997–1.0`), but separate float32 CUDA backward paths produce component-sum/cross-run relative-L2 discrepancies around `2e-4–1e-3`, exceeding R028's frozen `1e-5` parity rule. Therefore localization/confidence attribution is **NOT_REACHED**. Do not reinterpret the partial record or retroactively loosen R028.

The research priority is now adjusted: preserve the paper's main logic—**label-free test-time training of a small image-formation state with a frozen detector**—but allow objective/parameterization/trust-control details to change if they produce materially stronger performance. Continue falsifiable, precommitted experiments, but prefer direct performance-bearing candidates over repeated numerical micro-diagnostics when the core scientific question is already clear.

The immediate candidate is motivated by the largest remaining mismatch in Ours-v1: the current detector-native pseudo objective is essentially fixed-ROI classification confidence, whereas the real detector task includes ROI classification, ROI localization, RPN objectness, and RPN localization. Test one simple objective that preserves the entire TTT story while making the self-supervision structurally closer to detection.

---

# T018-A — Full-native pseudo-target detector loss for TTT-ISP

**Status: TODO. Target roughly one hour. One fixed candidate only; no objective-weight sweep.**

## Core hypothesis

Use the frozen detector on the original test image as a teacher to generate detached pseudo boxes/classes. On the ISP-transformed image, run the same frozen Faster R-CNN through its native training-loss path using those pseudo detections as targets. This yields the complete native pseudo-detection objective

`L_nativePT = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg`.

Only the ISP fast state is updated. Detector parameters remain frozen. No ground-truth label is used by adaptation. The existing CLIP signal remains magnitude-only / trust-radius control, not direction.

This changes the **self-supervised objective**, not the TAISP paradigm:

`test image -> frozen teacher detections -> pseudo targets -> differentiable ISP -> frozen detector native detection loss -> TTT update of phi -> enhanced image -> frozen detector`.

## Stage A — precommit exact candidate and source cohort before outcomes

Before any performance result, create a T018-A plan and commit:

1. Exact teacher support rule: reuse the current detector-pseudo support policy unchanged (`score >= 0.5`, top-20 after the existing ordering/filtering). Use only detached boxes and integer class labels. No score weighting, no soft-label temperature, no threshold search.
2. Exact native objective: unit-weight sum of the four torchvision Faster R-CNN native losses listed above. No loss coefficients, clipping, balancing, coordinate selection, or component dropping.
3. Exact adaptation rule: global 8-D ISP only; identity `phi0`; `K=3`; same ISP bounds, learning rate, epsilon, CLIP model/prompt/preprocessing, and CLIP-norm transfer as the current T009 Ours. At every TTT step,
   `g_native = grad_phi L_nativePT`,
   `g = g_native * ||g_clip|| / (||g_native|| + EPS)`.
   CLIP direction remains discarded.
4. Pseudo targets are generated exactly once from the original condition image before adaptation and are fixed/detached for all K steps. If no support exists, use exact no-update, matching current safety semantics.
5. Detector parameters/state/hash must remain unchanged. Any temporary training-loss mode must restore model state exactly; no detector optimizer exists.
6. Precommit exactly **100 new COCO train2017 images**, disjoint from every prior T013/T014/T015 source cohort and all validation cohorts, using a fixed seed and saved image IDs/JPEG hashes. Split into four fixed 25-image replication blocks before results.
7. Conditions are the existing T009 controlled set only: clean, gamma-s1/s2, contrast-s1/s2, color-cast-s1/s2. No new corruption or severity.

Ground-truth train2017 annotations are allowed **only for evaluation**, never for pseudo-target creation or adaptation.

## Stage B — implementation sanity, not a scientific gate

Implement the candidate in research/analysis code or a clearly isolated experimental loss path. Do not replace the deployed/current Ours path yet.

Required tests before the 100-image run:

- pseudo targets equal the frozen teacher's filtered boxes/labels and are detached;
- all four native loss keys are present and finite;
- detector parameters receive no gradients and state/hash are unchanged before/after adaptation;
- `phi` is the only updated state;
- empty support gives exact identity/no-update;
- K=3 executes on CUDA;
- current Ours remains bit-for-bit/functionally unchanged when the candidate is not selected.

Run focused tests and the full regression suite. A small runtime/smoke run may be used to catch engineering errors, but **do not use smoke performance to alter the objective, threshold, K, LR, or cohort**.

## Stage C — direct performance comparison on the frozen 100-image source cohort

Evaluate exactly three methods on all seven conditions:

1. `no_adapt`;
2. `current_ours` = authoritative T009 detector-pseudo direction + CLIP-norm scale, K=3;
3. `nativePT_ours` = the new full-native pseudo-target direction + the same CLIP-norm scale, K=3.

Use the frozen Faster R-CNN source detector for final predictions. Compute official COCO AP/AP50/AP75 for each condition and each of the four fixed blocks, plus the six-corruption macro average. Also report clean AP and adaptation latency. Preserve all negative conditions/blocks.

Additionally record, for diagnosis only:

- mean/median `||phi3||` on clean/corrupted images;
- fraction of clean images that update;
- per-step native pseudo loss;
- teacher-support counts;
- candidate/current paired per-image prediction-count changes if already available cheaply.

Do not tune from these diagnostics in T018-A.

## Frozen advancement rule — performance matters

Call `nativePT_ours` a **promising performance upgrade** only if all hold:

1. six-corruption macro AP is higher than `current_ours`;
2. candidate-minus-current corruption macro AP is at least **+0.10 AP** on this developmental source cohort;
3. at least `3/4` fixed blocks have positive candidate-minus-current corruption macro AP;
4. at least `4/6` corruption conditions have positive candidate-minus-current AP;
5. candidate corruption macro AP is above `no_adapt`;
6. clean AP is not worse than `current_ours` by more than `0.10 AP`.

These are developmental source-only criteria, not publication claims. If all pass, the next review may authorize a larger disjoint confirmatory run and then FCOS/SSD cross-detector evaluation. Do **not** run those targets during T018-A.

If the candidate is positive but fails the `+0.10 AP` materiality bar, report it as insufficient rather than launching coefficient/threshold tuning. If it is worse, close this exact native-pseudo-target formulation. In either case preserve all receipts.

## Scope boundaries

For this task, do not:

- modify detector weights;
- use ground-truth labels in adaptation;
- sweep pseudo thresholds/top-k/loss weights/K/LR/CLIP prompts;
- add spatial ISP yet;
- add localization-only/objectness-only variants;
- resume T017 component attribution;
- use FCOS/SSD or COCO-val;
- start meta-training, predictor redesign, gate/dose tuning, or mask/region search;
- replace the deployed Ours implementation before research review.

Append exact outcomes to `coordination/CODEX_TO_CHATGPT.md`, commit/push all plan/code/tests/results/receipts, and stop for review.

## Research intent

The purpose is deliberately practical: test whether the weak current AP gain is primarily caused by an under-specified self-supervised detector objective. This candidate keeps the paper-defining claim intact—**test-time adaptation happens in image-formation space, with the detector frozen and no test labels**—while giving the TTT gradient the full native detection structure needed to affect classification, localization, proposal objectness, and proposal geometry. Performance improvement is now a first-class acceptance criterion, not merely a secondary diagnostic.
