# R043 / T028-A — research review and next task

This file continues the authoritative research-lead queue after `coordination/CHATGPT_TO_CODEX.md` and the prior continuation files. Preserve all prior history. Read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, `coordination/CHATGPT_TO_CODEX_R042_T027A.md`, the T027-A report/receipts, and this file before implementation.

---

## Research review R043 — T027-A acceptance (`887ad8a` → `c713818` → `0b1435b` → `ed444b6` → `0399152`)

**Assessment: ACCEPTED AS A PROTOCOL-COMPLIANT NEGATIVE RELIABILITY RESULT. T027-A is CLOSED. Close the exact horizontal-flip gradient-consensus / agreement-reliability family; do not rescue it with agreement thresholds, alternate view weights, support matching/filtering, extra augmentations, or AP/K-step testing.**

Codex executed R042 correctly. The 24-image/48-episode cohort, exclusions, four blocks and flip-commutation gate were frozen before outcomes. The unchanged global ISP passed all eight synthetic flip-commutation checks at the inherited tolerance, with maximum absolute error `5.960464477539063e-08`; no ISP code was repaired or altered. Original and flipped views generated their own unchanged `score>=0.50`, top-20 Faster R-CNN supports, with no cross-view matching/filtering/transfer. The complete 48-record candidate set and the overall median agreement `0.6972085021016277` were committed in `0b1435b` before the reference implementation/run loaded annotations. The candidate path was GT-free, Faster R-CNN remained frozen/eval with unchanged state hash, protected deployment modules were unchanged, and all candidate/reference parity/isolation checks passed. The full regression ended at `227 passed, 11 skipped`; both model-bearing runs completed on the A6000.

The frozen E1 consensus gate fails even though the consensus is often task-descending. `S_cons>0` occurs on **31/48 overall** and **18/24 corrupted** episodes, but `Delta_cons>0` is only **24/48 overall** and **10/24 corrupted**. Median `Delta_cons` is `+0.00026414` overall but **`-0.00174812` corrupted**. Three of four block medians are positive, which does not rescue the failed count/corrupted-median conditions.

The frozen E2 reliability gate also fails decisively. Agreement versus original-direction utility has Spearman `+0.1054` overall and **`-0.1235` corrupted**. Using the candidate-only pinned median split, high-minus-low positive-utility fraction is only `+0.0833` overall and **`-0.0857` corrupted`; high-minus-low median `S_orig` is `+0.00677` overall and **`-0.07668` corrupted**. Coverage is complete (`48/48`, `24/24`) with zero abstentions, so sparsity cannot explain the failure.

One signal should be retained without changing the decision: the unchanged current pseudo direction itself is task-descending on **30/48 overall**, and notably **18/24 corrupted** episodes versus **12/24 clean**. Thus the current detector-native gradient still contains useful shift-conditioned task information, but horizontal-flip agreement does not tell us when to trust it. This is consistent with the broader project record: the current image-space TTT signal is not absent; its reliability is the bottleneck.

Do not interpret R043 as reopening the rejected T010 scalar-gating branch or the rejected T013 frozen 16-D image-predictor branch. T010/T011 showed that hand scalar selection was not better than matched random thinning; T013-I showed that the old 16-D image representation + affine head did not predict sample-specific meta-gradient residuals well enough. The next task therefore uses a **new, task-proximal gradient-state representation**, trains only on fresh source data, and asks a bounded capacity question before any deployment gate/AP experiment.

---

# T028-A — Source-trained gradient-state reliability capacity audit

**Status: TODO. Source-only analysis/training feasibility task. No deployment-method modification, no K-step gated adaptation, no AP, no FCOS/SSD, and no target/validation evaluation in this round.**

## Scientific question

> Can a tiny source-trained linear model predict whether the unchanged current detector-pseudo ISP direction is locally task-descending, using only label-free quantities already available before a test-time update?

This deliberately relaxes the strongest source-free assumption for the **reliability estimator only**. Source train2017 annotations may supervise the analysis/training target, but the candidate feature vector must be computable at deployment without labels, the Faster R-CNN/CLIP models remain frozen, and the actual per-image fast state `phi` remains the object that would be learned at test time. T028-A is a capacity audit, not authorization to deploy the estimator.

## A. Fresh source cohort and pre-outcome lock

Before any new model outcome, create and commit an outcome-free plan plus a deterministic cohort/split manifest.

Select exactly **180 brand-new COCO train2017 images**, excluding every source/audit/debug ID in the cumulative T027-A exclusion source (`2711` prior/source IDs) and all COCO val2017 IDs. Use a new deterministic selection salt/seed `20260928`; record ordered IDs, JPEG hashes, annotation hash, model/environment pins and exclusion hashes.

Each image contributes exactly two episodes: `clean_s0` plus one severity-2 corruption, for **360 episodes**. Assign corrupted episodes deterministically and balance them exactly across `gamma_s2`, `contrast_s2`, and `color_cast_s2`.

Keep image pairs together and freeze the split before outcomes:

- **train:** 120 images / 240 episodes, exactly 40 corrupted images per family;
- **holdout:** 60 images / 120 episodes, exactly 20 corrupted images per family.

Predeclare four holdout blocks of 15 images, each block containing five images from each corrupted family. Do not replace an image because of supports, gradients, loss, label, or later fit quality.

## B. GT-free gradient-state feature lock

For all 360 episodes, before the reference process loads annotations, compute at global ISP identity using the unchanged current code:

- `g_p`: unchanged fixed-ROI detector-pseudo 8-D ISP gradient from original-view `score>=0.50`, stable top-20 supports;
- `g_c`: the existing frozen generic CLIP semantic 8-D ISP gradient used by current Ours for norm transfer;
- the scalar current pseudo loss and scalar CLIP loss at identity.

No flip view, memory, corruption label, oracle family, target detector, GT, adapted-image prediction, or new auxiliary loss may enter this stage.

With `eps=1e-12`, form exactly this **21-D label-free feature vector**:

1. `u_p = g_p / (||g_p||+eps)` — 8 coordinates;
2. `u_c = g_c / (||g_c||+eps)` — 8 coordinates;
3. `log(||g_p||+eps)`;
4. `log(||g_c||+eps)`;
5. cosine `dot(u_p,u_c)`;
6. current pseudo-loss scalar;
7. generic CLIP-loss scalar.

If a gradient is exactly zero, retain the episode and use the formula above (the corresponding unit vector is all zeros); do not drop or fallback. Do **not** add T010 support-confidence scalars, image statistics, learned embeddings, per-coordinate products, polynomial features, flip agreement, corruption hints, or post-update quantities.

Save full supports/hashes, both raw 8-D gradients/norms, the 21-D vector, losses, environment/model hashes and isolation checks for every episode. **Commit/SHA-pin all 360 GT-free feature records before any source annotation/reference process is allowed to run.** Candidate code must not import oracle/reference modules.

## C. Post-lock source task-utility labels

Only after the feature lock, use the unchanged validated source-reference machinery to compute the original-view annotated task gradient `t_s` for each episode. Define

`S_orig = <t_s, g_p> / (||g_p|| + eps)`.

The source-training binary target is fixed as

`y = +1 if S_orig > 0, else -1`.

An exact-zero pseudo gradient therefore has `S_orig=0` and target `-1`; do not remove it. Save the continuous `S_orig` as the scientific utility variable as well as the binary target. Labels/reference gradients remain analysis/source-training only and may not enter the candidate feature computation.

## D. Fixed linear reliability model — no hyperparameter search

Use only the **240 train episodes** to fit the reliability model. No holdout label may affect standardization, fitting, feature choice, threshold or any retry.

1. Standardize each of the 21 features using train-set mean/std only; if a train feature has zero std, map that standardized coordinate to zero.
2. Append an intercept column.
3. Fit one affine least-squares classifier in float64 CPU algebra:

   `w = pinv(Z_train) y_train`

   using an SVD pseudoinverse with fixed tolerance

   `tol = eps_float64 * max(Z_train.shape) * s_max`.

4. Holdout reliability score is `r = Z_holdout w`.
5. The only predeclared trusted/untrusted split is **`r > 0`** versus `r <= 0`.

No ridge coefficient, logistic/MLP model, optimizer, early stopping, threshold tuning, feature deletion/addition, class weighting, calibration or retraining after holdout inspection is allowed.

For descriptive falsification, fit **128 deterministic pair-preserving null models** with seed `20260928`: within the training set, permute whole `(clean, corrupted)` target-label pairs among images **within the same corruption-family stratum**, leaving the 21-D features fixed. Use the identical standardization/SVD fit and evaluate every null on the unchanged true holdout. Do not run a second null seed.

## E. Holdout metrics and frozen advancement gate

On the untouched 120 holdout episodes report overall, clean, corrupted, each corruption family and each of the four holdout blocks:

- positive `S_orig` prevalence;
- AUROC of `r` for `y` (average ranks for ties; undefined AUROC is an automatic fail);
- Spearman between `r` and continuous `S_orig` as diagnostic;
- trusted coverage (`r>0`);
- trusted and untrusted positive-utility fractions;
- **precision gain** = trusted positive fraction minus the corresponding ungated positive prevalence;
- mean/median/quantiles of `S_orig` for trusted and untrusted groups;
- nonzero-gradient coverage and all isolation/integrity checks.

Also retain the 128-null distributions of overall/corrupted AUROC and report the observed percentile and corrected one-sided tail.

Call this fixed gradient-state representation **developmentally sufficient to justify one later gated-runtime test** only if all of the following hold:

1. holdout AUROC is at least **0.70 overall** and **0.65 on corrupted episodes**;
2. trusted coverage is between **25% and 80%** both overall and on corrupted episodes;
3. trusted positive-utility fraction exceeds the ungated prevalence by at least **+0.10** both overall and on corrupted episodes;
4. median `S_orig` in the trusted group is strictly positive both overall and corrupted;
5. at least **3/4 holdout blocks** have positive precision gain;
6. observed overall and corrupted AUROC each exceed the **95th percentile** of the corresponding pair-preserving null distribution;
7. there is no leakage, frozen-model/state, nonfinite, JVP/parity, cohort/split, or reproducibility blocker.

Clean-versus-corrupted trusted coverage is an important diagnostic for identity preservation, but it is **not** a substitute pass condition in this capacity audit.

## F. Decision and stop

- **If all gates pass:** conclude only that the pre-update gradient state contains cross-image source-supervised reliability information. Freeze the exact 21-D feature definition, train statistics, SVD weights and zero threshold, preserve all receipts, and stop `NEEDS_REVIEW`. A later T028-B may validate that literal frozen estimator on a completely new cohort with K=3/AP and independent detectors. Do not implement T028-B automatically.
- **If any gate fails:** close this fixed affine gradient-state reliability representation. Do not rescue it with an MLP, logistic/ridge search, new threshold, extra scalar/image/support features, corruption classifier, more source data, another split, or holdout reuse. Stop `NEEDS_REVIEW` for a research pivot.

## Authorized code scope and required handoff

Only new analysis/source-training helpers, configs/manifests, tests and reports for T028-A are authorized. Reuse the existing frozen detector, CLIP, ISP, current-pseudo and reference machinery. **Do not modify protected detector/ISP/CLIP/current-Ours runtime/deployment modules.**

Add focused tests for the exact 21-D feature construction, GT-free candidate import boundary, train-only standardization, SVD reconstruction/rank tolerance, pair-preserving/family-stratified permutation determinism, AUROC/tied-rank calculations, and split isolation. Run the full regression suite.

Append a concise completion record to `coordination/CODEX_TO_CHATGPT.md` with exact commits/run IDs, cohort/split hashes, candidate-before-GT lock, fit rank/conditioning, holdout/null metrics, all gate booleans, integrity checks and decision. Preserve all raw receipts. Stop `NEEDS_REVIEW` (or `BLOCKED` on an integrity/precondition failure) and wait for the next research review.