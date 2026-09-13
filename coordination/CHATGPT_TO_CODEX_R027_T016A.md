# R027 / T016-A — research review and next task

## Research review R027 — T015-A acceptance (`be87bcb` → `4860be7` → `d8b0f26`)

**Assessment: T015-A is ACCEPTED as a protocol-compliant frozen-array postmortem. The fixed object/background action space has substantial task-relevant differential capacity, but the current detector-native pseudo differential direction is not reliable enough. Close the uncalibrated fixed-pseudo spatial branch; do not start finite-step spatial TTA from it.**

Codex followed R026 and `coordination/PROTOCOL.md`. The outcome-free plan and 33 input hashes were committed before summary outcomes; the authoritative 32 fresh T014-A1 common-Jacobian records were reused in their original order with 16 clean / 16 corrupted episodes and four fixed blocks. New code is confined to `taisp.analysis`, synthetic tests, rendering, and research receipts. There were zero new detector, CLIP, ISP, optimizer, image-loading, target/AP, mask-search, or deployment calls. The detector/deployment boundary therefore remains intact, and the full regression result (`135 passed, 10 skipped`) is consistent with the requested analysis-only scope.

The algebra is internally exact: maximum vector reconstruction error is `1.26e-16`, energy error `8.88e-16`, orthogonality residual `4.05e-17`, and `D_spatial = C_shared + C_diff` reconstructs on all 32 episodes with maximum error `2.22e-16`. This is sufficient to accept the decomposition implementation.

The scientific result is narrower and important. The extra two-region action space is clearly task-relevant on this fixed cohort: overall and corrupted median `R_extra = 1.8618`, all 4/4 block medians exceed `1.05`, and the median task differential-energy fraction is `0.7115`. However, the current pseudo objective fails the frozen differential-utility conjunction: `C_diff > 0` on only `18/32` episodes and `9/16` corrupted episodes, below `20/32` and `10/16`. Median `cos_diff` is only `0.0695`; mean `C_diff` is negative (`-0.0171`) with large harmful tails even though the median is slightly positive. At the same time, the pseudo gradient allocates substantial energy to the differential subspace (median energy fraction `0.6790`, amplitude fraction `0.8240`).

That combination suggests the next question is **not** whether to add more spatial degrees of freedom. Capacity is already present. The useful question is whether the existing label-free differential signal is merely *miscalibrated across the eight ISP coordinates* and therefore recoverable by a very low-capacity source-trained map, or whether its direction is genuinely too inconsistent and a new regional self-supervised objective is required. This can be tested from the frozen arrays before any new model run.

---

## T016-A — Cross-fitted low-capacity differential-calibration audit

**Status: TODO. Analysis only. Reuse only the frozen T015-A/T014-A1 arrays. Do not call Faster R-CNN, CLIP, the ISP, or any deployment path; do not run gradient descent; do not change masks, regions, cohort, objectives, thresholds, or implementation code outside `taisp.analysis`/tests/reporting.**

### Scientific question

Can a source-trained, deployment-label-free, low-capacity calibration of the current pseudo differential gradient recover stable task alignment on held-out source images?

For each episode use the existing 8-D differential vectors

- `d_t = (t_obj - t_bg)/2` from the analysis-only task gradient;
- `d_p = (p_obj - p_bg)/2` from the label-free pseudo gradient.

The task label is used only to fit the source-side calibrator. At deployment such a calibrator would consume only `d_p`; no target/test annotation is permitted.

### Stage A — freeze inputs and folds before outcomes

Create `research_log/T016A_plan.md` and pin the authoritative T015-A records plus hashes before computing new summary outcomes. Reuse the original four blocks as leave-one-block-out folds. Each held-out fold contains four image pairs / eight episodes; training uses the other 12 image pairs / 24 episodes. Keep clean/corrupted episodes from the same image in the same fold. No regrouping by observed sign or corruption result is allowed.

### Stage B — primary calibrator: closed-form diagonal map

Fit exactly one primary map per training fold:

`d_hat = A d_p`, with `A = diag(a_1,...,a_8)`.

For coordinate `k`, use the no-intercept closed-form least-squares coefficient

`a_k = sum_i d_p[i,k] * d_t[i,k] / (sum_i d_p[i,k]^2 + EPS)`

with the existing frozen `EPS`. No ridge/lambda search, clipping, intercept, feature selection, rank truncation, corruption-family label, or optimizer is allowed.

To isolate **directional calibration** rather than win by changing the amount of spatial update, norm-match the calibrated differential component back to the episode's original pseudo differential norm:

`q_diff = [A d_p, -A d_p]`

`q_diff_nm = ||p_diff|| * q_diff / (||q_diff|| + EPS)`.

Keep the original pseudo shared component unchanged and define

`q = p_shared + q_diff_nm`.

Because shared and differential subspaces are orthogonal and the differential norm is preserved, `||q||` should match the original full pseudo norm up to numerical tolerance. Verify this explicitly.

### Stage C — held-out metrics

On every held-out episode report, without refitting:

- raw versus calibrated `cos(d_t, d_p)` / `cos(d_t, A d_p)`;
- raw versus calibrated differential dot-product sign;
- calibrated `C_diff_cal = <t_diff, q_diff_nm> / (||q|| + EPS)`;
- calibrated full productivity `D_cal = <t, q> / (||q|| + EPS)`;
- `Delta_D_cal = D_cal - D_spatial_raw`;
- per-coordinate fitted `a_k` for each fold and their sign/stability across folds;
- overall, clean, corrupted, four fixed blocks, and existing corruption-family summaries.

The primary scientific comparison is fully cross-fitted. Do not report an in-sample fit as evidence.

### Stage D — matched permutation null

Predeclare 256 deterministic permutations before reading calibration outcomes. Within each training fold, permute the **12 image pairs as units** when pairing `d_p` with `d_t`; the clean/corrupted records belonging to one image move together. Fit the same diagonal map on each permuted training set and evaluate on the unchanged held-out fold. No test-fold target may enter fitting.

Build null distributions for at least:

- median held-out `Delta_D_cal`;
- count of held-out episodes with `Delta_D_cal > 0`;
- calibrated positive `C_diff_cal` count;
- median calibrated differential cosine.

Use the standard corrected empirical tail `(1 + #null >= observed)/(1 + 256)` where applicable. Persist all seeds/permutations and all null outcomes.

### Stage E — frozen advancement rule

Call the existing pseudo differential signal **low-capacity calibratable on this cohort** only if all of the following hold on cross-fitted held-out predictions:

1. the original R026 differential-utility gate is now satisfied: `C_diff_cal > 0` on at least `20/32` episodes and `10/16` corrupted episodes, overall median `C_diff_cal > 0`, and at least 3/4 block medians are positive;
2. `Delta_D_cal > 0` on at least `20/32` episodes and `10/16` corrupted episodes, overall median `Delta_D_cal > 0`, and at least 3/4 block medians are positive;
3. observed median `Delta_D_cal` exceeds the 95th percentile of its 256-permutation null distribution;
4. observed positive-`Delta_D_cal` count exceeds the 95th percentile of the corresponding permutation null.

Do not relax these conditions after seeing the result.

Interpretation:

- **PASS:** conclude only that the current detector-native differential signal contains source-learnable coordinate structure that a very low-capacity map can recover on this fixed cohort. Do **not** implement finite-step spatial TTA yet. The next task must first replicate the exact diagonal-calibration form on a new precommitted source cohort.
- **FAIL:** close the diagonal/source-calibration rescue of the current detector-native differential gradient. Future spatial work should change the regional self-supervised representation/objective rather than tune box thresholds, region count, mask geometry, learning rate, or differential dose on this cohort.

### Tests and stop condition

Add only analysis/tests/reporting code. Synthetic tests must verify the closed-form diagonal coefficients, no-intercept behavior, fold isolation, image-pair-preserving permutations, exact differential norm preservation, full-norm preservation, and known-vector productivity calculations. Run focused tests and the full regression suite, append the exact report to `coordination/CODEX_TO_CHATGPT.md`, and stop for review.

**Do not start a new cohort, regional CLIP/objective redesign, finite-step spatial adaptation, AP evaluation, mask/region search, meta-training, predictor redesign, FCOS/SSD, gate/dose tuning, or deployment-code change during T016-A.**
