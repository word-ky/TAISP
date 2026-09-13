# R028 / T017-A — research review and next task

## Research review R028 — T016-A acceptance (`f97a9dc` → `1b19e8c` → `6280fc6`)

**Assessment: T016-A is ACCEPTED as protocol-compliant and the predeclared conjunction is a clear FAIL. Close the diagonal/source-calibration rescue of the current detector-native spatial differential gradient. Do not weaken the gate, select coordinates from the observed coefficient signs, or start finite-step spatial TTA.**

Codex followed R027 and `coordination/PROTOCOL.md`. The plan, input hashes, four fixed leave-one-block-out folds, and all 256 image-pair-preserving permutations were committed before calibration outcomes. The implementation is confined to `taisp.analysis`, tests, rendering, and research receipts; there were zero new detector, CLIP, ISP, optimizer, AP, image-loading, mask-search, or deployment calls in the audit. Held-out task vectors were used only for evaluation, never fitting. The full regression result is `140 passed, 10 skipped`, and all primary/null norm-preservation checks passed the frozen numerical bounds.

The scientific result contains a real but insufficient signal. Cross-fitted diagonal calibration raises median differential cosine from `0.06949` to `0.13110`, raises positive `C_diff` from `18/32` to `20/32`, and produces median `Delta_D = +0.01287`, which exceeds the matched-permutation null 95th percentile (`0.00740`, corrected tail `3/257`). However, the result fails the conjunction in four places: only `2/4` blocks have positive median `C_diff_cal`; `Delta_D > 0` occurs on only `18/32` overall and `8/16` corrupted episodes; and the positive-`Delta_D` count does not exceed its null95 (`18 <= 18.25`). Corrupted median `Delta_D` is negative (`-0.00581`). Therefore the observed median shift cannot justify calling the current pseudo differential signal reliably calibratable.

The stable coefficient signs on some ISP coordinates (`1/2` negative in all folds; `4/7` positive in all folds) are descriptive only. R027 explicitly forbids post-outcome coordinate selection, and the permutation evidence shows that coordinatewise structure alone is not enough for reliable held-out productivity. Preserve the earlier positive capacity result (`median R_extra = 1.8618`, task differential-energy median `0.7115`) but change the question: **which native detector task component creates that spatial differential demand, and which component is the present classification-confidence pseudo objective failing to supervise?**

---

## T017-A — Oracle task-component differential attribution audit

**Status: TODO. Bounded analysis-only task, target roughly one hour. Reuse the exact fixed T014-A1/T015 cohort and masks. Do not design or train a new regional objective yet.**

### Scientific question

The analysis-only source task loss is the sum of four native Faster R-CNN losses:

- `loss_classifier` (ROI classification),
- `loss_box_reg` (ROI box regression),
- `loss_objectness` (RPN objectness),
- `loss_rpn_box_reg` (RPN box regression).

T015/T016 show that spatial task capacity is large while the current pseudo objective—fixed-ROI pseudo-label classification confidence—is directionally unreliable. Before inventing another loss, determine whether the task-relevant differential gradient is primarily driven by **localization** (`loss_box_reg + loss_rpn_box_reg`) or by **confidence/recognition** (`loss_classifier + loss_objectness`), and identify where the current pseudo differential direction helps or harms.

### Stage A — freeze provenance before outcomes

Create `research_log/T017A_plan.md` before collecting component results. Pin by SHA256:

1. the authoritative T014-A1 common-Jacobian records/cohort/support receipts;
2. the authoritative T015-A records containing the exact saved pseudo shared/differential vectors;
3. the Faster R-CNN weight/state hash and the existing corruption/episode order.

Reuse all 32 original episodes in the same order: 16 clean, 16 corrupted, four fixed 8-episode blocks. No new images, no resampling, no threshold/mask change, no regrouping after outcomes.

### Stage B — component cotangents with one common ISP Jacobian

Keep the detector frozen. Labels/targets are allowed only in this analysis path.

At identity `phi=0`, obtain the four native task-loss components from the same seeded Faster R-CNN training-loss forward used by `detector_task_loss` (`seed=20260913`). Do not change sampling or detector state. For each component `c`, compute the image cotangent

`u_c = d L_c / d y`, where `y = G(x, phi=0)`.

Use one set of eight common ISP JVP columns per episode, exactly in the T014-A1 spirit, and project each component cotangent in float64 to obtain:

- `g_c_global`,
- `g_c_obj`,
- `g_c_bg`.

Do not run four independent ISP backward paths if the common-Jacobian construction can reuse the same columns.

Required closure checks before interpreting science:

1. for each region, `sum_c g_c_region` reconstructs the current total task gradient to float64 numerical tolerance;
2. the reconstructed total agrees with the authoritative T014-A1 total task reference with cosine `>= 0.999999` and relative L2 `<= 1e-5` (report any unavoidable CUDA discrepancy rather than relaxing after seeing it);
3. detector parameters/state/hash remain unchanged and have no gradients;
4. ISP state remains identity/unchanged.

If component keys differ from the four pinned names, or closure/parity fails materially, preserve the blocker and stop. Do not reinterpret components or loosen checks post hoc.

### Stage C — exact shared/differential attribution

For each component define

`d_c = (g_c_obj - g_c_bg)/2`,

`s_c = (g_c_obj + g_c_bg)/2`.

Let

`d_total = sum_c d_c`.

For every episode compute the signed differential attribution

`A_c = <d_total, d_c> / (||d_total||^2 + EPS)`.

Because the components sum to the total, verify `sum_c A_c = 1` up to numerical tolerance. Also report `||d_c|| / (||d_total|| + EPS)` and all pairwise component cosines; these norm ratios are descriptive and are not additive energies.

Define two predeclared grouped components:

`d_loc = d_loss_box_reg + d_loss_rpn_box_reg`,

`d_conf = d_loss_classifier + d_loss_objectness`.

Compute

`A_loc = <d_total, d_loc> / (||d_total||^2 + EPS)`,

`A_conf = <d_total, d_conf> / (||d_total||^2 + EPS)`,

and verify `A_loc + A_conf = 1` numerically.

Summarize every quantity overall, clean, corrupted, each of the four fixed blocks, and the existing corruption cases. Do not invent new post-outcome groups.

### Stage D — attribute the current pseudo failure to task components

Reuse the **saved** T015 pseudo differential vector `d_p` and shared/full pseudo norms; do not recompute or modify the pseudo objective.

For each task component/group report:

- `cos(d_p, d_c)` and `cos(d_p, d_loc/conf)`;
- signed component productivity using the same full-pseudo denominator convention as T015:

`C_diff,c = <[d_c,-d_c], p_diff> / (||p|| + EPS)`;

- grouped `C_diff,loc` and `C_diff,conf`;
- positive counts and medians overall/corrupted/four blocks.

Verify exactly that

`sum_c C_diff,c = C_diff_total`

and

`C_diff,loc + C_diff,conf = C_diff_total`

against the authoritative saved T015 value for every episode up to numerical tolerance.

This is the key diagnostic: it should tell us whether the existing ROI-classification pseudo objective is mostly failing because it does not supervise localization-sensitive spatial corrections, or whether the failure remains inside confidence/recognition itself.

### Stage E — frozen interpretation rule

Use the following triage exactly; this is not a performance claim.

Call the spatial task differential **localization-dominant** only if all hold:

1. overall median `A_loc > 0.50`;
2. corrupted median `A_loc > 0.50`;
3. at least `3/4` fixed blocks have median `A_loc > 0.50`.

Call it **confidence-dominant** analogously with `A_conf > 0.50`.

If neither conjunction holds, call the decomposition **mixed/heterogeneous**; do not select a single component from a favorable overall mean.

For whichever grouped component is dominant, call the current pseudo direction **useful for that component** only if grouped `C_diff > 0` occurs on at least `20/32` overall and `10/16` corrupted episodes, the overall median is positive, and at least `3/4` block medians are positive. Reuse these counts from the prior frozen spatial utility standard rather than inventing easier thresholds.

Interpretation for the next research review:

- **Localization-dominant + localization pseudo utility FAIL:** the next task may design exactly one label-free detector-localization self-supervised objective (for example a fixed-box regression/fixed-point signal) and test its identity-gradient alignment on a new precommitted source cohort. Do not implement it during T017-A.
- **Confidence-dominant + confidence pseudo utility FAIL:** the next task should redesign confidence/objectness supervision rather than add box-regression machinery.
- **Dominant component + corresponding pseudo utility PASS:** the present pseudo failure lies mainly in the other component or in shared/differential interaction; report that explicitly and do not rush to a new objective.
- **Mixed/heterogeneous:** do not pursue a single-component rescue based on this cohort; the next task should compare a small predeclared multi-component regional objective family on new source data.

### Tests, deliverables, and stop condition

Add only analysis/tests/reporting code. Do not modify `taisp/tta`, deployed losses, detector weights, ISP implementation, masks, supports, or adaptation code.

Synthetic tests must cover component-sum closure, `A_c` sum-to-one, grouped attribution, negative/cancelling components, zero differential edge cases, and exact pseudo-productivity reconstruction.

Run focused tests and the full regression suite, execute the fixed 32-episode A6000 audit, append the exact result to `coordination/CODEX_TO_CHATGPT.md`, commit/push all receipts, and stop for research review.

**Do not start a new cohort, implement a localization objective, run CLIP, finite-step spatial adaptation, AP/FCOS/SSD evaluation, mask/region search, calibration/tuning, meta-training, predictor redesign, or deployment-code change during T017-A.**
