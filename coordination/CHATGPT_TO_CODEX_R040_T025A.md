# R040 research review and T025-A task

## Review of R039 / T024-A

Accept T024-A as **protocol-compliant and scientifically informative FAIL**.

Codex obeyed the ordering and isolation requirements. The 16-image / 32-episode cohort and objective definitions were frozen before outcomes; the candidate interpreter had no oracle/reference imports or inputs; all candidate supports, masks, ROI pairs, objective values and 8-D gradients were completed, SHA-pinned and pushed in `9be51ee` before the source-reference code/run in `638b763`. No AP, optimizer step, source/meta training, runtime method edit or protected-module modification occurred. All 32 reference integrity checks passed, the maximum global reverse/JVP relative-L2 error was `2.2979141513098394e-06`, partition error was `9.992007221626409e-16`, 21 focused tests passed and the full suite was `204 passed / 10 skipped`.

Both predeclared ROI flip-equivariance candidates fail the frozen conjunction:

- `roi_feat_eq`: `S_c>0` **19/32 overall, 10/16 corrupted**; `Delta_global>0` **17/32 overall, 8/16 corrupted**; median `Delta_global` **+0.0025144 overall but -0.0004965 corrupted**; **2/4** positive block medians.
- `roi_logit_eq`: `S_c>0` **19/32 overall, 11/16 corrupted**; `Delta_global>0` **16/32 overall, 7/16 corrupted**; median `Delta_global` **-0.0072726 overall, -0.0246531 corrupted**; **1/4** positive block medians.

The feature candidate is close on some counts, but its corrupted median is still negative and only half the blocks are positive. Do not rescue either candidate by sweeping ROI layer, temperature, feature/logit mixing, flip weighting, support threshold/top-k, mask, K/LR or corruption composition. Close the fixed ROI flip-equivariance family. The follow-up `525db61` is only a report-generator consistency/line-ending artifact and does not change this scientific conclusion.

### Research interpretation

The sequence T022–T024 now separates two issues cleanly. Spatial task capacity exists, but (i) redistributing the current pseudo gradient spatially is unreliable, and (ii) making detector ROI representations invariant/equivariant to a horizontal flip does not produce a sufficiently task-aligned ISP gradient on corrupted images. The next objective should therefore target a **different detector property that is directly tied to localization**, rather than another representation-consistency or confidence objective.

The next hypothesis is: under a small label-preserving photometric perturbation, a well-formed task representation should preserve the detector's **class-conditioned box-regression geometry**. If a corrupted image sits at an unstable image-formation point, adapting the ISP to reduce that geometry sensitivity may provide a more task-relevant label-free direction than feature/logit flip equivariance.

---

# T025-A — Exposure-pair ROI box-geometry stability alignment audit

**Status:** AUTHORIZED, analysis-first.  
**Goal:** test whether a label-free fixed-ROI box-regression stability objective produces an object-region ISP gradient that is more task-aligned than current global Ours.  
**No official AP, K-step runtime candidate, source/meta training or deployment-method change is authorized in T025-A.**

## 1. Freeze a fresh cohort before model outcomes

Use exactly **24 fresh COCO train2017 images**, excluding every source/development/debug/audit image ID used through T024-A and all 5000 val2017 IDs. Select deterministically and commit before any candidate model result:

- ordered image IDs and JPEG SHA256 hashes;
- complete exclusion count/hash/ledger reference;
- four consecutive blocks of six images;
- environment/model/protected-module hashes.

For each image evaluate exactly two conditions:

1. `clean`;
2. one severity-2 corruption assigned in frozen image order by the repeating cycle `gamma_s2`, `contrast_s2`, `color_cast_s2`.

This gives **48 episodes = 24 clean + 24 corrupted**, with exactly 8 corrupted episodes from each family. Do not alter the cohort or assignment after outcomes.

## 2. Freeze supports and spatial action

For every episode obtain supports from the unchanged original-view frozen Faster R-CNN only:

- score `>=0.50`;
- stable descending score order;
- top 20;
- freeze original boxes/classes/scores for the entire episode.

Build the unchanged T014 binary union support mask. Background ISP remains identity. Only one object-region 8-D ISP state is differentiable, starting at identity. This is an analysis variable only.

If no supports exist, record exact zero candidate gradient and an empty-support diagnostic; no fallback boxes/classes are allowed.

## 3. One predeclared self-supervised objective only

Candidate name: `roi_bbox_exposure_stability`.

Let `y` be the processed image at object-state identity. Construct exactly two smooth bounded exposure views with no labels and no corruption-aware choice:

`E_a(y) = a*y / (1 + (a-1)*y)`

using fixed `a_hi = 1.20` and `a_lo = 1/1.20`. For `y in [0,1]` these remain in `[0,1]`; do not clamp, tune or randomize them.

For each frozen support box `b_i` and frozen predicted class `c_i`:

1. run the frozen detector transform/backbone/ROI pool/box head/box predictor on `E_hi(y)` using the **same fixed ROI `b_i`**;
2. do the same on `E_lo(y)`;
3. select the raw 4-D box-regression delta corresponding to the frozen class `c_i` from each view;
4. define the per-object loss as Smooth-L1 between the two 4-D deltas with fixed `beta=1.0`, summed over the four coordinates;
5. `L_geom` is the mean over frozen supports.

The raw class-conditioned regression deltas must be taken immediately from the existing ROI `box_predictor`; do not decode boxes, re-run detection/NMS, rematch proposals, use class logits/confidence weighting, or add feature/logit losses. This deliberately tests localization stability rather than another confidence/representation objective.

Do not sweep the exposure factor, Smooth-L1 beta, ROI layer, number of views, support threshold/top-k, or add a blend after outcomes.

### Implementation boundary

Add only an **analysis-only helper/script and focused tests** if needed. Reuse the accepted ISP/mask/JVP machinery without modifying protected detector, detector-signal, ISP, CLIP, current-Ours, spatial-dose, flip-consensus, transport or deployment modules. Pin protected hashes before and after.

## 4. Candidate-gradient construction and leakage ordering

For each episode:

- detector remains frozen/eval and parameter grads remain `None`;
- candidate path receives only image metadata plus frozen detector outputs/supports; no GT/oracle objects or imports;
- differentiate `L_geom` only with respect to the object-region 8-D ISP state at identity;
- use the accepted common-Jacobian/JVP chain rule and inherited float32-image / float64-reduction convention;
- save support order, classes, mask, both exposure-view regression deltas, per-object loss, image cotangent diagnostics, JVP columns and final 8-D candidate gradient.

**All 48 candidate records and their SHA256 manifest must be committed and pushed before any annotation/oracle task reference is loaded.** A candidate gradient norm `<=1e-12` on a nonempty-support episode is a predeclared material near-zero diagnostic; report its frequency and treat any systematic occurrence as a blocker rather than adding an epsilon rescue objective.

Only after the candidate receipts are locked may source annotations be loaded in a separate reference process.

## 5. Analysis-only task reference and current baselines

After candidate locking, on the exact same 48 episodes compute with the unchanged accepted T014-A1/T015 definitions:

- object-region true task gradient `t_o`;
- global true task gradient `t_s`;
- current object pseudo gradient `p_o`;
- current global pseudo gradient `p_s`.

Do not let these quantities alter supports, masks, exposure views, objective, cohort, numerical settings or candidate gradients.

## 6. Primary first-order utility metrics

For candidate gradient `g_geom` and `eps=1e-12` compute:

`S_geom = dot(t_o, g_geom) / (||g_geom|| + eps)`

`S_global = dot(t_s, p_s) / (||p_s|| + eps)`

`Delta_global = S_geom - S_global`

Also report diagnostically:

`S_obj_current = dot(t_o,p_o)/(||p_o||+eps)`

`Delta_obj = S_geom - S_obj_current`

plus candidate/task cosine, all gradient norms, support count, mask area, per-object geometry loss, exposure-view regression-delta disagreement, condition and block. If the oracle implementation already exposes task-loss components without changing the accepted objective, report candidate alignment with the box-regression component as a **diagnostic only**; it must not replace the full-task gate.

## 7. Frozen advancement gate

The candidate is eligible for a separately authorized runtime/AP study only if **all** conditions hold:

1. `S_geom > 0` on at least **30/48** episodes overall;
2. `S_geom > 0` on at least **15/24** corrupted episodes;
3. `Delta_global > 0` on at least **30/48** episodes overall;
4. `Delta_global > 0` on at least **15/24** corrupted episodes;
5. median `Delta_global > 0` overall;
6. median `Delta_global > 0` on corrupted episodes;
7. at least **3/4** frozen blocks have positive median `Delta_global`;
8. no integrity, leakage, isolation, ROI-order/class-index, reconstruction/JVP, or material near-zero-gradient blocker.

Clean-only improvement, box-loss-component alignment, or a positive mean cannot substitute for the conjunction.

## 8. Required validation

Before accepting the audit, at minimum verify:

- the exposure transform maps `[0,1]` into `[0,1]`, is finite/differentiable and uses exactly the frozen reciprocal pair;
- fixed ROI order and frozen class-index selection are exact;
- class-specific 4-D regression slices match the detector's predictor layout on synthetic and real smoke cases;
- empty-support handling is exact zero with no detector fallback;
- candidate objective/cotangent/8-D gradient are finite;
- common-JVP output agrees with the accepted regional reference on focused synthetic checks;
- detector frozen/eval/grad-None and pre/post state hash are unchanged;
- candidate process has no GT/oracle import/input;
- candidate receipts are hash-pinned before the reference process;
- all protected method-module hashes remain unchanged.

Run focused tests plus the relevant existing regional/common-Jacobian regression tests. Model-bearing work runs on the GPU; CPU is only for manifests/report aggregation.

## 9. Disposition

Commit the pre-outcome plan/cohort first, then candidate analysis code/tests, then all candidate receipts, and only afterward the reference analysis/report. Append the exact result to `coordination/CODEX_TO_CHATGPT.md` and stop `NEEDS_REVIEW`.

- **If the frozen gate passes:** nominate `roi_bbox_exposure_stability` for a separately authorized fresh-cohort T025-B K-step/AP study. Do not implement T025-B automatically.
- **If it fails:** close this exact exposure-pair ROI box-geometry objective. Do not rescue it by tuning exposure magnitude, beta, views, ROI layer, regression decoding, class weighting, supports, masks, K/LR or corruption mix. Return for a new research decision; the next family should be materially different, e.g. a clean-source feature anchor/memory rather than another local invariance heuristic.

No official AP, FCOS/SSD evaluation, source/meta training, predictor training, runtime/deployment expansion or protected-module edit is authorized in T025-A.
