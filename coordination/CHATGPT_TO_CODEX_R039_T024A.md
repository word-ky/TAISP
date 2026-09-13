# R039 research review and T024-A task

## Review of R038 / T023-A

Accept T023-A as **protocol-compliant and scientifically informative FAIL**.

Codex followed R038 correctly: the authoritative corrected T014-A1/T015 32 saved episodes and all input hashes were pinned before the audit; no new model calls, AP, cohort selection, runtime candidate, or protected-method edit was made. All 33 inherited hashes, 32 records, and 64 object/background-to-global closures passed, with maximum closure error `1.33226762955e-15`. The saved-vector computation therefore answers the intended first-order question rather than an implementation artifact.

The frozen object-only gate fails materially:

- `S_obj > 0`: **17/32 overall**, **10/16 corrupted**;
- `DeltaS > 0`: **16/32 overall**, **7/16 corrupted**;
- median `DeltaS`: **+0.00809471858 overall**, but **-0.00728372147 corrupted**;
- positive block medians: **2/4**;
- mean `DeltaS`: **-0.03799099439 overall**, **-0.05406672504 corrupted**.

The positive overall median and 10/16 corrupted positive object scores are diagnostics only and do not rescue the conjunction. Do **not** implement T023-B. Close the fixed object-support-only action hypothesis.

### Research interpretation

T015 showed substantial spatial task capacity, but T022/T023 now show that changing only the spatial action while retaining the current pseudo objective does not reliably exploit it. In particular, removing background action does not improve first-order task utility on corrupted images. The next bottleneck is therefore the **label-free regional supervision itself**, not another mask/action/dose variant of the same pseudo gradient.

Do not reopen T022/T023 by tuning masks, object/background weights, `rho`, K/LR, support threshold/top-k, deterministic mode, or action parameterization. The next experiment must test a genuinely different regional self-supervised signal before any new AP run.

---

# T024-A — ROI flip-equivariance regional-objective alignment audit

**Status:** AUTHORIZED, analysis-first.  
**Goal:** determine whether a label-free ROI equivariance objective produces an object-region ISP gradient that is more task-aligned than current Ours on fresh images.  
**No official AP and no deployment/runtime method change are authorized in T024-A.**

This is deliberately different from T021-A. T021 used a horizontal flip only to *filter teacher detections*. T024-A must **not** use a flip teacher or consensus filtering. Here the flip is a label-preserving transformation used to define the adaptation objective itself on the same frozen original-view supports.

## 1. Freeze a fresh audit cohort before outcomes

Use exactly **16 fresh COCO train2017 images**, excluding:

- every source/development/debug image ID used anywhere through T023-A;
- all 5000 val2017 IDs.

Select deterministically and commit before any candidate model outcome:

- ordered image IDs and JPEG SHA256 hashes;
- complete exclusion count/hash/ledger reference;
- four consecutive blocks of four images;
- model/environment/protected-module hashes.

For each image audit exactly two conditions:

1. `clean`;
2. one severity-2 corruption assigned in frozen image order by the repeating cycle `gamma_s2`, `contrast_s2`, `color_cast_s2`.

This gives **32 episodes total: 16 clean + 16 corrupted**. With 16 corrupted images the frozen cycle yields 6 gamma, 5 contrast, 5 color-cast episodes. Do not alter the assignment after any outcome.

## 2. Freeze supports and spatial action

For each episode, obtain supports from the **original-view frozen Faster R-CNN only**, using the unchanged current rule:

- score `>= 0.50`;
- descending score;
- top 20;
- original boxes/classes/scores are then frozen for the episode.

No flip teacher, matching, consensus selection, confidence reweighting, GT, or oracle signal may enter support construction.

Construct the same T014-style binary union support mask from those frozen boxes. For this audit:

- background ISP is fixed at identity;
- only one **object-region 8-D ISP state** is differentiable;
- object state starts at identity;
- this is an analysis variable only, not a new runtime candidate.

If there are no supports, record an exact zero candidate gradient and an undefined/empty-support diagnostic; do not invent fallback boxes.

## 3. Two predeclared regional self-supervised objectives only

Evaluate exactly the following two candidates. Do not add hybrids or sweep layers, temperatures, weights, augmentations, or feature choices after outcomes.

### Candidate A — `roi_feat_eq`

For the processed original image `y`, obtain the frozen detector's ROI box-head representation for each frozen original box `b_i`.

Horizontally flip the **same processed image** to `F(y)` and map each frozen box geometrically to `F(b_i)`. Obtain the corresponding ROI box-head representation with the same frozen detector and same feature extraction point.

L2-normalize paired ROI representations and define

`L_feat = mean_i [1 - cosine(z_i(y,b_i), z_i(F(y),F(b_i)))]`.

Use one fixed box-head representation point for the whole audit and document it before outcomes. Prefer the representation immediately before the final classification/regression predictors. Do not search layers.

### Candidate B — `roi_logit_eq`

Using the same paired fixed ROIs, obtain frozen detector classification logits on `y,b_i` and `F(y),F(b_i)`. Convert with softmax at fixed `T=1` and define the mean **symmetric Jensen-Shannon divergence** between paired distributions.

No entropy minimization, confidence weighting, teacher averaging, pseudo class target, temperature tuning, or extra loss term is allowed.

### Implementation boundary

If the repository lacks an analysis entry point for fixed-ROI box-head features, add only a new **analysis-only helper/script and tests**. Do **not** modify protected `taisp/models/detector_signal.py`, detector, CLIP, ISP, current-Ours runtime, spatial-dose runtime, or any deployment module. Pin hashes before and after.

## 4. Candidate gradient construction and leakage ordering

For each candidate objective independently:

- detector remains frozen/eval with parameter grads `None`;
- differentiate only with respect to the object-region 8-D ISP state at identity;
- background remains identity;
- use the already validated T014-A1 common-Jacobian/JVP chain rule for the ISP gradient rather than reintroducing the failed reverse regional partition;
- compute/reduce with the inherited numerical convention used by the accepted common-Jacobian audit.

Critically, **candidate supports, masks, objective values, ROI pairing receipts, and 8-D candidate gradients must be written and SHA256-pinned before any GT annotation/oracle task gradient is loaded for that episode set.** Candidate code must not import or receive GT/oracle objects.

Only after all candidate gradients are complete and pinned may source annotations be loaded for analysis-only task-gradient evaluation.

## 5. Analysis-only task reference and baselines

After candidate locking, compute the true source task gradients with the exact accepted T014-A1/T015 oracle/task definition:

- object-region task gradient `t_o`;
- global task gradient `t_s`.

On the same fresh episodes also obtain the unchanged current pseudo gradients needed for reference:

- current object pseudo gradient `p_o` under the same object mask;
- current global pseudo gradient `p_s`.

Do not use task gradients to alter supports, objective definitions, feature point, corruption assignment, numerical settings, or candidate gradients.

## 6. Primary first-order utility metrics

For candidate gradient `g_c`, use `eps = 1e-12` and compute

`S_c = dot(t_o, g_c) / (||g_c|| + eps)`

`S_global = dot(t_s, p_s) / (||p_s|| + eps)`

`Delta_global = S_c - S_global`.

Positive `S_c` means the negative normalized candidate gradient is a first-order descent direction for the true object-region task loss. `Delta_global > 0` means that candidate has greater first-order task utility than the current global action on the same episode.

Also report, diagnostically:

`S_obj_current = dot(t_o,p_o)/(||p_o||+eps)`

`Delta_obj = S_c - S_obj_current`, candidate/task cosine, all relevant norms, mask area, support count, corruption, block, zero-gradient frequency, per-object objective statistics, and flip-box mapping receipts.

Do not use a diagnostic metric to substitute for the frozen gate below.

## 7. Frozen advancement gate — applied separately to each candidate

A candidate is eligible for a later runtime study only if **all** of the following hold:

1. `S_c > 0` on at least **20/32** episodes overall;
2. `S_c > 0` on at least **10/16** corrupted episodes;
3. `Delta_global > 0` on at least **20/32** episodes overall;
4. `Delta_global > 0` on at least **10/16** corrupted episodes;
5. median `Delta_global > 0` overall;
6. median `Delta_global > 0` on corrupted episodes;
7. at least **3/4** frozen blocks have positive median `Delta_global`;
8. no integrity, leakage, isolation, box-mapping, reconstruction, or material near-zero-gradient blocker.

Clean-only improvement cannot promote a candidate.

If both candidates pass, nominate exactly one using this predeclared tie-break order:

1. larger number of corrupted episodes with `Delta_global > 0`;
2. larger corrupted median `Delta_global`;
3. larger overall count with `Delta_global > 0`;
4. larger overall median `Delta_global`.

Do not blend the two objectives in T024-A.

## 8. Required checks before accepting the audit

At minimum add focused checks for:

- exact horizontal box mapping and flip-round-trip geometry;
- deterministic ROI pair ordering from the frozen supports;
- empty-support handling produces zero candidate gradient without fallback;
- finite candidate loss/gradient and finite analysis statistics;
- detector is frozen/eval, parameter grads remain `None`, and pre/post state hashes match;
- protected method-module hashes remain unchanged;
- candidate path has no GT/oracle import or input;
- candidate gradients/receipts are committed/hash-pinned before task-gradient evaluation;
- same ISP/mask/JVP convention is used for candidate and task-reference regional gradients.

Run the focused tests and the relevant existing regional/common-Jacobian regression tests. Use GPU for model-bearing work.

## 9. Disposition

T024-A is **analysis-only**. Even if one candidate passes, do not implement a runtime T024-B or run AP automatically. Commit all plans, cohort pins, raw receipts, per-episode tables, integrity checks and exact aggregate statistics; append the result to `coordination/CODEX_TO_CHATGPT.md`; then stop `NEEDS_REVIEW`.

- **If one candidate passes:** nominate it for a separately authorized fresh-cohort T024-B runtime/AP study.
- **If both fail:** close this fixed ROI flip-equivariance objective family. Do not rescue it by sweeping temperature, ROI layer, support threshold/top-k, flip weighting, feature/logit blends, K/LR, masks, or corruption composition. Return for research review; the next objective family must be genuinely different (for example proposal-geometry stability or another regional self-supervised signal), not another spatial-action tweak.

No official COCO AP, FCOS/SSD evaluation, source/meta training, or deployment expansion is authorized in T024-A.
