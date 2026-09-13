# R042 — T027-A: Horizontal-Flip Gradient-Consensus Reliability Audit

## Research review of T026-A

Accept T026-A as **protocol-compliant and scientifically informative FAIL**.

The R041 ordering was respected: the 24-image audit cohort and deterministic source-memory selection were frozen first; the complete balanced 1280-entry memory was generated and SHA-pinned before candidate outcomes; all 48 candidate supports/retrievals/anchors/8-D gradients were then completed and pinned before the separate reference process loaded audit annotations. The audit-time path used only frozen-detector predicted classes, the detector remained frozen/eval, no AP/K-step runtime was run, and protected detector/ISP/CLIP/deployment modules were unchanged. The added code stayed within the authorized analysis/source-memory scope, with full regression passing.

The frozen gate failed:

- `S_mem > 0`: **26/48 overall**, **18/24 corrupted**;
- `Delta_global > 0`: **24/48 overall**, **12/24 corrupted**;
- median `Delta_global`: **+0.0014528 overall**, **+0.0014652 corrupted**;
- mean `Delta_global`: **-0.02298 overall**, **-0.04111 corrupted**;
- positive block medians: **2/4**.

The useful signal is limited but worth interpreting correctly: the clean-source memory direction is task-descending on 18/24 corrupted episodes, yet it does not beat the current global pseudo direction consistently and has harmful outliers large enough to make the means negative. Therefore close the exact class-conditional nearest-memory objective exactly as R041 required. Do not rescue it with memory size/top-k/layer/projection/source selection/weights/support/mask/K/LR sweeps or objective blending.

Also record a research-history correction without rewriting prior files: the authoritative T025 receipts are `S_geom>0 = 28/48 overall, 15/24 corrupted`, `Delta_global>0 = 23/48, 14/24`, median `Delta_global = -0.0195430 overall, +0.00938987 corrupted`, and `2/4` positive block medians. R041 quoted different T025 summary numbers, but both versions imply FAIL and T026-A was independent of that discrepancy.

### Interpretation

After several objective/action variants, the strongest remaining diagnosis is **reliability**, not raw gradient availability. Current pseudo gradients contain genuine task-useful signal but are inconsistent; source memory, flip feature/logit equivariance, box-geometry stability, object-only reuse, and spatial-dose variants have not produced a stable replacement direction. The next experiment should therefore test whether two label-free views can tell us when the current pseudo direction is trustworthy, while returning to the source-free deployment assumption.

T027-A is not another AP run and not another source-memory objective. It asks whether the unchanged current pseudo gradient, estimated independently on an image and its horizontal flip, has a useful **view-consensus structure**. Horizontal flip is label-preserving and should commute with the global ISP state. If the two 8-D directions agree, they may be two noisy observations of the same useful test-time direction; if they disagree, that disagreement may itself be a label-free reliability signal.

---

# T027-A — horizontal-flip gradient-consensus reliability audit

## Goal

Test two predeclared hypotheses on a fresh cohort:

1. A symmetric consensus of original-view and horizontal-flip current-pseudo gradients is more task-aligned than the original current-pseudo gradient alone.
2. The label-free cosine agreement between the two view gradients predicts whether the original current-pseudo direction is task-useful.

This round is **analysis only**. No AP, no K-step adaptation, no FCOS/SSD, no source memory, and no deployment-method modification are authorized.

## A. Fresh cohort and exclusions

Use **24 brand-new COCO train2017 images**, four fixed blocks of six. Each image contributes `clean` plus exactly one frozen severity-2 corruption, giving **48 episodes** total: 8 gamma, 8 contrast, 8 color-cast corrupted episodes.

Before any model outcome:

- use `research_log/T026A_source_exclusion_manifest.json` as the cumulative train2017 exclusion source, so all prior source/audit/debug images **and all 1187 T026-A memory images** remain excluded;
- exclude all COCO val2017 images;
- freeze ordered image IDs/JPEG hashes, block membership, corruption assignment, selection seed, model seed, protected-module hashes and environment assumptions;
- use a new deterministic cohort-selection salt; do not select or replace images from outcomes.

## B. ISP/flip commutation preflight

Before candidate model outcomes, add a pure numerical test that the unchanged **global** ISP parameterization is horizontal-flip equivariant:

`flip(G(x, phi)) == G(flip(x), phi)`

for identity and several fixed nonzero in-range 8-D states on synthetic tensors. Reuse the existing processed-image numerical tolerance (`atol=2e-7`, `rtol=1e-6`). If a material failure occurs, stop `BLOCKED`; do not change ISP operators to make this task pass.

## C. GT-free two-view candidate stage

For each episode image `x`, define `x_f = horizontal_flip(x)`.

For each view **independently**:

1. Run the unchanged frozen Faster R-CNN teacher on the raw view.
2. Select supports with the unchanged current rule: score `>=0.50`, stable descending score, top 20.
3. Freeze that view's support boxes/classes/scores; do not match, filter or transfer supports across views.
4. At global ISP identity, compute the unchanged current fixed-ROI pseudo-loss gradient in the same 8-D ISP coordinate system:
   - `g_o` from the original view;
   - `g_f` from the flipped view.
5. No CLIP magnitude is needed for this direction/alignment audit. Do not change the current pseudo objective.

Because the ISP coordinates are global and the commutation preflight must pass, **do not remap ISP coordinates** for `g_f`.

Define `eps = 1e-12` and:

- if `||g_o|| <= eps` or `||g_f|| <= eps`, set `agreement = -1` and `g_cons = 0` with an explicit zero/abstention receipt;
- otherwise `u_o = g_o / ||g_o||`, `u_f = g_f / ||g_f||`, `agreement = <u_o,u_f>`, `q = u_o + u_f`;
- if `||q|| <= eps`, set `g_cons = 0`; otherwise `g_cons = q / ||q||`.

This is a symmetric, parameter-free consensus. No learned weighting, threshold, confidence weighting, support consensus, fallback direction, memory, CLIP, geometry, feature/logit auxiliary loss or corruption-aware branch is allowed.

For every episode save at least: both support receipts/hashes, both 8-D gradients and norms, `agreement`, `g_cons`, zero/abstention flags, detector/module/environment hashes and frozen/eval/state checks.

Complete and SHA-pin **all 48 GT-free candidate records** before any reference process loads audit annotations.

## D. Post-lock reference stage

Only after the candidate lock, use the unchanged validated source-reference machinery on the **original-view episode** to compute the authoritative global task gradient `t_s`. Audit GT is analysis-only and must not affect either view's supports, gradients, agreement or consensus.

For each episode compute:

- `S_orig = <t_s, g_o> / (||g_o|| + eps)`;
- `S_cons = <t_s, g_cons> / (||g_cons|| + eps)`, with exact zero defined as `0`;
- `Delta_cons = S_cons - S_orig`;
- `S_flip = <t_s, g_f> / (||g_f|| + eps)` as diagnostic only.

Report overall, clean, corrupted, each corruption family and each block: positive counts/fractions, mean/median/quantiles for `S_orig`, `S_cons`, `Delta_cons`, agreement distributions, zero/abstention frequency and all integrity/JVP/isolation checks.

## E. Frozen decision rules

### E1. Consensus-direction gate

The consensus direction is developmental-PASS only if **all** hold:

1. `S_cons > 0` on at least **30/48 overall** and **15/24 corrupted** episodes.
2. `Delta_cons > 0` on at least **30/48 overall** and **15/24 corrupted** episodes.
3. Overall median `Delta_cons > 0`.
4. Corrupted median `Delta_cons > 0`.
5. At least **3/4 blocks** have median `Delta_cons > 0`.
6. No leakage, commutation, detector-state, nonfinite, JVP/isolation or systematic-zero blocker.

A PASS still does **not** authorize AP/K-step runtime in this round. Stop `NEEDS_REVIEW` for a separate fresh-cohort runtime decision.

### E2. Reliability-signal gate

Before loading GT, compute and SHA-pin the candidate-only **overall median agreement** across the 48 episodes. After reference is available, use that pinned median only to split episodes into high-agreement and low-agreement halves; do not tune a threshold against task labels.

Treat view agreement as a potentially useful reliability statistic only if **all** hold:

1. Spearman correlation between `agreement` and `S_orig` is at least **+0.30 overall** and **+0.30 on corrupted episodes**.
2. The high-agreement half has at least **+0.20 higher fraction of `S_orig>0`** than the low-agreement half, both overall and within corrupted episodes.
3. Median `S_orig` is strictly higher in the high-agreement half than the low-agreement half, both overall and corrupted.
4. At least **44/48 overall** and **22/24 corrupted** episodes have nonzero gradients in both views; otherwise the reliability estimate is too sparse.

Bootstrap confidence intervals may be reported as diagnostics but do not replace these frozen criteria.

### Decision

- If **E1 passes**, report consensus as the nominated developmental direction and stop for review; no AP yet.
- If **E1 fails but E2 passes**, do **not** rescue the consensus direction and do not run AP. Report that agreement may be useful only as a reliability/gating signal; a later task, on a new cohort, may predeclare a gating rule.
- If **both E1 and E2 fail**, close horizontal-flip gradient consensus/reliability as a useful family under the current pseudo objective. Do not sweep agreement thresholds, support thresholds/top-k, augmentations, K/LR, ISP ranges or consensus weights.

## F. Authorized code scope

Only new analysis helpers/runners/configs/tests/reports needed for T027-A are authorized. Reuse existing frozen detector, ISP and current-pseudo implementations. **Do not modify protected detector, ISP, CLIP, current-Ours runtime/deployment, source-memory or prior-study implementation modules.**

## Required handoff

Append a concise T027-A completion record to `coordination/CODEX_TO_CHATGPT.md` with exact commits/run IDs, cohort hash, commutation test, candidate-before-GT lock, full E1/E2 metrics, integrity checks and decision. Preserve all raw receipts and stop `NEEDS_REVIEW` or `BLOCKED` as specified.