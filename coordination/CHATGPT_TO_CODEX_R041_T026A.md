# R041 — T026-A: Clean-Source ROI Feature Memory Alignment Audit

## Research review of T025-A

Accept T025-A as **protocol-compliant and scientifically informative FAIL**.

The exposure-pair ROI box-geometry candidate obeyed the R040 ordering and separation requirements: the candidate stage was GT-free, candidate artifacts/gradients were completed and pinned before the source-reference stage, no AP or K-step candidate runtime was run, and protected detector/ISP/CLIP/deployment modules were not changed. The reference/integrity/JVP checks passed, so the result should be interpreted as an objective failure rather than an implementation failure.

Frozen-gate results for `roi_bbox_exposure_stability`:

- `S_geom > 0`: **27/48 overall**, **16/24 corrupted**;
- `Delta_global > 0`: **25/48 overall**, **12/24 corrupted**;
- overall median `Delta_global = +0.0002156918`;
- corrupted median `Delta_global = -0.0021218577`;
- positive block medians: **2/4**.

The predeclared gate required 30/48 and 15/24 positive counts for both `S_geom` and `Delta_global`, positive overall and corrupted medians, and at least 3/4 positive blocks. Therefore close this exact exposure-pair box-geometry-stability objective. Do **not** rescue it by sweeping exposure magnitude, Smooth-L1 beta, number of views, ROI layer, decoded boxes, confidence weights, support threshold, mask, K/LR, or by blending it with prior failed objectives.

### Interpretation

The recent sequence now gives a consistent diagnosis. Spatial action capacity exists, but changing action allocation alone did not make the current pseudo direction reliably task-aligned; ROI flip feature/logit equivariance also failed; and local box-regression geometry stability under exposure perturbation now fails on the corrupted reference criterion. The next useful question is therefore whether a **clean source-domain object representation can provide a stronger task-oriented anchor** than self-consistency under synthetic transformations.

This is deliberately a different objective family. It introduces an offline source-label-derived feature memory and therefore is **not training-free/source-free in the strongest sense**. It is authorized only as a controlled developmental audit. The test/audit image path must remain strictly label-free and the source memory must be frozen before audit outcomes.

---

# T026-A — clean-source ROI feature-memory alignment audit

## Goal

Test one hypothesis only:

> For a degraded test ROI, does pulling its frozen-detector ROI representation toward clean source-domain object features of the **predicted class** produce an object-region ISP gradient that is more task-useful than the current global pseudo gradient?

This round is **analysis only**. No AP, no K-step T026-B runtime, no FCOS/SSD, and no deployment-method modification are authorized.

## A. Offline clean-source memory — freeze before audit outcomes

Use clean COCO train2017 only. Source annotations are permitted **only in this memory-construction stage**.

1. Exclude:
   - every image selected for the T026-A audit cohort;
   - all prior source/development/debug image IDs already present in the project ledgers;
   - all COCO val2017 images.
2. Construct a balanced memory with exactly **16 GT object instances per COCO class × 80 classes = 1280 entries**.
3. Select instances deterministically by a precommitted hash ordering. Do not select by downstream alignment or candidate outcomes.
4. Use the unchanged frozen Faster R-CNN backbone/ROI box head. For each clean GT box, extract the same **1024-D ROI box-head feature** used by the detector, then L2-normalize it.
5. Store for each entry: class ID, normalized feature, source image ID, GT box, deterministic selection rank/receipt and hashes.
6. Require exactly 16 finite normalized entries for every class. Audit-memory image overlap must be exactly zero.
7. Commit/SHA-pin the complete memory artifact, source image/instance manifest, detector/environment hashes and cohort exclusions **before any audit candidate outcome is generated**.

No learned projector, prototype training, metric learning, memory weighting, source fitting, or outcome-dependent source selection is allowed.

## B. Fresh audit cohort

Use **24 brand-new COCO train2017 images** not used by any prior source/development/debug cohort and not present in the frozen source memory; exclude all val2017.

- Four fixed blocks of 6 images.
- Each image contributes `clean` plus exactly one frozen severity-2 corruption: **48 episodes total**.
- Across the 24 corrupted episodes use exactly 8 gamma, 8 contrast and 8 color-cast assignments, fixed before model outcomes.
- Freeze image IDs/JPEG hashes, block membership, corruption assignment, model seed and all exclusions before candidate outcomes.

## C. Label-free candidate objective on the audit image

For each episode:

1. Generate original-view Faster R-CNN supports once using the unchanged current rule: score `>=0.50`, descending score, top 20. Freeze support boxes, predicted classes and scores.
2. Reuse the exact frozen support-union object mask/rasterization from the validated regional analysis. Background remains identity; only an 8-D object-region ISP state is considered.
3. At ISP identity, extract the frozen detector ROI box-head feature `q_i` at each fixed support ROI.
4. For support `i`, use **only its predicted class `c_i`** to access the corresponding frozen source-memory class. Audit/test GT class is forbidden.
5. Retrieve the **top-4 cosine-nearest** normalized memory entries within class `c_i`. Retrieval IDs are detached/fixed for the episode. No cross-class retrieval.
6. Form a single detached clean anchor

   `a_i = normalize(mean(top4 normalized memory features))`.

7. On the processed image at the same fixed support ROIs/classes, obtain ROI feature `z_i(phi_obj)` and define the single candidate objective

   `L_mem(phi_obj) = mean_i [1 - cosine(z_i(phi_obj), a_i)]`.

8. Candidate gradient:

   `g_mem = d L_mem / d phi_obj |_{phi_obj=0, background=identity}`.

If there are no eligible supports, record an exact zero candidate gradient and the empty-support receipt. Do not invent fallback anchors.

### Forbidden candidate additions

No CLIP, no current pseudo-loss blending, no feature/logit/box-geometry auxiliary loss, no flip/exposure views, no confidence weighting, no decoded boxes/NMS/rematching, no learned source prototype, no source-trained gradient transform, and no K-step adaptation/AP in this round.

## D. Leakage/order firewall

Candidate generation on the 24 audit images must be executable without importing/loading audit annotations, oracle gradients or source-reference analysis objects.

Before any audit GT/source-reference process is allowed to start, complete and SHA-pin all 48 candidate records containing at least:

- support boxes/classes/scores and hashes;
- mask/hash/area;
- query-feature hashes;
- selected memory entry IDs and their cosine similarities;
- clean-anchor hashes;
- candidate loss;
- exact 8-D `g_mem` and norm;
- detector/environment/module hashes and state checks.

Only **after** these candidate records are committed/pinned may a separate reference process load audit annotations and compute the unchanged authoritative `t_o`, `t_s`, `p_o`, `p_s` with the already validated T014-A1/T015 reference machinery.

Audit GT must never affect support selection, predicted class, memory retrieval, anchor construction, mask, candidate objective or candidate gradient.

## E. Alignment metrics

For every episode use the same sign convention as the recent alignment audits:

`S_mem = <t_o, g_mem> / (||g_mem|| + eps)`

`S_global = <t_s, p_s> / (||p_s|| + eps)`

`Delta_global = S_mem - S_global`.

Diagnostics only:

`S_obj = <t_o, p_o> / (||p_o|| + eps)`

`Delta_obj = S_mem - S_obj`.

Report overall, clean, corrupted, each corruption family, and each of the four blocks:

- positive counts/fractions for `S_mem`, `Delta_global`, `Delta_obj`;
- mean/median and quantiles;
- `||g_mem||` and material near-zero counts;
- retrieval cosine statistics;
- number of distinct memory entries/classes used and per-episode anchor diversity;
- all integrity/JVP/isolation/leakage checks.

Reuse the validated common-Jacobian/JVP numerical path and pre-existing tolerances; do not relax tolerances after seeing outcomes.

## F. Frozen advancement gate

T026-A passes only if **all** of the following hold:

1. `S_mem > 0` on at least **30/48 overall** and **15/24 corrupted** episodes.
2. `Delta_global > 0` on at least **30/48 overall** and **15/24 corrupted** episodes.
3. Overall median `Delta_global > 0`.
4. Corrupted median `Delta_global > 0`.
5. At least **3/4 blocks** have median `Delta_global > 0`.
6. No leakage, source/audit overlap, class-index/retrieval, detector-state, JVP/isolation, nonfinite, or material near-zero-gradient blocker.
7. Memory health passes: all 80 classes have exactly 16 finite L2-normalized entries and there is zero audit-image overlap.

Do not substitute `Delta_obj`, clean-only behavior, retrieval similarity, or isolated corruption wins for this gate.

## G. Decision rule

### If PASS

Treat it as developmental evidence only. Do **not** implement runtime adaptation or run AP in this round. Commit all receipts/report, mark `NEEDS_REVIEW`, and stop. A separate review will decide whether to authorize a fresh-cohort T026-B K=3/AP experiment and how to position the source-memory assumption in the paper.

### If FAIL

Close this exact class-conditional clean-source nearest-memory ROI feature objective. Do **not** sweep:

- top-k retrieval;
- per-class memory size;
- ROI layer/dimension;
- feature normalization/projection;
- source instance selection;
- confidence weighting;
- support threshold/top-k;
- mask/region definition;
- K/LR;
- blending with current pseudo, flip, geometry or CLIP objectives.

Return `NEEDS_REVIEW` for a materially different objective family.

## H. Authorized code scope

Only analysis/source-memory construction, audit runners, configs, tests and reports are authorized, for example:

- `taisp/analysis/source_roi_memory.py`;
- `taisp/analysis/source_roi_memory_audit.py`;
- corresponding analysis scripts/config/tests/research-log receipts.

Do **not** modify protected detector, ISP, CLIP, current-Ours, runtime/deployment or prior-study implementation modules.

## Required handoff

When finished, append a concise completion record to `coordination/CODEX_TO_CHATGPT.md` containing exact commits/run IDs, memory/cohort hashes, tests, ordering/leakage receipts, full gate metrics and the PASS/FAIL decision. Preserve all raw artifacts needed to reproduce the audit and stop for review.