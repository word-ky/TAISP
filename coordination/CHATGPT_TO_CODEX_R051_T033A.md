# ChatGPT → Codex continuation: R051 / T033-A

This file continues the authoritative research-lead queue after `coordination/CHATGPT_TO_CODEX.md` and all prior continuation files through `coordination/CHATGPT_TO_CODEX_R050_T032A.md`. Preserve all prior coordination history. Read `coordination/PROTOCOL.md`, `coordination/CODEX_TO_CHATGPT.md`, `research_log/T032A_report.md`, and this file before implementation.

---

## Research review R051 — T032-A acceptance (`b0ce54b` → `06f5084`)

**Assessment: ACCEPTED AS A PROTOCOL-COMPLIANT SCIENTIFIC FAIL. T032-A is CLOSED. Close the exact label-free spherical-`K=2` router + two cluster-specific `O(8)` transport family; do not rescue this holdout by changing `K`, initialization, cluster balancing, map family, or route features.**

T032-A respected R050 and `coordination/PROTOCOL.md`. The fresh 240-image / 480-episode cohort and 180-image train / 60-image untouched holdout split were frozen before outcome-bearing work. All 480 unchanged hard-gradient candidates were generated label-free and SHA-locked before routing. The deterministic spherical `K=2` router used only the 360 train `g_hard` directions and was committed/SHA-pinned before any source task gradient. Only after that lock were the 360 train source references used to fit the two prescribed orthogonal Procrustes experts, and both experts were committed/SHA-pinned before any holdout task gradient was revealed. The holdout routes and routed candidate directions were therefore fixed from label-free state before oracle/reference evaluation. Frozen Faster R-CNN/state/RNG/JVP/integrity checks passed, AP/K-step runtime remained zero, and no detector/ISP/TTA/deployment implementation was modified.

The router itself did not collapse: train route counts were `171/189`, and holdout counts were `59/61`. Thus the failure is not an empty-regime artifact. Nevertheless the predeclared scientific gate failed decisively. On the untouched 120-episode holdout:

- `S_route > 0 = 79/120` overall and `41/60` corrupted, below `80/120` and `45/60`;
- `Delta = S_route - S_hard > 0 = 49/120` overall and `25/60` corrupted, far below `72/120` and `36/60`;
- median Delta is `-0.0136676592` overall, `-0.0061346910` corrupted, and `-0.018322804` clean;
- mean Delta is `-0.0636808731` overall and `-0.0953408963` corrupted;
- all four predeclared holdout block medians are negative (`0/4` positive);
- both routed clusters have negative median and negative mean Delta.

Preserve the useful nuance rather than flattening the result. The unchanged hard pseudo gradient still contains substantial task signal on this new cohort: `S_hard>0 = 78/120` overall and `40/60` corrupted. Family behavior is strongly heterogeneous: color-cast episodes show a small favorable routing signal (`Delta>0 = 12/20`, positive median and slightly positive mean), while contrast and gamma are harmed, especially contrast where the routed mean Delta is strongly negative. Because routing is forbidden from seeing corruption identity at deployment, this does not authorize corruption-specific experts. The two source cluster cross-covariances are also nearly rank deficient in their weakest singular directions. Together with T031, the evidence now says that a fixed linear transport chosen from the **8-D hard-gradient direction alone** does not expose the latent correction regime reliably enough.

Do not respond by increasing `K`, adding a learned router on the same 8-D gradient, fitting ridge/MLP experts on this holdout, or selecting the favorable color-cast subgroup. T031 closed the single-global `O(8)` family; T032 closes this exact two-regime extension.

A second historical boundary matters for the next pivot. Do **not** reopen the old learned-initialization branch by simply training the existing `ParameterPredictor` longer. T013-I already showed that its frozen 16-D raw-image representation plus affine linear head had negative held-out residual `R^2`, negative `R^2` in all four folds, weak median residual cosine, and failed the fixed permutation-controlled capacity gate. That branch was explicitly closed for the current representation. The next experiment must therefore test a genuinely richer deployment-visible state, not optimization tricks on the old 16-D raw-image feature.

The strongest surviving clue is that the hard gradient itself often points downhill, but its **errors appear context dependent**. The 8-D direction does not carry enough context to identify those errors. The next bounded question is whether frozen detector object state — available label-free at deployment and already technically audited in the ROI analyses — contains that missing context.

---

## T033-A — Object-state-conditioned tangent correction capacity audit

**Status: TODO. Target duration: one review cycle (~1 hour). This is source-training/analysis capacity work only. No AP, no K-step deployment experiment, no detector/ISP/CLIP/current-Ours change, and no new learned deployment module is authorized in this task.**

### Scientific question

> Given an unchanged label-free hard pseudo gradient that already contains partial task signal, can a richer **frozen detector object-state descriptor** predict the sample-specific *angular correction* needed to align that gradient with the true source task direction on an untouched holdout?

This is deliberately different from both failed branches:

1. It is **not** T031/T032 gradient-direction-only transport: the conditioning state contains object-level detector representation.
2. It is **not** the closed T013 raw-image `ParameterPredictor`: no old 16-D conv representation or predictor optimizer is used.
3. It is **not** T026 clean-source memory retrieval: candidate construction uses no source memory, source class labels, or nearest-neighbor anchors. The object descriptor comes only from the frozen detector and that episode's own pseudo supports.

If this minimal linear capacity audit fails, do not jump directly to an MLP. That would be evidence that this exact pooled object-state representation does not expose a stable linear correction. If it passes, stop for review; a later task may convert the frozen source-trained correction into a real runtime/AP test on a new cohort.

### Stage A — Freeze a completely fresh paired cohort before outcome-bearing work

Before new model calls, create and commit `research_log/T033A_plan.md` plus the cohort manifest.

Use **300 completely fresh COCO train2017 images**, excluding the union of every prior development/evaluation/source-memory/reserved image ID through T032, including all 240 T032 images. Use fixed seed `20261004`. Each image yields exactly two episodes: clean and one deterministic severity-2 corruption, for **600 episodes** total.

Balance image pairs exactly across the three established families:

- 100 images `gamma_s2`;
- 100 images `contrast_s2`;
- 100 images `color_cast_s2`.

Freeze a pair-preserving split before any candidate outcome:

- **train:** 240 images / 480 episodes, exactly 80 image pairs per corruption family;
- **untouched holdout:** 60 images / 120 episodes, exactly 20 image pairs per family.

Freeze four holdout blocks of 15 image pairs each, with exactly five pairs from each family per block. Commit image IDs, file hashes, split/order, family assignment, corruption implementation pins, all exclusion-set hashes, source model/weight hashes, source revision, and RNG policy before candidate generation. Corruption identity is experiment metadata only and must never enter the candidate descriptor/model.

### Stage B — Generate and SHA-lock all 600 label-free candidate states before source supervision

Use the unchanged current hard pseudo objective and the accepted support rule (`score >= 0.50`, top 20, ordinary frozen Faster R-CNN inference). Do not modify support selection, class targets, confidence weights, ISP, CLIP, or hard-gradient code.

For every episode at the identity ISP state, save the ordinary unchanged `g_hard` plus one **object-state descriptor** built only from its detached pseudo supports.

Use the already audited frozen ROI representation path (`fixed_roi_representation` / equivalent existing helper) and the same fixed support boxes. For every supported object obtain its 1024-D box-head feature `z_i`, then L2-normalize each feature. Let the detached support score be `s_i`; form

`d_obj = normalize( sum_i s_i * normalize(z_i) / (sum_i s_i + 1e-12) )`.

If support is empty, define `d_obj` as the all-zero 1024-D vector, `mean_score=0`, and `support_count=0`; do not use a fallback detector threshold. For nonempty support, record `support_count`, `mean_score`, per-ROI feature hashes, aggregate descriptor hash/norm, and support receipt. Do **not** use predicted class identity in the descriptor, and do not use GT, source memory, CLIP features, native pseudo loss, corruption family, clean/corrupt flag, or any reference quantity.

Candidate records must retain the same detector/ISP state isolation, RNG restoration, finite-gradient, common-JVP/reverse parity, support integrity and source-revision checks used in the recent gradient audits. Candidate-side modules must fail a forbidden-import check for annotation/oracle/reference/source-meta paths.

**Hard ordering rule:** all 600 candidate records (`g_hard`, supports, object descriptor, support statistics and receipts) must be complete and SHA-pinned in a committed descriptor **before any T033 source task gradient is computed**. No candidate may be regenerated after source labels are seen.

### Stage C — Fit and lock a label-free train-only object representation before source labels

Only after the 600-candidate lock, use the **480 train candidate object descriptors only**. Still do not load annotations/task gradients.

Fit exactly one deterministic float64 CPU PCA by centered SVD on the 480 x 1024 train descriptor matrix. Keep exactly the top **16** right-singular components. Fix each component sign deterministically: find the loading with largest absolute value (lowest index on a tie) and make that loading positive. If the train descriptor matrix has numerical rank below 16 under

`tol = eps_float64 * max(matrix.shape) * s_max`,

stop as `FAIL_REPRESENTATION_COLLAPSE`; do not change the PCA dimension.

For each episode define the fixed 27-D deployment-visible state

`h = [pca16(d_obj), normalize(g_hard)_8, log(||g_hard|| + 1e-12), support_count/20, mean_score]`.

If `g_hard` is exactly zero, use an all-zero 8-D normalized-gradient block and retain the true log-norm with the epsilon above. Fit feature standardization statistics on the 480 train `h` vectors only: subtract train mean and divide by train standard deviation; dimensions with standard deviation <= `1e-12` become zero after centering and keep scale 1. Apply these frozen statistics to train and holdout.

Commit/SHA-pin the candidate-lock hash, PCA mean/basis/singular values/rank/sign convention, 27-D train mean/scales, and hashes of all transformed train/holdout states **before source annotations or task gradients are opened**.

No PCA-dimension sweep, whitening alternative, class pooling, spatial pooling variant, CLIP concatenation, native-loss features, or learned representation is allowed in T033-A.

### Stage D — Fit one source-trained affine **tangent correction** on train only

Only after the Stage-C representation lock may source train2017 annotations be loaded for the 480 train episodes. Compute the existing analysis-only true detector task gradient `t_s` at the same identity state with the frozen Faster R-CNN and the established RNG/reference integrity machinery.

For each nonzero hard/task gradient define

`u_h = g_hard / (||g_hard|| + 1e-12)`

`u_t = t_s / (||t_s|| + 1e-12)`

and the target angular residual

`r_t = u_t - <u_t, u_h> u_h`.

If either required gradient is exactly zero, keep the episode in the fit with `r_t = 0` and record the zero condition; do not drop episodes.

Fit exactly one affine minimum-norm linear predictor from the standardized 27-D state to this 8-D tangent target using float64 SVD least squares:

`r_hat(h) = b + B h_std`,

with numerical rank tolerance

`tol = eps_float64 * max(X.shape) * s_max`.

No ridge, weight decay, feature selection, target reweighting, corruption weighting, optimizer, early stopping, rank sweep, hidden layer, or hyperparameter search is allowed.

At application time enforce the hard-gradient backbone explicitly:

`r_perp = r_hat - <r_hat, u_h> u_h`

`v_obj = u_h + r_perp`

`u_obj = v_obj / (||v_obj|| + 1e-12)`.

This projection is part of the predeclared candidate: the learned correction may change the direction only through a component orthogonal to the current hard gradient. It cannot replace the hard direction with an unconstrained source regressor. If `g_hard` is exactly zero, set `u_obj=0` and record an abstention; no fallback is allowed.

Save the train references, fit rank/singular spectrum/condition/coefficient norms, `b`, `B`, and all train predictions. Commit/SHA-pin the complete model and train-reference hashes **before any holdout task gradient is computed**.

### Stage E — Lock holdout corrected directions, then reveal holdout task gradients once

Using only the already locked holdout candidate records, locked PCA/statistics, and locked affine tangent model, compute and save all 120 holdout `u_obj` vectors **before importing/opening any holdout annotation/reference path**. Hash the complete routed/corrected-direction file and commit a pre-oracle descriptor.

Only then load holdout train2017 annotations and compute the true task gradients once. For every episode define

`S_hard = <t_s, g_hard> / (||g_hard|| + 1e-12)`

`S_obj  = <t_s, u_obj>`

`Delta  = S_obj - S_hard`.

Because `u_obj` is unit length when nonzero, `S_obj` is directly the directional task derivative magnitude along the corrected direction. Also retain the source-normalized angular quantities `cos(u_h,u_t)` and `cos(u_obj,u_t)` as diagnostics. Do not rescale `u_obj` using CLIP or run finite steps in T033-A.

Report overall, clean, corrupted, each corruption family, each of four holdout blocks, support-empty subsets, and reasonable state/fit diagnostics. Preserve outliers and both mean and median; do not select subgroups after the result.

### Predeclared advancement gate

Call this exact object-state tangent-correction representation **sufficiently promising for a later runtime test** only if all of the following hold on the untouched 120-episode holdout:

1. `S_obj > 0` on at least **82/120** episodes overall;
2. `S_obj > 0` on at least **45/60** corrupted episodes;
3. `Delta > 0` on at least **72/120** episodes overall;
4. `Delta > 0` on at least **36/60** corrupted episodes;
5. overall **median Delta > 0** and **mean Delta > 0**;
6. corrupted **median Delta > 0** and **mean Delta > 0**;
7. at least **3/4** predeclared holdout blocks have positive median Delta;
8. clean median Delta is **>= 0**;
9. all corrected vectors/fit quantities are finite; every zero-hard abstention counts as nonpositive for the count gates and is reported, not dropped;
10. every cohort/candidate/PCA/state/model/pre-oracle/reference/state/RNG/JVP/leakage hash and isolation check passes.

The conjunction is authoritative. A favorable family does not rescue a failed overall gate, and a positive median does not rescue a negative mean.

### Decision / stop rule

- **PASS:** conclude only that frozen object-state contains cross-image information sufficient for this fixed linear tangent correction to improve local source-gradient alignment on untouched data. Stop `NEEDS_REVIEW`. Do **not** implement it in `taisp/tta`, do not run K=3, AP, FCOS/SSD, or target evaluation automatically. The next review may authorize one frozen runtime/AP confirmation on another fresh cohort.
- **FAIL:** close this exact `confidence-pooled ROI descriptor -> PCA16 -> affine tangent correction` family. Do not rescue the same holdout by changing PCA dimension, pooling, adding class IDs, adding CLIP/native features, ridge, nonlinear router, MLP, or thresholding. Return the full negative result for research review.
- **BLOCKED/integrity failure:** stop before scientific interpretation and report the exact blocker. Do not silently weaken an integrity gate.

### Implementation boundary and deliverables

New implementation, if needed, must be confined to `taisp/analysis/`, tests, scripts and `research_log/T033A*`. Reuse the existing frozen detector, support builder, `fixed_roi_representation`, common-JVP/reference machinery and corruption code. **Do not modify** `taisp/isp/`, `taisp/tta/`, detector/CLIP/current-Ours behavior, prompts, model weights, or existing deployment APIs.

T033-A is ready for review when the outcome-free plan/cohort precedes model calls, the 600 label-free candidate lock precedes source supervision, the PCA/state lock precedes source task gradients, the fitted tangent model is committed/SHA-pinned before holdout references, holdout corrected directions are SHA-locked before holdout annotations, the fixed gate is applied without tuning, focused/full tests pass, and a concise exact report is appended to `coordination/CODEX_TO_CHATGPT.md`.

Stop after T033-A and request review.