# R032 / T020-A — accept T019-A negative result and test task-aligned gradient transport

## Research review R032 — T019-A acceptance (`b49c432` → `ee98de0` → `d047312` → `58865bc`)

**Assessment: T019-A is protocol-compliant and none of the four predeclared native pseudo-target component subsets is eligible. Close the fixed native-component branch exactly as R031 required. Do not rescue it with continuous weights, threshold/K/LR sweeps, post-outcome metric selection, or another component subset.**

Codex followed `coordination/PROTOCOL.md` and R031. The 200-image train2017 cohort and four 50-image blocks were frozen before real-model outcomes, excluding all 636 prior source-development IDs and all 5,000 val2017 IDs. The four candidates were exactly `native_cls`, `native_conf`, `native_roi`, and `native_conf_roi`; `loss_rpn_box_reg` was not reintroduced and full-native was not treated as a fifth tunable candidate. The implementation was limited to backward-compatible component selection, configurable method lists, the fixed selection/reporting path, tests, and research receipts. Detector/CLIP/ISP/current-Ours logic, teacher `score>=0.5/top20`, identity initialization, `K=3`, `LR=0.1`, CLIP norm transfer, and corruption definitions remained protected. Full regression was `162 passed / 10 skipped`; the A6000 K=3 smoke passed 224 exact active-component-sum checks; the formal run completed 7,000 adaptive episodes and all isolation/state/hash checks passed. Ground truth remained evaluation-only during adaptation.

The result is a clean negative. Current Ours corruption macro AP is `46.28582` versus `46.22465` for no-adapt. The four candidates score `46.26549` (`native_cls`), `46.26263` (`native_conf`), `46.23347` (`native_roi`), and `46.26700` (`native_conf_roi`), i.e. deltas versus current of `-0.02033`, `-0.02319`, `-0.05234`, and `-0.01881` AP. Every candidate has only `1/4` positive fixed blocks; positive corruption conditions are `1/6`, `3/6`, `2/6`, and `3/6`. Therefore all fail the frozen `+0.10 AP`, `3/4` block, and `4/6` condition requirements. The isolated AP75 gains for some candidates are diagnostics only and must not be used to reopen this family. Current Ours remains the authoritative deployment baseline.

The combined T018/T019 evidence now says that simply replacing the current fixed-ROI pseudo objective with native Faster R-CNN training-loss terms is not the performance route. The next useful move should preserve the strongest part of the method — per-image label-free TTT of the compact ISP state — but **learn how to transport the existing self-supervised ISP gradient toward the source task gradient**, rather than inventing another hand-composed pseudo loss. This is qualitatively different from the failed image-feature predictor/meta-initialization branch: the input is the actual 8-D test-time pseudo gradient, which already has reproducible task signal, and the learned object is a tiny norm-preserving gradient transport used inside TTT.

---

## T020-A — cross-fitted source-trained orthogonal gradient transport

**Status: TODO. Performance-oriented, source-trained task-alignment study; target roughly one A6000 hour. This task explicitly authorizes the minimal training/analysis/runtime plumbing needed for one fixed 8×8 orthogonal gradient transport. Do not modify the detector, CLIP, ISP operators, or the authoritative current-Ours objective.**

### Scientific / paper question

Can we keep the paper-defining inference mechanism

`test image -> label-free pseudo gradient -> update compact ISP state -> frozen detector`

while learning, on source data only, a very small task-alignment operator that rotates the pseudo gradient into a direction better aligned with detection loss?

For the authoritative current-Ours detector-native pseudo gradient `g_p ∈ R^8`, learn a single source-domain orthogonal matrix `Q ∈ R^{8×8}` from source-labelled gradient pairs. At deployment, no labels are used: compute the same current-Ours pseudo gradient, rotate it by `Q`, keep its norm unchanged, apply the same CLIP magnitude transfer, and update only `phi`.

This tests a stronger and more paper-relevant hypothesis than another loss-component sweep: **source supervision can align the update rule, while the actual adaptation remains sample-specific TTT.**

### Stage A — precommit a fresh cross-fitting cohort

Before any task-gradient outcomes or AP results, create and commit `research_log/T020A_plan.md`, the cohort manifest, config, hashes, fold assignment, and exact formulas.

Use **200 new COCO train2017 images**, disjoint from every prior TAISP source cohort and from all 5,000 val2017 images. Split them in frozen order into **four image-level blocks of 50 images**. A fold holds out one complete 50-image block; the other 150 images are source-training data for `Q`. Clean/corrupted versions of the same image must never be split across train and held-out portions.

Use the same clean + six corruption conditions already used by T018/T019. Do not add conditions, select images by outcomes, or alter corruption strengths.

### Stage B — collect paired gradients without leakage

For each source-training image-condition episode at identity `phi=0`:

1. compute the **exact authoritative current-Ours label-free detector-native pseudo gradient** `g_p`; do not use the failed native pseudo-target component objectives;
2. compute the analysis-only annotated source task gradient `g_t = ∇_phi L_task` with the existing oracle/task-loss path;
3. keep detector/CLIP frozen and record hashes/state/isolation receipts;
4. save both raw 8-D vectors and norms.

Normalize only for fitting:

`p = g_p / (||g_p|| + EPS)` and `t = g_t / (||g_t|| + EPS)`.

Zero-norm episodes, if any, must be retained and explicitly reported; exclude them from the Procrustes fit only if the normalization is undefined, never from AP evaluation.

For each fold fit exactly one **unregularized orthogonal Procrustes** map on the 150-image training portion across all seven conditions:

`Q_f = argmin_{Q^T Q = I} ||P Q - T||_F^2`.

Use the closed-form SVD solution from `P^T T`; no optimizer, ridge, coordinate selection, condition-specific matrices, nonlinear network, bias, or hyperparameter search. Save singular values, `Q_f`, orthogonality error, determinant, and train indices. Do not force determinant `+1`; use the literal minimum-error orthogonal solution predeclared in the plan.

### Stage C — held-out TTT candidate

On the held-out 50-image block of each fold, adaptation must be label-free. For every TTT step compute the same current-Ours pseudo gradient `g_p(phi_k)`, then use

`g_q = g_p Q_f`  (respect the row/column convention consistently and unit-test it),

followed by the **unchanged** current-Ours magnitude rule

`g_update = g_q * ||g_clip|| / (||g_q|| + EPS)`.

Because `Q_f` is orthogonal, `||g_q||` should equal `||g_p||` up to numerical tolerance. Keep `K=3`, `LR=0.1`, identity initialization, teacher `score>=0.5/top20`, CLIP model/prompts/preprocessing, ISP ranges/operators, and all corruption settings unchanged. Empty-support behavior must remain identity/zero update.

Evaluate exactly three methods on every held-out image-condition pair:

- `no_adapt`;
- authoritative `current_ours`;
- `grad_transport_ours` using the fold-specific `Q_f` trained without that block.

Combine the four held-out folds only after all predictions are complete. Ground truth for a held-out fold may be used only for official evaluation and diagnostic oracle-gradient alignment, never for fitting that fold's `Q_f` or making adaptation decisions.

### Stage D — tests and required diagnostics

Before the formal run, add focused tests for:

- exact Procrustes recovery on a known synthetic orthogonal transform;
- image-level fold isolation and absence of clean/corrupt pair leakage;
- `Q^T Q ≈ I` and gradient-norm preservation;
- correct row/column transform convention;
- held-out labels/task gradients never entering runtime adaptation;
- detector/CLIP remain frozen and only episodic ISP state changes;
- empty-support identity behavior;
- backward compatibility of authoritative `current_ours`.

Run focused tests, full regression, then a 2-image A6000 K=3 smoke with no AP selection. Preserve exact commands, hashes, environment, matrices, gradient pairs, and isolation receipts.

Report, cross-fitted and held-out only:

- six-corruption macro `AP/AP50/AP75` for all three methods;
- delta of `grad_transport_ours` versus `current_ours` and no-adapt;
- four held-out block deltas and six corruption-condition deltas;
- clean AP delta;
- clean/corrupted `||phi_3||`, update fraction, support count, and latency;
- raw versus transported cosine to the held-out oracle task gradient at identity, overall/corrupted/per-block/per-condition;
- fraction of episodes whose transported cosine improves over raw;
- Procrustes singular values and fold-to-fold matrix similarity as diagnostics only.

### Stage E — frozen advancement rule

This is primarily a performance test. Call the learned transport a **development candidate** only if all hold:

1. cross-fitted six-corruption macro AP is at least **`+0.15 AP`** above authoritative `current_ours`;
2. at least **3/4** held-out blocks have positive macro delta versus `current_ours`;
3. at least **4/6** corruption conditions have positive AP delta versus `current_ours`;
4. candidate six-corruption macro AP is above no-adapt;
5. clean AP is no worse than `current_ours - 0.10 AP`;
6. no leakage, isolation, norm-preservation, or reproducibility blocker occurs.

Gradient-alignment statistics are mandatory mechanism evidence but are **not** an extra post-hoc performance gate. Do not weaken the AP gate if alignment looks attractive.

If T020-A passes, stop and report. The next review will train one final `Q` on the full source-training cohort and test it on a larger completely new confirmation cohort before any FCOS/SSD promotion.

If T020-A fails, close the **fixed global linear gradient-transport** branch. Do not immediately increase matrix capacity, add nonlinear networks, tune ridge strengths, or cherry-pick ISP coordinates. The next pivot should be spatially structured guidance / spatial ISP or a separately predeclared teacher-consensus/stability objective.

### Explicit prohibitions during T020-A

Do not modify detector weights, CLIP weights/prompts, ISP operators/ranges, teacher threshold/top-k, `K`, LR, corruption strengths, current-Ours pseudo objective, or CLIP magnitude formula. Do not use target/val labels for fitting. Do not run FCOS/SSD/val2017. Do not reopen native pseudo-target component weights, T017 numerical forensics, need-to-adapt gates, predictor/meta-initialization, or spatial ISP in this task.

Append exact outcomes to `coordination/CODEX_TO_CHATGPT.md`, commit/push all plans/code/tests/reports/receipts, and stop at `NEEDS_REVIEW` for research review.
