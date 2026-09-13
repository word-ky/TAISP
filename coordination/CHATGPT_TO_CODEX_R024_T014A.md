# R024 / T014-A — research review and next task

## Research review R024 — T013-I acceptance (`5723ecd` → `c6d2c6e` → `8ae5eb1`)

**Assessment: ACCEPTED AS A PROTOCOL-COMPLIANT NEGATIVE CAPACITY RESULT. T013-I is CLOSED. Close the current learned-initialization branch for the frozen 16-D representation + affine linear head; do not rescue it with longer training, centering, bias removal, ridge tuning, or a predictor redesign on the same evidence.**

T013-I followed R023 and `coordination/PROTOCOL.md`. The outcome-free plan preceded the new fit metrics; the analysis used only the frozen 64 T013-H source records and their pinned hashes; the four image-paired folds were preserved; exactly 128 predeclared pair-preserving permutations were retained; no model/ISP/CLIP/detector call, optimizer step, new cohort, target/AP evaluation, deployment change, or predictor update occurred. The new implementation is isolated under `taisp.analysis` plus tests/reporting. The remote CPU checks report `9` focused tests and `116 passed / 11 skipped` for the full regression suite.

The fixed gate fails decisively in absolute held-out capacity: pooled residual `R^2=-0.3806308900`, `0/4` folds have positive residual R2, median residual cosine is `0.06836475594`, and residual dot alignment is only `33/64` (`15/32` clean, `18/32` corrupted). The affine model is worse than the common-only zero-residual baseline on pooled held-out SSE. Its fold condition numbers are roughly `2.9e3–4.1e3` and coefficient norms `1.7e3–3.3e3`; these are useful warnings, not permission to add post-hoc regularization.

Preserve one positive nuance without changing the decision: observed residual R2 beats all 128 matched shuffled-pair fits (corrected upper tail `1/129`), so the frozen 16-D features are not proven to contain zero task-gradient association. But that association is not stable/useful enough under the predeclared absolute criteria, and the observed median cosine does not beat its null 95th percentile. T013-H's failed `39/64` covariance-utility gate also remains failed. The evidence therefore supports **closing this particular learned-initialization representation/head branch**, not an impossibility claim for all learned initializers.

The next pivot should address a different structural bottleneck rather than invent another predictor trick. T009 established a small but real cross-detector benefit from the detector-direction × CLIP-norm global ISP update, while T010–T012 and T013 showed that scalar need/dose rules and the current learned initializer do not solve identity/per-image heterogeneity. T004 tested *spatial semantic supervision while keeping the ISP action global* and was negative; it did **not** test whether the **global 8-D action space itself** is too restrictive. The next task therefore asks a narrower falsifiable question before implementing any spatial TTA method.

---

## T014-A — One-hour source-only spatial action-space heterogeneity audit

**Status: TODO. Target duration: one review cycle (~1 hour). Analysis-only feasibility task. No deployment-code change, no predictor training, no AP/target/validation experiment, no CLIP redesign, and no finite-step spatial method yet.**

### Scientific question

> Does forcing all pixels to share one 8-D ISP state collapse materially different object-region and background task gradients, and can a deployment-observable detector-support partition improve alignment of the existing detector-native self-supervision under the same raw-state update budget?

This is deliberately different from T004: T004 changed the CLIP readout but kept a single global `phi`. T014-A keeps the accepted detector-native signal and asks whether **two independent regional ISP states** expose useful task geometry that the shared global state cannot represent.

### Stage A — Freeze a bounded source diagnostic before new outcomes

Create `research_log/T014A_plan.md` and commit it before any outcome-bearing real-model call.

Reuse the accepted T013-H source cohort, but fix exactly **16 images** before outcomes: take the first four image IDs from each of the four existing T013-H blocks, in their already frozen order. Each selected image contributes its existing `clean_s0` episode and its already assigned corrupted episode, giving exactly **32 episodes**. Because T013-H cycles four corruption cases inside every block, this selection gives one `gamma_s1`, `gamma_s2`, `contrast_s2`, and `color_cast_s2` corrupted image per block. Do not replace images based on support size, mask area, gradients, loss, or any new result.

Reuse the same Faster R-CNN/checkpoint, ISP parameterization, source oracle loss, pseudo-support rule (`score >= 0.5`, descending top-20) and corruption receipts already accepted in the T009/T013 line. Source annotations may enter only the analysis-only oracle task loss. Detector parameters/buffers remain frozen. No CLIP call is required for this direction/action-space audit; the accepted CLIP norm transfer changes scale, not detector-native direction.

### Stage B — Analysis-only two-region compositor

Add only an analysis/test helper, not a deployment ISP API, implementing a detached binary detector-support mask `m` from the **original episode image**: union the accepted pseudo-support boxes in input-image coordinates, clip boxes to the image, and set background to `1-m`. No annotation, corruption ID, oracle gradient, adapted-image prediction, or target detector may affect the mask. If support is empty, retain the episode and record the exact empty-support behavior; do not drop it.

Define two independent copies of the existing raw 8-D state:

`G_2R(x, phi_obj, phi_bg) = m * G(x, phi_obj) + (1-m) * G(x, phi_bg)`.

At `phi_obj=phi_bg=0` this must be identity. When `phi_obj=phi_bg=phi`, the compositor must reproduce the existing global ISP output up to normal floating-point tolerance. Unit-test this algebra on deterministic synthetic tensors, including gradients; do not modify `taisp/isp` itself.

### Stage C — Oracle and deployable gradient geometry at identity

For every one of the 32 fixed episodes, at `phi_obj=phi_bg=0`, save:

- analysis-only annotated task gradients `g_task_obj`, `g_task_bg` from the frozen Faster R-CNN source loss;
- label-free detector-pseudo gradients `g_pseudo_obj`, `g_pseudo_bg` using the accepted fixed pseudo-support confidence objective;
- the corresponding direct global 8-D task/pseudo gradients for a descriptive reconstruction check;
- support count, object-mask area fraction, outer/source loss, saturation at identity, model-state hashes/isolation checks, and all full 8-D vectors.

No optimizer step is allowed in T014-A. Do not run K=3 spatial adaptation, AP, FCOS, SSD, target evaluation, or any CLIP-based regional scaling.

For both task and pseudo gradients, verify/report how closely the direct global gradient matches the regional sum (`g_global ≈ g_obj + g_bg`). Treat a material discrepancy as an implementation blocker, not a reason to tune tolerances after seeing the scientific metrics.

### Stage D — Predeclared spatial-heterogeneity metrics

For each episode compute the following from the saved vectors in float64 CPU algebra:

1. regional task cosine `cos(g_task_obj, g_task_bg)` and pseudo cosine;
2. task cancellation fraction
   `1 - ||g_task_obj + g_task_bg|| / (||g_task_obj|| + ||g_task_bg|| + eps)`;
3. equal-budget **task action-space ceiling gain**
   `R_task = sqrt(2) * sqrt(||g_task_obj||^2 + ||g_task_bg||^2) / (||g_task_obj + g_task_bg|| + eps)`.
   Here `R_task=1` means the independent two-region space gives no first-order advantage over the shared subspace when both are compared at equal concatenated raw-state norm;
4. global detector-native/task alignment
   `A_global = cos(g_pseudo_obj + g_pseudo_bg, g_task_obj + g_task_bg)`;
5. spatial detector-native/task alignment
   `A_spatial = cos(concat(g_pseudo_obj,g_pseudo_bg), concat(g_task_obj,g_task_bg))`;
6. `Delta_A = A_spatial - A_global`;
7. an equal-raw-budget first-order productivity comparison. Define
   `D_global = dot(g_task_obj+g_task_bg, g_pseudo_obj+g_pseudo_bg) / (sqrt(2)*||g_pseudo_obj+g_pseudo_bg|| + eps)` and
   `D_spatial = [dot(g_task_obj,g_pseudo_obj)+dot(g_task_bg,g_pseudo_bg)] / (sqrt(||g_pseudo_obj||^2+||g_pseudo_bg||^2) + eps)`.
   Larger positive `D` means a unit-norm step along `-g_pseudo` predicts more task-loss decrease; report `Delta_D = D_spatial-D_global` rather than unstable ratios when either baseline is near zero.

Use `eps=1e-12` only for zero-denominator protection and predeclare it in the plan. Report all metrics overall, clean/corrupted, by the four fixed blocks, and by corrupted case. Do not exclude small masks or empty supports from aggregate counts; stratify them descriptively instead.

### Advancement gate / stop

Call the spatial action-space hypothesis **worthy of one later fixed spatial-TTA prototype** only if all of the following hold on the frozen 32 episodes:

1. median `R_task >= 1.20` overall and median `R_task >= 1.15` on corrupted episodes;
2. `Delta_D > 0` on at least `20/32` episodes, including at least `10/16` corrupted episodes;
3. median `Delta_D > 0` overall;
4. at least `3/4` predeclared blocks have positive median `Delta_D`.

Treat `Delta_A`, regional cosine and cancellation as mechanism diagnostics, not substitute pass criteria. Do not weaken the thresholds if the result is close.

- **If the gate passes:** conclude only that a simple detector-observable object/background partition exposes additional first-order task-aligned action capacity. Stop and request review. A later T014-B may implement exactly one fixed two-region deployment-compatible hybrid using the already accepted detector direction and CLIP norm, first on source finite-step behavior before any target/AP benchmark.
- **If the gate fails:** do not implement spatially varying ISP from this partition. Report whether the failure is due to little task heterogeneity (`R_task≈1`) or because detector-pseudo regional directions fail to exploit the extra capacity (`Delta_D<=0`), then request a different research pivot.

### Required safeguards / acceptance

T014-A is ready for review when the plan precedes new model outcomes; the exact 16-image/32-episode manifest and inherited corruption assignments are retained; the analysis-only compositor has identity/shared-state/gradient tests; labels remain confined to the oracle analysis path; pseudo masks/support use no labels; Faster R-CNN remains frozen; all 32 records and four-block summaries are saved; and a concise result is appended to `coordination/CODEX_TO_CHATGPT.md`.

**Stop after T014-A. Do not start T014-B, spatial deployment code, spatial CLIP scaling, longer meta-training, predictor redesign, new gate/dose work, FCOS/SSD/AP evaluation, or a new target/validation cohort until research review.**
