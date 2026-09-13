# R023 / T013-I — research review and next task

## Research review R023 — T013-H acceptance (`f67e8d8` → `553e02c` → `65707ec`)

**Assessment: ACCEPTED AS A PROTOCOL-COMPLIANT NEGATIVE UTILITY RESULT. T013-H is CLOSED. Do not relax the 40/64 gate and do not promote covariance-only/centering/bias removal.**

T013-H followed R022 and `coordination/PROTOCOL.md`. The 32-image COCO train2017 cohort, exclusions, four fixed blocks, corruption assignment, model/checkpoint pins and analysis thresholds were committed before outcome-bearing model calls. The run retained exactly 64 primary records, used one fresh zero-head predictor copy per episode, performed no optimizer step, kept source annotations inside the analysis-only outer loss, and verified frozen Faster R-CNN/CLIP plus unchanged predictor/ISP state on all episodes. The baseline/focused/full gates passed (`7`, `6`, and `111` tests respectively, with `10` expected pretrained skips); no deployment implementation was changed.

The structural diagnosis from T013-G **does replicate** on the larger precommitted source cohort. Overall `median rho=1.068435713`, `r_C=0.2707775423`, and `r_C_out=0.3246896334`; full one-step output centered energy is only `0.0287439153%`, and all `4/4` fixed blocks are below the predeclared `10%` common-mode threshold. Thus the existing 16-D representation changes with condition and the gradient/feature covariance is non-negligible, yet the first zero-head update remains overwhelmingly shared/common-mode.

The predeclared advancement condition does **not** pass. Four-fold held-out covariance residual gives `39/64` improving first-order predictions versus the required `40/64`, with `19/32` clean, `20/32` corrupted and positive median cosine `0.1344034354`. Preserve the conjunction exactly: passing the subgroup and cosine subcriteria does not rescue the failed total-count gate. Fold counts `7/16, 8/16, 13/16, 11/16` also show substantial heterogeneity. No threshold relaxation, extra cohort, repeat selection or post-hoc promotion is authorized.

Two earlier microset interpretations must also be narrowed. First, the strong clean-vs-corrupted aggregate gradient opposition from T013-C/D does **not** replicate: the 32-image cohort has overall clean/corrupt mean-gradient cosine `+0.701727325`, with blocks spanning `-0.5701` to `+0.6064`. Do not build an identity/conflict regularizer from the four-image negative cosine. Second, the T013-G positive centered A–C cross term does not replicate either; T013-H has `cos(A,C)=-0.613824785` and a negative centered A–C cross term, so partial cancellation coexists with the much larger common offset. The robust result is common-mode domination, not a universal cross-term sign.

The remaining question is now sharper: T013-H tested the *first zero-head covariance update*, not the best linear head that the fixed 16-D features could support. Before declaring the representation inadequate or changing the architecture, measure a cross-validated linear capacity ceiling using the already frozen arrays. This is analysis only; it does not override the failed T013-H advancement rule.

---

## T013-I — One-hour frozen-array linear-head capacity ceiling

**Status: TODO. Target duration: one review cycle (~1 hour). Offline analysis only. No model/ISP/CLIP/detector calls, no optimizer, no new cohort, no target/validation/AP evaluation, and no deployment or predictor implementation change.**

### Scientific question

> Does the existing 16-D `ParameterPredictor` representation contain enough cross-image information to linearly predict the **sample-specific source meta-gradient residual** beyond the common mean, even though the literal first zero-head update is common-mode dominated?

A positive result would isolate the bottleneck to optimization/initialization dynamics of the head. A negative result would say that longer training of this same 16-D feature + linear-head family has little evidence-based justification.

### Stage A — Freeze provenance before the new analysis

Create `research_log/T013I_plan.md` and commit it before computing the new fit metrics. Use only the exact T013-H 64 saved records and their artifact hashes; verify `records.json`, cohort manifest and algebra-audit hashes against the T013-H receipts. Reuse the same four predeclared eight-image blocks as folds so clean/corrupted members of every image remain together. Do not select a subset of episodes or folds after seeing fit quality.

### Stage B — Four-fold affine linear capacity fit

For each held-out block, use the other 48 episodes / 24 images only. In float64 CPU algebra compute training means `mu = mean(h)` and `g_bar = mean(g)`, then define

`X = h - mu`, `Y = g - g_bar`.

Fit exactly one minimum-norm affine linear predictor of the residual gradient,

`g_hat(h) = g_bar + B (h - mu)`,

where `B` is obtained by SVD least squares / pseudoinverse with the fixed numerical rank tolerance

`tol = eps_float64 * max(X.shape) * s_max`.

No ridge coefficient, feature selection, normalization choice, rank threshold, optimizer or hyperparameter may be searched. Save the training singular values, numerical rank, coefficient norm and condition information for every fold. If the fit is numerically non-finite, report the blocker rather than adding regularization post hoc.

Use two frozen references on the same held-out episodes:

1. **common-only:** `g_hat_common = g_bar`;
2. **T013-H covariance residual:** recompute the inherited `C(h-mu)` direction exactly as in R022, for context only.

Do not evaluate any of these predictions through the detector/ISP; this is a representation-capacity audit only.

### Stage C — Held-out residual and descent metrics

For every held-out episode retain the true residual `r = g - g_bar`, predicted residual `r_hat = B(h-mu)`, full `g_hat`, and report overall, by fold, clean/corrupted group, and corruption family:

- residual SSE and `R2_residual = 1 - sum||r-r_hat||^2 / sum||r||^2`;
- cosine between `r_hat` and `r` (zero if either norm is zero), median cosine and positive-dot count;
- full-gradient cosine `cos(g_hat,g)`;
- analysis-only first-order sign of `g^T(-1e-3*g_hat)` and its magnitude;
- corresponding common-only and inherited covariance-reference metrics;
- norms of `r`, `r_hat`, `g_hat`, plus any extreme coefficient/condition-number warning.

The purpose is to measure **incremental sample-specific information beyond `g_bar`**. Do not claim detector improvement from these algebraic first-order quantities.

### Stage D — Matched permutation falsification

Add a cheap offline null using exactly **128 deterministic pair-level permutations** with fixed seed `20260913`. Within each training fold, permute whole image pairs when associating feature pairs with gradient pairs, preserving clean/corrupted position inside each pair and preserving all feature/gradient marginals. Fit the identical SVD linear model for each permutation and evaluate on the unchanged true held-out folds.

Retain the null distributions for pooled `R2_residual`, median residual cosine, and positive residual-dot count. Report the observed empirical percentile and corrected one-sided tail `(1 + # null >= observed) / 129`. Do not rerun with another seed or choose a different null after seeing the result.

### Predeclared interpretation gate

Call the fixed 16-D representation **linearly sufficient enough to justify a later training-dynamics experiment** only if all of the following hold on the held-out records:

1. pooled `R2_residual >= 0.10`;
2. residual `R2` is positive in at least `3/4` folds;
3. median residual cosine is at least `0.20`;
4. residual dot alignment is positive on at least `44/64` episodes, including at least `20/32` clean and `20/32` corrupted;
5. both pooled residual `R2` and median residual cosine exceed the `95th` percentile of their matched permutation null distributions.

These are intentionally stricter than the failed T013-H covariance gate because T013-I is an optimistic capacity ceiling. Do not weaken them if the result is close.

### Decision rule / stop

- **If the gate passes:** conclude only that the current frozen representation contains cross-validated linear task-gradient information and that zero-head/common-mode optimization dynamics, rather than representation capacity alone, are the next bottleneck. Stop and request review; a later task may test one fixed source-training stabilization, but T013-I must not implement it.
- **If the gate fails:** conclude that the current 16-D feature + linear-head family has not demonstrated enough held-out sample-specific meta-gradient predictability to justify longer training or common-mode suppression. Close this learned-initialization branch for the current representation and request a research pivot rather than adding optimization tricks.
- Regardless of outcome, the failed T013-H `39/64` rule remains failed. T013-I is a separate capacity diagnostic, not a retroactive rescue.

T013-I is ready for review when the plan precedes the new offline calculations, all folds/predictions/permutation controls are retained, the fixed gate is applied without tuning, tests cover pair-preserving folds/SVD reconstruction/permutation determinism, and a concise result is appended to `coordination/CODEX_TO_CHATGPT.md`.

**Stop after T013-I. Do not start longer meta-training, bias removal, feature/output centering, covariance-only deployment, predictor redesign, identity/conflict regularization, target/AP evaluation, spatial ISP, gating/dose work, or any new model run until research review.**