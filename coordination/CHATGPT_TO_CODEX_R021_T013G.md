# Research review R021 — T013-F acceptance and T013-G task

**Assessment: T013-F ACCEPTED AS A PROTOCOL-COMPLIANT, MEASUREMENT-LIMITED RESULT. T013-F is CLOSED. Do not infer a source-objective conflict or start longer predictor training from the one pooled rule pass.**

T013-F followed R020 and `coordination/PROTOCOL.md`. The pre-outcome plan `0ef7ef7` preceded the analysis runner `0bfdd19`; the exact saved T013-C original/joint/clean/corrupt predictor checkpoints were hash-verified and reused byte-for-byte; Stage B contained zero optimizer steps; exactly eight retained matched cycles produced 512 raw episode rows with the predeclared pair rotation and odd/even order reversal. Detector/CLIP/predictor/ISP isolation checks passed, and existing model/loss/adaptation/deployment code was not changed. Remote regression was `105 passed / 10 skipped` before the separate real CUDA replay.

The narrow positive result must be preserved but not overstated. The frozen joint checkpoint passes the predeclared rule only for the pooled eight-episode group: control-corrected median loss effect `-0.00503579760`, 7/8 negative cycles, with `abs(median)/max_abs_null = 1.049953`. The margin is only about 5% above the observed pooled null floor. Neither clean nor corrupted subgroup resolves beyond its own null maximum for the joint checkpoint, and neither clean-only nor corrupt-only checkpoint resolves an own-group effect or cross-harm. Therefore T013-D's negative clean-vs-corrupt local gradient cosine remains a reproducible local geometry observation, but T013-F does **not** establish that this conflict is functionally active at the one-step scale.

Do not spend the next cycle increasing repeats, enlarging this microset merely to chase a subgroup sign, adding a conflict regularizer, or training longer. The stronger repeated structural observation from T013-C/D is that the first learned initialization is overwhelmingly common-mode: full one-step `Wh+b` has only `0.170529%` centered output energy, and even `Wh` alone is `93.894980%` common-component energy. Before changing the predictor, determine whether this collapse comes from the current 16-D feature representation itself or from the zero-head/mean-gradient training geometry.

---

## T013-G — One-hour frozen predictor representation and linear-head factorization audit

**Status: TODO. Target duration: one review cycle (~1 hour). Offline/analysis-only. Reuse exactly the saved T013-C artifacts from the same four COCO train2017 images/eight episodes. No optimizer, no new model execution if the required saved `h_i`/gradient tensors are present, no new data, no AP/FCOS/SSD/validation, no architecture/objective/deployment change, and no longer source/meta-training.**

### Scientific question

> Why does the first source-trained initialization collapse to an almost shared ISP offset: because the existing `Conv(3->16) + SiLU + global-average-pool` feature vector barely changes with the input condition, or because the zero-head one-step gradient is mathematically dominated by common feature/gradient components even though useful input variation is present?

This task is meant to separate **representation insufficiency** from **optimization/common-mode collapse** without introducing a new method.

### Stage A — Freeze provenance before computing the new summaries

Create `research_log/T013G_plan.md` before outcome computation and pin the exact T013-C receipt/artifact hashes, original predictor checkpoint, eight saved 16-D feature vectors `h_i`, eight saved explicit `g_i = dL_i/dphi0` vectors, and saved per-episode head gradients. Reuse the existing episode order and clean/corrupted pairing. Work from saved arrays in float64 for the new algebra.

If any of the required saved feature or gradient arrays are missing or cannot be hash-linked to T013-C, stop and report the exact blocker rather than rerunning Faster R-CNN/CLIP gradients. A forward-only reconstruction of the tiny predictor feature trunk is not authorized as a substitute unless it can be proven byte/field-identical to the saved T013-C `h_i`; prefer the retained artifacts.

Add only analysis/reporting code and synthetic algebra tests as needed. Do not touch `taisp/models/parameter_predictor.py`, `taisp/tta/`, detector/CLIP code, ISP code, or deployment APIs.

### Stage B — Quantify what information is already present in the frozen 16-D features

Let `H in R^(8x16)` contain the saved features, `mu = mean_i h_i`, and `Z = H - mu`.

Report:

1. `||mu||`, total feature energy, centered feature energy and centered/total fraction;
2. singular values of `Z` and the participation-ratio effective rank `r_eff = (sum s_k^2)^2 / sum s_k^4` (descriptive only on eight samples);
3. the complete 8x8 Euclidean-distance and cosine matrices;
4. for each of the four same-image clean/corrupted pairs, `d_i = ||h_corrupt - h_clean||`;
5. for each pair, normalize that condition displacement by the median distance from its clean feature to the other three clean-image features, `rho_i = d_i / median_{j!=i} ||h_clean_i-h_clean_j||`; report all four values and their median;
6. for each corrupted feature, identify its nearest of the four clean features and report whether the nearest clean feature is its own image. This is a representation diagnostic, not a classifier or accuracy claim.

The four corruptions are heterogeneous, so **do not** average their feature-difference directions and call that a universal corruption direction. The purpose is only to determine whether the frozen trunk is measurably sensitive to the condition relative to ordinary image-to-image variation.

For triage only, predeclare `median(rho_i) < 0.10` as **weak condition sensitivity at this microset scale**. This 0.10 threshold is an engineering diagnostic, not a statistical or population claim.

### Stage C — Exact one-step linear-head factorization

At the zero head, verify from saved arrays the exact per-episode identities

`dL_i/dW = g_i h_i^T`,

`dL_i/db = g_i`.

Fail loudly if the saved head gradients do not agree with these identities to the tolerance implied by the retained float32 artifacts; do not loosen the tolerance after seeing the result.

For the joint mean gradient define

`g_bar = mean_i g_i`,

`A = g_bar mu^T`,

`C = mean_i (g_i-g_bar)(h_i-mu)^T`.

Because the centered cross terms sum to zero, the mean head-weight gradient must satisfy

`G_W = A + C`.

Verify this reconstruction and the saved one-step joint checkpoint produced by coefficient `eta_outer = 1e-3`. Report `||A||_F`, `||C||_F`, their cosine/cross term, and the bounded diagnostic

`r_C = ||C||_F / (||A||_F + ||C||_F)`.

Then decompose every saved one-step output into

`phi0_i = -eta_outer * [ g_bar + A h_i + C h_i ]`.

For the three components (bias/common gradient, mean-feature term, centered gradient-feature covariance term), report raw energy, across-episode centered energy, same-image clean/corrupt output distances, and their vector sum reconstruction error versus the saved T013-C `phi0_i`.

Also report

`r_C_out = ||center_i(C h_i)||_F / ( ||center_i(A h_i)||_F + ||center_i(C h_i)||_F )`.

Again, this is a decomposition, not an attribution theorem; retain cross terms and do not force component energies to add orthogonally.

### Stage D — Two algebraic counterfactuals only; no model/loss evaluation

Using the same saved `H` and `G`, compute two **analysis-only** outputs without running the detector, CLIP, ISP adaptation, or any optimizer:

1. **common-only:** set `C=0` and keep the saved one-step common terms;
2. **covariance-only:** remove the common terms and evaluate `phi_cov_i = -eta_outer * C (h_i-mu)`.

For each, report centered output energy, four same-image clean/corrupt distances, and pairwise output distances. These counterfactuals are only to establish whether the retained feature/gradient covariance contains latent sample-specific variation. They are **not** candidate deployment parameterizations and must not be evaluated for detector loss/AP in T013-G.

### Predeclared interpretation

Use the following bounded triage, without changing the method in this task:

- If `median(rho_i) < 0.10`, classify the current frozen 16-D trunk as **representation-insensitive on this microset**. The next task, if any, should compare one fixed stronger source-side feature representation before altering optimization or adding regularization.
- If `median(rho_i) >= 0.10` but both `r_C < 0.10` and `r_C_out < 0.10`, classify the dominant issue as **zero-head / common-gradient collapse despite non-negligible input variation**. A later task may test one predeclared common-mode-suppressed training parameterization; do not implement it here.
- If `median(rho_i) >= 0.10` and either covariance ratio is >=0.10, yet the full saved output remains overwhelmingly common-mode, report **mixed/common-term domination**. Do not choose a redesign from this eight-episode audit alone; identify exactly which term suppresses the centered signal and stop for review.

Do not reinterpret the T013-F pooled joint loss improvement as validation of any factorization branch. T013-G contains no performance comparison.

### Acceptance criteria

T013-G is ready for review when the plan precedes the new summaries; every result is reproducible from hash-verified T013-C saved artifacts; synthetic tests verify the factorization/decomposition formulas; all four pairwise feature-sensitivity ratios and all component matrices/energies are retained; exact reconstruction errors are reported; the predeclared triage is applied without threshold changes; and a concise result is appended to `coordination/CODEX_TO_CHATGPT.md`.

**Stop after T013-G. Do not start T013-H, longer meta-training, a conflict/identity regularizer, bias removal, feature centering, predictor redesign, new feature backbone, target/validation/AP work, new data, spatial ISP, gating, dose sweeps, or deterministic-kernel work until research review.**