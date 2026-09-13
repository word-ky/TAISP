# T013-I — NEEDS_REVIEW: fixed linear capacity gate fails

The current frozen 16-D features plus affine linear head did not demonstrate the
held-out sample-specific source meta-gradient predictability required by R023.
Pooled residual R2 is **-0.3806308900**, all four fold R2 values are negative, median
residual cosine is **0.06836475594**, and residual alignment is positive on **33/64**
episodes (15 clean, 18 corrupted). The predeclared conjunction fails without tuning.
Under R023, close this learned-initialization branch for the current representation
and return a research pivot decision to the research lead. No pivot is implemented.

## Provenance, implementation and execution

- Authority: R023/c016e51 in `coordination/CHATGPT_TO_CODEX_R023_T013I.md`;
  T013-H accepted and closed as a negative utility result.
- Outcome-free plan **5723ecd**, then implementation **c6d2c6e**; both pushed before
  the new analysis. New code is confined to `taisp/analysis/linear_capacity.py`,
  its synthetic tests, report rendering and research records. Existing predictor,
  deployment, detector, CLIP, ISP and training implementations are unchanged.
- Release `20260913-091116-taisp-t013i-linear-capacity`; run
  **20260913-091200-taisp-t013i-linear-capacity**, under
  `/home/liujianhua/wjq/TAISP/runs/` on the A6000 server, **CPU only**.
- Run 2026-09-13 **09:12:05–09:12:13 +08:00**, exit **0**. Audit elapsed
  **0.5177526710 seconds** excluding Python startup/tests; 4 observed + 512 null
  SVD fits, exactly 128 null replicates. No real-model execution, optimizer or new data.
- Baseline on unchanged H release: **6 passed in 1.54s**. Focused new/affected tests:
  **9 passed in 1.66s**. Full CPU regression: **116 passed, 11 skipped in 5.91s**
  (ten explicit pretrained integration skips and one CUDA-only skip). Two existing
  protobuf deprecation warnings; no blocker. Local syntax check and stdlib report
  rendering passed. Used the existing remote CPU environment because the local
  NumPy+torch OpenMP conflict was already documented; no runtime workaround.
- [Raw run](remote_runs/20260913-091200-taisp-t013i-linear-capacity/) retains exact
  command in `meta.json`/`run.sh`, test output, provenance and all fit results.

All three source pins matched the accepted H artifact manifest and on-disk files:

| H input | SHA256 |
| --- | --- |
| records.json | ac3d462521ad84e71af298eb57ae91b732f97e74fcf5336de91ab0967c41c2ab |
| cohort.json | 99f15bc4329b8431a6718bc1c2ae1ef3fa6221db0869a7cc593ef95bdca2d357 |
| algebra/audit.json | ef31b0394eca3e1d4ee773380b65f2a1a11f37620b67d05c3bb77b2da6ba0a72 |

All 64 records, their clean/corrupted order and four precommitted blocks were kept.
Each fold trains on 48 episodes / 24 images and holds out 16 episodes / eight images;
both members of an image stay together. Held-out gradients never enter training means
or coefficients. Full fold identities, means, B matrices and predictions are retained.

## Fixed fit and numerical condition

The fit uses train-only X=H-mu and Y=G-gbar, SVD minimum-norm least squares with
`tol=eps_float64*max(X.shape)*s_max`, retaining singular values strictly above tol.
The held-out residual estimate is B(h-mu), and full gradient estimate is gbar+B(h-mu).
No normalization, feature selection, ridge, rank sweep or optimizer was introduced.

| Fold | Numerical rank | Rank tolerance | B Frobenius norm | Condition number | Training residual SSE |
| --- | --- | --- | --- | --- | --- |
| 0 | 16 | 7.223369785e-15 | 2679.072202 | 4128.546267 | 7.813965224 |
| 1 | 16 | 6.675106769e-15 | 1722.680119 | 3548.243181 | 5.774596548 |
| 2 | 16 | 6.722710389e-15 | 3140.094321 | 2907.713752 | 6.018064387 |
| 3 | 16 | 6.789530257e-15 | 3327.319758 | 2936.073422 | 5.731092725 |

All fits and predictions are finite; no singular directions were removed by the fixed
numerical cutoff. The feature matrices are nevertheless anisotropic: condition numbers
are about 2.9–4.1 thousand and coefficient norms about 1.7–3.3 thousand. These large
coefficients warrant caution about small feature perturbations; they are not a numerical
failure or authorization to add regularization. Full singular spectra are in the tables.
Numerical rank16 does not contradict H's low participation effective rank near2.

## Held-out residual capacity

Pooled SSE is **19.35733960088**, compared with the common-only residual energy/SSE
**14.02064790876**. Thus the fitted head is worse than predicting zero residual on
this held-out cohort, despite fitting the training residuals. R2 uses pooled sums,
not an average of per-fold R2 values.

| Scope | Residual R2 | Median residual cosine | Positive residual dot |
| --- | --- | --- | --- |
| Overall | -0.3806308900 | 0.06836475594 | 33/64 |
| Clean | -2.597423311 | -0.02254290790 | 15/32 |
| Corrupted | 0.009309191867 | 0.1843268178 | 18/32 |
| Fold 0 | -6.226104798 | 0.2574256589 | 9/16 |
| Fold 1 | -0.1384326675 | -0.3866351801 | 6/16 |
| Fold 2 | -0.1545159727 | 0.2001646460 | 10/16 |
| Fold 3 | -0.1046360453 | 0.007714449917 | 8/16 |

Full corruption-case/family and fold-by-clean/corrupted statistics are retained in
[complete tables](T013I/tables.md). Cases are heterogeneous; no case, episode or fold
was selected to rescue the pooled decision. For example, gamma_s1 has R2=.4205351
but median residual cosine=-.3923475 and only3/8 positive residual dots.

The pooled mean norms of true residual, predicted residual and full predicted gradient
are **.3009862663**, **.3473317678**, and **.3533690222**, respectively. Their maxima
are **1.680661717**, **1.296872076**, and **1.205488184**. All per-episode values are saved.

## Frozen references and first-order context

| Method | Residual R2 | Residual median cosine | Residual dot >0 | Full-direction g dot delta <0 | Mean g dot delta |
| --- | --- | --- | --- | --- | --- |
| Affine linear | -0.3806308900 | 0.06836475594 | 33/64 | 37/64 | -4.647974066e-5 |
| Common-only gbar | 0 | 0 | 0/64 | 29/64 | -4.116131275e-6 |
| H covariance residual | 0.0007984359813 | 0.2243513870 | 40/64 | **39/64** | -8.722545644e-8 |

Common-only predicts zero residual and gbar as the full gradient. The covariance
reference predicts C(h-mu) as the residual and uses that residual alone for its
inherited descent delta=-.001*C(h-mu), without adding gbar. Its deltas reconstruct H's
saved deltas within the unchanged8eps32 bound: max error **3.176373552e-22**.

Residual alignment and full-gradient descent are different quantities. The covariance
reference's40/64 residual-dot count does **not** change H's failed39/64 full-gradient
descent gate. Likewise, the linear model's37/64 negative predicted changes do not
rescue its failed residual-capacity criteria. No delta was applied to a predictor or
evaluated through an ISP/detector, and no finite-step or AP improvement was measured.

## Matched pair-permutation null

Exactly128 replicates from NumPy default_rng(20260913), ordered replicate then fold.
Within each training fold, all24 gradient image pairs are permuted against the fixed
feature pairs, preserving clean/corrupt positions and both marginals. True held-out
folds and residuals stay unchanged. The identical SVD fit is used for all512 null fits.
Every mapping, coefficient and held-out prediction is saved; no reseeding or rerun.

| Metric | Observed | Null95 (linear quantile) | Strict empirical percentile | #null >= observed | Corrected upper tail |
| --- | --- | --- | --- | --- | --- |
| Residual R2 | -0.3806308900 | -0.5450267464 | 100% | 0 | 1/129 = 0.007751938 |
| Median residual cosine | 0.06836475594 | 0.2305152311 | 72.65625% | 35 | 36/129 = 0.279069767 |
| Positive residual dots | 33 | 40 | 54.6875% | 58 | 59/129 = 0.457364341 |

The dot-count null has14 ties at33; the other comparisons have zero ties. Percentiles
use count(null<observed)/128; tails use (1+count(null>=observed))/129. Quantile method
and strict exceedance were fixed in the plan.

Preserve the positive nuance: observed R2 beats every shuffled-pair fit. That shows
an advantage over this matched null, but the absolute R2 is still negative versus
the common baseline, and the cosine fails its null comparison. This supports neither
the conjunction nor a claim that the representation contains exactly zero information.

## Gate and research handoff

| Predeclared criterion | Observed | Outcome |
| --- | --- | --- |
| Pooled R2 >= .10 | -.3806308900 | FAIL |
| Positive R2 in >=3/4 folds | 0/4 | FAIL |
| Median residual cosine >= .20 | .06836475594 | FAIL |
| Positive residual dots >=44/64 | 33/64 | FAIL |
| Clean >=20/32 | 15/32 | FAIL |
| Corrupted >=20/32 | 18/32 | FAIL |
| R2 strictly above matched null95 | -.38063 > -.54503 | PASS |
| Cosine strictly above matched null95 | .06836 < .23052 | FAIL |

**Conjunction FAIL.** Per R023, the current frozen16D feature + linear-head family
has not demonstrated sufficient held-out sample-specific predictability to justify
longer training or common-mode suppression. Close this current-representation
learned-initialization branch and request a research pivot in the coordination review.
This is a bounded source-array result, not an impossibility theorem for other learned
representations. T013-H remains failed and is not retroactively rescued.

[Artifact manifest](T013I/artifact_manifest.json) hashes **137 files / 8,432,184 bytes**:
all observed folds/64 predictions/three references,128 complete null fits, schedule,
summary, provenance, reconstruction check, completion, launch command and logs.
No analysis remains running. Stop **NEEDS_REVIEW**; no new model run, meta-training,
centering/bias removal, covariance-only deployment, predictor redesign, regularizer,
targetAP, spatialISP or gating/dose experiment started.
