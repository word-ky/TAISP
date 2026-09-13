# T013-H — NEEDS_REVIEW: structure replicated, covariance utility fails

R022's structural replication rule passes on the 32 new source images. The fixed
cross-fitted covariance-utility rule fails: 39/64 improving first-order predictions
versus the required 40/64. Preserve that negative decision despite passing clean,
corrupted and median-cosine subcriteria. No training or parameterization change follows.

## Provenance and execution

- Authority: R022/4802c88 in CHATGPT_TO_CODEX.md. T013-G is closed.
- Pre-outcome plan and cohort commit: **f67e8d8**, pushed before model calls.
- Collection/analysis code: **553e02c**. Deployment code and model/loss paths unchanged.
- Release: `20260913-081014-taisp-t013h-source-replication`.
- Run: `20260913-081117-taisp-t013h-source-replication`, A6000,
  `/home/liujianhua/wjq/TAISP/runs/20260913-081117-taisp-t013h-source-replication`.
- Run start/end: 2026-09-13 08:11:21–08:12:06 +08:00; exit **0**.
- Collection elapsed: 34.17053716699593 seconds including setup; peak allocated CUDA
  memory 4,024,015,360 bytes. Algebra used saved arrays on CPU in float64.
- Baseline on unchanged T013-G release: **7 passed in 1.47s**. New/affected CPU
  tests: **6 passed in 1.60s**. Full regression: **111 passed, 10 skipped in 6.16s**.
  The 10 pretrained integration tests were explicitly disabled by TAISP_REAL_MODELS=0;
  the subsequent 64-record primary collection used the real frozen models.
  Existing NVML/deprecation warnings did not block CUDA. Local syntax compile passed;
  NumPy+torch tests used the existing remote environment due the previously documented
  Windows OpenMP conflict. No duplicate-runtime workaround.
- Exact commands, source revision, package/model/prompt pins and logs are retained in
  [raw run](remote_runs/20260913-081117-taisp-t013h-source-replication/).

## Cohort and primary-record integrity

`T013H_train_cohort.json` SHA256:
`99f15bc4329b8431a6718bc1c2ae1ef3fa6221db0869a7cc593ef95bdca2d357`.
The manifest and plan predate gradients. The hash rule ranked 13,794 eligible images
from 13,798 readable source images with valid annotations. Exclusions included the
original four source IDs, all 5,000 official val2017 IDs, and the union of 1,400
evaluation IDs from 20 T002–T012 environment receipts. The latter were verified to
be included in the official val set. No outcome-dependent filtering or replacement.

Exactly 32 images / 64 episodes, clean followed by one fixed corruption. The four
corruptions each cover eight images; four consecutive eight-image blocks each have
two images per corruption. Selected JPEG hashes, full source annotations, annotation
hashes, exclusion ledger and block membership are retained. No target outcomes used.

The seed and constructor order reproduced the pinned original T013-C zero-head
checkpoint exactly. One predictor feature forward per primary episode was captured
by a hook in the same K3 source trajectory; all 16-D h, 8-D g, head gradients, loss,
phi0/phi3, supports, saturation and isolation records were saved. Pseudo supports were
selected once per episode (score .5, top20); labels entered only the analysis outer
loss. No optimizer, no predictor update, no counterfactual model re-evaluation.

All 64 isolation checks passed: source/CLIP parameters and buffers unchanged,
frozen/eval with no parameter gradients; original/copied predictors unchanged with
grad=None; ISP state unchanged. All primary phi0 are zero and trunk gradients zero,
as expected at the zero head. Empty supports: **0/64**. Maximum saturation:
clean **0.1168747693**, corrupt **0.2382955104**; all images retained.
Mean phi3 norms: clean **0.03956614327**, corrupt **0.02718205791**.
Mean source outer loss: clean **0.3364305513**, corrupt **0.4000436723**.
These are primary source diagnostics, not before/after performance improvements.

All three inherited fixed-bound gradient reconstructions pass. Max absolute error:
per-episode weight outer product **7.028177151e-9**, bias identity **0**,
mean weight factorization **2.023458363e-10**. Maximum error/bound ratio **0.02075525**.
The bound remains 8*eps_float32*(max_abs_saved+abs_saved_element), unchanged.

## Structural replication

| Scope | Median rho | r_C | r_C_out | Full centered energy (%) | Clean/corrupt mean-gradient cosine |
| --- | --- | --- | --- | --- | --- |
| Overall | 1.068435713 | 0.270777542 | 0.324689633 | 0.0287439153 | 0.701727325 |
| Block 0 | 1.297340993 | 0.181406526 | 0.216339063 | 0.0518793024 | -0.570084017 |
| Block 1 | 0.905160811 | 0.328321834 | 0.402314387 | 0.0381815423 | 0.606389789 |
| Block 2 | 1.258853884 | 0.223284505 | 0.317006392 | 0.0291102942 | -0.018649237 |
| Block 3 | 0.778511693 | 0.250975753 | 0.287578800 | 0.0330229464 | 0.581051289 |

Overall median rho exceeds .10; covariance ratios exceed .10. Full centered energy
is below 5% overall and below 10% in **4/4** blocks (required >=3/4). Thus the fixed
structural rule passes. Overall feature centered fraction is **8.62149813%**, mean
norm **0.3841084472**, participation effective rank **2.005216624**. The expanded
cohort still has low effective rank; these are engineering criteria, not population tests.

The finer cross-term sign pattern differs from T013-G. Overall ||A||=.02921003239,
||C||=.01084637575, cos(A,C)=**-.6138247850**. Raw bias energy is 3.701147002e-7
versus full 4.595833961e-7, a descriptive nonorthogonal ratio. Centered A energy is
2.687489687e-10, centered C energy 6.212649018e-11, and the centered A–C cross term
is **-1.987731969e-10**, yielding full centered energy 1.321022620e-10.
Hence this cohort has partial centered cancellation as well as common-mode domination.
Do not generalize the old microset's positive-cross-term / no-cancellation finding.

## Four-fold held-out first-order utility

Each held-out block contains 16 episodes from eight images; moments use only the
other 48 episodes / 24 images. Both members of each image remain together. All fold
indices, fitted mu/gbar/A/C, deltas, dot products, cosines, norms and separations are
retained. No fitting of a predictor and no detector/ISP/CLIP evaluation of these deltas.

| Direction | Improving all | Clean | Corrupt | Overall median cos(delta,-g) | Mean g dot delta |
| --- | --- | --- | --- | --- | --- |
| Full | 29/64 | 12/32 | 17/32 | -0.0421150610 | -4.456314632e-6 |
| Common | 29/64 | 12/32 | 17/32 | -0.0404019222 | -4.542288987e-6 |
| Covariance residual | **39/64** | 19/32 | 20/32 | 0.1344034354 | -8.722545644e-8 |

Covariance improving counts by fold: **7, 8, 13, 11** out of 16 each. Its mean norm
is 8.547260387e-7; the clean mean predicted change is +1.975883560e-9 and corrupted
mean is -1.764267964e-7. Preserve both magnitude and sign rather than treating a
positive cosine/count as a measured finite-step or AP gain.

The rule requires >=40 overall AND >=18 clean AND >=18 corrupted AND positive
overall median cosine. **Only the overall count fails, and the conjunction fails.**
There is no threshold relaxation, added repeat, new cohort, calibration or selection.

## Delivery and stop

[Complete tables](T013H/tables.md) contain all scope statistics, cross terms,
per-episode first-order predictions and primary diagnostics. Full matrices and vectors
remain in the raw JSON. [Artifact manifest](T013H/artifact_manifest.json) hashes all
**76 files / 2,937,962 bytes** including 64 individual records, aggregate records,
supports, checkpoint, cohort, environment, algebra, completion receipts and launch logs.

Primary records SHA256: `ac3d462521ad84e71af298eb57ae91b732f97e74fcf5336de91ab0967c41c2ab`.
Algebra audit SHA256: `ef31b0394eca3e1d4ee773380b65f2a1a11f37620b67d05c3bb77b2da6ba0a72`.

**Decision: structural replication PASS; cross-fitted covariance utility FAIL;
joint advancement criterion FAIL.** Centered variation is measurable but has not met
the predeclared task-utility requirement. Stop T013-H for research review. Do not
promote centering, bias removal, covariance-only deployment, a new parameterization,
training, regularization, target/AP evaluation or T013-I automatically.
