# T016-A — diagonal calibration rescue fails the frozen conjunction

Status: NEEDS_REVIEW. R027 completed on 2026-09-13. The cross-fitted map improves
some aggregate signals, including median Delta_D beyond its permutation null95,
but fails the full predeclared gate. Close the diagonal/source-calibration rescue
of the current detector-native differential gradient, as R027 specifies.

## Provenance and execution

- R027 pointer1107946; pre-outcome plan/input/schedule commit **f97a9dc**;
  implementation **1b19e8c**.
- Parent T015-A records SHA256
  `ae01cea586d0585b510293460ce174e2cf65350cc3d2ba3bd364c24b7de55b12`.
  Four pinned inputs (T015 records/environment/summary and A1 records) are recorded
  in `T016A_input_manifest.json` and the raw provenance receipt.
- All256 pair-permutation schedules committed before fits, NumPy default_rng seed20260913;
  schedule SHA256 `cc8deaad302ab0880350f95ff14af08b2871182d9c05b0a95d611a0f48597e15`.
- Release `20260913-135925-taisp-t016a-differential-calibration`;
  run `20260913-140107-taisp-t016a-differential-calibration`.
  Started14:01:12, finished14:01:24 +08:00; exit0.
- Raw receipts: `research_log/remote_runs/20260913-140107-taisp-t016a-differential-calibration/`.
  All264 files,26,055,799 bytes fetched and hashed in `T016A/artifact_manifest.json`.
  Transfer archive SHA2563738791f64b71a8b1fc219cbeb6f0d9310a50d879da781f8b5f3e4d3554f185d.
- All primary/group metrics, coefficients, norm diagnostics, vectors, null outcomes and
  flags: `T016A/tables.md`; numerical extrema: `T016A/norm_summary.json`.

Only `taisp.analysis`, tests, reporting and receipts changed. The new module reuses
pair_rows and differential-subspace summary/cosine helpers. Each fold fits the exact
no-intercept diagonal least-squares formula with EPS1e-12. There is no centering,
clipping, ridge search, feature selection or optimizer. All four image-paired blocks
are preserved:24training/8heldout episodes,12training/4heldout imagepairs per fold.
Heldout task vectors enter only metric calculation, never coefficient fitting.

The calibrated differential vector follows the prescribed EPS norm match; the saved
shared pseudo component is unchanged. Training nulls permute task imagepairs relative
to fixed pseudo rows; clean/corrupted positions stay paired. Each null evaluates the
same heldout images. All1024 null maps/mappings and8192 heldout null predictions are
retained, together with4 primary maps and32 primary predictions: **1028 closed-form fits**.

Python3.12.12, NumPy1.26.4, float64 CPU audit on the A6000 server; elapsed2.603052380s.
Zero new detector/ISP/CLIP/optimizer/image-loading/AP calls in the audit. GPU remains
available for applicable regression workloads and is preferred for future model work;
these small saved-array fits use CPU. No new scientific GPU result is implied.

## Tests and command

- Baseline on unchanged T015 release: `pytest tests/test_differential_subspace.py tests/test_linear_capacity.py -q`:
  **13passed in1.50s**.
- Focused: `pytest tests/test_differential_calibration.py tests/test_differential_subspace.py -q`:
  **12passed in1.52s**.
- Full regression before formal fits: **140passed,10skipped**,4warnings in6.70s.
  Optional real-model tests remain skipped; this is not140real-model tests. Existing
  NVML/protobuf warnings are retained in train.log and did not prevent completion.
- Local syntax and saved-record table rendering passed.

Synthetic checks cover exact coefficients/no intercept, heldout-target isolation,
all256 image-pair schedules, norm/shared-component preservation, known productivity,
zero/one-region vectors, collapsed-map diagnosis, exact gates and corrected null tails.

Exact launch also retained in raw meta.json/run.sh:

```bash
unset CUBLAS_WORKSPACE_CONFIG
export TAISP_SOURCE_REVISION=1b19e8c TAISP_REAL_MODELS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q &&
/home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.differential_calibration \
  --prior-root /home/liujianhua/wjq/TAISP/runs \
  --manifest research_log/T016A_input_manifest.json \
  --schedule research_log/T016A_permutation_schedule.json \
  --output "$AUTODL_ARTIFACTS_DIR/audit"
```

## Numerical norm checks

All32 primary and8192 null records satisfy the precommitted absolute1e-10 plus relative
1e-8 norm bounds for both differential and full pseudo norms. Primary maximum errors:
1.70983227576e-11 differential,1.70534697475e-11 full. Null maximum errors:
4.09108302790e-10 differential,4.08026945564e-10 full. Worst error/bound ratios are
.03944357387/.02582195926 primary and .69682102465/.45617779377 null.
The explicit EPS shrinkage accounts for the small norm deficits; no second scaling,
fallback direction, tolerance adjustment or claim of bitwise norm equality is used.
Zero raw/calibrated cosine cases:0primary,0null.

## Heldout scientific results

| Scope | N | Median raw/cal cosine | Positive C_diff_cal | Median C_diff_cal | Positive Delta_D | Median Delta_D |
| --- | --- | --- | --- | --- | --- | --- |
| Overall |32|.069492 / .131101|20|.009115873|18|.012870634|
| Clean |16|.065872 / .163694|10|.029781525|10|.030192603|
| Corrupted |16|.171336 / .095804|10|.006101233|8|-.005814599|
| Block0 |8|.324431 / -.052507|4|-.000934452|4|.003878427|
| Block1 |8|-.428461 / .176587|7|.017285309|6|.067389817|
| Block2 |8|.067868 / .362689|6|.039053222|6|.039926753|
| Block3 |8|.371071 / -.026421|3|-.017501423|2|-.049570771|
| Color cast s2 |4|-.331307 / .143687|2|.000922725|3|.019053567|
| Contrast s2 |4|-.249294 / .048373|2|.183197185|2|.301930482|
| Gamma s1 |4|.345793 / .000272|2|-.025951739|2|-.081189977|
| Gamma s2 |4|.399398 / .163912|4|.011776381|1|-.013929115|

Overall C_diff positive count rises18to20 and mean rises-.01708248to.05909935,
but its median decreases.01043942to.00911587. Full D mean rises-.01701703to.05916480;
mean Delta_D is.07618183, median.01287063, range[-1.40652079,1.87079108].
Corrupted median Delta_D is negative and only8/16 improve. The fixed block/episode
conditions therefore prevent an aggregate-improvement claim from becoming a pass.

## Matched null comparisons

Linear95th percentiles; corrected empirical tail=(1+#null>=observed)/257.

| Quantity | Observed | Null95 | Null >= observed | Ties | Corrected tail |
| --- | --- | --- | --- | --- | --- |
| Median Delta_D |.0128706335091|.00740041297933|2|0|3/257=.01167315|
| Positive Delta_D count |18|18.25|30|17|31/257=.12062257|
| Positive C_diff_cal count |20|21|32|18|33/257=.12840467|
| Median calibrated cosine |.131101470250|.207458100496|43|0|44/257=.17120623|

Median improvement exceeds the matched null95 and is at the99.21875 strict empirical
percentile. Positive improvement count does not exceed its null95 (88.28125percentile).
The two other requested null diagnostics also remain below their95th percentiles.
No seeds, permutations or quantile conventions were changed after seeing outcomes.

Coordinate signs across folds (zero-based ISP order): coordinates1,2 are negative in
all4 folds;4,7 positive inall4. Coordinates0,3,6 have[-,+,-,-], coordinate5[+,+,+,-].
Full values and population standard deviations are retained; no coordinate is removed
or selected from this post-outcome stability description.

## Literal R027 advancement gate

| Flag | Result |
| --- | --- |
| C_diff_cal positive >=20overall |PASS:20|
| C_diff_cal positive >=10corrupted |PASS:10|
| Overall median C_diff_cal >0 |PASS:.009115873|
| At least3positive C_diff_cal block medians |**FAIL:2/4**|
| Delta_D positive >=20overall |**FAIL:18**|
| Delta_D positive >=10corrupted |**FAIL:8**|
| Overall median Delta_D >0 |PASS:.012870634|
| At least3positive Delta_D block medians |PASS:3/4|
| Median Delta_D above null95 |PASS|
| Positive Delta_D count above null95 |**FAIL:18<=18.25**|
| Conjunction |**FAIL**|

The median-null signal is retained, but it does not satisfy the full frozen criterion
for low-capacity calibratability. PerR027 close this diagonal/source-calibration rescue.
This does not establish absence of all learnable information or invalidate the earlier
task-capacity result. Future regional representation/objective changes require a new
explicit research task; none is implemented here. Stop NEEDS_REVIEW afterT016-A.
No newcohort, spatialfinite-step/deployment, regionalCLIP/objective redesign, AP/FCOS/SSD,
mask/region search, source/meta-training, predictorredesign or gate/dose tuning occurred.
