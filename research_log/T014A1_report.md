# T014-A1 — numerical replay complete; R024 scientific gate FAIL

Status: NEEDS_REVIEW. R025 completed on 2026-09-13. The common-Jacobian
reference passes the predeclared numerical checks on all three debug episodes and
the subsequent fresh 32-episode cohort. The unchanged scientific conjunction fails:
positive Delta_D counts are **19/32 overall and 9/16 corrupted**, below 20 and 10.
No T014-B or spatial deployment experiment is justified by this task's gate.

## Provenance and implementation

- Research instruction: `coordination/CHATGPT_TO_CODEX_R025_T014A1.md`, pointer c584887.
- Pre-outcome plan: 89cac87; implementation: 0ad53b6.
- Release: `20260913-113310-taisp-t014a1-common-jacobian`.
- Run: `20260913-113401-taisp-t014a1-common-jacobian`.
- Started 11:34:06, finished 11:34:35 +08:00; exit 0.
- Raw receipts: `research_log/remote_runs/20260913-113401-taisp-t014a1-common-jacobian/`.
- SHA256 manifest: `T014A1/artifact_manifest.json` (45 files, 1,092,765 bytes).
- Complete numerical, geometry, subgroup, loss/saturation and 8-D vector tables:
  `T014A1/tables.md`; compact numerical extrema: `T014A1/numerical_summary.json`.

Only analysis code changed: `common_jacobian.py` reuses the original loss cotangent
and computes eight JVP columns through the existing global ISP. Each column is shared
between task and pseudo objectives. Cotangent, mask and column are cast to float64
before three independent global/object/background reductions; no region is defined
by subtraction. `spatial_action.identity_gradients` optionally returns its detached
cotangent, preserving its default behavior and original assertions. The old reverse
regional vectors remain diagnostics. `tests/test_common_jacobian.py` covers the new
reference. `scripts/report_t014a1.py` renders saved records without model calls.

The exact A manifest, H supports/cohort/environment, source weights/state and old
failed receipt/blocker hashes were checked before and after the run; see raw
`environment.json`. The original A record and `T014A/blocker.json` remain unchanged.
The old run is still BLOCKED at its original assertion. This new explicitly authorized
reference validates the subsequent rerun; it does not retrospectively pass that run.

Same 16 source images, 32 clean/corrupted episodes, four fixed blocks, cached pseudo
supports, binary detached masks, seed 20260913 and original losses. Source annotations
enter only the analysis task oracle. Models remain frozen; there are no test labels,
optimizer steps, CLIP calls, target/AP measurements or finite-step spatial updates.

## Execution and tests

Python 3.12.12, torch 2.4.0+cu121, NumPy 1.26.4, CUDA 12.1, NVIDIA RTX A6000.
Original float32 model/ISP path, float64 reference reductions; deterministic algorithms
false, cuDNN benchmark false, CUBLAS_WORKSPACE_CONFIG unset. Existing NVML warning is
retained in the log; CUDA execution completed successfully.

- Unchanged baseline: 10 passed in 2.09 s.
- Focused analysis tests: 11 passed in 2.17 s on remote CPU.
- Full regression before real replay: **128 passed, 10 skipped**, 4 warnings in 6.48 s.
  Optional tests remain skipped; these are not described as 128 real-model tests.
- Local syntax compile and saved-receipt table rendering passed.
- Real replay: 3 fixed debug episodes, then all 32 fresh episodes from index 0;
  70 source-loss calls, 280 ISP JVP columns, zero optimizer steps/CLIP calls.
- Audit elapsed 19.707982376 s; peak CUDA allocated 1,877,246,976 bytes.

The full exact launch is retained in raw `meta.json` and `run.sh`:

```bash
unset CUBLAS_WORKSPACE_CONFIG
export TAISP_SOURCE_REVISION=0ad53b6 TAISP_REAL_MODELS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q &&
/home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.common_jacobian \
  --manifest research_log/T014A_source_manifest.json \
  --prior-root /home/liujianhua/wjq/TAISP/runs/20260913-081117-taisp-t013h-source-replication/artifacts/collection \
  --old-run /home/liujianhua/wjq/TAISP/runs/20260913-101408-taisp-t014a-spatial-action \
  --old-blocker research_log/T014A/blocker.json \
  --output "$AUTODL_ARTIFACTS_DIR/audit"
```

## Stage C and full-cohort numerical results

Each objective passes coordinate closure
`error <= 1e-12 + 1e-10*(abs(object)+abs(background))`, direct-global parity
cosine >= .999999 and relative L2 <= 1e-5, plus image and frozen-state checks.

| Debug index | Image/case | Objective | Max closure error | Parity relative L2 | Old reverse check |
| --- | --- | --- | --- | --- | --- |
| 0 | 182164 clean | task | 9.71445e-17 | 2.99257e-7 | pass |
| 0 | 182164 clean | pseudo | 3.81639e-17 | 2.31001e-7 | pass |
| 5 | 425480 contrast_s2 | task | 2.22045e-16 | 2.02016e-7 | pass |
| 5 | 425480 contrast_s2 | pseudo | 1.11022e-15 | 2.42786e-7 | pass |
| 6 | 410054 clean | task | 3.46945e-17 | 4.89543e-7 | pass |
| 6 | 410054 clean | pseudo | 2.91434e-16 | 5.06491e-7 | FAIL, diagnostic |

All six debug parity cosines are at least .9999999999998749. Worst closure/bound
ratio is 1.87976011048e-5. The old debug index-6 pseudo discrepancy remains visible
(4.76837158203e-7, ratio 1.44508367721); the original A receipt is independently
preserved with its original discrepancy 6.25848770142e-7 and ratio 1.89668603069.

Across the 64 fresh full-cohort objectives, maximum closure error is 1.33226762955e-15,
maximum closure/bound ratio .000124111170226, minimum parity cosine
.9999999999979398, maximum relative L2 2.04213488758e-6. All numerical and source/ISP
isolation checks pass. Old reverse diagnostics fail for pseudo at indices 6 and 13;
neither is hidden or used to change the new thresholds. Normal CUDA execution is not
bitwise deterministic; debug/full cotangents are separately evaluated and retained.
Only fresh full records contribute to scientific geometry.

## Unchanged R024 scientific result

| Scope | N | Median R_task | Median Delta_D | Positive Delta_D |
| --- | --- | --- | --- | --- |
| Overall | 32 | 1.86180558637 | .0129286963886 | 19 |
| Clean | 16 | 1.89056600799 | .0144530724448 | 10 |
| Corrupted | 16 | 1.86180558637 | .0125861896369 | 9 |
| Block 0 | 8 | 2.08956702442 | .0129286963886 | 5 |
| Block 1 | 8 | 1.40557134869 | -.0114778201327 | 3 |
| Block 2 | 8 | 1.75500799149 | .02781605548 | 6 |
| Block 3 | 8 | 2.27520968721 | .0174252303778 | 5 |
| Color cast s2 | 4 | 1.56643670350 | .00571408914297 | 2 |
| Contrast s2 | 4 | 2.07234642315 | -.115582268208 | 1 |
| Gamma s1 | 4 | 2.00626026244 | .0221449556246 | 3 |
| Gamma s2 | 4 | 1.64267428342 | .0485288524527 | 3 |

Overall task regional cosine median -.489268480551 (6/32 positive), pseudo regional
cosine median -.565595607834 (8/32 positive); task cancellation median .446780215489.
A_global median .0323934971304, A_spatial .0700014296686, Delta_A -.00940817066410.
D_global median .000853277264387, D_spatial .00412155976051. Delta_D median is positive
but its mean is **-.0302653653821** (range -1.50697471877 to 1.14297992328); the tails
and the failed positive counts prevent a broad improvement claim.

All 32 supports are nonempty. Mean mask area .520579713284; mask strata contain
3 small, 27 middle and 2 large masks, with zero empty/full masks. Full subgroup
statistics are retained. R_task alone is not a sufficient heterogeneity/productivity
criterion: the norm-budget convention can give sqrt(2) even for a degenerate partition.

| Frozen gate | Result |
| --- | --- |
| Overall median R_task >= 1.20 | PASS |
| Corrupted median R_task >= 1.15 | PASS |
| Positive Delta_D >= 20/32 overall | **FAIL: 19/32** |
| Positive Delta_D >= 10/16 corrupted | **FAIL: 9/16** |
| Overall median Delta_D > 0 | PASS |
| At least 3/4 positive block medians | PASS: 3/4 |
| Conjunction | **FAIL** |

The numerical blocker is resolved for the new reference. Task gradients show regional
differences, but the current pseudo direction under this fixed partition does not meet
the predeclared utility criterion. This is a local first-order source audit, not an AP
result or a theorem against spatial ISP. No thresholds, masks, prompts, learning rates,
ISP ranges or cohorts were adjusted. Stop NEEDS_REVIEW; return the next research
decision to the research lead. No T014-B, deployment, regional CLIP, meta-training,
predictor redesign, FCOS/SSD/AP or additional search was started.
