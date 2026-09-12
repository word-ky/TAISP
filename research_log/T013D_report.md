# T013-D final report — NEEDS_REVIEW

2026-09-13, R018. Completed the predeclared no-update repeatability and saved-data
common-mode audit. T013-C was accepted/closed by R018. Stop here for research review.

Plan commit `78ef20e` preceded outcome-bearing code `81097bf` and the real run.
Run `20260913-033722-taisp-t013d-repeatability-common-mode`, release
`20260913-033259-taisp-t013d-repeatability`, A6000 project
`/home/liujianhua/wjq/TAISP`. Started 03:37:26+08, finished 03:38:31+08, exit 0.
Analysis elapsed 54.852330033 seconds; peak allocated CUDA memory 3,714,830,848 bytes.

## Scope and validation

- Exactly 12 primary repeats × 8 fixed episodes, zero optimizer steps; all retained.
- Same four train2017 images, corruption/order, detached supports, original zero-head
  predictor, K=3/lr=0.1/eps=1e-12 and source outer objective as T013-C.
- Manifest/support/prior receipt/original checkpoint hashes verified against
  [the pre-outcome plan](T013D_plan.md). Constructor state matched the saved original
  exactly. Each fresh predictor remained unchanged; original ISP remained zero.
- Faster R-CNN and CLIP weight hashes pinned, all parameters/buffers unchanged,
  frozen model `.grad=None`. Labels used only in the source-analysis outer loss.
- No deployment/predictor/ISP/loss changes, AP/validation/new data, or optimizer probes.
- Local baseline: 9 passed in 18.05s. Focused: 12 passed in 17.15s.
  Remote regression: **98 passed, 10 skipped in 6.17s**. The real diagnostic then ran
  separately on CUDA; skipped tests are not counted as real-model passes.
- All primary JSON measurements finite. Saturation across the 96 episodes ranged
  from 0 to 0.0516609512; full states, losses, components and gradients are retained.

Only `taisp/analysis/repeatability.py` and `tests/test_repeatability.py` were added
for computation. The offline renderer `scripts/report_t013d.py` and project records
were added for reporting. Its initial integer table-header formatting error was
fixed and rendering completed; no experiment was repeated for this reporting fix.

## 1. Opposing aggregate gradients recur, but are not numerically tight

| Space | Median cosine | Minimum | Maximum | Population std | Negative repeats |
| --- | --- | --- | --- | --- | --- |
| Explicit phi0 | -0.7340253331 | -0.8587400335 | -0.5016293222 | 0.0958469579 | 12/12 |
| Predictor head | -0.7129362944 | -0.8406946249 | -0.4776152348 | 0.0966456867 | 12/12 |

Opposition is reproducible in sign and remains substantial on this fixed microset.
Its magnitude spans about 0.36, so it is not an exact `-0.73` measurement. Clean
aggregate norms vary more than corrupt norms: phi norm means/std are
0.0926621/0.00981843 (clean), 0.349017/0.00423962 (corrupt); head equivalents are
0.100620/0.0107287 and 0.397196/0.00480321.

The clean episode of image 541157 is especially variable: its repeated gradient
cosine against repeat 0 reaches 0.455802 in both spaces, while the other seven
episode minima are at least 0.981512. Maximum per-episode loss deviation from
repeat 0 is 0.0383286476, and maximum phi3 coordinate deviation is 0.00664875843,
both on that clean image. Every episode and repeat remains in the report.

This supports a stable **local geometric conflict** at the original initialization
on these four images. It does not establish finite-step cross-harm, identify the
cause of the numerical variation, or justify a conflict regularizer by itself.

## 2. Saved one-step effects relative to no-update controls

No-update group loss population std/range:
joint 0.00174623572 / 0.00660717650;
clean 0.00319986598 / 0.0122590051;
corrupt 0.000815520460 / 0.00257409830.

| T013-C probe | Measured group | Signed loss effect | Abs / std | Abs / range | Equal/larger no-update change |
| --- | --- | --- | --- | --- | --- |
| joint | joint | -0.00512896688 | 2.93716 | 0.776272 | 3/12 |
| joint | clean | -0.00356810587 | 1.11508 | 0.291060 | 11/12 |
| joint | corrupt | -0.00668982789 | 8.20314 | 2.59890 | 0/12 |
| clean | joint | -0.00183736626 | 1.05219 | 0.278086 | 11/12 |
| clean | clean | -0.00296309404 | 0.926006 | 0.241708 | 11/12 |
| clean | corrupt | -0.000711638480 | 0.872619 | 0.276461 | 4/12 |
| corrupt | joint | -0.00309802080 | 1.77411 | 0.468887 | 10/12 |
| corrupt | clean | -0.00615196582 | 1.92257 | 0.501832 | 10/12 |
| corrupt | corrupt | -0.0000440757722 | 0.0540462 | 0.0171228 | 11/12 |

Counts compare absolute repeat-minus-repeat0 changes with the absolute saved effect;
denominator 12 includes repeat 0. Repeat 0 happens to have the largest joint/clean
group loss, which affects these anchored counts. These are descriptive controls,
not p-values, confidence intervals, or optimizer rankings. No T013-C probe was rerun.

Most old effects lie within this no-update range. The exception is the joint probe's
corrupt-group effect (2.60 times the range); it should not be erased by a blanket
"all effects are noise" conclusion. Nevertheless this is one saved optimizer outcome
from the earlier run, compared with one later 12-repeat control set on a tiny microset.
It does not show repeatable optimizer superiority or restore the missing cross-harm
pattern. Full per-episode, group-radius and all phi3-coordinate comparisons are in
[the complete tables](T013D/tables.md) and raw summary.

## 3. Explicit bias is not the only shared component

Algebraic float32 one-step W/b reconstruction, then CPU float64 saved-feature
evaluation, matches saved T013-C Wh, bias and outputs with **zero maximum error**.
No model was executed in this stage. `||mean Wh|| = 2.81007567e-5`.

| Output | Total energy | Centered energy | Centered fraction |
| --- | --- | --- | --- |
| Original Wh+b | 2.40864159e-7 | 4.10743522e-10 | 0.170529% |
| Bias removed: Wh | 6.72796374e-9 | 4.10743522e-10 | 6.105020% |
| Microset common mode removed | 4.10743522e-10 | 4.10743522e-10 | 100% |

Wh itself has **93.894980% common-component energy**. Bias removal increases the
relative centered fraction but neither adds centered energy nor increases separation.
The four same-image clean/corrupt distances are 8.77128753e-6, 3.31031679e-6,
9.41395258e-6 and 5.75226767e-6. All pairwise distances are translation invariant
across these decompositions, up to floating-point rounding.

Centered Wh singular values are approximately
`[1.90643e-5, 6.87711e-6, 3.26706e-8, 8.71317e-9, 3.31544e-10, 1.76042e-11,
8.24050e-14, 1.35764e-21]`; default float64 numerical matrix rank is 7.
The spectrum is strongly concentrated in its first two directions; numerical rank
alone should not be read as rich conditioning.

Simply deleting the explicit bias is unlikely to remove shared-offset dominance
in this first update. This does not prove the feature trunk is incapable of learning.
Whole-microset centering is only the requested nondeployable diagnostic upper bound;
its 100% centered fraction is true by construction. Neither counterfactual is a
proposed method or a measured detection improvement.

## 4. Optional deterministic branch: recorded failure, stopped as specified

Primary repetitions, summaries and algebra were saved first. The single attempted
first-pair execution with `torch.use_deterministic_algorithms(True)` raised in
CLIP visual projection `F.linear`: CuBLAS requires `CUBLAS_WORKSPACE_CONFIG` for
deterministic behavior on this CUDA version. Complete traceback is retained in
`deterministic_diagnostic.json`. Zero pair repeats completed. The flag was restored,
and frozen-state checks still passed. No environment workaround or retry was added.
This diagnoses one operational requirement, not the cause of all primary variation.
The earlier T013-A CUDA assertion/tolerances remain unchanged.

## Reproduction and retained artifacts

Full [run metadata and exact shell command](remote_runs/20260913-033722-taisp-t013d-repeatability-common-mode/meta.json),
[stdout/test receipt](remote_runs/20260913-033722-taisp-t013d-repeatability-common-mode/train.log),
[completion](remote_runs/20260913-033722-taisp-t013d-repeatability-common-mode/artifacts/audit/completion.json),
[artifact SHA256 manifest](T013D/artifact_manifest.json), and
[all tables](T013D/tables.md) are retained. Audit directory contains all 12 raw
repetitions, summary, common-mode algebra, environment metadata and optional traceback.
Original predictor checkpoint and supports remain in their pinned T013-C/T013-B runs;
each primary copy was verified identical, so there are no new learned checkpoints.

Local table reproduction (no models):

```powershell
D:\anaconda3\python.exe scripts/report_t013d.py --audit research_log/remote_runs/20260913-033722-taisp-t013d-repeatability-common-mode/artifacts/audit --output research_log/T013D/tables.md
```

Ready for R018 acceptance review. No T013-E, longer training, regularizer, centering,
bias-free architecture, new data, deployment changes, or target/AP experiment started.
