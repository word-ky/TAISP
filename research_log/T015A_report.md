# T015-A — task-relevant differential capacity; pseudo utility FAIL

Status: NEEDS_REVIEW. R026 completed on 2026-09-13. The fixed source cohort has
task-relevant differential capacity under all three predeclared conditions. The
current pseudo objective fails the differential utility conjunction: **C_diff > 0
on 18/32 episodes and 9/16 corrupted episodes**, below 20/32 and 10/16.
Per R026, close the current fixed pseudo-spatial branch and stop for research review.

## Provenance, implementation and execution

- R026 pointer f27a996; pre-outcome plan and 33 input hashes committed be87bcb;
  analysis implementation 4860be7.
- Parent run: `20260913-113401-taisp-t014a1-common-jacobian`, source0ad53b6.
  Authoritative records.json SHA256
  `a0a8c59dbb6b2fd83343f774755c6061a28542bce3c1d2f00b1458e8bb655a17`.
  All individual full record hashes are in `T015A_input_manifest.json` and the new
  raw environment receipt. Debug and old reverse-mode records are not analyzed.
- Release: `20260913-122950-taisp-t015a-differential-subspace`.
- Run: `20260913-123142-taisp-t015a-differential-subspace`.
  Started12:31:49, finished12:31:58 +08:00; exit0.
- Raw receipts: `research_log/remote_runs/20260913-123142-taisp-t015a-differential-subspace/`.
  All7 files,154584 bytes fetched and hashed in `T015A/artifact_manifest.json`.
- Complete scope/per-episode metrics, norms, s/d vectors, zero counts and triage:
  `T015A/tables.md`. Numeric error extrema: `T015A/reconstruction_summary.json`.

New code is confined to `taisp/analysis/differential_subspace.py`, its synthetic tests,
the saved-record renderer `scripts/report_t015a.py`, and research records. It reuses
spatial_action.EPS=1e-12 and the saved D_spatial comparator. No ISP/model/loss/deployment
changes. All32 original episodes,16clean/16corrupted,4fixed blocks retain their order,
case labels, masks and cached supports. Existing numerical/isolation flags were checked;
reference vectors match all32 individual hashed files. No filtering or regrouping.

The audit makes zero model, ISP, CLIP, optimizer, image/data-loading or target/AP calls.
It uses NumPy1.26.4 float64 CPU algebra on the existing Python3.12.12 A6000 server
environment; elapsed **.023246104 s**. The user's GPU preference is recorded: GPU stays
available for applicable regression/model workloads, while these tiny saved-array
operations use CPU. No new GPU scientific experiment is represented by this audit.

## Tests and exact commands

- Baseline unchanged A1 release: `pytest tests/test_spatial_action.py tests/test_common_jacobian.py -q`:
  **11 passed in2.28s**.
- Focused new/affected tests: `pytest tests/test_differential_subspace.py tests/test_spatial_action.py -q`:
  **13 passed in2.05s**.
- Full regression before saved-array analysis: **135 passed,10 skipped**,4warnings in6.49s.
  Optional tests remain skipped; these are not135 real-model tests. The existing NVML
  and protobuf warnings remain in train.log; they did not block execution.
- Local syntax and stdlib saved-record rendering passed.

Synthetic tests cover shared/opposing/one-region/zero/general vectors, vector and energy
reconstruction, orthogonality, exact productivity splitting, null zero-norm cosine
reporting, and the inclusive/strict triage boundaries.

Full launch (also preserved verbatim in raw meta.json/run.sh):

```bash
unset CUBLAS_WORKSPACE_CONFIG
export TAISP_SOURCE_REVISION=4860be7 TAISP_REAL_MODELS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q &&
/home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.differential_subspace \
  --input-root /home/liujianhua/wjq/TAISP/runs/20260913-113401-taisp-t014a1-common-jacobian/artifacts/audit \
  --manifest research_log/T015A_input_manifest.json \
  --output "$AUTODL_ARTIFACTS_DIR/audit"
```

## Orthogonal reconstruction

Each objective uses s=(obj+bg)/2 and d=(obj-bg)/2, with shared=[s,s],diff=[d,-d].
The maximum vector reconstruction error over64 objectives is1.25668713465e-16,
maximum energy error8.88178419700e-16, maximum absolute orthogonality residual
4.05491612510e-17. Saved D_spatial=C_shared+C_diff holds on all32 episodes with
maximum absolute error **2.22044604925e-16**, below the frozen1e-12 bound.
There are zero task/pseudo differential zero-norm cases and zero shared zero-norm
cases. Zero-norm conventions are nevertheless exercised by the synthetic tests.

R_extra is algebraically the same ratio as the prior R_task apart from epsilon
placement: norm(t_shared)=norm(t_obj+t_bg)/sqrt(2). Its high value is therefore not
an independent replication of capacity. This postmortem additionally exposes the
differential energy and the pseudo contribution within the orthogonal subspace.

## Fixed-cohort results

| Scope | N | Median R_extra | Median task diff energy | Median C_diff | Positive C_diff |
| --- | --- | --- | --- | --- | --- |
| Overall |32|1.86180558637|.711499912159|.0104394215118|18|
| Clean |16|1.89056600798|.716356440647|.00730584706017|9|
| Corrupted |16|1.86180558637|.711499912159|.0218212500191|9|
| Block0 |8|2.08956702441|.766055836914|.0396491955948|5|
| Block1 |8|1.40557134868|.492998757464|-.0330313949246|3|
| Block2 |8|1.75500799148|.675311148953|.0144040641595|5|
| Block3 |8|2.27520968721|.798099303615|.0241169653713|5|
| Color cast s2 |4|1.56643670349|.527025109411|-.0137463637006|1|
| Contrast s2 |4|2.07234642315|.766285544989|-.118733297361|1|
| Gamma s1 |4|2.00626026243|.738780270712|.0399190266725|3|
| Gamma s2 |4|1.64267428341|.610196965089|.0306272639973|4|

Overall pseudo differential energy fraction median .679021438565 and amplitude norm
fraction median .823982149301: the pseudo objective assigns substantial norm to the
differential subspace. Median cos_diff is .0694917699356;18dots positive,14negative,
none zero. C_diff mean is **-.0170824788497**, despite positive median, with range
-1.02495339263 to1.34312462605. C_shared mean is .0000654485751770 and median
.00129567245812; D_spatial mean is -.0170170302745. The positive unnormalized mean
differential dot (.0344847381864) does not reverse the negative mean normalized
C_diff; the specified productivity denominator is retained.

## Literal R026 triage and stop

| Predeclared flag | Result |
| --- | --- |
| Overall median R_extra >=1.10 | PASS:1.86180558637 |
| Corrupted median R_extra >=1.10 | PASS:1.86180558637 |
| At least3blocks median R_extra >1.05 | PASS:4/4 |
| C_diff positive >=20/32 overall | **FAIL:18/32** |
| C_diff positive >=10/16 corrupted | **FAIL:9/16** |
| Overall median C_diff >0 | PASS:.0104394215118 |
| At least3positive C_diff block medians | PASS:3/4 |

Task-relevant YES; pseudo differentially useful NO. On this frozen source cohort the
object/background space contains substantial task-relevant differential capacity,
but the accepted detector-native pseudo objective does not exploit it reliably
enough under the specified rule. Close this fixed pseudo-spatial branch per R026.
The earlier R024 failed gate remains failed; no threshold is softened, and no spatial
finite-step/AP or population-level claim follows from this algebraic postmortem.

No additional model runs or changes are needed to finish T015-A. Stop NEEDS_REVIEW;
await the next explicit research task. No T014-B/T015-B, differential-only deployment,
finite-step spatial adaptation, region/mask search, regionalCLIP, source/meta-training,
predictor redesign, FCOS/SSD/AP, newcohort or gate/dose tuning was started.
