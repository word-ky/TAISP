# T017-A — BLOCKED before scientific attribution

R028 was executed through the first numerical record. The GPU run stopped on the
first episode, index0/image182164/clean_s0. Component-sum reconstruction and saved-A1
relative-L2 parity failed. **1/32 records collected,0passed,31unexecuted; localization
versus confidence attribution and the scientific triage are NOT_REACHED.**
No tolerance change, numerical workaround, alternate precision or second run occurred.

## Provenance and scope

- R028 pointerf8fd9eb; pre-outcome plan/inputhashes **e506f79**; code **5e0724b**.
- Release `20260913-142715-taisp-t017a-task-components`.
- Run `20260913-142838-taisp-t017a-task-components`.
  Started14:28:44, finished14:28:57 +08:00; exit1.
- Raw receipts: `research_log/remote_runs/20260913-142838-taisp-t017a-task-components/`.
  All6 files,24,467 bytes fetched and hashed in `T017A/artifact_manifest.json`.
  All component/current-total8-D vectors, coordinate errors and parity tables are
  retained in `T017A/numerical_tables.md`; raw blocker.json records the stop/counters.
- A1 records/cohort/supports/environment and T015 records hashes are pinned in
  `T017A_input_manifest.json` and verified before execution. Source weight/state hashes
  match the inherited258fb6.../73eed6... pins. Exact full hashes are in raw environment.json.

Only new analysis collector/attribution code and tests/reporting were added. The
existing oracle, model, ISP, deployed losses, masks, supports and adaptation paths
remain unchanged. The one seeded native detector forward returns all four requested
losses. Each component cotangent and the independent summed-loss cotangent are
projected through the same eight ISP JVP columns with float64 regional accumulation.
The current-total comparator is not defined as a component sum.

## Tests and environment

- Baseline unchanged T016 release: common_jacobian+differential_subspace,
  **12passed in1.88s**.
- Focused task_components+common_jacobian: **9passed in1.94s**.
- Full regression before GPU collection: **144passed,10skipped**,4warnings in6.66s.
  Optional tests remain skipped; this is not144real-model tests.
- Local syntax and saved-receipt renderer passed.

Synthetic tests cover a single forward/common-JVP component closure, independent-total
reconstruction, signed/cancelling and grouped attribution, zero differential cases,
pseudo-productivity reconstruction and fixed dominance/utility thresholds.

Python3.12.12, torch2.4.0+cu121, NumPy1.26.4, NVIDIA RTX A6000 cuda:0.
Float32 detector/ISP, float64 projections, seed20260913, deterministic algorithmsfalse,
cuDNN benchmarkfalse, CUBLASunset, threads1. The existing NVML warning is in the log;
CUDA execution did run. No driver or environment change was made.

Exact launch (also preserved in raw meta.json/run.sh):

```bash
unset CUBLAS_WORKSPACE_CONFIG
export TAISP_SOURCE_REVISION=5e0724b TAISP_REAL_MODELS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q &&
/home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.task_components \
  --prior-root /home/liujianhua/wjq/TAISP/runs \
  --manifest research_log/T017A_input_manifest.json \
  --output "$AUTODL_ARTIFACTS_DIR/audit"
```

## First-record blocker

All four scalar losses and their total exactly match the saved A1 record:

| Component | Loss |
| --- | --- |
| loss_classifier |.0756167620420456|
| loss_box_reg |.1729314923286438|
| loss_objectness |.02523311227560043|
| loss_rpn_box_reg |.0075882477685809135|
| Native total |.2813695967197418|

The mask hash/rectangles match A1;5supports, area.30939700704225354. All source frozen,
eval,gradNone,state-hash and ISP identity/gradNone checks pass. All five objective
object/background partition closures pass. JVP primal identity max error5.960464478e-8.
Direct global reverse vs current total common-Jacobian parity passes:
cosine.9999999999998989,relativeL2 **4.897136043e-7**.

However, independently projected component sums fail the precommitted float64 bound
`1e-12 + 1e-10*sum(abs(component_gradient))` in all regions:

| Region | Max absolute discrepancy | Max error/bound |
| --- | --- | --- |
| Global |6.986881543e-5|9,612,367.59|
| Object |4.271232911e-5|4,842,434.21|
| Background |4.570763189e-5|7,330,959.81|

Saved-A1 parity also fails the explicitly required relativeL2<=1e-5, even though all
cosines exceed.999999. Thus this is not only a consequence of the strict float64
component-closure threshold:

| Region | Component sum relativeL2 vs A1 | Independent current total relativeL2 vs A1 |
| --- | --- | --- |
| Global |9.819567371e-4|2.633194794e-4|
| Object |4.945390400e-4|2.224348290e-4|
| Background |5.237177251e-4|3.098840982e-4|
| Concatenated regions |5.014846381e-4|2.455934294e-4|

For example, global coordinate2 component sum=-.0934066919520435 versus independently
projected total=-.09333682313661347, an absolute difference6.986881543e-5.
The raw record contains every coordinate and all four component vectors.

Identical scalar forward losses and unchanged state localize the observed discrepancy
to the gradient/replay comparison, but do not establish its root cause. Separate
float32 detector backwards and CUDA execution differences are possible contributors;
they have not been isolated. Float64 ISP projection alone cannot guarantee additive
equality of cotangents already produced by those backward computations. No diagnostic
model rerun was authorized after this material failure, so none was performed.

## Counts and stop

One detector forward,5image-cotangent requests,8ISP JVP columns. Zero pseudo-objective
recomputations,CLIP,optimizer,finite-step updates or AP calls. Collection elapsed
1.311475714s; peak CUDA allocated1,859,385,344bytes. The remaining31episodes did not run.
No A_loc/A_conf,component-productivity statistics or dominance decision was computed
from this partial cohort. The earlier T015/T016 conclusions remain unchanged.

Status **BLOCKED**, not completed scientific attribution and not a negative result for
either localization or confidence. Request research review of the independent-total
and cross-run gradient reconstruction discrepancy before continuing. Preserve the
original record and thresholds. No newcohort, regional objective/localization loss,
CLIP,spatial adaptation,AP/FCOS/SSD,mask search,calibration/training,predictor redesign
or deployment change was started. No active experiment remains.
