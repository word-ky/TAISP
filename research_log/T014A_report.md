# T014-A — BLOCKED at fixed gradient reconstruction check

The identity spatial audit is **incomplete**: six episodes passed, the seventh was
saved but failed the predeclared pseudo-gradient reconstruction bound, and the
remaining25 episodes were not executed. The scientific geometry stage and advancement
gate were **NOT_REACHED**. This is a numerical implementation blocker, not a negative
result about spatial ISP capacity. No tolerance change or rerun was made.

## Plan, implementation and tests

- R024/f5da87a accepts/closes T013-I and authorizes only T014-A.
- Plan/cohort commit **90ad9e3**, pushed before model calls; code **767ba1c**.
- The exact first four images of each H block were fixed, giving16images/32episodes
  with inherited corruption assignments. All original cached H support/cohort/environment
  hashes matched the pins. Selected manifest SHA256:
  `a4b250a877ae1171cd6f30899f531ba2645cb37ff90abf1a26b8de48d6aa558e`.
- Added only `taisp.analysis.spatial_action` and tests. The compositor unions detached
  cached detector boxes using clipped floor/ceil rasterization, then combines two
  independent existing ISP calls. No changes under ISP, deployment, predictor or losses.
- Direct global gradients and regional ISP VJPs use the same image-output cotangent
  from one loss call per objective. At identical output this is the exact chain rule;
  it avoids a second detector evaluation contaminating the reconstruction check.
  No H K3 gradient was reused, and no oracle label entered the mask/pseudo path.
- Baseline remote CPU: **8 passed, 1 skipped in1.75s**.
- Focused compositor/ISP: **10 passed in2.02s**, including mask clipping/detachment,
  identity/shared-state output, shared-state gradients, independent-state gradcheck,
  common-cotangent equivalence to direct regional loss gradients, empty/full masks,
  zero-pseudo behavior and fixed geometry/gate boundaries. Local syntax check passed.
- Full regression before the real audit: **123 passed, 10 skipped in6.84s**.
  Ten pretrained integration tests were explicitly disabled; the following audit used
  real frozen Faster R-CNN. Existing NVML/protobuf warnings did not prevent execution.

## Exact run and failure

Run **20260913-101408-taisp-t014a-spatial-action**, release
`20260913-101321-taisp-t014a-spatial-action`, source767ba1c, on A6000.
Start/end2026-09-13 **10:14:13–10:14:28 +08:00**, exit **1**.
[Raw run](remote_runs/20260913-101408-taisp-t014a-spatial-action/) retains the exact
command in meta.json/run.sh, full test/runtime log and all saved records.

The precommitted gradient bound is elementwise
`abs(regional_sum-direct) <= 1e-7 + 1e-5*abs(direct)`.
Failure is episode index6 (seventh), **image410054 / clean_s0**, pseudo objective,
brightness raw-state coordinate5:

| Quantity | Value |
| --- | --- |
| Direct global gradient | -0.022996962070465088 |
| Object gradient | 0.20501339435577393 |
| Background gradient | -0.22801098227500916 |
| Regional sum | -0.02299758791923523 |
| Absolute error | 6.258487701416016e-7 |
| Predeclared coordinate bound | 3.2996962070465087e-7 |
| Error / bound | **1.8966860306870073** |
| Whole-vector relative L2 error | 1.0989012348867046e-6 |

Only this coordinate exceeded the bound. Re-adding the two already saved float32
regional values in float64 gives the same offending difference; changing only the
last addition precision would not remove it. The two regional contributions oppose
and nearly cancel in this coordinate. The pattern is compatible with accumulated
float32 reduction error, but the root cause is **not established** by these records.
The small relative L2 discrepancy does not retroactively override the elementwise rule.

For this episode, the task-gradient sum check passed (maximum error2.980232239e-8).
Both objectives' regional/global output images were bitwise equal, and identity versus
input had maximum error5.960464478e-8, within the inherited image bound. The mask had
area fraction **0.316496** and three pseudo supports; empty support was not the cause.

All seven saved episodes have finite gradients and passed source frozen/eval/gradNone,
source-state hash and ISP unchanged/gradNone checks. Source state stayed
`73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf`.
The real source weight hash was verified. There were14 source-loss calls in the saved
records, zero CLIP calls, zero optimizer steps and zero finite-step spatial updates.
Peak memory and full-run completion receipt were not written because execution stopped
at the assertion; they must not be inferred as zero.

## Retained evidence and next action

[Blocker tables](T014A/blocker_tables.md) retain all seven reconstruction receipts and
all eight coordinates of the failed pseudo comparison. [Blocker receipt](T014A/blocker.json)
records6passed/1failed/25unexecuted and NOT_REACHED scientific gate.
[Artifact manifest](T014A/artifact_manifest.json) hashes **13 files /217,657bytes**,
including all seven full vector records, environment,16-image cohort, cached supports,
launch command and failure log. Mask rectangles, area, shape and uint8 hash permit exact
mask recovery from each record; no image was replaced or dropped.
Failed raw record SHA256:
`ffb8e0561043d6e5d1e9bbdd466fe53bc9b8192912660076a837ccdade52fcae`.

No R_task, Delta_D, block summary or advancement conclusion was computed from the
partial cohort. The geometry code is ready but was never reached by this run.
The next research review should resolve the numerical reconstruction blocker before
authorizing continuation; no tolerance tuning, alternate kernel, new precision path
or selected repeat was tried. Preserve the current failure as the source of truth.

**T014-A remains BLOCKED, not DONE.** No active job/transfer remains. Do not begin
T014-B, spatial deployment, CLIP regional scaling, meta-training, predictor redesign,
gate/dose, FCOS/SSD/AP or target/validation work from these incomplete records.
