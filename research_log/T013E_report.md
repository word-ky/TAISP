# T013-E report — NEEDS_REVIEW; deterministic measurement blocked

2026-09-13. R019/74cbdce accepted and closed T013-D. T013-E stopped at the
predeclared unsupported-operator condition. **No deterministic repeat completed;
the exact repeatability gate was not reached; zero optimizer probes were run.**
No finite-step scientific conclusion can be drawn from this task.

## Exact blocker

The analysis launcher set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python started.
The runner enabled `torch.use_deterministic_algorithms(True)` and retained
`torch.backends.cudnn.benchmark=False`. The earlier CLIP CuBLAS projection blocker
was passed, but the first attempted repeat failed during the CLIP gradient:

```text
RuntimeError: upsample_bicubic2d_aa_backward_out_cuda does not have a deterministic
implementation, but you set 'torch.use_deterministic_algorithms(True)'.
```

Call path: `deterministic_replay.run -> gradient_conflict.evaluate ->
source_meta_smoke.source_outer_episode -> adapt_from_initialization ->
trust_radius._adapt_radius -> torch.autograd.grad(clip, probe)`.
The operator is the backward of the existing antialiased bicubic CLIP tensor
resize. This is a different blocker from T013-D's missing CuBLAS environment setting.

The complete [structured traceback](remote_runs/20260913-045756-taisp-t013e-deterministic-replay-fixed/artifacts/audit/runtime_failure.json)
and [run log](remote_runs/20260913-045756-taisp-t013e-deterministic-replay-fixed/train.log)
are retained. No `warn_only`, deterministic-flag disable, CPU fallback, different
resize/kernel, additional environment setting, precision change, or retry was applied
after this failure. Existing numerical tolerances and T013-A assertion were untouched.

R019 explicitly requires: “If another unsupported deterministic operator raises,
preserve the full error and stop T013-E before optimizer probes.” This stopping
condition has been followed; the task does not authorize implementing a new kernel.

## Execution and tests

Pre-outcome plan [T013E_plan.md](T013E_plan.md), commit `604528c`, preceded analysis
code `b1278c4` and all real outcomes. Final runner source is `a585749`.

| Stage | Result |
| --- | --- |
| Local baseline, existing affected suites | 6 passed, 10.06s |
| New exact-gate/correlation/decision and affected tests | 9 passed, 7.14s |
| Focused tests after setup-only repair | 9 passed, 6.99s |
| First remote regression | 101 passed / 10 skipped, 5.91s |
| Final remote regression | 101 passed / 10 skipped, 5.83s |
| Required 3 no-update repeats | 0 completed; first attempt raised |
| Bitwise gate | Not reached, not passed |
| Joint / clean / corrupt SGD probes | 0 / 0 / 0 |
| Group losses, phi3, gradient geometry, comparisons | Unavailable for T013-E |

The regression uses `TAISP_REAL_MODELS=0`; skipped checks are not counted as real
model passes. The separate CUDA diagnostic reached the real CLIP backward and failed.
The conditional probe path is implemented with reused T013-C functions and focused
tests, but it was **not exercised end to end** because the scientific gate was unmet.

Final run `20260913-045756-taisp-t013e-deterministic-replay-fixed`, release
`20260913-045732-taisp-t013e-eval-init`, source `a585749`.
Started 04:58:00+08, stopped 04:58:30+08, exit 1. No job remains active from this run.
Server root `/home/liujianhua/wjq/TAISP`.

Environment: NVIDIA RTX A6000; Python 3.12.12, torch 2.4.0 / CUDA 12.1,
torchvision 0.19.0+cu121, transformers 4.44.2, numpy 1.26.4.
Runtime flags, prompts, model file checksums and initial model state hashes are in
[environment.json](remote_runs/20260913-045756-taisp-t013e-deterministic-replay-fixed/artifacts/audit/environment.json).
The known NVML warning did not prevent CUDA execution and was not changed.

## Setup-only failure and smallest repair

The first launch `20260913-045510-taisp-t013e-deterministic-replay` (release
`20260913-045442-taisp-t013e-deterministic`, source `b1278c4`) stopped at 04:55:28+08,
exit 1, before any sample evaluation. The newly added initial isolation check expected
every wrapper's `training` flag to be false. `load_clip_guidance` returns a
`SemanticDirectionLoss` wrapper whose default flag is true even though its encoder
and actual CLIP model are already eval/frozen. The existing shared inner loop sets
that outer wrapper to eval before computation.

The [local diagnostic](T013E/initial_wrapper_diagnostic.txt) reproduced this flag issue:
initial frozen check false; after `wrapper.eval()` true, with unchanged tensors.
The only repair was `clip.eval()` before the new early isolation check in the
analysis runner. No production code, weights, data, preprocessing or objective changed.
The first run contained no primary or probe outcome; it was retained in full.
This engineering restart did not discard a scientific mismatch or unfavorable outcome.

## Inputs and isolation

Same four train2017 images/eight episodes, original zero-head predictor checkpoint,
frozen supports, seed 20260913, K=3, inner lr=0.1, norm epsilon=1e-12, and source outer
objective as T013-C/T013-D. No new annotations were used beyond the fixed source
analysis targets. All task input hashes are listed in the pre-outcome plan; the runner
verified manifest/support/T013-C receipt/original checkpoint and all 12 used T013-D
raw control files against the pinned T013-D artifact manifest. No supports recomputed.

Initial predictor construction exactly matched the saved checkpoint. Initial
source/CLIP parameters and buffers were unchanged and `.grad=None`; original
predictor and ISP were unchanged. Initial in-memory state SHA256 values:

- Source: `73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf`
- CLIP: `6b38ac3696ff07a4e0cdbe436db05551707486e413df526256526b5777fe147c`

These hashes include state names, dtype, shape and bytes; they are distinct from the
pinned model-file SHA256 values. The process exited during unsupported backward,
so a post-failure isolation receipt was not recorded. We do not claim that missing
check passed. No optimizer or model update was reached, and no new checkpoint exists.

## Code, records and exact reproduction

Added only `taisp/analysis/deterministic_replay.py`, `tests/test_deterministic_replay.py`
and project records. Reused `gradient_conflict` evaluation/geometry/directions/SGD/
first-order comparison and `source_meta_smoke` episodes/outer loss/frozen checks/group
statistics. New logic is limited to strict output-bit comparison, the conditional gate,
prior-range reporting, average-tie Spearman and the predeclared signed decision rule.

[Exact launch metadata](remote_runs/20260913-045756-taisp-t013e-deterministic-replay-fixed/meta.json)
and `run.sh` retain the complete command. Command structure:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TAISP_SOURCE_REVISION=a585749 TAISP_REAL_MODELS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q
# Only after pytest success, the metadata's exact deterministic_replay command ran.
```

Both launches and all available outputs are indexed by the
[artifact SHA256 manifest](T013E/artifact_manifest.json). The final audit directory
contains `environment.json` and `runtime_failure.json`; no missing repeat/probe
JSON has been replaced with synthetic results. Operational disposition is retained in
[T013E/disposition.json](T013E/disposition.json).

## Research consequence and stop

R019's finite-step discriminator remains unanswered. T013-D's robust local gradient
opposition and predominantly common-mode Wh remain the latest scientific evidence.
This operator error neither proves nor refutes finite-step cross-harm, and does not
identify the source of all earlier nondeterministic variation. No objective or
predictor redesign follows from this failure.

**NEEDS_REVIEW with Stage A blocked.** Request the research lead's next explicit
decision about measurement feasibility. No T013-F, longer training, regularizer,
bias deletion, centering, predictor redesign, new data, validation/AP, spatial ISP,
gating or dose work started. The heartbeat may consume a new explicit task after review.
