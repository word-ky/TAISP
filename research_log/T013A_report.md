# T013-A first-order initialization feasibility

**NEEDS_REVIEW. Required synthetic plumbing checks completed; optional CUDA
bitwise parity failed and is retained as a limitation. No real predictor training
or COCO experiment was run.** R015 task5e8afe6; plan **cd3060c** was committed and
pushed before code. Implementation **bfd2484**, unchanged after verification.

## Implementation and approximation

`taisp.training.initialization.adapt_from_initialization(..., phi0, steps=3,
lr=.1, eps=1e-12)` calls the existing private full-hybrid loop with explicit phi0.
The public `adapt_clip_radius` signature/defaults and detached deployment outputs
are unchanged. `adapt_half_dose` remains unchanged as well. No detector, CLIP,
ISP operator, ParameterPredictor architecture, loss, prompt, data or evaluator
was modified. No external donor code was introduced.

The explicit initializer is cloned without detaching; each detector/CLIP gradient
is computed at a detached copy of the current state with `create_graph=False`.
The accepted norm transfer is applied to these detached gradients, then
`phi_next = phi - .1 * stopgrad(update)` preserves the chain to phi0. The terminal
enhanced image is recomputed through only the small ISP graph. The resulting
state Jacobian is identity: this is the prescribed **FOMAML-style first-order
approximation**, not an exact detector/CLIP meta-gradient. It exposes an outer
gradient; it does not implement a source-training policy or outer optimizer in
the deployment API. The sole optimizer is in the synthetic sanity script.

## Required tests and synthetic result

Six new pytest cases in `tests/test_initialization.py` cover:

- Exact full trajectories, diagnostics (excluding wall time), terminal phi and
  images for zero and nonzero explicit initializers, against accepted deployment.
- Independent episode resets, unchanged caller/owned state, and exact empty
  support fallback relative to nonzero phi0, including its identity Jacobian.
- K=3 terminal-image outer gradient to phi0; explicit 8x8 identity state Jacobian;
  all eight inner gradient calls use fresh leaves and no second-order graph.
- Nonzero predictor-head gradient at its existing zero initialization; a fixed
  nonzero-head fixture also gives finite nonzero feature-trunk gradient.
- Detector/CLIP frozen/eval with no parameter gradients; ISP-owned state untouched;
  original deployment signature and output detachment retained. Existing support
  isolation and half-dose tests remain green.

`scripts/sanity_t013a.py` uses CPU seed **20260913**, one fixed 1x3x9x11 synthetic
image, target `x+.08`, pixel MSE, existing predictor, K=3/lr=.1/eps=1e-12. Eight
SGD outer updates at lr=.2 decrease loss monotonically:

`0.005294277333 -> 0.004555691965 -> 0.003916862886 -> 0.003364897100 ->
0.002888521412 -> 0.002477848670 -> 0.002124246908 -> 0.001820139238 -> 0.001558897318`.

All eight recorded head-weight gradient norms are finite/nonzero (.02437968 down
to .01425504). Exact values/environment are in `T013A/synthetic_sanity.json`.
This is **software/gradient sanity only, not detection or learned-initialization
performance evidence**.

## Commands and verification

Local Python `D:/anaconda3/python.exe`3.12.7, torch2.13.0+cpu; PYTHONUTF8=1,
OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1.

| Command | Outcome |
| --- | --- |
| Pre-change `python -m pytest tests/test_trust_radius.py tests/test_half_dose.py -q` | 5 passed,3 skipped,10.60s |
| `python -m pytest tests/test_initialization.py tests/test_trust_radius.py tests/test_half_dose.py -q` | 11 passed,3 skipped,10.43s |
| `python -m scripts.sanity_t013a --output research_log/T013A/synthetic_sanity.json` | exit0, loss decrease above |
| `python -m pytest tests -q` locally | 84 passed,14 skipped,1 failed,19.40s: existing replication test requires missing pycocotools |
| Same complete suite on A6000 with TAISP_REAL_MODELS=0 | **89 passed,10 skipped**,5.09s;2 existing NVML warnings |

Remote Python3.12.12/torch2.4.0+cu121; existing project venv and cached pinned
models. Release **20260913-000723-taisp-t013a-plumbing**. Main receipt directory
`remote_runs/20260913-000838-taisp-t013a-feasibility-fixed/`; exact workflow command
is in its `meta.json` and `run.sh`. The final regression was run from that resolved
release with `/home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q`,
TAISP_REAL_MODELS=0 and OPENBLAS/OMP/MKL_NUM_THREADS=1; exit0 output is saved as
`regression.log`. The ten opt-in real-model tests were intentionally not launched
as a full model suite; the task permits only one tiny optional real-model smoke.

## Optional real-model smoke: retained failure and diagnostic

One synthetic-image smoke fixture (seed20260913, CUDA0 RTX A6000,1x3x129x161,
one detached fixed synthetic box/class/score) uses the existing frozen source
Faster R-CNN and CLIP. No COCO images/annotations or target detectors are loaded.
The strict zero-tolerance assertion in `scripts/smoke_t013a.py` **failed**:
maximum phi discrepancy **2.1182699129e-5**. The assertion remains unchanged and
the failure is preserved in `smoke.log`; this is not reported as a passing test.

A bounded diagnostic of this same fixture was then run, with predictor output
squeezed to the deployment state's (8,) shape and one repeated original deployment
trajectory for comparison. `T013A/diagnose_cuda.py` deliberately records numerical
differences rather than declaring a parity pass. Its receipt reports:

| Measurement | Value |
| --- | ---: |
| Repeated deployment vs original deployment max phi difference | 8.5884239525e-6 |
| Connected vs original deployment max phi difference | 9.0480316430e-6 |
| Connected vs original deployment max image difference | 5.0663948059e-6 |
| Outer phi0 gradient norm | .1245225221 |
| Predictor head-weight gradient norm | .0540611036 |
| Diagnostic execution time (including model loading) | 3.591554s |

All source/CLIP parameters and buffers are unchanged, all submodules eval, no
model/support gradient is populated, and ISP-owned state stays zero. There is no
observed autograd or memory blocker. Repeated deployment itself is not bitwise
stable on this CUDA fixture. That provides context for the small discrepancy,
but does not establish its exact kernel cause or a numerical acceptance bound.
**GPU bitwise parity is not established.** Mandatory CPU/mock same-gradient
forward parity is exact. No algorithm change, tolerance relaxation or outcome
tuning was made to turn the optional failure into a pass.

Diagnostic command: from the same release, with PYTHONPATH=.,
TAISP_SOURCE_REVISION=bfd2484 and OPENBLAS/OMP/MKL_NUM_THREADS=1,
`/home/liujianhua/wjq/TAISP/.venv/bin/python research_log/T013A/diagnose_cuda.py
--output /home/liujianhua/wjq/TAISP/runs/20260913-000838-taisp-t013a-feasibility-fixed/artifacts/cuda_diagnostic.json`.
Exit0 output is retained in `cuda_diagnostic.log`; final JSON and diagnostic source
are archived locally and on A6000.

## Execution failures and recovery

- First run **20260913-000746-taisp-t013a-feasibility** exited127 before tests:
  release-relative `.venv/bin/python` absent. Used the verified existing absolute
  project venv; no installation or environment change.
- Next workflow launch disconnected via SSH255 before tmux started. Verified no
  session/log, then resumed its existing script. This manual tmux resume omitted
  output redirection; the session ended with no retained log or final artifact.
  Its result is unknown and not counted as passing. Repeated the regression and
  same smoke fixture with explicit separate logs; the latter failure is above.
  The optional fixture therefore had retries/diagnosis, not multiple datasets or
  new experiment variants. No active process remains.
- A local handoff write initially hit PowerShell interpolation syntax `$stamp:`;
  corrected to `${stamp}` before writing. No source code or model impact.
- Local missing pycocotools is unchanged. Remote complete regression resolves the
  validation gap without modifying dependencies or suppressing an existing test.

## Review boundary

The requested first-order software plumbing is implemented and its mandatory
synthetic checks pass. Review the optional CUDA numerical limitation before
requiring bitwise real-model equivalence in future work. This task supplies no
evidence that learned initialization improves detection. Stop at T013-A:
**no T013-B, COCO training, real predictor/source/meta-training, new split/cohort,
gate, alpha, cap, spatial ISP or model update is started.**
