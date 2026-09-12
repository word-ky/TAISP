# T013-A pre-implementation plan

2026-09-13 00:05 +08. R015 task synchronized at 5e8afe6; T012 CLOSED.
Mode: bounded implementation on the accepted T009 full-hybrid, not model training.

API: add `taisp.training.initialization.adapt_from_initialization(image, isp,
detector_loss, clip_loss, phi0, *, steps=3, lr=.1, eps=1e-12)`.
Keep public `adapt_clip_radius` and `adapt_half_dose` signatures and defaults
unchanged. Reuse their private loop with an optional explicit initialization.
Reuse DifferentiableISP, AdaptResult, transfer_norm, and ParameterPredictor
without architectural changes or external donor code.

Training-only convention: u = stopgrad(g_det * ||g_clip||/(||g_det||+eps));
phi_next = phi - lr*u retains the graph to phi0. Compute inner gradients on a
detached current-phi probe, then recompute only the terminal ISP image on the
connected phi. This avoids retaining detector/CLIP graphs or requesting Hessians.
The outer state Jacobian is identity (FOMAML-style approximation), not an exact
meta-gradient. Explicit initialization is cloned, never stored in isp.phi.

Baseline: local Python D:/anaconda3/python.exe, OMP/OPENBLAS threads=1;
`python -m pytest tests/test_trust_radius.py tests/test_half_dose.py -q`:
5 passed, 3 skipped in 10.60s (two opt-in real-model tests, one pycocotools test).
Receipt: T013A/baseline_tests.txt. No code changes before this plan commit.

Increment 1: minimal shared-loop plumbing + focused parity (zero and nonzero),
reset/fallback, phi0/predictor head/trunk gradient, detached update/identity
Jacobian and model freeze tests. Preserve deployment detach and all diagnostics.
Increment 2 after focused green: deterministic CPU toy optimizer sanity script
using existing predictor, K=3/lr=.1/eps=1e-12; seed20260913, eight outer SGD steps
at lr=.2, fixed synthetic brightness target. Save every loss and before/after.
Final: all locally available tests; report skipped dependencies explicitly.
Optional one tiny real-model smoke only if time permits. No COCO training,
new cohort, gate/dose search, real predictor training, or T013-B.
