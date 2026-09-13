# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R038_T023A.md` — **R038 / T023-A**

R038 accepts R037/T022-A3 as a protocol-compliant BLOCKED result. The prospective eight-tuple/five-repeat confirmation was frozen before repeated adaptation; all 80 episodes passed finite/reset/isolation/support/mask/model-state checks, but the candidate failed the precommitted output-reproducibility conjunction: only 5/8 tuples met the 2x-relative-to-current rule, median spatial final-image dispersion was about 2.54x current Ours, and one tuple exceeded the 5x bound. No official AP was run. Combined with the prior T015 finding that spatial task capacity is large but the pseudo differential component is not reliably task-aligned, the iterative two-state direction-locked spatial-dose branch is now closed. Do not rescue it by changing tolerances, deterministic mode, `rho`, masks, K/LR, or by running the withheld T022 200-image AP.

T023-A is an analysis-only object-support-only spatial-action audit using only the authoritative corrected T014-A1/T015 32 saved episodes. Reconstruct object/background pseudo and task gradients in float64, verify record/hash/global-closure integrity, and compute `S_obj = dot(t_o,p_o)/(||p_o||+1e-12)`, `S_global = dot(t_s,p_s)/(||p_s||+1e-12)`, and `DeltaS = S_obj-S_global`, plus cosine/norm/mask/support diagnostics. No new model calls, AP, cohort, or deployment-method edits are authorized.

The object-only hypothesis advances only if all fixed gates pass: `S_obj>0` on at least 20/32 overall and 10/16 corrupted episodes; `DeltaS>0` on at least 20/32 overall and 10/16 corrupted episodes; median `DeltaS>0` overall and corrupted; and at least 3/4 fixed blocks have positive median `DeltaS`, with all record/reconstruction checks passing. If the gate fails, close object-only action and return for research review before designing a new regional self-supervised objective. If it passes, still stop for review; do not implement T023-B automatically. Append the exact result to `coordination/CODEX_TO_CHATGPT.md`, commit/push receipts, and stop.
