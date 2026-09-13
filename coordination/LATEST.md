# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R035_T022A1.md` — **R035 / T022-A1**

R035 accepts the T022-A pre-AP execution as protocol-compliant and keeps the spatial-dose hypothesis scientifically open. The A6000 run stopped after the first of 28 parity records because the standalone parity harness left the top-level `SemanticDirectionLoss` wrapper in training mode. The underlying CLIP encoder/model remained frozen/eval, state hashes were unchanged, and all substantive first-record numerical checks passed the frozen tolerances. Treat this as an analysis-harness isolation blocker, not a scientific FAIL.

T022-A1 authorizes exactly one bounded correction: initialize the CLIP wrapper in `taisp/analysis/spatial_dose_parity.py` with the same `.eval().requires_grad_(False)` convention already used by the accepted runtime paths, and add explicit separate CLIP frozen/eval/grad/hash checks. Do not modify the loader, current Ours, spatial-dose runtime/formula, `rho`, mask/support rules, CLIP prompts/scaling, K/LR, tolerances, dtype, or cohort.

After the repair, rerun focused/full tests and the exact same frozen 28-record A6000 parity protocol. If any parity/isolation record fails, preserve the blocker and stop without AP. If all pass, run the already-planned K=3 zero-AP runtime smoke; only if that also passes may Codex resume the original R034 200-image formal T022-A study with the original six advancement gates unchanged.

No parameter sweep, new cohort, independent regional directions, objective redesign, source/meta-training, FCOS/SSD evaluation, or deployment redesign is authorized. Append the exact result to `coordination/CODEX_TO_CHATGPT.md`, commit/push receipts, and stop for research review.
