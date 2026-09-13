# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R036_T022A2.md` — **R036 / T022-A2**

R036 accepts T022-A1 as protocol-compliant and keeps the spatial-dose hypothesis scientifically open. The exact R035 CLIP-wrapper repair is fully validated: all 28 frozen parity records and CLIP isolation/hash checks pass within the original numerical tolerances. The subsequent zero-AP K=3 runtime smoke nevertheless stopped on the precommitted exact-repeat edge diagnostic: empty-mask/nonempty-support repeated episodes diverged at final state (`1.458e-4` max abs, `4.52e-3` relative L2), full-mask repeats differed only at `2.79e-9` max abs, and empty-support remained exactly repeatable. No AP was run.

T022-A2 is a bounded analysis-only repeatability attribution audit. Do not modify current Ours, the spatial-dose runtime/formula, ISP, detector/CLIP, `rho`, masks/support rules, K/LR, dtype, tolerances or cohort. On the same saved image/support/state, separately measure repeated detector/CLIP image cotangents, repeated regional JVP/reductions with a fixed cotangent, repeated pure `dose_step` with fixed vectors, and five K=3 repeats of current Ours versus spatial dose using the real support-derived mask plus the frozen empty/full/empty-support controls. Run one separate `torch.use_deterministic_algorithms(True)` diagnostic only to identify whether PyTorch reports an upstream nondeterministic operator; do not use deterministic mode for performance evaluation.

The goal is to locate the first non-repeatable quantity and determine whether the edge divergence is inherited upstream CUDA backward variability, variability introduced by the spatial JVP/dose implementation, or trajectory amplification specific to degenerate masks/real spatial runtime. Do not define a new tolerance after observing outcomes. Zero official AP evaluations are authorized in T022-A2. Append exact receipts to `coordination/CODEX_TO_CHATGPT.md`, commit/push, and stop for research review.
