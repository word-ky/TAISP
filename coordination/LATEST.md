# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R019_T013E.md` — **R019 / T013-E**

R019 accepts T013-D as a bounded repeatability/common-mode diagnostic. Across all 12 frozen no-update CUDA repeats, clean-vs-corrupt aggregate source meta-gradients remain opposed in both explicit `phi0` and predictor-head space, so the local gradient conflict is real on this microset. However most earlier one-step effects remain comparable to nondeterministic no-update variation. Separately, `Wh` itself is about 93.9% common-component energy, so deleting the explicit head bias alone is not a justified conditional-initialization fix.

T013-E is a one-hour measurement-only deterministic replay. Reuse exactly the same four train2017 images/eight episodes, frozen supports, original zero-head predictor, accepted K=3 hybrid and source outer loss. Launch Python with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before torch import, enable deterministic algorithms, and first require three bitwise-identical no-update repeats. Only if that gate passes, replay exactly the fixed joint/clean-only/corrupt-only one-step SGD probes at `1e-3`. Use the predeclared cross-harm rule to decide whether source-objective conflict is functionally active. Do not tune, train longer, change architecture/objective/deployment code, remove bias, add regularization, run validation/FCOS/SSD/AP, add data, or start T013-F automatically.