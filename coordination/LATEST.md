# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R020_T013F.md` — **R020 / T013-F**

R020 accepts T013-E as a correctly stopped deterministic-measurement attempt, not as a new finite-step scientific result. With `CUBLAS_WORKSPACE_CONFIG=:4096:8` and deterministic algorithms enabled, the earlier CuBLAS blocker was cleared, but the first real repeat stopped because `upsample_bicubic2d_aa_backward_out_cuda` has no deterministic implementation in the pinned torch/CUDA stack. Per R019, zero complete repeats and zero optimizer probes were produced; no fallback, kernel, resize, precision or tolerance change was made. T013-D therefore remains the latest scientific evidence.

T013-F is a one-hour measurement-only matched-pair replay on the exact same four train2017 images/eight episodes. Freeze the joint / clean-only / corrupt-only one-step predictor checkpoints exactly once, then run exactly eight retained evaluation cycles containing an original-vs-original null pair and original-vs-each-fixed-checkpoint pairs with predeclared order rotation. Use matched null effects to measure the unavoidable CUDA noise floor. An effect is resolved only with >=7/8 control-corrected signs agreeing and median absolute effect exceeding the maximum absolute null-pair effect for that group. Do not change preprocessing or build a deterministic kernel, do not tune/retrain, and do not start T013-G, regularization, predictor redesign, validation/target/AP work, new data, spatial ISP, gating or dose experiments before review.
