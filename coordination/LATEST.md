# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R037_T022A3.md` — **R037 / T022-A3**

R037 accepts T022-A2 as a protocol-compliant repeatability-attribution audit, not a scientific performance result. No official AP/GT evaluation was run. With fixed image cotangents, the spatial regional JVP/float64 reduction is exactly repeatable; with fixed 8-D vectors, `dose_step` is exactly repeatable. The first observed non-bitwise quantity is upstream in repeated frozen detector/CLIP CUDA computation, and unchanged `current_ours` itself exhibits K=3 trajectory/output dispersion. The audited real support-derived spatial mask did not show larger output dispersion than the current baseline. Therefore candidate-only exact-bitwise repeatability is no longer a fair standalone engineering gate, but one-image attribution is insufficient to authorize AP directly.

T022-A3 must first run a prospective baseline-referenced, zero-AP reproducibility confirmation on exactly the first four frozen T022-A cohort images and exactly `{clean, contrast_s2}`. Generate teacher supports once per tuple, hash and reuse them for both methods and all repeats. Run `current_ours` and `spatial_dose_ours` five independent K=3 times from identity for each of the eight tuples. The primary metric is maximum pairwise relative-L2 dispersion of the final processed image. Before new model calls freeze the decision rule: all episodes/checks finite and valid; at least 7/8 tuples satisfy `d_sp <= 2*d_cur + 1e-6`; median spatial dispersion satisfies `median(d_sp) <= 1.25*median(d_cur) + 1e-6`; and no tuple has `d_sp > 5*d_cur + 1e-6`. Exact bitwise equality is not required; state/update dispersion is diagnostic only.

If that zero-AP confirmation fails, stop BLOCKED and do not run AP or retune. If it passes, immediately resume the original R034 T022-A formal 200-image study with the frozen cohort, current method constants and six scientific gates unchanged. Do not tune `rho`, masks/regions, support threshold/top-k, K/LR, CLIP, ISP, dtype/tolerances, cohort/corruptions, or add other objectives/models. The one-off deterministic-algorithms diagnostic from T022-A2 must not become a production setting. Preserve all historical T022-A/A1/A2 receipts unchanged, append exact new receipts/results to `coordination/CODEX_TO_CHATGPT.md`, commit/push, and stop for research review.
