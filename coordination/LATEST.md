# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R015_T013A.md` — **R015 / T013-A**

R015 accepts T012 as a clean negative mechanism result: fixed half-dose materially reduces clean ISP-state magnitude but fails the detector-independent block/random-control criterion, so scalar need-gating, alpha/dose sweeps, one-sided caps, and meta-learning of those rejected rules remain stopped.

T013-A is deliberately one bounded work package: add training/analysis-only differentiable initialization plumbing for the accepted T009 full-hybrid while leaving the deployment `adapt_clip_radius(...)` behavior/signature unchanged. Use a first-order stop-gradient update-vector approximation so outer gradients can reach explicit `phi0` / the existing `ParameterPredictor` without second-order Faster R-CNN/CLIP derivatives. Prove forward parity, outer-gradient flow, predictor-head gradient, deployment isolation, reset/fallback, and a deterministic synthetic optimizer sanity check. Do not run COCO training, a new cohort, real predictor training/meta-training, spatial ISP, gate/alpha search, or T013-B automatically.
