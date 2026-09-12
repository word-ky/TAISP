# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R016_T013B.md` — **R016 / T013-B**

R016 accepts T013-A as software/gradient feasibility: the training-only explicit-initialization path preserves the accepted hybrid forward rule, exposes finite outer gradients to `phi0`/`ParameterPredictor`, keeps detector/CLIP frozen, and passes the required synthetic/reset/isolation checks. The optional strict CUDA bitwise parity failure remains recorded and is not converted into a pass; same-path CUDA repeat drift is of the same order, so it is treated as a numerical limitation rather than a blocker.

T013-B is one bounded ~1-hour package. First audit the current FOMAML-style stop-gradient meta-gradient against an exact tiny unroll and finite differences on 12 deterministic synthetic episodes. Proceed to a tiny real source-labeled meta-step smoke only if the predeclared fidelity gate passes. The real smoke may use only an already-available fixed 4-image COCO train2017 microset, never COCO-val/T002–T012 cohorts; optimize only the existing `ParameterPredictor` for exactly 3 SGD outer steps while the accepted K=3 hybrid, Faster R-CNN, CLIP, ISP, prompts, pseudo support, and deployment API remain frozen. If train2017 is unavailable, stop rather than substituting validation data. Do not start longer meta-training, target evaluation, new cohorts, spatial ISP, gating/dose work, or T013-C automatically.
