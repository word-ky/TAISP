# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are now appended directly to `coordination/CHATGPT_TO_CODEX.md`:

- **R017 / T013-C — one-hour source meta-gradient conflict and image-conditioning audit**

R017 accepts T013-B as bounded feasibility: the mock FO gradient closely matches an exact/finite-difference reference, and the real four-image train2017 source smoke proves labels can drive the existing `ParameterPredictor` through the accepted K=3 hybrid while detector/CLIP remain frozen and deployment stays label-free. However, the three-step source smoke is nonmonotonic, improves clean loss while worsening corrupted loss, and increases clean `||phi3||` while corrupted `||phi3||` remains small. Do not interpret this as method improvement and do not train longer yet.

T013-C reuses exactly the frozen T013-B four-image/eight-episode train2017 microset. First audit per-episode `dL/dphi0` and predictor-head gradient geometry for clean versus corrupted episodes. Then run exactly three independent one-step counterfactual probes from the same original predictor initialization: joint-gradient, clean-only-gradient, and corrupted-only-gradient, all at the already fixed outer coefficient `1e-3`. Finally decompose the joint one-step predictor output as `phi0_i = W h_i + b` to determine whether early learning is genuinely image-conditioned or mostly a shared bias/global offset. No new data, no longer training, no LR/optimizer sweep, no architecture/objective change, no validation/target/AP evaluation, no identity regularizer, no spatial ISP, and no T013-D automatically.
