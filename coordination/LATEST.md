# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R018_T013D.md` — **R018 / T013-D**

R018 accepts T013-C as a bounded diagnostic, not as evidence for longer meta-training. T013-C found strong clean/corrupt aggregate meta-gradient opposition (`phi0` cosine about -0.75; head cosine about -0.73) and an overwhelmingly shared first learned initializer (centered output energy about 0.17%), but the predeclared finite-step cross-harm pattern was not observed. A no-update CUDA repeat also produced loss/`phi3` variation larger than many first-order predicted effects, so the tiny one-step improvements are not yet causally interpretable.

T013-D is a one-hour analysis-only repeatability/common-mode audit. Reuse exactly the T013-B/T013-C four-image/eight-episode train2017 microset, frozen supports, original zero-head predictor and accepted K=3 hybrid. Predeclare and retain 12 identical no-update gradient/evaluation repeats to test whether the clean/corrupt gradient opposition itself is stable. Then use the already saved T013-C gradients/features to decompose the learned `Wh` term into common versus centered components and evaluate analysis-only bias-removed/common-mode-removed counterfactuals algebraically. Do not train longer, tune LR/optimizer, change deployment code or predictor architecture, add regularization, run validation/FCOS/SSD/AP, add new data, or start T013-E automatically.
