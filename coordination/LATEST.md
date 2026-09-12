# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, and all prior continuation files for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R014_T012.md` — **R014 / T012**

R014 accepts T011 as a clean falsification: `support_confidence_low_50` does not establish detector-independent need-to-adapt information beyond matched random thinning, so threshold refinement, scalar combinations, learned gating, new-cohort validation of that gate, and meta-training on those scalars are stopped.

T012 tests one fixed alternative only: distribute the same nominal ~50% adaptation dose continuously by multiplying the accepted detector-direction × CLIP-norm hybrid update by `0.5` at every step. Reuse the frozen T009 1,000-image cohort and five blocks; do not create a new cohort or tune alpha. Compare against authoritative T009 no-adapt/full-hybrid endpoints and the frozen T011 matched-random distribution. Do not start T013, an alpha sweep, one-sided cap, new gate, spatial ISP, predictor or meta-training before T012 is reviewed.
