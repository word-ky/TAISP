# T011 predeclared matched-random falsification

2026-09-12, before any T011 random-selector AP. R013 research `26ff476`, pointer `ad06718`, accepts/closes T010. The authoritative R013/T011 is appended to coordination/CHATGPT_TO_CODEX.md. This task is entirely offline; no model, ISP or adaptation rerun, new cohort, threshold tuning or deployment change.

## Frozen candidate and exact randomization

Candidate: `support_confidence_low_50`, literal T010 cutoff `0.8499477751114789`; the saved T010 decision vector is authoritative, not a new ranking. Its SHA256 is `7a02e6d66336ef39fbd4b55c3a5aba564aacee56bd2d7405b62ce9d811a511f6`. Input preparation is research_log/T010/preparation, committed901560f before T010 AP; T010 finalreporta634c93. Keep exactly this candidate even if controls defeat it.

Predeclare 200 integer seeds: **2026091600 through 2026091799 inclusive**, in ascending order. For each seed, rank the 200 observations within each of seven conditions × five fixed T009 blocks by the SHA256 digest of UTF-8 text `TAISP-T011|{seed}|{image_id}|{case}`. Compare digest bytes ascending; ties use ascending image ID. Select exactly the number selected by the candidate in that stratum. Image ID and condition serve only immutable identity and the explicitly requested stratum match. No label, target output, prediction quality, AP or post-update quantity enters hashing or selection.

The same 7,000-element mask is used for source, FCOS and SSD. Preserve all 200 seed masks even if outcomes are identical. Save the seed list, matrix, grid, all 35 candidate stratum counts and hashes before any new official AP. Anchors are no-adapt, full-hybrid and the exact candidate (203 configurations total). Do not introduce another T010 candidate or feature combination.

## Reuse and bounded increments

| Responsibility | Existing implementation | T011 change / test |
| --- | --- | --- |
| Inputs / candidate | T010 preparation scores.jsonl, decisions.npz, grid.json, manifest.json | Read the existing candidate vector verbatim; hash equality and no reranking test. |
| Random selectors | New minimal hash ranking inside fixed strata | Determinism, seed variation, exact condition×block counts, zero/full strata and no outcome-field influence tests. |
| Composition and COCOeval | gating.compose_predictions; analyze_t010.evaluate_case/run; replication_ap/subset_ap | Reuse unchanged composition/evaluator; add bounded configuration batches for the 200-draw workload, preserving default T010 behavior. Test batching and complete configuration coverage. |
| Anchors | Frozen T009 metrics and T010 candidate panels | Reuse T009 0/100 metrics; compose/evaluate the fixed candidate and verify exact equality to T010. No model inference. |
| Randomization report | Existing T010 AP/clean/block conventions | New empirical distribution/tail and R013 rule summaries; hand-calculated ties, corrected-tail and both-target criteria fixtures. |

Fresh baseline before edits: remote14affectedoffline/AP/regressiontests passed2.31s in the exact T010release. All older model/ISP inference tests remain untouched and will not be rerun for an offline task.

Use CPU-only24workers, one BLAS/OpenMP/MKL thread each, configuration batches of25. Previous server inventory128logicalCPUs/~498GB available; no GPU required. This changes scheduling only. Keep all outcome panels/hashes; no partial-result-driven selector changes.

## Evaluation and descriptive costs

There are 203 ×3detectors×7conditions×6groups = **25,578 official AP/AP50/AP75 rows**: 25,200 random-control evaluations, 126 fixed-candidate evaluations, 252 reused no-adapt/full-hybrid anchors. All six corruption conditions, clean, aggregate and five fixed200-image blocks are retained. Each composition keeps native per-image prediction order and fresh record dictionaries so COCOeval cannot mutate reused anchors. Raw inputs plus masks and composition hashes reconstruct outputs without tens of GB of redundant prediction JSONs.

Coverage is exactly matched in all35strata. Effective phi3 = mask×saved hybrid phi3; effective nonzero coverage is reported separately because empty-support fallback can differ among selected images. Descriptive source-path latency estimate for every selector is S+mask×A, using saved source/support setup S and hybrid adaptation A. No new timing benchmark; target inference, IO and random-rank construction are excluded. Random controls are offline condition-stratified controls, not deployable selectors.

## Predeclared statistics and decision

For source/FCOS/SSD report all AP/AP50/AP75, macro deltas vs no-adapt/full hybrid, block macros/signs, clean AP, and candidate-minus-random distributions. Primary percentiles/tails are FCOS and SSD aggregate corruption macro AP. Define empirical percentile as **100×#(random < candidate)/200**, report exact tie counts, and one-sided corrected upper tail as **(1+#(random >= candidate))/201**. Report random mean, median, 5th and95th percentiles using NumPy linear quantile interpolation; these are control distributions, not AP confidence intervals or bootstraps.

Count how many of200random selectors pass the original R012 rule: clean selectedcoverage<=.5, cleanAPdelta>=−.10 onall3detectors, positive aggregate macro onbothindependenttargets and>=4/5positiveblocks foreach. Use the identical frozen candidate margin/coverage conventions asT010.

R013 selection information is supported only if candidate aggregate macro>0 onboth targets, candidate **strictly exceeds** random95thpercentile onboth, candidate **strictly beats** the within-block randommedian in>=4/5blocks foreach target, and candidatecleanAPdelta>=−.10 forall3detectors. Preserve all failures. If any condition fails, conclude T010 has not established need selection beyond generic thinning; do not validate this candidate on a new cohort or refine these scalar thresholds automatically. If allpass, it remains developmental afterthe42-wayT010search; awaitresearchreviewbefore literal-cutoff independentvalidation. No T012/meta/newGPUexperiment automatically.
