# T018-A — full native pseudo-target loss: NEEDS_REVIEW

**PROMISING: all six developmental criteria pass; confirmation still requires research authorization.**

The six-corruption macro AP is 54.108149 for nativePT_ours,
53.891408 for current_ours and 53.914903 for no_adapt.
Candidate minus current is +0.216741 AP points;
candidate minus no-adapt is +0.193247.
Positive candidate-minus-current macro deltas occur in 3/4 fixed blocks,
and 4/6 corruption conditions improve. Clean AP delta is +0.619965.
These are developmental source-only results, with no independent confirmation or cross-detector claim.

## Provenance and implementation

- Research R029 / pointer 53f8662. Pre-outcome plan/cohort/config commit cd2ed29; implementation 7c43f1f.
- Release 20260913-153309-taisp-t018a-nativept. Formal run `20260913-153604-taisp-t018a-source100`; smoke `20260913-153419-taisp-t018a-runtime-smoke`.
- Exactly 100 new COCO train2017 images, four precommitted blocks of 25, and seven unchanged conditions.
  Cohort SHA256 a01dfb1d40a6daceddccc1b7aa7f3f2e74871fd4a511d8d6c9acf8e50c2c111f.
  Zero overlap with 36 prior source IDs or all 5000 validation IDs. Selection and JPEG hashes are preserved.
  The hash-ordered selection pool contained 13,762 available eligible images after exclusions, rather than
  the entire train2017 split. Cohort preparation used the existing non-crowd positive-box annotation eligibility;
  this metadata filtering is disclosed separately from the label-free adaptation path.
- Original-condition frozen teacher predictions are computed once, filtered with existing score>=0.5/top20 ordering.
  Native targets contain detached boxes and integer labels only. All four native losses have unit weight.
- Existing global 8-D ISP and adapt_clip_radius are unchanged: identity initialization, K=3, LR=0.1,
  epsilon=1e-12 and current-phi CLIP gradient magnitude. CLIP direction is discarded.
  RPN/ROI sampling uses the precommitted seed 20260912 for each loss call. Four diagnostic evaluations apply three updates.
- Candidate code is isolated in analysis modules. Current Ours, teacher filtering, CLIP prompts/preprocessing,
  corruption definitions, detector and ISP bounds are unchanged. No deployed method replacement occurred.
- The adaptation loader sees image metadata/JPEGs only. Train annotations enter official COCO evaluation after
  every prediction file is written; annotation targets never enter adaptation.

## Validation and execution

- Baseline: 6 passed, 2 skipped in 1.40 s. Focused: 10 passed, 2 skipped in 1.70 s.
- Full regression: 148 passed, 10 skipped in 6.94 s. Optional skips are retained; these are not 148 real-model tests.
- CUDA smoke: two images, seven conditions, 28 adaptive episodes (14 native), K=3, four finite native
  component histories per episode, identity initialization and all frozen-state checks passed. No AP was computed.
- Formal: 100 images, 700 shared teacher forwards, 1400 adaptive episodes, 21 prediction files and
  105 official aggregate/block evaluations. Completion elapsed 605.118242 s; exit status is preserved in train.log.
  Started 2026-09-13 15:36:09 +08:00; finished 15:46:22 +08:00, exit 0.
- A6000 cuda:0, Python 3.12.12, torch 2.4.0+cu121, float32 model/ISP, one CPU thread.
  Existing NVML warning did not prevent CUDA execution. Package versions, weight/state hashes and exact commands
  are in environment.json, meta.json and run.sh. Model work used GPU; official COCO aggregation used CPU.
- Source state, requires_grad flags, eval modes and gradient absence are checked after every adaptive episode.
  Source and CLIP state hashes match before/after the whole study; every isolation check passes.
  ISP-owned phi remains zero with no accumulated gradient. Only the functional episodic phi is updated.
  Empty-support exact-zero phi and existing identity-ISP output semantics passed the unit test;
  there were no empty-support episodes in this formal cohort. All 700 method pairs have identical teacher supports,
  and all saved phi, detector-gradient and CLIP-gradient coordinates are finite.

## Aggregate official AP

| Condition | No adapt | Current | NativePT | Native-current |
| --- | --- | --- | --- | --- |
| gamma_s1 | 56.5707864 | 57.0963555 | 57.5354042 | 0.439048692 |
| gamma_s2 | 54.7445706 | 54.7236828 | 54.9007194 | 0.177036614 |
| contrast_s1 | 55.7381666 | 54.6155321 | 55.0809362 | 0.465404115 |
| contrast_s2 | 45.543434 | 45.9623675 | 45.8846745 | -0.0776929989 |
| color_cast_s1 | 56.5290615 | 56.9136209 | 56.8102264 | -0.103394513 |
| color_cast_s2 | 54.3633961 | 54.0368893 | 54.4369345 | 0.400045186 |
| clean_s0 | 59.7798201 | 59.7445635 | 60.3645281 | 0.61996451 |

## Fixed block corruption macro AP

| Block | No adapt | Current | NativePT | Delta |
| --- | --- | --- | --- | --- |
| block0 | 53.4499538 | 53.3878248 | 53.8249267 | 0.437101816 |
| block1 | 55.7934026 | 55.912447 | 55.9938211 | 0.0813741261 |
| block2 | 54.8894352 | 54.8368715 | 54.8633123 | 0.0264407625 |
| block3 | 57.1198186 | 57.0462538 | 56.8403212 | -0.205932539 |

## Frozen advancement rule

| Criterion | Passed |
| --- | --- |
| corruption_macro_above_current | True |
| corruption_macro_delta_at_least_point10 | True |
| at_least3_positive_block_macros | True |
| at_least4_positive_corruption_conditions | True |
| corruption_macro_above_no_adapt | True |
| clean_delta_at_least_minus_point10 | True |

## Diagnostics and limits

- current_ours/clean: phi norm mean/median 0.0386662962/0.0320180971; update fraction 1; support mean/median 9.75/8; adaptation mean/median 0.301226/0.301903 s.
- current_ours/corrupted: phi norm mean/median 0.0309632282/0.0269624116; update fraction 1; support mean/median 9.13833/7; adaptation mean/median 0.301075/0.302535 s.
- nativePT_ours/clean: phi norm mean/median 0.033542294/0.0301730707; update fraction 1; support mean/median 9.75/8; adaptation mean/median 0.368588/0.369076 s.
- nativePT_ours/corrupted: phi norm mean/median 0.0286475098/0.0257296767; update fraction 1; support mean/median 9.13833/7; adaptation mean/median 0.367953/0.368851 s.

Per-step component losses, paired prediction counts, AP/AP50/AP75 for all conditions/blocks and macro averages
are in [complete_tables.md](T018A/complete_tables.md). All negative conditions and blocks are retained.
Timing includes terminal diagnostics and excludes final enhanced-image inference/state verification/evaluation;
the adapt+teacher column adds the shared original teacher forward. No end-to-end throughput claim is made.
Peak allocated memory includes the frozen state copy used for verification.

The 100-image source set is developmental, not a publication or generalization claim. Diagnostics do not establish
a causal explanation of AP changes. No threshold, loss weight, step count, learning rate or cohort was tuned.
Both methods update every clean image; the smaller nativePT phi norms do not establish clean-image selectivity.
T017-A remains numerically blocked and unchanged; no component-attribution conclusion follows from T018-A.

## Artifacts and next action

All 62 raw files (30,654,090 bytes), including smoke and formal receipts, are hashed in [artifact_manifest.json](T018A/artifact_manifest.json). Raw root: `research_log/remote_runs/20260913-153604-taisp-t018a-source100/`.

Stop for research review. No active experiment remains. Do not launch confirmatory, FCOS/SSD, COCO-val, spatial ISP, component-only variants, sweeps or deployment changes without a new queued task.
