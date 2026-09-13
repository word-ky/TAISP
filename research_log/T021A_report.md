# T021-A - close_fixed_horizontal_flip_support_filter; NEEDS_REVIEW

R033/a5bae26; pre-outcome plan0207885; experimental code daa79e5.
Formal `20260913-222407-taisp-t021a-source200-consensus`; smoke `20260913-222054-taisp-t021a-runtime-smoke`; exact commands/environment in raw receipts.
200newtrain2017 images/four50blocks,1036prior-source/debug IDs/all5000val excluded.
CohortSHA0fdb6a815d380104542c324ba2146944b47232dae22a416fa12d5f72a5d29725; selectionseed20260922, unchanged modelseed20260912.
No selection by predictions/retention. No source labels or fitting enter adaptation.

Frozen rule: original and horizontal flip teacher,score>=.50 each view,sameclass IoU>=.60,
one-to-one greedy geometric confidence descending, original/flip index ties,top20after
matching. Original boxes/classes/**scores** retained; geometric mean is ranking only.
No box/score averaging or rematching. Empty consensus means exact identity/no fallback.
Unchanged current-Ours pseudo loss,global8D ISP/identity,K3/LR.1,CLIP magnitude rule,
frozen weights/prompts/teacher/corruptions. Final inference is a single enhanced view.

Corruption macro AP: {"no_adapt": 50.331294197165455, "current_ours": 50.15130896713511, "flip_consensus_ours": 50.12094182205353}.
Candidate-current **-0.030367145 AP**, candidate-raw **-0.210352375 AP**.
Positive blocks **1/4**, positive conditions **4/6**.
Clean-current **+0.263891538 AP**, clean-raw **-0.149335804 AP**.
Macro AP50/AP75 deltas versus current: {"AP50": 0.0446170629117546, "AP75": 0.02264903794497286}.
Frozen decision passed: **False**. Retention/AP50/AP75 never replace AP gates.

## Frozen criteria

| Criterion | Passed |
| --- | --- |
| corruption_macro_delta_at_least_point10 | False |
| at_least3_positive_block_macros | False |
| at_least4_positive_corruption_conditions | True |
| corruption_macro_above_no_adapt | False |
| clean_delta_at_least_minus_point10 | True |
| no_support_matching_isolation_or_contamination_blocker | True |

## Fixed block macro deltas

| Block | AP-current | AP-raw | AP50-current | AP75-current |
| --- | --- | --- | --- | --- |
| block0 | 0.0975245813 | -0.000325636012 | 0.092397898 | -0.21491122 |
| block1 | -0.0180727143 | -0.00768123096 | 0.300767471 | -0.25817891 |
| block2 | -0.0847939799 | -0.0933518395 | -0.122475205 | 0.0902558466 |
| block3 | -0.242832941 | -1.23017981 | 0.0355337139 | -0.231657885 |

## Consensus support retention

| Group | N | Mean original eligible | Mean current top20 | Mean matched | Mean retained top20 | Mean retention defined | Median retention defined | Undefined retention | Zero count | Zero fraction | Extra flip+match s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 1400 | 8.75071429 | 7.97714286 | 7.23857143 | 6.90857143 | 0.847174612 | 0.875 | 5 | 7 | 0.005 | 0.0294333422 |
| clean | 200 | 9.42 | 8.525 | 7.8 | 7.395 | 0.84878297 | 0.872282609 | 0 | 0 | 0 | 0.0293978395 |
| corrupted | 1200 | 8.63916667 | 7.88583333 | 7.145 | 6.8275 | 0.846905431 | 0.875 | 5 | 7 | 0.00583333333 | 0.0294392593 |
| block0 | 350 | 9.61714286 | 8.81428571 | 7.86285714 | 7.52 | 0.834970566 | 0.857142857 | 1 | 2 | 0.00571428571 | 0.0295669733 |
| block1 | 350 | 7.64857143 | 6.70857143 | 6.43714286 | 5.90285714 | 0.846780031 | 0.888888889 | 2 | 2 | 0.00571428571 | 0.0297635248 |
| block2 | 350 | 8.11142857 | 7.6 | 6.68857143 | 6.54857143 | 0.858444472 | 0.888888889 | 0 | 0 | 0 | 0.0293690344 |
| block3 | 350 | 9.62571429 | 8.78571429 | 7.96571429 | 7.66285714 | 0.848473678 | 0.875 | 2 | 3 | 0.00857142857 | 0.0290338361 |
| clean_s0 | 200 | 9.42 | 8.525 | 7.8 | 7.395 | 0.84878297 | 0.872282609 | 0 | 0 | 0 | 0.0293978395 |
| color_cast_s1 | 200 | 9.205 | 8.265 | 7.56 | 7.17 | 0.847223194 | 0.875 | 0 | 0 | 0 | 0.0294337263 |
| color_cast_s2 | 200 | 9.025 | 8.11 | 7.39 | 7.03 | 0.845503041 | 0.868115942 | 0 | 0 | 0 | 0.0294121544 |
| contrast_s1 | 200 | 8.86 | 8.06 | 7.24 | 6.955 | 0.835853174 | 0.846153846 | 0 | 1 | 0.005 | 0.0294124847 |
| contrast_s2 | 200 | 6.88 | 6.46 | 5.86 | 5.67 | 0.867934355 | 1 | 5 | 5 | 0.025 | 0.0293933064 |
| gamma_s1 | 200 | 9.155 | 8.335 | 7.575 | 7.18 | 0.839412249 | 0.857142857 | 0 | 0 | 0 | 0.029556521 |
| gamma_s2 | 200 | 8.71 | 8.085 | 7.245 | 6.96 | 0.846032294 | 0.875 | 0 | 1 | 0.005 | 0.0294273628 |

## Adaptation diagnostics

| Method/group | N | Mean phi | Median phi | Update fraction | Mean support | Median support | Mean adapt s | Median adapt s | Mean teacher-inclusive s | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_ours/clean | 200 | 0.0384209952 | 0.0309232464 | 1 | 8.525 | 6 | 0.303178218 | 0.308434777 | 0.331832789 | 3.05834723 |
| current_ours/corrupted | 1200 | 0.0296465212 | 0.0246728454 | 0.995833333 | 7.88583333 | 6 | 0.302770695 | 0.308912815 | 0.332613133 | 3.06434488 |
| flip_consensus_ours/clean | 200 | 0.0384710176 | 0.0308409147 | 1 | 7.395 | 5.5 | 0.302365645 | 0.308311822 | 0.360418056 | 3.05820179 |
| flip_consensus_ours/corrupted | 1200 | 0.0302778448 | 0.0248258989 | 0.994166667 | 6.8275 | 5 | 0.301494675 | 0.308374672 | 0.360776373 | 3.0618577 |

## Validation and artifacts

Baseline12passed2skipped1.77s; increment1 11passed2skipped2.42s.
Focused: 13 passed, 2 skipped, 2 warnings in 2.50s. Full regression: 175 passed, 10 skipped, 4 warnings in 7.52s.
Two-image CUDA smoke:28teacherforwards/28adaptiveK3episodes/zeroAP; all support checks pass.
Formal:2800teacherforwards/2800adaptiveepisodes/21predictions/105officialCOCOevaluations,
1128.431519s, A6000 CUDA models/gradients/adaptation; CPU official AP.
All1400consensus receipt checks and9672
retained supports preserve original boxes/classes/scores, ranking/ties/one-to-one/IoU.
All baseline supports reproduce original top20. Frozen state/hash/gradNone/ISP/support
checks pass. Empty candidate/current episodes:7/5.
All step histories retained. No native loss components or transported gradients appear.

Adaptation latency includes four diagnostic evaluations/threeupdates, excludes final
prediction/state/AP. Candidate teacher-inclusive includes original plus extra flip/matching;
current includes original teacher only. Extra overhead is separately reported above.
Peak memory includes verification copies. Support retention divides retained top20 by
original eligible pre-top20 count; a zero denominator is null and explicitly counted.

66rawfiles,59995881bytes,all SHA256 recorded inT021A/artifact_manifest.json.
Full AP/AP50/AP75/condition/block/step/support tables:T021A/complete_tables.md.
Receipt audit:T021A/receipt_audit.json. No thresholds/augmentation/topk/fallback tuning,
extra cohort,FCOS/SSD/val,spatial/meta/predictor/gate/dose or closed-branch experiments.
Stop NEEDS_REVIEW: pass means developmental candidate only; fail closes the fixed
horizontal-flip support filter under R033. Current Ours remains authoritative pending review.

Actual remote release:20260913-222014-taisp-t021a-consensus; all17protected/authorized module hashes verified directly on A6000 and saved inT021A/remote_code_hashes.json. Formal started22:24:12 and finished22:43:09+08,exit0. No development candidate; this fixed branch is closed under R033. Both current Ours and consensus are below raw corruption AP on this cohort.
