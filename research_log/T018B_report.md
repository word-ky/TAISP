# T018-B — NOT SOURCE-CONFIRMED; NEEDS_REVIEW

The fixed nativePT candidate achieves six-corruption macro AP 44.345497746,
versus current Ours 44.683726195 and no-adapt 44.479799444.
Native-current is -0.338228449 AP; native-no-adapt is -0.134301698 AP.
Positive blocks: 0/5. Positive corruption conditions: 0/6.
Clean AP delta: +0.291859560. R030 rule passed: **False**.
Macro AP50 delta: -0.354632226; macro AP75 delta: -0.427867441.
AP75 nonnegative diagnostic: False; it is not a confirmation gate.

## Protocol and provenance

R030 / pointer7902e3a. Pre-outcome plan/config/cohort958d898; experiment code d8ff14f.
Release20260913-161116-taisp-t018b-source500; formal `20260913-161818-taisp-t018b-source500`; smoke `20260913-161225-taisp-t018b-runtime-smoke`.
Cohort SHA256 e7a771126ae2fee9844dde456648b50f4c01ebcdc404318c21f9a33260da7b17.
500 new train2017 images, five precommitted consecutive100-image blocks, zero overlap with
136 prior-source IDs or all5000val IDs. Same disclosed readable/non-crowd positive-box/area eligibility
as T018-A: 13,662 available eligible images after exclusions. Selection uses hash seed20260919.

NativePseudoTargetLoss, adapt_clip_radius, current Ours, source loading/teacher selection,
CLIP, oracle, ISP and corruptions are unchanged from accepted T018-A; 11 protected modules
are pinned in T018B_method_pins.json. The runner only extends cohort count and result summary.
Same score>=.5/top20 detached teacher boxes/int labels, unit four-native-loss sum, global8D
identity phi, K3/LR.1/EPS1e-12, CLIP norm transfer and seeded native sampling20260912.
No detector/CLIP optimizer or parameter update. Ground-truth eligibility metadata is used in
cohort preparation; adaptation receives JPEGs and detached teacher targets only. Official
annotations are loaded after all prediction files are saved, for evaluation only.

## Tests and execution

- Unchanged baseline10passed2skipped1.74s; focused12passed2skipped1.77s.
- Full regression150passed10skipped7.06s. Optional skips retained; these are not150real-model tests.
- CUDAK3 smoke:2images,28adaptiveepisodes,14teacherforwards,all isolation checks passed,0AP,exit0.
- Formal:500images,3500teacherforwards,7000adaptiveepisodes,21predictionfiles,126official evaluations.
  Collection/evaluation elapsed 2891.522734s. Exact timestamps/exit status in train.log.
- A6000cuda:0,float32 models/ISP,oneCPUthread; source/CLIP frozen with before-after state hashes and
  per-episode source/ISP/support isolation checks. The run completed with all isolation checks passing.
  Existing NVML warning did not prevent CUDA execution. Environment/package/hash/command receipts retained.
- Empty-support identity behavior is covered by existing tests; actual empty fraction is reported in tables.
  Adaptation includes four diagnostic evaluations with exactly three updates; its latency excludes
  final prediction, state verification and evaluation. Teacher-inclusive adds original teacher forward.

## Aggregate results

| Condition | No-adapt AP | Current AP | Native AP | AP delta | AP50 delta | AP75 delta |
| --- | --- | --- | --- | --- | --- | --- |
| gamma_s1 | 47.846829 | 47.748073 | 47.6550685 | -0.0930045238 | 0.124860288 | -0.357620781 |
| gamma_s2 | 44.8015693 | 45.1176088 | 44.781196 | -0.336412793 | -0.58344862 | -0.237215019 |
| contrast_s1 | 46.2021226 | 46.6419296 | 45.4826868 | -1.15924281 | -1.16131822 | -1.60912452 |
| contrast_s2 | 36.6888794 | 36.9787041 | 36.8250642 | -0.153639942 | -0.282717386 | -0.212101435 |
| color_cast_s1 | 47.0295274 | 47.2092334 | 47.0124075 | -0.196825883 | -0.0132984973 | -0.044626635 |
| color_cast_s2 | 44.3098691 | 44.4068082 | 44.3165635 | -0.090244742 | -0.211870927 | -0.106516255 |
| clean_s0 | 50.1588525 | 49.6133386 | 49.9051981 | 0.29185956 | -0.120648437 | 2.10154617 |

## Fixed block macro AP deltas

| Block | Delta | AP50 delta | AP75 delta |
| --- | --- | --- | --- |
| block0 | -0.620467499 | -0.46896686 | -0.798597681 |
| block1 | -0.238126574 | -0.144392382 | -0.448413648 |
| block2 | -0.0352389743 | -0.0366880193 | -0.0845632672 |
| block3 | -0.178669267 | -0.275498278 | 0.0269483811 |
| block4 | -0.113510921 | 0.127720594 | -0.0354386454 |

## All six R030 criteria

| Criterion | Passed |
| --- | --- |
| macro_AP_delta_at_least_point10 | False |
| at_least4_positive_block_macros | False |
| at_least4_positive_corruption_conditions | False |
| corruption_macro_above_no_adapt | False |
| clean_delta_at_least_minus_point10 | True |
| no_isolation_or_reproducibility_blocker | True |

## Diagnostic summary and limitations

- current_ours/clean: phi norm mean/median 0.0380710889/0.0311116287; update fraction 0.998; support mean/median 8.966/7; adapt mean/median 0.302323465/0.305183042s.
- current_ours/corrupted: phi norm mean/median 0.0308243236/0.0256403983; update fraction 0.995333333; support mean/median 8.319/6; adapt mean/median 0.301079962/0.305161021s.
- nativePT_ours/clean: phi norm mean/median 0.0353240972/0.0286556892; update fraction 0.998; support mean/median 8.966/7; adapt mean/median 0.36829309/0.372465699s.
- nativePT_ours/corrupted: phi norm mean/median 0.0289826476/0.0244086403; update fraction 0.995333333; support mean/median 8.319/6; adapt mean/median 0.367649081/0.372027645s.

All conditions, blocks, AP/AP50/AP75 metrics, native component histories and paired counts are retained in [complete_tables.md](T018B/complete_tables.md). The full per-step records remain in samples.jsonl. Negative results are not omitted.

The T018-A developmental gain did not replicate. Under the frozen R030 decision rule, this exact fixed native pseudo-target formulation is not confirmed and should be closed pending research review, without tuning or cross-detector promotion.

Clean native-minus-no-adapt AP is -0.253654359; the clean comparison against current Ours must be read alongside this raw baseline.

This is a disjoint confirmation on the same source detector and availability-defined train population. Source-detector self-consistency remains an alternative explanation; no cross-detector/generalization conclusion is supported. Clean phi/update diagnostics do not by themselves establish selective adaptation. T017 attribution remains blocked and untouched.

All62raw files (139,528,015bytes), including smoke, are hashed in [artifact_manifest.json](T018B/artifact_manifest.json). Exact settings, JPEG/cohort hashes and commands are retained.

Stop NEEDS_REVIEW. No follow-on FCOS/SSD, COCO-val, hyperparameter/component search, spatial ISP, T017 forensics, meta-training or deployment replacement was executed. A new explicit research task is required for the next experiment.
