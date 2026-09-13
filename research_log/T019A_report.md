# T019-A — No candidate eligible; close this fixed component family; NEEDS_REVIEW

Eligible candidates: []. Selected: **None**.
Near-best candidates within0.01AP: [].
Selection uses only frozen six-corruption macro AP, then macro AP75 within the0.01AP tie range,
then number of positive blocks. Any selected method is a developmental candidate, not confirmed.
Baseline corruption macro AP: no-adapt 46.224650701, current Ours 46.285815818.

## Protocol and provenance

R031/pointerd60e72c; pre-outcome plan/config/cohort/method hashes b49c432;
experiment codeee98de0; CUDA smoke receipt d047312.
Release20260913-182322-taisp-t019a-components; formal `20260913-182636-taisp-t019a-source200-components`; smoke `20260913-182437-taisp-t019a-runtime-smoke`.
200newtrain2017images,fourprecommitted50-imageblocks,636prior-source/all5000val excluded.
CohortSHA534b17343ccba995beb9d112d552eefd7f1b2116479c2132dd99265b1d6735b4.
Same readable/non-crowd positive-box/area eligibility as T018-A/B;13,162eligible after exclusions.
Hash selection seed20260920; model/native sampling seed20260912. Annotation eligibility is used
only for cohort preparation. Adaptation sees JPEGs and detached teacher predictions; official
GT annotations enter evaluation only after all prediction files are saved.

Only named native-loss subset selection/configuration and study reporting were added.
Ten protected modules remain unchanged: existing current Ours, adaptation, source detector,
CLIP, teacher filtering, oracle, ISP and corruptions. Default/full native behavior remains
backward compatible; the failed full-native formulation was not evaluated as a fifth candidate.
All candidates use detachedscore>=.5/top20 teacher boxes/int labels, unit active coefficients,
global8D identity phi,K3/LR.1/EPS1e-12 and current-phi CLIP-norm transfer with direction discarded.
Detector and CLIP remain frozen; only functional episodic ISP state changes.

## Exact fixed family

| Candidate | Active keys |
| --- | --- |
| native_cls | loss_classifier |
| native_conf | loss_classifier, loss_objectness |
| native_roi | loss_classifier, loss_box_reg |
| native_conf_roi | loss_classifier, loss_objectness, loss_box_reg |

## Validation and execution

- Unchanged baseline12passed2skipped1.57s; focused24passed2skipped1.82s.
- Full162passed10skipped6.84s. Optional tests remain skipped; these are not162real-model tests.
- Two-imageCUDAK3smoke:70adaptiveepisodes,56candidateepisodes,224exactfloat32active-sum checks,
  allstate/K3checks passed,0AP,exit0. Exact receipt inT019A_smoke_audit.json.
- Formal200images,1400teacherforwards,7000adaptiveepisodes,42predictionfiles,210officialevaluations.
  Collection/evaluation elapsed2887.666868s. Exact start/finish/exit in train.log.
- A6000cuda:0,float32,threads1. Source/CLIP before-after hashes and all per-episode isolation checks pass.
  CUDA model work completed despite the existing NVML warning. Exact environment/commands are retained.
- Empty-support semantics are covered by focused tests and per-episode checks. Actual empty fractions appear below.
  Four diagnostic evaluations apply three updates. Latency includes terminal diagnostics and excludes final
  prediction/state verification/evaluation; teacher-inclusive adds the original detector teacher forward.

## Aggregate candidate outcomes

| Candidate | Macro AP | Delta current | Delta raw | AP50 delta current | AP75 delta current | Positive blocks/4 | Positive conditions/6 | Clean delta current | Clean delta raw | Eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| native_cls | 46.2654886 | -0.0203272635 | 0.0408378534 | -0.0447491063 | 0.268549269 | 1 | 1 | -0.0915215888 | -0.421743862 | False |
| native_conf | 46.262627 | -0.0231888536 | 0.0379762634 | -0.0298936328 | -0.0545570098 | 1 | 3 | 0.153617006 | -0.176605267 | False |
| native_roi | 46.2334727 | -0.0523431487 | 0.00882196822 | 0.0014770446 | 0.13791045 | 1 | 2 | -0.0922907317 | -0.422513005 | False |
| native_conf_roi | 46.2670042 | -0.0188115829 | 0.042353534 | -0.0060565217 | 0.00228427523 | 1 | 3 | 0.20033921 | -0.129883063 | False |

## Aggregate AP across all conditions

| Condition | no_adapt | current_ours | native_cls | native_conf | native_roi | native_conf_roi |
| --- | --- | --- | --- | --- | --- | --- |
| gamma_s1 | 49.5901657 | 49.4080697 | 49.2703167 | 49.2834328 | 49.2473864 | 49.291481 |
| gamma_s2 | 45.7607905 | 46.2601615 | 46.1031225 | 45.9808983 | 46.083146 | 46.0608268 |
| contrast_s1 | 47.4913619 | 47.6649845 | 47.613763 | 47.6846187 | 47.5163084 | 47.4594837 |
| contrast_s2 | 38.6475226 | 38.4729897 | 38.4166862 | 38.2284587 | 38.4444419 | 38.6465956 |
| color_cast_s1 | 48.9878659 | 49.2062757 | 49.4960041 | 49.5231103 | 49.395812 | 49.4288696 |
| color_cast_s2 | 46.8701977 | 46.7024138 | 46.693039 | 46.8752429 | 46.7137413 | 46.7147687 |
| clean_s0 | 52.04403 | 51.7138077 | 51.6222861 | 51.8674247 | 51.621517 | 51.9141469 |

## All four block deltas

| Candidate | Block | AP delta current | AP50 delta current | AP75 delta current |
| --- | --- | --- | --- | --- |
| native_cls | block0 | -0.0406247248 | -0.0567126647 | 0.288258015 |
| native_cls | block1 | -0.0373283496 | 0.0115625407 | 0.143338947 |
| native_cls | block2 | -0.199824676 | -0.216632072 | -0.193218986 |
| native_cls | block3 | 0.131034011 | -0.121396042 | 0.505533804 |
| native_conf | block0 | -0.0643781287 | -0.0345129666 | 0.193625518 |
| native_conf | block1 | -0.297888833 | -0.158568239 | -0.408483285 |
| native_conf | block2 | -0.0242702589 | -0.0924320373 | -0.350014483 |
| native_conf | block3 | 0.0598880035 | -0.189453102 | 0.210631595 |
| native_roi | block0 | -0.246648349 | 0.0923008379 | -0.944331076 |
| native_roi | block1 | -0.345227253 | -0.108285644 | 0.0107261108 |
| native_roi | block2 | -0.293485095 | -0.269089512 | -0.0820024376 |
| native_roi | block3 | 0.0768842557 | -0.107391639 | 0.727016522 |
| native_conf_roi | block0 | -0.0973876755 | -0.0368951791 | -0.42633658 |
| native_conf_roi | block1 | -0.133335361 | -0.0249798823 | -0.126797416 |
| native_conf_roi | block2 | -0.203850888 | -0.0421749002 | 0.112650923 |
| native_conf_roi | block3 | 0.0833835311 | -0.168985802 | 0.420422021 |

## All frozen eligibility checks

| Candidate | Criterion | Passed |
| --- | --- | --- |
| native_cls | corruption_macro_delta_at_least_point10 | False |
| native_cls | at_least3_positive_block_macros | False |
| native_cls | at_least4_positive_corruption_conditions | False |
| native_cls | corruption_macro_above_no_adapt | True |
| native_cls | clean_delta_at_least_minus_point10 | True |
| native_cls | no_isolation_or_reproducibility_blocker | True |
| native_conf | corruption_macro_delta_at_least_point10 | False |
| native_conf | at_least3_positive_block_macros | False |
| native_conf | at_least4_positive_corruption_conditions | False |
| native_conf | corruption_macro_above_no_adapt | True |
| native_conf | clean_delta_at_least_minus_point10 | True |
| native_conf | no_isolation_or_reproducibility_blocker | True |
| native_roi | corruption_macro_delta_at_least_point10 | False |
| native_roi | at_least3_positive_block_macros | False |
| native_roi | at_least4_positive_corruption_conditions | False |
| native_roi | corruption_macro_above_no_adapt | True |
| native_roi | clean_delta_at_least_minus_point10 | True |
| native_roi | no_isolation_or_reproducibility_blocker | True |
| native_conf_roi | corruption_macro_delta_at_least_point10 | False |
| native_conf_roi | at_least3_positive_block_macros | False |
| native_conf_roi | at_least4_positive_corruption_conditions | False |
| native_conf_roi | corruption_macro_above_no_adapt | True |
| native_conf_roi | clean_delta_at_least_minus_point10 | True |
| native_conf_roi | no_isolation_or_reproducibility_blocker | True |

Every candidate fails: corruption_macro_delta_at_least_point10, at_least3_positive_block_macros, at_least4_positive_corruption_conditions. No tie-breaker is invoked because none is eligible. Close this fixed component-subset branch without weights/thresholds or another cohort search.

## Adaptation diagnostics

| Method/group | N | Mean phi | Median phi | Update fraction | Mean support | Median support | Empty fraction | Mean adapt s | Median adapt s | Mean adapt+teacher s | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_ours/clean | 200 | 0.0377769114 | 0.0314012226 | 0.995 | 9.03 | 6 | 0.005 | 0.297411886 | 0.298980041 | 0.325570382 | 3.04440975 |
| current_ours/corrupted | 1200 | 0.0306389889 | 0.0254503032 | 0.995 | 8.26583333 | 5 | 0.005 | 0.29759234 | 0.299573776 | 0.326740824 | 3.048769 |
| native_cls/clean | 200 | 0.0365645393 | 0.0284771249 | 0.995 | 9.03 | 6 | 0.005 | 0.351654874 | 0.35330491 | 0.379813371 | 3.05197954 |
| native_cls/corrupted | 1200 | 0.0288262436 | 0.0238801073 | 0.995 | 8.26583333 | 5 | 0.005 | 0.35125429 | 0.353189668 | 0.380402773 | 3.05456734 |
| native_conf/clean | 200 | 0.0363951075 | 0.0300274054 | 0.995 | 9.03 | 6 | 0.005 | 0.36089721 | 0.36246129 | 0.389055706 | 3.22216225 |
| native_conf/corrupted | 1200 | 0.0291163369 | 0.0242138272 | 0.995 | 8.26583333 | 5 | 0.005 | 0.360661097 | 0.362693137 | 0.389809581 | 3.22547531 |
| native_roi/clean | 200 | 0.0347095637 | 0.0283422433 | 0.995 | 9.03 | 6 | 0.005 | 0.354263408 | 0.356049948 | 0.382421904 | 3.05147648 |
| native_roi/corrupted | 1200 | 0.0283636398 | 0.0242336057 | 0.995 | 8.26583333 | 5 | 0.005 | 0.354170862 | 0.355761024 | 0.383319346 | 3.05285263 |
| native_conf_roi/clean | 200 | 0.0354220146 | 0.0277842712 | 0.995 | 9.03 | 6 | 0.005 | 0.361702628 | 0.363279155 | 0.389861124 | 3.2232933 |
| native_conf_roi/corrupted | 1200 | 0.0289198909 | 0.0243089888 | 0.995 | 8.26583333 | 5 | 0.005 | 0.361763753 | 0.363787022 | 0.390912237 | 3.22571516 |

All condition/block AP/AP50/AP75 values and deltas against both baselines, active per-step losses and gradient norms are in [complete_tables.md](T019A/complete_tables.md); per-episode raw traces retain all supports/phi/gradients/counts/timings. No negative candidate, condition or block is omitted.

This is a four-candidate source-development study in the availability-defined training population. No candidate is independently confirmed here, and no cross-detector/generalization claim follows. Component comparisons do not prove a causal account of teacher geometry noise. T018-B remains a negative full-native confirmation; T017 attribution remains blocked.

All104raw files (124,041,334bytes), including smoke, are hashed in [artifact_manifest.json](T019A/artifact_manifest.json).

StopNEEDS_REVIEW. No coefficient/threshold/K/LR/prompt tuning, new cohort, FCOS/SSD/val2017, spatial ISP, T017 forensics, gate/dose/meta-training or deployment replacement occurred. Follow-on work requires a new explicit research task.
