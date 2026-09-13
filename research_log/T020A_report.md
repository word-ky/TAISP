# T020-A — close_fixed_global_linear_gradient_transport; NEEDS_REVIEW

R032/2654048. Plan/cohort precommit766ae89; experiment code0546b05.
Formal `20260913-203203-taisp-t020a-source200-crossfit`; smoke `20260913-202806-taisp-t020a-runtime-smoke`; release20260913-202732-taisp-t020a-crossfit.
200 new train2017 images, four image-level 50-image folds, exclude836prior-source/all5000val;
cohortSHAc364fb4d0bf2880318dbbcd22a47605cedccb31f6ee783ffdc56f6650d1f9ade, selectionseed20260921, model/oracle seed20260912.
For each held-out fold Q=U Vh from normalized training P^T T (150images/1050pairs).
All1400rawpairs/fourmatrices/singularvalues/trainindices retained. No determinant correction,
optimizer, ridge, hyperparameter search, per-condition matrix or post-outcome exclusions.
Source labels enter only the separate collection process. Held-out gt is diagnostic only
for its own fold. Runtime receives only matrix/provenance and the existing label-free
pseudo loss; gd @ Q then unchanged CLIP norm transfer. K3/LR.1/identity/score.5top20,
frozen source/CLIP/ISP operators/prompts/corruptions unchanged. Public current Ours is
unchanged by default; identity-Q equivalence tested. No target/val/cross-detector work.

## Exact performance result

Corruption macro AP: {"no_adapt": 45.728536946294895, "current_ours": 45.86888482962125, "grad_transport_ours": 45.876769932637814}.
Candidate minus current: **+0.007885103 AP**; minus raw **+0.148232986 AP**.
Positive blocks **1/4**; positive conditions **3/6**.
Clean delta current **-0.027302686 AP**, clean delta raw **+0.230903596 AP**.
Macro AP50/AP75 deltas current: {"AP50": 0.017578982611539118, "AP75": -0.014407043471898362}.
Frozen decision passed: **False**. Alignment is diagnostic and does not override AP criteria.

## R032 frozen criteria

| Criterion | Passed |
| --- | --- |
| at_least3_positive_block_macros | False |
| at_least4_positive_corruption_conditions | False |
| corruption_macro_above_no_adapt | True |
| clean_delta_at_least_minus_point10 | True |
| corruption_macro_delta_at_least_point15 | False |
| no_leakage_isolation_norm_or_reproducibility_blocker | True |

## Held-out block deltas

| Block | AP delta current | AP delta raw | AP50 delta current | AP75 delta current |
| --- | --- | --- | --- | --- |
| block0 | -0.0513109386 | 0.243254965 | 0.05579032 | -0.486594508 |
| block1 | -0.0667877776 | 0.0470332441 | 0.0570227055 | 0.16282306 |
| block2 | 0.168861644 | 0.341828325 | 0.369034187 | 0.258177304 |
| block3 | -0.0851499125 | 0.129080871 | -0.149576139 | -0.257400717 |

## Held-out identity gradient alignment

| Group | N | Defined | Undefined zero | Raw mean | Transport mean | Raw median | Transport median | Improved fraction defined |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 1400 | 1388 | 12 | 0.154443289 | 0.134915686 | 0.252397539 | 0.223561864 | 0.425792507 |
| corrupted | 1200 | 1190 | 10 | 0.172892568 | 0.153082777 | 0.28155645 | 0.259796358 | 0.423529412 |
| clean | 200 | 198 | 2 | 0.0435612594 | 0.0257296366 | 0.0179917289 | 0.0162991211 | 0.439393939 |
| block0 | 350 | 348 | 2 | 0.15146433 | 0.117725105 | 0.259835862 | 0.205998982 | 0.413793103 |
| block1 | 350 | 346 | 4 | 0.225569814 | 0.200882597 | 0.325971211 | 0.29849185 | 0.413294798 |
| block2 | 350 | 344 | 6 | 0.13696613 | 0.128634492 | 0.210010469 | 0.222781746 | 0.453488372 |
| block3 | 350 | 350 | 0 | 0.104269126 | 0.0929685494 | 0.211616711 | 0.191368466 | 0.422857143 |
| clean_s0 | 200 | 198 | 2 | 0.0435612594 | 0.0257296366 | 0.0179917289 | 0.0162991211 | 0.439393939 |
| color_cast_s1 | 200 | 198 | 2 | 0.120905884 | 0.102717565 | 0.170279354 | 0.192424695 | 0.404040404 |
| color_cast_s2 | 200 | 197 | 3 | 0.11013093 | 0.088560946 | 0.13141649 | 0.12671727 | 0.436548223 |
| contrast_s1 | 200 | 199 | 1 | 0.240000604 | 0.225118702 | 0.423148637 | 0.378065715 | 0.43718593 |
| contrast_s2 | 200 | 198 | 2 | 0.28063397 | 0.255027109 | 0.524278753 | 0.429671189 | 0.388888889 |
| gamma_s1 | 200 | 199 | 1 | 0.0892020651 | 0.0758087031 | 0.117540835 | 0.0964720875 | 0.472361809 |
| gamma_s2 | 200 | 199 | 1 | 0.196131361 | 0.170874367 | 0.256825335 | 0.24743637 | 0.40201005 |

## Adaptation diagnostics

| Method/group | N | Mean phi | Median phi | Update fraction | Mean support | Median support | Mean adapt s | Median adapt s | Mean teacher-inclusive s | Peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_ours/clean | 200 | 0.0355016243 | 0.0276657511 | 0.99 | 8.255 | 6.5 | 0.301963056 | 0.305063554 | 0.33042509 | 3.06136894 |
| current_ours/corrupted | 1200 | 0.0292750998 | 0.0245520892 | 0.991666667 | 7.70583333 | 6 | 0.302520396 | 0.305138241 | 0.332118154 | 3.06378603 |
| grad_transport_ours/clean | 200 | 0.0357404708 | 0.0280565089 | 0.99 | 8.255 | 6.5 | 0.305419086 | 0.309160605 | 0.333881121 | 3.06136942 |
| grad_transport_ours/corrupted | 1200 | 0.0292663144 | 0.0246495586 | 0.991666667 | 7.70583333 | 6 | 0.305364458 | 0.308845787 | 0.334962215 | 3.06350708 |

## Validation, timing and artifacts

Baseline10passed2skipped1.79s; increment1 12passed1skipped1.69s.
Focused: 16 passed, 2 skipped in 1.70s
Full regression: 168 passed, 10 skipped, 4 warnings in 7.49s
Two-image smoke:14sourcepairs,twoleave-one-image-outmaps,28adaptiveK3episodes,0AP.
Formal:1400sourcepairs,2800adaptiveepisodes,21predictionfiles,105officialevaluations.
GPU collection 196.470625s; CUDA float64 fit 0.040319s;
CUDA float32 runtime collection/officialCPUevaluation 1092.429241s.
All source/runtime isolation and code pins pass. 5600runtime norm checks pass,
max relative norm error 1.57913324819e-07; paired teacher supports match all1400episodes.
Zero pseudo/task pairs: 12/0; retained in fitting
because EPS normalization is defined. Zero-vector alignment is null and reported explicitly.
Empty adaptive episodes 24; exact identity updates verified.
Source-vs-runtime identity pseudo maximum absolute L2 diagnostic 0.00193120774416.
Separate float32 collection/runtime gradients are not bit-identical: maximum relative L2
0.00225706747473, median 9.94599721946e-07,
minimum defined cosine 0.999999125542. Same objective/code/supports
are verified; the numerical differences are disclosed without claiming their cause,
rerunning models, changing tolerances, or resuming the closed T017 forensics task.
All16actual remote release code hashes match pins; see T020A/remote_code_hashes.json.
Adaptation latency includes terminal gradient diagnostics (four evaluations,threeupdates),
excludes source fitting, final prediction, state verification and AP. Teacher-inclusive adds
shared teacher setup. Peak memory includes frozen-state verification copies.

76rawfiles, 57905525bytes; SHA256 manifest inT020A/artifact_manifest.json.
All AP/AP50/AP75 tables,clean/block/condition deltas,Q/singularvalues/matrixsimilarity and
diagnostics: T020A/complete_tables.md. Isolation/norm/hash audit:T020A/receipt_audit.json.
No tuning or extra cohort was run. StopNEEDS_REVIEW. A pass permits only a developmental
candidate; a failure closes the fixed global linear gradient-transport branch underR032.
