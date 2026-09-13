# T025-A — NEEDS_REVIEW; fixed geometry objective FAIL

R040/0b6e3ca executed. The fixed `roi_bbox_exposure_stability` candidate fails the original conjunction. Close this exact exposure-pair ROI box-geometry objective. No nomination, AP, K-step runtime, FCOS/SSD, source/meta/predictor training or protected-method change.

## Protocol and ordering

Plan/cohort `65416d5`; candidate code `8738f05`; all48 candidate receipts committed and pushed `ef14e99` before reference code/run `2cd7415`. Separate candidate interpreter loaded no GT/oracle/reference module. Only the inherited cohort-preparation process read annotations for noncrowd valid-box/readable-image eligibility; candidate input contained image metadata and conditions only.

24 fresh train2017 images, seed20260925, excluding1452 prior source/development/debug/audit IDs plus all5000val IDs (6452total). Four consecutive6imageblocks.48episodes=24clean+8gamma_s2+8contrast_s2+8color_cast_s2. Ordered IDs/JPEG hashes/full exclusion ledger are committed.

Original-view FasterRCNN score>=.50/stable descending top20 supports were frozen. Same binary union mask; background identity and object8D identity as analysis variable only. Exposure views are exactly a*y/(1+(a-1)*y), factors1.2 and1/1.2, no clamp/randomness. Raw predictor regression layout(N,91,4) uses original ROI index/frozen predicted class. Per-object SmoothL1(beta1) sums four coordinates; objective means supports. No decoding/NMS/rematching/weights/feature-logit blend.

Accepted common8-column ISP JVP: float32 image/model/ISP, independent float64 global/object/background reductions onCUDA. Same unmodified helper for candidates/reference, equality to A1 checked. Source full-task oracle is unchanged native four-loss sum at seed20260913; current pseudo is unchanged det_pseudo. Additional ROI box-regression and combined RPN+ROI localization gradients come from the same oracle forward as diagnostics only.

## Frozen gate

|Metric|Observed|Required|
|---|---:|---:|
|S_geom>0 overall|28/48|>=30/48|
|S_geom>0 corrupt|15/24|>=15/24|
|Delta_global>0 overall|23/48|>=30/48|
|Delta_global>0 corrupt|14/24|>=15/24|
|Median Delta_global overall|-0.019543044761259266|>0|
|Median Delta_global corrupt|0.009389868326732737|>0|
|Positive block medians|2/4|>=3/4|
|Conjunction|FAIL|all conditions|

S_geom=dot(t_o,g)/(norm(g)+1e-12); Delta_global subtracts dot(t_s,p_s)/(norm(p_s)+1e-12). Overall meanDelta=-0.030934327594878755; corrupted meanDelta=-0.06356534479306966. Positive corrupted median does not substitute for the failed overall/count/block conditions. No post-outcome change was made.

ROI-box diagnostic alignment is positive on27/48overall and13/24corrupt; combined localization positivity is27/48 and13/24. These do not alter the full-task gate. Exact condition/block summaries, object-current diagnostics, cosines, all gradient norms, mask/support strata values and regression-delta disagreement are retained in summary.json/per_episode.tsv and raw records.

## Integrity and tests

All48 reference integrity checks passed. Max global reverse/JVP relativeL2=1.7862943797059111e-06; max partitionerror=8.881784197001252e-16. Empty supports=0. Zero/near-zero candidategradients=0/0; minimum norm=0.001093627708136352. Before outcomes, systematic near-zero was specified as nonempty norm<=1e-12 on at least2distinctimages; none occurred.

Source state before/after both stages:73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf. Detector frozen/eval/parameter-grad-None, ISP state unchanged. All23protected modules and2inherited analysis helpers unchanged locally/remotely; all4new source/test pins verified on remote. Class slice and ROI ordering passed synthetic and real FasterRCNN CUDA smoke on a generated tensor (no extra COCO image).

Baseline18tests passed3.91s; candidate increment24passed6.04s; final focused30passed6.13s including cached real-model CUDA smoke. Full regression212passed/11skipped/4warnings in10.56s. Existing NVML and protobuf warnings retained; CUDA succeeds, no driver/environment changes.

Candidate run `20260914-052059-taisp-t025a-candidates48`: 10.715751997020561s; reference `20260914-052527-taisp-t025a-reference48`: 23.271904545021243s, both exit0. NVIDIA RTX A6000/CUDA12.1. CPU only for manifests, synthetic unit checks and lightweight aggregation.

## Artifacts and disposition

`research_log/T025A/` contains per_episode.tsv, exact summary.json, integrity and code-hash checks. Raw48candidate records retain supports/classes/order, mask rectangles/hash, both Nx4exposure deltas, per-object losses, cotangent shape/norm/hash, eight JVP column norms/identity checks and final global/object/background8D gradients. Raw48reference records retain unchanged task/pseudo plus diagnostic gradients, numerical checks and all metrics. Candidate and reference roots are `research_log/remote_runs/20260914-052059-taisp-t025a-candidates48/artifacts/candidates` and `research_log/remote_runs/20260914-052527-taisp-t025a-reference48/artifacts/reference`. Plans, JPEG/exclusion pins, pre-reference candidate commit/hash lock and tests are retained.

Stop NEEDS_REVIEW. Close this exact exposure-pair ROI box-geometry objective per R040. Do not tune exposure/beta/views/layer/decoding/classes/support/mask/K/LR/corruptions, implement T025-B or start a clean-source anchor/memory family without a new explicit research task.
