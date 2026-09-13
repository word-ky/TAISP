# T026-A — NEEDS_REVIEW; clean-source ROI memory objective FAIL

R041/d5a9d7a executed. Close this exact class-conditional clean-source nearest-memory ROI feature objective. No nomination/AP/K-step/runtime/FCOS/SSD/training or protected-method change.

## Source assumption, cohort and ordering

This candidate explicitly uses an offline source-GT-derived feature memory. It is not source-free/training-free in the strongest sense specified by R041. Detector weights remain frozen, with no source fitting/projector/prototype training; audit-time support and retrieval use only predicted classes.

Plan/cohort `a8610b6` precommitted source-instance hash ordering. Source manifest/code `cfcb895`; complete memory committed/pushed `2befc7f` BEFORE candidate outcomes. Candidate code `051d7ee`; all48 candidate records/anchors committed/pushed `b4c1c04` BEFORE reference code/run `fb8de70`. Separate candidate interpreter loaded no oracle/reference/audit annotations. Separate source-cohort preparation uses inherited GT-valid-box/readable-image eligibility, disclosed before outcomes. Source GT instances are used only to construct the frozen memory; audit GT enters only the post-lock reference process.

24 brand-new train2017 audit images, seed20260926, four6imageblocks,24clean plus8each gamma_s2/contrast_s2/color_cast_s2. Audit excludes1476previous source/audit/debug IDs and5000val IDs; memory additionally excludes24audit images,6500total. From readable clean training JPEGs, each class selects first16 noncrowd positive-area instances by sha256('T026A:20260926:'+annotationID), annotationID tie-break.1280 entries from1187distinctsourceimages, zero audit/prior/val overlap. No selection by model outcomes. Minimum available class count was23; no class substitutions or source downloads.

Source manifest SHA256 `ccff02e5c5786def9b6fcdfe4d4e98cfa3f23716c1521039852d486919f50766`. Complete memory.pt SHA256 `e51e7179f7a08e0652439b0708a1e0e275828097024558e9ed3b9b7370c9f12f`. Cohort SHA256 `01b7f61925e6a377e11a4fe20a12e9759569b57410ce384d740c0242afe8fceb`. All80classes have16 finite1024D unit vectors; max norm error=9.547524437714117e-08. No learned normalization/projection.

## Fixed objective and reference

Original-view FasterRCNN supports score>=.50, stable descending top20, original boxes/classes/scores frozen. Same binary support union mask; background identity and object8D identity analysis only. Query and current features use unchanged1024D box_head output before predictor. Same-predicted-class cosine top4, stable memory-index ties; anchor is detached normalize(mean4 normalized memory features). Single unweighted mean ROI cosine distance. No confidence weighting, cross-class retrieval, CLIP/pseudo/flip/exposure/geometry blend, fallback anchors or adaptation step.

Shared accepted common8-column ISP JVP; float32 image/model/ISP and independent float64 global/object/background reductions. Full source reference remains unchanged native four-loss sum at seed20260913 and currentdet_pseudo, eps1e-12. Exact supports, mask, selected IDs/similarities, query/anchor hashes, actual anchor tensors, per-object losses, cotangent diagnostics, JVP column norms/identity checks and8D gradients are retained. Memory and candidate hashes are checked before audit annotation loading.

## Frozen gate

|Metric|Observed|Required|
|---|---:|---:|
|S_mem>0 overall|26/48|>=30/48|
|S_mem>0 corrupt|18/24|>=15/24|
|Delta_global>0 overall|24/48|>=30/48|
|Delta_global>0 corrupt|12/24|>=15/24|
|MedianDelta overall|0.0014527534056069202|>0|
|MedianDelta corrupt|0.0014651648237539966|>0|
|Positiveblockmedians|2/4|>=3/4|
|Memory health/integrity|PASS|PASS|
|Conjunction|FAIL|allconditions|

S_mem=dot(t_o,g)/(norm(g)+eps); Delta_global subtracts dot(t_s,p_s)/(norm(p_s)+eps). Positive medians, retrieval similarity or Delta_obj do not rescue the failed counts/block requirement. Overall meanDelta=-0.022976433802312433; corrupted meanDelta=-0.041112448910959325.

## Retrieval and distribution diagnostics

Across all48 episodes,335distinctmemory entries from45predicted classes were used; corrupted subset273entries/40classes. Retrieval cosine overall mean/median=0.81916668744101/0.8293895125389099; corrupted=0.8159190622589937/0.8228422701358795.

`T026A/summary.json` reports overall,clean,corrupt,eachcorruption and all4blocks: positive counts/fractions,means,medians,linear quantiles[0,.25,.5,.75,1] for S_mem/Delta_global/Delta_obj; gradient norms/near-zero frequency; retrieval similarities, distinct entries/classes, unique-anchor counts and pairwise anchor-cosine diversity. Per-episode values/norms/cosines/mask/support/anchor-diversity/loss statistics are in per_episode.tsv. Undefined pairwise diversity for fewer than2supports is null, without excluding episodes.

## Integrity, validation and execution

All48reference integrity checks passed. Maxreverse/JVPrelativeL2=3.216667555966973e-06; partitionerror=1.7763568394002505e-15. Empty supports=0;zero/nearzero candidategradients=0/0;minimumgradientnorm=0.004442874961598745. Prospective nearzero convention remains norm<=1e-12, systematic if>=2distinct nonempty images; none occurred.

Source before/after every stage:73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf. Frozen/eval/parameter-grad-None and ISP-state checks passed.23protected+2inherited modules unchanged locally/remotely; all8new source/test pins match remote release. Tests: baseline18pass3.90s; memoryincrement13pass2.89s; candidate24pass3.83s; finalfocused26pass3.89s; full220pass11skip4warnings10.50s. Existing NVML/protobuf warnings retained, no driver/environment change.

All model-bearing execution on NVIDIA RTX A6000 CUDA12.1: memory `20260914-065856-taisp-t026a-memory1280` 30.942629848024808s; candidate `20260914-070245-taisp-t026a-candidates48` 9.698264045000542s; reference `20260914-070746-taisp-t026a-reference48` 21.361190908995923s; all exit0. CPU only manifests/synthetic checks/lightweight aggregation. Aborted launcher065756 left meta.json only and no tmux/run.sh; confirmed before successful065856 launch. Later SCPtimeout recovered through existingworkflow retry; no method change or repeated outcome run.

## Research-review numeric discrepancy and handoff

R041 quotes T025 figures that differ from authoritative `a268f90` receipts. Actual T025 S_geom28/48overall,15/24corrupt;Delta23/48,14/24;medians-.019543044761259266overall,+.009389868326732737corrupt;2/4blocks. R041 quotes27/48,16/24 and25/48,12/24 with different medians. Both concludeFAIL; T025 rawdata unchanged, no rerun, research-owned files not edited. T026 was executed as its independent newly authorized hypothesis.

Preserve all memory/48candidate/48reference raw artifacts under the three remote_runs roots, source-instance/JPEG/exclusion manifests, codepins and commit locks. Future fresh cohorts MUST use `research_log/T026A_source_exclusion_manifest.json` as the cumulative additional_source: includes all1187memoryimages+24audit and1476prior IDs,2687source IDs total, plusval5000 exclusion. Do not use only T026A_train_cohort, which omits memory images.

Stop NEEDS_REVIEW; close this exact clean-source nearest-memory ROI objective. No top-k/memory-size/layer/projection/source-selection/weights/support/mask/K/LR/blend rescue and no T026-B/AP/new family without a new explicit research task.
