# T032-A / R050 — scientific FAIL; NEEDS_REVIEW

The frozen spherical K=2 router plus two O(8) source experts fails the predeclared holdout conjunction. Close this exact two-regime transport family without rescue tuning. This is only a first-order source gradient capacity audit. No AP, K-step adaptation or deployment performance is claimed. T031 remains closed; none of its records/maps/outcomes entered this fit.

## Frozen stages, provenance and label separation

Research R050 195bfc8 / latest8d45ec2. Fresh cohort/split/protocol **b0ce54b**; candidate code is unchanged R049 orthogonal_candidates.py. New router code/tests **d71bd34**; reference integration/tests **0d22b7d**, both committed before real router/source fits. Candidate480 commit **5ae8bb68087d9739b5f69c7524a9f4b75effc829**, descriptor **0401c8e** BEFORE router fitting. Router+assignments commit **844bc0d0a8dbaf2f24b11ab35cbddc50aad6cb98**, descriptor **2742f87**, BEFORE annotations/task gradients. Both source maps+train references commit **a0ecfdab59b7425fb93ba298f647f1bf33794d6e** BEFORE holdout task gradients. Map descriptor **57bf9b6** is committed before the holdout launch; its SHA is 611ce63d828fef3d74b6e48f53f3468ab60af67f3ff0b19920c49fce31be3cfd. Exact map-lock receipt: {"map_commit": "a0ecfdab59b7425fb93ba298f647f1bf33794d6e", "maps_sha256": "36a0e62b7696b8ea99bc92f6806c8f30dc7d1aa4a6414cfd2ae81dfb5219ec4e", "map_lock_sha256": "611ce63d828fef3d74b6e48f53f3468ab60af67f3ff0b19920c49fce31be3cfd"}.

240 completely fresh COCO train2017 images, selectionseed20261002, excluding3251 prior/reserved IDs (including T031) plus5000val and1400evaluation IDs. Existing prepare_t018a eligibility/order; cyclic frozen severity2 family assignment. First180 images train360episodes (60/family), last60 holdout120episodes (20/family); clean then assignedcorruption perimage; four holdout blocks15pairs/5perfamily. Cohort SHAa51028db007f2530ef1ae01c7d683aa0c7e70dac2acd3cc0a78b31b1291325b0. Outcome-free preparation parsed annotations to screen valid boxes and write separate train/holdout subsets. Candidate manifest contains no annotation paths; no candidate oracle/reference import. Train process opened only train annotations after the router commit. Holdout process wrote ALL120 routes/transformed gradients before lazyoracle imports and task-gradient computation.

Router fit accepts only360 ordered train hard-gradient vectors. It receives no annotation, task outcome, image/family/case metadata or holdout vectors. For nonzero g, u=g/(norm+1e-12); exactzero routes0 and stays counted. First nonzero initializes centroid0; smallest cosine initializes centroid1, ties earliest. Maximum-cosine assignments tie0; centroids normalized arithmetic means; unchanged assignments or20iteration cap; no restarts or K search. Empty/zero-mean centroid collapse was predeclared before outcomes. Router iterations=6, converged=True, initial indices=[0, 299], counts=[171, 189], zero count=1, counttrace=[[195, 165], [183, 177], [173, 187], [169, 191], [171, 189], [171, 189]]. Both>=72; no router-collapse stop. Router SHA2e150aadf557db106ebf73a74372b91d91751d52e80b041de3bf8db6010f738a, candidate-lock SHA55db016f1a43d5ed04ccea7fc0ffd07ce547a7cea678af210f52aee66cd9e40e.

Candidate run20260914-181428-taisp-t032a-candidates480, release20260914-181346-taisp-t032a-candidates, sourceb0ce54b,480episodes/297.5412268320215s. Router command and outputs recorded in project progress; fitted once in release20260914-181842-taisp-t032a-reference. Train run20260914-182353-taisp-t032a-train360-maps, source2742f87,360episodes/222.71183044300415s. Holdout run20260914-183220-taisp-t032a-holdout120, source57bf9b6,120episodes/76.50184720597463s. Train/holdout use the same frozen release20260914-181842-taisp-t032a-reference; exact commands/timestamps/source revisions retained in raw runs. A6000CUDA0 for all model forward/backward/JVP; CPUfloat64 only router, two8x8SVD and summaries. Same frozen Faster R-CNN, 8-D identityISP, score>=.50 stabletop20 confidence-weighted hard objective and taskreferenceseed20260913. Source stateSHA73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf unchanged.

## Source expert diagnostics

Exactly one Procrustes SVD per frozen train cluster, two total; unit normalize iff norm>1e-12 elsezero; C=sum u_t u_h.T; R=U@Vt, fullO(8), no determinant correction/bias/ridge/weights. Stored transformed gradients are not renormalized. Maps SHA36a0e62b7696b8ea99bc92f6806c8f30dc7d1aa4a6414cfd2ae81dfb5219ec4e; full matrices/C/spectra in T032A/maps.json. Maximum holdout norm-preservation absolute error=1.3322676295501878e-15.

Cluster0: train count171; zero unit hard/task1/0; determinant0.9999999999999987; orthogonality maxabs8.881784197001252e-16; singular spectrum[24.623088278334762, 5.6530900982464205, 3.275284526834716, 1.4253554825569181, 0.7504357458207236, 0.25342242598764236, 0.0472728306163204, 3.7299897666596714e-13]; SVD calls1.
Cluster1: train count189; zero unit hard/task0/0; determinant-0.9999999999999994; orthogonality maxabs5.551115123125783e-16; singular spectrum[16.723406962358702, 9.914379018363183, 5.485847250541336, 1.7055392930423905, 0.9636525002414557, 0.30661934393509627, 0.007165717009272259, 9.662256352583755e-14]; SVD calls1.

## Frozen R050 gate

Required S_routepositive>=80/120overall,45/60corrupt; Deltapositive>=72/120overall,36/60corrupt; mean+medianDeltaoverall/corrupt>0; >=3/4positive block medians; cleanmedian>=0; bothholdoutroutes>=12; all integrity. Exactzero episodes retained in denominators. S_route=<task,g_route>/(norm(g_route)+1e-12), Delta=S_route-S_hard. Summary reuses R049 distribution field names S_cal/g_cal/cos_cal_task as aliases for S_route/g_route/routed cosine; no old T031 map is evaluated.

|Gate|Pass|
|---|---|
|Delta_overall|False|
|Delta_corrupt|False|
|median_overall|False|
|median_corrupt|False|
|mean_overall|False|
|mean_corrupt|False|
|positive_blocks|False|
|median_clean|False|
|integrity|True|
|S_route_overall|False|
|S_route_corrupt|False|
|holdout_route_coverage|True|

|Group|N|S_hard>0|S_route>0|Delta>0|MedianDelta|MeanDelta|Median cos(h,task)|Median cos(route,task)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|overall|120|78|79|49|-0.0136676592432203|-0.06368087305755213|0.3883204364055197|0.20813009776477268|
|clean|60|38|38|24|-0.0183228039167913|-0.03202084978958428|0.31450517090683205|0.13212984766147934|
|corrupt|60|40|41|25|-0.006134691018266185|-0.09534089632552|0.4506687042514488|0.28127371022723474|
|clean_s0|60|38|38|24|-0.0183228039167913|-0.03202084978958428|0.31450517090683205|0.13212984766147934|
|color_cast_s2|20|11|13|12|0.011977380328050242|0.0004421921250471416|0.10020388325747168|0.13466435266903223|
|contrast_s2|20|14|15|7|-0.027951687943026146|-0.23626656641770766|0.4856945173934663|0.415386045437099|
|gamma_s2|20|15|13|6|-0.020262423843788215|-0.050198314683899445|0.46076074389405086|0.12152893062399556|
|block0|30|21|20|9|-0.03903835433007794|-0.053466896354947877|0.407152343847638|0.16533909377928047|
|block1|30|20|20|14|-0.001609350358977203|-0.07537409765550403|0.28633148493000415|0.1111203612936231|
|block2|30|18|19|14|-0.0031494872970173806|-0.022948554128773238|0.21265264076797716|0.20648697333847083|
|block3|30|19|20|12|-0.01849145291147203|-0.10293394409098337|0.5284239041395957|0.30445574155055444|
|cluster0|59|42|39|19|-0.020880671369171155|-0.04720042487043893|0.4223291133319539|0.19017214232337892|
|cluster1|61|36|40|30|-0.0012506616054783724|-0.07962097868115343|0.23913313348186124|0.22608805320616643|

Route counts by scope (diagnostics only; never routing inputs): {"overall": [59, 61], "clean": [32, 28], "corrupt": [27, 33], "clean_s0": [32, 28], "color_cast_s2": [9, 11], "contrast_s2": [9, 11], "gamma_s2": [9, 11], "block0": [20, 10], "block1": [11, 19], "block2": [14, 16], "block3": [14, 16]}.

## Coordinate concentration

Full8-coordinate mean/std/meanabs/RMS/energy share by scope/cluster are in summary.json. Energy share=sum g_j²/sum_all g²; indices zero-based. No diagnostic-driven refit.

|Group|Gradient|Max-energy coordinate|Max share|
|---|---|---:|---:|
|overall|g_hard|2|0.4288587059503316|
|overall|g_cal|3|0.2248202580879998|
|clean|g_hard|2|0.4334622398389448|
|clean|g_cal|3|0.3207614967297679|
|corrupt|g_hard|2|0.4270663551907254|
|corrupt|g_cal|2|0.21060552431929866|
|clean_s0|g_hard|2|0.4334622398389448|
|clean_s0|g_cal|3|0.3207614967297679|
|color_cast_s2|g_hard|2|0.3712840444182266|
|color_cast_s2|g_cal|3|0.3646667390519541|
|contrast_s2|g_hard|2|0.5010419097404004|
|contrast_s2|g_cal|2|0.2512080566098174|
|gamma_s2|g_hard|1|0.5174622508073121|
|gamma_s2|g_cal|1|0.41588113835981805|
|block0|g_hard|1|0.40783294379596896|
|block0|g_cal|3|0.3767305117299513|
|block1|g_hard|2|0.6545732289555473|
|block1|g_cal|4|0.3068959626330034|
|block2|g_hard|2|0.33906420213419736|
|block2|g_cal|1|0.28513804964231454|
|block3|g_hard|2|0.26587930431994294|
|block3|g_cal|3|0.2701745425515943|
|cluster0|g_hard|2|0.37115585798245765|
|cluster0|g_route|3|0.35509330495564995|
|cluster1|g_hard|2|0.49094401510914665|
|cluster1|g_route|4|0.24967887114951687|

## Integrity, tests and artifacts

All480candidate+360train+120holdout integrity/RNG/state/parity receipts pass. Stored hard vectors unchanged; pair-preserving train/holdout disjoint; train assignments exactly router lock; holdout routes/transformed gradients exactly equal pre-oracle file. All65 source/code pins match locally/remotely, including61protected/prior files. Candidate/router/source-model/deployment code not changed after outcomes. AP0; no K-step/runtime/FCOS/SSD. Raw hashes: {"candidate": {"file_count": 483, "records_sha256": "d72c8f40316e05cf2b87c75fcac6c0d9a0f626dabc1006bc5372d258f79e1adc", "manifest_sha256": "c416b04b11438b94ace47e23b20372e0d6ca94bb39639651a1bb77606aef7f67"}, "train": {"file_count": 365, "records_sha256": "98f151f7351f2e77732e22f4a317c6f4ba43dd381bcdb6f74cf693e03b75e286", "manifest_sha256": "a4fcfa95f346a8653498b8169d0c134517601750d16dd7e69499dba6ac6bc204"}, "holdout": {"file_count": 126, "records_sha256": "7a740412ae86aa974d8f7e04e4c177fff4f05544a157388698e159aa38e45ccd", "manifest_sha256": "cb5e2efaa86d88b8c7c12a4161ae8bcb5d22f904421b846736d6f674291c458c"}}.

Baseline7passed3warnings7.28s; routerfocused4passed3.09s; router+referencefocused7passed2warnings6.36s; full278passed11skipped4warnings33.75s on server. Tests cover deterministic initialization/ties/zeros/minimum cluster boundary, no detector/oracle import, original mean/count gates, >=12routecoverage and empty-cluster diagnostics. Initial test incorrectly assumed package import excludes torch; existing taisp.__init__ imports ISP/torch. Corrected only that test to enforce the actual detector/oracle boundary; no implementation or task constraint changed. Candidate manifest download initially failed due missing newlocaldirectory; recreated exact JSON from SHA-verified cohort, no model rerun. Train archive SCP stalled at229376bytes; stopped only matching transfer PID54660, existing legacy fallback completed; full archive SHA24cb957225aa68e438bd0a260e45775fae55160b17ae3102dfc4786f22ff6ad9 and all365file hashes verified before mapscommit. No training rerun.

New code: taisp/analysis/gradient_regime_router.py, gradient_regime_reference.py; tests/test_gradient_regime_router.py, test_gradient_regime_reference.py; this stdlib report renderer under research_log/T032A. Existing R049 verifier/Procrustes/summary algebra and exact task-gradient loop reused; donor same repository under existing license. New logic is restricted to router, lock ordering, two experts and cluster diagnostics. Detector/ISP/tta/current-Ours/CLIP/deployment unchanged.

Report research_log/T032A_report.md; T032A/router.json,maps.json,summary.json,per_episode.tsv,integrity.json; frozen descriptors and allrawJSON/logs under research_log/remote_runs. Annotation subsets remain under remote project research_log/T032A, hashes pinned in cohort. No raw artifacts deleted. Future fresh selection additional_source T032A_train_cohort.json preserves3491 cumulative prior/reserved IDs plusval/evaluation.

Final disposition **close_exact_K2_two_O8_family**. Stop NEEDS_REVIEW. No alternative clustering/K, learned/soft/family routing, map tuning, CLIP/native inputs or AP selection authorized.
