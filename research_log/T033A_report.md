# T033-A / R051 — scientific FAIL; NEEDS_REVIEW

The exact confidence-pooled own-ROI -> PCA16 -> affine tangent-correction family fails the frozen source holdout conjunction and is closed without same-holdout rescue. No AP, finite-step, cross-detector or runtime result is claimed. T031, T032, T013 and memory branches remain closed and were not reused as learned models.

## Frozen inputs and ordering

Research R051 1d17ff9 / latest ae70654. Plan/cohort commit **4126a98** preceded new model calls. Candidate code/pins **b5d4c6fd024cc45c9a1c417be34709cd2532d58a**; numerical code **c0d70fc**; stage/reference code **13ecdd9**, all fixed before actual PCA/source fitting. Candidate600 commit **25d74e9142e58224d2589ac00be18872292606c5**, descriptor **dff5355**, BEFORE representation fit. Train-only representation/all600states commit **d0080523a423a90321129fbfe34cc29d46a259fa**, descriptor **fabdfda**, BEFORE source annotation/reference process. Train480/model commit **7976196999be1042452350c7ad7fc0cd9696735a** BEFORE any holdout task gradient. All120 corrected directions commit **21a0ee43bc012e4efb3ddcf225d559741d6fd5ed** and pre-oracle descriptor BEFORE the holdout reference process. Complete lock receipt: {"status": "directions_complete", "episodes": 120, "GT_loaded": false, "task_gradients_computed": 0, "candidate_lock_sha256": "9dd19b7c5df3bf1cb0ea5ac623a75dd451dbb64da7367a9ba8bb8306260fc872", "representation_lock_sha256": "f87461011ffc98a04f3bf03db694bc220a6a0789cefa0da5bab15d52342f4261", "model_commit": "7976196999be1042452350c7ad7fc0cd9696735a", "model_lock_sha256": "d3dcfa5a788c9e4ff06040dd805fba758356170baedc1fa1a8b929a6a458ceb0", "model_sha256": "79dc0866f76d1f0fcb4c22ff7aaa0d1016ae27b770a6258d0a8667be6db0251e", "AP_calls": 0, "directions_commit": "21a0ee43bc012e4efb3ddcf225d559741d6fd5ed", "directions_sha256": "cac18406ffc23a42b4d53a2d6384bb93926136d37b4fc1704e81caae3eb29786", "directions_lock_sha256": "83def049f69306bcbc3b14ad3ae61c0a4ed39da9196b6d597d75fe1fea425a89"}.

Fresh300 COCOtrain2017images, selectionseed20261004; excludes3491 prior/reserved IDs including T032, plus5000val and1400evaluation IDs. Existing eligible validbbox/readableJPEG selection sha256(seed:ID),ID; cyclic three established severity2families100pairs each. First240images train480episodes,80pairs/family; last60holdout120episodes,20pairs/family; four15pairblocks5/family. Clean then assignedcorruption perimage. CohortSHA690eadca95f37a40acc78f80284e55c802ea02a451ed8bed37e66aa99d5dd515; allimage/exclusion-set provenance hashes in cohort. Preparation parsed source annotations only for outcome-free eligibility and writing separate train/holdout annotation subsets. Candidate manifest has no annotation paths; candidate construction/representation/direction stage imports no oracle/reference/source-meta path. Actual train process opens only train annotations; holdout annotations only after committed corrected directions.

## Candidate and representation

Unchanged hard candidate calls existing orthogonal_candidates.candidate / original view_gradient, frozen score>=.50 stabletop20 confidence-weighted pseudo loss and 8-D identityISP. No support regeneration after source labels. The stored own-support boxes feed existing fixed_roi_representation at identityISP. Per-ROI1024-D float32feature is L2-normalized, detached-score weighted mean uses denominator sumscore+1e-12, pooled vector L2-normalized with eps1e-12. No predicted class identity, GT, family/case flags, memory, native losses or CLIP enter descriptor/state. PerROI rawfeature hashes, pooleddescriptor hash/vector/norm, exactsupports/hash,count,meanscore retained. Empty support=>zero1024descriptor/count0/score0. All600 candidates include3empty supports and3zero hard gradients, all retained.

PCA fits centered480x1024train descriptors only, onefloat64CPUthinSVD; numericalrank=477, threshold=1.2732997437130769e-12, top16 variance fraction=0.7352461848526368. Largest-absolute-loading (earliest tie) signpositive; no dimension/rank sweep. State27=[PCA16, g/(normg+1e-12)8, log(normg+1e-12),count/20,meanscore]. Train-only populationmean/std; constantdims=[] withscale1/forcedzero convention precommitted. All600states saved with individual float64-vector hashes before source labels. Fullbasis/spectrum/signindices/state means/scales in representation.json. NumPy1.26.4.

## One affine tangent fit

Train480 source taskgradients use unchanged oracle at identityISP, reference seed20260913 and existing common-JVP/reverse parity. u_h=g/(normg+1e-12),u_t=t/(normt+1e-12), target=u_t-dot(u_t,u_h)u_h; exactzero hard/task=>zerotarget, keepallrows. Onefloat64SVDminimum-norm affine fit X=[1,h_std], no ridge/weights/optimizer/search. Rank=28, ranktolerance=3.709766592833112e-12; singular spectrum=[34.80688217732233, 29.61617368330472, 28.362498686544928, 26.579347724773097, 26.415787924346585, 25.49848280167168, 23.98498602034068, 22.905214639391534, 22.175417388141515, 21.940758018785882, 21.908902300206663, 21.90890230020666, 21.90890230020666, 21.908902300206652, 21.90890230020664, 21.90890230020664, 21.90890230020664, 21.434710135864375, 21.035106218783735, 20.698831075241856, 20.000583698400973, 19.088971358097844, 18.548125638219567, 16.84944833242021, 12.218934179912333, 11.219241606124296, 7.8448951627513965, 1.8362676431904858e-06]; effectivecondition=18955233.626426008, fullcondition=18955233.626426008; coefficientFrobenius=485350.62770457513,Bnorm=485350.6277045686,bnorm=0.07946850143203116,trainresidualSSE=278.012365430567. Zero hard/task/target counts=3/0/3. All480train targets/predictions, b/B and fit diagnostics retained.

Application is fixed r_perp=r_hat-dot(r_hat,u_h)u_h;u_obj=(u_h+r_perp)/(norm(u_h+r_perp)+1e-12); exactzerohard=>zero/abstain. This exact eps-normalized formula is retained; projectiondot and finalnorm are diagnostics, not a new fitted correction or tolerance. All120 corrected directions computed and committed in a separate label-free process; reference consumes those exact vectors without recomputation. Maximum absolute projectiondot=2.1302502817288627e-09. Holdout abstentions=0, counted nonpositive.

## Execution and integrity

Candidate run20260914-191144-taisp-t033a-candidates600:600episodes/535.4537428139593s, sourceb5d4c6f,release20260914-191003-taisp-t033a-candidates. Representation run20260914-192538-taisp-t033a-representation, sourcedff5355. Train run20260914-192812-taisp-t033a-train480-model:480episodes/268.65787857800024s, sourcefabdfda. Directions run20260914-193512-taisp-t033a-preoracle120. Holdout run20260914-193723-taisp-t033a-holdout120:120episodes/70.2446526850108s. Postcandidate stages use identical frozen release20260914-192036-taisp-t033a-reference. Exact commands, commits, timestamps and exit statuses retained in raw run.sh/meta/train.log. Allmodel forwards/gradients onA6000CUDA0; CPUonlyPCA/affineSVD/summaries. Frozen detectorstate73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf andweight258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384 unchanged. No CUDA/RNG/seed/model/ISP setting changes.

All600candidate+480train+120holdout integrity/state/RNG/JVP checks pass; allhardvectors unchanged; train/holdoutdisjoint;alllocks verifiedbeforeannotations; savedu_obj consumedexactly. All72local/remote pinsmatch,including65protected/prior paths. Rawmanifests: {"candidate": {"files": 603, "manifest_sha256": "e360863bd90e86fcd7315877d75b95be437df392c61cf3cc319ac56fddd7b4cc"}, "representation": {"files": 3, "manifest_sha256": "050720ba8477e2ab5ba2a290c21821f084dee8fc295c12f5f64df71c6571501f"}, "train": {"files": 486, "manifest_sha256": "4cd09ab172f6f2080f43c606db605567bba702ef1d48a639c1c98b8efa5040e0"}, "directions": {"files": 2, "manifest_sha256": "92fb73ac81f1ae4d1a6230a46edd2a6877c2687dd39753121bc22a7f31360f85"}, "holdout": {"files": 125, "manifest_sha256": "896ba9465b05bbf3c595076023fcd1611eeb85703915bf1244528d8456ec774e"}}. ArtifactSHA receipt: {"candidate_records_sha256": "265451316d20725999d7117788f0874cff1faa617351e5320957b854cc5cb64a", "representation_sha256": "42970a5e12953de024a86dee466c13d73ad893374681001ef2848caf389016ed", "states_sha256": "e1a05feae334d1a31667c01ae65bc95ec6b1e12c5a95eac2d68896f4e10cefe4", "train_records_sha256": "5a2aeb4bda68f1aca2ba441e70b2e02e5ff0ff23fa31e2ae4032dd87642c42fb", "model_sha256": "79dc0866f76d1f0fcb4c22ff7aaa0d1016ae27b770a6258d0a8667be6db0251e", "directions_sha256": "cac18406ffc23a42b4d53a2d6384bb93926136d37b4fc1704e81caae3eb29786", "holdout_records_sha256": "3e76459fe88464f2d4797df0e14e092de4ac23977a1fe34b54c576f8def05d2c"}.

## Exact R051 gate

S_hard=dot(task,hard)/(normhard+1e-12);S_obj=dot(task,u_obj);Delta=S_obj-S_hard. Required S_objpositive82/120overall45/60corrupt;Deltapositive72/120overall36/60corrupt;positive mean+medianoverall/corrupt;>=3/4positiveblockmedians;cleanmedian>=0;finite/zeroabstention/integrity. Summary S_cal/g_cal/cos_cal_task aliases refer only to S_obj/u_obj/object-corrected cosine. No oldmap comparator.

|Gate|Pass|
|---|---|
|Delta_overall|False|
|Delta_corrupt|False|
|median_overall|True|
|median_corrupt|True|
|mean_overall|True|
|mean_corrupt|True|
|positive_blocks|False|
|median_clean|True|
|integrity|True|
|S_obj_overall|False|
|S_obj_corrupt|False|
|finite_zero_integrity|True|

|Group|N|S_hard>0|S_obj>0|Delta>0|MedianDelta|MeanDelta|Median cos(h,task)|Median cos(obj,task)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|overall|120|58|60|65|0.000723623151379132|0.00042972172600682246|-0.010801368053622612|0.006130770421784245|
|clean|60|26|26|33|0.000723623151379132|-0.0003708873935141803|-0.05011099960543351|-0.09506729549991755|
|corrupt|60|32|34|32|0.0006816998436826177|0.0012303308455278252|0.11998508924588766|0.09524619585534483|
|clean_s0|60|26|26|33|0.000723623151379132|-0.0003708873935141803|-0.05011099960543351|-0.09506729549991755|
|color_cast_s2|20|11|12|10|-0.00027035559716054834|-0.0007052194406223528|0.22737900183967227|0.2596286388020921|
|contrast_s2|20|12|12|10|-1.1742394687195385e-05|-0.002083890767219383|0.19921302540842037|0.15416019761295352|
|gamma_s2|20|9|10|12|0.0015327523452159132|0.006480102744425211|-0.08606272212858919|-0.07307066087739512|
|block0|30|14|14|21|0.001923482383748714|0.006254950462799177|-0.2207851963116435|-0.18095370854084025|
|block1|30|14|15|14|-0.00038386191869355796|1.6152513575247333e-05|-0.0038450254104919965|0.006130770421784245|
|block2|30|12|13|12|-0.001460136473717441|-0.006117031443850666|-0.1319984649844188|-0.14296996940148596|
|block3|30|18|18|18|0.0030631854966106003|0.001564815371503531|0.3990398806616972|0.4283768246928158|
|support_empty|0|0|0|0|None|None|None|None|
|support_nonempty|120|58|60|65|0.000723623151379132|0.00042972172600682246|-0.010801368053622612|0.006130770421784245|

Outliers and empty/zero episodes remain in totals; undefined zero cosine is None, not an invented direction. Full distributions and per-episode TSV retained.

|Group|Gradient|Max-energy coordinate|Max share|
|---|---|---:|---:|
|overall|g_hard|2|0.28569598154735354|
|overall|g_cal|2|0.2864676305186043|
|clean|g_hard|2|0.282626690183213|
|clean|g_cal|2|0.29684344184964556|
|corrupt|g_hard|2|0.28804563154253476|
|corrupt|g_cal|2|0.2760918191875629|
|clean_s0|g_hard|2|0.282626690183213|
|clean_s0|g_cal|2|0.29684344184964556|
|color_cast_s2|g_hard|1|0.39835175079591106|
|color_cast_s2|g_cal|1|0.33593178018333303|
|contrast_s2|g_hard|2|0.2979266641554666|
|contrast_s2|g_cal|2|0.2966575731405998|
|gamma_s2|g_hard|1|0.3325401083131684|
|gamma_s2|g_cal|2|0.2475855204113496|
|block0|g_hard|2|0.39986576822775116|
|block0|g_cal|2|0.2595816334445046|
|block1|g_hard|2|0.27424932823042875|
|block1|g_cal|2|0.2982552988898222|
|block2|g_hard|1|0.3132152497064908|
|block2|g_cal|2|0.29702521632472023|
|block3|g_hard|1|0.3113112706636989|
|block3|g_cal|2|0.29100837341537006|
|support_nonempty|g_hard|2|0.28569598154735354|
|support_nonempty|u_obj|2|0.2864676305186043|

Energy shares compare rawhard to unit corrected direction and are descriptive only, not matched-norm performance. Percoordinate mean/std/meanabs/RMS in summary. No subgroup-selected rescue.

## Validation, changes and recovery

Baseline5passed3warnings5.12s; candidate+hardfocused5passed3warnings7.64s; mathfocused3passed1.61s; stage+mathfocused5passed5.45s; full286passed11skipped4warnings38.81s, completed before actual representation/source outcomes. Tests cover normalized confidence pool, empty supports, class irrelevance, original hard reuse, PCA sign/rankcollapse/train-onlystandardization, one affineSVD/minimumnorm, tangent andzero formulas,82positive threshold, abstention accounting, meanoutlier andlazyoracle boundary. Real600+480+120 serves as end-to-end integration. No outcome-driven core code change or extra tests after frozen code.

New analysis object_state_candidates/math/stages/reference modules and focusedtests; project-local report/plan/cohort/codepins/locks/rawreceipts. Existing detector/ISP/tta/current-Ours/CLIP andpastanalysis protected. AP0/Kstep0/no runtime changes. Source-only fit, deployment-visible features, separateholdoutdirectioncommit all preserved.

Local Git auto-GC filled D disk before mathcommit; it exited with out-of-space, leaving two read-only incomplete temp packs. Normal deletion failed; automaticapprovalreview rejected force deletion with 'blocked by policy', so no alternate deletion was attempted. Available disk space subsequently sufficed for small commits; one-off gc.auto=0/maintenance.auto=false avoided recurrence without changing repo config. All experiment archives/logs preserved; no model rerun. Project-local progress and .autodl receipts preserve this recovery history.

Report research_log/T033A_report.md; representation/model/directions/summary/integrity/per_episode.tsv andallrawruns retained. Annotation subsets stay underremoteproject research_log/T033A withcohortSHA pins. Futurefreshadditional_source T033A_train_cohort.json preserves3791cumulativeprior/reserved plusval/evaluation.

Final **close_exact_pooled_ROI_PCA16_affine_tangent_family**, NEEDS_REVIEW. No pooling/PCA/class/CLIP/native/ridge/MLP/threshold rescue; no runtime/AP extension authorized.
