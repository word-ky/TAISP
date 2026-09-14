[2026-09-12 00:17:55 +08:00] Checked workspace D:\work\fightccfa-agin\CVPR2027\TTT-ISP: initially empty, not a Git repository. GitHub word-ky/TAISP webpage reports public empty repository. git ls-remote exit=0. No clone, remote configuration, or push performed.
[2026-09-12 00:27:12 +08:00] T001 stage 1: PyTorch 2.13.0+cpu / Python 3.12.7 / pytest 9.1.1 / PyYAML 6.0.3. ISP tests: 4 passed in 15.55s; identity, bounds, finite black gradients, gradcheck for every coordinate, state_dict roundtrip. User additionally authorized A6000 project setup and experiments.
[2026-09-12 00:29:40 +08:00] T001 stage 2: loss/predictor tests 4 passed in 19.11s. Semantic mock uses negative directional projection, finite at identity; no claim of CLIP equivalence. Stage 1 commit 60853a6 pushed to origin/main.
[2026-09-12 00:32:09 +08:00] T001 stage 3: python -m pytest tests/test_adapt.py -q -> 5 passed in 12.94s. A6000 connection succeeded via existing workflow; two idle RTX A6000 49140 MiB. Created /home/liujianhua/wjq/TAISP/research_log under current writable account. No other projects modified.
[2026-09-12 00:35:58 +08:00] Integrated research R001 (2cfbbbb) with merge aa2e2c2. Added per-step raw/physical state, signed and absolute coordinate gradients, output saturation fraction. 7 adaptation tests passed in 9.57s. Fully clipped image reproduces zero gradient (reported, not hidden). Predictor outer gradient passes. Remote venv created with existing system torch plus pytest 9.1.1 and PyYAML 6.0.3.
[2026-09-12 00:37:43 +08:00] Local full suite: 17 passed, 1 CUDA skip in 8.81s. CPU demo seed 42: total -1.0323826415969961e-07 -> -0.44069603085517883; gamma 0.77050513; brightness 0.09697855; no repeatedly near-zero (<1e-8) coordinate. Remote Python 3.12.12 / torch 2.4.0+cu121 / CUDA available / pytest 9.1.1 / PyYAML 6.0.3. Preparing deployment; SSH config kept in ignored project .autodl/config.json.
[2026-09-12 00:41:08 +08:00] T001 DONE: A6000 run 20260912-003812-taisp-t001-a6000 exit 0, 18 passed in 3.50s; CUDA demo total -5.1619135632563484e-8 -> -0.44069597125053406. Full receipts fetched under research_log/remote_runs; final report and handoff written. Editable package install/import passed. Sharpening's weak mock gradient and fully saturated zero-gradient case disclosed. Research acceptance pending.
[2026-09-12 00:54:52 +08:00] Project recheck: fast-forwarded to 19f362d. Research review R002 formally ACCEPTED/CLOSED T001; T002 real CLIP + frozen detector + fixed COCO-subset gradient-alignment feasibility study is TODO. Created active thread heartbeat automation id=taisp every 15 minutes, checking changes and reporting meaningful events only. No T002 experiments launched by this monitoring request.
[2026-09-12 01:02:14 +08:00] User authorized direct execution of current/future queued tasks. Heartbeat taisp updated accordingly. T002 started; T001 baseline regression 17 passed, 1 CUDA skip in 26.71s. Pre-result protocol recorded in T002_plan.md. Real CLIP module added; official COCO data preparation run 20260912-010134-taisp-t002-data launched.
[2026-09-12 01:06:27 +08:00] T002 real CLIP gate: 3 passed in 3.09s on A6000 (run 20260912-010431-taisp-t002-clip-check). CUDA antialiased bicubic backward nondeterminism identified by deterministic_algorithms; exact episode initial state retained with numerical update comparison. Reused full cached COCO annotations from existing WeDetect source after slow official download; original data download stopped, new run 20260912-010501-taisp-t002-data-cached.
[2026-09-12 01:10:04 +08:00] T002 detector gate: 2 passed in 4.12s (A6000 run 20260912-010621-taisp-t002-detector-tests). Native oracle losses have finite nonzero phi gradients; parameters/buffers and sampling RNG unchanged; eval inference works. Fixed 200-image dataset prepared in shared/coco200/subset.json.
[2026-09-12 01:12:50 +08:00] T002 full real-model suite and 2-image/12-corruption study smoke succeeded, run 20260912-011042-taisp-t002-study-smoke exit 0. Corruption equations and official evaluator perfect/empty cases included. No hyperparameter changes from pre-result plan; proceeding to fixed 200-image full study.
[2026-09-12 01:14:45 +08:00] Full T002 run started: 20260912-011328-taisp-t002-coco200, release 20260912-011324-taisp-t002-full, source 4922ffe. 200 images, 1200 corruption samples; fixed protocol unchanged. Prior real suite 25 passed in 6.17s and 2-image smoke exit 0 (5.60s experiment). Live recovery instructions in T002_handoff.md.
[2026-09-12 01:18:55 +08:00] Full T002 attempt 20260912-011328-taisp-t002-coco200 failed after 42 images at COCO 105335 (612x612): int(612*(224/612))=223 led to CLIP patch/position mismatch 37 vs 50. Failed run preserved locally. Small repair uses exact integer shortest-side resize; added observed-shape gradient regression. No sample exclusion or scientific-setting change. Fresh A6000 full-suite run 20260912-011821-taisp-t002-regression.
[2026-09-12 01:19:58 +08:00] Post-fix full real suite: 26 passed in 5.89s (20260912-011821-taisp-t002-regression). Restarted full200 run 20260912-011915-taisp-t002-coco200-fixed; source c639969; release 20260912-011912-taisp-t002-full-fixed. Same IDs/config; no exclusions/tuning.
[2026-09-12 01:27:13 +08:00] Synced R003 checkpoint d6fd863: approved implementation and unchanged current run. Required final additions: first-order dot-product prediction vs observed loss (sign/correlations/distributions/CI), plus forward-only HF processor parity including odd/rectangular/612-square shapes. Execute directly; no scientific hyperparameter tuning.
[2026-09-12 01:30:53 +08:00] R003 parity-before (20260912-012847-taisp-t002-parity-before) found material one-pixel odd-center crop mismatch: candidate round vs pinned HF floor; max normalized geometry error 2.8535867 and RMSE 0.7695012 on diagnostic content. Completed run 20260912-011915-taisp-t002-coco200-fixed is PRELIMINARY per R003, preserved. Correct crop floor only; same scientific settings/IDs will be rerun after parity/full-test pass.
[2026-09-12 01:32:12 +08:00] R003 corrected parity passes on 14 natural/noise shape cases: geometry error exactly 0; max normalized RMSE 0.005910955 < declared 0.02. Full A6000 suite 27 passed in 6.04s, run 20260912-013058-taisp-t002-parity-after. Completed previous200 run marked PRELIMINARY, all receipts preserved. Preparing final unchanged-protocol rerun with pinned HF crop geometry.
[2026-09-12 01:36:30 +08:00] Final R003-corrected run active: 20260912-013248-taisp-t002-coco200-final; source0b8a888; release20260912-013244-taisp-t002-final-parity. Postprocessing Taylor/cluster-bootstrap helper implemented and known-relation test passed (1 in3.05s); preliminary data used only to verify renderer/analysis execution, marked superseded. No runtime protocol/hyperparameter changes.
[2026-09-12 01:46:19 +08:00] T002 final run 20260912-013248-taisp-t002-coco200-final exit0: 200 images/1200 observations,499.2096s. R003 postprocessing complete; overall cosine0.04530, positive alignment53.83%, semantic step detector-loss decrease47.58%; Taylor sign agreement54.75% (CI52.00-57.25), Spearman0.153 (CI0.094-0.213). AP mixed; no meta-training recommendation. Full report/receipts prepared; status NEEDS_REVIEW.

[2026-09-12 02:56:10 +08:00] Synced R004 edf75c0: T002 accepted/closed; T003 directly started. Baseline A6000 suite27 passed6.24s; local12 passed7.43s. Fixed prompts/temperature0.05/masks/comparisons recorded in T003_plan.md before new experiments.

[2026-09-12 02:59:04 +08:00] T003 Stage A complete from1200 saved T002 observations; hand-computed decomposition test1 passed0.38s. Overall outside-subspace norm0.8244, outside energy0.7205. Full per-coordinate and joint saturation strata saved research_log/T003_stage_a/coordinates.{json,md}; no GPU rerun.

[2026-09-12 03:02:23 +08:00] Conditioner/gate real A6000 suite32 passed8.18s (run20260912-025949-taisp-t003-conditioner-tests,release20260912-025945-taisp-t003-conditioner,source2416208). Deployment weights/gates detached; exact frozen-state and fresh-episode tests pass. Analysis driver reuses T002 initial oracle gradients and generic receipts, with per-case two-image numerical equivalence checks in smoke before full reuse.

[2026-09-12 03:04:21 +08:00] T003 smoke20260912-030241 failed after33 real tests passed: initial detector gradient cross-run reuse mismatch maxabs0.0001080167 (7/8 coords), though loss check passed. Investigating cold/repeated/post-clean forward before deciding repair; no tolerance widening or hyperparameter change. Smoke command source-revision label was mistyped35d074b; actual release sourcec8ac8ff verified run_t003.py SHA25672b18c1bea1c972d019fc66fecb8c725ddfef61c5b3647032d76e9de113200d8. Raw logs preserved; full run will use git-derived revision.

[2026-09-12 03:07:06 +08:00] Paired smoke20260912-030559:33 tests passed, then old generic phi3 cache equality failed maxabs6.2841e-7 on real COCO. Known nondeterministic CUDA backward accumulates across3 steps more than toy test; retain cross-run differences as audit, not equality prerequisite, since T003 now reruns generic and uses fresh within-run pairing. No equality tolerance changed, scientific settings unchanged.

[2026-09-12 03:10:57 +08:00] T003 fresh smoke20260912-030712 exit0:33 real tests passed7.28s;2 images/72 variant observations; all AP evaluations complete24.139s. Report pipeline passed on smoke, paired-bootstrap known-key test1 passed1.38s. Small fetch ordering error (local report started before SCP finished) resolved by waiting for completed transfer; no code change. Ready for same fixed200-image run.

[2026-09-12 03:13:19 +08:00] Full T003 run20260912-031123-taisp-t003-coco200 launched,release20260912-031119-taisp-t003-full,source8d8913e. Fresh A6000 suite34 passed7.79s;first6/200 images complete57.2s. Target7200 variant observations and72 AP evaluations. Active run and next postprocessing in T003_handoff.md; notes/state mirrored remote. Do not duplicate; next heartbeat monitors and collects when complete.

[2026-09-12 03:46:31 +08:00] Synced R00543fa056 checkpoint approval. Extended reporting only: paired positive alignment/saturation/latency, within-run AP differences, separate raw/effective coordinate energy and norms. Two focused analysis tests passed1.18s; new report rendered on smoke. Full T003 run20260912-031123 finished200 images/7200 observations at03:45:51+08,exit0. Fetching all raw receipts before final postprocessing.

[2026-09-12 03:53:16 +08:00] T003 final postprocessing complete:200 images/7200 observations,72 AP evaluations,2049.095s,exit0. R005 paired statistics/raw-effective analysis saved. Soft mean cosine0.045298 vs generic0.045314;oracle-both0.093684,benefit50.583% vs46.667%;AP heterogeneous. Full report and mailbox written,NEEDS_REVIEW,no active run,no meta-training. Raw120094186-byte JSONL compressed losslessly for GitHub single-file limit (37666421 bytes,round-trip verified);raw retained local/remote. Receipt audit counts/phi0/sharedgdet/gate equation passed;APfigure visually verified.

[2026-09-12 05:01:01 +08:00] R006cfdec6e accepted/closed T003; T004 started directly. Local baseline9 passed10.77s;A6000 baseline34 passed7.69s. Inspected pinned HF vision forward: last_hidden_state before post_layernorm, pooled CLS postnorm. T004_plan.md locks last-block projected normalized patch tokens, score0.5/top20 region overlap weighting and all six comparisons before new experiments.

[2026-09-12 05:03:09 +08:00] T004 feature gate: A6000 patch/preprocess/adapt15 passed5.23s,run20260912-050140-taisp-t004-patch-tests. Last hidden patches postnorm/projected match direct model equation, gradients finite, frozen-state/reset/odd-shape tests green. Region overlap mapper added next with original-only inference and predeclared fallback.

[2026-09-12 05:11:14 +08:00] T004 full real-model suite42 passed8.52s;2-image/72-observation/all-evaluation smoke20260912-050701 exit0,27.5378s. Report pipeline passed on smoke; analysis-related3 tests passed5.89s. Smoke receipt checks phi0,patch weight sums,norm-matched target norms pass. No failures or scientific-setting changes. Preparing fixed200 run.

[2026-09-12T05:14:35+08:00] T004 full run20260912-051214-taisp-t004-coco200 active;release20260912-051210-taisp-t004-full,source4817825. Fresh42 real tests passed8.17s,first4/200 images52.3s. Recovery in T004_handoff.md. Memory reporting explicitly scoped to adaptation-phase peak; original detector inference peak not separately recorded; no run change.

[2026-09-12T06:04:16+08:00] T004 full run20260912-051214-taisp-t004-coco200 finished200 images/7200 observations/72 AP evaluations at05:56:05+08,exit0,2612.53855s. All raw receipts fetched; count/zero-phi0/shared-gdet/norm-match/patch-weight audit passed. Raw JSONL96709726 bytes fits GitHub limit and is retained without compression. Final paired analysis running.

[2026-09-12T06:06:49+08:00] T004 postprocessing added Stage C matched-text global controls for all local variants (generic/global_generic and oracle/global_oracle, primary and norm-matched). This separates privileged direction choice from visual representation; no data/model/protocol change. Three focused paired/norm/partition tests passed6.85s. Recomputing report; no experiment rerun.

[2026-09-12T06:11:41+08:00] T004 final report complete,NEEDS_REVIEW. Generic/patch-generic/region-generic mean cos0.04531/0.01529/0.02003; no supported paired alignment improvement. Norm-matched region-oracle benefit51.50%,+4.33pp vs global-generic, but same-text global-oracle control+1.58pp CI[-1.92,5.08]. Mixed AP/negative families retained. Full report appended to mailbox,42 real tests and3 analysis tests pass. Raw7200 rows96.7MB with SHA; figure inspected. No active run or meta-training.

[2026-09-12T06:37:22+08:00] R00724114d2 accepted T004; T005 started directly. Baseline11 local passed6.38s,42 remote passed8.23s. Protocol fixed in T005_plan.md before smoke. ROI5 tests passed4.67s. Loss GPU repeat assertion failed maxabs1.49682e-5 (14pass/1fail); same tolerance moved to real CPU repeat, GPU gradient/freeze checks retained. Active focused run20260912-063641-taisp-t005-repeat-tests; no full study launched.

[2026-09-12T06:45:10+08:00] T005 fixed support/loss gate15passed81.40s (real CPU repeat same1e-6, GPU gradients/freeze). Smoke20260912-064017 finished2images56observations/all63APevals,exit0; full suite50passed. Fetching receipts; reporting focused4tests passed11.02s. No scientific changes/tuning.

[2026-09-12T06:48:04+08:00] T005 smoke validated:50real tests passed87.80s,2images56rows63APevals27.37618s,exit0. Raw phi0/sharedgdet/weights/normtarget audit passed; no empty support among these2images (explicit unit no-update tests pass). Report pipeline and4focused report tests pass; figure label overlap fixed by aspect auto. Ready for fixed200run, no scientific changes.

[2026-09-12T06:49:43+08:00] Launched T005 full run20260912-064836-taisp-t005-coco200,release20260912-064831-taisp-t005-full,source9fb96c1. Fresh51tests before fixed200study withclean control. Active recovery in T005_handoff.md; no duplicate/tuning.

[2026-09-12T06:51:12+08:00] T005 full run20260912-064836 passed51real tests85.36s;first2/200images complete26.0s, no blocking error. Active state mirrored remote. No intervention/tuning/duplicate.

[2026-09-12T07:38:49+08:00] T005 full run20260912-064836 completed200images/5600observations at07:33:51+08,exit0. No active experiment. Fetching packed raw receipts for final paired/clean/support analysis.

[2026-09-12T07:44:02+08:00] T005 final paired/clean/support analysis complete,NEEDS_REVIEW. Mean cosine0.04532/0.11757/0.12431/0.12271; allnative paired cosine CIs positive. Raw benefit46.83/50.25/50.00/50.17%, differencesCIcross0; norm-matched improvements3.17-3.75pp positive exploratoryCIs. APmixed; cleanAPpositive but largerphi; stable/JS incremental benefit unsupported. Full rawauditpassed5600rows63metrics,fallbackzero. Raw50817303bytesSHA87554e26f6452d2063e92a50fb27d4cb9dcd5931dfd62942f48b6fac47e7bd87. Report/figure complete, no active run or meta-training.

[2026-09-12T07:45:09+08:00] Final report remote mirror tar initially refused writing through research_log/remote_runs symlink (invalid cross-device link). Re-extracted generated artifacts to actual runs target via archive path transform; succeeded. Raw SHA matches local. No experiment/data change.

[2026-09-12T09:06:34+08:00] Synced R00806c1541 andLATEST864b0ee: T005accepted/closed,T006started directly. Predeclared FCOS independenttarget, sameISP/sourcepseudo/CLIPbaseline, severity/clean controls. Baseline51test running; initialFCOSdownload failed PythonSSLCA, retrywithsystem CA bundle withoutdisablingverification. No modelreplacement or fullstudy.

[2026-09-12T09:08:37+08:00] T006 FCOS weights obtained usingSSL_CERT_FILE systemCA; oneSSHtimeout retried. Exacthash/defaults inT006_fcos_pin.json; installedFCOSforwardinspected. Baseline51passed84.76s. Addedanalysis-onlyloader/oracle andstrict source-with/withouttarget CPUisolation test; GPUfreeze/gradient checks pending.

[2026-09-12T09:14:59+08:00] T006 targetoracle/sourceisolation4tests passed20.60s,run20260912-090851. Sourceadaptation absent/presenttarget exactCPUequal; targetnativeGPUgradientfinite/frozen/RNGunchanged. Smoke20260912-091109 finished2images28rows/all70APevals,exit0at09:13:17+08. Report/diagnostic3localtests passed4.49s (1realmodelskip); fetchingrawsmoke forfinalaudit beforefullstudy. No scientificchange.

[2026-09-12T09:23:17+08:00] T006 smoke receipt audit passed: 53 real tests in100.40s; 2images/28rows/70APevals in16.28047s. Both annotated gradients/losses shared, phi0zero, CLIP norm reference, signed products verified. Raw226866bytes SHA c7c7a8c79231f517d91e81d050423419fdb119356a14ade8e464d213d56de4d8. Default SFTP download stalled at229376bytes; stopped only the matching scp transfer and existing workflow retried legacy SCP, exit0; archive295285bytes SHA745b8c8e0ac1aa105dea5b5edaf0577c58d6a98c33e127eb5538abcebce04e85 verified. Report and figure inspected. No scientific settings change. Ready for fixed fullrun.

[2026-09-12T09:25:30+08:00] T006 full run launched 20260912-092401-taisp-t006-coco200,release20260912-092347-taisp-t006-full,sourced7d0510 pushedmain. Fresh54tests precede fixed200study. Heartbeat preserved15min and updated to read LATEST continuation. Active recovery saved; awaiting fulltest/firstimage verification.

[2026-09-12T09:26:39+08:00] Fullrun20260912-092401-taisp-t006-coco200 passed54real tests102.35s (7knownwarnings); first2/200images completed15.1s, no blocking error. Experiment remains active; next heartbeat monitors existingrun and collects only after completion. Rootproject logs mirrored remote.

[2026-09-12T09:58:40+08:00] T006 full run20260912-092401-taisp-t006-coco200 completed200images/2800rows at09:51:22+08,exit0. No active experiment. Fetching complete receipts for paired transfer/clean/severity analysis. First packaging SSHconnection closed(exit255), retry succeeded; no model/data rerun. Archive31081757bytes SHA1c121d5904a0df5dd5b1a72d3e658752dbbfa9343a8d69be83c3c95d75071553.

[2026-09-12T10:06:42+08:00] T006 finalreport complete,NEEDS_REVIEW. All2800rows/70AP auditpassed;FCOS pairedcos+0.093CIpositive,rawbenefit+5.42ppCIpositive,matched+7.25ppCIpositive. FCOSAP3vsCLIP5/6positive butvsbefore3/6positive;rawmeanloss notimproved,matchedrelativeimproved. No severity-only orrobustrestoration claim. Reportassembly optionalcachepathcomparison narrowedtocorresponding pinnedfields(allmatch);GBKread fixedexplicitUTF8. Allnegative/fallbacks retained,figureinspected,no activeexperiment/meta.

[2026-09-12T10:48:56+08:00] R009 a8b5475 acceptedT006;T007started directly. Disjointmanifest55e83e0 overlap0 committed/pushed before newdatasetmodelexecution. Hybrid/driver4e87904,local4tests13.39s,report1test1.19s. Real smoke20260912-104530:58tests126.41s passed,firstimagecomplete. Awaitsmoke/reportaudit beforefull. No failures/scientificchanges.

[2026-09-12T10:50:34+08:00] T007 smoke20260912-104530 completed2images42rows98APevals35.857011s,exit0at10:48:25+08;58realtests126.41s. Receipt auditpassed samecommitteddisjointmanifest,sharedsource/targetgrads/support,phi0,every-stepnormtransfer/updateequation. Same-forwardcollinearityerror9.99e-16; independentraw-vshybridcosmaxdiff.00013274 fromseparateCUDAforwards,not directiongain. Report/figurechecked,allnegativecasesretained,no parameterchange. Readyforfixednew200.

[2026-09-12T10:52:27+08:00] LaunchedT007fullrun20260912-105119-taisp-t007-coco200,release20260912-105102-taisp-t007-full,source0df7e13 aftervalidatedsmoke. Priorfull deploy105051connectionclosed duringupload/extraction; retrysucceeded, noexperimentfromfailedrelease. Fresh59tests beforefixeddisjoint200;handoffrecordsactualrun.

[2026-09-12T10:54:02+08:00] Fullrun20260912-105119-taisp-t007-coco200 passed59realtests127.00s(8knownwarnings). Study nowloading frozenmodels;no blockingerror,no duplicate.

[2026-09-12T11:43:01+08:00] T007 fullrun20260912-105119-taisp-t007-coco200 completed200images/4200rows at11:35:47+08,exit0. No activeexperiment;fetching fullreceipts forfinal audit/report. Protocolunchanged.

[2026-09-12T11:53:11+08:00] T007 finalreport/audit complete,NEEDS_REVIEW. 4200rows98APallverified;partialscalebenefit: FCOSbenefit+3.33ppCIpositivevsraw,butmeanlosscontrastCIcross0;AP3vsraw6/6positive,vsnoadapt4/6positive;cleanmeanphi3-46.53%yet57/200largerthanraw. No all-threeconfirmation/meta. Raw35250130bytesSHAfd41b75af58e6b8eaffdc79405ee2cd604acf85505df9d402f4c9ea2c1d2f20a. FirstpostrunpackagingSSHtimeout retried,completearchivehashverified. Figure/linkschecked,allnegativeandfallbacksretained.

[2026-09-12T13:06:59+08:00] R010acceptedT007;T008offline diagnosis started. Plancommita85e3b3. Matching4tests pass5.21s; bootstrap matrixmul causednativePythonabortexit3, replacedwithmathematicallyequivalent einsum;6tests nowpass5.60s. Savedtestreceipt. Existingannotationsfetched;no newadaptation/modelinference ordeployment edits.


## 2026-09-12T13:21:40+08:00 — T008 offline diagnosis complete

Plan a85e3b3; code cff2193/38627d4. Reconstructed all98 retained predictionfiles,19600proxyrows and22400pairedrows. Corrected observed undefined-GT category denominators; final7focusedtests passed4.23s. Offline computation/report/plot exit0; no model inference. Audit2816quadrantpartitions and uniquejoins passed; figure inspected. BLASabort and plottingOpenMPcollision fixed without changing statistical settings or enabling unsafe runtime override.

Full report T008_report.md and raw/CSV/figures in T008/. T008NEEDS_REVIEW: FP component ofhybrid-vs-raw gain, sourceconfidence/FPinflation,cleanperturbation; targetAPnegativeconditionsmixed/unclear. No nextmethod/newexperiment. Preparing GitHubpush andA6000mirror.

## 2026-09-12T13:23:30+08:00 — T008 delivered

Report/data commit702e5da pushed toGitHubmain. A6000archiveSHA256 9a5d95f35f9926bceffd227bd498d69f35395766f629d7637e0d92b8500f962b matched; remoteanalysis/proxies/pairedhashes and19600/22400rows verified. Durabledata under/home/liujianhua/wjq/TAISP/research_log/T008. Existingtaispheartbeat confirmedACTIVE every15min; no duplicateautomation. No activeexperiment; awaitR011/researchreview.

## 2026-09-12T14:14:29+08:00 — R011/T009 started

Fetched184a3a0/5885716. T008accepted/closed. Baseline3tests pass10.74s,1real skip; cohort extension2tests pass0.57s. Preparing1000 disjoint IDs/JPEGhashes and5random-order blocks before any T009model execution. PlanT009_plan.md; no adaptation changes.

## 2026-09-12T14:24:35+08:00 — T009 evaluation plumbing

SSD exactinstalledtorchvision defaults inspected; evaluation-only adapter and realcompatibilitytest prepared,notexecuted beforecohortcommit. FixedblockCOCOeval/macrotest localfailed due missingpycocotools; server4tests passed1.52s in plumbingrelease20260912-141922. K3driver andreport reuseacceptedadapt/evaluator functions; staticcompilepass. Download750/1000 ongoing,no model execution.

## 2026-09-12T14:28:10+08:00 — T009 cohort pinned

1000images downloaded exit0; manifest e98a7fd pushed before any T009model execution. SHA155bb6f047d374342623f48488bcb2b33601a4ecbd372976a433c1b289546e49; zerooverlap with historical200/T007200,exact5random-order200blocks. Preparing SSDrealcompatibility. An initial local record/commit command ran from workflow cwd and failed missingproject paths without creating a commit; corrected to projectcwd. Deployment used explicit correct project source.

## 2026-09-12T14:29:11+08:00 — SSD compatibility running

Release20260912-142737-taisp-t009-ssd-compat,run20260912-142824-taisp-t009-ssd-compat,sourceb492c9d. Exact SSD COCO_V1; real historical-image CPUhybrid K3 absence/presence comparison and GPU SSD repeat/mapping/freeze/COCOeval. Fullcohort notrun.

## 2026-09-12T14:33:39+08:00 — SSD compatibility passed

Retry143109:1passed66.32s,4knownwarnings,exit0. ExactCOCO_V1modelhash b556d3b43ab6c3f63d81bfb8835fe8756ac22da664357da100dccf96b6a6b42d,metadata T009_ssd_pin.json; targetisolated and source support/phi/enhanced unchanged. Commitpin before smoke/fullstudy; nextfreshregression plus2imagesmoke.

## 2026-09-12T14:35:42+08:00 — T009 smoke/regression started

Run20260912-143436-taisp-t009-study-smoke,release143351,sourcebe5c797. Fullfreshsuite then2imageK3three-detector/APblocksmoke/report. SSDpinbe5c797 andcohorte98a7fd precede study; exactmethodunchanged.

## 2026-09-12T14:43:08+08:00 — T009 regression/smoke green

Run20260912-143436,sourcebe5c797:70real-model/regressiontests passed183.95s(9knownwarnings);2images42rows84predictionfiles252APevals16.453258s,exit0at14:38:14. Manifest/modelpins/sharedsupport/phi0/all3updates/hybridnormtransfer/terminalno-update audited; maxreconstructednormtransferroundingerror1.31e-8. Smoke signs retained,notusedfortuning. Report-onlypartialcohortcriterion now unassessed(null),figureexplicitlySMOKE; regeneratedreport/plot exit0 andvisualchecked. No adaptation/driver changes after fullsuite; preparingfull1000.

## 2026-09-12T14:46:53+08:00 — T009 full study started

Run20260912-144439-taisp-t009-coco1000,release20260912-144331-taisp-t009-full,source/report69bfb66. Formal1000cohort/fiveblocks,3detectors7conditions4methodsK3. Earlier70tests+2imagesmoke passed; driver/adaptationunchanged. First6/1000normal (~35.7s); no activeblockingerror. RemoteMatplotlibabsent,finalplotwilluseexistinglocalruntime; no remotepackageupgrade. Monitoring via15minheartbeat; fullscientificresultspending.

## 2026-09-12T14:48:12+08:00 — T009 launch handoff synchronized

Active-runhandoffcad6205 pushed and mirrored under A6000 projectrootresearch_log/coordination. Verifiedactual started_at14:44:47+08; latest27/1000images elapsed156.2s,normal. Launchreceipt research_log/T009/launch_receipt.json. No finalscientificresultyet; scheduledmonitorcontinues.

## 2026-09-12T16:26:00+08:00 — T009 inference complete, official AP running

Active run20260912-144439 completed1000/1000inference at5913.0s. Exactly21000sample rows and84predictionfiles saved; source/target/SSD aggregate+5block COCOeval stillrunning. Packaging immutable input/trajectory/prediction receipts whileevaluation continues; no protocolchanges or scientificconclusion frompartialmetrics.

## 2026-09-12T16:36:30+08:00 — T009 raw receipt audit passed

Remote audit scripts/audit_t009_raw.py exit0:1000images21000unique rows7000sharedsupport84predictionfiles; cohort/model/config pins match,allphi0/allthree updates/hybridscale verified. Maxphi rounding9.39e-8,maxtransfer5.60e-8. Samples125507835bytes SHAcdbe9cf51feb17dc5ec2e60f0ea396227f4b4af301b2f6dc135f7382f0e8910c. Receipt fetched toT009/receipt_audit.json. APstillrunning. Rawarchive download session9887 active; no duplicate inference.

## 2026-09-12T16:47:00+08:00 — T009 formal completion and audited report

Formalrun20260912-144439 exit0at16:41:56;1000images21000rows84predictions504APevals;driver7011.39484s. Rawintegritycheckedremotely;finalderivedarchivehashverifiedlocally. Independentlyreconstructed4536metricdeltas/allmacrocounts. PlotPNG/PDFgeneratedlocallyandvisuallychecked. ExternalAPcriterionpasseswithtinygains FCOS+.070272 SSD+.032161vsnoadapt,each4/5positiveblocks;vsraweach3/5positiveblocks. Cleanphi54.263%lowerbut99.3%stillupdates. ReportNEEDS_REVIEW;noT010/meta. Raw371MBarchivedownloadsession9887ongoing;remoteallrawsavedandhashed;derivedscienceoutputsreadyforGitHubnow.

## 2026-09-12T17:57:25+08:00 — R012 accepted; T010 started

Synced1a4df0c/R0126012f1d. T009closed; T010offline7scalarneed-to-adaptgrid authorized. FrozenT009receipts only. Predeclared T010_plan.md beforegatedAP. Baseline localmacro1passed8.54s;remoteAP/block2passed1.29s after correcting initialwrongcwd(filemissingexit4). No model/deploymentchanges. T009rawdownload9887stillactive.

## 2026-09-12T18:02:37+08:00 — T010 scalar preparation green

Pureextract/rank/composition/timing/preparationtests7passed11.83s. Two-imagefrozenreceiptpreparationsmoke14rows44configs exit0 without labels/predictions. Scoresandrulesplanfd88938 alreadycommitted beforeAP. Deployingscorepreparationforfull7000rawrows;all42gatesretained.

## 2026-09-12T18:06:42+08:00 — Full T010 pre-AP decisions frozen

Preparationcode32a89e3,release20260912-180251-taisp-t010-scores. DeploymentSSH255atcurrent-symlinkstep; verifiedalreadyextractedfilehash andcompletedonlysymlink/state step. Fullprepare exit0:7000rows44configs,all7scores available,70zero-vectorcosines coded0 perplan. Noannotations/predictionsread. All42ranks selectexact1750/3500/5250rows. DownloadedpreparationarchiveSHA21e03e82e58fff2d11938d3e8d55e39c3ee20e6fce0184f1dad4be0151ab53fe verified; samples/cohort/artifacthashes match. CommittingallbeforegatedAP.

## 2026-09-12T18:13:41+08:00 — T010 offline AP/report smoke green

Run20260912-180743-taisp-t010-offline-smoke release180659 source901560f:13affectedofflineAP/gating/regressiontests passed2.35s;2images14rows44configs21panels2772officialAPresults(126reusedanchors),16.57865s,exit0at18:08:12. Anchorcompositions exacthashmatch;reportaudit924shareddecisions pass,2772AProws396macro rows,smokecriterionnull. Reportcriterionboundaryunit1passed15.98s. PlotinitialOMP15 fromunusedgatingimport;removedthatimportandreadscorelabelsfromsavedreport,standaloneplotexit0andvisuallychecked. Nopackageupgrade,nogatedAPoutcome-basedchange. Fullprepare901560f committedbeforeAP;fullgridready.

## 2026-09-12T18:17:49+08:00 — T010 full run started; T009 raw archival complete

T010run20260912-181436,release181357,source/report9efe032,started18:14:44+08.
Fresh14offline/regressiontests passed2.45s;12CPUworkers evaluatefrozen44configs,
no modelinference. Full5,544AProwreportpending. T009archive9887completed,fullSHAand
84individualpredictionhashes verified. Sample125507835bytes compressedlosslesslyto
31110348bytes,GzipSHA31d808e7094b539a9f9b02a8ac2b6b7761d8fc3db8387fdc2dfb71ca638633a2,
roundtripSHA matches. Originalpreserved,exactrawpathGitignored;84predictions+Gzipreadyforarchivalcommit.

## 2026-09-12T18:54:45+08:00 — T010 formal offline study complete and audited

Run20260912-181436 completed18:36:59+08,exit0;source/report9efe032;1330.08296s.
All44configs21panels6groups5544AProws792macro rows. 14tests2.45s beforefullrun.
Archive485bd11b924ac81ec63a6c6f29baae16ac430c673fccc8c0052eb24f91d8227f verified.
21panelhashes/252anchors match;924shareddecisionschecked;33264pairedmetricdeltas,
allmacro/sign/rule outcomes independentlyrecomputed. Finalplotexit0/visuallychecked.
9of42passdevelopmentalrule. Proposedsupport_confidence_low_50:clean49.7%,phi.017600,
FCOS+.036793/SSD+.032926vsnoadapt,4/5and5/5blocks. Tinygain,FCOSvsfull-.033478,
clean/corruptcoveragessimilar;noindependentneeddiscriminationclaim. Allnegativesretained.
T009rawarchivalpushsession56743 completeda23b618. Noactivejobs/transfers,nonewT011/meta.

## 2026-09-12T19:32:06+08:00 — R013 accepted; T011 matched-random control started

Syncedad06718/R01326ff476;T010closed. LatestpointerlocatesR013appendinmainqueue(initialguessedcontinuationfilenameabsent,correctqueueimmediatelyread). PlanT011 freezesexactcandidatevectorand200seeds2026091600..2026091799,35stratumcounts,hashrule,percentile/tail conventions beforeAP. Remoteofflinebaseline14passed2.31s. No scientificparameter ormodelchange.

## 2026-09-12T19:37:10+08:00 — T011 selector preparation green

5matched-control/preparation/statistic tests passed8.05s. Two-imageexistingT010fixture:14rows203configs200controls14partialstrata exactcounts,exit0. Partialstrataoneimageeachmakeidenticalrandommasks;smokehasnoscientificinterpretation. Fullcandidatevector isreadverbatim,notreranked. Code/seedrule readyforfull35-stratumpreparationbeforeAP.

## 2026-09-12T19:39:20+08:00 — T011 all 200 matched selectors frozen before AP

Preparationb49405d generated203x7000decisionmatrix with200unique randommasks;all200x35candidate stratumcounts matchexactly. Candidatehash7a02e6d66336ef39fbd4b55c3a5aba564aacee56bd2d7405b62ce9d811a511f6 unchanged. SeedsSHA3706edc0a3408cf87368eddf507324d7b5658fa46f7f495a2a052bcaeb0d0a01;matrixNPZSHA8e9458588ca3a614a32b86fd5c39a58a8618c53ccd7fe17ac190b9940f42deb9. NoAP/predictions/labelsread.6control/preparation/batch/statistic tests passed11.17s. T010evaluator gains optional25-configschedulingonly,defaultbehaviorpreserved;nextbatchedofficialsmoke.

## 2026-09-12T19:43:38+08:00 — T011 full-control smoke running

Run20260912-194144/source8dd776d/release193946 started19:41:54+08;20affectedoffline/regressiontests passed3.14s. FourCPUworkers evaluate203configs in25-configbatches,189panels,2-imagefixtureonly. Fullpre-AP200masks alreadycommitted8dd776d. NoformalT011fullrunyet.

## 2026-09-12T19:53:07+08:00 — T011 offline smoke and report complete

Run 20260912-194144-taisp-t011-offline-smoke finished 19:43:25+08, exit 0.
20 affected tests passed in 3.14s. All 203 configurations / 189 batched panels /
12,789 AP rows / 1,827 macro rows completed in 85.972176s with four CPU workers.
Archive SHA afb4b41579e84bec02d93f43469392a68a5f2ce8f4745cadfd1cc1fe618e90ab verified.
Report checks all 189 candidate/endpoint group metrics and composition hashes against
T010, plus 4,263 shared detector-condition decisions. Two-image strata force all
200 masks equal to the candidate; exact ties, percentile 0 and tail 1 confirmed.
Scientific criterion remains unassessed on smoke. Standalone plot generated and
visually checked. Full masks were precommitted 8dd776d; no full T011 AP run yet.
Next deploy report/evaluator and run the frozen 25,578-row CPU study with 24 workers.

## 2026-09-12T19:55:18+08:00 — T011 full frozen CPU analysis launched

Run **20260912-195353-taisp-t011-coco1000-offline**, release **20260912-195321-taisp-t011-full**,
source/report **8b2143f**, started 2026-09-12T19:54:03+08:00.
Fresh **20 affected offline/regression tests passed in 2.90s**. tmux active;
24 CPU workers are evaluating 203 fixed configurations, batches of 25.
Full decision matrix remote SHA matches pre-AP commit 8dd776d exactly.
Expected 25,578 AP rows (25,200 random +126 candidate +252 reused anchors),
3,654 macro rows, 189 panels. Report runs automatically after all panels complete.
Exact command/release stored in research_log/T011/full_run_meta.json.
No model/adaptation rerun or scientific parameter change. Final result pending.
15-minute heartbeat taisp verified ACTIVE; resume this run without duplication.

## 2026-09-12T20:46:19+08:00 — T011 formal run finished; receipt retrieval in progress

Run 20260912-195353-taisp-t011-coco1000-offline completed 20:39:00+08, exit 0.
All 189 panels / 203 configs / 25,578 AP rows completed in 2688.745720s.
Automatic R013 outcome false; random R012 pass count 96/200.
Full archive SHA ba1b20f2c022eda264da19d057b267b7ace60a17ae6c1e82dd3c4e39bcc01ebe,
12,021,457 bytes; transfer active to project .autodl/t011_full_receipts.tar.gz.
Final arithmetic audit, plot and report pending. No new experiment/task started.

## 2026-09-12T20:50:54+08:00 — T011 final report and audit complete

Full archive downloaded and SHA verified; 189 panels /25,578 AP rows /203 configs
retained. Independent arithmetic audit confirms all 153,468 paired deltas, macros,
control statistics, R012 frequency 96/200 and R013 FAIL (FCOS percentile 47%, 2/5
block medians; SSD 96%, 4/5). Plot generated/visually checked. Source/report8b2143f.
T011_report.md and CODEX_TO_CHATGPT.md report NEEDS_REVIEW. No active run/transfer.
No scalar refinement, candidate validation on new cohort or T012/meta started.

## 2026-09-12T21:09:21+08:00 — R014 accepted; T012 fixed half-dose started

Synced cbbc8d8/4afc07a. T011 closed, scalar branch stopped. T012_plan.md freezes .5 attenuation and all R014 criteria before model execution. Fresh remote baseline 4 passed, 1 opt-in real-model test skipped, 1.56s. No new cohort or dose search.

## 2026-09-12T21:12:17+08:00 — T012 update algebra green

Minimal fixed half-dose wrapper shares the accepted loop; full public signature unchanged. Local5 passed2 real-model skipped in5.44s: exact half algebra at each current phi, freeze/reset/empty/support and deployable isolation. T009 driver gains reference-only endpoint reuse for half variant and requested pin/config/ID comparison. Reference hashes frozen in T012_references.json. Next remote reference/real-model tests before two-image smoke.

## 2026-09-12T21:14:14+08:00 — T012 remote integration checks

Initial deployment20260912-211222 timed out at first SSH mkdir (255), before upload or run. Normal retry deployed release20260912-211252-taisp-t012-tests successfully; sourceaeef36b. Real-model opt-in half/full/reference/replication tests now running from exact release; no scientific or environment changes.

## 2026-09-12T21:20:37+08:00 — T012 real regression passed; smoke pin comparison repaired

Remote sourceaeef36b focused10real/reference tests passed63.51s. Full90real-model/regression tests passed222.54s in run20260912-211448. Then smoke aborted before any adaptation with AssertionError positive_prompts: current constants are tuples, saved JSON arrays are lists, text identical. Minimal repair converts only the two prompt tuples to lists for comparison; focused fixture now covers this observed serialization difference. No prompt/model/scientific changes. Failed run exit1 at21:18:48 retained; archiveSHA121840736022f314dd0467d3d7225d1d1c278e4e874df26c72f7e372bc9b0c16 verified. Report criterion/ratio/reference helper4 tests passed10.51s. Next repaired smoke and report validation.

## 2026-09-12T21:24:10+08:00 — T012 repaired smoke/report green

Run20260912-212148-taisp-t012-study-smoke-fixed, sourcedddc238, release212045,
finished21:22:15+08 exit0. Focused9passed1real-opt-in-skipped in2.30s (earlier90
fullreal/regression passed222.54s). Twoimages14episodes,21predictionfiles,189AP
rows including126 exactreusedendpoints; runtime7.205856s. All14originalsupports
exactlymatchT009; allstepalgebra/reset/fallbackchecks andreferenceSHA checks pass.
Reportcriteria null onsmoke; plotgeneratedandvisuallychecked. ArchiveSHA
79ec8eb7f9a2a0f4c0f532a6bfcf0edab5e9eb4e0a5139e432f2383f8180e2ae verified.
Next formal1000imagefixedalpha study, latestfull93tests beforeexperiment.
No alpha/candidate/cohort/model/prompt change.

## 2026-09-12T21:27:24+08:00 — T012 formal fixed half-dose job launched

Run **20260912-212537-taisp-t012-coco1000-half-dose**, release **20260912-212421-taisp-t012-full**,
source **3c82267** (kernel/report dddc238), started **21:25:48+08**.
Latest full93real-model/regression suite currently running, followed automatically
by the fixed1000image/7000episode half-dose driver and T012 report. Exactcommand
in research_log/T012_full_run_meta.json. tmuxactive; do notduplicate.
Smoke90fulltests earlierpassed; repaired9focused and189APsmokereport passed.
GitHub push3c82267 initially timedout (443), normalretry succeeded. Full deployment
extracted then SSHtimedout atcurrentlink (255); four kernel/report/config/reference
hashes matchlocal, completed onlylink+last-release state. No method/configchanges.
Expected21newpredictionfiles126newAP+252reusedendpoints=378AProws. Final resultpending.

## 2026-09-12T21:44:23+08:00 — T012 formal regression green; full study progressing

Run20260912-212537: **93real-model/regression tests passed in222.69s**.
All model/ID/config/reference checks passed and formal inference is active;
latest observed290/1000images,830.6s inference elapsed. No failure or new task.
GitHub remainsR014/T012, frozenalpha=.5 unchanged. Final AP/report pending.

## 2026-09-12T22:20:59+08:00 — T012 all1000 inference complete; AP and archival active

All7000half-dose episodes/21predictionfiles written before AP. Final SSD evaluations underway. Immutable102728966byte rawarchive SHA6ce8db0e5957ee805ce9bd0057a122a410c1f6805ddefe8adfc8fde3e97e20c2 prepared; download session75110 active. See T012/raw_archive_delivery.json. No new adaptation execution or parameter changes.

## 2026-09-12T22:28:30+08:00 — T012 final science audited; raw archival continues

Runfinished22:22:35+08,exit0;378AProws/7000episodes/21rawpredictions.
Reportarchive SHA046e543450523a7faa78d19f0ee38ee87d413c353cbcebd66299809e3e3a7522
verified locally. Fullreport confirms7000exact supports/algebra/252anchors.
Independentaudit2268metricdeltas/allmacros/controlcriteria andpaireddose pass;
plotvisuallychecked. R014FAIL:SSD3positiveblocks;FCOS/SSD3/2abovecontrolmedians;
SSDaggregatelowerthancontrolmedian. Cleanratio.530583,allcleanAPbounds pass.
T012_report.md/mailbox NEEDS_REVIEW. Rawdownload75110stillactive,102728966expectedbytes;
noactiveexperiment. No alpha/dose/gate/newcohort/T013/meta expansion.

## 2026-09-12T23:07:03+08:00 — T012 raw archival verified and ready for GitHub delivery

Download75110 completedexit0;102728966bytearchiveSHA matches. All21predictionSHA
values match report_receipt,314186349bytes total,max31194559. Samples60392049bytes,
7000rows,SHAedd356414275c1ee828a9fe53bf4ef47f33ef57328ef5993ada44f0532407884
verified. Originalfiles retained unmodified. Noactiveexperiment/transfer; science
unchanged from591425b. GitHub taskqueue stillR014/T012,awaitresearchreview.

## 2026-09-13T00:07:11.7672054+08:00 — T013-A first-order plumbing and local sanity

R015 accepted T012 and issued T013-A (5e8afe6). Pre-code plan cd3060c pushed.
New training-only entry point reuses private full-hybrid loop; deployment signature
and forward unchanged. Six new synthetic contract cases pass: 11 passed/3 skipped
with affected baseline tests in10.43s. Tiny predictor SGD seed20260913,8steps lr.2:
outerloss .005294277333 -> .001558897318; all8 head gradients finite/nonzero.
Full local suite84passed/14skipped/1failed19.40s: missingpycocotools in existing
replication test. No scientific code repair needed. Use existing remote environment
for complete non-real regression and one optional synthetic-image real-model smoke.
No dataset/COCO training or real predictor optimizer. Receipts inresearch_log/T013A.

## 2026-09-13T00:14:32.7268912+08:00 — T013-A ready for review with optional CUDA numerical limitation

Required synthetic initialization plumbing and predictor-gradient checks complete.
Plan cd3060c; source bfd2484. Remote full regression 89 passed/10 skipped in5.09s.
Synthetic seed20260913,8 outer SGD steps, loss .005294277333 -> .001558897318.
Optional strict real CUDA smoke failed bitwise parity (2.11827e-5 phi discrepancy).
Bounded same-fixture diagnostic records repeated original drift8.58842e-6,
connected drift9.04803e-6; outer phi0/head gradients .1245225221/.0540611036;
frozen parameters/buffers/support unchanged. No algorithm/tolerance changed.
Report T013A_report.md preserves this limitation and all operational failures:
relative venv127, SSH255, uncaptured manual resume, local missing pycocotools.
All available logs/JSON/source receipts fetched under research_log. No active
job/transfer or COCO/real predictor training. T013-A NEEDS_REVIEW, await next task.

## 2026-09-13T01:37:14.0450633+08:00 — T013-B started from R016

R016/83f734e accepts T013-A with its optional CUDA limitation intact. Plan b73bcd2
predeclares12 CPU float64 fixtures, FD1e-5, K3/lr.1 and original continuation gate.
Baseline8passed/1skipped19.17s. Exact unroll added only inanalysis; focused tests
inprogress. No deployment/predictor/ISP/model change, no real optimizer started.

## 2026-09-13T01:45:29.1157459+08:00 — T013-B Part A passes; train microset committed

Part A source42072e7, median FO/exact cosine .9980674725590613;12/12positive;
min exact/FD .9999999999999998; allfinite; state/image forwarddifferencesexactzero.
Train microset fc88531 committed before optimizer:65088,426525,541157,129068.
All official coco_url values train2017. 13,798 readable eligible images after
skipping998 inaccessible/root symlinks; no dataset modification or download.
Part B source-only outer path ready;23passed/2skipped17.88s, including target
variation proving labels change outer objective without changing inner trajectory.
Next remote complete non-real regression and exactly3 predictor SGD updates.

## 2026-09-13T01:46:44.5537489+08:00 — T013-B bounded source smoke launched

Run20260913-014612-taisp-t013b-source-three-steps,release014535,source1685fb6.
Command complete non-real suite then fixed8episodes/3SGDsteps; exactcommand in
runmeta and T013B_handoff.md. No duplicate, no further optimizer steps permitted.

## 2026-09-13T01:51:35.3833075+08:00 — T013-B complete, receipts fetched, ready for review

Part A12episode exact/FD reference valid and FO gatePASS; allforwardtrajectoriesexact.
PartB run014612 completed01:46:40+08exit0;93passed10skipped5.94s beforethreesteps.
All3predictorupdates finite, finaldelta.000527524506; models/buffers remainunchanged.
Meanouterloss .474083306,.472119273,.475948031,.473895423 nonmonotonic.
Cleanloss decreased .004128285 whilecorruptloss increased .003752520; retained.
Cleanphi3mean rises .071202720 to.090419135; no identity/performanceclaim.
Rawsteps/supports/receipt/checkpoint/logs fetched, source inputhashes verified.
Report T013B_report.md; NEEDS_REVIEW; no furthertraining oractiveprocess.

## 2026-09-13T03:02:19.0363509+08:00 — T013-C diagnostic prepared

R017 mainqueue accepts T013-B. Plan cdd5c58 committedbeforeoutcomes.
Baseline7passed15.42s;focused14passed/1skipped16.99s. Newanalysis gradient_conflict.py
reuses source_outer_episode, savedsupport and frozenmanifest. Threeindependent
originalcopies use joint/clean/corrupt meanheadgradient oneSGDstep1e-3 each.
Trunk expectedzero atinitialhead; one originalno-update repeatforCUDA drift.
No deployment,ISP,models,losses,predictorarchitecture orT013Aassertion changes.

## 2026-09-13T03:03:35.5291325+08:00 — T013-C remote diagnostic launched

Run20260913-030301-taisp-t013c-conflict-conditioning,release030230,source68a6b07.
Fullregression then exactly3independentone-step probes plus no-update repeat.
Manifest/supporthashes frozen. Runmetadata records exactcommand; no duplicate.

## 2026-09-13T03:09:00.2485404+08:00 — T013-C completed and archived for review

Run030301/source68a6b07 exit0 at03:03:31+08;95passed10skipped6.28s.
8episode gradients plus3independentone-step probes plusoriginalrepeatcomplete.
Aggregatephi/headcos-.747508/-.726191 opposing; finitegroupcross-harm notobserved.
Jointoutputbiascomponentshare96.172876%,centeredvariation.170529%totalenergy.
No-updateCUDArepeatmaxlossdifference.006211102,maxphi3difference.002021194;
initialphi0exactreset, allfrozenstatesunchanged. Preserve numerical/causallimitations.
FullreportT013C_report.md andalltables/rawJSON/checkpoints fetched; no rerun.
No longertraining/regularizer/redesign/T013-D; NEEDS_REVIEW.

## 2026-09-13T03:32:49.1738691+08:00 — T013-D prepared, zero optimizer steps

R018/b9e59da accepts T013-C; pre-outcomeplan78ef20e. Baseline9passed18.05s,
focused12passed17.15s. Analysis-only repeatability/common-modecode ready.
All12identicalno-update repeats predeclared, alloutcomesretained; originalstate and
model/data/support/priorreceipt hashes pinned. No training/probe reruns authorized.

## 2026-09-13T03:37:46.0096273+08:00 — T013-D launched
Run 20260913-033722-taisp-t013d-repeatability-common-mode; source 81097bf; release 20260913-033259-taisp-t013d-repeatability. Full regression then exactly 12 no-update repeats. No duplicate launch.

## 2026-09-13T03:43:30.9991492+08:00 — T013-D completed, NEEDS_REVIEW

Run 033722/source81097bf exit0 at03:38:31+08. 98passed10skipped6.17s.
12x8 no-update repeats retained; phi/head opposing in12/12, medians-.734025/-.712936.
Wh93.894980%common energy. Most saved effects within repeat variation; joint-corrupt
exception exceeds range2.5989x, no causal optimizer ranking. Optional CuBLAS failure
retained, stopped as planned. No model/setting/tolerance changes. All artifacts fetched.
Offline table-header TypeError repaired (integer labels to strings); renderer succeeded,
no model rerun. Full report/tables/artifact manifest ready; stop for research review.

## 2026-09-13T04:54:23.3490338+08:00 — R019 synchronized; T013-E prepared

Research accepts T013-D via74cbdce. Pre-outcome plan604528c pushed before code/results.
Baseline6passed10.06s; focused9passed7.14s. Reuse T013C evaluate/directions/one_step,
source-meta data/loss/isolation and T013D raw inputs. New exact-output gate, priorrange
comparison, descriptive Spearman and predeclared decision only. No deployment changes.
Next fullremote regression, 3no-update repeats, conditional3independent one-step probes.

## 2026-09-13T04:55:41.6601322+08:00 — T013-E launched
Run20260913-045510-taisp-t013e-deterministic-replay, sourceb1278c4, release045442. Fullremote tests then exact gate; conditionalprobes only. No duplicate job.

## 2026-09-13T04:57:15.4407989+08:00 — T013-E initial wrapper check repaired before outcomes

Run045510 exit1 at04:55:28+08 after101passed10skipped5.91s. Assertion occurred in
initial isolation before any sample/repeat/probe; full failed run retained. CLIP loss
wrapper defaults training=True although its encoder/model are eval/frozen; existing
trust_radius lines28-29 set wrappers eval before each normal computation. Local mock
reproduces frozen check False before wrapper.eval(), True after, with unchangedstate.
Smallest repair: call clip.eval() before the new initial check in analysis runner only.
Focused9passed6.99s. No deterministic-operator failure, no output mismatch, no scientific
outcome rerun or tolerance relaxation. Relaunch same predeclared protocol after deploy.

## 2026-09-13T04:58:16.3274215+08:00 — T013-E setup repair deployed, fixed run launched
Run20260913-045756-taisp-t013e-deterministic-replay-fixed, sourcea585749, release045732. Prior failedrun retained; no duplicate activejob.

## 2026-09-13T05:01:31.3830697+08:00 — T013-E exact operator blocker; stopped for review

Finalrun045756/sourcea585749 exit1 at04:58:30+08, regression101passed10skipped5.83s.
CUBLAS env fixed beforeimport and runtime recorded. First attempted CLIP gradient
failed at upsample_bicubic2d_aa_backward_out_cuda, no deterministic implementation.
No complete repeats, exact gate not reached, zero SGD probes. Per R019 stoppedbranch
with full traceback, no workaround/alternatekernel/warn_only/tolerance change.
Initial isolation passed; postfailure check not recorded. Bothrun receipts fetched,
artifact SHAmanifest/disposition/report retained. No finite-step scientific conclusion;
NEEDS_REVIEW with StageA blocked. No T013-F or method changes automatically.

## 2026-09-13T05:56:13.8490767+08:00 — R020 synchronized; T013-F ready

R020/7efdb89 accepts T013-E blocked outcome. Pre-outcomeplan0ef7ef7 pushed.
Reuse exact original/joint/clean/corrupted T013C checkpoint bytes; allhashes pinned.
No gradient recomputation/optimizer. New scheduler/effectsummary only; existing models,
ISP, losses, episode lifecycle unchanged. Baseline6passed8.31s; focused10passed7.06s.
Next fullremote tests and8 matchedcycles. Keep all512raw episode rows before summary.

## 2026-09-13T05:58:08.0226457+08:00 — T013-F launched
Run20260913-055654-taisp-t013f-matched-checkpoints, source0bfdd19, release055629. Fullregression then exactly8cycles/512episodes, no optimizer.

## 2026-09-13T06:05:35.5593680+08:00 — T013-F finished; all matched outcomes retained

Run055654/source0bfdd19 exit0 at06:01:34+08.105passed10skipped5.78s.
8cycles/64evaluations/512rows, zerooptimizersteps; allstate/hash/isolation checks passed.
Only joint pooled effect resolved under exact7/8+strictnullmax rule: median-.00503579760,
7/8negative versus maxabsnull.00479621300,ratio1.049953. All subgroup effects unresolved;
no resolved cross-harm or both-group improvement. Scientificdecision measurement-limited.
No extra repeats/tuning. All83files fetched and hashed; offline renderer succeeded,
full tables andreport ready. NEEDS_REVIEW; stopafterT013F, no nextmethod task automatically.

## 2026-09-13T07:01:40.4102208+08:00 — T013-G offline implementation; local environment blocker isolated

R021/3a1f363 acceptsT013F. Plan21eaa31 committed/pushed before newoutcomes.
All required H/G/headgradient/output arrays present and canonicalhashpinned.
Baseline5passed13.51s. Added NumPyfloat64 feature/factorization/decomposition and
synthetic tests only. Local focusedprocess nativeabort atmatmul; minimal NumPy+torch
repro gives OMP Error15 duplicate libiomp5md.dll. Logs preserved underT013G.
No unsafe duplicate-runtime flag, formula/tolerance relaxation or modelrerun.
Next use existing remoteCPU runtime: focusedtests thenfullregression thenofflineaudit.

## 2026-09-13T07:03:22.9023358+08:00 — T013-G CPU run launched
Run20260913-070237-taisp-t013g-offline-factorization, sourcea7a5494/release070203. Focusedtests thenfullregression thenofflineaudit only.

## 2026-09-13T07:08:24.9918219+08:00 — T013-G offline audit finished, NEEDS_REVIEW

Run070237/sourcea7a5494 exit0 at07:02:52+08. CPUfocused8passed2.29s, regression107passed
11skipped5.18s. All7reconstructionchecks pass predeclared8eps32 bound; no tolerancechange.
medianrho1.433201502,rC.4570598744,rCout.5297500659 => mixed/commonterm domination.
Featurecenteredenergy8.612005%, participationrank1.68541. Biasrawenergy1.690688908e-7;
fullcenteredfraction.170529127%. CenteredA-Ccrosspositive1.188630969e-10: no cancellation.
Savedarraysonly, zero auditmodelcalls/optimizer. All5files fetched/hashed; matrices,
crossterms, errors, twoalgebraiccounterfactuals rendered toT013G/tables.md. Stop forreview.
2026-09-13T08:07:49.2556007+08:00 T013-H Stage A: deterministic hash cohort selected 32 of 13794 eligible source images (13798 readable valid before exclusion); full 5000 val IDs plus 1400 ledger eval IDs and prior4 excluded. No model calls. Manifest and pre-outcome plan committed before gradients.
2026-09-13T08:10:13.4820868+08:00 T013-H baseline: unchanged T013G remote release, CPU tests/test_predictor_factorization.py and test_matched_replay.py: 7 passed in 1.47s. Manifest SHA256 99f15bc4329b8431a6718bc1c2ae1ef3fa6221db0869a7cc593ef95bdca2d357 committed as f67e8d8 and pushed before any model calls. Implemented variable-N array statistics, fixed image folds and single-forward feature-hook collector.
2026-09-13T08:11:47.4170718+08:00 T013-H focused6passed1.60s and syntax pass; launched 20260913-081117-taisp-t013h-source-replication once on source553e02c, release081014; full regression then64primaryrecords thenCPUalgebra.
2026-09-13T08:16:55.6400851+08:00 T013-H complete exit0, 111passed10skipped6.16s. Fetched76files2937962bytes. StructuralreplicationPASS; covutilityFAIL39/64<40 despite19clean20corrupt andmedian.1344034354. All64isolation and3gradientidentitychecks pass. Negative centeredA-Ccross differs fromoldmicroset. Fullreport/tables/hashes retained; stop NEEDS_REVIEW.
2026-09-13T09:06:41.0745398+08:00 R023/c016e51 fetched and fast-forwarded; T013-H accepted/closed as negativeutility. T013-I offline capacity task nowactive. Wrote pre-outcome plan with exact H hashes, SVD tolerance, 4folds, 128pairpermutations and fixed conjunction; no fit metrics computed yet.
2026-09-13T09:11:02.0621260+08:00 T013-I baseline unchanged H remoteCPU:6passed1.54s. Implemented one analysis module reusing H fold/cosine/factorization helpers; SVD fit, fixed pair null and predeclared gate tests added. Local syntax compile passed; no new real-array fit computed.
2026-09-13T09:12:30.6402018+08:00 T013-I focused9passed1.66s remoteCPU. Run20260913-091200-taisp-t013i-linear-capacity launched once, release091116, codec6d2c6e; fullCPUregression then4observed+512nullSVD fits; no model calls.
2026-09-13T09:17:33.9758946+08:00 T013-I complete run091200exit0; regression116passed11skipped5.91s. PooledR2-.3806308900, residualcos.06836475594, positive33/64(15clean18corrupt), all4foldR2negative: fixedgateFAIL. R2aboveall128null (tail1/129) butcosbelow95th(tail36/129); preserveboth. 137files8432184bytes fetched/hashed, tables/report completed; stopNEEDS_REVIEW andclosecurrentrepresentationlearned-init branch perR023, awaitresearchpivot.
2026-09-13T10:08:43.5351910+08:00 R024/f5da87a fast-forwarded: T013-I accepted andclosed; T014-A spatialaction-space audit authorized. Pre-outcome16image/32episode manifest derived first4images perHblock; exactcachedoriginalsupport/model pins and singlecotangent regionalJacobian procedure predeclared. No newmodeloutcomes yet.
2026-09-13T10:13:05.2438731+08:00 T014-A baseline onunchanged Irelease remoteCPU:8passed1skipped1.75s. Implemented analysis-only compositor/cachedmask, identitycotangent gradients and saved-vector geometry/gate; synthetictests andsyntaxcheck added. No ISP/deployment modifications, no newrealmodeloutcomes.
2026-09-13T10:14:43.3573272+08:00 T014-A focused10passed2.02s; single run20260913-101408-taisp-t014a-spatial-action launched from767ba1c/release101321. Fullregression then32identitygradientrecords; no modelupdates.
2026-09-13T10:18:39.2228037+08:00 T014-A run101408exit1 after123passed10skipped6.84s. Saved7/32records:6pass, episode6image410054clean pseudo brightnessreconstructionerror6.258487701e-7 exceeds3.299696207e-7 bound(1.896686x). Allisolation/imagechecks pass. 25notrun; geometry/gateNOT_REACHED. Fetched13files217657bytes; blocker/report/tables persisted. No tolerancechange/rerun; BLOCKEDforresearchreview.
2026-09-13T11:24:26.0229762+08:00 R025/c584887 fetched andfast-forwarded: Ablockeraccepted, A1analysisnumericalcorrection authorized. Wrote pre-outcome plan for commonISPJVP + independentfloat64reductions, debugindices0/5/6, conditionalfresh32rerun; preservesoldfailedreceipt andallscientificthresholds.
2026-09-13T11:32:58.2927790+08:00 A1baseline onunchanged Arelease10passed2.09s. Implemented analysis-only optionalcotangentreturn andcommon8columnISP JVP withindependentfloat64reductions; exactStage-C/conditionalfullgates and5synthetictests. No modelcalls yet; syntaxcompilepass.
2026-09-13T11:34:38.4379469+08:00 A1focused11passed2.17s; run20260913-113401-taisp-t014a1-common-jacobian launchedonce on0ad53b6/release113310. Fullregression thenfixeddebug0/5/6 andconditionalfresh32. No oldreceipt/tolerance/sciencegate change.

2026-09-13T11:47:49.6295582+08:00 T014-A1 completed exit0: 128passed10skipped6.48s; 3fixeddebug plus32freshfull pass numerical/isolation checks. R024 conjunctionFAIL19/32 overall,9/16corrupted positiveDeltaD; no thresholds changed. OldA receipts unchanged. Fetched45files1092765bytes, rendered fulltables andSHAmanifest; report ready, NEEDS_REVIEW. No further experiment launched.

2026-09-13T12:25:17.6716539+08:00 User requested GPU priority. Future model/gradient/experiment workloads should prefer A6000 GPU where supported; retain CPU for lightweight file/report work and small offline algebra. Existing frozen device/precision protocols remain unchanged.

2026-09-13T12:27:33.4292103+08:00 R026/f27a996 synchronized during GPU preference update. T014-A/A1 accepted/closed; T015-A saved-array task authorized. Verified33 pinned inputs/order/reference provenance with no new summary outcomes; unchanged A1 baseline11passed2.28s. Pre-outcome plan written; existing GPU preference retained.

2026-09-13T12:31:17.1777836+08:00 T015-A analysis-only decomposition/triage implemented; reuse A1 reference records and spatial EPS. Synthetic+affected focused13passed2.05s on unchanged-version A6000 environment; syntax passed. Release20260913-122950-taisp-t015a-differential-subspace deployed; no new scientific summaries computed.

2026-09-13T12:32:20.2485818+08:00 T015-A formalrun20260913-123142 launchedonce, source4860be7/release122950; fullregression thenfrozenarrayaudit. GPUavailableforapplicableregression, tiny16DalgebraCPU; no modelaudit calls.

2026-09-13T12:35:45.8294123+08:00 T015-A complete exit0:135passed10skipped6.49s. Audit.023246104s CPU,0model/optimizer calls,32originalrecords; allD reconstructions passmax2.22044604925e-16. TaskcapacityYES, pseudo differentialutilityNO18/32overall9/16corrupt; closefixedpseudo-spatialbranch byR026. All7rawfiles154584bytes fetched/hashed; fullreport/tablesready. StopNEEDS_REVIEW; GPUpriority preference retained.

2026-09-13T13:56:11.7788928+08:00 R027/1107946 fetched/fast-forwarded: T015-Aaccepted; T016-A diagonalcalibrationaudit authorized. Verifiedfrozenpair/foldprovenance, pinned4inputs and256deterministicpairpermutations beforefits; baseline13passed1.50s onunchangedT015release. Pre-outcome planwritten.

2026-09-13T14:00:48.3737316+08:00 T016-A analysis-only diagonalfit/heldoutnormmatch/permutationaudit implemented; reusepair_rows andsavedarraysummary helpers. Syntaxpass; focused12passed1.52s onA6000environment. Release20260913-135925 deployed; no actualcalibrationoutcomes yet.

2026-09-13T14:01:35.8437424+08:00 T016-A run20260913-140107 launchedonce from1b19e8c/release135925; fullregression then fixed4fold/256pairnull audit. Normchecks andallscientificgatefrozen.

2026-09-13T14:05:13.7232304+08:00 T016-A completeexit0:140passed10skipped6.70s;1028closedformfits2.603052380s,0model/optimizercalls. Allnormchecks32primary+8192nullpass. R027FAIL Cdiffblocks2/4,DeltaD18/32overall8/16corrupt,nullcount18<=18.25;medianDeltaD.0128706335091>null95.00740041297933,tail3/257retained.264rawfiles26055799bytes fetched/hashed;fullreport/tablescompleted. Closecalibrationrescue; stopNEEDS_REVIEW.

2026-09-13T14:24:04.6231254+08:00 R028/f8fd9eb synchronized; T016-Aacceptedandclosed. T017-A fixed32A6000componentaudit authorized. Pinned5inputreceipts/modelhashes; baseline12passed1.88s. Pre-outcomeplan usesoneoracleforward, commonJVP, independenttotalcotangentclosure andfixedparity. No componentoutcomes computed.

2026-09-13T14:27:55.9467796+08:00 T017-A analysiscollector/attribution implemented reusingoneoracleforward/common8JVPcolumns; no existingmoduleedits. Syntaxpassed; focused9passed1.94s onA6000environment. Release20260913-142715 deployed. No realcomponentoutcomes yet; nextfullregressionthenfixed32GPUaudit.

2026-09-13T14:29:07.7066497+08:00 T017-A code5e0724b pushed afteroneunchangednetworkretry; run20260913-142838 launchedonce onA6000GPU,release142715. Fullregression thenfixed32componentaudit. No outcomes interpreted yet.

2026-09-13T14:33:19.5123074+08:00 T017-A GPUrun142838exit1 after144passed10skipped6.66s. Firstrecordindex0image182164clean failedindependentcomponentclosure(globalmax6.986881543e-5) ANDsavedA1relativeL2(sum9.819567371e-4,total2.633194794e-4 >1e-5). Scalarforwardlossesexactlymatchold; source/ISPisolationandallpartitionspass.1record0pass31notrun; noattributionortriage.6rawfiles24467bytesfetched/hashed,reportcompleted; BLOCKEDforresearchreview, noadjustment/rerun.

2026-09-13T15:25:40.9512825+08:00 R029/53f8662 synchronized; performancefirstT018-A authorized. Baseline6passed2skipped1.40s. Prepared100newsourceimages from13762eligible,0overlap36prior-source/5000val; JPEGhashmanifestSHAa01dfb1d40a6daceddccc1b7aa7f3f2e74871fd4a511d8d6c9acf8e50c2c111f. Candidateplan/currentT009settingsandcohortready forpre-outcomecommit. No model/performancecalls yet.

2026-09-13T15:34:03.4781952+08:00 T018-A experimental native loss/driver added; current modules unchanged. Focused10passed2skipped1.70s onA6000 environment. Release20260913-153309-taisp-t018a-nativept. Next fullregression plus CUDAK3two-image smoke without AP.

2026-09-13T15:36:32.3121768+08:00 T018-A code7c43f1f pushed. Smoke20260913-153419: full148passed10skipped6.94s; CUDA2images7conditions28adaptiveepisodes/14native,4finitecomponenthistories,K3,phi0zero,allstatechecks passed; 0AP,exit0. Formal20260913-153604-taisp-t018a-source100 started once with identical release153309/config/cohort onA6000cuda:0.

2026-09-13T15:42:53.3810104+08:00 Existing taisp heartbeat verified ACTIVE every15minutes; appended userGPU preference to saved prompt through automation_update, preserving schedule/task scope/quiet unchanged behavior. Model forwards/gradients/adaptation preferA6000CUDA; lightweight summaries/officialCOCO aggregationCPU.

2026-09-13T15:49:14.418244+08:00 T018-A fullrunexit0:100images/1400adaptiveepisodes/105officialevals,605.118242s collection+eval. All6R029flagsPASS; macrodelta+.216741182649AP vs current,+.193246665691vsraw;3/4blocks,4/6conditions;clean+.619964510336. Allisolation,700pairedsupports,finitegradients pass.62rawfiles30,654,090bytes fetched; archiveSHAa3fda68005b1367153d5eadbc0dde2351f64af61c71d0bf9351b01869e765578. Reportcomplete; NEEDS_REVIEW, noactivejob.

2026-09-13T16:10:55.738133+08:00 R030/7902e3a synced; T018-A accepted. T018-B preparation: unchanged baseline10passed2skipped1.74s.500newsource images,5fixed100blocks,136prior-source and5000val exclusions; manifestSHAe7a771126ae2fee9844dde456648b50f4c01ebcdc404318c21f9a33260da7b17. Same candidate/settings; no new model outcomes.

2026-09-13T16:12:10.7631363+08:00 T018-B runner cohortcount/tasksummary extension complete; source/current/native adaptation loop unchanged. Focused12passed2skipped1.77s,syntaxpassed. Deployed release20260913-161116-taisp-t018b-source500; no new model outcomes.

2026-09-13T16:19:05.2298210+08:00 T018-B code d8ff14f pushed. Full150passed10skipped7.06s; runtime smoke161225 exit0,28CUDAK3episodes/allisolationpass/0AP. Launched formal20260913-161818-taisp-t018b-source500 once onA6000 using same release161116/settings/cohort. Expected~50minutes; no outcomes yet.

2026-09-13T16:21:47.9838355+08:00 T018-B formal first29/500images completed,noerror. Report renderer scripts/report_t018b.py prepared/syntaxpassed; not run without completion. Durable fetch/report/resume instructions saved.

2026-09-13T17:12:49.443006+08:00 T018-B completeexit0:500images/7000adaptiveepisodes/126officialevals,2891.522734s. FixedR030confirmationFAIL:macro-.338228449022APvs current,-.134301697631vsraw;0/5blocks,0/6conditions;clean+.291859559754currentbut-.253654358664raw;AP50-.354632226472/AP75-.427867441159. All7000isolation/3500pairedsupports/finitegradients/11modulepins pass.62rawfiles139,528,015bytes fetched/hashverified; reportcomplete,closeexactformulation pendingreview. Noactivejob,no tuning.

2026-09-13T18:22:46.334177+08:00 R031/d60e72c synced: T018-B acceptednegative; T019-A fixed4componentfamily authorized. Baseline12passed2skipped1.57s. New200cohort4x50,636prior-source/5000val excluded;SHA534b17343ccba995beb9d112d552eefd7f1b2116479c2132dd99265b1d6735b4. Plan/config/methodhashes ready forpre-outcomecommit; no real-model outcomes.

2026-09-13T18:24:23.1437975+08:00 T019-A minimalnamedsubset/driver/selection implementation passedfocused24tests2skips1.82s; local syntaxpassed. Release20260913-182322-taisp-t019a-components deployed. No realmodeloutcomes; nextfullregression+2imageGPUsmoke0AP.

2026-09-13T18:26:31.3180888+08:00 T019-A full162passed10skipped6.84s. CUDA smoke20260913-182437 exit0:2images/70adaptiveepisodes,56candidateepisodes,224active-componentfloat32sum comparisons exactlymatch returnedtotals;allK3/statechecks pass,0AP. T019A_smoke_audit.json saved. No tuning; ready forformal200.

2026-09-13T18:27:30.1252481+08:00 T019-A formal20260913-182636-taisp-t019a-source200-components launchedonce afteralltests/smokeaudit. Same release182322/codeee98de0/config/cohort. Expected7000GPUadaptiveepisodes/210officialevals; noAPoutcomesyet.

2026-09-13T18:30:17.1461645+08:00 T019-A formal first12/200images completed,noerror. scripts/report_t019a.py prepared/syntaxpassed; fullfetch/audit/report instructions saved forheartbeat recovery. No scientificoutcomesyet.

2026-09-13T19:25:28.334825+08:00 T019-A completeexit0:200images/7000adaptiveepisodes/210officialevals,2887.666868s. Noeligiblecandidate:all4macro-current negative(-.0203273,-.0231889,-.0523431,-.0188116AP),all1/4positiveblocks and1/3/2/3positiveconditions outof6. Allabove-raw/clean/isolationpass,but+.10andreplicationfail. Formal22400active-sums exact;allstate/pairedsupports/codepins pass.104rawfiles124,041,334bytes fetched/hashverified;reportcomplete,closefixedbranch,NEEDS_REVIEW,noactivejob.

2026-09-13T20:21:52.634022+08:00 R032/2654048 synchronized; T020-A pre-outcome plan/config/cohort/pins prepared. Baseline10passed2skipped1.79s. Cohort200/four50folds excludes836prior-source/all5000val,eligible12962; SHAc364fb4d0bf2880318dbbcd22a47605cedccb31f6ee783ffdc56f6650d1f9ade. Preparation root lacked prepare_t013b import; used intact accepted release preparer unchanged. No model outcomes.

2026-09-13T20:26:39.896844+08:00 T020-A increment1 passed12tests1skip1.69s on release20260913-202408. Fixed Procrustes/row transport/identity-current/zero/fold isolation verified. First deploy SSHclosed before upload; unchanged retry succeeded. Added separate source-labelled collector and minimal shared-driver Q plumbing; syntax passed; next focused/full/CUDA smoke.

2026-09-13T20:31:59.3981083+08:00 T020-A full168passed10skipped7.49s; focused16passed2skipped1.70s. CUDA smoke20260913-202806 exit0,14pairs/two leave-one-image-outfits/28K3episodes/56normchecks/0AP,allisolationpassed. Release20260913-202732 used after transient202644extractSSHtimeout; unchanged redeploy succeeded. Formal200 ready, same code0546b05/config/cohort.

2026-09-13T20:34:18.392571+08:00 T020-A formal203203 started once, A6000 release202732/code0546b05 after all tests/smoke. Rendererf03d904 prepared; initial local AST read hit GBKdecode error, explicitUTF8 fixed, no experiment impact. Durable recovery saved.

2026-09-13T21:24:23.775106+08:00 T020-A complete exit0:200images/1400pairs/fourcrossfits/2800adaptiveepisodes/105APevals. FrozenR032fail:macro-current+.007885103017AP,1/4blocks,3/6conditions; meanheldoutalignment .154443289188 -> .134915686288. Clean/raw/isolation pass. All5600norms/2800episodechecks/1400supports/16remotehashpins pass.76rawfiles57905525bytes fetched/hashverified; reportcomplete. Close fixedglobal linear transport, NEEDS_REVIEW, noactivejob.

2026-09-13T22:15:42.322885+08:00. R033/a5bae26 synchronized. Baseline12passed2skipped1.77s.
200newimages/four50blocks; excludes1036prior source/debug IDs/all5000val.
Eligible pool12762; cohortSHA0fdb6a815d380104542c324ba2146944b47232dae22a416fa12d5f72a5d29725.
Fixed score.5/IoU.6/geometric-confidence greedy/ties-by-original-index/top20aftermatching.
Keep original box/class/score, existing loss/CLIP/ISP/K3/LR.1 unchanged. Empty=>identity.
No model outcomes or active job. Next commit plan, implement/test, fullregression,
2image CUDA smoke0AP, then formal200 on A6000. No thresholds or method expansion.

2026-09-13T22:19:55.619123+08:00 T021-A increment1 passed11tests2skips2.42s on release221708. Inversion/scoreIoUboundary/greedyconfidence/tie/top20/originalweights/emptyidentity verified. Added shared driver support-only branch and fixed assessment/diagnostics; syntaxpassed. Next focused/full/CUDA smoke0AP.

2026-09-13T22:24:03.006130+08:00 T021-A focused13passed2skipped2.50s; full175passed10skipped7.52s. CUDA smoke222054exit0:28K3episodes/14consensus episodes/28teachers/0AP. Offline real-support audit passed all originalbox/class/score retention, greedy/tie/one-to-one receipts, unchanged baseline supports and isolation. Ready formal200 using same release222014/code daa79e5.

2026-09-13T22:25:13.164576+08:00 T021-A formal222407 started once on A6000, same release222014/code daa79e5 after175passed10skipped and real-support smoke d276fc0. Expected2800teachers/2800K3episodes/105APevaluations; no outcomes. Durable recovery/report instructions saved.

2026-09-13T23:00:03.394627+08:00 T021-A completedexit0:200images/2800teachers/2800adaptiveepisodes/105APevals,1128.431519s. R033FAIL macro-current-.030367145082AP/macro-raw-.210352375112AP,1/4blocks,4/6conditions;clean+.263891538209vs current but-.149335803521vsraw. All2800isolation/1400consensus/9672retained supports/17remotehashpins pass.66rawfiles59995881bytes fetched/hashverified; reportcomplete,closefixedflipfilter,NEEDS_REVIEW,noactivejob.

2026-09-13T23:36:26.050998+08:00. R034/6001e14 synchronized. Baseline13passed1skipped2.38s.
200newtrain2017/four50blocks,1236prior-source/debug plus5000val excluded (6236IDs).
CohortSHA589878ad7af1d86bf73514d63dd58a4e40d58797d119ea4981be9e4292772690;eligible12562,firstIDs160585/114830.
Direction locked to go+gb; rho.5 regional coefficients bounded. Reuse validated A1
JVP/float64 chain-rule reduction to avoid previously observed reverse partition error.
Exact mask/ISP/loss/CLIP/globalbaseline preserved. Plan freezes parity tolerances before
new model calls. No activejob or scientific outcomes. Next implement/test isolated
candidate, fullregression and GPU parity/runtime smoke; stop on numericalfailure.

2026-09-13T23:42:42.966070+08:00 T022-A increment1 initial unit233843:3failed16passed1skip (detach inside JVP lambda dispatch error). Minimal donor-convention repair detach outside; unit233948 passed19tests1skip3.94s. Deployment current-symlink SSHclosed after extraction, tests used explicit intact release path. New real-model parity module prepared; frozen tolerances unchanged; no AP.

2026-09-13T23:44:05.485879+08:00 T022-A numericalrun234326 launched once onA6000 after module units passed. Fullregression andfrozen28parity checks; noAP. Sourcecodee8272c5,release234251.


# T022-A BLOCKED — pre-AP CLIP wrapper initialization

2026-09-13. R034/6001e14; plan b677a0c; code e8272c5.
Run 20260913-234326-taisp-t022a-numerical-parity ended exit 1; no active job.
Full 182 passed / 10 skipped; focused 20 passed / 1 skipped.
First of 28 real records passes all numerical checks but fails CLIP eval isolation.
Source/CLIP state hashes unchanged. Standalone parity entry omitted wrapper eval;
accepted and candidate adaptation runtimes already initialize it correctly.
Stop per R034; do not run AP or silently patch/re-run. This is an implementation
blocker, not a scientific FAIL. Minimal proposed repair and full receipts are in
research_log/T022A_report.md. Remaining 27 checks, real K3 smoke, formal driver
and 200-image study are pending. Await new explicit research queue decision.

# T022-A1 IN_PROGRESS — exact R035 parity repair

2026-09-14 +08. R035/0b46353 authorizes wrapper initialization and separate CLIP
parameter/eval/hash receipts only before the same 28-record rerun. Baseline is
T022-A's 182 passed/10 skipped and preserved first-record harness blocker.
Changed only parity harness and one focused test. Reuse accepted runtime
.eval().requires_grad_(False); loader, runtime/formula, ISP, masks, cohort and
all tolerances unchanged. No active run yet. Next focused/full tests and exact
A6000 parity; any failed record stops without AP. Only all-pass permits K3 smoke,
then shared driver completion and original R034 study. See R035 exact contract.

2026-09-14T00:10+08:00 R035 correction8948d4c deployed20260914-001000-taisp-t022a1-parity; run20260914-001023-taisp-t022a1-numerical-parity launched once. Focused21passed1skipped3.57s; full/parity ongoing. Exact same config/cohort, original failed rawrun immutable. NoAP.

2026-09-14T00:12+08:00 T022-A1 parity001023 completed exit0 at00:11:23: all28originalrecords/allCLIPsubchecks/finalhashes pass; focused21passed1skip3.57s/full183passed10skip8.25s. RawSHA655ad1ec8e13e0b015cb81d430fa5c11e2b97f037ce30b6ea8c45cf1f4c10afc fetched/verified. Proceed original R034 driver and K3zeroAPsmoke, no formula/runtime changes.

2026-09-14T00:15+08:00 Original R034 shared-driver integration completed only for spatial_dose_ours. Current branch unchanged; pure gate mapping and per-episode/edge diagnostics added. Protected18 modules and prior spatial runtime/ISP hashes unchanged. Increment focused18passed2warnings4.84s on release20260914-001435. Next fullregression + real K3smoke, including repeated fullmask/emptymask/empty-support episodes; zero AP.

# T022-A1 BLOCKED — runtime edge repeatability

2026-09-14 +08, R035/0b46353. Code7001d0c; no active run.
Parity repair8948d4c passed28real checks, separateCLIPisolation and hashes;
parityrun20260914-001023,receipt c713150. Smoke20260914-001557 exit1 after
186passed10skipped: repeated empty/full-mask K3 states not exactly equal.
Empty-mask maxstate difference1.458087936e-4 /relativeL2.0045155768;
fullmask2.7939677e-9. Empty support exactzero. All other edge/isolationchecks pass.
Stop per R035, no second repair, noAP. Normal14candidateepisode smoke incomplete;
only one current row persisted plus six complete edge trajectories. Not scientificFAIL.
See research_log/T022A1_report.md and T022A1/repeat_differences.json. Original
T022Acohort/formula/runtime/tolerances unchanged. Await explicit new research decision.

# T022-A2 IN_PROGRESS — R036 repeatability attribution only

2026-09-14 +08; research3e366b7, runtime7001d0c/release001534.
Reuse saved image160585 gamma_s1, original support from saved current row,
original empty-mask first trajectory states0..3 and exact T022A cohort.
No teacher/GT/AP/newcohort or method edits. Protected23 modules pinned in
T022A2_pins.json, including prior parity and failed edge checks.

B: identity plus saved empty-mask states1..3, 5independent pseudo/CLIP loss and
image-cotangent calls at each fixed image, all10repeat-pair metrics and raw tensors.
C: same4states, real/zero/one masks, fixed first B cotangents, 5regionalJVP calls
perstate/mask, allpairs; 20pure dose calls on one fixed tuple. No averaging.
D: fiveK3 repeats per current/realspatial/emptymask/fullmask/empty-support.
Use analysis-only loss/output hooks to retain actual runtime image cotangents,
not an alternative update implementation. Hooks returnNone and never modify
gradients; focused test proves intercepted cotangents and unchanged output.
Raw tensors chunked perrepeat, loss/gradient/state/image all-pair exact/maxabs/
relativeL2/cosine. Zero norms cosine undefined(null) unless bothzero(exact1).
A separate process enables deterministic_algorithms(True,warn_only=False) for
one representative cotangent computation, preserving full traceback/operator.
Do not enable deterministic mode in normal audit or performance evaluation.

No new pass tolerance. If fixed-input spatialJVP/dose varies, save blocker and
stop perR036 without patching. Otherwise finish all25runtime repeats and report
first observed variability and amplification scales; efficacy remainsopen.
Focused/full tests required; artifacts and reports committed/pushed/mirrored.

2026-09-14T02:01+08:00 T022-A2 analysis-only hook/metric/rawtensor tests15passed5.36s onA6000 release020010. Hooks reproduce uninstrumented current/spatial outputs and exact expected toy image cotangents. Existing23runtime modules unchanged. Next fullregression, separate deterministicprocess, then defaultCUDAfixed-input/runtime audit. NoAP.

2026-09-14T02:01+08:00 Run20260914-020134-taisp-t022a2-repeatability started once; release20260914-020108/codeee20040. Full regression then separate deterministic diagnostic process then default CUDAaudit. Collect existing run; do notduplicate. FixedJVP/dosevariability =>preserve/blockstop; otherwise25runtime episodes. ZeroAP.

# T022-A2 NEEDS_REVIEW — upstream repeatability audit complete

2026-09-14 +08. R036/3e366b7; plan2b669bc; codeee20040.
Run20260914-020134-taisp-t022a2-repeatability ended02:03:38 exit0; no active job.
190passed10skipped. Fixed image cp/cc vary; scalar losses exact. Fixed-cotangent
60JVP calls/120pairs exact; fixedtuple20dosecalls/190pairs exact. All25K3episodes
completed. Current max finalimage relativeL2.0001657856; realspatial.0000813436;
empty mask.0002225833; fullmask.0000668022. No tolerance or efficacy conclusion.
Empty-support states/pseudo/image exact; rawcompletion's controlblocker includes
irrelevant CLIPcotangent variation. Report explicitly separates that analysis
classification bug; raw unchanged. Separate deterministicprocess errors in
CLIPvisual_projection/F.linear/CuBLAS before backward, not evidence of ROI cause.
25episode isolation/reset checks pass. 129rawfiles583264937bytes fetched/hashverified.
See research_log/T022A2_report.md and T022A2/summary.json. T022-A remains preAP
BLOCKED; no new threshold, run, patch or AP until explicit next research decision.

# T022-A3 IN_PROGRESS — prospective R037 output repeatability

R037/3a3c598. T022A3_plan.json freezes first4cohortIDs, clean/contrast_s2, 5repeats
per current/spatial method, exact common pairwise normalization and allfour
R037criteria before newmodelcalls. First prepare8teacher supports once and save
support/mask/module hashes; commit/push those receipts BEFORE80adaptationepisodes.
Keep original23modules/constants/cohort unchanged. Reuse R036 compare/pairs and
lossless tensor persistence; no cotangent hooks needed. If confirmationfails,
stopBLOCKED/noAP. Ifallpass, original frozen200study uses existing7001d0c driver
without smokeedge invocation. Allhistoricalreceipts remainunchanged.
Noactivejob. Fullregression and focused confirmation tests required.

2026-09-14T02:41+08:00 New confirmation analysis17focusedtests pass5.21s; prior23modules unchanged. Preparation will make exactly8teacher calls and zero adaptations/AP, then supports/hashes committed before80repeats. No outcome yet.

2026-09-14T02:44+08:00 Prepare run20260914-024325 completed exit0, exactly8teacher calls/zero adaptations/AP. Full192passed10skip10.09s. Earlier024219 launcher SSHtimeout before creating remote run directory, confirmed no targetdir/session; safe retry024325. Supports/mask hashes now in immutable raw prepare receipt; commit BEFORE80repeats. Same release024158/code9b27c49.

2026-09-14T02:45+08:00 Supports committed5c35f4c before repeated gradients. Confirmation run20260914-024501-taisp-t022a3-confirmation80 started once with explicitrelease024158/code9b27c49, supportfileSHA0d7115a0acfae5cd81eaf6cc55be39fadf36bf2203fdcbcb88237349efc9eeea. NoAP. Next inspectcompletion, failure=>reportBLOCKED, pass=>original200formal unchanged.

# T022-A3 BLOCKED — prospective R037 output confirmation failed

2026-09-14 +08. R037/3a3c598; plan70df971; code9b27c49; supports5c35f4c.
Prepare024325:8teacher calls,192passed10skipped;80repeat run024501 completed
02:47:28 exit0 with gateBLOCKED. 80/80finite/reset/isolation/hashchecks pass,
but5/8tuples pass2x (need7), spatialmedian.000792959 > bound.000390756,
and332316/contrast_s2 exceeds5x. Allthree dispersioncriteriafail; noAP.
127rawfiles191292120bytes fetched/hashverified. SeeT022A3_report.md and
T022A3/complete_tables.md. No activejob; no formal200, tuning, newrepeats or
method change. Stop perR037, await explicit research decision. Previous
R036diagnosis and historical rawreceipts preserved; no AP efficacy conclusion.

# T023-A IN_PROGRESS — R038 saved-gradient object-only audit

R038/9f2d045 closes iterativeT022spatialdose. No rescue/withheldAP authorized.
Only corrected A1 common-Jacobian 32records (16clean/16corrupt/4fixedblocks),
authoritative33hashes verified before newmetrics. PlanT023A_plan.json freezes
input hashes/formulas/eps/closure bound/thresholds and diagnostic strata.
Use Python binary64 CPU on saved8D arrays only. No model calls/data loading,
AP, newcohort or deployment edit. Focused synthetic stdlib tests then32row
analysis, complete tables, commit/push/server mirror; stopNEEDS_REVIEW eitheroutcome.

2026-09-14T03:34+08:00 Four focused stdlib mathematical tests passed0.001s: utility signs, zero/undefined, original A1closure tolerance, conjunctive gate. No model imports/calls. Code restricted to scripts/audit_t023a.py and tests/test_object_action.py; binary64 Pythonfloat/math.fsum, original scores/strict thresholds. Next saved32record calculation.

# T023-A NEEDS_REVIEW — object-only audit FAIL; branch closed

2026-09-14 +08; R038/9f2d045; plan23c0169; analysis7d523a2.
32authoritative correctedA1episodes,33hashes/64closures pass. No model/AP calls.
S_obj positive17/32overall,10/16corrupt; DeltaS positive16/32overall,7/16corrupt;
medianDeltaS +.00809471858overall,-.00728372147corrupt;2/4positiveblockmedians.
Fixed conjunction FAIL=>close_object_only_action, no implementation/T023-B.
4focusedstdlibtests pass. All23protectedmodulehashes unchanged. Full tables,
vectors, strata, zero/undefined reporting in research_log/T023A/ and T023A_report.md.
R038 also closed T022 iterativetwo-state spatialdose; no withheld200AP or rescue.
Noactivejob. StopNEEDS_REVIEW; waitnewexplicitresearchdecision before designing
newregional self-supervised objective or any other experiment.

2026-09-14 +08 T024-A: R039 read;13baseline tests pass3.04s;16fresh selected seed20260924 excluding6436IDs. GPU preference recorded. Candidate/oracle ordering frozen.

2026-09-14 04:19+08 T024-A candidate increment:18focused/commonJVP/regionaltests passed3.77s. New pure JVP helper equals acceptedA1 exactly on tested inputs. GT-free separate interpreter; no protected edits. Ready candidate CUDA run after codecommit.

2026-09-14 04:23+08 T024-A: candidate32complete CUDA9.267749s, statehashunchanged/gradNone/nooracleimports. Allcandidate receipts committed/pushed9be51ee before any source reference. Referenceincrement21focusedpass3.76s/full204pass10skip10.04s. Runpostlockreference next.

2026-09-14 04:28+08 T024-A finished: both32episode CUDA stagesexit0;21focused/full204pass10skip. Feature/logit fixedgatesFAIL, corruptionDelta8/16 and7/16;nointegrity/nearzero blocker. Report+raw+tables complete. StopNEEDS_REVIEW perR039; noAP/runtime.

2026-09-14 05:18+08 R040 read, T025-A fresh24selected seed20260925 excluding6452.18baseline tests pass3.91s. Candidate exposurepair1.2/reciprocal SmoothL1beta1 classconditioned rawbboxdelta frozen; GT-free candidates before reference.

2026-09-14 05:21+08 T025-A candidate implementation passes24focused/regional tests incl realGPU predictorlayout/freeze smoke6.04s. Protected/inherited code unchanged. Ready48GT-free candidates after commit.

2026-09-14 05:25+08 T025-A candidates48complete10.715752s onA6000;locked/pushedef14e99 before referenceimplementation. Reference increment30focused incl realGPU pass6.13s/full212pass11skip10.56s. Source full-task and optional ROI/localization diagnostics next; no candidate change.

2026-09-14 05:29+08 T025-A complete:48candidate+48reference CUDAexit0;30focused/full212pass11skip. FixedgateFAIL overallDelta23/48/corrupt14/24,overallmediannegative,2/4blocks.48integritypass/nozero. Report/raw/tables complete; close exactgeometryfamily perR040 and stopNEEDS_REVIEW.

2026-09-14 06:53+08 R041/T026-A read; T025numericmisquote noted, authoritativea268f90 preserved. Baseline18pass3.90s.24fresh selectedseed20260926. Memoryavailability80classes>=16 (minimumtoaster23,hairdryer26); skip unreadable JPEGs using inherited os.access. SSHone timeout recoveredstatus; noactivejob. Freeze memoryhashselection before building.

2026-09-14 06:57+08 T026memoryselection:1280instances/1187distinctcleantrainimages,16per80classes,zeroaudit/prior/valoverlap;source manifestccff02e5c5786def9b6fcdfe4d4e98cfa3f23716c1521039852d486919f50766.13memory/inheritedtests pass2.89s. Commitmanifest/code before GPUmemory build.

2026-09-14 07:01+08 T026memoryGPUcomplete1280entries/1187images;healthy80x16,zerooverlap. Firstlauncher065756SSHtimeout leftmetaonly/no run.sh/tmux, confirmedbefore065856retry. Bothreceiptsretained. Commitmemorybeforecandidateoutcomes.

2026-09-14 07:03+08 T026memory1280healthpass/maxnormerror9.547524437714117e-08;locked/pushed2befc7f. Candidateincrement24tests pass3.83s. Predicted-class-only stabletop4,detachedanchor; no auditGT. Candidate GPU48 next.

2026-09-14 07:08+08 T026all48candidate9.698264s complete,anchor tensors/receipts locked/pushedb4c1c04 before reference.26focused/full220pass11skip. Futureexclusion manifest includesall1187memoryimages+24audit,priorunion2687. Reference deployscp timeout recovered via existingworkflow legacySCP retry, release070621 complete; source/candidatecodeunchanged.

2026-09-14 07:12+08 T026-A complete:healthy80x16memory,48candidate+48reference CUDAexit0. FrozenFAIL S26/48,18/24;Delta24/48,12/24;positivemedians but2/4blocks.48integritypass/nozero.26focused/full220pass11skip. Exactsource-memoryfamily closed;report/raw/fullstratastats complete. Futureexclude2687sourceIDs plusval;stopNEEDS_REVIEW.

2026-09-14 07:32+08 R042/T027-A read. Baseline13tests pass3.33s;new24cohortseed20260927 excludesall2687prior including1187memory images plusval5000. Predeclarecommutationstates/tolerance,E1systematiczero,E2medianties/averageranks beforeoutcomes. CohortcopySSHtimeout recoveredexistingSCPfallback. Noactivejob yet.

2026-09-14 07:35+08 T027preflight8synthetic/state cases passed CUDA atoriginalatol2e-7/rtol1e-6;no ISPedit/modeloutcomes.17focused/regional tests pass3.85s. Candidatecode/preflightreceipts commit before two-view modelstage.

2026-09-14 07:41+08 T027all48candidatescomplete14.410640s; both-viewgradients/overallagreementmedian.6972085021016277 locked/pushed0b1435b beforeGT.20focused/full227pass11skip. Reference onlyoriginaltaskgradient; exactE1/E2 and prelockedmedian split, no thresholdtuning.

2026-09-14 T027-A completed on A6000: E1/E2 FAIL; 48 integrity checks pass, no zero/abstention. Reference exit0 18.390367s, raw SHA verified. Full distributions/report generated from immutable receipts; exact family closed, stopNEEDS_REVIEW. Protected/prior code unchanged.

2026-09-14 08:24+08 R043/T028-A read and180freshcohort prepared withseed20260928;2711prior excluded. Baseline12pass2.82s. Predeclared std/null/ties/split beforemodeloutcomes; existing CLIP displacement identityloss retained, no objective change.

2026-09-14 08:26+08 T028candidate21D tests and inheritedJVP12pass5.33s; GT-free transitive import check passed. Newcode uses unchanged original-view pseudo and genericCLIP loss, both global8D; candidate release082414 ready.

2026-09-14 08:30+08 T028all360GT-freefeature records complete CUDAexit0;allhashes/isolation/parity verified. Raw source_revision accidentally placeholder; preservedraw plus codeSHA mapping3614d6e in candidate_provenance.json (verifiedequalcodepins), no rerun. Exactsplit/familybalances verified. Candidate lock before source references.

2026-09-14 08:32+08 T028post-lock reference/fixed affine fit implemented.17focusedpass5.15s;full236pass11skip4warnings12.56s onA6000environment. Train-only populationstd/SVDcutoff,exactzero threshold,128PCG64pair/family permutations and tiedAUROC tested. Reference/fit launchnext usingcandidate lock08dd3fd.

2026-09-14 08:35+08 T028-A completed:all360reference integritypassed, fixedfit/128nulls complete;holdoutAUROC.52625/.65767,precisiongain.02954/.08062,2/4blocks,bothbelow null95th=>FAIL.17focused/full236pass11skip. Report generatedfromexactSHAverifiedraw;freezeestimator,closefamily,stopNEEDS_REVIEW.

2026-09-14 10:12+08 T029-A/R044read;fresh60cohortseed20260929 preparednooutcomes,2891prior excluded,balancedfourblocks. Baseline9pass2.92s. Exactalpha2/detached91class/sharedsupportweight and numericconventions pinned;actualcommit infuturelaunches.

2026-09-14T11:38:10.405317 T029candidatefocused10pass5.62s. Initialtest usedliteral1.4 causingfloat32normalization mismatch; fixedtest toexisting scores.sum(), no methodchange. Ready120GPUcandidate;actualsourcecommit setbeforelaunch.

2026-09-14T11:40:59.319298 T029120candidates CUDAexit0;source_revision correct a2bc921;archive/perfileSHA verified;all support/target/isolation/JVPpass. Commitall120beforeGTreference.

2026-09-14T11:43:05.579477 T029reference/gatefocused12pass5.84s;full243pass11skip4warnings15.43s. All120candidate locked4ec7df4beforeGT. Referencecode prepared, noAP/CLIP/modelchange.

2026-09-14T11:48:12.486654 T029-A complete:all120candidate/referenceintegritypass;Ssoft63/120,31/60;Delta69/120,36/60,positivemedians but2/4blocks=>FAIL.12focused/full243pass11skip. RawSHAverified/reportcomplete;closeexactpower2family,stopNEEDS_REVIEW.

2026-09-14T12:58:08.304508 R045/T030-A read;SSH intermittenttimeouts recovered withworkflowstatus/retry;no modeloutcomes started.60cohort2951prior excluded prepared;baseline10pass5.41s. Native samplingseed20260930/reference20260913 andsumtolerance predeclared.

2026-09-14T13:00:19.221021 T030candidatefocused11pass5.81s; neutral native helper ASTbody identical existingoracle, numerical/RNG/eval/freezestate/component-JVP tests passed. Candidatecode ready for120GPUepisodes;samplingseed20260930.

2026-09-14T13:05:53.321772 T030-A BLOCKED onfirstcandidate componentgradient sum,maxerror/bound13.68;allindividualJVPparity/targets/RNG/freeze flags pass. NoGTreference orremainingepisodes/rerun. Full249pass11skip;failure persistedraw/report,42protected unchanged45remotepinsmatch. Awaitresearch numericaldiagnosis scope.

2026-09-14T13:24:35.139025 R046/T030-A1 acceptednewscope; first4pairs8episodes fixed,5reps, noGT. Pairwise/ordering/directparity/reverse-orderdiagnostic conventions precommitted. OriginalT030failure retained.

2026-09-14T13:29:24.773000 T030-A1 numericaldiagnostic implemented;16focusedpass7.94s,full254pass11skip19.80s. Fourcotangents/rawtensors+projectedvectors, freshdirectphi, RNG/state/support hashes andfixed8x5 gate;non-gatingep0probe restorationtested. ReadyA6000run.

2026-09-14T13:32:30.278361 T030-A1 actualGPUrun133102 started;132943launcherSSHfailureconfirmedno tmux/process/train.log,rawmetadata retained. 46codepins match. All8x5regularreps useoriginalnondeterministicsetting;ep0probeonly/restored.

2026-09-14T13:36:00.025186 T030-A1 complete8x5;replacementgatefaildirectfreshphi+multi3/8. Allnumericalreceipts/deterministicprobe restored;nofull120/noGT. Raw826MBcotangents preservedremote,fetchingandreporting.

## 2026-09-14T13:44:13.716379+08:00 T030-A1 final attribution audit
# T030-A1 / R046 — BLOCKED; numerical audit complete
R046/27fed9c; plan9ce9b5e, codea2844cd. Run20260914-133102-taisp-t030a1-attribution8x5 completed8episodes x5repetitions onA6000CUDA in64.28838972502854s, exit0, replacementgate FAIL.
Native repeatability all8pass: maximumrelative dispersion0.0014573985962872816, minimumcosine0.9999990149641096. Freshdirectphi relativeparity0/40pass (cosine40/40pass); multi-output no-worse3/8(required7). All40target/RNG/frozenstate/import checks pass. Non-gating deterministicprobe CuBLASRuntimeError, originalsettings/RNG/state restored.
No assertion correction, full120rerun, GTreference, AP or scientific T030PASS/FAIL. Stop and await explicit research decision. No activejob; every15minuteheartbeat and A6000CUDA preference retained. Futureadditional_sourceT030A_train_cohort.json preserves3011reserved/prior IDs plus5000val.
Report research_log/T030A1_report.md; all40 per-repetition and8episode JSON plus TSV, summary/integrity, code/tests and SHA index retained. Local complete archive .autodl/t030a1_diagnostic_raw.tar.gz SHA256 d1eeb81fdd7f1ea470a92a316e016c6154892c7636adfe2527bbd83864049be3 (664831514 bytes) contains all40 verified tensor files and both launcher receipts. Expanded tensor copies exhausted local D drive and only those new copies were removed; original archive retained. Full tensors also remain in remote runs/<run>/artifacts/diagnostic; tensor_storage_index.json records each path/size/hash. GitHub contains non-tensor receipts and tensor index, not826MB tensor objects.
Initial launcher20260914-132943 failedSSH before train.log/process; absence checked before retry. All outcomes from successfulrun only. No seed/tolerance/objective/support/deterministic-mode changes.
Focused16passed; full254passed11skipped4warnings19.80s. Report generation verified all raw file hashes; tensor hashes streamed from complete archive. See .autodl/last-heartbeat.json for final commit/mirror receipt after publication.

2026-09-14T15:07:00.949326+08:00 # T030-A2 / R047 IN_PROGRESS
Next four T030 pairs27717,74938,347235,413056 frozen;8episodes5reps; noGT. Pre-outcome manifest and gate plan saved. Baseline5tests passed5.15s onremote. Implement bounded samegraph/samecotangent audit; A6000CUDA; do not change old R046 result. No active run yet.

2026-09-14T15:10:46.728247+08:00 T030-A2 audit module focused15pass4warnings10.45s onA6000; samegraph3scalarcalls andsamecotangentchain verified synthetic; exactgate boundary tests/import boundary pass. Fullregression running before outcomes.

2026-09-14T15:12:13.538692+08:00 # T030-A2 / R047 IN_PROGRESS — GPU audit running
Manifest1d4b158; diagnosticcodee5ed293; release20260914-150932-taisp-t030a2-attribution. Run20260914-151141-taisp-t030a2-attribution8x5. Frozen next4pairs27717,74938,347235,413056;8episodes5reps. NoGT.
Focused15pass10.45s/full258pass11skip22.76s;49remotehashesmatch. Check this exactrun before any relaunch; collect summary and apply R047Gates1-4. Onlyallpass authorizes mechanical120candidate correction/rerun, thenstopbeforeGT; anyfailBLOCKED. Large tensors retained server,fetchJSON/hashindex only due localdisk.

2026-09-14T15:14:00.306506+08:00 # T030-A2 / R047 diagnostic PASS — conditional candidate correction in progress
Manifest1d4b158;codee5ed293;run20260914-151141-taisp-t030a2-attribution8x5;40reps48.186700168s. Gates1-4allpass;8/8noiseexplained;maxnativepairrel.0017662375053865307;minnativecos.9999984580853699;maxchainrel8.961290554393774e-7. NoGT/AP.
Diagnostic raw archiveSHA caede3c11bbf9c1d835c5ccb151d2950075f4aee5e814f4ada16f89ced36b5d6 verified, remoteallrawfilesSHAverifiedincluding40tensorfiles;localJSON+index retained. Nowauthorized mechanicalanalysisrunnercorrection+focused/fulltests thenfull120GTfreecandidatefrom0 undernewcommittedcode;stopbeforeGT. OldR046BLOCKED remains.

2026-09-14T15:15:54.343567+08:00 R047 conditional correction after diagnosticlock8c5991e: authoritative native scalar construction unchanged; remove oldcomponentadditivity assertion, retain diagnostic andaddmulti cotangent; retain samecotangent ISPparity (no freshdetectorphi gate existed inoriginalT030runner).16focusedpassed3warnings10.37s.

2026-09-14T15:16:52.181068+08:00 # T030-A2 IN_PROGRESS — diagnosticPASS; full120candidate running
R047/b083ffd; manifest1d4b158; diagnostice5ed293/results8c5991e; correctedrunner59efa11. Diagnostic8x5allgatesPASS, noGT. Fullregression259pass11skip22.60s;50remotecodepinsmatch.
Active run20260914-151622-taisp-t030a2-candidates120; release20260914-151439-taisp-t030a2-candidates. Candidatefromrecord0originalT030A/candidate_images.json, scalarunchanged, componentdiagnosticsnotstop. Check exactrun beforeanyrelaunch. Collect all120records/SHA/report,stopNEEDS_REVIEWbeforeanyGT/reference/AP/Kstep.

2026-09-14T15:20:45.386905+08:00
# T030-A2 / R047 — NEEDS_REVIEW; numerical PASS and full120candidate lock complete
Researchb083ffd; manifest1d4b158; diagnosticcodee5ed293; diagnosticresultlock8c5991e; correctedrunner59efa11. All4R047gatesPASS onnext4T030pairs(27717,74938,347235,413056),8episodes5reps. Native maxpairrel.0017662375053865307/mincos.9999984580853699; maxsamecotangentchainrel8.961290554393774e-7; formulationnoiseboundpasses8/8.
Diagnosticrun20260914-151141-taisp-t030a2-attribution8x5 completed48.18670016800752s; candidaterun20260914-151622-taisp-t030a2-candidates120 completed48.15833077405114s. BothA6000CUDAexit0, detectorstateunchanged, noGT/reference/AP/Kstep. Full120records freshly regeneratedfrom0under59efa11; no oldfailedrecordreuse. All120integrity and7perrecordsamecotangentparitiespass; oldcomponentadditivity remainsdiagnostic(0/120pass), nevercandidate orstop.
Candidate recordsSHA 031731120f517c03bfb8a6a5ae595676ae26b6bb6ba6076cbcfde00be6a7bdb6; rawmanifestSHA e5b019de2034f1697c5e630538869bf487674ca9cd06b10a3ed489bbdf6deac7. Original60cohort/support/native4lossunitweights/seed unchanged. OnlyPASS-authorizedanalysisrunner correction+tests; method/deploymentfilesprotected. Full259passed11skipped4warnings22.60s; correctedfocused16pass10.37s. Diagnosticfocused15pass10.45s/full258pass11skip22.76s. Diagnostic49andcorrected50remotehashesmatch.
Raw JSON/TSV/loss/gradient/support/target/RNG/state/provenance retainedlocal+GitHub;40six-cotangenttensorfiles remainserverwithSHA/norm/pathindex (localdiskconstrained). DiagnosticreceiptarchiveSHA caede3c11bbf9c1d835c5ccb151d2950075f4aee5e814f4ada16f89ced36b5d6; candidatearchiveSHA fc9b507b3d2bd605eea55aacfe3a84a6aa2b6ac01ee664002cd6edd175e7fdcb. BothlocalSHAverified; allrawserverfilesverified. CandidateSCPtimeout recoveredautomaticlegacyfallback,no rerun.
Reportresearch_log/T030A2_report.md; summary,per_repetition.tsv,candidate_per_episode.tsv,integrity.json. Noactivejob. STOPforresearchreviewbeforeanyannotation-bearingprocess. R045/R046stillBLOCKEDunderoriginalrules. NumericalPASSdoesnotestablishtaskutility. Futurefreshcohortadditional_sourceT030A_train_cohort.json remains3011reserved/priorIDs+5000val. Heartbeat15minutes/A6000GPUpreference unchanged. Finalcommit/mirrorreceipts inproject.autodl/last-heartbeat.json.

2026-09-14T15:56:03.363283+08:00 # T030-A3 / R048 IN_PROGRESS
Candidate lock localpreflight verified123filehashes/120order/exactreviewedandcandidatecommit. DescriptorandfrozengatesprecommittedbeforeGT. Implement reference-onlydraftcorrection, lazyoracleimportsafterpreflight, focused/fulltests,oneA6000reference120run. No candidates recomputed orGTloaded yet.

2026-09-14T16:00:06.374703+08:00 T030-A3 baseline11pass7.87s; referencefocused12pass3warnings7.77s. Remote preflight123hashes/120order/commit/state/integrity passedbeforeoracleimports/GT. Deployment155750SSHtimeoutbeforeupload; statusnoactivejobs then155837deployed. No modeloutcomes yet; fullregressionpending.

2026-09-14T16:01:18.329625+08:00 # T030-A3 / R048 IN_PROGRESS — reference120 running
Inputlockea67513; referencecoded480de9 pre-outcome; release20260914-155837-taisp-t030a3-reference. Run20260914-160040-taisp-t030a3-reference120 onA6000CUDA. All55pinsmatch, strictpreGTpreflight123hash/120orderpassed; oracleimportsdeferreduntilverification. CandidateimmutableR047recordsSHA031731120f517c03bfb8a6a5ae595676ae26b6bb6ba6076cbcfde00be6a7bdb6.
Focused12pass7.77s/full264pass11skip25.03s. Checkexactrunbeforeanyrelaunch; collectreference120/summary/hashes; applyoriginalR045conjunctionunchanged. Scientificfailcloseexactobjective; integrityfailBLOCKED. NoAP/Kstep/familyorcomponentrescue.

2026-09-14T16:05:07.799682+08:00
# T030-A3 / R048 — scientific FAIL; NEEDS_REVIEW
R048/860f4d5; inputlockea67513; referencecoded480de9 committedbeforeoutcomes. Run20260914-160040-taisp-t030a3-reference120, release20260914-155837-taisp-t030a3-reference, A6000CUDA120episodes87.01122627704171s exit0. CandidateR047 immutable59efa11/f9ca28e, no regeneration. ExactpreGT123hash/120order/commits/cohort/state/parity checks passedbeforeoracleimport/annotationload.
R045gateFAIL: S_nativepositive74/120(required80),42/60corrupted(required45); Deltapositive66/120(required68),32/60corrupted(required35). MedianDeltaoverall0.003484209199114012,corrupted0.0042116080176258855,clean0.003374409078447967;positiveblocks3/4, values[-0.0005735732333550583,0.01640573342034415,0.0029184774353523753,0.0023496265121768153]. MeansDeltaoverall-0.021331682095508665,corrupted-0.046962139222488215. Firstfourcountconditionsfail;medians/blocks/integritypass. Closeexactunweightedfour-loss pseudo-nativeobjective; no component/support/seed/confidence/CLIP/KLR rescue.
All120referenceintegrity/RNG/state/JVPpass; stored hard/nativevectors exactlyunchanged; all123candidatehashes unchanged. Detectorstate73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf. NoAP/Kstep/FCOS/SSD. Labelsusedonlypostlocksource-reference;no candidate recomputation. R045/R046historicalBLOCKED andR047numericalPASS unchanged.
Candidate recordsSHA031731120f517c03bfb8a6a5ae595676ae26b6bb6ba6076cbcfde00be6a7bdb6. Reference recordsSHAb492a6b2b3435dcef3dda0b7dd81a59811701d10d0a40a03b1ae037f1dda81a0; rawmanifestSHA149aaf824e04b1d40c35bfe4460f8ba0ccfc8ecd6d31e0b40ae9b09dc923dbdd. Rawarchive2ad9d2b4446d69e3ff873104231a58c16b653aecfbbdefb6e1bd5638c9eb07a3 verifiedlocal/server. All120rawrecords, score/component/family/blockdiagnostics andreportunderresearch_log/T030A3. No postoutcome referencecode edits.
Baseline11pass7.87s;focused12pass7.77s;full264pass11skip4warnings25.03s. 53protected/priorunchanged,55remotepinsmatch. Deployment155750SSHtimeoutprecededupload;statusnojob,155837success;onlyoneannotatedrun. Reportresearch_log/T030A3_report.md. Noactivejob;stopNEEDS_REVIEWwaitexplicitnewresearchtask. Futurefreshcohortadditional_sourceT030A_train_cohort.json remains3011reserved/priorIDs+5000val. Heartbeat15minutes/A6000preference unchanged. Finalcommit/mirrorreceiptinproject.autodl/last-heartbeat.json.

2026-09-14T16:56:43.076206+08:00 # T031-A / R049 IN_PROGRESS
240freshimages prepared(seed20261001),3011prior/reservedexcluded plusval/evaluation;180train60holdout pair-preserving balancedfamiliesandblocks. No modeloutcomes yet. Baseline9pass5.42s. Candidate480label-free thenlocked; train360reference/SVDoneRcommit beforeholdout120taskgradient. Annotationpreparer wrote separatepartitionsonserver; candidatefile containsnopaths/GT. NoT030revival/AP/Kstep.

2026-09-14T16:58:37.063731+08:00 T031candidatefocused6pass4warnings5.18s;unchangedoriginalviewhardreuse,zerorecord/RNG,noextraobjectiveimports tested. Ready480GTfreeCUDA.

2026-09-14T17:02:49.636914+08:00 # T031-A / R049 IN_PROGRESS — 480candidate run active
Cohortb019a89; candidatecode620076f; release20260914-165741-taisp-t031a-candidates; run20260914-165905-taisp-t031a-candidates480. Latest307/480 atcheck; no GT/reference/model fitting yet. Candidatefocused6passed5.18s;58pinsmatch.
Reference/Procrustes analysis module and tests drafted locally, pendingtest/deploy. Awaitcompletecandidate,fetchall480/commitlock beforetrain360. Trainannotationsonlyfrompartitionfile; oneCPUfloat64SVD;commitRbeforeholdout120. No candidate rerun or T030revival/AP.

2026-09-14T17:03:57.109067+08:00 T031reference/Procrustesfocused5pass3.88s; oneSVD O8reflection/columnconvention/zeros/meanoutliergate tests pass. Referencecodecommittedbeforetrainorholdoutgradients. FullsuitewillrunaftercandidateGPUjobcompletes.

2026-09-14T17:06:15.545699+08:00 T031all480candidate complete296.448390157s, sourcehashunchanged/noGT/imports/AP. ArchiveSHA6f4f44b12eec9f0de838290f23e021aba5f901b3b52aa347271be4a49386e433 currentlyfetching session2497. Full271pass11skip4warnings29.78s. No source train/holdoutgradients yet; nextcandidate commitlock.

2026-09-14T17:07:50.365691+08:00 All480GTfreecandidates rawSHAverified,483filehashespass;lockingbeforeGT. SCPstalledat131072bytes; stoppedonlymatchingdownloadscpPID46952,existingworkflowlegacyretrycompleted. No modelrerun. recordsSHA4a4736c99da7f7baf2c332d397133f449418e6fad27aa1c2b4d428a3fb7863d3

2026-09-14T17:08:46.391586+08:00 # T031-A / R049 candidate lock complete; trainreference next
Cohortb019a89; candidatecode620076f;all480resultlockd997b61. Candidate recordsSHA4a4736c99da7f7baf2c332d397133f449418e6fad27aa1c2b4d428a3fb7863d3,manifest8fb32640c416ca237762be805c26d0d66105aba6f0d11c0dfc2fc96cea4d891c;0zeros/allintegrity. Referencecode14b5c68,5focusedpass3.88s/full271pass11skip29.78s;61remotepinsmatch. No modeljobactiveyet. Launchtrain360onlyafterdescriptorupload;oneRthencommitbeforeholdout.

2026-09-14T17:13:43.3151697+08:00 # T031-A / R049 IN_PROGRESS — train360 reference active
Run 20260914-170909-taisp-t031a-train360-map; release 20260914-170250-taisp-t031a-reference; source 7569ed4. All480 candidate lock d997b6152d1aef4875498f55179664748c17b183; descriptor 7569ed4. Train360 CUDA reference and one CPU float64 8x8 SVD active. Holdout task gradients not started. Fetch/verify train results and commit R before holdout120. Full tests 271 passed, 11 skipped; 61 remote pins matched. Frozen R049 protocol; no tuning/AP/K-step.

2026-09-14T17:14:47.8710884+08:00 T031-A train360 complete in221.0398s; all360 integrity pass,180trainIDs only,zero holdout gradients. OneSVD R determinant -0.9999999999999991; orthogonality maxabs5.551115123125783e-16; zero unit hard/task0/0. R SHA333e7dcdb0286cfd5957e68f0533c613ab71d1b68cadc614da58a641207f01ec; trainrecordsSHA942a3bc20b154583317beeff02a2f8aab86dea366400d48f7b5b9f1bd184ee0a. Raw archive28599283fabe725c7f841ee9dca3ed081d809aad5abb5d17710df49a60e8bc63 and allfiles verified. Commit R now before holdout120.

2026-09-14T17:17:36.2567737+08:00 T031-A holdout120 active:20260914-171544-taisp-t031a-holdout120, reference release20260914-170250-taisp-t031a-reference/sourcebc8ee6d. Train360+R committed/pushed6ee34b06813918a70c2a2022ed40985bbb65452b BEFORE holdout; map descriptor bc8ee6d. Frozen code; no retuning. Fetch all holdout records, report original R049 conjunction, stop NEEDS_REVIEW. GPU A6000.

2026-09-14T17:19:29.301778+08:00 # T031-A / R049 — scientific FAIL; NEEDS_REVIEW
R049 c76b2e2; cohort/split b019a89; candidate code620076f; reference/fit14b5c68; all480 candidate lockd997b6152d1aef4875498f55179664748c17b183 and descriptor7569ed4 BEFOREtrain. Train360+R commit6ee34b06813918a70c2a2022ed40985bbb65452b and map descriptorbc8ee6df75bb1781ecba0be27a891ce76438b4e0 pushed BEFOREholdout120.
Runs: candidate20260914-165905-taisp-t031a-candidates480; train20260914-170909-taisp-t031a-train360-map; holdout20260914-171544-taisp-t031a-holdout120. All A6000 CUDA model calculations exit0; oneCPUfloat64 SVD. Train360/180images, holdout120/60images, disjoint; unchanged hard objective; no holdout before R.
FAIL: S_cal positive72/120(required80),38/60corrupt(required45); Delta positive53/120(required72),21/60corrupt(required36). MedianDeltaoverall-0.004003608916161447,corrupt-0.013859426945399671,clean0.00076676711481; meanDeltaoverall-0.014199087303298098,corrupt-0.026211238528914486. Positiveblocks1/4, medians[-0.009910628170084219, -0.0044470606185848475, 0.003078115054953973, -0.007902528659567403]. Onlycleanmedian andintegrity gatespass. Exactglobal O8familyclosed; no rescue tuning/alternative fit/AP/Kstep/runtime change.
R SHA333e7dcdb0286cfd5957e68f0533c613ab71d1b68cadc614da58a641207f01ec; determinant-0.9999999999999991,orthogonalitymaxabs5.551115123125783e-16,zero unit hard/task0/0;oneSVD. All480candidate+480reference integrity/RNG/state/JVPpass; unchanged candidate vectors;483+365+125rawfiles SHAverified;56protectedpriorunchanged;61reference remote pinsmatched. Full271pass11skip4warnings29.78s; candidatefocused6/referencefocused5pass. AP0.
Candidate recordsSHA4a4736c99da7f7baf2c332d397133f449418e6fad27aa1c2b4d428a3fb7863d3; trainrecordsSHA942a3bc20b154583317beeff02a2f8aab86dea366400d48f7b5b9f1bd184ee0a; holdoutrecordsSHAd2334c22651afc5f13679a481f86c623853120dd1c72a2f953c9a464b6949cc8. Fullreport research_log/T031A_report.md; matrix/summary/per_episode/integrity/rawruns retained. Overall coordinateenergymax decreases47.20% to40.40%; clean increases43.35% to48.79%; diagnostic only. No outcome-driven referencecode changes.
Noactivejob. Stop NEEDS_REVIEW, await explicit research continuation. Future fresh cohort additional_source T031A_train_cohort.json preserves3251 cumulative prior/reserved IDs plusval/evaluation. Heartbeat15minutes/A6000preference unchanged; finalpublication/mirrorreceipt inproject .autodl/last-heartbeat.json.

2026-09-14T18:12:30.496791+08:00 # T032-A / R050 IN_PROGRESS
Fresh240images seed20261002,3251prior/reservedexcluded;train180/holdout60,480episodes. Cohort and split prepared without outcomes; candidate module unchanged. Baseline7pass7.28s. Next deployprecommittedcohort and run480GTfreecandidateonA6000. Router thencommittedbeforetrainlabels;mapsbeforeholdout. No activejobyet.
Candidate manifest download failed because localnewdirectory absent; reproduced exact JSON locally from fetched cohort, no model rerun.

2026-09-14T18:14:49.1985790+08:00 T032-A IN_PROGRESS: candidate480 run20260914-181428-taisp-t032a-candidates480, release20260914-181346-taisp-t032a-candidates, sourceb0ce54b. 61protectedhashespass. Unchanged candidate code, no annotations. Routercode drafted, pending focusedtests; no real router fitting before candidatecommit. No source gradients yet.

2026-09-14T18:16:40.122375+08:00 Router focused4pass3.09s. Initial test incorrectly assumed package import excludes torch; existing taisp.__init__ imports ISP/torch. Corrected assertion to actual R050 detector/oracle/reference exclusion, unchanged package/model code; no real router fitted yet.

2026-09-14T18:19:43.350725+08:00 T032 router+reference focused7pass2warnings6.36s. Candidate at397/480 priorcheck. New reference reuses R049 verifier and exact source gradientloop; routes stored before lazyoracle imports; no real router/sourcegradient outcomes yet.

2026-09-14T18:21:05.835836+08:00 All480 candidates complete,483fileSHApassed; no GT, no routerfityet. Lockbeforeclustering. {"episodes": 480, "all_integrity_passed": true, "zeros": 1, "records_sha256": "d72c8f40316e05cf2b87c75fcac6c0d9a0f626dabc1006bc5372d258f79e1adc", "manifest_sha256": "c416b04b11438b94ace47e23b20372e0d6ca94bb39639651a1bb77606aef7f67", "archive_sha256": "92dfa7025f7f803fdbbe16b22493b31bf4eb9f92be2d4d44a0052d167f948c6e", "completion": {"status": "candidate_complete", "episodes": 480, "seconds": 297.5412268320215, "source_hash_after": "73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf", "forbidden_modules_loaded": [], "AP_calls": 0, "GT_loaded": false}}

2026-09-14T18:23:05.367468+08:00 # T032-A / R050 IN_PROGRESS — router complete, lock before train
Candidate480 locked5ae8bb6,descriptor0401c8e. Router code d71bd34; onefit only,6iterations,clusters171/189,minimum72 passed; trainzero count1; routerSHA2e150aadf557db106ebf73a74372b91d91751d52e80b041de3bf8db6010f738a. No annotations/taskgradients yet. Referencecode0d22b7d;full278pass11skip4warnings33.75s;65remotehashesmatch. Next commitrouter before train360/twoSVD onA6000. Existing reference release20260914-181842-taisp-t032a-reference.

2026-09-14T18:26:22.3858943+08:00 T032-A IN_PROGRESS train360 active:20260914-182353-taisp-t032a-train360-maps; release20260914-181842-taisp-t032a-reference; source2742f87. Routercommit844bc0d/descriptor2742f87 BEFOREannotations,counts171/189,6iterations. Candidate5ae8bb6/0401c8e locked. Full278pass11skip33.75s;65codepins pass. Fetchtrain360+two maps andcommitbeforeholdout120. No holdout taskgradients yet. Code frozen.

2026-09-14T18:30:30.8998860+08:00 Train360 completed; SCP t032a_train_raw.tar.gz stalled229376bytes for over60s. Stopped only matching transfer PID54660 so existing workflow legacy fallback can complete. No experiment rerun; await archiveSHA before map commit.

2026-09-14T18:31:09.371443+08:00 # T032-A train360 and two maps complete, lock before holdout
Run20260914-182353-taisp-t032a-train360-maps; all360integritypass,180trainIDs only; oneSVDpercluster171/189. MapsSHA36a0e62b7696b8ea99bc92f6806c8f30dc7d1aa4a6414cfd2ae81dfb5219ec4e; trainrecordsSHA98f151f7351f2e77732e22f4a317c6f4ba43dd381bcdb6f74cf693e03b75e286. Both orthogonalityerrors<9e-16. No holdouttaskgradient yet. Commit/push alltrain andmaps now before holdout120. Full278pass11skip. Download legacyfallback recovered; archiveSHA24cb957225aa68e438bd0a260e45775fae55160b17ae3102dfc4786f22ff6ad9 and all365rawfile hashespass.

2026-09-14T18:32:50.5472574+08:00 T032-A holdout120 active20260914-183220-taisp-t032a-holdout120, reference release20260914-181842-taisp-t032a-reference/source57bf9b6. Bothmaps andtrain360 committed/pusheda0ecfdab59b7425fb93ba298f647f1bf33794d6e, descriptor57bf9b6 BEFOREholdout. Router844bc0d/2742f87, candidate5ae8bb6/0401c8e. Code frozen; route120savedbeforeoracle. Fetchfinal and report exactR050 gates; no rescue/AP/Kstep. Run report renderer research_log/T032A/report.py --holdout-run 20260914-183220-taisp-t032a-holdout120 afterfetch.

2026-09-14T18:36:03.083312+08:00 # T032-A / R050 — scientific FAIL; NEEDS_REVIEW
R050195bfc8/latest8d45ec2 accepted T031FAIL. Fresh240cohort/split b0ce54b (seed20261002,3251excludedprior); routercoded71bd34 andreference0d22b7d BEFOREoutcomes. All480candidate lock5ae8bb68087d9739b5f69c7524a9f4b75effc829/descriptor0401c8e BEFORErouterfit. Routerlock844bc0d0a8dbaf2f24b11ab35cbddc50aad6cb98/2742f87 BEFOREtrainlabels. Bothmaps+train360 locka0ecfdab59b7425fb93ba298f647f1bf33794d6e/57bf9b6 BEFOREholdout120. All120routes/g_route fixedandsavedbeforeoracleimport; no reassign/refit.
Router6iterations,train171/189,holdout59/61; no collapse,coveragepassed. ScientificFAIL: S_routepositive79/120(required80),41/60corrupt(required45); Deltapositive49/120(required72),25/60corrupt(required36). MedianDeltaoverall-0.0136676592432203,corrupt-0.006134691018266185,clean-0.0183228039167913;meanDeltaoverall-0.06368087305755213,corrupt-0.09534089632552. Blockspositive0/4,medians[-0.03903835433007794, -0.001609350358977203, -0.0031494872970173806, -0.01849145291147203]. BothclustersnegativeDelta median/mean. Onlyintegrity androutecoveragegatespass. Closeexact sphericalK2+twoO8 family; no K/cluster/map/threshold/CLIP/native/AP rescue.
AllmodelworkA6000CUDA;CPUrouterandtwo8x8SVDonly. Candidate20260914-181428-taisp-t032a-candidates480:297.541226832s;train20260914-182353-taisp-t032a-train360-maps:222.711830443s;holdout20260914-183220-taisp-t032a-holdout120:76.50184720597463s;all exit0. Candidate480/train360/holdout120 state/RNG/JVP/integritypass;candidatevectorsunchanged; trainholdoutdisjoint;noAP/Kstep/runtimechanges. Raw483+365+126fileSHApass;65local/remote codepinsmatch(61protectedprior). Source detectorhash73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf. Full278passed11skipped4warnings33.75s;baseline7,routerfocused4,router+referencefocused7passed.
RouterSHA2e150aadf557db106ebf73a74372b91d91751d52e80b041de3bf8db6010f738a; mapsSHA36a0e62b7696b8ea99bc92f6806c8f30dc7d1aa4a6414cfd2ae81dfb5219ec4e; determinants[0.9999999999999987, -0.9999999999999994],orthogonalitymaxabs[8.881784197001252e-16, 5.551115123125783e-16]. Candidate/train/holdoutrecordSHAs['d72c8f40316e05cf2b87c75fcac6c0d9a0f626dabc1006bc5372d258f79e1adc', '98f151f7351f2e77732e22f4a317c6f4ba43dd381bcdb6f74cf693e03b75e286', '7a740412ae86aa974d8f7e04e4c177fff4f05544a157388698e159aa38e45ccd']. Percluster singularspectra andcoordinate/family/cluster/blockdiagnostics inreport andsummary. Coordinateenergymaxoverall42.89%hard->22.48%routed,diagnosticonly.
Report research_log/T032A_report.md; rawruns,centroids,assignments,maps,per_episode.tsv,integrity andallhashes retained. TrainSCPstallrecoveredlegacyfallback afterstoppingonlytransferPID54660; no experimentrerun. No postoutcome reference/router code edits.
Noactivejob. StopNEEDS_REVIEWwaitexplicitresearchcontinuation. Futureadditional_source T032A_train_cohort.json preserves3491cumulativeprior/reservedIDs plusval/evaluation. Heartbeat15minutes;A6000preferenceunchanged. Finalcommit/mirrorreceipt project .autodl/last-heartbeat.json.

2026-09-14T19:07:50.306002+08:00 # T033-A / R051 IN_PROGRESS
Plan andfresh300cohort readybeforemodelcalls;seed20261004,3491prior excluded,train240/holdout60. Candidate600nextafterbaselineandfocusedtests. Noactivejob/no newmodelcalls yet. Read T033A_plan.md andR051fullcontinuation. A6000CUDA.

2026-09-14T19:10:13.738299+08:00 T033baseline5pass3warnings5.12s existing orthogonal_candidates+source_roi_memory_audit tests onremote AFTER plan/cohortcommit4126a98. Candidate module reuseshardcandidate andfixedROI, newconfidencepool only; pendingfocusedtests.

2026-09-14T19:11:14.4197677+08:00 T033candidate+existinghard focused5pass3warnings7.64s; reuse/pool/empty/classirrelevance/importboundary green. Commit code/pins beforecandidate600.

2026-09-14T19:13:45.3787796+08:00 T033-A IN_PROGRESS candidate600 active20260914-191144-taisp-t033a-candidates600; release20260914-191003-taisp-t033a-candidates/sourceb5d4c6f. Cohort/plan4126a98 beforemodelcalls;baseline5pass5.12s,candidatefocused5pass7.64s;67remotehashesmatched. Candidate unchangedhard + own-supportROIpool only. No PCA/sourcegradients yet. Next pureNumPy PCA16/27Dstate/tangentSVD implementation andtests; all600candidatecommit beforeactualPCA, representationcommitbeforetrainlabels, modelcommitandpreoraclecorrecteddirectionscommitbeforeholdout.

2026-09-14T19:15:18.933621+08:00 T033 PCA/tangentmathfocused3pass1.61s: PCA sign/rank/standardization, affine singleSVD/minimum norm, zero target andabstention. Realcandidate600active; no actualPCA/labels yet.

2026-09-14T19:17:03.5729855+08:00 Math commit initially failed: D diskfull during Git auto-GC fromheartbeatfetch. GC exitedwithoutofspace. Two aborted tmp_pack files total~1.98GB remained read-only; normal Remove-Itemfailed, forcedremoval rejectedbyautomaticpolicy, no alternate deletion attempted. Disklaterreported304779264free. Existing code/pins intact. Continue smallcommits with one-off gc.auto=0/maintenance.auto=false; no configchange/no experimentrerun.

2026-09-14T19:22:23.220574+08:00 T033stages+mathfocused5pass5.45s. Candidate600completedexit0. Stage/referencecode freeze now before realPCAandtaskgradients; preoracle corrected directionsseparateprocessandcommit. Nextfetch600rawandfulltests.
