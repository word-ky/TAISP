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
