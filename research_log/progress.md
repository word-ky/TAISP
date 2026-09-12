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
