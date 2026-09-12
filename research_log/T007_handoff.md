# T007 recovery checkpoint

Updated 2026-09-12T10:48:56+08:00. T006 accepted/closed byR009(a8b5475); T007 IN_PROGRESS.
Read coordination/LATEST.md -> CHATGPT_TO_CODEX_R009_T007.md.
No full T007 study launched yet. Disjoint200manifest committed/pushed55e83e0
before newdatasetmodel execution; algorithm/predeclarationc5dc90a, seed20260913,
excludeallhistorical200,randomsample sortedremaining4800 then sort200.
ManifestSHA1738d9d504233513893cc4a3f3b1e0a2f58d4665eb4df522e87255082fade1ea;
overlap0,annotationhashunchanged; allJPEGhashes research_log/T007_subset.json.
Remote data /home/liujianhua/wjq/TAISP/shared/coco200_t007.

Hybrid/driver4e87904, fixed eps1e-12, CLIPnorm recomputedcurrentphi eachstep,
no target/labels inadapter; same direction,zeroexactfallback, fixedsource support.
Source/CLIPfreeze/exactCPUrepeat realtests passed aspartof58tests126.41s.
Localalgebra/selection/isolation4tests13.39s(2real skips);reportflip/stratum1test1.19s.
Smoke active20260912-104530-taisp-t007-study-smoke,
release20260912-104516-taisp-t007-smoke,source4e87904.
58tests passed; firstimage1490complete16.2s. Checkcompletion beforefetch.
Expected2images42rows98APevals. Report scripts/report_t007.py uncommitted,
localfocusedtestpasses; run onfullsmoke,inspectfigureandpaireddata beforefull.
No scientificparameterchange fromsmokeresults.

Fullrun aftersmokeaudit: commit/push allcode+smokereceipts, deploy sourcecommit,
TAISP_REAL_MODELS=1 pytest -q (expected59tests) then python -m taisp.analysis.run_t007
--data-root /home/liujianhua/wjq/TAISP/shared/coco200_t007
--output "$AUTODL_ARTIFACTS_DIR/study" (NO--limit).
Expected200images4200rows1400cases98APevals. No duplicate existingrun.

Reporthybrid/raw/CLIP/noadapt onsourceFCOS,AP50/AP75/AP beforeafter1/3,
all3pairedlosscontrasts,coscollinearity—notimproveddirection,FCOSstrictlossflip
strata ratios[0,1),[1,2),[2,4),[4,inf),clipzero. Clusterbootstrap2000seed20260912,
keepselectedcaseswithimage; stratavariableclustersweightedfractions. Cleanprimary,
phi/saturation/support/fallback/currentphi norms scales andtimings. NoAP CIs.
No T008/meta/gates/predictors/spatialISP/smoothclamp/outcome-driven tuning.
Keepresearch_log rawresults andfinalreport,mirrorrootremote andcommit/push.

[2026-09-12T10:50:34+08:00] T007 smoke20260912-104530 completed2images42rows98APevals35.857011s,exit0at10:48:25+08;58realtests126.41s. Receipt auditpassed samecommitteddisjointmanifest,sharedsource/targetgrads/support,phi0,every-stepnormtransfer/updateequation. Same-forwardcollinearityerror9.99e-16; independentraw-vshybridcosmaxdiff.00013274 fromseparateCUDAforwards,not directiongain. Report/figurechecked,allnegativecasesretained,no parameterchange. Readyforfixednew200.
