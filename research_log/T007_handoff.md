# T007 active full-run handoff

Updated 2026-09-12T10:52:27+08:00. T007 IN_PROGRESS; T001-T006 accepted/closed. R009research task
in coordination/LATEST.md -> CHATGPT_TO_CODEX_R009_T007.md. No T008/meta/gate.

Active run 20260912-105119-taisp-t007-coco200,release20260912-105102-taisp-t007-full,source/report0df7e13,driver4e87904.
Remote root /home/liujianhua/wjq/TAISP via project .autodl/config.json.
Artifacts /home/liujianhua/wjq/TAISP/runs/20260912-105119-taisp-t007-coco200/artifacts/study.
Command export TAISP_SOURCE_REVISION=0df7e13 TAISP_REAL_MODELS=1;
/home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q &&
/home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.run_t007
--data-root /home/liujianhua/wjq/TAISP/shared/coco200_t007
--output "$AUTODL_ARTIFACTS_DIR/study"
No --limit. Fresh59tests precede200fixed images/4200rows/1400cases/98APevals.
Check train.log, no duplicate launch. Collect only after finished exit0.

Disjoint selection predeclaredc5dc90a; fullmanifest55e83e0 pushedbefore anynew
subsetmodel execution. Seed20260913,excludehistorical200,randomsample sorted4800
then sort200;overlap0. research_log/T007_subset.json allIDs/JPEGhashes,
SHA1738d9d504233513893cc4a3f3b1e0a2f58d4665eb4df522e87255082fade1ea.
Annotationhashsamehistorical. Dataset remote shared/coco200_t007.

Fixedhybrid computes sourcepseudo andCLIPgrads atcurrentphi eachstep,eps1e-12,
gd*norm(gc)/(norm(gd)+eps), gd0=>exact0. Samefrozenmodels/promptbank/sourcefixed
support/8DzeroISP/lr.1/hardclamp/K1/3. Step3gradients diagnostic-only; no4thupdate.
Target/labels absentfromadapter. Sameenhancedimages for source/FCOS.

Smoke20260912-104530-taisp-t007-study-smoke release20260912-104516,source4e87904:
58realtests126.41s,2images42rows98APevals35.857011s,exit0at10:48:25+08.
Allreceipts/figure/report audited;42rows14cases exactpairing, sharedoracle
source/targetgrads/beforeloss,support,zero phi,epsnormrelation andstep equations.
Sameforward collinearityerror9.99e-16,independentCUDAraw/hybridcosmaxdiff.00013274;
do notcall thisdirectiongain. Raw368406bytes SHAa52395fbb2976ecf3ead5a3064e6cdfcbd2a8247bbf92362c8adeee3af056c7e.
SmokearchiveSHA8fb98fb28860953bab1f05d4292d06dd90ae0ffa9249d0c05ada871dd9b2a28c.
Local4tests13.39s plusreportflip/stratum1test1.19s pass. No code/test failure.

Operationalfailure: deployment20260912-105051 upload connectionclosed,
legacySCP succeeded then extractionSSHclosed(exit255); no experimentstarted.
Retryfreshrelease20260912-105102-taisp-t007-full succeeded. Actualfullrun uniqueabove.
KnownNVMLwarning remains, CUDAworks; no driver/TLS/scientificchanges.

Oncompletion packrun,fetcharchive usingworkflow, waitexit0thenextract under
research_log/remote_runs. Runlocal PYTHONUTF8=1 D:/anaconda3/python.exe -m
scripts.report_t007 research_log/remote_runs/20260912-105119-taisp-t007-coco200/artifacts/study.
Audit4200rows21variantcasegroupsx200,1400cases,98APevals,committed
manifestexactmatch,sourceANDtargetoraclegrads/beforelossshared,fixedsource support,
phi0,every-stepnormscale/updateequation,collinearity,zerofallback. Saveaudit/hash.
Reportallnegativeconditions,AP/AP50/AP75 K1/3 vsbefore/CLIP/raw,
all3pairedoraclelosscontrasts,raw/hybridlossflipsbyfixedratio bins[0,1),[1,2),
[2,4),[4,inf),clipzero; cleanprimaryphi/AP/saturation/support. Cosinecollinear,
no alignmentgainclaim. 2000pairedimageclusterseed20260912; unequalratio strata
clusterweightedfractions,noAP CIs. Meanloss andbenefitrates distinct,lossunits
source/FCOSdiffer. Freezehypotheses/protocol; no tuningfromoutcomes.
Complete T007_report.md,mailbox NEEDS_REVIEW,allrawreceipts,commit/push,remote
rootresearch_logmirror. For symlink runarchives transform research_log/remote_runs/
to actual runs/. Thenawait researchnewtask; no T008/meta inferred.

[2026-09-12T10:54:02+08:00] Fullrun20260912-105119-taisp-t007-coco200 passed59realtests127.00s(8knownwarnings). Study nowloading frozenmodels;no blockingerror,no duplicate.
