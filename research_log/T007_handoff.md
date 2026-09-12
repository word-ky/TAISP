# T007 final recovery handoff

Updated 2026-09-12T11:53:11+08:00. T007 NEEDS_REVIEW, no activeexperiment. T001-T006accepted/closed.
Latestqueue R009 via coordination/LATEST.md and CHATGPT_TO_CODEX_R009_T007.md.
No T008/meta/gate/newsource objective authorized. Userrequires FCOSfinite-step
loss vsraw, crossdetectorAPvsnoadapt, cleanphi reduction considered separately.

Fullrun20260912-105119-taisp-t007-coco200,release20260912-105102-taisp-t007-full,source/report0df7e13,
driver4e87904. Disjointselectionc5dc90a,manifest55e83e0 pushedbeforemodels.
200IDs/JPEGhashes research_log/T007_subset.json,overlap0,seed20260913.
ManifestSHA1738d9d504233513893cc4a3f3b1e0a2f58d4665eb4df522e87255082fade1ea.
59realtests127.00s;200images4200rows1400cases21groups98APevals2527.919981s,
finished2026-09-12T11:35:47+08 exit0. No rerunneeded.

Localfullartifacts research_log/remote_runs/20260912-105119-taisp-t007-coco200/artifacts/study.
Remote /home/liujianhua/wjq/TAISP/runs/20260912-105119-taisp-t007-coco200/artifacts/study.
Fullreport research_log/T007_report.md appended to CODEX_TO_CHATGPT.md.
Raw35250130bytes SHAfd41b75af58e6b8eaffdc79405ee2cd604acf85505df9d402f4c9ea2c1d2f20a.
Rawuncompressed retainlocalremoteGitHub. Archive45709998bytes
SHA45312112a35ce567a0e93e136897f0d3de63324d74f802d3b020ec29cdae81d9 verified.
Auditpasses pairing/sharedsourceANDtargetgrads/beforeloss/fixedsupport,
zero phi/weights/every-stepnormequation/stepupdate/predictions/pins.
10emptyepisodespernativevariant (9corrupt+1clean),exactzero phi/losschanges;
40zero hybridgradientdiagnosticsteps. Sameforwardcollinearityerror1.11e-15,
independentCUDAraw/hybridcosmaxdiff.001384,notalignmentgain.

Threeendpointinterpretation: partial scale-benefit replication,APmixed.
FCOSrawbenefit53.33%,hybrid56.67%,paired+3.33ppCI[1.33,5.67].
FCOSloss1raw+.000471,hybrid-.000423;paired-.000894CI[-.002396,.000533]cross0.
HybridabsoluteCI[-.000753,-.000083],hybridminusCLIP-.000958CI[-.001500,-.000417].
AP3vsrawtarget6/6positive,vsbefore4/6positive,negativegamma_s2-.302/contrast_s1-.108;
sourcegamma1/2negative-.231/-.216. Bothdetectorspositiveonly3/6conditions.
Cleanphi3.072558raw->.038796hybrid46.53%lower,paired-.033762CInegative;
yethigherthanCLIP.029245;58/200initialscale>1,57/200finalphi3>raw. Noidentityclaim.
CleanAP3source+.228,target+.011,AP1bothnegative;sourceoracleloss3worse+ .001705.
No simultaneous strongconfirmation ofallthreeendpoints. Awaitresearchjudgment.

Otherreceipts: smoke20260912-104530 58tests126.41s,42rows/98AP35.857011s;
local4tests13.39s+report1test1.19s. Fullreportfigureinspected,artifactlinksresolve.
Operationalfailurefull deployment105051connectionclosed;freshrelease105102success.
PostrunpackagingSSHtimeout(exit255),retrysucceeded;full46MBdownloadcompleted/hashverified.
KnownNVMLwarning only;no modeldata changes/TLSbypass/drivermodification.

Remote rootresearch_log mirrors reports/recovery; for runsymlink archives transform
research_log/remote_runs/ to runs/. Finaldeliverycommit recoverfromgitlog. No active
processsessiontokeepalive. Nextheartbeat fetchqueue,quietifno newresearchtask.
