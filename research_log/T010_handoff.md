# T010 active full offline evaluation

2026-09-12 18:14+08: T010 IN_PROGRESS, R0126012f1d/latest1a4df0c. T009closed.
Active formal run **20260912-181436-taisp-t010-coco1000-offline**,
release20260912-181357-taisp-t010-full, source/report **9efe032**.
Started18:14:44+08. Fresh14affectedoffline/regressiontests passed2.45s, then
CPU-only12-worker officialAP evaluation. Do not duplicate or rerun models.

## Frozen contract and inputs

Readcoordination/CHATGPT_TO_CODEX_R012_T010.md andT010_plan.md(planfd88938).
SevenidentityscalarsexactlyasR012,zero-vectorcosine0andemptysupportconfidence0.
Full7000-rowpreparation generatedby32a89e3 andcommitted **901560f before gatedAP**.
42gates (7scores*2orientations*3coverages) +no-adapt/fullhybridanchors.
Pooledrankoverall7000observations,notbycondition/block;tiesimageIDthenimmutable
hybridrowordinal. Exactly1750/3500/5250 selectedrows; alldecisionsandcutoffs saved.
Do not change afteroutcomes. StageA didnotreadannotations/predictionfiles.
All7scoresavailable;70zero-vectorcosinescoded0. PreparationinputSHA matchesT009.

Local preparation research_log/T010/preparation; remote root sameprojectrelativepath.
Frozeninputstudy remote runs/20260912-144439-taisp-t009-coco1000/artifacts/study.
Annotations shared/coco1000_t009/instances_val2017.json,officialhash e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f.
LocalT009rawarchivefullydownloadedandhashed;samplesoriginalpreserved,Gziptracked.

## Validation and actual command

Core/preparation7tests localpassed11.83s. OfflineAPsmoke run20260912-180743,
source901560f:13tests2.35s,2images14rows44configs21panels2772officialresults,
16.57865s,exit0at18:08:12. Exact0/100anchorhashes;report924decisionsaudited.
Reportcriterionboundaryunit1pass15.98s. StandaloneplotOMP15 fixedbyremoving
unusedgatingimport(readlabelsfromsavedreport),noenvironmentchange;plotexit0/visualchecked.
Full14tests2.45scoverlatestreportcriterion too. No model/ISP/adaptation rerun.

ActualrunexportsTAISP_SOURCE_REVISION=9efe032,OPENBLAS_NUM_THREADS=1,
OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,CUDA_VISIBLE_DEVICES=empty.
Thenpytesttests/test_t010_gating.py tests/test_t009_replication.py tests/test_t008_analysis.py -q,
then .venv/bin/python -m scripts.analyze_t010 --study <T009study> --prepared
/home/liujianhua/wjq/TAISP/research_log/T010/preparation --annotations <above>
--output "$AUTODL_ARTIFACTS_DIR/study" --workers12,
then -m scripts.report_t010 --study "$AUTODL_ARTIFACTS_DIR/study" --prepared <above>.

## Completion work

Monitorviaworkflow; finaldriverexpects21panels44configs6groups =5544AProws,
including5292newgatedevaluationsand252exactreusedT009anchorvalues.
EachpanelstoresselectedIDs,decisionhashes,composedpredictionhashes,allCOCOstats.
Rawanchors+precommittedmatrixreconstructallcompositions; noGBduplicatedJSONneeded.
Do not interpretpartialpanels ormodifygrid. Fetchfinalruncompressedarchive,
verifySHA,extractunderresearch_log/remote_runs/<run>. Reportscriptalreadyrunsremote.
Generatelocalplotwithscripts.plot_t010;remotematplotlibabsent. InspectPNG.
Auditall44configurations/5544AProws,allshareddecisions,andindependentlyreconstruct
macro/sign/cleancriterionarithmetic. Retainallnegativeconditions/blocks/configs.
Reportallpassingconfigurations,nounreportedoutcomeselection;developmentcohortonly.
Cleanselectedcoverage<=.5,all3cleanAPdeltas>=-.1,bothindependenttargetmacro>0,
each>=4/5positiveblocks. SourcecannotreplaceFCOS/SSD. Effectivephi,safety/coverage
andcoarsereceipt-derivedlatenciesalreadyinpreparation/safety.json.
WriteT010_report.md,appendCODEX_TO_CHATGPT.md,statusNEEDS_REVIEW,commitpush,
mirrorfinalprojectlocaldocs/resultsunderremoteprojectrootresearch_log.
No T011,learnedgate,featurecombination,loss,predictor,spatialISP ormetaautomatically.

WorkflowcwdD:/work/claude-autodl/autodl-workflow-clean;projectconfig.autodl/config.json.
Projectread/edit/gitfromprojectcwd. Neverprintconfigsecrets. Remote root/home/liujianhua/wjq/TAISP.
InitialscoredeploySSH255afterextract;verifiedfiles/completedonlycurrentlinkstep.
Otherknownfailure:firstbaselinewrongremoteprojectcwdexit4;correctreleasepassed.
