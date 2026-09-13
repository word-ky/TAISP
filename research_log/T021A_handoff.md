# T021-A NEEDS_REVIEW - fixed horizontal-flip support filter closed

2026-09-13T23:00:03.394627+08:00. R033/a5bae26 completed; plan0207885, code daa79e5, smoke d276fc0.
Formal20260913-222407-taisp-t021a-source200-consensus,release20260913-222014-taisp-t021a-consensus.
Smoke20260913-222054-taisp-t021a-runtime-smoke. Formal22:24:12-22:43:09+08 exit0,
1128.431519s collection/evaluation. No active job.
200newtrain2017/four50blocks,1036prior-source/debug/all5000valexcluded; cohortSHA
0fdb6a815d380104542c324ba2146944b47232dae22a416fa12d5f72a5d29725.
2800teacherforwards/2800adaptiveepisodes/21predictionfiles/105officialCOCOevals.
Model/gradient/adaptation/IoU onA6000CUDA; smallsorting/report/APaggregationCPU.

Macro AP raw50.3312941972,current50.1513089671,consensus50.1209418221.
Candidate-current -0.030367145082AP; candidate-raw -0.210352375112AP.
Only1/4positiveblocks;4/6positiveconditions. Materiality/block/above-rawcriteriafail.
Clean-current +0.263891538209AP,clean-raw -0.149335803521AP; cleancriterionpasses.
AP50 +0.044617062912/AP75 +0.022649037945 versuscurrent arediagnostic only.
No eligible development candidate; close fixed horizontal-flip support filter underR033.
Current Ours remains authoritative despite below-raw AP on this cohort; no tuning.

Meanretention84.71746%defined,9672retainedsupports across1400episodes. Zero candidate7
versuscurrent5,allcorrupted;clean0empty and100%updateboth. No cleanselectivity signal.
Meanoriginaleligible8.750714,currenttop20 7.977143,retained6.908571. Original confidence
weights/boxes/classes preserved. Extra flip/matchingmean.029433s; teacherinclusive
current~.332s,candidate~.360-.361s. Full diagnostics/blocks/conditions retained.
Full175passed10skipped7.52s; focused13passed2skipped2.50s. Smoke28CUDAK3episodes,
168retainedsupports/0APpassed. Formal2800isolation/1400consensuschecks/9672original
supports/baselinepreservation/17actualreleasehashpins pass. All modelhashes unchanged.
66rawfiles59,995,881bytes fetched/SHAverified; archiveSHA
577dea0d1a264bb9e7bbc1bc16e41d7fa57866e523dd463af575b2c4eca38547.
ReportT021A_report.md; T021A/complete_tables.md,receipt_audit.json,remote_code_hashes.json,
artifact_manifest.json; raw runs retained. StopNEEDS_REVIEW for new explicit task.
No score/IoU/topk/augmentation/fallback sweep,newcohort,FCOS/SSD/val,spatial/meta/predictor/
gate/dose/native/Q/T017forensics experiments. Continue15minuteheartbeat quietly unchanged.

Recovery: completed raw runs remain in research_log/remote_runs and remote runs/. Report renderer scripts/report_t021a.py is stdlib-only; use PYTHONUTF8=1 on Windows. Do not rerun the experiment. Final push/mirror/report hashes are recorded in .autodl/last-heartbeat.json.
