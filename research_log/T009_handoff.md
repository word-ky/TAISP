# T009 active full-run handoff

2026-09-12T14:46:53+08:00: T009 IN_PROGRESS, **active full run 20260912-144439-taisp-t009-coco1000**.
Remote root /home/liujianhua/wjq/TAISP; release20260912-144331-taisp-t009-full.
Source/report69bfb66. No other active T009 run. Do not duplicate or change current
release while running. First6/1000 images finished normally (~6s/image).

## Read/recover

Research R011 in coordination/CHATGPT_TO_CODEX_R011_T009.md via LATEST5885716.
T008closed. Plan92aa4df;cohort e98a7fd before any T009model execution.
Data shared/coco1000_t009;manifest T009_subset.json 1000unique IDs,0/0historical/
T007overlap,5fixed200blocks from random.Random(20260914)selectionorder.
Manifest SHA155bb6f047d374342623f48488bcb2b33601a4ecbd372976a433c1b289546e49.
SSD pinbe5c797 exact COCO_V1 SHA b556d3b43ab6c3f63d81bfb8835fe8756ac22da664357da100dccf96b6a6b42d.
Source/FCOS/CLIP/T007adaptation untouched. No target gradients or oraclelossstudy.

## Evidence before full run

SSDcompat retry143109 passed1test66.32s; first142824 failed PythonTLS chain,
normal systemcurl verifiedTLS and downloadedsameofficialweights,no modelsubstitute.
Pin/modelgate receipt T009_ssd_gate_receipt.md. Smoke20260912-143436,sourcebe5c797:
70real/regressiontests passed183.95s(9knownwarnings);2images42rows84predfiles252
APevals16.453258s,exit0. Auditsharedsupport/phi0/all3updates/normtransfer/terminal
no fourthupdate. All smokeoutcomes retained but notusedfortuning. Only subsequent
changes are reportpartialcoverage labeling and standaloneplot; these ranlocally
exit0 and figureinspected. No affected adaptation/driver change requiring retest.

## Actual full command and monitoring

Use existing workflow scripts from D:/work/claude-autodl/autodl-workflow-clean,
AUTODL_CONFIG_PATH points to thisproject .autodl/config.json (never printsecrets).
Logs: autodl-logs.ps1 -RunId 20260912-144439-taisp-t009-coco1000 -Lines30.
Remote command exports TAISP_SOURCE_REVISION=69bfb66, then .venv/bin/python
-m taisp.analysis.run_t009 --data-root /home/liujianhua/wjq/TAISP/shared/coco1000_t009
--output "$AUTODL_ARTIFACTS_DIR/study", followedby
-m scripts.report_t009 --study "$AUTODL_ARTIFACTS_DIR/study".
Artifacts remote runs/20260912-144439-taisp-t009-coco1000/artifacts/study, logtrain.log.
Remote Matplotlib absent; finalfigure generated locally by scripts/plot_t009.py,
using existing D:/anaconda3/python.exe, then copied back; no remotepackagechanges.

## Completion work

Expect1000images21000variantrows,84predictionJSONfiles,504officialaggregate+
5blockAP/AP50/AP75 evaluations. Saveallrawreceipts,negativeconditions/blocks and
safety/runtime/normratio/scaledistributions. Fetcha compressedarchiveof fullrun,
verifyhash andextractunderproject research_log/remote_runs (avoid many small SCPs).
Audit cohort/weights/ids/counts/sharedsupport/phi0/allthree updateequations and
hybridscale unchanged. Report allthree detectors,all7conditions,all4methods,
aggregate+fiveblockvalues,hybrid-minus-noadapt/raw signs with fixedR011criteria.
No AP CIs/perimageAP/proxymining. Plotlocally andinspect. WriteT009_report.md,
append CODEX_TO_CHATGPT.md, NEEDS_REVIEW onlyaftercomplete audit/report;commitpush
andmirrorprojectrootresearch_log onA6000. Preservefailurelogs andrawdata.
No T010/meta/loss/gate/cap/predictor/spatialISP automatically.
