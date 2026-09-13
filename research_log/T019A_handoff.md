# T019-A IN_PROGRESS — formal four-component-family source200 run

2026-09-13T18:27:30.1252481+08:00 — R031/d60e72c. Pre-outcomeplan/cohort/methodpins b49c432; codeee98de0; smokereceiptd047312.
Active run20260913-182636-taisp-t019a-source200-components onA6000cuda:0; release20260913-182322-taisp-t019a-components.
Expected~50minutes; do notduplicate/restart. Full162passed10skipped6.84s; focused24passed2skipped1.82s.
Smoke20260913-182437-taisp-t019a-runtime-smoke exit0:2images/70adaptiveepisodes/56candidateepisodes,
224exactfloat32activecomponent sums,allK3/statechecks pass,0AP. SeeT019A_smoke_audit.json.
200newimages/four50blocks,636prior-source/all5000val excluded;cohortSHA534b17343ccba995beb9d112d552eefd7f1b2116479c2132dd99265b1d6735b4.
Exactlyno_adapt,current_ours,native_cls,native_conf,native_roi,native_conf_roi; nofullnativecandidate.
Expected1400teacherforwards/7000adaptiveepisodes/42predictionfiles/210officialevals.
Native component selection onlyauthorizedmethodchange; defaultfull behavior preserved;10protectedmodules unchanged.
Nextheartbeatmonitor existingrun. Whenexit0complete,fetch BOTH182437smoke and182636formal,
verifyarchiveSHA,reporteachcandidate's allconditions/blocks/AP/AP50/AP75/diagnostics andfrozeneligibility/selection.
Newsummaryschema: candidates[name].groups/flags/eligible/positive_blocks; gate.eligible_candidates/selected.
Do notuseoldA/Breporter unmodified; T019needsfourcandidate summary. Retainallnegativeoutcomes.
ThenappendCODEXmailbox,updatehandoff/state/progress,commit/push/mirrorA6000,stopNEEDS_REVIEW.
NoFCOS/SSD/val,spatialISP,T017forensics,hyperparametersearch ordeploymentreplacement. GPUpriority/15minuteheartbeatactive.

2026-09-13T18:30:17.1461645+08:00 — First12/200images completed without error; noAPoutcomes yet. Renderer scripts/report_t019a.py ready,syntaxpassed.
Whenformalexit0complete, tar/fetch both20260913-182437-taisp-t019a-runtime-smoke and20260913-182636-taisp-t019a-source200-components.
VerifyarchiveSHA,extract underresearch_log/remote_runs, then:
D:\anaconda3\python.exe scripts/report_t019a.py --project . --run 20260913-182636-taisp-t019a-source200-components --smoke 20260913-182437-taisp-t019a-runtime-smoke
Auditall1400pairedconditions/fiveadaptivemethods,K3/identityphi0/activecomponentfloat32sums/finitegradients,
protected10modulepins and3precommittednew/modifiedLFsourcehashes againstcodeee98de0. RunsummaryselectionisfrozenR031.
Inspectallcandidateflags,condition/blockAP/AP50/AP75 andtie-breakerreceipt. UpdateCODEX/state/handoff/progress,
commit/pushfullreceipts/report,mirrorA6000 andstopNEEDS_REVIEW. Do notduplicateactive182636runoraddvariants.
