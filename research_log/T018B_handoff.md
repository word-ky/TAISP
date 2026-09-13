# T018-B IN_PROGRESS — formal source500 GPU confirmation

2026-09-13T16:19:05.2298210+08:00 — R030/7902e3a; plan/cohort958d898; code d8ff14f.
Active run:20260913-161818-taisp-t018b-source500 onA6000cuda:0.
Release:20260913-161116-taisp-t018b-source500. Expected~50minutes; do not duplicate or restart.
Smoke:20260913-161225-taisp-t018b-runtime-smoke exit0,150passed10skipped7.06s,
2images/28adaptiveepisodes/K3CUDA/allisolationpassed/0AP. Focused12passed2skipped1.77s.
Fixed500newsourceimages,5blocks100;136prior-source/all5000val excluded. CohortSHA e7a771126ae2fee9844dde456648b50f4c01ebcdc404318c21f9a33260da7b17.
Unchanged accepted native/current model paths andhyperparameters. Protected11modulepins inT018B_method_pins.json.
Formalexpected3500teacherforwards/7000adaptiveepisodes/21predictionfiles/126officialevals.
Next: monitor this run at heartbeat; when completed fetch BOTH smoke andformal raw directories,
verify archival hashes, create T018B report with allAP/AP50/AP75 and5block/7conditionresults,
apply exactR030criteria fromsummary.json, retainnegatives,appendCODEXmailbox,commit/push/mirrorA6000.
AP50/AP75 are diagnostics,notgates. NoFCOS/SSD/COCOval/deploymentreplacement/tuning/T017forensics.
Reports canreuse scripts/report_t018a.py table formatting but need B-specific counts/verdict/provenance.
StopNEEDS_REVIEW aftercompletion. UserGPUpriority and15minuteheartbeat active; quietunchanged.
