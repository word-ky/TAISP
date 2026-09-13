# T020-A IN_PROGRESS - formal source200 running on A6000

2026-09-13T20:34:18.392571+08:00. R032/2654048; plan766ae89, code0546b05, smoke31e0267.
Release: 20260913-202732-taisp-t020a-crossfit.
Active run: 20260913-203203-taisp-t020a-source200-crossfit (started once).
Smoke: 20260913-202806-taisp-t020a-runtime-smoke, exit0.
Baseline10passed2skipped1.79s; increment1 12passed1skipped1.69s;
focused16passed2skipped1.70s; full168passed10skipped7.49s.
CUDA smoke14sourcepairs, two leave-one-image-out matrices,28K3episodes,
56normchecks, allisolation passed, zeroAP. See T020A_smoke_audit.json.
Cohort200 new train2017, four50-image folds,836prior-source/all5000val excluded.
SHA c364fb4d0bf2880318dbbcd22a47605cedccb31f6ee783ffdc56f6650d1f9ade.
Formal collector1400pairs/four150-image Q fits; separate runtime2800adaptiveepisodes,
21predictionfiles/105officialevals. No formal scientific results yet.
Source/CLIP frozen, original pseudo objective/global8D identity/K3/LR.1 unchanged.
Only row gradient gd@Q before unchanged CLIP norm transfer is new.
Do not launch duplicate. Monitor,fetch,audit,report,push,mirror,stopNEEDS_REVIEW.
Heartbeat15minutes; model/gradients/fitting/runtime CUDA preferred.
