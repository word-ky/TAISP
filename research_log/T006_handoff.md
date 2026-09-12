# T006 active run handoff

Updated 2026-09-12T09:25:30+08:00. T006 IN_PROGRESS; T001-T005 accepted/closed.
Authoritative queue: coordination/LATEST.md -> CHATGPT_TO_CODEX_R008_T006.md.
Protocol/pin: T006_plan.md and T006_fcos_pin.json.

Active run: 20260912-092401-taisp-t006-coco200
Release: 20260912-092347-taisp-t006-full
Source: d7d0510 (pushed main); driver8c0f480, FCOSloader6429337.
Remote artifacts: /home/liujianhua/wjq/TAISP/runs/20260912-092401-taisp-t006-coco200/artifacts/study
Remote tmux: autodl-20260912-092401-taisp-t006-coco200
Command: export TAISP_SOURCE_REVISION=d7d0510 TAISP_REAL_MODELS=1;
/home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q &&
/home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.run_t006
--data-root /home/liujianhua/wjq/TAISP/shared/coco200
--output "$AUTODL_ARTIFACTS_DIR/study"

Full tests expected54, then200fixed images/2800variant observations/70APevals.
Check train.log with project AutoDL config; no duplicate launch. No intervention
on mixed scientific results. Raw per-image progress flushes to samples.jsonl.
On finish pack this run under remote runs/, fetch archive through Copy-FromAutodl,
wait exit0, then extract to research_log/remote_runs. Run local
D:/anaconda3/python.exe -m scripts.report_t006
research_log/remote_runs/20260912-092401-taisp-t006-coco200/artifacts/study.
Audit complete rows/pairs, source AND target shared oraclegradients/beforeloss,
source support, fresh zero phi, CLIP-only norm reference, signed products and
zero-update fallbacks. Check figure. Report both AP/AP50/AP75 percase vsbefore
andCLIP, raw/normmatched mechanism, jointbenefit, source-target disagreement,
s1/s2,cleanphi/AP, costs. Preserve negatives. Source/target native losses have
different units; APseveritymacro averages3officialconditions, not pooledCOCOAP.
Bootstrap2000 pairedimageclusters seed20260912. Report NEEDS_REVIEW to mailbox,
commit/push code/fullraw/artifacts; mirror rootresearch_log remote. For remote
research_log/remote_runs symlinks use tar transform to actual runs/ targets.
No meta/T007, stable/JS, gates, newprompts, spatialISP or result-driven tuning.

Validation: baseline51realtests84.76s. Target/isolation4realtests20.60s run
20260912-090851-taisp-t006-target-tests. Smoke20260912-091109-taisp-t006-study-smoke
source8c0f480:53realtests100.40s,2images28rows70APevals16.28047s,exit0.
Smoke receipt_audit.json and figure verified. Report projection/joint/Taylor
focused localtest passed4.67s after adding3step loss summaries.
Target FCOSnative GPUgradient finite/nonzero, weights/buffers frozen, RNGunchanged;
sourceadaptCPU result exactsame with/withouttarget, targetforward trap unused.

Operational history: FCOS download SSL fixed with systemCA (no TLSbypass), one
SSHtimeout retried. Smoke SFTP archive transfer stalled at229376bytes; stopped
only matching scpPID, existing workflow retried legacySCP successfully; full
archive295285bytes SHA745b8c8e0ac1aa105dea5b5edaf0577c58d6a98c33e127eb5538abcebce04e85.
No data/model change. Existing NVMLwarning remains; CUDA experiments work.
Heartbeat every15minutes includes LATEST continuation reading; no duplicate.

[2026-09-12T09:26:39+08:00] Fullrun20260912-092401-taisp-t006-coco200 passed54real tests102.35s (7knownwarnings); first2/200images completed15.1s, no blocking error. Experiment remains active; next heartbeat monitors existingrun and collects only after completion. Rootproject logs mirrored remote.
