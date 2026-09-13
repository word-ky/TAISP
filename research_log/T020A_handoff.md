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

## Recovery

Workflow D:/work/claude-autodl/autodl-workflow-clean, AUTODL_CONFIG_PATH=
D:/work/fightccfa-agin/CVPR2027/TTT-ISP/.autodl/config.json.
Check autodl-logs.ps1 -RunId 20260913-203203-taisp-t020a-source200-crossfit -Lines30.
Remote meta/run.sh retain exact commands; TAISP_SOURCE_REVISION=0546b05.
Absolute /home/liujianhua/wjq/TAISP/.venv/bin/python (not a release .venv).
Collector -m taisp.analysis.collect_t020a writes artifacts/source_fit; next separate
process -m taisp.analysis.run_t018a --config configs/t020a.yaml --manifest
research_log/T020A_train_cohort.json --transport-fits artifacts/source_fit/fits.json
--output artifacts/study. No retry/duplicate without observed blocking failure.

After exit0 archive both smoke/formal raw directories under remote shared/t020a_raw.tar.gz,
sha256sum, Copy-FromAutodl to local .autodl, verify SHA; extract with tarfile filter=data
into research_log/remote_runs. Preserve raw bytes. Run stdlib-only renderer:
D:/anaconda3/python.exe scripts/report_t020a.py --project . --run
20260913-203203-taisp-t020a-source200-crossfit --smoke 20260913-202806-taisp-t020a-runtime-smoke.
It checks1400pairedsupports/fold exclusions/5600norms/codepins and all metrics/matrices.
Renderer prepared in f03d904 before outcomes. Use explicit UTF-8 reads on Windows
(default local Python encoding GBK caused only an AST-read error; no remote issue).
Do not import local torch (known OMP duplicate issue). Fix report bugs only.
Frozen performance rule +.15APcurrent,3of4blocks,4of6conditions,above raw,clean>=-.10,
noisolation/norm/leakage blocker. Alignment diagnostic only. Fail closes fixedglobal
linear transport; pass development candidate pending confirmation. No extra work.

Inspect source_fit/{gradient_pairs.jsonl,fits,alignment,environment,completion}.json
and study/{samples.jsonl,predictions,metrics,summary,diagnostics,transport_fits,
environment,isolation,completion}. Raw smoke has focused/fulltest outputs.
All source/CLIP before-after hashes and zero gradients recorded. Runtime never reads
source task gradients/labels; fit indices exclude complete held-out image blocks.

Append CODEX report; update state/handoff/progress; commit/push allraw/report/code.
Add raw with core.autocrlf=false; docs with true,safecrlf=false. GituserCodex,
emailcodex@users.noreply.github.com. Preserve large rawfiles (under100MB) evenif50MBwarning.
Mirror docs/code/cohort/pins/report/taskfolder/state/mailbox/queue/LATEST via sharedarchive
into remote root; SHAverify. Create new research_log/remote_runs/<id> symlinks to
../../runs/<id>. Record final revision/mirror/report hashes in ignored heartbeat files.
