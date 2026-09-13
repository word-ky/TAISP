# T021-A IN_PROGRESS - formal source200 flip-consensus run on A6000

2026-09-13T22:25:13.164576+08:00. R033/a5bae26; pre-outcome plan0207885; code daa79e5;
smoke/report-preparation d276fc0. Release20260913-222014-taisp-t021a-consensus.
Active run: 20260913-222407-taisp-t021a-source200-consensus (launched once).
Smoke: 20260913-222054-taisp-t021a-runtime-smoke, exit0.
Baseline12passed2skipped1.77s; increment1 11passed2skipped2.42s;
focused13passed2skipped2.50s; full175passed10skipped7.52s.
Smoke28CUDAK3episodes/14consensus episodes/28teachers/0AP; all168retained supports
preserve original boxes/classes/scores, baseline supports unchanged, all isolation passed.
No empty smoke supports; unit tests verify empty-consensus identity/no fallback.

200newtrain2017 images/four50blocks; excludes1036prior-source/debug/all5000val.
CohortSHA0fdb6a815d380104542c324ba2146944b47232dae22a416fa12d5f72a5d29725.
Original+horizontal flip teachers, score>=.50 each, sameclass IoU>=.60,
geometric-confidence greedy/index ties/top20 aftermatching; original confidences for
unchanged current-Ours loss weights. Frozen detector/CLIP/global8D ISP/K3/LR.1 unchanged.
Formal expects2800teachers/2800adaptiveepisodes/21predictions/105officialCOCOevals.
All model/gradient/adaptation and IoU work CUDA; tiny sorting/reports/AP aggregation CPU.
No formal scientific results yet. Do not launch duplicate or alter constants.
After completion fetch both raw runs, audit/report/push/mirror; stop NEEDS_REVIEW.
Heartbeat every15minutes; only execute explicit new research tasks after this one.

## Recovery and final reporting

Use workflow D:/work/claude-autodl/autodl-workflow-clean, AUTODL_CONFIG_PATH=
D:/work/fightccfa-agin/CVPR2027/TTT-ISP/.autodl/config.json.
Check autodl-logs.ps1 -RunId 20260913-222407-taisp-t021a-source200-consensus -Lines30.
Exact command in remote meta.json/run.sh: export TAISP_SOURCE_REVISION=daa79e5;
/home/liujianhua/wjq/TAISP/.venv/bin/python -u -m taisp.analysis.run_t018a
--config configs/t021a.yaml --manifest research_log/T021A_train_cohort.json
--output "$AUTODL_ARTIFACTS_DIR/study". No interpreter inside release; use absolute path.

After exit0 archive both smoke/formal raw directories under remote shared/t021a_raw.tar.gz;
sha256sum and Copy-FromAutodl to local .autodl; wait for download completion before reads.
Verify SHA then extract tarfile filter=data into research_log/remote_runs. Keep rawbytes.
PYTHONUTF8=1; D:/anaconda3/python.exe scripts/report_t021a.py --project . --run
20260913-222407-taisp-t021a-source200-consensus --smoke 20260913-222054-taisp-t021a-runtime-smoke.
Stdlib renderer prepared before outcomes and its audit_supports passed real smoke.
It audits frozen/originalscore support receipts, ranking/eligibility/ties/one-to-one,
original top20 baseline, K3/identity/isolations, codepins and prints fullAP/support tables.
No local torch imports (known duplicateOMP). Correct only actual report bugs, no new
scientific runs. Verify actual remote release code files against14protected+3authorized
pins and save research_log/T021A/remote_code_hashes.json before claiming release match.
Report candidate-current/raw AP, all blocks/conditions, cleanAP,AP50/AP75, all support
retention/zero incidence andphi/update/extra flip/matching overhead. Pre-top20original
count is retention denominator; geometric mean only affects matching, notlossweights.

FixedR033 gate: macro-current>=+.10AP,3of4positiveblocks,4of6positiveconditions,
above raw,clean>=current-.10AP,no support/isolation/repro blocker. Retention/AP50/AP75
are diagnostics only. Fail closes fixedhorizontal-flipfilter; pass developmental only.
No score/IoU/topk/augmentation/fallback changes,newcohort,FCOS/SSD/val,spatial/metawork.

Append CODEX final report; update state/handoff/progress. Commit/push docs/code and allraw.
Raw gitadd core.autocrlf=false,safecrlf=false; docstrue/safecrlffalse; userCodex,
emailcodex@users.noreply.github.com. Mirror complete report/taskfolder/cohort/pins/code/
state/mailbox/queue/LATEST to remote root via sharedarchive; SHAverify. Make new
research_log/remote_runs/<run> symlinks to ../../runs/<run>, without deleting old paths.
Final revision/mirror/report hashes into .autodl/last-heartbeat.json and history.
