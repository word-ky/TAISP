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
