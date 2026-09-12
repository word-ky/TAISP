# T011 handoff

2026-09-12T19:53:07+08:00: IN_PROGRESS; R013 (26ff476, pointer ad06718) closes T010 and
issues T011 in coordination/CHATGPT_TO_CODEX.md. Read research_log/T011_plan.md.

Candidate support_confidence_low_50, cutoff 0.8499477751114789, saved mask SHA
7a02e6d66336ef39fbd4b55c3a5aba564aacee56bd2d7405b62ce9d811a511f6.
Seeds 2026091600..2026091799, identity hash ranks within 35 condition/block strata.
All 200 unique masks and counts committed 8dd776d BEFORE any new AP. Preparation
code b49405d; research_log/T011/preparation; matrix SHA
8e9458588ca3a614a32b86fd5c39a58a8618c53ccd7fe17ac190b9940f42deb9.

Smoke 20260912-194144-taisp-t011-offline-smoke completed exit 0 at 19:43:25+08.
20 tests passed; 12,789 AP rows; report and plot checked. Formal run now ACTIVE: **20260912-195353-taisp-t011-coco1000-offline**, release
**20260912-195321-taisp-t011-full**, source/report **8b2143f**.
Started 19:54:03+08; fresh 20 tests passed 2.90s. Exact command in
research_log/T011/full_run_meta.json. Do not duplicate this run.
The active full CPU command is:
python -m scripts.analyze_t010 --study <T009 study> --prepared research_log/T011/preparation
--annotations <COCO annotations> --output <run artifacts/study> --workers 24 --config-batch-size 25
followed by python -m scripts.report_t011 with same study/prepared and T010 candidate-study.
Expected 25,578 AP rows / 3,654 macro rows / 189 panels / 203 configs; 378 exact
anchor comparisons and 4,263 shared detector-condition decisions. Fresh 20 offline tests first.
No GPU/model/ISP/adaptation execution. Use one BLAS/OpenMP/MKL thread each.

Remote root /home/liujianhua/wjq/TAISP, Python .venv/bin/python.
T009 input runs/20260912-144439-taisp-t009-coco1000/artifacts/study.
T010 reference runs/20260912-181436-taisp-t010-coco1000-offline/artifacts/study.
Annotations shared/coco1000_t009/instances_val2017.json.
Full preparation is in deployed releases; do not assume project-root preparation exists.
Workflow D:/work/claude-autodl/autodl-workflow-clean with project .autodl/config.json.
Remote matplotlib absent; plot locally after SHA-verified receipt fetch.

R013 requires both positive macro deltas, both strictly above random p95, both
beating block random medians in at least 4/5 blocks, all three clean deltas >= -.10.
Report all 200 controls, ties, corrected tails (1+#random>=candidate)/201, strict
percentile, linear quantiles, R012 control-pass frequency and descriptive costs.
If any criterion fails, stop this scalar-gate route; no new cohort for candidate.
Even if passing, await research review. No automatic T012, learned gate, feature
combination, new objective, spatial ISP, predictor or meta-training.

Next heartbeat: inspect active run train.log/tmux and completion/report_receipt.
On success, fetch run archive with SHA verification, plot locally using
python -m scripts.plot_t011 --study <local run artifacts/study>, inspect figure,
check all 200 controls/statistics and write T011_report.md plus NEEDS_REVIEW.
Commit/push all results and mirror durable logs to remote project root.
Report automatically compares 378 candidate/endpoint evaluations with T010.
Preserve all negatives and distinguish developmental evidence from validation.
