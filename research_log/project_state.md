# Current project state

Updated: 2026-09-12, Asia/Shanghai.

- T001 accepted/closed by R002. Latest research review R003 (d6fd863) accepted T002's implementation checkpoint and required Taylor diagnostics plus preprocessing parity.
- T002 implementation, final200-image experiment and all R003 diagnostics are complete. Status NEEDS_REVIEW; no experiment is active. Read research_log/T002_report.md for final interpretation and exact receipts.
- Final run: 20260912-013248-taisp-t002-coco200-final, source0b8a888, release20260912-013244-taisp-t002-final-parity; exit0 at01:41:17+08,200 images/1200 observations. Raw results and generated reports are under research_log/remote_runs/<run>/artifacts/study/.
- Do not use 20260912-011915-taisp-t002-coco200-fixed as final evidence: it is marked PRELIMINARY because R003 found a one-pixel odd crop mismatch. Earlier 20260912-011328-taisp-t002-coco200 failed on612-square resize and is retained.
- Final parity: 14 cases exact geometry; max normalized RMSE0.005910955 below0.02. Full model/regression suite27 passed; local Taylor/cluster test1 passed.
- Conclusion: generic CLIP direction has weak heterogeneous detector alignment; mean cosine0.0453, detector-loss decrease47.58%, Taylor sign agreement54.75%, mixed AP changes. No automatic meta-training. Follow research lead's next explicit diagnostic task.
- User explicitly authorized direct execution of newly assigned tasks without asking whether to start. Active heartbeat id taisp checks every15 minutes and resumes/implements/tests/experiments/reports/pushes assigned work. It remains quiet on unchanged non-actionable state.
- Remote project /home/liujianhua/wjq/TAISP; use project .autodl/config.json with the existing AutoDL workflow. All final recovery notes/results are mirrored under project-local research_log; current final report supersedes the earlier live handoff.
