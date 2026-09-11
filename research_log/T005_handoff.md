# T005 handoff

2026-09-12: T005 IN_PROGRESS following R00724114d2. T004 accepted/closed.
Read T005_plan.md for predeclared matching/weights/loss/zero-cosine conventions.
Source f64151a includes study driver; local report_t005.py added afterward.
All experiments use existing project remote /home/liujianhua/wjq/TAISP.

Completed baseline42 real tests8.23s; fixed ROI5tests4.67s; native loss/reset
15tests81.40s. Failed GPU repeated update test retained at
research_log/remote_runs/20260912-063551-taisp-t005-loss-tests (1fail14pass).
Replaced test device with CPU for same1e-6 repeat tolerance; GPU freeze/grad/reset
invariants kept. No method adjustment. Local empty-support assertion corrected
from raw image bitwise equality to exact ISP(phi0), because baseline arithmetic
roundoff exists; phi and no-update image remain exact relative to initial ISP.

Smoke run20260912-064017-taisp-t005-study-smoke, sourcef64151a,
release20260912-064013-taisp-t005-smoke:50real tests pass,2images56observations,
all63AP evaluations,exit0at06:42:23+08. Fetch currently under way; wait for SCP
completion before running local report. Report focused4tests passed11.02s.
No full study launched yet. Next: audit smoke, run python -m scripts.report_t005
research_log/remote_runs/20260912-064017-taisp-t005-study-smoke/artifacts/study;
inspect report/figure, commit/push, deploy and full run with fresh51-test suite,
no --limit. Configt005.yaml fixed200images6corruptions+clean4variants,5600rows.

Remote launch uses TAISP_REAL_MODELS=1 and TAISP_SOURCE_REVISION=<commit>,
/home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q then same python -m
 taisp.analysis.run_t005
 --data-root /home/liujianhua/wjq/TAISP/shared/coco200
 --output "$AUTODL_ARTIFACTS_DIR/study".
Use existing AutoDL PowerShell scripts, project .autodl/config.json. Do not launch
a duplicate full study; replace this note with actual run details after launch.

After full completion collect all raw logs/predictions/rows and postprocess with
scripts.report_t005. Use zero-coded aggregate cosine explicitly, valid-only cosine
separately; no dropped fallbackcases. Clean AP/phi controls mandatory. Append full
report to CODEX_TO_CHATGPT.md and set NEEDS_REVIEW, then mirror reports remote and
commit/push. Do not modify research lead mailbox. No T006/meta/spatialISP until
explicit nexttask. User authorizes direct execution of queuedtasks.
