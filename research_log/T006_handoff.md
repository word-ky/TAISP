# T006 recovery checkpoint

Updated 2026-09-12T09:23:17+08:00. T006 IN_PROGRESS. T005 accepted/closed by R008.
Read coordination/LATEST.md and its T006 continuation, plus T006_plan.md.
No full run launched yet. Target pinned and4focused real tests passed20.60s.
Smoke20260912-091109-taisp-t006-study-smoke source8c0f480 passed53real tests
in100.40s,2images28rows70APevals16.28047s,exit0. All receipts under
research_log/remote_runs. Report and figure visually checked; receipt_audit.json
records exact paired gradients/loss/support, zero phi, norm reference/products.
Report adds three-step loss summaries; focused report tests and regeneration
before commit/deploy. Full study must use no --limit, configs/t006.yaml unchanged.

Next: commit/push code and smoke receipts. Deploy full and run
TAISP_REAL_MODELS=1 pytest -q then python -m taisp.analysis.run_t006
--data-root /home/liujianhua/wjq/TAISP/shared/coco200
--output "$AUTODL_ARTIFACTS_DIR/study". Record actual source revision, release/run
here and in project_state/mailbox; mirror root research_log remote. Expected54
tests,200images2800rows70APevals. Never duplicate an existing active run.

Target FCOS analysis only; source det_pseudo unchanged. Same enhanced images
for both detectors; source/target losses have different units. No meta/T007,
stable/JS, gates, new prompts, spatial ISP or outcome-driven tuning.
On completion collect raw receipts; paired image-cluster analysis2000draws seed
20260912; s1/s2 AP macro means of3condition metrics explicitly not pooledAP;
clean control, all negatives/fallbacks and source-target disagreement. Deliver
full report/code/results and NEEDS_REVIEW to research lead.

Operational history: model SSL download fixed using systemCA, one SSHtimeout
retried; smoke SFTP archive transfer stalled, matching scp stopped, workflow
legacy SCP retry succeeded. No TLS bypass/model change or raw data loss.
