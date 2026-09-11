# Current project state

Updated: 2026-09-12 (Asia/Shanghai).

- Latest research decision: `19f362d`, R002 accepts and closes T001.
- T001_handoff.md is the historical implementation receipt; its pending-acceptance statement is superseded by R002.
- Current task: T002, IN_PROGRESS. User explicitly authorized direct execution of all clearly assigned queued tasks, without asking whether to start. Frozen real CLIP and detector integration tests have passed. The fixed 200-image COCO-val subset is available on A6000 under shared/coco200. Full protocol: research_log/T002_plan.md.
- Current A6000 run: `20260912-011328-taisp-t002-coco200`, full 200-image/1200-corruption study. Tested source revision `4922ffe`; release `20260912-011324-taisp-t002-full`. Do not launch a duplicate. Two-image smoke completed (12 corruption samples, all 25 comparisons) and all 25 tests passed before this run.
- Active Codex thread heartbeat: id `taisp`, name `TAISP 项目检查`, interval 15 minutes. It now executes clearly assigned new tasks, resumes existing implementation/experiments, collects results, reports and pushes stages automatically within their assigned scope. Check project logs to avoid duplicate jobs. No repeated unchanged-status notifications.
- No TAISP automation was found before creating this one.
- Durable implementation receipts: research_log/T001_handoff.md and research_log/remote_runs/20260912-003812-taisp-t001-a6000/.
