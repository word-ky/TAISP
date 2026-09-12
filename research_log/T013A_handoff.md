# T013-A handoff — required plumbing complete, optional limitation reported

2026-09-13T00:14:32.7268912+08:00 — NEEDS_REVIEW. Read T013A_report.md first.
Plan cd3060c committed before implementation bfd2484. No source changes after tests.
Release 20260913-000723-taisp-t013a-plumbing.
Receipts: research_log/remote_runs/20260913-000838-taisp-t013a-feasibility-fixed.
Full remote regression 89 passed / 10 skipped, 5.09s. Local focused 11/3,10.43s.
Local full 84/14/1 failure (missing pycocotools) is retained; remote covers it.
Synthetic receipt T013A/synthetic_sanity.json: .005294277333 -> .001558897318.
Strict CUDA smoke failed exact parity, unchanged assertion in scripts/smoke_t013a.py.
Diagnostic source T013A/diagnose_cuda.py and cuda_diagnostic.json retain differences:
full repeat 8.58842e-6, connected 9.04803e-6, image 5.06639e-6; phi0/head gradients
.1245225221/.0540611036. Mandatory same-gradient CPU parity exact. No GPU bitwise claim.
First run 000746 failed before tests (relative venv). Second initial SSH launch
failed before tmux; manual resume lost output. Subsequent logged commands/results
are authoritative. No active experiment or transfer. Do not rerun unchanged work.
Next action await research review/new explicit task; never automatically start
T013-B, real/source/meta-training, new cohort, spatial ISP, gate/alpha/cap search.
Remote root /home/liujianhua/wjq/TAISP. Use absolute project .venv/bin/python;
release does not contain a .venv link. Workflow config remains project .autodl/config.json.
