# Current project state

2026-09-13T04:54:23.3490338+08:00 — T013-E IN_PROGRESS under R019/74cbdce; T013-D CLOSED.
Pre-outcome plan604528c. Baseline6passed10.06s; focused9passed7.14s.
New analysis-only deterministic runner ready; no production/model/data/loss change.
Three exact no-update repeats gate the three independent fixed SGD1e-3 probes.
Launcher must export CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python. No other workaround.
All input/checkpoint/control hashes pinned; unsupported operator or exact mismatch stops.
Next commit/deploy, full remote regression then one real diagnostic. Stop after T013-E.
