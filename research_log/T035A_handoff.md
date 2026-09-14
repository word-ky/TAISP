# T035-A / R053 — literature/design complete; NEEDS_REVIEW

Authority: 44030b0 / coordination/CHATGPT_TO_CODEX_R053_T035A.md. Prior T034 scientific FAIL accepted; old H/C/N mixing family closed. No GPU/model experiment, finite ISP update, implementation or deployment change in T035-A.

Deliverables: T035A_literature_review.md (15 recent primary-source papers, including 8 seeds and 6 independent priority-conference additions, plus a WACV addition; 2 older boundary checks), T035A_overlap_matrix.md (12 required branches plus restricted untested scalar), T035A_method_cards.md (exactly one NOT IMPLEMENTED card), T035A/search-notes.md and papers.csv/papers.md. Seven principles and access/reading limitations are explicit.

Strongest threat: AdaptiveISP already combines task-driven image-adaptive ISP and a frozen detector. VPTTA covers frozen-model per-image input optimization; learned losses have NC-TTT/older meta-TTT/Learn-to-Steer antecedents. Remaining untested project question: source-supervised current-observation scalar producing signed and finite-step useful ISP gradients, beyond matched controls. Novelty and usefulness remain unproven. Second spatial card declined because capacity is not supervision.

T035 is documents only: no regression/model tests needed; bibliography count, local links, file hashes and changed-path scope checked before commit. Final commit/push and document-mirror receipts are recorded in .autodl/last-heartbeat.json and heartbeat-history.jsonl after publication. T034 raw evidence and all implementation files remain unchanged.

Next action: read LATEST and any explicit new research-lead continuation. Do not implement the proposed card until next review/task. Heartbeat taisp every15minutes; unchanged/non-actionable state stays quiet. Future authorized model work prefers A6000 CUDA. No active model job. Preserve T034 exclusion lineage (3911 prior/reserved IDs plus val/evaluation). Use one-off git gc.auto=0/maintenance.auto=false; do not force-delete Git temp packs or artifacts.
