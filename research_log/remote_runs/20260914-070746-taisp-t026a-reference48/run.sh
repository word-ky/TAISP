#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-070746-taisp-t026a-reference48'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-070746-taisp-t026a-reference48'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-070746-taisp-t026a-reference48/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-070621-taisp-t026a-reference && export TAISP_SOURCE_REVISION=fb8de70 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.source_roi_memory_reference --cohort research_log/T026A_train_cohort.json --candidate-root /home/liujianhua/wjq/TAISP/runs/20260914-070245-taisp-t026a-candidates48/artifacts/candidates --candidate-lock research_log/T026A_candidate_lock.json --memory-root /home/liujianhua/wjq/TAISP/runs/20260914-065856-taisp-t026a-memory1280/artifacts/memory --memory-lock research_log/T026A_memory_lock.json --output "$AUTODL_ARTIFACTS_DIR/reference"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
