#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-070245-taisp-t026a-candidates48'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-070245-taisp-t026a-candidates48'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-070245-taisp-t026a-candidates48/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-070133-taisp-t026a-candidate && export TAISP_SOURCE_REVISION=051d7ee && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.source_roi_memory_audit --manifest research_log/T026A/candidate_images.json --memory-root /home/liujianhua/wjq/TAISP/runs/20260914-065856-taisp-t026a-memory1280/artifacts/memory --memory-lock research_log/T026A_memory_lock.json --output "$AUTODL_ARTIFACTS_DIR/candidates"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
