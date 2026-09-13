#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-065856-taisp-t026a-memory1280'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-065856-taisp-t026a-memory1280'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-065856-taisp-t026a-memory1280/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-065617-taisp-t026a-memory && export TAISP_SOURCE_REVISION=cfcb895 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.source_roi_memory --manifest /home/liujianhua/wjq/TAISP/shared/t026a_memory_manifest.json --output "$AUTODL_ARTIFACTS_DIR/memory"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
