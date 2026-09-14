#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-151141-taisp-t030a2-attribution8x5'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-151141-taisp-t030a2-attribution8x5'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-151141-taisp-t030a2-attribution8x5/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-150932-taisp-t030a2-attribution && export TAISP_SOURCE_REVISION=e5ed293 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.native_chain_attribution --manifest research_log/T030A2/diagnostic_images.json --output "$AUTODL_ARTIFACTS_DIR/diagnostic"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
