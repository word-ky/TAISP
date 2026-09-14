#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-130040-taisp-t030a-candidates120'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-130040-taisp-t030a-candidates120'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-130040-taisp-t030a-candidates120/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-125919-taisp-t030a-candidate && export TAISP_SOURCE_REVISION=9fd4a7e && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.pseudo_native_candidates --manifest research_log/T030A/candidate_images.json --output "$AUTODL_ARTIFACTS_DIR/candidates"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
