#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-191144-taisp-t033a-candidates600'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-191144-taisp-t033a-candidates600'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-191144-taisp-t033a-candidates600/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-191003-taisp-t033a-candidates && export TAISP_SOURCE_REVISION=b5d4c6f && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.object_state_candidates --manifest research_log/T033A/candidate_images.json --output "$AUTODL_ARTIFACTS_DIR/candidate"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
