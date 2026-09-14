#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-181428-taisp-t032a-candidates480'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-181428-taisp-t032a-candidates480'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-181428-taisp-t032a-candidates480/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-181346-taisp-t032a-candidates && export TAISP_SOURCE_REVISION=b0ce54b && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.orthogonal_candidates --manifest research_log/T032A/candidate_images.json --output "$AUTODL_ARTIFACTS_DIR/candidate"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
