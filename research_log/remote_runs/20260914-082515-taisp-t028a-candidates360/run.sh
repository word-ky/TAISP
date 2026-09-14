#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-082515-taisp-t028a-candidates360'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-082515-taisp-t028a-candidates360'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-082515-taisp-t028a-candidates360/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-082414-taisp-t028a-candidate && export TAISP_SOURCE_REVISION=$(git_not_used=1; echo placeholder) && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.gradient_state_candidates --manifest research_log/T028A/candidate_images.json --output "$AUTODL_ARTIFACTS_DIR/candidates"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
