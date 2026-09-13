#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-073543-taisp-t027a-candidates48'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-073543-taisp-t027a-candidates48'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-073543-taisp-t027a-candidates48/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-073411-taisp-t027a-candidate && export TAISP_SOURCE_REVISION=c713818 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.flip_gradient_consensus candidate --manifest research_log/T027A/candidate_images.json --preflight-receipt /home/liujianhua/wjq/TAISP/shared/t027a_commutation.json --output "$AUTODL_ARTIFACTS_DIR/candidates"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
