#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-113829-taisp-t029a-candidates120'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-113829-taisp-t029a-candidates120'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-113829-taisp-t029a-candidates120/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-113707-taisp-t029a-candidate-tested && export TAISP_SOURCE_REVISION=a2bc921 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.soft_pseudo_candidates --manifest research_log/T029A/candidate_images.json --output "$AUTODL_ARTIFACTS_DIR/candidates"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
