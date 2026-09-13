#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-052527-taisp-t025a-reference48'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-052527-taisp-t025a-reference48'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-052527-taisp-t025a-reference48/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-052403-taisp-t025a-reference && export TAISP_SOURCE_REVISION=2cd7415 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.roi_geometry_reference --cohort research_log/T025A_train_cohort.json --candidate-root /home/liujianhua/wjq/TAISP/runs/20260914-052059-taisp-t025a-candidates48/artifacts/candidates --candidate-lock research_log/T025A_candidate_lock.json --output "$AUTODL_ARTIFACTS_DIR/reference"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
