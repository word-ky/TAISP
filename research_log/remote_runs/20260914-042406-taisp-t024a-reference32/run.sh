#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-042406-taisp-t024a-reference32'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-042406-taisp-t024a-reference32'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-042406-taisp-t024a-reference32/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-042234-taisp-t024a-reference && export TAISP_SOURCE_REVISION=638b763 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.roi_equivariance_reference --cohort research_log/T024A_train_cohort.json --candidate-root /home/liujianhua/wjq/TAISP/runs/20260914-041922-taisp-t024a-candidates32/artifacts/candidates --candidate-lock research_log/T024A_candidate_lock.json --output "$AUTODL_ARTIFACTS_DIR/reference"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
