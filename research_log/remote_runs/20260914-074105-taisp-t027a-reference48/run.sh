#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-074105-taisp-t027a-reference48'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-074105-taisp-t027a-reference48'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-074105-taisp-t027a-reference48/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-073937-taisp-t027a-reference && export TAISP_SOURCE_REVISION=ed444b6 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.flip_gradient_reference --cohort research_log/T027A_train_cohort.json --candidate-root /home/liujianhua/wjq/TAISP/runs/20260914-073543-taisp-t027a-candidates48/artifacts/candidates --candidate-lock research_log/T027A_candidate_lock.json --output "$AUTODL_ARTIFACTS_DIR/reference"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
