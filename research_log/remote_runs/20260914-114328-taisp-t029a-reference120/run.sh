#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-114328-taisp-t029a-reference120'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-114328-taisp-t029a-reference120'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-114328-taisp-t029a-reference120/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-114121-taisp-t029a-reference && export TAISP_SOURCE_REVISION=8ef7b3e && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.soft_pseudo_reference --cohort research_log/T029A_train_cohort.json --candidate-root /home/liujianhua/wjq/TAISP/runs/20260914-113829-taisp-t029a-candidates120/artifacts/candidates --candidate-lock research_log/T029A_candidate_lock.json --output "$AUTODL_ARTIFACTS_DIR/reference"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
