#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-160040-taisp-t030a3-reference120'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-160040-taisp-t030a3-reference120'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-160040-taisp-t030a3-reference120/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-155837-taisp-t030a3-reference && export TAISP_SOURCE_REVISION=d480de9 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.pseudo_native_reference --cohort research_log/T030A_train_cohort.json --candidate-root /home/liujianhua/wjq/TAISP/runs/20260914-151622-taisp-t030a2-candidates120/artifacts/candidate --candidate-lock research_log/T030A3_candidate_lock.json --output "$AUTODL_ARTIFACTS_DIR/reference"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
