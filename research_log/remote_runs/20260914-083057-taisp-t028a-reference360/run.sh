#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-083057-taisp-t028a-reference360'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-083057-taisp-t028a-reference360'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-083057-taisp-t028a-reference360/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-082925-taisp-t028a-reference && export TAISP_SOURCE_REVISION=b502d15 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.gradient_state_reference --cohort research_log/T028A_train_cohort.json --candidate-root /home/liujianhua/wjq/TAISP/runs/20260914-082515-taisp-t028a-candidates360/artifacts/candidates --candidate-lock research_log/T028A_candidate_lock.json --output "$AUTODL_ARTIFACTS_DIR/reference" && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.gradient_state_fit --candidate-root /home/liujianhua/wjq/TAISP/runs/20260914-082515-taisp-t028a-candidates360/artifacts/candidates --reference-root "$AUTODL_ARTIFACTS_DIR/reference" --output "$AUTODL_ARTIFACTS_DIR/fit"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
