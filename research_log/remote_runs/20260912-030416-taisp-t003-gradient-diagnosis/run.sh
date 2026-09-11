#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260912-030416-taisp-t003-gradient-diagnosis'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-030416-taisp-t003-gradient-diagnosis'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-030416-taisp-t003-gradient-diagnosis/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export PYTHONPATH="$PWD"; /home/liujianhua/wjq/TAISP/.venv/bin/python /home/liujianhua/wjq/TAISP/shared/diagnose_t003_baseline_gradient.py
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
