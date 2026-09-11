#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260912-063551-taisp-t005-loss-tests'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-063551-taisp-t005-loss-tests'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-063551-taisp-t005-loss-tests/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
TAISP_REAL_MODELS=1 /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q tests/test_detector_native.py tests/test_detector_signal.py tests/test_adapt.py
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
