#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260912-003812-taisp-t001-a6000'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-003812-taisp-t001-a6000'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-003812-taisp-t001-a6000/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export PATH=/home/liujianhua/wjq/TAISP/.venv/bin:$PATH; export TAISP_SOURCE_REVISION=54a9e64; bash scripts/run_a6000_smoke.sh
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
