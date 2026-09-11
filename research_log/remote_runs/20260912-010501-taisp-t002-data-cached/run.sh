#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260912-010501-taisp-t002-data-cached'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-010501-taisp-t002-data-cached'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-010501-taisp-t002-data-cached/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
/home/liujianhua/wjq/TAISP/.venv/bin/python /home/liujianhua/wjq/TAISP/shared/prepare_coco_subset.py --root /home/liujianhua/wjq/TAISP/shared/coco200
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
