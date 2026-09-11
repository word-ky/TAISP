#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260912-012847-taisp-t002-parity-before'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-012847-taisp-t002-parity-before'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-012847-taisp-t002-parity-before/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export USE_TF=0; export PYTHONPATH=$PWD; /home/liujianhua/wjq/TAISP/.venv/bin/python /home/liujianhua/wjq/TAISP/shared/check_clip_preprocess_parity.py --image /home/liujianhua/wjq/TAISP/shared/coco200/val2017/000000105335.jpg --output "$AUTODL_ARTIFACTS_DIR/parity.json"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
