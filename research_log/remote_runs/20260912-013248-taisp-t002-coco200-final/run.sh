#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260912-013248-taisp-t002-coco200-final'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-013248-taisp-t002-coco200-final'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-013248-taisp-t002-coco200-final/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export OMP_NUM_THREADS=1; export TAISP_SOURCE_REVISION=0b8a888; export PATH=/home/liujianhua/wjq/TAISP/.venv/bin:$PATH; python -m taisp.analysis.run_t002 --data-root /home/liujianhua/wjq/TAISP/shared/coco200 --output "$AUTODL_ARTIFACTS_DIR/study"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
