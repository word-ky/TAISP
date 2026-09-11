#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260912-031123-taisp-t003-coco200'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-031123-taisp-t003-coco200'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-031123-taisp-t003-coco200/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export OMP_NUM_THREADS=1; export TAISP_SOURCE_REVISION=8d8913e; export PATH=/home/liujianhua/wjq/TAISP/.venv/bin:$PATH; TAISP_REAL_MODELS=1 python -m pytest -q && python -m taisp.analysis.run_t003 --data-root /home/liujianhua/wjq/TAISP/shared/coco200 --baseline /home/liujianhua/wjq/TAISP/runs/20260912-013248-taisp-t002-coco200-final/artifacts/study --output "$AUTODL_ARTIFACTS_DIR/study"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
