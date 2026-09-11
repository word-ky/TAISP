#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260912-050701-taisp-t004-study-smoke'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-050701-taisp-t004-study-smoke'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-050701-taisp-t004-study-smoke/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export OMP_NUM_THREADS=1; export TAISP_SOURCE_REVISION=681ea1b; export PATH=/home/liujianhua/wjq/TAISP/.venv/bin:$PATH; TAISP_REAL_MODELS=1 python -m pytest -q && python -m taisp.analysis.run_t004 --data-root /home/liujianhua/wjq/TAISP/shared/coco200 --baseline /home/liujianhua/wjq/TAISP/runs/20260912-013248-taisp-t002-coco200-final/artifacts/study --output "$AUTODL_ARTIFACTS_DIR/study" --limit 2
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
