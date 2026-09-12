#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260912-212148-taisp-t012-study-smoke-fixed'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-212148-taisp-t012-study-smoke-fixed'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-212148-taisp-t012-study-smoke-fixed/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export TAISP_SOURCE_REVISION=dddc238 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests/test_half_dose.py tests/test_t012_report.py tests/test_t009_replication.py -q && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.run_t009 --config configs/t012.yaml --data-root /home/liujianhua/wjq/TAISP/shared/coco1000_t009 --reference-study /home/liujianhua/wjq/TAISP/runs/20260912-143436-taisp-t009-study-smoke/artifacts/study --output "$AUTODL_ARTIFACTS_DIR/study" --limit 2
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
