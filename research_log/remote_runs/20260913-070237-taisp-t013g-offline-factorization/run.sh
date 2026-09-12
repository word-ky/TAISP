#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260913-070237-taisp-t013g-offline-factorization'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-070237-taisp-t013g-offline-factorization'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-070237-taisp-t013g-offline-factorization/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES="" TAISP_SOURCE_REVISION=a7a5494 TAISP_REAL_MODELS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests/test_predictor_factorization.py tests/test_repeatability.py tests/test_gradient_conflict.py -q && /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.predictor_factorization --prior-root /home/liujianhua/wjq/TAISP/runs/20260913-030301-taisp-t013c-conflict-conditioning/artifacts/audit --output "$AUTODL_ARTIFACTS_DIR/audit"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
