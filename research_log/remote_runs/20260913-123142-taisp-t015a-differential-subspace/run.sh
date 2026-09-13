#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260913-123142-taisp-t015a-differential-subspace'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-123142-taisp-t015a-differential-subspace'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-123142-taisp-t015a-differential-subspace/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
unset CUBLAS_WORKSPACE_CONFIG; export TAISP_SOURCE_REVISION=4860be7 TAISP_REAL_MODELS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.differential_subspace --input-root /home/liujianhua/wjq/TAISP/runs/20260913-113401-taisp-t014a1-common-jacobian/artifacts/audit --manifest research_log/T015A_input_manifest.json --output "$AUTODL_ARTIFACTS_DIR/audit"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
