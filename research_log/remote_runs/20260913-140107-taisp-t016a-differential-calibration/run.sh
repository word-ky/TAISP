#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260913-140107-taisp-t016a-differential-calibration'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-140107-taisp-t016a-differential-calibration'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-140107-taisp-t016a-differential-calibration/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
unset CUBLAS_WORKSPACE_CONFIG; export TAISP_SOURCE_REVISION=1b19e8c TAISP_REAL_MODELS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.differential_calibration --prior-root /home/liujianhua/wjq/TAISP/runs --manifest research_log/T016A_input_manifest.json --schedule research_log/T016A_permutation_schedule.json --output "$AUTODL_ARTIFACTS_DIR/audit"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
