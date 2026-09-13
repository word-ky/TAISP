#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260913-091200-taisp-t013i-linear-capacity'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-091200-taisp-t013i-linear-capacity'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-091200-taisp-t013i-linear-capacity/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUDA_VISIBLE_DEVICES="" TAISP_SOURCE_REVISION=c6d2c6e TAISP_REAL_MODELS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.linear_capacity --prior-root /home/liujianhua/wjq/TAISP/runs/20260913-081117-taisp-t013h-source-replication --artifact-manifest research_log/T013H/artifact_manifest.json --output "$AUTODL_ARTIFACTS_DIR/audit"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
