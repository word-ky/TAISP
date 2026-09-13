#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260913-081117-taisp-t013h-source-replication'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-081117-taisp-t013h-source-replication'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-081117-taisp-t013h-source-replication/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
unset CUBLAS_WORKSPACE_CONFIG; export TAISP_SOURCE_REVISION=553e02c TAISP_REAL_MODELS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.collect_source_replication --manifest research_log/T013H_train_cohort.json --checkpoint /home/liujianhua/wjq/TAISP/runs/20260913-030301-taisp-t013c-conflict-conditioning/artifacts/audit/original_predictor.pt --output "$AUTODL_ARTIFACTS_DIR/collection" && CUDA_VISIBLE_DEVICES="" /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.source_replication --records "$AUTODL_ARTIFACTS_DIR/collection/records.json" --output "$AUTODL_ARTIFACTS_DIR/algebra"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
