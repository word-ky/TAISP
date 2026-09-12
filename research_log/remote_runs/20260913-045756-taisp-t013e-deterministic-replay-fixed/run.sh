#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260913-045756-taisp-t013e-deterministic-replay-fixed'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-045756-taisp-t013e-deterministic-replay-fixed'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-045756-taisp-t013e-deterministic-replay-fixed/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TAISP_SOURCE_REVISION=a585749 TAISP_REAL_MODELS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.deterministic_replay --manifest research_log/T013B_train_microset.json --supports /home/liujianhua/wjq/TAISP/runs/20260913-014612-taisp-t013b-source-three-steps/artifacts/smoke/supports.json --prior /home/liujianhua/wjq/TAISP/runs/20260913-030301-taisp-t013c-conflict-conditioning/artifacts/audit/receipt.json --original /home/liujianhua/wjq/TAISP/runs/20260913-030301-taisp-t013c-conflict-conditioning/artifacts/audit/original_predictor.pt --control-root /home/liujianhua/wjq/TAISP/runs/20260913-033722-taisp-t013d-repeatability-common-mode --control-manifest research_log/T013D/artifact_manifest.json --output "$AUTODL_ARTIFACTS_DIR/audit"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
