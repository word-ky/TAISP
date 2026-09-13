#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260913-182437-taisp-t019a-runtime-smoke'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-182437-taisp-t019a-runtime-smoke'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-182437-taisp-t019a-runtime-smoke/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export TAISP_SOURCE_REVISION=ee98de0; export TAISP_REAL_MODELS=0; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q && /home/liujianhua/wjq/TAISP/.venv/bin/python -u -m taisp.analysis.run_t018a --config configs/t019a.yaml --manifest research_log/T019A_train_cohort.json --smoke --output "$AUTODL_ARTIFACTS_DIR/study"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
