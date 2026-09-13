#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260913-203203-taisp-t020a-source200-crossfit'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-203203-taisp-t020a-source200-crossfit'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-203203-taisp-t020a-source200-crossfit/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export TAISP_SOURCE_REVISION=0546b05; /home/liujianhua/wjq/TAISP/.venv/bin/python -u -m taisp.analysis.collect_t020a --output "$AUTODL_ARTIFACTS_DIR/source_fit" && /home/liujianhua/wjq/TAISP/.venv/bin/python -u -m taisp.analysis.run_t018a --config configs/t020a.yaml --manifest research_log/T020A_train_cohort.json --transport-fits "$AUTODL_ARTIFACTS_DIR/source_fit/fits.json" --output "$AUTODL_ARTIFACTS_DIR/study"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
