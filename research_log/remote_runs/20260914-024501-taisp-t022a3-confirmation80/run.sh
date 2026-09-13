#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-024501-taisp-t022a3-confirmation80'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-024501-taisp-t022a3-confirmation80'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-024501-taisp-t022a3-confirmation80/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-024158-taisp-t022a3-confirmation && export TAISP_SOURCE_REVISION=9b27c49 && /home/liujianhua/wjq/TAISP/.venv/bin/python -u -m taisp.analysis.spatial_confirmation --supports /home/liujianhua/wjq/TAISP/runs/20260914-024325-taisp-t022a3-supports/artifacts/prepare/supports.json --output "$AUTODL_ARTIFACTS_DIR/confirmation"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
