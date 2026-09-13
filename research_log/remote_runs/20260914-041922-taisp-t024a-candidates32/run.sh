#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-041922-taisp-t024a-candidates32'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-041922-taisp-t024a-candidates32'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-041922-taisp-t024a-candidates32/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-041825-taisp-t024a-candidate && export TAISP_SOURCE_REVISION=bf78bc5 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.roi_equivariance --manifest research_log/T024A/candidate_images.json --output "$AUTODL_ARTIFACTS_DIR/candidates"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
