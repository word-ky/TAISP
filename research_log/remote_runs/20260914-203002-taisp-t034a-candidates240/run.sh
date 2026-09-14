#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-203002-taisp-t034a-candidates240'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-203002-taisp-t034a-candidates240'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-203002-taisp-t034a-candidates240/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-202639-taisp-t034a-candidates && export TAISP_SOURCE_REVISION=0f1ddd264d392e0536953d29a632a6382a3c5cf2 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.gradient_basis_candidates --manifest /home/liujianhua/wjq/TAISP/research_log/T034A_candidate_images.json --output "$AUTODL_ARTIFACTS_DIR/candidates"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
