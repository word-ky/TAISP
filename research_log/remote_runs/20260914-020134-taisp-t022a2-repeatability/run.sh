#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-020134-taisp-t022a2-repeatability'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-020134-taisp-t022a2-repeatability'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-020134-taisp-t022a2-repeatability/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export TAISP_SOURCE_REVISION=ee20040; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q tests/test_spatial_repeatability.py tests/test_spatial_dose.py tests/test_spatial_dose_study.py 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/focused_tests.txt" && /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/full_tests.txt" && /home/liujianhua/wjq/TAISP/.venv/bin/python -u -m taisp.analysis.spatial_repeatability --deterministic --output "$AUTODL_ARTIFACTS_DIR/deterministic" && /home/liujianhua/wjq/TAISP/.venv/bin/python -u -m taisp.analysis.spatial_repeatability --output "$AUTODL_ARTIFACTS_DIR/audit"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
