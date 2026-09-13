#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-024325-taisp-t022a3-supports'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-024325-taisp-t022a3-supports'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-024325-taisp-t022a3-supports/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export TAISP_SOURCE_REVISION=9b27c49; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q tests/test_spatial_confirmation.py tests/test_spatial_repeatability.py tests/test_spatial_dose_study.py tests/test_spatial_dose.py 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/focused_tests.txt" && /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/full_tests.txt" && /home/liujianhua/wjq/TAISP/.venv/bin/python -u -m taisp.analysis.spatial_confirmation --prepare --output "$AUTODL_ARTIFACTS_DIR/prepare"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
