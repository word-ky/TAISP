#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-001023-taisp-t022a1-numerical-parity'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-001023-taisp-t022a1-numerical-parity'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-001023-taisp-t022a1-numerical-parity/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export TAISP_SOURCE_REVISION=8948d4c; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q tests/test_spatial_dose.py tests/test_common_jacobian.py tests/test_spatial_action.py tests/test_trust_radius.py 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/focused_tests.txt" && /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/full_tests.txt" && /home/liujianhua/wjq/TAISP/.venv/bin/python -u -m taisp.analysis.spatial_dose_parity --output "$AUTODL_ARTIFACTS_DIR/parity"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
