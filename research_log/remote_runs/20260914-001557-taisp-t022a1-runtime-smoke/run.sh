#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-001557-taisp-t022a1-runtime-smoke'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-001557-taisp-t022a1-runtime-smoke'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-001557-taisp-t022a1-runtime-smoke/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export TAISP_SOURCE_REVISION=7001d0c; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q tests/test_spatial_dose_study.py tests/test_spatial_dose.py tests/test_flip_consensus.py 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/focused_tests.txt" && /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/full_tests.txt" && /home/liujianhua/wjq/TAISP/.venv/bin/python -u -m taisp.analysis.run_t018a --config configs/t022a.yaml --manifest research_log/T022A_train_cohort.json --smoke --output "$AUTODL_ARTIFACTS_DIR/study"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
