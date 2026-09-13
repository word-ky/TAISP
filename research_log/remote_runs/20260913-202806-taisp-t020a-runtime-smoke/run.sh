#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260913-202806-taisp-t020a-runtime-smoke'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-202806-taisp-t020a-runtime-smoke'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260913-202806-taisp-t020a-runtime-smoke/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export TAISP_SOURCE_REVISION=0546b05; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q tests/test_gradient_transport.py tests/test_trust_radius.py tests/test_detector_native.py tests/test_native_pseudo_target.py 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/focused_tests.txt" && /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest -q 2>&1 | tee "$AUTODL_ARTIFACTS_DIR/full_tests.txt" && /home/liujianhua/wjq/TAISP/.venv/bin/python -u -m taisp.analysis.collect_t020a --smoke --output "$AUTODL_ARTIFACTS_DIR/source_fit" && /home/liujianhua/wjq/TAISP/.venv/bin/python -u -m taisp.analysis.run_t018a --config configs/t020a.yaml --manifest research_log/T020A_train_cohort.json --transport-fits "$AUTODL_ARTIFACTS_DIR/source_fit/fits.json" --smoke --output "$AUTODL_ARTIFACTS_DIR/study"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
