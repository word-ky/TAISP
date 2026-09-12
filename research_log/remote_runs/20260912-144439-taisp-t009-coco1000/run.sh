#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260912-144439-taisp-t009-coco1000'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-144439-taisp-t009-coco1000'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-144439-taisp-t009-coco1000/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export TAISP_SOURCE_REVISION=69bfb66; /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.run_t009 --data-root /home/liujianhua/wjq/TAISP/shared/coco1000_t009 --output "$AUTODL_ARTIFACTS_DIR/study" && /home/liujianhua/wjq/TAISP/.venv/bin/python -m scripts.report_t009 --study "$AUTODL_ARTIFACTS_DIR/study"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
