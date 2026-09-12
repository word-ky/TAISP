#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260912-181436-taisp-t010-coco1000-offline'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-181436-taisp-t010-coco1000-offline'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-181436-taisp-t010-coco1000-offline/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export TAISP_SOURCE_REVISION=9efe032 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests/test_t010_gating.py tests/test_t009_replication.py tests/test_t008_analysis.py -q && /home/liujianhua/wjq/TAISP/.venv/bin/python -m scripts.analyze_t010 --study /home/liujianhua/wjq/TAISP/runs/20260912-144439-taisp-t009-coco1000/artifacts/study --prepared /home/liujianhua/wjq/TAISP/research_log/T010/preparation --annotations /home/liujianhua/wjq/TAISP/shared/coco1000_t009/instances_val2017.json --output "$AUTODL_ARTIFACTS_DIR/study" --workers 12 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m scripts.report_t010 --study "$AUTODL_ARTIFACTS_DIR/study" --prepared /home/liujianhua/wjq/TAISP/research_log/T010/preparation
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
