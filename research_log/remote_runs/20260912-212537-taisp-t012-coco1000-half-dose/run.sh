#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260912-212537-taisp-t012-coco1000-half-dose'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-212537-taisp-t012-coco1000-half-dose'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-212537-taisp-t012-coco1000-half-dose/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export TAISP_SOURCE_REVISION=3c82267 TAISP_REAL_MODELS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; /home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.run_t009 --config configs/t012.yaml --data-root /home/liujianhua/wjq/TAISP/shared/coco1000_t009 --reference-study /home/liujianhua/wjq/TAISP/runs/20260912-144439-taisp-t009-coco1000/artifacts/study --output "$AUTODL_ARTIFACTS_DIR/study" && /home/liujianhua/wjq/TAISP/.venv/bin/python -m scripts.report_t012 --study "$AUTODL_ARTIFACTS_DIR/study" --reference /home/liujianhua/wjq/TAISP/runs/20260912-144439-taisp-t009-coco1000/artifacts/study --controls /home/liujianhua/wjq/TAISP/runs/20260912-195353-taisp-t011-coco1000-offline/artifacts/study
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
