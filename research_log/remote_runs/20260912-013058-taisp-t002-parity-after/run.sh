#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260912-013058-taisp-t002-parity-after'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-013058-taisp-t002-parity-after'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260912-013058-taisp-t002-parity-after/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
export USE_TF=0; export OMP_NUM_THREADS=1; export TAISP_REAL_MODELS=1; export PATH=/home/liujianhua/wjq/TAISP/.venv/bin:$PATH; python -m pytest -q && PYTHONPATH=$PWD python scripts/check_clip_preprocess_parity.py --image /home/liujianhua/wjq/TAISP/shared/coco200/val2017/000000105335.jpg --output "$AUTODL_ARTIFACTS_DIR/parity.json"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
