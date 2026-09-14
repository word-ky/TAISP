#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-170909-taisp-t031a-train360-map'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-170909-taisp-t031a-train360-map'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-170909-taisp-t031a-train360-map/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-170250-taisp-t031a-reference && export TAISP_SOURCE_REVISION=7569ed4 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.orthogonal_reference --cohort research_log/T031A_train_cohort.json --candidate-root /home/liujianhua/wjq/TAISP/runs/20260914-165905-taisp-t031a-candidates480/artifacts/candidate --candidate-lock research_log/T031A_candidate_lock.json --phase train --output "$AUTODL_ARTIFACTS_DIR/train"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
