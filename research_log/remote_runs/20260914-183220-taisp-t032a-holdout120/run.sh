#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-183220-taisp-t032a-holdout120'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-183220-taisp-t032a-holdout120'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-183220-taisp-t032a-holdout120/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-181842-taisp-t032a-reference && export TAISP_SOURCE_REVISION=57bf9b6 && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.gradient_regime_reference --cohort research_log/T032A_train_cohort.json --candidate-root /home/liujianhua/wjq/TAISP/runs/20260914-181428-taisp-t032a-candidates480/artifacts/candidate --candidate-lock research_log/T032A_candidate_lock.json --router research_log/T032A/router.json --router-lock research_log/T032A_router_lock.json --phase holdout --map-root /home/liujianhua/wjq/TAISP/runs/20260914-182353-taisp-t032a-train360-maps/artifacts/train --map-lock research_log/T032A_map_lock.json --output "$AUTODL_ARTIFACTS_DIR/holdout"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
