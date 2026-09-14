#!/usr/bin/env bash
set -uo pipefail
cd '/home/liujianhua/wjq/TAISP/current'
export AUTODL_RUN_ID='20260914-192812-taisp-t033a-train480-model'
export AUTODL_RUN_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-192812-taisp-t033a-train480-model'
export AUTODL_ARTIFACTS_DIR='/home/liujianhua/wjq/TAISP/runs/20260914-192812-taisp-t033a-train480-model/artifacts'
mkdir -p "$AUTODL_ARTIFACTS_DIR"
echo "[autodl] run_id=$AUTODL_RUN_ID"
echo "[autodl] started_at=$(date -Is)"
{
cd /home/liujianhua/wjq/TAISP/releases/20260914-192036-taisp-t033a-reference && export TAISP_SOURCE_REVISION=fabdfda && /home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.object_state_reference --cohort research_log/T033A_train_cohort.json --candidate-root /home/liujianhua/wjq/TAISP/runs/20260914-191144-taisp-t033a-candidates600/artifacts/candidate --candidate-lock research_log/T033A_candidate_lock.json --representation-root /home/liujianhua/wjq/TAISP/runs/20260914-192538-taisp-t033a-representation/artifacts/representation --representation-lock research_log/T033A_representation_lock.json --phase train --output "$AUTODL_ARTIFACTS_DIR/train"
}
status=$?
echo "[autodl] finished_at=$(date -Is)"
echo "[autodl] exit_code=$status"
exit $status
