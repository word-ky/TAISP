# T015-A pre-outcome frozen differential-subspace plan

R026/f27a996 accepts and closes T014-A/A1; this task is a saved-array postmortem only.
This plan and input manifest must be committed before new scientific summary outcomes.

## Inputs and reuse

Use only run20260913-113401-taisp-t014a1-common-jacobian, implementation0ad53b6.
Authoritative records.json SHA256:
a0a8c59dbb6b2fd83343f774755c6061a28542bce3c1d2f00b1458e8bb655a17.
All 32 full record hashes are pinned in T015A_input_manifest.json. Verified original
indices0..31,16clean/16corrupted,4blocks of8, reference vectors match individual files,
and existing numerical/isolation flags are true. Never use reverse diagnostic vectors.
Preserve image/case/block/support/mask metadata and all samples without regrouping.

Reuse existing spatial_action.EPS=1e-12, saved geometry.D_spatial, NumPy float64
analysis conventions and existing run/deploy/fetch workflow. Add only differential
decomposition, triage, tests and report; no model/deployment/ISP/loss edits.
Baseline on unchanged A1 release: tests/test_spatial_action.py and
tests/test_common_jacobian.py, 11 passed in2.28s, remote environment.

## Exact algebra

For each task/pseudo pair of8Dvectors, s=(obj+bg)/2,d=(obj-bg)/2;
g=concat(obj,bg),shared=concat(s,s),diff=concat(d,-d).
Save s,d,three norms,differential squared-energy fraction using EPS in denominator,
vector/energy reconstruction errors and shared-dot-diff orthogonality residual.
R_extra=norm(task)/(norm(task_shared)+EPS). No epsilon changes or clipping.
C_shared=dot(task_shared,pseudo_shared)/(norm(pseudo)+EPS),
C_diff=dot(task_diff,pseudo_diff)/(norm(pseudo)+EPS).
Require abs(C_shared+C_diff-savedD_spatial)<=1e-12 for every episode.
Save the component dots, differential dot sign, cos_shared/cos_diff, pseudo differential
energy fraction and amplitude norm fraction, task differential energy fraction.
For cosine with either norm exactly zero, record null plus explicit task/pseudo/either
zero counts; do not count such cases as aligned or omit their C_diff from cohort counts.
Report mean/median/min/max/positive counts, with valid counts for null cosines.
No extra mask correlations or thresholds are needed for this bounded question.

## Frozen triage

Task relevant iff medianR_extra>=1.10 overall AND corrupted AND at least3/4 block
medianR_extra>1.05. Pseudo differentially useful iff C_diff>0 counts>=20/32 overall
AND>=10/16corrupted AND overallmedianC_diff>0 AND at least3/4 positiveblockmedians.
Scopes: overall,clean,corrupted,original4blocks,originalcase labels (fourcorruptions).
TaskYES/pseudoNO => close fixed pseudo-spatial branch; taskNO => close partition;
bothYES => report shared/differential interaction/normalization diagnosis. All cases
stop forresearchreview without automatic method implementation. Do not rescue R024.

## Execution and tests

Single bounded increment: new analysis module and synthetic tests. Verify reconstruction,
orthogonality, Pythagorean identity, D_spatial parity including shared-only, opposing,
one-region-only and zero vectors, and exact triage boundaries. Focused tests then full
regression before one saved-array audit. Local syntax/stdlib preparation; remote known
Python3.12.12/NumPy1.26.4 runtime for analysis. User prefers A6000 GPU: enable CUDA for
applicable regression checks; tiny16D float64 array reductions stay on CPU. No new
FasterR-CNN/CLIP/ISP audit call, optimizer, image read or target/AP execution.
Persist commands, environment, per-episode decomposition, subgroup statistics, triage,
full raw run and SHAmanifest. Commit/push CODEX mailbox report and mirrorA6000.
Stop NEEDS_REVIEW after T015-A; no T014-B/T015-B, differential-only deployment,
finite-step spatial update, mask/region search, regionalCLIP, source/meta-training,
predictor redesign, newcohort or gate/dose tuning.
