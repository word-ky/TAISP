# T023-A — FAIL; close object-only action; NEEDS_REVIEW

R038/9f2d045; pre-outcome plan23c0169; analysis7d523a2. Completed
2026-09-14T03:34:58.965626+08:00.
**The fixed object-only first-order utility conjunction fails.** Per R038,
close this object-only action hypothesis without implementing it. T022 iterative
two-state spatial dose was already closed by R038; its withheld200-image AP
remains unrun. No new runtime, objective design or T023-B was started.

## Authoritative inputs and integrity

Only corrected common-Jacobian A1 records from
`20260913-113401-taisp-t014a1-common-jacobian/artifacts/audit` were analyzed.
The records.json hash is
`a0a8c59dbb6b2fd83343f774755c6061a28542bce3c1d2f00b1458e8bb655a17`.
All33hashes (aggregate plus32full records) match the inherited T015-A manifest,
pinned before new summaries. Aggregate task/pseudo.reference vectors exactly
match each individual authoritative record. Existing numerical and isolation
flags pass; original episode order,16clean/16corrupted,4blocks of8, supports and
mask hashes were preserved. No old failed reverse-partition vectors were used.

Reconstructed p_s=p_o+p_b and t_s=t_o+t_b were checked against saved reference
`global` for all64objectives at the original bound
`1e-12 + 1e-10*(abs(object)+abs(background))` percoordinate. All pass;
maximum closure absolute error **1.33226762955e-15**.
All23protected method-module LF hashes remain unchanged.

## Computation and focused validation

One standalone analysis script and four standard-library synthetic tests were
added. Tests cover unit action/descent signs, cancellation, zero/undefined
cosines, originalclosure tolerance and the conjunction of count/median/block/
integrity requirements. **4tests passed in0.001s**. Exact commands are saved in
T023A/commands_and_tests.txt. No model-bearing regression or pretrained-model
experiment was needed for this saved-vector calculation.

Python3.12.7 on local CPU, IEEE754 binary64 Pythonfloat, math.fsum dot
products; eps=1e-12. Arithmetic capability is checked (53-bit mantissa). Audit
elapsed0.3064298s. No torch/numpy/model import, detector/CLIP/ISP
call, image/annotation loading, optimizer, AP or new data was used. This is the
lightweight offline-summary exception to the user's GPU preference.

For each episode:
S_obj=dot(t_o,p_o)/(norm(p_o)+eps);
S_global=dot(t_s,p_s)/(norm(p_s)+eps);
DeltaS=S_obj-S_global. All reconstructed8-D vectors, cosine/norm/mask/support
information and closure errors are retained. PositiveS means first-order task
descent under the negative normalized pseudo direction. The same positive
CLIP-radius factor is omitted as specified; no gradient/radius is re-estimated.

## Frozen gate results

| Criterion | Observed | Result |
| --- | --- | --- |
| Record/hash/reconstruction checks | All33hashes /32records /64closures pass | PASS |
| S_obj>0 overall >=20/32 | 17/32 | FAIL |
| S_obj>0 corrupted >=10/16 | 10/16 | PASS |
| DeltaS>0 overall >=20/32 | 16/32 | FAIL |
| DeltaS>0 corrupted >=10/16 | 7/16 | FAIL |
| Overall median DeltaS>0 | +0.008094718581960925 | PASS |
| Corrupted median DeltaS>0 | -0.007283721473185255 | FAIL |
| >=3/4 positive block medianDeltaS | 2/4 | FAIL |

Block medians, in frozen order:
- block0: -0.009311543143670779;
- block1: -0.0014895791902435783;
- block2: +0.027607223102328684;
- block3: +0.019864270447509092.

Object/global meanS overall are -0.019255019209484678 /+0.018735975185190056;
overall meanDeltaS=-0.03799099439467474. Corrupted meanDeltaS=-0.05406672504050771.
The overall positive median and corrupted10/16positive object scores do not
rescue the failed conjunction. This does not show that an implemented runtime
would harm AP; none was implemented/evaluated.

## Full diagnostics and decision

T023A/tables.md contains all32episode rows, all six vector norms, clean/corrupt,
allfour blocks, fixedcorruption cases and all predeclared mask/support strata.
Mask bins:zero,(0,.01],(.01,.05],(.05,.2],(.2,1]; support bins:zero,1–5,6–10,
11–20,>20. Boundaries were pinned in23c0169 before results; no favorable bin
substitutes for the gate. All six norms are nonzero on all32episodes, all64
cosines are defined, and no utility/delta equals zero. Empty bins have n=0/null
statistics; no episodes were excluded from the analysis.

Stop **NEEDS_REVIEW** and **close_object_only_action**, exactly as R038 directs.
No object-only runtime, T023-B, new regional objective, AP, newcohort, source/meta
training, dose/rho/mask sweep, deterministic performance setting or method edit
was performed. A new regional self-supervised objective requires a new research
decision. No active experiment remains. Original T014/T015/T022 receipts unchanged.
