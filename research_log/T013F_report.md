# T013-F final report — NEEDS_REVIEW

2026-09-13, R020/7efdb89. The fixed matched-checkpoint replay is complete.
**The predeclared scientific decision is measurement-limited / unresolved or mixed.**
Only the joint checkpoint's pooled eight-episode group passes the resolution rule;
neither clean nor corrupted subgroup passes for any checkpoint. Functional cross-harm
and resolved improvement on both subgroups are therefore not established.

## Scope, provenance and execution

Pre-outcome plan [T013F_plan.md](T013F_plan.md), commit `0ef7ef7`, preceded runner
`0bfdd19` and all repeated outcomes. Exact saved T013-C checkpoints were available:
the original and joint/clean/corrupt files were hash-verified and copied byte-for-byte.
No new original-state gradient computation, direction selection, or optimizer step
occurred in T013-F. The saved full gradient audit and three directions were retained.

| Checkpoint | Parameter delta norm from original | Changed tensors |
| --- | --- | --- |
| original | 0 | none |
| joint | 0.0001682057773 | head.weight, head.bias |
| clean | 0.00009830247291 | head.weight, head.bias |
| corrupted | 0.0004009396944 | head.weight, head.bias |

All three heads exactly match their saved direction multiplied by `-1e-3`;
feature-trunk tensors remain identical to original. Full checkpoint file/state hashes
and gradient vectors are in
[checkpoint_freeze.json](remote_runs/20260913-055654-taisp-t013f-matched-checkpoints/artifacts/audit/checkpoint_freeze.json).
The original constructor state also exactly matched its saved checkpoint.

Same T013-B four train2017 images/eight episodes, annotations, corruption order and
detached supports, all pinned in the plan. Same source/CLIP weight files and prompts,
RGB/float32 preprocessing, ISP, K=3, inner lr=0.1, epsilon=1e-12, and source outer loss.
Seed20260913 reset before each eight-episode evaluation; oracle sampling remains fixed.
Normal T013-C/D CUDA path: deterministic algorithms false, cudnn benchmark false,
`CUBLAS_WORKSPACE_CONFIG` unset. No new kernel, preprocessing, precision or tolerance change.

Run `20260913-055654-taisp-t013f-matched-checkpoints`, release
`20260913-055629-taisp-t013f-matched-replay`, source `0bfdd19`, at
`/home/liujianhua/wjq/TAISP` on RTX A6000. Started 05:56:58+08, finished 06:01:34+08,
exit 0. Analysis elapsed **267.411184s**; peak allocated CUDA **4,842,729,984 bytes**.

Tests: baseline **6 passed / 8.31s**; focused **10 passed / 7.06s**;
remote regression **105 passed / 10 skipped / 5.78s**, then the separate real CUDA
study. Skipped tests are not counted as real-model passes. No failure or retry occurred.

## Matched schedule and retained measurements

Exactly **8 cycles × 4 pairs × 2 evaluations × 8 episodes = 512 raw rows**.
All 64 eight-episode evaluations were saved separately before the final effect summary.
Pairs are null (original A / independent original B), original / joint checkpoint,
original / clean checkpoint, original / corrupt checkpoint. Rotate the four blocks
left by `(cycle-1) modulo 4`; baseline first in odd cycles, probe/control-copy first
in even cycles. Each block occupies each position twice across the eight cycles.
The planned and executed order is recorded in [the complete tables](T013F/tables.md).

Every evaluation used a fresh checkpoint copy and existing episodic ISP initialization.
Every raw row retains cycle/pair/block/role/order, image/case, outer loss/components,
phi0/phi3 vectors and norms, saturation and empty-support status. All 64 receipts
confirmed predictor unchanged with `.grad=None`, source/CLIP parameter-buffer hashes
unchanged and `.grad=None`, and ISP unchanged. All original checkpoint files and
template states were rechecked at completion. There were **zero optimizer steps**.
Empty supports: **0/512**. Saturation ranged from 0 to **0.0568152368**.

## Exact predeclared rule and noise controls

Use logical A/B roles regardless of execution order:
`delta_probe = mean(loss_B-loss_A)`;
`delta_null = mean(loss_original_B-loss_original_A)` in the same cycle;
`delta_cc = delta_probe-delta_null`.
Negative is improvement. Median averages the two middle values of all eight cycles.

An effect is resolved only if **at least 7/8 corrected signs match the median** and
**abs(median corrected effect) > max absolute null effect for that group**.
The strict `>` threshold and all cycles were retained without post-outcome adjustment.

| Group | Null signed median | Null minimum | Null maximum | Null absolute min–max | Max abs null used as threshold |
| --- | --- | --- | --- | --- | --- |
| pooled joint (8 episodes) | -0.0000603842782 | -0.00324923871 | 0.00479621300 | 0.000850379700–0.00479621300 | 0.00479621300 |
| clean (4 episodes) | -0.00227782968 | -0.00786287524 | 0.00444629975 | 0.000432928558–0.00786287524 | 0.00786287524 |
| corrupted (4 episodes) | 0.00131611433 | -0.00448579807 | 0.00941403583 | 0.000211596489–0.00941403583 | 0.00941403583 |

Null signed negative/positive counts are 4/4 pooled, 5/3 clean, 2/6 corrupt;
there are no zero null effects. Absolute null ranges (max minus min) are
0.00394583330, 0.00742994668 and 0.00920243934 respectively.

## All probe/group decisions

| Checkpoint | Group | Paired median | Corrected median | Corrected min | Corrected max | Median sign / 8 | Resolved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| joint | pooled | -0.00626181124 | -0.00503579760 | -0.0114577140 | 0.000386401778 | 7 negative | **yes, improvement** |
| joint | clean | -0.00655866344 | -0.00387978251 | -0.0125554572 | -0.000945163425 | 8 negative | no, below null maximum |
| joint | corrupted | -0.00508284708 | -0.00613804255 | -0.0165368309 | 0.00541674625 | 7 negative | no, below null maximum |
| clean | pooled | -0.000830582692 | 0.000107601867 | -0.00591056724 | 0.00402586767 | 4 positive | no |
| clean | clean | -0.00459561148 | -0.00283594034 | -0.00690851780 | 0.00484240986 | 5 negative | no |
| clean | corrupted | 0.00306009501 | 0.000463171862 | -0.00502406247 | 0.00758421607 | 4 positive | no |
| corrupted | pooled | -0.00165019976 | -0.00216311275 | -0.00846532406 | 0.00407093810 | 5 negative | no |
| corrupted | clean | -0.00460326695 | -0.00489598629 | -0.0136581114 | 0.00477537792 | 5 negative | no |
| corrupted | corrupted | 0.00391785940 | 0.00298669096 | -0.0127135394 | 0.0133312559 | 5 positive | no |

All eight paired, null and corrected values, sign counts, min/max summaries and
per-episode paired differences are in [T013F/tables.md](T013F/tables.md), rendered
from [summary.json](remote_runs/20260913-055654-taisp-t013f-matched-checkpoints/artifacts/audit/summary.json).

The pooled joint result must be preserved: its absolute corrected median is
**1.049953 times** the pooled null maximum and its corrected sign is negative in 7/8
cycles, so it passes the predefined rule. The margin is small (about 5% above the
observed floor); this is a descriptive rule pass on the fixed microset.
It does **not** satisfy the separate requirement of resolved clean and corrupt effects.

The joint checkpoint improves clean loss in all eight raw pairs and all eight
corrected cycles, but its corrected median remains below the clean null threshold.
Do not relabel that result resolved just because its sign is consistent. The clean
and corrupt checkpoint subgroup medians show mixed directions; none resolves beyond
the matched floor. Thus neither group-specific resolved cross-harm pattern holds.

## Interpretation and limitations

The R020 discriminator remains **measurement-limited**: no checkpoint has resolved
improvement in both clean and corrupted groups, and neither group-specific checkpoint
has resolved own-group improvement plus opposite-group worsening. The single pooled
joint improvement is insufficient to choose a source-objective regularizer or a
conditional-predictor redesign.

These are eight dependent cycles on four fixed source images, not independent dataset
replicates or evidence of population-level significance. The same-cycle null correction
is the predeclared descriptive control; it does not guarantee all CUDA variation is
cancelled between blocks. No confidence intervals, added repeats, outlier deletion,
optimizer ranking, validation or AP inference was performed. T013-D's local gradient
opposition and common-mode Wh findings remain in force; bias deletion remains unjustified.

## Artifacts and reproduction

Added `taisp/analysis/matched_replay.py`, `tests/test_matched_replay.py`, offline
`scripts/report_t013f.py` and project records. Existing model, loss, adaptation and
deployment code remains untouched. The runner reuses T013-C evaluation and T013-B
source helpers; new code only loads fixed checkpoints, schedules pairs and applies
the prescribed control/decision rule.

[Exact command and release metadata](remote_runs/20260913-055654-taisp-t013f-matched-checkpoints/meta.json),
[run/test log](remote_runs/20260913-055654-taisp-t013f-matched-checkpoints/train.log),
[completion receipt](remote_runs/20260913-055654-taisp-t013f-matched-checkpoints/artifacts/audit/completion.json),
all 64 raw evaluation files, eight cycle files, original/three unchanged checkpoint
copies, environment and frozen-gradient receipts are retained in the run directory.
[The artifact manifest](T013F/artifact_manifest.json) records SHA256 and sizes for
all 83 run files (1,494,476 bytes).

Offline table reproduction, without models:

```powershell
D:\anaconda3\python.exe scripts/report_t013f.py --audit research_log/remote_runs/20260913-055654-taisp-t013f-matched-checkpoints/artifacts/audit --output research_log/T013F/tables.md
```

**NEEDS_REVIEW. Stop after T013-F.** No T013-G, longer training, regularizer, bias
removal, centering, predictor redesign, deterministic-kernel work, preprocessing
change, new data, target/AP, spatial ISP, gating or dose experiment started.
