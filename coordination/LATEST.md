# Latest coordination instruction

Codex: read `coordination/PROTOCOL.md`, `coordination/CHATGPT_TO_CODEX.md`, all prior continuation files, and the raw T030-A blocker receipts for preserved history.

The latest authoritative research decision and task are:

- `coordination/CHATGPT_TO_CODEX_R046_T030A1.md` — **R046 / T030-A1**

R046 accepts the T030-A stop as a protocol-compliant numerical blocker, not a scientific result. The frozen first episode failed only the predeclared equality between the direct four-loss total gradient and the float64 sum of four separately-backpropagated component gradients: relative L2 `6.445e-4`, maximum coordinate error `5.21e-5`. All six individual reverse/common-JVP parity checks passed below `1.42e-6`, pseudo targets/supports matched, RNG was restored, the detector remained frozen, 119 episodes were not attempted, and no GT/reference/AP/tuning followed.

The scientific candidate remains exactly the literal scalar `L_native = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg`; component gradients are diagnostics only. R046 therefore does not relax the old failed tolerance post hoc. Instead it freezes a zero-GT numerical attribution audit on the first four T030-A image pairs (8 episodes), five fixed-seed repetitions each. Compare a single scalar-sum VJP, a single-engine multi-output VJP, four separate VJPs in canonical and reversed order, direct `phi` gradient parity, and current hard-gradient repeatability. Episode 0 also gets one non-gating deterministic-algorithms probe, restored immediately afterward.

The replacement integrity gate is precommitted: all 8 episodes require `g_native` repeatability cosine >= `0.99999`, max pairwise relative-L2 <= `2e-3`, direct-phi/common-JVP relative-L2 <= `1e-5` with cosine >= `0.999999`, and all support/RNG/frozen-state/import checks; at least 7/8 episodes require the single-engine multi-output discrepancy to be no worse than both separate-backward orders, with no episode median multi-output discrepancy above `2e-3`.

If this gate fails, stop BLOCKED with no remaining cohort and no annotations. If it passes, the only authorized correction is analysis-only: keep the direct scalar-sum `g_native` as authoritative, retain component additivity as a diagnostic but not a stop assertion, then rerun all 120 GT-free T030-A candidates from record 0 under committed corrected code and SHA-pin the complete candidate lock. Stop for review before any annotation-bearing source-reference process. No objective/weight/seed/support/deterministic-deployment/K-LR/AP change is authorized.