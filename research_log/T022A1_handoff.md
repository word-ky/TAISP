# T022-A1 IN_PROGRESS — exact R035 parity repair

2026-09-14 +08. R035/0b46353 authorizes wrapper initialization and separate CLIP
parameter/eval/hash receipts only before the same 28-record rerun. Baseline is
T022-A's 182 passed/10 skipped and preserved first-record harness blocker.
Changed only parity harness and one focused test. Reuse accepted runtime
.eval().requires_grad_(False); loader, runtime/formula, ISP, masks, cohort and
all tolerances unchanged. No active run yet. Next focused/full tests and exact
A6000 parity; any failed record stops without AP. Only all-pass permits K3 smoke,
then shared driver completion and original R034 study. See R035 exact contract.
