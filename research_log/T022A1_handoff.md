# T022-A1 IN_PROGRESS — exact R035 parity repair

2026-09-14 +08. R035/0b46353 authorizes wrapper initialization and separate CLIP
parameter/eval/hash receipts only before the same 28-record rerun. Baseline is
T022-A's 182 passed/10 skipped and preserved first-record harness blocker.
Changed only parity harness and one focused test. Reuse accepted runtime
.eval().requires_grad_(False); loader, runtime/formula, ISP, masks, cohort and
all tolerances unchanged. No active run yet. Next focused/full tests and exact
A6000 parity; any failed record stops without AP. Only all-pass permits K3 smoke,
then shared driver completion and original R034 study. See R035 exact contract.

2026-09-14T00:10+08:00 R035 correction8948d4c deployed20260914-001000-taisp-t022a1-parity; run20260914-001023-taisp-t022a1-numerical-parity launched once. Focused21passed1skipped3.57s; full/parity ongoing. Exact same config/cohort, original failed rawrun immutable. NoAP.

2026-09-14T00:12+08:00 T022-A1 parity001023 completed exit0 at00:11:23: all28originalrecords/allCLIPsubchecks/finalhashes pass; focused21passed1skip3.57s/full183passed10skip8.25s. RawSHA655ad1ec8e13e0b015cb81d430fa5c11e2b97f037ce30b6ea8c45cf1f4c10afc fetched/verified. Proceed original R034 driver and K3zeroAPsmoke, no formula/runtime changes.

2026-09-14T00:15+08:00 Original R034 shared-driver integration completed only for spatial_dose_ours. Current branch unchanged; pure gate mapping and per-episode/edge diagnostics added. Protected18 modules and prior spatial runtime/ISP hashes unchanged. Increment focused18passed2warnings4.84s on release20260914-001435. Next fullregression + real K3smoke, including repeated fullmask/emptymask/empty-support episodes; zero AP.
