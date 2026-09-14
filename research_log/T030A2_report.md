# T030-A2 / R047 — numerical PASS; 120 candidates locked; NEEDS_REVIEW

All four precommitted numerical gates pass on the fresh next four T030 image pairs. The conditionally authorized analysis correction and full 120-episode GT-free candidate regeneration are complete. Stop for research review. There is no source-reference alignment, AP, K-step adaptation or scientific utility conclusion. R045/R046 remain BLOCKED under their original rules; no retrospective relabeling.

## Provenance and reuse

Research b083ffd; pre-outcome plan/manifest1d4b158; diagnostic codee5ed293; diagnostic result lock8c5991e; corrected runner59efa11. Audit cohort IDs27717,74938,347235,413056 retain originalT030 order/corruptions and JPEG hashes. Original60-image cohortSHA3ac9eb54750f0a2b197fa37205d33f493e039b9932b785830288ba997dd32752. Eight-episode manifestSHA4b43fc9bb5ec5f94e9fd00fda3bdf37be16c44c6c53be9963f2589d07c6fbf82.

Unchanged frozen Faster R-CNN weightSHA258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384 and stateSHA73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf. A6000CUDA0, torch2.4.0+cu121, native samplingseed20260930/setupseed20260913, unchanged nondeterministic CUDA environment. No CuBLAS setting/probe change. Existing native_task_signal, DetectorNativeLoss pseudo-target support, roi_equivariance setup/images/common_jvp reused. No annotation JSON opened by diagnostic/candidate process.

Audit run20260914-151141-taisp-t030a2-attribution8x5, release20260914-150932-taisp-t030a2-attribution, completed40repetitions in48.18670016800752s. Exact command in saved run.sh: python -m taisp.analysis.native_chain_attribution --manifest research_log/T030A2/diagnostic_images.json --output "$AUTODL_ARTIFACTS_DIR/diagnostic", TAISP_SOURCE_REVISION=e5ed293.

Support is generated once per episode, score>=.50/stabletop20, identical detached boxes/classes for five fresh native forward graphs. Each retained graph executes sum_a, sum_b, sum_c, one-engine multi-output, canonical4 and reversed4 component VJPs in that exact order. The exact detached sum_a cotangent contracts both the existing common8-column ISP JVP and the corresponding original ISP graph VJP. No second detector call enters the chain comparison. Native candidate is sum_a's projected vector, never a mean or selected diagnostic. All relative discrepancies use norm(a-b)/max(norm(a),norm(b),1e-12).

## Frozen gates

|Gate|Observed|Verdict|
|---|---|---|
|1: all8 candidate cosine>=.99999 and pairwise rel<=.002|mincos0.9999984580853699; maxrel0.0017662375053865307|PASS|
|2: every same-cotangent ISP chain rel<=1e-5, cos>=.999999, finite/nonzero|40/40; maxrel8.961290554393774e-07; mincos0.9999999999996394|PASS|
|3: all8 finite scalar self-noise, median<=.002, withinrep cos>=.99999|maxepisode median0.0004972147051116427; mincos0.9999998619067585|PASS|
|4: >=7/8 allthree medians<=2*noise+1e-5; individual<=.002; integrity|8/8; maxindividual0.0005720718925654897; all40integrity|PASS|

|Image / case|candidate maxrel|candidate mincos|chain maxrel|scalar median noise|multi median|separate median|reverse median|noise explained|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|27717 / clean_s0|0.000517896223|0.999999877714|6.63623556e-07|0.000406934529|0.000417229061|0.000453578779|0.000453606892|True|
|27717 / contrast_s2|0.000410223829|0.999999978162|6.87147496e-07|0.000401983496|0.000422488718|0.00046858538|0.000471247855|True|
|74938 / clean_s0|0.000555409561|0.999999975317|5.40554988e-07|0.000408987153|0.00041216377|0.000519609183|0.00051813097|True|
|74938 / color_cast_s2|0.000211479077|0.999999992371|5.17643461e-07|0.000472431065|0.000472049226|0.000522489749|0.000534176924|True|
|347235 / clean_s0|0.000656055516|0.99999980331|6.29187871e-07|0.00046416793|0.000448420964|0.000527598242|0.000528038749|True|
|347235 / gamma_s2|0.00176623751|0.999998458085|8.96129055e-07|0.000497214705|0.000501630518|0.00055329586|0.000548765921|True|
|413056 / clean_s0|0.000400245101|0.999999920714|4.43297196e-07|0.000426913884|0.000398318574|0.000468892604|0.000475737194|True|
|413056 / contrast_s2|0.000268381167|0.999999965952|6.12386764e-07|0.000422675955|0.000423245646|0.000497293474|0.000486751772|True|

Fullprecision all40 comparisons inper_repetition.tsv; all8D vectors, image cotangent SHA/norm/dtype, scalar losses, pseudo-target/support and RNG/state hashes in raw per-repetition/episodeJSON. Six full cotangents per repetition retained in40server .pt files, verified against rawsha256.json; tensor_storage_index.json locates all raw tensors. Local D disk constraint means tensors stay expanded onserver; small complete JSON/TSV/hash receipts are local andGitHub. No tensor deletion or new image selection occurred.

These measurements separate the ISP chain from detector reverse-sweep variation: same-cotangent ISP errors are below1e-6, while repeated scalar reverse sweeps on one detector graph show roughly4e-4–5e-4 image-gradient noise. Formulation discrepancies fit the frozen measured-noise bounds. This supports the bounded numerical attribution; it does not identify a specific CUDA operator or establish task utility.

## Conditional correction and 120-record lock

After diagnostic PASS was committed, only pseudo_native_candidates.py and its tests were corrected: keep authoritative scalar-sum reverse; keep four component/additivity receipts; remove old component-additivity from stop assertion; log one-engine multi-output diagnostic; keep same-cotangent ISP parity for every gradient. The original candidate runner had no fresh-detector direct-phi equality assertion, so none needed removal. No native objective, weights, support, seed, cohort, ISP or deployment change.

Run20260914-151622-taisp-t030a2-candidates120, release20260914-151439-taisp-t030a2-candidates, source59efa11, fromrecord0 originalresearch_log/T030A/candidate_images.json. Exact saved command: python -m taisp.analysis.pseudo_native_candidates --manifest research_log/T030A/candidate_images.json --output "$AUTODL_ARTIFACTS_DIR/candidate". Completed120episodes in48.15833077405114s. All120 support/target/RNG/freeze and all7same-cotangent gradient parity checks pass. Component-additivity diagnostic passes0/120; these diagnostics do not alter the candidate or stop it.

Candidate recordsSHA256 **031731120f517c03bfb8a6a5ae595676ae26b6bb6ba6076cbcfde00be6a7bdb6**; candidate rawmanifestSHA256 **e5b019de2034f1697c5e630538869bf487674ca9cd06b10a3ed489bbdf6deac7**. All120freshrecords preserved individually plus records.json. No failedR045record reused. Everyrecord includes supports, boxes/classes, allfour native losses, authoritative gradient, hard gradient, multi/component diagnostics and isolation; environment.json/run.sh provide code/model/data provenance.

## Tests and integrity

Baseline5passed5.15s (initial invocation referenced a nonexistent test filename and ranzero tests, corrected before code edits). Auditfocused15passed4warnings10.45s/full258passed11skipped4warnings22.76s. Correctedfocused16passed3warnings10.37s/full259passed11skipped4warnings22.60s. Existing NVML/protobuf warnings retained. Tests cover sharedgraph reuse, chain parity with exactcotangent, symmetric relative/zero semantics, exactGate1–4 boundaries, native scalar identity, non-gating decomposition logging, frozenstate/RNG restoration and no-reference import boundary. Model-bearing tests and both real runs onCUDA; local CPU report/hash work only.

46protected/prior files unchanged; diagnostic49and corrected50remote hashes match. Both runs finaldetectorstate identical, no reference/oracle/CLIP import, GT_loadedfalse, APcalls0. Oldreports/receipts preserved. Task statusNEEDS_REVIEW: authorized numerical audit andcandidate lock complete; stopbeforeannotations/reference/AP/K-step. Futurefreshcohort exclusions remainT030A_train_cohort.json (3011reserved/prior IDs plus5000val).
