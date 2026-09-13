# T022-A BLOCKED — pre-AP CLIP wrapper initialization

2026-09-13. R034 / 6001e14; plan b677a0c; implementation e8272c5.
Run `20260913-234326-taisp-t022a-numerical-parity`, release
`20260913-234251-taisp-t022a-parity`, 23:43:31–23:43:53 +08, exit 1.
A6000 CUDA was used for real models, gradients and numerical checks.

**Implementation blocker, not a scientific FAIL.** One of 28 planned real-model
records was written; 27 remain unexecuted. No official AP evaluations occurred.
The formal candidate driver, real K=3 smoke and 200-image study remain pending.
The isolated candidate and its unit tests are implemented. No frozen method,
prompt, learning rate, rho, mask rule, threshold or tolerance was changed.

## Tests and first real record

Focused: **20 passed, 1 skipped, 3.74 s**. Full regression: **182 passed,
10 skipped, 4 warnings, 8.40 s**. Earlier focused unit repair and its failed
receipt are retained in `T022A_unit_failure.md`; the repaired suite passed
19 tests with 1 skip. The repair moved image detachment outside the JVP closure.

First real record: image 160585, gamma_s1, equal states zero, 3 supports,
object-mask fraction 0.005126953125.

| Check | Observed | Result |
| --- | --- | --- |
| Pseudo common-shift relative L2 | 1.2348761934872946e-6 | PASS |
| CLIP common-shift relative L2 | 2.0502338781629528e-7 | PASS |
| Mean update relative L2 | 7.401833374038719e-7 | PASS |
| Minimum of the three cosine similarities | 0.9999999999997277 | PASS |
| Pseudo float64 partition max error | 3.3306690738754696e-16 | PASS |
| CLIP float64 partition max error | 6.938893903907228e-17 | PASS |
| Equal-state image max error | 0 | PASS |
| Common CLIP norm relative error | 0 | PASS |
| Dose coefficients | c=-1; multipliers 0.5 / 1.5; mean 1 | PASS |
| Source frozen/eval/grad-none | true | PASS |
| CLIP frozen/eval/grad-none | false | BLOCKER |
| ISP identity/grad-none | true | PASS |
| Final source and CLIP state hashes unchanged | both true | PASS |

These results cover one record only; they do not establish complete parity,
real episode reset, full/empty real masks, or method efficacy.

## Cause and bounded next repair

Code inspection identifies a missing initialization in the standalone parity
entry point: `load_clip_guidance` returns a `SemanticDirectionLoss` wrapper whose
top-level `training` flag defaults to true. Its encoder and underlying CLIP model
are initialized eval/frozen. `collect` in `spatial_dose_parity.py` does not apply
`.eval().requires_grad_(False)` to that wrapper after loading. Both the accepted
`adapt_clip_radius` runtime and new `adapt_spatial_dose` runtime already do so.
The raw combined isolation flag does not identify individual subchecks, so the
wrapper explanation is based on code inspection; unchanged weight hashes are
independently recorded. This is not evidence of trained or mutated CLIP weights.

The minimal proposed repair is to initialize the wrapper in this standalone
entry point exactly as the runtime does, then repeat the same frozen 28-record
pre-AP checks. No protected loader change or numerical tolerance relaxation is
needed. **This repair/re-run has not been applied after the stop.** R034 states:
“Any material parity/isolation failure is a blocker; preserve it and stop rather
than loosening tolerances after seeing AP.” The failed receipt is preserved for
research review under that instruction.

Raw `blocker.json` uses the generic wording “numerical parity failed”; its first
record shows that all numerical checks passed and isolation caused the stop.
Raw files are unmodified. No scientific negative inference is drawn from c=-1
or this single image.

## Provenance and advancement gate

Cohort: 200 precommitted fresh train2017 IDs, four fixed blocks of 50;
SHA256 `589878ad7af1d86bf73514d63dd58a4e40d58797d119ea4981be9e4292772690`.
Exclusion set: 1236 prior source/debug IDs plus all 5000 val2017 IDs.
Actual remote release hashes match all 18 protected modules, 3 new modules,
and the unchanged shared driver baseline (22 files total).

Gate flags: macro improvement >= +0.10 **NOT EVALUATED**; positive blocks >=3/4
**NOT EVALUATED**; positive conditions >=4/6 **NOT EVALUATED**; above raw
**NOT EVALUATED**; clean delta >=-0.10 **NOT EVALUATED**; no blocker **FALSE**.
Overall **BLOCKED**, with no scientific PASS/FAIL and no candidate promotion.
No active job remains; no AP, new cohort, parameter sweep or subsequent task
was started. Resume only from a new explicit research decision in the queue.

Raw archive SHA256 `e2313190af626aff4adf339c525ff7f0ec94b64339807c02daaa484b13ffcc36`; 9 files, 20,745 bytes.
See `T022A/artifact_manifest.json`, `T022A/remote_code_hashes.json`, and
`remote_runs/20260913-234326-taisp-t022a-numerical-parity/` for exact receipts.
