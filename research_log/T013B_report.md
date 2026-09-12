# T013-B meta-gradient fidelity and source meta-step feasibility

**NEEDS_REVIEW. Both authorized parts completed.** Part A passes the predeclared
fidelity gate; Part B completes exactly three predictor-only SGD updates on the
fixed four-image train2017 microset. No AP, independent target evaluation or
generalization claim is produced. No further training is active.

Plan **b73bcd2** preceded outcome-bearing code/runs. Part A code **42072e7**;
microset **fc88531** was committed/pushed before optimizer execution; Part B code
**1685fb6**. R016/83f734e accepted T013-A, whose strict CUDA failure is untouched.

## Part A: exact reference validated; FO direction passes this toy audit

All 12 fixed fixtures use the existing 8D ISP, CPU float64, K=3, lr=.1, norm
epsilon1e-12, central FD epsilon1e-5, and the exact predeclared image/state/loss
construction in `T013B_plan.md`. EXACT uses `create_graph=True` only inside
`taisp.analysis.meta_gradient`; deployment/training APIs have no exact option.
FO calls the existing first-order initializer path without changes.

| Episode | cos(FO,EXACT) | cos(EXACT,FD) | Coordinate sign agreement | FO/exact norm ratio |
| --- | ---: | ---: | ---: | ---: |
| 0 | .998700311 | ~1 | 8/8 | 1.012788355 |
| 1 | .998032202 | 1 | 8/8 | 1.005225850 |
| 2 | .997488054 | 1 | 8/8 | 1.005234316 |
| 3 | .998102743 | 1 | 8/8 | 1.005316361 |
| 4 | .997636091 | 1 | 8/8 | 1.005321127 |
| 5 | .997498174 | 1 | 8/8 | 1.011927007 |
| 6 | .997815375 | ~1 | 8/8 | 1.009309542 |
| 7 | .997952252 | 1 | 8/8 | 1.017309930 |
| 8 | .998441977 | 1 | 8/8 | 1.026804283 |
| 9 | .998711858 | 1 | 8/8 | 1.025358076 |
| 10 | .999203421 | ~1 | 8/8 | 1.031209237 |
| 11 | .999601896 | 1 | 8/8 | 1.033592462 |

EXACT/FD minimum cosine is **.9999999999999998**, and maximum absolute gradient
error across all coordinates/episodes is **5.7693017030e-12**. The reference is
valid under the precommitted >=.999 criterion. Every trajectory and final image
is exactly equal between FO and EXACT; all terminal saturation rates are zero.
All gradients are finite; no singular all-zero case was removed or replaced.

Continuation gate: median cos(FO,EXACT) **.9980674725590613 >= .5**;
**12/12 positive >=9/12**; **all finite**. Therefore Part B was allowed.
The raw vectors, complete trajectories and exact per-episode metrics are retained
in [audit.json](T013B/part_a/audit.json) and [audit.csv](T013B/part_a/audit.csv).
This establishes fidelity only on these small quadratic mock objectives, not on
the true Faster R-CNN/CLIP meta-gradient.

## Part B: fixed source-only data and update protocol

Existing train2017 annotation file and image directory were located under
`/home/wenchang/asdasdsad/wjq/dapd_query_opt44/assets`. The directory contains
14,874 JPG entries: 998 inaccessible symlinks and 13,876 readable images. Of these,
13,798 have valid non-crowd positive-area annotations. Selection uses the planned
`random.Random(20260913).sample(sorted(valid_available_ids),4)`.

| Ordered image ID | Image size | Corrupted episode (in addition to clean) |
| --- | --- | --- |
| 65088 | 500x350 | gamma_s2 |
| 426525 | 640x424 | contrast_s2 |
| 541157 | 500x375 | color_cast_s2 |
| 129068 | 640x480 | gamma_s1 |

All four original COCO URLs explicitly identify train2017. Image bytes, annotation
file hash, IDs, original annotations, order and corruption assignments were
committed in [T013B_train_microset.json](T013B_train_microset.json) at **fc88531**.
No validation image, download, target cohort or data alteration was used.

Each episode's original source prediction produces detached score>=.5/top20
pseudo support once. Each outer evaluation resets from `P_psi(x_episode)` and runs
the unchanged K=3 accepted full hybrid. Ground truth is passed only to the
analysis `source_outer_episode` / existing `oracle.detector_task_loss` outer loss.
The oracle's existing loss-return branch flags are temporarily enabled, then all
modules restored to eval; parameters stay frozen throughout. No label or
corruption identifier enters predictor inputs or the inner loss.

Only the existing predictor is optimized: plain SGD1e-3, mean of eight episode
losses, exactly three updates in fixed order. Detector sampling seed20260913 is
reset by the existing oracle machinery. Step3 is the final measurement/backward
for gradient diagnostics; **no fourth optimizer update** is applied.

## Part B measured results (not detection-performance evidence)

| Step before update / final | Mean outer loss | Clean outer loss | Corrupted outer loss | Head gradient norm | Trunk gradient norm | Predictor change norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | .474083306 | .355870174 | .592296437 | .168584347 | 0 | 0 |
| 1 | .472119273 | .349765647 | .594472898 | .174203992 | 1.493231e-5 | .000168584360 |
| 2 | .475948031 | .351801782 | .600094279 | .187235892 | 3.299718e-5 | .000341098668 |
| 3 final | .473895423 | .351741889 | .596048958 | .191115558 | 5.162177e-5 | .000527524506 |

The sequence is **nonmonotonic**. Overall final loss is only .000187882455 lower;
clean loss decreases while corrupted loss increases. This is not interpreted as
a method improvement. Loss decrease was not an acceptance condition. The initially
zero head necessarily gives zero feature-trunk gradient at step0; after the first
head update the trunk receives finite, nonzero gradients. Every head gradient is
finite/nonzero, parameters actually change, and the final predictor checkpoint
is retained without promotion to deployment.

| Step | Clean phi0 mean | Corrupted phi0 mean | Clean phi3 mean | Corrupted phi3 mean | Clean saturation mean | Corrupted saturation mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | .071202720 | .019964728 | .013971020 | .013220601 |
| 1 | .000175177 | .000173393 | .098663848 | .019716651 | .018109011 | .012089172 |
| 2 | .000355062 | .000351340 | .097237059 | .019831144 | .016801470 | .012741553 |
| 3 final | .000550002 | .000544378 | .090419135 | .019958154 | .015670632 | .014551076 |

Clean phi3 remains markedly larger and rises from its initial mean; the test does
not establish identity preservation. Final clean/corrupted phi0 maxima are
.000564660/.000576353, phi3 maxima .190454721/.027042057, and saturation maxima
.045055769/.056859046. All eight supports are nonempty at every measurement;
no full-image saturation or nonfinite predictor gradients occurs. Per-episode
states, all intermediate group maxima, original saturation and all four native
loss components are in the raw receipt.

All detector/CLIP parameters and buffers match their pre-run state at every
measurement, all `.grad` fields remain None, and all submodules are eval after
the outer call. ISP-owned phi remains zero with no gradient. Peak allocated CUDA
memory is **3,712,453,632 bytes**; source smoke elapsed **15.003842537s**, including
model loading/support preparation and final measurement.

## Implementation and tests

New analysis modules: `meta_gradient.py` (mock exact/FD audit) and
`source_meta_smoke.py` (fixed source-only outer runner). New preparation script:
`scripts/prepare_t013b.py`. Tests: `test_meta_gradient.py` and
`test_source_meta_smoke.py`. The latter varies outer labels and confirms identical
inner trajectories/images for the same input across intervening episodes.

`git diff bfd2484 -- taisp/tta taisp/training taisp/isp taisp/models taisp/losses
scripts/smoke_t013a.py` is empty: all accepted deployment/initializer/model/loss
code and the T013-A failed strict assertion are untouched.

| Validation | Result |
| --- | --- |
| Pre-edit initialization + trust-radius baseline | 8 passed,1 skipped,19.17s |
| Part A focused tests + initialization/trust-radius | 11 passed,1 skipped,16.39s |
| Part B affected tests (source smoke, meta gradient, initialization, trust radius, detector native, adapt) | 23 passed,2 skipped,17.88s |
| Complete A6000 suite, TAISP_REAL_MODELS=0 | **93 passed,10 skipped,5.94s**;2 pre-existing NVML warnings |
| Part B source three-step run | **exit0**, exactly3 updates |

The ten opt-in model tests are skipped in the full regression command; the actual
source/CLIP test is the separate authorized Part B smoke, not a claim that all
opt-in tests ran. No tests were weakened or removed.

Local environment: Python3.12.7,torch2.13.0+cpu, one thread, float64 audit.
Remote: Python3.12.12,torch2.4.0+cu121,NVIDIA RTX A6000, one thread, float32 models.
Run **20260913-014612-taisp-t013b-source-three-steps**, release
**20260913-014535-taisp-t013b-source-smoke**, source **1685fb6**.
Started01:46:16+08; finished01:46:40+08 on2026-09-13, exit0.

Exact remote command (from release cwd, absolute existing project venv):

```bash
export TAISP_SOURCE_REVISION=1685fb6 TAISP_REAL_MODELS=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/liujianhua/wjq/TAISP/.venv/bin/python -m pytest tests -q &&
/home/liujianhua/wjq/TAISP/.venv/bin/python -m taisp.analysis.source_meta_smoke --manifest research_log/T013B_train_microset.json --audit research_log/T013B/part_a/audit.json --output "$AUTODL_ARTIFACTS_DIR/smoke"
```

Part A command: `D:/anaconda3/python.exe -m taisp.analysis.meta_gradient --output
research_log/T013B/part_a`, with PYTHONUTF8=1 and OPENBLAS/OMP_NUM_THREADS=1.
All exact commands and outputs are retained in local logs/run metadata.

## Failures, receipts and stop

The initial data-preparation attempt hit PermissionError on an inaccessible
symlink pointing below root-owned `/root`. Its cause was confirmed with `namei`;
readability inspection found the existing usable images. The minimal repair
checks readability before testing file existence. It implements the predeclared
available-image filter; it does not change the seed, objective or dataset.
The initial inventory shell also returned exit2 for absent optional root paths;
positive findings were verified separately. No model-run failure or retry occurred.
Known NVML/tokenizer warnings remain unchanged.

Raw Part B logs, support sets,32 episode measurements, four aggregate records,
final checkpoint and run metadata are under
`research_log/remote_runs/20260913-014612-taisp-t013b-source-three-steps` locally
and `/home/liujianhua/wjq/TAISP/runs/20260913-014612-taisp-t013b-source-three-steps`
remotely. Source manifest SHA164bef1b809fd8fa763ed8954386ea39443a2d7786b90c54ca0673de4bf49493
and Part A JSON SHA8176e57c52fcfa67eebeb3e53008b903bc45c4d7f2657066ae65c534fe65a2a4
match the recorded remote inputs.

**Recommendation:** review this as successful bounded software/gradient feasibility
with limited toy fidelity evidence and a nonmonotonic real source objective.
The clean/corrupted asymmetry remains unresolved; no detection or generalization
benefit is established. Stop here and await the next explicit package. No T013-C,
longer training, schedule selection, architecture change, val/FCOS/SSD evaluation,
new target cohort, spatial ISP or gating/dose work is started.
