# T003 final report — NEEDS_REVIEW

2026-09-12, Asia/Shanghai. R004 assigned the study; R005 approved its implementation and within-run pairing. All requested T003 implementation, experiment and reporting deliverables are now complete. Research acceptance is pending. No experiment remains active.

## Main finding and recommendation

The deployable CLIP condition inference is too diffuse to materially repair generic guidance at the fixed temperature: soft-direction mean cosine is 0.04530 versus generic 0.04531, paired difference −0.00002 (95% CI −0.00805 to 0.00715). Soft gating gives cosine 0.05127, but the paired gain also crosses zero. Soft gating lowers saturation and shrinks the mean per-sample gradient norm to 32.98% of the raw norm, so these changes cannot be attributed solely to better update direction.

Correct-family information does expose a limited mechanism benefit: oracle direction plus oracle gate reaches cosine 0.09368, positive alignment 62.33%, and detector-loss benefit 50.58%. Against the contemporaneous generic these differences are +0.04837 cosine [0.00472, 0.09020], +8.50 percentage points positive alignment [4.33, 12.67], and +3.92 points beneficial steps [0.58, 7.08]. Its mean observed one-step loss delta is −0.000557 versus +0.001969 for generic; paired difference −0.002526 [−0.004398, −0.000648]. These exploratory effects are real evidence to review, not robust restoration across conditions.

Downstream AP remains mixed even with correct family information. Oracle-both improves gamma-s2 relative to generic but hurts gamma-s1, contrast-s1 and color-cast-s1; both color-cast severities remain below the corrupted-input AP. Oracle-prompt alone substantially improves gamma-s2 but harms contrast-s2 and color-cast-s1. Therefore condition identification is one bottleneck, and better identification alone does not establish a usable global CLIP supervisory signal. The full result does not support starting meta-training. A next separately assigned study should compare final-global versus layer/patch-local supervision, with condition inference treated as a distinct diagnostic. No T004 or meta-training is launched.

## Experiment and reproducibility

- Run `20260912-031123-taisp-t003-coco200`; release `20260912-031119-taisp-t003-full`; experiment source `8d8913e`; R005 postprocessing source `724c04f`. Exit 0 at 2026-09-12T03:45:51+08:00. Study elapsed 2049.095 seconds (34.15 minutes), excluding startup/tests.
- Same 200 COCO-val IDs as final T002, six settings, six variants: 7200 observations, 36 groups of 200 and 72 AP evaluations. No omitted images, hyperparameter selection or outcome-based rerun.
- Same frozen CLIP ViT-B/32 and FasterRCNN ResNet50-FPN COCO_V1. Same differentiable tensor preprocessing and hard-clamped eight-coordinate ISP. Initial phi=0, lr=0.1, K=3, consistency and regularization weights=0. Oracle native-loss sampling seed=20260912+image_id.
- Corruptions unchanged: gamma 1.5/2.0; contrast 0.6/0.3 about 0.5; RGB gains [1.2,1,0.8]/[1.4,1,0.6], with the same T002 implementations.
- Each image/case has one freshly computed annotated detector gradient/loss shared by all six variants. Generic adaptation is rerun in the same experiment. Clean/corrupted AP is reused on the identical subset; initial detector forward losses match T002 exactly for all 1200 cases. Labels/family selection are confined to analysis.

```sh
export OMP_NUM_THREADS=1
export TAISP_SOURCE_REVISION=8d8913e
export PATH=/home/liujianhua/wjq/TAISP/.venv/bin:$PATH
TAISP_REAL_MODELS=1 python -m pytest -q
python -m taisp.analysis.run_t003 --data-root /home/liujianhua/wjq/TAISP/shared/coco200 --baseline /home/liujianhua/wjq/TAISP/runs/20260912-013248-taisp-t002-coco200-final/artifacts/study --output "$AUTODL_ARTIFACTS_DIR/study"
# Local postprocessing from project root:
python -m scripts.report_t003 research_log/remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study
```

## Exact conditioning and masks

All positive/negative prompts are the original T002 text; only grouping and weighting change. Positive bank:
- `a clear natural photograph`
- `a well-lit photograph with natural colors`
- `a sharp and clear photograph with good visibility`

darkness:
- `a dark underexposed photograph`
- `a poorly lit photograph with low visibility`

low_contrast:
- `a low contrast washed-out photograph`
- `a hazy photograph with faded details`

color_cast:
- `a photograph with an unnatural color cast`
- `a photograph with distorted and unbalanced colors`

Normalize each text embedding, average within each bank and normalize the bank. From original corrupted image only, z=normalize(CLIP(x)); w=softmax(z·t_neg / 0.05). No CLIP learned logit-scale factor. The episode direction is normalize(t_pos − sum(w_m t_neg,m)), without renormalizing the weighted negative mixture. Original features, weights, direction and gate are detached. They are never recomputed from adapted images.

Coordinate order: gamma, red_gain, green_gain, blue_gain, contrast, brightness, tone, sharpening. Masks are exactly:

```text
darkness:     [1,0,0,0,0,1,1,0]
low_contrast: [0,0,0,0,1,0,1,0]
color_cast:   [0,1,1,1,0,0,0,0]
```

Soft gate m=w@M. Actual update phi <- phi −0.1*(m*g_sem), with no norm compensation. Generic/soft/oracle_prompt use all coordinates. soft_gate uses soft direction and soft mask; soft_oracle_gate uses soft direction and the correct-family mask; oracle_both uses correct-family direction and mask. Oracle is a diagnostic with privileged family information, not a guaranteed performance upper bound.

## Overall paired findings

Each variant has 1200 observations from the same 200 images. Intervals: 2000 paired image-cluster percentile bootstrap draws, seed20260912, all six conditions travel together overall; exploratory, no multiplicity adjustment.

| Variant | Mean / median cosine | Positive alignment | Beneficial detector step | Paired Δ benefit, pp [95% CI] |
|---|---:|---:|---:|---:|
| generic | 0.04531 / 0.05630 | 53.83% | 46.67% | +0.000 [+0.000, +0.000] |
| soft | 0.04530 / 0.06837 | 53.25% | 49.00% | +2.333 [-0.500, +5.083] |
| oracle_prompt | 0.07074 / 0.10308 | 55.58% | 50.67% | +4.000 [+0.917, +7.169] |
| soft_gate | 0.05127 / 0.07416 | 54.00% | 47.17% | +0.500 [-2.667, +3.583] |
| soft_oracle_gate | 0.09505 / 0.11424 | 61.75% | 48.00% | +1.333 [-1.750, +4.417] |
| oracle_both | 0.09368 / 0.11896 | 62.33% | 50.58% | +3.917 [+0.583, +7.083] |

Generic Taylor sign agreement is 54.50%, Spearman 0.1472; oracle-both is 51.92%, Spearman 0.0377 (CI −0.0207 to 0.0952). Better effective cosine does not imply a reliable finite-step loss prediction. Native detector proposal/assignment behavior and observed backward numerical variability remain relevant.

## All six families/severities: AP1 / AP3

COCO subset AP in 0–100 points. Clean AP=38.012. Values below are measured, not full-COCO benchmark claims.

| Case | Corrupted | Generic | Soft | Oracle prompt | Soft gate | Soft + oracle gate | Oracle both |
|---|---:|---:|---:|---:|---:|---:|---:|
| gamma_s1 | 38.145 | 38.356 / 38.676 | 38.321 / 38.681 | 38.573 / 38.431 | 38.275 / 38.189 | 38.065 / 38.279 | 38.176 / 38.347 |
| gamma_s2 | 36.446 | 36.594 / 36.399 | 36.439 / 36.489 | 36.621 / 37.108 | 36.511 / 36.399 | 36.421 / 36.534 | 36.449 / 36.614 |
| contrast_s1 | 36.175 | 36.376 / 36.507 | 36.421 / 36.756 | 36.302 / 36.547 | 36.239 / 36.499 | 36.245 / 36.300 | 36.233 / 36.296 |
| contrast_s2 | 31.175 | 31.115 / 31.313 | 31.108 / 31.069 | 30.888 / 30.917 | 31.241 / 31.101 | 31.173 / 31.298 | 31.265 / 31.339 |
| color_cast_s1 | 37.642 | 37.667 / 37.723 | 37.632 / 37.657 | 37.274 / 37.267 | 37.566 / 37.727 | 37.707 / 37.667 | 37.314 / 37.311 |
| color_cast_s2 | 35.908 | 35.842 / 35.528 | 35.859 / 35.599 | 35.801 / 35.804 | 35.918 / 35.884 | 35.775 / 35.613 | 35.848 / 35.798 |

Full within-run AP1/AP3 differences, per-case cosine/positive alignment/beneficial-step changes, paired loss/saturation/latency CIs and Taylor Pearson/Spearman distributions are in [results.md](remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/results.md) and [analysis.json](remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/analysis.json). AP differences use the same fixed images, but no AP bootstrap intervals were computed; AP is not an average of per-image AP.

## Degradation-state inference

Overall top-1 family accuracy is 45.50% on these balanced synthetic settings (uniform-chance reference 33.33%). Weight entropy is approximately 98.1–98.4% of its maximum, and mean maximum weight is only about 0.40. Color-cast identification is poor; high contrast identification accuracy still produces diffuse weights. Family IDs are used only to calculate this analysis table.

| Case | Mean darkness / contrast / color weights | Correct top-1 | Normalized entropy |
|---|---:|---:|---:|
| gamma_s1 | 0.3350 / 0.3305 / 0.3345 | 38.5% | 0.9841 |
| gamma_s2 | 0.3478 / 0.3212 / 0.3310 | 43.0% | 0.9835 |
| contrast_s1 | 0.3158 / 0.3780 / 0.3062 | 61.0% | 0.9826 |
| contrast_s2 | 0.3114 / 0.3935 / 0.2951 | 71.5% | 0.9814 |
| color_cast_s1 | 0.3105 / 0.3561 / 0.3334 | 30.5% | 0.9831 |
| color_cast_s2 | 0.3014 / 0.3660 / 0.3327 | 28.5% | 0.9821 |

## Coordinate mechanism, saturation and runtime

T002 offline Stage A is complete in [coordinates.md](T003_stage_a/coordinates.md): mean outside-subspace norm fraction0.8244, energy0.7205. Darkness and contrast have especially large cross-talk; contrast has essentially zero saturation. These are descriptive subspace diagnostics, not a causal decomposition of failure.

In T003, raw soft-direction outside-subspace energy is 72.41%; the soft gate still leaves 70.14% of effective energy outside the correct analysis subspace. Correct-family hard gates make that energy zero by construction, which is not itself evidence of task alignment.

| Variant | Raw / effective outside energy | Effective/raw norm ratio | Mean saturation3 | Own CLIP loss decreases at3 | Mean seconds3 |
|---|---:|---:|---:|---:|---:|
| generic | 72.05% / 72.05% | 1.0000 | 3.470% | 93.33% | 0.1179 |
| soft | 72.41% / 72.41% | 1.0000 | 3.463% | 93.50% | 0.1232 |
| oracle_prompt | 69.10% / 69.10% | 1.0000 | 3.250% | 94.75% | 0.1261 |
| soft_gate | 72.41% / 70.14% | 0.3298 | 2.651% | 97.58% | 0.1248 |
| soft_oracle_gate | 72.41% / 0.00% | 0.4413 | 2.438% | 94.67% | 0.1252 |
| oracle_both | 69.10% / 0.00% | 0.4672 | 2.414% | 93.92% | 0.1264 |

Gamma-s2 saturation after3 steps is 8.33% generic, 5.55% soft-gate, 3.20% oracle-both; p95 is 31.16%,20.35%,15.91% respectively. Yet beneficial one-step frequencies are45.5%,45.5%,52.5%, and the paired oracle-both improvement CI still includes zero within gamma-s2. Its post1 saturation buckets show harmful fractions47.5%,59.1%,28.6%,40.0% (N=120,44,21,15), without a monotonic pattern. Gamma-s1 has some high-saturation small groups with more harm, but these do not justify a broad causal claim. Both contrast settings remain at zero saturation across variants while still showing mixed detector/AP effects. A clamp ablation should remain a separate research decision.

Runtime includes condition inference/setup and three-step adaptation with diagnostics/GPU synchronization, excluding annotated loss/AP evaluation, image loading and common-CLIP diagnostics. Mean extra time versus generic is about5–9ms. Mean peak allocated GPU memory is about839MiB, with model residency included. Oracle timing includes the common conditioner invocation and is implementation timing, not a minimal theoretical oracle cost. Per-case latency/saturation paired intervals and complete raw/physical trajectories are retained.

## Validation, environment and observed failures

- Final A6000 full suite:34 passed in7.79s, immediately before full study. Relevant tests cover detached condition equations, masks/SGD, frozen parameter/buffer state, phi gradients, episode reset, detector oracle isolation and baseline preprocessing/evaluation.
- R005 postprocessing: paired-bootstrap and coordinate known-example tests2 passed in1.18s; full smoke renderer passed. Final receipt audit verified7200 rows,36 groups×200,200 IDs, all-zero fresh phi0, identical initial detector gradient/loss across variants, and effective-gradient=gate×raw-gradient.
- Two strict cached-equality smokes failed and are retained:20260912-030241 (native detector gradients max discrepancy1.08e-4) and20260912-030559 (three-step generic phi discrepancy6.2841e-7). Repeat diagnostic20260912-030416 reproduced same-process detector gradient variation despite exact forward loss. Protocol repair uses fresh paired gradients and fresh generic; equality tolerances were not widened.
- First failed smoke command mislabeled its revision35d074b; actual c8ac8ff was verified by local/remote SHA and documented in SOURCE_CORRECTION.md. A local fetch/report ordering mistake was fixed by waiting for SCP completion; no model/code change.
- Final cross-run audit: initial detector forward loss max difference0; detector gradient max difference0.0015283 versus T002; generic AP differences versus T002 are below0.01 points. Primary conclusions use within-T003 pairing, never cached T002 gradients.
- CUDA backward nondeterminism and this one-subset, one-seed exploratory study limit generalization. Confidence intervals reflect image sampling, not between-run numerical variance or multiple comparisons. No temperature/lr/prompt tuning was performed.

Remote environment: Python3.12.12, torch2.4.0, torchvision0.19.0+cu121, Transformers4.44.2, pycocotools2.0.10, NumPy1.26.4, Pillow12.3.0, CUDA12.1, RTX A6000. Local postprocessing uses Python3.12.7, NumPy1.26.4, SciPy1.13.1, Matplotlib3.9.2.

CLIP revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; weight SHA256 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`. Detector weight SHA256 `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`. Dataset annotation SHA256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`. The subset manifest retains exact IDs/image hashes.

## Artifacts and next action

- [Full paired tables](remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/results.md), [all statistics](remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/analysis.json), [AP figure](remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/ap_change.pdf), [environment](remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/environment.json), [completion](remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/completion.json).
- Raw7200 rows exceed GitHub single-file size; lossless [samples.jsonl.gz](remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/samples.jsonl.gz) is tracked instead, with [SHA256/round-trip receipt](remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/samples_archive.json) and [decompression instructions](remote_runs/20260912-031123-taisp-t003-coco200/artifacts/study/ARCHIVE.md). Raw JSONL remains local and on A6000; no receipt was deleted. All72 prediction files are retained.
- Remote experiment artifacts remain in /home/liujianhua/wjq/TAISP/runs/20260912-031123-taisp-t003-coco200/artifacts/study. Reports and recovery notes are mirrored to the remote project research_log.
- Source changes: conditioned_clip.py, optional gate in adapt.py, run_t003.py/config, coordinate/report scripts and focused tests. All existing model/data/ISP/oracle/evaluator owners were reused. No learned prompts, predictor/source/meta-training, detector updates, smooth clamp or ViT³ structures.
- T003 status NEEDS_REVIEW. Await research lead acceptance/next explicit task; heartbeat executes newly assigned work directly and avoids rerunning this completed experiment.
