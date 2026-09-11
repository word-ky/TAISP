# T004 final report — NEEDS_REVIEW

2026-09-12, Asia/Shanghai. R006 cfdec6e assigned this study after accepting T003. All T004 implementation, fixed-subset experiment and report deliverables are complete. No active experiment remains; research acceptance is pending.

## Finding and recommendation

Last-layer patch-token supervision does not provide a supported overall improvement in deployable semantic-gradient alignment. Global-generic mean cosine is0.04531, versus0.01529 for uniform patches and0.02003 for region-weighted patches; paired differences are−0.03002 [−0.08513,0.02399] and−0.02528 [−0.07905,0.02981]. Primary beneficial detector-loss steps rise slightly from47.17% to48.58%/48.75%, but both paired intervals cross zero. These results do not support global pooling being the main bottleneck under the tested readout.

The positive oracle result must be separated from the representation effect. Norm-matched region-oracle updates benefit51.50% of samples, +4.33 percentage points [1.17,7.58] versus global-generic. Contrast-s2 contributes a+11.0-point improvement [2.0,19.5]. However, holding the oracle text choice fixed, the same norm-matched region-oracle improvement over norm-matched global-oracle is only+1.58 points [−1.92,5.08]. Region-oracle cosine is lower than global-oracle by−0.06353 [−0.11187,−0.01523]; uniform-oracle is lower by−0.05464 [−0.10160,−0.00666]. Thus the oracle-only positive frequency signal does not establish a patch-representation advantage.

Region selection reduces effective patches from49 to24.09 on average, but it does not materially beat uniform patches in the overall paired mechanism metrics. With generic text, region-minus-uniform cosine is+0.00474 [−0.02486,0.03413], beneficial-step gain+0.17pp [−3.00,3.50]. With oracle text the corresponding differences are−0.00889 [−0.03941,0.02089] and+0.83pp [−2.17,3.58].

AP remains heterogeneous. Region-oracle AP3 improves over corrupted input in five of six settings, including+0.353 on contrast-s2 and+0.039 on color-cast-s2, but gamma-s1 drops0.039. Against the stronger contemporaneous global-generic, region-oracle gains0.224 AP on gamma-s2,0.261 on contrast-s2 and0.418 on color-cast-s2, while losing0.574 on gamma-s1 and also losing on contrast-s1/color-cast-s1. Preserve these positive and negative cases; aggregate cosine alone is not a performance predictor.

The evidence remains weak/mixed even with oracle family information and norm matching. Under the R006 decision rule, recommend that the next explicitly assigned task compare a different self-supervised signal (for example frozen detector consistency or self-distillation), rather than meta-learning this CLIP directional loss. No T005, meta-training or spatially varying ISP is launched. This conclusion concerns the tested final-layer projected patch readout and fixed region weighting, not every possible CLIP feature/signal.

## Experiment receipt

- Run `20260912-051214-taisp-t004-coco200`; release `20260912-051210-taisp-t004-full`; experiment source `4817825`. Final report helper source `8cd0eb0` adds matched-text controls without changing experiment data/settings.
- Exit0 at2026-09-12T05:56:05+08:00; study elapsed2612.53855 seconds (43.54 minutes), excluding test/model startup.
- Same200 COCO-val IDs and six corruptions as T002/T003; six primary variants;7200 observations,36 groups×200,72 AP evaluations, plus7200 norm-matched one-step diagnostics. No sample exclusion or tuning.
- Same frozen CLIP ViT-B/32 and FasterRCNN ResNet50-FPN COCO_V1; same hard-clamped global8D ISP. phi0=0,lr0.1,K3,consistency/reg0,seed20260912. Annotated native-loss sampling seed20260912+image_id.
- A fresh g_det/loss is computed once per image/case and shared across all variants. No cached cross-run detector gradients are used. Generic and oracle global baselines are rerun in T004. Clean/corrupted AP uses identical T002 subset receipts.

```sh
export OMP_NUM_THREADS=1
export TAISP_SOURCE_REVISION=4817825
export PATH=/home/liujianhua/wjq/TAISP/.venv/bin:$PATH
TAISP_REAL_MODELS=1 python -m pytest -q
python -m taisp.analysis.run_t004 --data-root /home/liujianhua/wjq/TAISP/shared/coco200 --baseline /home/liujianhua/wjq/TAISP/runs/20260912-013248-taisp-t002-coco200-final/artifacts/study --output "$AUTODL_ARTIFACTS_DIR/study"
# Local report from project root:
python -m scripts.report_t004 research_log/remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study
```

## Exact representations and weighting

Global baseline uses the existing final normalized CLS embedding. Local features use the parity-correct differentiable shortest-side224 resize/floor center crop; take last_hidden_state[:,1:], apply frozen vision post_layernorm, then the same visual_projection768→512, then per-token L2 normalization. This yields7×7 patch features. No intermediate block was selected. Pinned embedding code confirms row-major flatten(2).transpose(1,2) after CLS. Patch-wise text alignment is diagnostic and was not directly guaranteed by global CLIP pretraining; patch tokens retain global self-attention context.

Uniform loss is the average negative projected enhanced-minus-original token displacement. Region loss is the weighted sum of the same per-token scores. The original image reference is detached and re-encoded under no_grad within each loss call, as in the global baseline. CLIP weights/buffers remain frozen; gradients reach phi through enhanced pixels.

Detector-region weights use exactly one frozen eval/no_grad detector inference on original corrupted input: score>=0.5,descending top20. Map original xyxy boxes with the exact integer resized width/height and floor crop offsets, clip to224 crop, intersect each32×32 patch. Raw weight is sum(score×intersection_area/1024), then normalize over49 patches. If no visible valid region exists, use uniform1/49. Boxes/scores/support/weights are fixed and detached for the episode; no annotation, corruption ID, detector gradient or adapted-image prediction determines them.

All six primary variants update all eight ISP coordinates with the same fixed lr; there are no oracle masks. The extra diagnostic rescales each initial semantic gradient to the contemporaneous global-generic gradient norm and takes one step. Labels do not determine scaling. No zero gradients occurred. Maximum norm-target absolute discrepancy is1.1921e-7 from floating arithmetic. Primary fixed-lr AP remains the deployment comparison.

Positive text bank (unchanged):
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

Normalize text embeddings, average each bank and normalize its mean. Generic direction remains normalize(t_pos−normalized aggregate negative mean); oracle direction uses the correct analysis-family negative bank. T004 reuses stored bank construction but never runs the T003 condition classifier prepare() method. No learned prompts.

## Overall primary and norm-matched results

Each variant contains1200 observations. Intervals below use2000 paired image-cluster percentile bootstrap draws,seed20260912; all six conditions follow each sampled image. Exploratory95% intervals, no multiplicity adjustment.

| Variant | Mean gradient norm | Mean / median cosine | Positive alignment | Primary loss benefit | Norm-matched benefit | Norm-matched Δbenefit pp vs global-generic [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| global_generic | 0.12575 | 0.04531 / 0.05628 | 53.83% | 47.17% | 47.17% | +0.000 [+0.000, +0.000] |
| global_oracle | 0.12457 | 0.07074 / 0.10302 | 55.58% | 50.83% | 49.92% | +2.750 [-0.250, +5.833] |
| patch_generic | 0.03470 | 0.01529 / 0.04155 | 51.92% | 48.58% | 48.33% | +1.167 [-2.333, +4.252] |
| patch_oracle | 0.03510 | 0.01610 / 0.02802 | 51.67% | 48.58% | 48.00% | +0.833 [-2.417, +4.085] |
| region_generic | 0.04038 | 0.02003 / 0.04505 | 52.83% | 48.75% | 48.00% | +0.833 [-2.417, +4.000] |
| region_oracle | 0.04179 | 0.00721 / 0.03496 | 51.58% | 49.42% | 51.50% | +4.333 [+1.167, +7.583] |

Mean raw local-gradient norms are0.0347–0.0418 compared with0.12575 global-generic. Norm matching is therefore essential to interpreting the apparent stability. Even matched region-oracle mean loss delta remains+0.000276; its paired mean-loss improvement versus global-generic−0.001484 has interval[−0.003298,+0.000568], crossing zero. Beneficial-step frequency and mean-loss change answer different questions.

### Visual effect with text choice held fixed

Both sides of the norm-matched comparison use each image/case global-generic target norm.

| Local variant | Global control | Δcosine [95% CI] | Δprimary benefit pp [95% CI] | Δmatched benefit pp [95% CI] |
|---|---|---:|---:|---:|
| patch_generic | global_generic | -0.030 [-0.085, +0.024] | +1.417 [-2.167, +4.919] | +1.167 [-2.333, +4.252] |
| patch_oracle | global_oracle | -0.055 [-0.102, -0.007] | -2.250 [-5.419, +1.000] | -1.917 [-5.250, +1.500] |
| region_generic | global_generic | -0.025 [-0.079, +0.030] | +1.583 [-1.917, +5.083] | +0.833 [-2.417, +4.000] |
| region_oracle | global_oracle | -0.064 [-0.112, -0.015] | -1.417 [-4.585, +1.917] | +1.583 [-1.917, +5.083] |

## All family/severity AP1 / AP3

COCO subset AP in0–100 points; clean AP38.012. No full-COCO benchmark claim, no per-image AP averaging or AP confidence intervals.

| Case | Corrupted | Global generic | Global oracle | Uniform generic | Uniform oracle | Region generic | Region oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| gamma_s1 | 38.145 | 38.351 / 38.680 | 38.568 / 38.437 | 38.177 / 38.253 | 38.152 / 38.481 | 38.211 / 38.112 | 38.184 / 38.106 |
| gamma_s2 | 36.446 | 36.573 / 36.388 | 36.619 / 37.113 | 36.488 / 36.486 | 36.490 / 36.558 | 36.492 / 36.604 | 36.492 / 36.612 |
| contrast_s1 | 36.175 | 36.369 / 36.508 | 36.301 / 36.553 | 36.154 / 36.136 | 36.212 / 36.191 | 36.172 / 36.138 | 36.315 / 36.434 |
| contrast_s2 | 31.175 | 31.115 / 31.267 | 30.888 / 31.016 | 31.389 / 31.267 | 31.320 / 31.277 | 31.388 / 31.369 | 31.363 / 31.528 |
| color_cast_s1 | 37.642 | 37.668 / 37.724 | 37.274 / 37.266 | 37.399 / 37.718 | 37.349 / 37.324 | 37.738 / 37.804 | 37.688 / 37.664 |
| color_cast_s2 | 35.908 | 35.841 / 35.528 | 35.801 / 35.784 | 35.828 / 35.813 | 35.839 / 35.795 | 35.891 / 35.784 | 35.963 / 35.947 |

All per-case gradient statistics, paired deltas versus global-generic, same-text global controls, region-versus-uniform intervals, norm-matched/Taylor analyses, AP differences, saturation and runtime are in [results.md](remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/results.md) and [analysis.json](remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/analysis.json). No negative family is omitted.

## Patch evidence and coordinate mechanism

Original detector predictions cover62.12% of the crop patch grid on average. Mean selected boxes7.16, visible boxes6.24. Exactly11/1200 image/cases use uniform fallback (0.92%). Region weights affect63.03% of patches including fallback and have effective count24.09 versus49 for uniform weighting. Weights are based on predictions, not GT object masks.

| Local variant | Effective patches | Object / background initial gradient norm | Object / background weighted loss3 | Partition-gradient residual max |
|---|---:|---:|---:|---:|
| patch_generic | 49.00 | 0.02134 / 0.01943 | -0.000158 / -0.000152 | 9.17e-05 |
| patch_oracle | 49.00 | 0.02183 / 0.01942 | -0.000162 / -0.000161 | 0.000117 |
| region_generic | 24.09 | 0.03958 / 0.00080 | -0.000407 / -0.000013 | 8.87e-08 |
| region_oracle | 24.09 | 0.04097 / 0.00083 | -0.000426 / -0.000014 | 6.8e-08 |

Region background weighted loss is zero by construction when object support exists. The small nonzero overall background contribution comes from the11 uniform fallback cases. Uniform-patch object and background gradient norms are similar despite different support sizes; selecting object tokens does not guarantee task-aligned gradients. The raw records retain object/background gradients, all49 weighted/unweighted patch scores at1/3 steps, boxes, scores, weights and support.

Gradient-energy percentages below normalize squared coordinates per observation before averaging. RGB gains still dominate local objectives (approximately73–74% combined), so token readout has not removed coordinate cross-talk.

| Variant | Gamma | Red | Green | Blue | Contrast | Brightness | Tone | Sharpen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global_generic | 3.41 | 29.00 | 26.18 | 12.30 | 19.14 | 8.55 | 0.82 | 0.60 |
| global_oracle | 5.20 | 28.21 | 27.61 | 12.40 | 16.19 | 8.49 | 1.27 | 0.63 |
| patch_generic | 4.53 | 32.29 | 27.66 | 13.21 | 13.47 | 7.69 | 0.94 | 0.21 |
| patch_oracle | 4.20 | 32.35 | 28.40 | 12.86 | 13.32 | 7.81 | 0.87 | 0.19 |
| region_generic | 5.46 | 31.89 | 27.82 | 14.07 | 12.43 | 6.93 | 1.16 | 0.22 |
| region_oracle | 4.96 | 31.34 | 28.76 | 13.43 | 13.10 | 7.12 | 1.05 | 0.24 |

## Semantic progress, saturation and costs

Local own semantic losses decrease after3 steps in96.67–98.08% of observations, yet primary detector-loss benefit remains48.58–49.42%. Mean saturation3 drops from3.47% global-generic to2.75% uniform-generic and2.37% region-generic. These reductions accompany substantially smaller raw gradients and do not establish improved semantic direction. Both contrast settings remain at essentially zero saturation.

| Variant | Own semantic loss decreases at3 | Mean saturation3 | Adapt seconds3 | Deployment seconds3 | Adaptation peak MiB |
|---|---:|---:|---:|---:|---:|
| global_generic | 93.33% | 3.470% | 0.1106 | 0.1106 | 841.8 |
| global_oracle | 94.75% | 3.250% | 0.1122 | 0.1122 | 841.9 |
| patch_generic | 96.67% | 2.753% | 0.1143 | 0.1143 | 842.0 |
| patch_oracle | 98.08% | 2.386% | 0.1096 | 0.1096 | 842.0 |
| region_generic | 97.08% | 2.372% | 0.1088 | 0.1386 | 841.9 |
| region_oracle | 97.58% | 2.335% | 0.1091 | 0.1390 | 842.0 |

Region deployment time includes the original detector inference/map, averaging29.86ms extra setup; that inference is diagnostic-only and excluded for the global/uniform paths. Adaptation timing includes its diagnostics and GPU synchronization but excludes annotated-loss/AP/norm-match/partition computation. Peak memory is adaptation-phase allocated memory only; detector-setup peak was not separately recorded. Variant execution order was fixed, so millisecond differences are descriptive and not a hardware-optimized speed benchmark.

## Validation and reproducibility

- Baseline:9 local tests passed10.77s;34 A6000 tests passed7.69s. New feature/preprocessing/adapt gate15 passed5.23s.
- Smoke run20260912-050701-taisp-t004-study-smoke:42 real-model tests passed8.52s;2 images/72 observations/all AP evaluations,27.53785s,exit0. No smoke-driven setting changes.
- Full run starts with42 real-model/regression tests passed8.17s. Tests cover odd/rectangular/square crop geometry, direct post-LN/projection equivalence, frozen model state, finite phi gradients, repeated fresh episodes, region overlap/score/top-k/fallback, original-only detector invocation, norm matching and partition gradients.
- Final analysis-related tests3 passed6.85s. Complete receipt audit passed:7200 rows,36 groups×200,200 IDs, all-zero fresh phi0, exact shared g_det/loss across variants, patch weight normalization and norm-match target checks. Figure visually inspected.
- No T004 implementation or experiment failures. Two documentation trailing-CR lines were normalized after git diff --check identified them; no scientific change.
- Recomputed uniform patch partition gradients differ from the saved total by at most1.1682e-4; region partition residuals are below9e-8. These discrepancies are retained rather than assuming bitwise CUDA backward reproducibility. CPU partition sum tests pass. All scientific comparisons use fresh within-run detector gradients.
- One fixed200-image subset and one seed; intervals reflect image resampling, not run-to-run numerical uncertainty or multiplicity control. Oracle directions are analysis-only. No meta-training, learned prompts, intermediate-layer search, predictor/source training, smooth clamp, detector update or ViT³ mechanism was introduced.

Remote environment: Python3.12.12,torch2.4.0,torchvision0.19.0+cu121,Transformers4.44.2,pycocotools2.0.10,NumPy1.26.4,Pillow12.3.0,CUDA12.1,RTX A6000. Local reports use Python3.12.7,NumPy1.26.4,SciPy1.13.1,Matplotlib3.9.2.

CLIP revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; SHA256 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`. Detector SHA256 `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`. Annotation SHA256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`. The subset manifest retains exact image IDs/hashes.

## Delivery and next action

- [Full paired tables](remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/results.md), [all analysis](remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/analysis.json), [AP figure](remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/ap_change.pdf), [raw7200 observations](remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/samples.jsonl), [receipt audit](remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/receipt_audit.json), [environment](remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/environment.json), [completion](remote_runs/20260912-051214-taisp-t004-coco200/artifacts/study/completion.json).
- Raw samples.jsonl is96,709,726 bytes, below GitHub single-file limit; retained unchanged without compression. SHA25681a75284aed074427f82e6b743d62192ef5cf93882f4f7ab2f31ad34d92c56ca. All72 prediction files and raw logs are retained.
- Remote artifacts: /home/liujianhua/wjq/TAISP/runs/20260912-051214-taisp-t004-coco200/artifacts/study; reports/recovery notes mirrored under project research_log.
- Source: shared CLIP geometry/patch extension, spatial loss, region mapper, spatial diagnostics/runner/config/tests and report script. Existing ISP, adaptation, detector, COCO and oracle/evaluator owners were reused.
- T004 NEEDS_REVIEW. Await research lead acceptance/next explicit task. Heartbeat executes newly assigned work directly; do not rerun this completed study or start meta-training automatically.
