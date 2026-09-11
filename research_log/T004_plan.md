# T004 pre-run contract

2026-09-12 Asia/Shanghai. R006 cfdec6e accepts/closes T003 and assigns T004.
Representation diagnosis only: eight-dimensional global ISP, frozen models,
hard clamp, no training, learned prompts, spatial ISP, condition classifier tuning
or intermediate-layer search. Follow existing baseline-reuse and AutoDL workflow.

## Definitions fixed before GPU experiments

Same final200 IDs, six corruptions, CLIP/model revision/detector, phi0=0, lr0.1,
K3, consistency/reg0, seed20260912. Fresh g_det once per image/case shared by all
variants. Six primary variants, in order: global_generic, global_oracle,
patch_generic, patch_oracle, region_generic, region_oracle. Generic banks unchanged;
oracle selects only the correct family's T003 negative bank, analysis-only.
No coordinate masks; every primary variant updates all eight raw coordinates.

Spatial encoder: unchanged CLIPTensorPreprocess,224x224. Pinned HF vision_model
last_hidden_state excludes CLS, giving7x7 tokens. Apply the frozen vision_model
post_layernorm to each patch, then model.visual_projection (768->512), then
L2 normalize each token. No intermediate block. Patch text alignment is a
diagnostic use, not guaranteed by CLIP's global contrastive pretraining.

Patch objective: per-token negative displacement projection on the same normalized
text direction; stop-gradient original-image reference. Uniform average over49
patches or normalized fixed nonnegative region weights. Reference remains a frozen
forward inside each loss call as in global baseline (no asymmetric reference cache).

Region weights: frozen detector predicts once on the original corrupted image,
eval/no_grad; score>=0.5, descending top20. Detector boxes in original image xyxy
coordinates are mapped with the exact integer resized height/width then floor
center crop offsets. Intersect mapped boxes with224 crop and each32x32 patch.
Unnormalized patch weight=sum_box(score * intersection_area/1024). Normalize by
sum over49 patches. No valid visible positive-area region => uniform1/49 fallback
(explicitly requested by T004). Save selected/visible boxes, scores, coverage,
normalized weights, fallback flag, support fraction and effective N=1/sum(w^2).
No annotation/corruption ID/gradient/adapted prediction enters this path.

Object/background diagnostic: object patches are those with positive mapped region
overlap, background is the complement, fixed from original inference. For every
patch variant save initial g_object and g_background obtained by differentiating
the corresponding weighted loss contributions; report their sum against g_sem.
Also save per-patch loss contributions after1/3 steps and partition sums/unweighted
partition means. For region variants background contribution is zero by design;
fallback is explicitly identified (no object support). Do not infer causality from
object selection alone. Uniform variants reuse the same support for analysis only.

Direction-only diagnostic: one step using g_scaled=g_variant*(norm(g_global_generic)
/norm(g_variant)) at phi0; no annotation determines scaling. Global_generic is
unchanged; also report scaled global_oracle for completeness. Primary deployable
updates remain fixed-lr. If an actual zero gradient occurs, retain a zero diagnostic
step and report it (do not fabricate a direction). Evaluate oracle loss, saturation,
phi and semantic loss for this extra step; no extra AP series is required for it.

Report every case/variant: raw coordinate gradients/energy, cosine/positive rate,
measured loss benefit/delta, norm-matched results, AP1/AP3, own/common semantic loss,
saturation, physical/raw trajectories, timing/memory, patch contribution/support.
Paired deltas vs contemporaneous global_generic and region-vs-uniform with matched
text choice; 2000 image-cluster bootstrap draws seed20260912 where applicable.
AP paired fixed-subset differences without per-image-AP averaging. No tuning.
Region deployment latency includes the one original detector inference/map; other
variants exclude this diagnostic-only inference. Report costs separately.

## Reuse and bounded increments

1. Baseline: existing adaptation/regression and real-model tests, record results.
2. Extend FrozenCLIPEncoder with last-layer spatial features, shared preprocessing
   geometry helper; weighted patch loss. Hand/real feature shape, projection,
   frozen-state/phi-gradient/episode and odd/rectangular tests before continuing.
3. Region mapper uses existing detector inference; known boxes/score filtering,
   crop clipping, empty-region fallback, frozen original input/weights tests.
4. New analysis driver reuses existing data/corruptions/oracle/evaluator/metadata/
   physical-vector/summary helpers and adapt. No changes to evaluator or ISP.
   Two-image smoke verifies all six variants and scaled-step/partition diagnostics.
5. Fresh full suite then fixed200 run. Persist all failures/results; postprocess
   and append complete report. No duplicate run or automatic T005/meta-training.
