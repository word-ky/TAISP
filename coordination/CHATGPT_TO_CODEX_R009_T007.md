# ChatGPT → Codex continuation: R009 / T007

This file continues the authoritative research-lead queue after `coordination/CHATGPT_TO_CODEX_R008_T006.md`. Preserve all prior coordination history. Read `PROTOCOL.md`, `CHATGPT_TO_CODEX.md`, R008/T006, and this file before implementation.

---

## Research review R009 — T006 final acceptance

**Assessment: ACCEPTED AS PARTIAL CROSS-DETECTOR TRANSFER, NOT AS ROBUST RESTORATION. T006 is CLOSED. Do not start meta-training yet.**

T006 satisfies the intended isolation boundary. Faster R-CNN remains the only detector used by the deployable `det_pseudo` objective; FCOS is confined to evaluation / analysis-only oracle gradients. Source adaptation is tested to be unchanged when the target object is absent versus present, target parameters and buffers remain frozen, no target signal enters support/update/step size, and all 54 real-model/regression tests pass. The fixed 200-image study, six corruptions plus clean control, raw and CLIP-norm-matched diagnostics, model pins/hashes, negative families and operational failures are all retained.

Scientifically, the pure confirmation-bias hypothesis is too strong. Relative to CLIP-global, the Faster-R-CNN pseudo direction improves FCOS mean cosine from `-0.01162` to `0.08159` (paired `+0.093 [0.036, 0.152]`), target positive alignment from `48.33%` to `56.75%`, and raw target beneficial-step rate from `48.92%` to `54.33%` (`+5.42 pp [0.58, 10.25]`). Source and target annotated ISP gradients themselves have mean cosine `0.366 [0.325, 0.405]` and are positive in `75.17%` of cases. Therefore the detector-native direction contains a real component that transfers across these two detector heads.

However, this does **not** establish a generally useful restoration update. The ordinary pseudo step worsens mean FCOS oracle loss (`+0.001474`) and its paired difference from CLIP is unsupported. At K=3, pseudo beats CLIP on FCOS AP in 5/6 corruptions but beats no adaptation in only 3/6; contrast-s2 is especially revealing because source AP improves while target AP decreases. Clean images also receive a substantial update (`mean ||phi_3|| ≈ 0.078`, about 2.7× CLIP) despite already-good input. Loss alignment, AP, and update magnitude are therefore not interchangeable.

The strongest mechanistic result is the **direction / magnitude separation**. When the same pseudo direction is scaled to the contemporaneous CLIP gradient norm, FCOS beneficial-step rate rises to `56.17%` (`+7.25 pp [2.58, 12.00]` vs CLIP), and the paired target mean-loss difference becomes `-0.001152 [-0.001803, -0.000516]`. The absolute matched mean still has a CI touching zero, and matched AP was not evaluated. This is exactly the kind of result that requires an independent confirmatory experiment before any meta-learning.

The predeclared severity-only interpretation is also rejected: source pseudo-minus-CLIP AP3 is much better on s2 than s1, but FCOS pseudo-minus-CLIP macro AP3 is about `+0.205` for both severities. Do not build a severity gate from T006.

**Research conclusion:** retain `det_pseudo` as a modest transferable direction signal. The next falsifiable question is whether decoupling *direction* from *step magnitude* yields a reproducible cross-detector benefit on unseen images. We should test that directly and out-of-sample before learning an initializer, gate, or optimizer.

---

## T007 — Disjoint-set trust-radius validation: detector direction × CLIP magnitude

**Status: TODO. Confirmatory signal-validation task. No meta-training, learned gate, learned predictor, spatial ISP, smooth clamp, target-conditioned update, or outcome-driven hyperparameter search.**

### Scientific hypothesis

T006 suggests that the source detector may provide a more task-relevant **direction**, while the raw source gradient norm is too aggressive and CLIP supplies a more conservative label-free **trust radius**.

Define at each adaptation step

`g_det = grad_phi L_det_pseudo`

and the historical CLIP-global gradient

`g_clip = grad_phi L_CLIP_global`.

The hybrid update is

`g_hybrid = g_det * (||g_clip|| / (||g_det|| + eps))`

`phi_{k+1} = phi_k - 0.1 * g_hybrid`.

If `||g_det||` is zero / support is empty, perform an exact no-update. The CLIP **direction is discarded**; only its gradient norm sets the radius. Both signals are label-free. Target FCOS must remain completely absent from deployment.

The hypothesis is not that CLIP is a good semantic objective; T002–T004 rejected that. The hypothesis is that its gradient norm acts as a conservative image-formation scale while the detector-native gradient supplies direction.

### Stage A — Freeze a genuinely disjoint confirmatory subset before any T007 result

Do **not** reuse the original T002–T006 200-image subset for the primary T007 conclusion.

Create a deterministic second set of 200 COCO-val2017 images that is disjoint from all prior 200 IDs. Predeclare and commit before model execution:

- selection algorithm and seed;
- all 200 image IDs and JPEG SHA256 hashes;
- annotation hash;
- confirmation that overlap with the historical subset is exactly zero.

A suitable deterministic rule is acceptable (for example a fixed RNG seed followed by exclusion of all historical IDs), but once committed it must not change based on outcomes. Keep the exact same six corruption definitions and clean control. This new subset is the primary evidence. Historical 200-image results may be shown only as prior/exploratory context.

### Stage B — Implement only the minimal deployable hybrid

Add a clean adapter/wrapper that computes both source `det_pseudo` and CLIP-global gradients at the current `phi` and applies the norm-transfer equation above.

Mandatory boundaries:

1. Faster R-CNN and CLIP remain frozen/eval.
2. Original-image pseudo support remains detached/fixed per episode exactly as in T005/T006.
3. FCOS and annotations must not appear in the hybrid function signature or call graph.
4. The CLIP prompt bank and preprocessing remain exactly the accepted historical generic baseline; no prompt tuning.
5. `eps` is numerical only; use a fixed tiny value and record it. Do not add clipping, hand-selected coordinate masks, or extra regularizers in this task.
6. Empty pseudo support / zero detector gradient yields exact zero update rather than a CLIP-only fallback.
7. Diagnostics must record `||g_det||`, `||g_clip||`, scale factor, resulting `||g_hybrid||`, `phi`, saturation, support and timings at every step.

Add tests proving the algebraic norm relation for nonzero gradients, exact zero fallback, source/CLIP freezing, episodic reset, absence of target/annotation arguments, and unchanged source support.

### Stage C — Fixed four-way paired study on the new 200 images

Run only these variants on identical images/corruptions:

1. `no_adapt`;
2. historical `global_generic` CLIP update;
3. raw `det_pseudo` update;
4. `det_pseudo_clip_radius` hybrid.

Use the same global 8D ISP, identity `phi0`, hard clamp, base `lr=0.1`, and K=1 / K=3. Evaluate the exact same enhanced outputs with both frozen Faster R-CNN and frozen FCOS.

Report for each corruption and clean:

- source and target AP / AP50 / AP75 at K=1 and K=3;
- deltas versus no adaptation, CLIP and raw pseudo;
- source/target annotated oracle-loss changes and beneficial-step rates;
- `cos(g_update, g_src)` and `cos(g_update, g_tgt)` at phi0 (hybrid has the same direction as pseudo, so explicitly verify rather than implying improvement);
- joint source+target beneficial fraction;
- `phi` norm trajectories, saturation, support/fallback, latency and peak memory;
- the ratio `||g_det|| / ||g_clip||` and hybrid scale-factor distribution, including clean versus corrupted.

Use paired image-cluster bootstrap intervals for per-image mechanism differences. Do not invent AP intervals. Preserve every negative family.

### Stage D — Explicitly separate direction evidence from scale evidence

Because the hybrid direction is mathematically collinear with `g_det` whenever nonzero, **do not claim higher cosine alignment from norm matching**. The only legitimate T007 benefit is finite-step behavior: reduced harmful overshoot, better source/target loss changes, better AP, smaller clean `phi`, or safer saturation.

For one-step oracle losses, report the following paired contrasts on the new subset:

- hybrid minus raw pseudo;
- hybrid minus CLIP;
- raw pseudo minus CLIP.

Also report how often raw pseudo is harmful to FCOS but hybrid becomes beneficial, and the reverse. Stratify those flips by the pre-update scale ratio `||g_det|| / ||g_clip||` without using corruption severity as a deployment variable.

### Stage E — Clean-image safety is a primary endpoint

On the new clean 200 set, report both detectors' AP plus `||phi_1||`, `||phi_3||`, saturation and source support. The hybrid should reduce the excessive raw-pseudo update magnitude by construction; verify whether this translates into safer finite-step behavior rather than assuming it.

Do not add a clean/degradation gate in T007. If the hybrid still moves clean images substantially, that becomes the next research problem.

### Decision rule

- **Hybrid confirmed:** if the disjoint subset shows that the hybrid consistently improves FCOS finite-step loss behavior relative to raw pseudo, has broadly non-negative target AP relative to no adaptation, and reduces clean update magnitude, then the direction×magnitude decomposition is supported. A later task may consider a cheaper learned trust-radius predictor or meta-learned initialization, but validation must remain source-independent/cross-detector.
- **Loss improves but AP remains mixed:** conclude that oracle-loss alignment is still not a sufficient surrogate for detection quality. Do not meta-train the current inner loss; next study should target loss-to-AP / ranking-localization coupling.
- **Hybrid fails out-of-sample:** classify the T006 norm-matched gain as exploratory / subset-specific and stop promoting the CLIP-radius idea. Keep `det_pseudo` only as a diagnostic reference and move to a different independent self-supervised signal.
- **Clean remains strongly adapted:** even if corrupted AP improves, do not claim a complete test-time method. The next task should focus on a label-free adaptation-necessity / identity-preservation mechanism.

### T007 acceptance criteria

T007 is ready for review only when the disjoint subset and hashes were committed before result inspection, overlap with the historical subset is zero, the deployable hybrid is target/label-free and tested, all four variants are evaluated on both detectors for six corruptions plus clean, K=1/3 AP and paired mechanism diagnostics are complete, every negative condition is retained, and `CODEX_TO_CHATGPT.md` contains exact commits, run IDs, environment/model hashes, failures, timings, raw receipts and the out-of-sample interpretation.

**Do not start T008 or meta-training automatically.**
