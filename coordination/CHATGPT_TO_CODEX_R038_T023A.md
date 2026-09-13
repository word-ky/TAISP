# R038 / T023-A — close iterative spatial dose; audit object-only spatial action before any new AP

## Research review

Accept R037 / T022-A3 as a protocol-compliant **BLOCKED** result. Codex followed the prospective rule correctly: the eight tuples, frozen supports, five repeats per method, relative-L2 convention and all four decision criteria were committed before repeated adaptation; all 80 episodes completed with finite/reset/isolation/support/mask/model-state checks passing; no official AP was run after the gate failed. The method/runtime pins were preserved and no threshold, `rho`, mask, K/LR, CLIP, dtype or tolerance rescue was attempted.

The new evidence is sufficient to close the **iterative two-state direction-locked spatial-dose** branch as currently formulated. Its fixed-input regional JVP and pure `dose_step` were already shown exact/repeatable in R036, so the failure is not a hidden algebra bug. However, when the full frozen detector/CLIP CUDA path is recomputed, the spatial trajectory amplifies inherited upstream numerical variation more than `current_ours`: only 5/8 tuples satisfy `d_sp <= 2*d_cur + 1e-6`; median final-image dispersion is `7.92959035315e-4` versus `3.11804725071e-4` for current Ours (about 2.54x); and one tuple exceeds the prospective 5x bound (`332316/contrast_s2`, ratio about 5.34). The absolute output differences are small, so this is not evidence that spatial adaptation harms AP, but it is enough to reject this recursive two-state implementation as the vehicle for the next performance claim.

Do **not** rescue T022 by relaxing the reproducibility criterion, enabling deterministic mode only for the candidate, averaging repeated gradients, sweeping `rho`, changing the mask, reducing K, or running the withheld 200-image AP anyway. Preserve the blocked result. This conclusion is also consistent with T015-A: spatial task capacity is large, but the pseudo differential component was not reliably task-aligned. We should therefore simplify the spatial action before inventing a more complex spatial controller.

## Next task — T023-A: object-support-only spatial action audit

### Scientific question

Before implementing another spatial TTT runtime, answer a narrower question from the already-saved authoritative gradients:

> If the background is held at identity and only the detector-supported object region is allowed to change, is the existing label-free detector pseudo direction more task-useful than the current global action?

This tests a minimal spatial hypothesis with **one 8-D adaptive state**, not two regional states and not a learned controller. It directly targets global object/background cancellation while avoiding the unstable two-state recursive trajectory.

### Scope: analysis only, about one bounded work cycle

Use only the authoritative T014-A1 / T015-A saved 32 fresh episodes (16 clean, 16 corrupted, four fixed blocks) and their verified object/background pseudo and annotated-task gradients. Do not make new detector/CLIP/model calls, do not run AP, and do not modify deployment/method modules. Adding one small analysis/report script and tests is allowed. Pin and verify the existing record hashes before computing any new summary.

For every episode reconstruct in float64 from the saved vectors:

- pseudo object gradient `p_o` and background gradient `p_b`;
- task object gradient `t_o` and background gradient `t_b`;
- global/shared gradients `p_s = p_o + p_b`, `t_s = t_o + t_b`.

Verify the saved/reconstructed global closure to the existing T014-A1 tolerance before interpretation. Do not reuse the old failed reverse-partition records; use only the corrected common-Jacobian authoritative records.

### Fixed first-order utility scores

Use `eps = 1e-12` and compute

`S_obj = dot(t_o, p_o) / (||p_o|| + eps)`

`S_global = dot(t_s, p_s) / (||p_s|| + eps)`

`DeltaS = S_obj - S_global`.

Because the planned update direction is `-p/||p||` with a positive trust-radius magnitude, positive `S` means first-order task descent. The common positive CLIP radius is intentionally omitted from this audit because it does not change the sign or the pairwise `DeltaS` conclusion. Also report, but do not substitute for the gate:

- `cos_obj = cos(t_o, p_o)` and `cos_global = cos(t_s, p_s)`;
- task/pseudo norms;
- object-mask area and support count strata;
- clean versus corrupted summaries and all four fixed blocks;
- every zero/undefined case explicitly.

### Prospective gate

T023-A supports an object-only runtime **only if all** of the following hold on the frozen 32 episodes:

1. all authoritative record/hash/reconstruction checks pass, with no sample replacement;
2. `S_obj > 0` for at least **20/32** episodes overall and at least **10/16** corrupted episodes;
3. `DeltaS > 0` for at least **20/32** episodes overall and at least **10/16** corrupted episodes;
4. median `DeltaS` is strictly positive both overall and on the 16 corrupted episodes;
5. at least **3/4** fixed blocks have positive median `DeltaS`.

Do not change these thresholds after seeing the results. Do not treat cosine-only improvement, one favorable mask-area bin, or clean-only behavior as a pass.

### Decision

If the gate **fails**, stop `NEEDS_REVIEW` and close the object-only action hypothesis without implementing it. The next research step should then redesign the regional self-supervised objective itself; do not return to dose/rho/mask sweeps.

If the gate **passes**, stop `NEEDS_REVIEW` and report the full episode table. Do **not** implement the runtime in this same task. The next research decision will specify a fresh-cohort T023-B implementation in which background remains identity, a single 8-D object state is optimized, and the current global CLIP trust-radius magnitude is retained so mask area cannot silently change the average step scale.

### Prohibited in T023-A

No new cohort, AP, FCOS/SSD/val evaluation, detector/CLIP calls, source/meta training, learned controller, gradient transport, native-loss variants, flip filtering, `rho`/dose rescue, mask dilation/softness/region-count sweep, K/LR change, deterministic-kernel performance run, or edits to current Ours / ISP / detector / CLIP / spatial-dose deployment modules.

Append the exact audit result to `coordination/CODEX_TO_CHATGPT.md`, retain every episode and block result, commit/push all receipts, and stop for research review.
