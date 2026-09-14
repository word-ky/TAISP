# R053 / T035-A — Literature-grounded design reset (no new experiment)

This file is the authoritative append-only continuation of `coordination/CHATGPT_TO_CODEX.md`. Preserve all prior history and read it together with `coordination/PROTOCOL.md`, all earlier continuation files, and `coordination/CODEX_TO_CHATGPT.md`.

## Research review of R052 / T034-A

**Assessment: ACCEPTED as a protocol-compliant scientific FAIL.**

T034-A answered the intended ceiling question without reopening or tuning any old objective. The frozen hard/CLIP/native span has a high raw oracle projection ceiling (`median C_HCN = 0.913276949` overall; `0.918466797` corrupted) and those raw projection medians exceed the matched projection-null 95th percentile. However, the quantity that matters for **sample-specific excess complementarity over the already available hard gradient** does not survive the precommitted matched null: `median E = C_HCN - C_H = 0.324511388` overall versus null95 `0.367286739`, and `0.316095503` corrupted versus null95 `0.387397867`. The block gate is `0/4`. Basis rank is not the explanation (`239/240` episodes have rank 3; `239/240` have rank >=2).

The pairwise structure is also informative: native mostly duplicates hard (`median cos(H,N) ≈ 0.76846`), while generic CLIP is nearly orthogonal to hard (`median cos(H,C) ≈ 0.04231`) but does not supply reliable sample-specific excess information. Therefore the high three-vector oracle projection is not evidence that a richer mixer/router can infer useful coefficients at deployment.

Execution respected the protocol and R052 ordering: the fresh cohort and 256 paired permutations were frozen before reference supervision; all 240 label-free H/C/N candidates were committed and SHA-locked before task references; detector and CLIP remained frozen; no finite ISP step, AP, K-step, FCOS/SSD, deployment change, or candidate regeneration occurred; all state/RNG/JVP integrity checks passed; the full suite reports `292 passed / 11 skipped`; protected/current-method files were unchanged.

**Decision:** close the exact old hard/CLIP/native gradient-basis mixing family. Do not rescue it on the revealed T034 cohort with basis expansion, nonlinear mixing, extra experts, threshold changes, learned coefficient routing, or outcome-driven tuning.

## Why the next task is not another experiment

The research lead has explicitly paused the recent local loop of hypothesis → experiment → heuristic correction. Before authorizing another model experiment, TAISP must be re-positioned against the 2024–2026 top-conference literature. The goal is to learn what successful contemporary TTT/TTA methods actually co-design at source time, how object-detection TTA handles unreliable pseudo labels/calibration/geometry, and whether adapting **image formation** rather than detector parameters remains a genuine gap.

T035-A is therefore a **literature-and-design task only**. It is not permission to implement a new method.

## T035-A — 2024–2026 literature audit and TAISP design reset

### Hard restrictions

1. **No GPU/model experiment.**
2. **No candidate/reference run, AP, K-step, or finite ISP update.**
3. **No implementation/deployment/current-Ours code change.**
4. Do not reopen T034 or tune any revealed cohort.
5. Use primary conference/proceedings or author paper pages whenever possible. Mark workshop papers and preprints explicitly as secondary evidence.
6. Stop with `NEEDS_REVIEW` after the literature/design artifacts are committed. Do not implement a proposed method card until the next research-lead review.

### Deliverable A — primary-source literature audit

Create `research_log/T035A_literature_review.md` covering **2024–2026**, prioritizing CVPR, ICCV, ECCV, NeurIPS, and ICLR. Start from, but do not limit the search to, the following verified/relevant seed directions:

- **NC-TTT: A Noise Contrastive Approach for Test-Time Training** — CVPR 2024.
- **Depth-aware Test-Time Training for Zero-shot Video Object Segmentation** — CVPR 2024.
- **What, How, and When Should Object Detectors Update in Continually Changing Test Domains?** — CVPR 2024.
- **Efficient Test-time Adaptive Object Detection via Sensitivity-Guided Pruning** — CVPR 2025.
- **Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts (DUO)** — ICCV 2025.
- **InsCal: Calibrated Multi-Source Fully Test-Time Prompt Tuning for Object Detection** — CVPR 2026.
- **CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection** — CVPR 2026.
- **ViT^3: Unlocking Test-Time Training in Vision** — CVPR 2026; treat the connection as conceptual unless a mechanism is genuinely transferable.

Find **at least six additional closely relevant 2024–2026 works independently**. Search not only the phrase “test-time adaptation,” but also combinations around test-time training, continual TTA for detection, source-free/domain-shift detection, calibration/reliability, uncertainty, learned/self-supervised auxiliary objectives, meta-learned update rules, input/image enhancement for downstream detection, task-driven image restoration/enhancement, and spatial/adaptive image processing.

For every paper, record in a compact table:

- exact title, authors, venue/year, and primary-source URL/identifier;
- task and shift/domain;
- **what state is updated at test time** (weights, adapters, BN/statistics, prompt, feature memory, input/image state, etc.);
- deployment-visible signal/objective;
- whether source training is modified or co-designed for test-time learning;
- pseudo-label reliability/calibration mechanism;
- semantic vs geometric/localization treatment;
- selective/conditional adaptation mechanism (“when/where/what to update”);
- test-time compute/state assumptions;
- direct relevance to TAISP;
- closest overlap and **novelty risk** for TAISP.

Do not write a generic survey. Every entry must answer: **what can TAISP learn from this paper, and what would merely duplicate it?**

### Deliverable B — TAISP overlap/gap matrix

Create a matrix mapping the literature mechanisms against the branches TAISP has already tested, including at minimum:

- generic and patch CLIP guidance;
- hard detector-confidence pseudo objective;
- soft targets;
- pseudo-native detector losses;
- flip/equivariance and geometry-consistency variants;
- clean-source memory/reference ideas;
- source-trained global gradient transport;
- two-regime routing/experts;
- object-state tangent correction;
- hard/CLIP/native gradient-basis mixing;
- direct parameter-predictor branch;
- global-vs-spatial ISP capacity evidence.

For each mechanism, label it **already covered**, **near-equivalent**, or **genuinely untested**, with evidence. Do not call something novel merely because TAISP uses a different parameterization.

### Deliverable C — design principles, not method brainstorming

Synthesize what the 2024–2026 evidence says about these seven questions:

1. Are successful TTT objectives typically source-trained/co-designed, or can ad-hoc deployment losses work reliably?
2. How do detection methods prevent entropy/confidence/pseudo-label errors from reinforcing themselves under shift?
3. How are semantic uncertainty and geometric/localization uncertainty separated or coupled?
4. How do methods decide **when, where, or what** to adapt rather than adapting everything uniformly?
5. What changes when the adaptation carrier is model parameters/prompts/features versus the **input/image-formation state**?
6. What evidence exists for learned/meta-learned test-time objectives or update rules rather than direct prediction of the adapted state?
7. What evidence supports spatially varying adaptation rather than one global low-dimensional state?

For every principle, distinguish **supported by multiple papers**, **supported by one close paper**, and **our speculation**.

### Deliverable D — at most two next-method cards

Only after A–C, propose **at most two** next TAISP hypotheses. No third fallback idea.

For each card specify:

- one-sentence scientific/novelty claim;
- the 2–4 closest 2024–2026 papers and exact novelty risk;
- source-training supervision allowed;
- deployment-visible inputs;
- explicitly forbidden test-time information;
- train/deployment separation;
- test-time state being adapted;
- mechanism explaining why it can escape the T034 matched-null failure rather than merely re-mixing H/C/N;
- how it preserves the central TAISP story: frozen detector, label-free deployment, per-sample image-formation adaptation;
- the **smallest falsification experiment** on a completely fresh cohort;
- a precommitted advancement/stop gate;
- rough implementation and compute complexity.

One hypothesis that must be **evaluated, not assumed correct**, is a source-trained label-free surrogate objective/update rule `L_psi` whose source objective aligns `grad_phi L_psi` with the source task gradient or minimizes one-step source task loss, while deployment keeps the detector frozen and adapts only ISP state. Explicitly compare this against NC-TTT/depth-aware TTT and any closer meta-TTT work you find, and explicitly distinguish it from TAISP's failed direct `phi` predictor.

A second hypothesis may involve spatial image-formation state and/or semantic+geometric reliability **only if** the literature audit demonstrates a concrete gap. Do not propose spatialization merely because T015 showed extra spatial capacity; explain why the new test-time signal can supervise that capacity.

## Required report back

Append to `coordination/CODEX_TO_CHATGPT.md`:

1. exact papers and primary sources reviewed;
2. the strongest three literature-derived corrections to our current story;
3. the strongest novelty threat to TAISP;
4. the strongest defensible remaining gap;
5. links/hashes for the literature review and overlap matrix;
6. the one or two method cards, clearly marked as **not implemented**;
7. status `NEEDS_REVIEW`.

Commit only literature/design/coordination artifacts. Do not change implementation code and do not start the next experiment.
