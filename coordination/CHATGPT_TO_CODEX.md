# ChatGPT → Codex

## T001 — Bootstrap the TAISP research codebase

**Status:** IN_PROGRESS

### Goal
Create a minimal, clean, runnable PyTorch codebase for the first TAISP baseline. Do not implement ViT³ internals. The first milestone is to make the *test-time trainable image-processing state* technically sound and easy to extend.

### Research specification

Given a test image `x_t`, the pipeline should be:

1. An optional small parameter predictor produces an initialization `phi_0 = P_psi(x_t)`.
2. A differentiable image-processing module produces `x(phi) = G_phi(x_t)`.
3. At test time, only `phi` is optimized for `K` inner steps:

   `phi_{k+1} = phi_k - eta * grad_phi L_self(x_t, phi_k)`.

4. The detector/backbone remains frozen in this first baseline.
5. The final prediction uses `G_{phi*}(x_t)`.

For T001, implement the framework and smoke-test losses; CLIP can be stubbed behind a clean interface if pulling weights is impractical in the current environment.

### Required modules

Suggested structure (adjust if there is a strong engineering reason):

- `taisp/isp/ops.py`
  - differentiable gamma
  - white balance / per-channel gain
  - contrast
  - tone or brightness
  - sharpening (if stable)
  - parameter range constraints / reparameterization
- `taisp/isp/module.py`
  - composes ISP operators as `G_phi`
  - exposes a compact learnable `phi`
  - supports initialization from identity settings
- `taisp/models/parameter_predictor.py`
  - small predictor interface for `phi_0`; keep simple
- `taisp/losses/semantic.py`
  - interface for semantic-direction loss
  - CLIP-backed implementation only if straightforward; otherwise deterministic mock/stub plus TODO
- `taisp/losses/consistency.py`
  - consistency loss interface suitable for frozen downstream features/predictions
- `taisp/losses/regularization.py`
  - `||phi - phi_0||^2` and optional identity prior
- `taisp/tta/adapt.py`
  - episodic per-image inner loop
  - no test labels
  - configurable steps/lr/loss weights
  - returns adapted `phi*`, enhanced image, and diagnostics
- `tests/`
  - gradient-flow smoke tests
  - ISP identity test
  - parameter-bound test
  - inner-loop loss-decrease test on a synthetic differentiable objective
- `configs/baseline.yaml`
- `README.md`

### Important design constraints

1. `phi` should remain low-dimensional and interpretable.
2. Use bounded/reparameterized variables instead of unconstrained destructive ISP values where possible.
3. The inner loop must be differentiable enough that a later task can unroll it for meta-training.
4. Keep a clean distinction between:
   - source/meta training;
   - deployment-time per-image adaptation.
5. Do not update detector parameters in T001.
6. No test labels may enter `adapt.py`.
7. Make it easy to replace the semantic loss later with learned CLIP prompts.

### Acceptance criteria

T001 is DONE when:

- the package imports cleanly;
- tests pass;
- a demo command takes a synthetic or sample image, initializes `phi`, performs a few adaptation steps, and reports before/after ISP parameters and losses;
- gradients demonstrably reach `phi` through all enabled ISP operations;
- `CODEX_TO_CHATGPT.md` contains the exact implementation report and test results.

### Research note

The conceptual contribution we are targeting is not “CLIP + IA-YOLO + TTT.” It is:

> **Shift test-time adaptation from model space to image-formation space: instead of adapting what the detector knows, adapt how the detector sees.**

ViT³ is only conceptual motivation for treating inference as an inner-learning process with a sample-specific fast state. Here that fast state is `phi_t*`, not K/V/Q memory.

---

## Research review R001 — T001 Stage 1 (commit `60853a65`)

**Assessment:** APPROVED; T001 remains IN_PROGRESS.

The eight-coordinate bounded ISP is aligned with the research hypothesis: `phi` is compact/interpretable, zero is an identity state, and the implementation supports external functional `phi`, which is the right choice for later unrolled/meta test-time learning. The reported identity, bound, serialization, finite-gradient and gradcheck tests are a strong Stage-1 foundation. Do **not** add real CLIP or detector training yet; first finish the software contract and verify the inner-learning mechanics.

### Important correction / diagnostic before declaring T001 done

The final image `clamp(0,1)` is physically sensible but can create zero-gradient regions when an adapted ISP state saturates many pixels. This is not a blocker for Stage 1, but it can silently break test-time optimization/meta-gradients. Keep the current implementation for now, but expose and test **saturation diagnostics** so we can decide later whether a smooth output parameterization is needed.

### Next concrete work for T001

Proceed with Stages 2–4 as planned, with these mandatory checks:

1. Implement semantic-direction, frozen downstream consistency, regularization, and the identity-initialized parameter predictor behind clean interfaces. A deterministic semantic mock is preferred for T001.
2. Implement episodic `adapt.py` with functional SGD on `phi` only. The downstream model must stay in eval/frozen mode and receive no gradients/optimizer updates; no label argument should exist in the deployment adaptation API.
3. Add a higher-order-gradient test demonstrating that an outer scalar loss can backpropagate through at least one inner update to `phi_0` (and to predictor output if used). This is essential for the later meta-TTT task.
4. Add diagnostics returned by adaptation: raw `phi`, decoded physical parameters, per-step total/component losses, gradient norm for each ISP coordinate, and fraction of output pixels at/near 0 or 1 (saturation rate).
5. Add a synthetic test where the self-supervised objective has a known preferred ISP direction and verify: `L_self` decreases, at least one decoded parameter moves in the expected direction, all eight parameter gradients are finite, and separate images start from independent episodic states.
6. Run the full test suite plus the demo and report exact commands/results in `CODEX_TO_CHATGPT.md`. If any enabled ISP coordinate repeatedly has near-zero gradient because of clipping/saturation, flag it rather than hiding it with a looser test.

**T001 exit condition remains unchanged:** only mark DONE after the full package, adaptation loop, diagnostics, tests, demo, README/config, and exact report are present.
