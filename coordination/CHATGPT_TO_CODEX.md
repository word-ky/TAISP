# ChatGPT → Codex

## T001 — Bootstrap the TAISP research codebase

**Status:** TODO

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
