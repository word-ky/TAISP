# TAISP

Adapt how a frozen model sees: per-image test-time optimization of an
eight-dimensional differentiable image-processing state. T001 is a runnable
PyTorch framework with **synthetic mock guidance**, not a trained detector or a
validated restoration method. No ViT³ architecture is implemented.

## Run

Python >=3.10, PyTorch >=2.2, PyYAML >=6; pytest >=8 for tests.
Install a PyTorch build appropriate for your CPU/CUDA environment first.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m taisp.demo --config configs/baseline.yaml --output research_log/demo_cpu.json
python -m taisp.demo --config configs/baseline.yaml --device cuda:0 --output research_log/demo_cuda.json
```

Run from the repository root. The demo uses seed 42, a generated 32x32 RGB
image, deterministic RGB-mean semantic mock, and a randomly initialized frozen
feature model. No datasets, annotations, pretrained weights, or downloads are
needed by the demo. It reports raw and physical parameters before/after,
component losses, predictions, and all per-step diagnostics. Set
`use_predictor: true` in YAML to use the identity-initialized small predictor.

## State and ISP

`DifferentiableISP` takes float RGB tensors `(B,3,H,W)` in `[0,1]` and raw
`phi` of shape `(8,)` (shared) or `(B,8)` (independent). Operators run in this
order, followed by output clipping to `[0,1]`:

| Raw coordinate | Physical mapping | Range | Identity |
|---|---|---|---|
| gamma | exp(log(2) tanh(phi[0])) | [0.5,2] | 1 |
| RGB gains | exp(log(2) tanh(phi[1:4])) | [0.5,2] | 1,1,1 |
| contrast | exp(log(2) tanh(phi[4])) | [0.5,2] | 1 |
| brightness | 0.25 tanh(phi[5]) | [-0.25,0.25] | 0 |
| tone | 0.5 tanh(phi[6]) | [-0.5,0.5] | 0 |
| sharpening | 0.5 tanh(phi[7]) | [-0.5,0.5] | 0 |

Zero raw state is identity. Gamma is a shifted, endpoint-normalized power with
epsilon 1e-6 for finite black-pixel gradients. Contrast pivots about 0.5;
tone adds `t*x*(1-x)`. Sharpening adds a signed 3x3 unsharp residual (negative
values smooth). Intermediate values can exceed [0,1]; clipping occurs once at
the output. Physical mappings bound parameters, not every intermediate pixel.

The final clamp can zero gradients for clipped pixels. Adaptation records the
fraction at/within 1e-4 of 0 or 1. A deliberate fully saturated test reproduces
zero image-loss gradients. No smooth clipping change was made; this is a known
limitation to evaluate on real data. Ordinary non-saturated tests verify all
eight coordinates against finite differences.

## Adaptation and loss interfaces

`taisp.tta.adapt(image, isp, semantic_loss, ...)` accepts **one image per call**.
Initialization precedence is explicit `phi0`, then optional predictor, then
`isp.phi`. A new state is cloned on every call; no episode mutates `isp.phi`.
Functional SGD performs `phi = phi - lr * grad_phi(loss)` for the YAML step count.
Returns `phi0`, `phi`, final enhanced image, frozen-model prediction, diagnostics.

The three weighted terms are:

- Semantic: callable `(original, enhanced) -> scalar`. T001 uses negative
  feature displacement projected onto a unit desired direction:
  `-mean(<E(enhanced)-stopgrad(E(original)), normalize(direction)>)`.
  This avoids division by zero displacement at identity. The mock encoder is
  RGB means and the demo direction brightens RGB. This is not a CLIP cosine loss.
- Consistency: MSE between frozen downstream tensors for enhanced and original
  images; the original reference is detached. A real detector adapter should
  expose aligned differentiable pre-NMS features/predictions. T001's demo model
  is only a feature-model stand-in, with no detection metric.
- Regularization: `||phi-phi0||² + identity_weight*||phi||²`, in raw coordinates,
  averaged over states and multiplied by `regularization_weight`.

The downstream module is set permanently to eval and `requires_grad=False`.
Gradients still pass through its enhanced-image branch into phi. Predictor
weights are not optimized. Deployment outputs and initialization are detached;
there is no label argument and no source/meta-training routine in `adapt.py`.

`differentiable=True` retains the functional update graph with `create_graph`;
an outer scalar may differentiate into supplied phi0 or the optional predictor.
This is an extension seam, not an implemented meta-training method. Higher-order
finite-difference and predictor-gradient tests exercise it. All modules/input
must be on the same device. Gradients can be enabled inside outer `no_grad`,
but do not call adaptation inside `torch.inference_mode()`.

Diagnostics contain K+1 states/losses and K update gradients: signed/absolute
per-coordinate values, total gradient norm, raw/decoded state and saturation.
The final state has no update gradient because no further update is performed.
The demo flags coordinates with absolute gradient below 1e-8 on every update.

## Layout and evidence

`taisp/isp`: operators/state; `taisp/models`: optional predictor;
`taisp/losses`: interchangeable objectives; `taisp/tta`: deployment adaptation;
`tests`: identity/bounds/serialization, first/higher-order derivatives, freeze,
independent episodes, loss direction/decrease, saturation, demo and CUDA parity.

Coordination rules and task queue live in `coordination/`. Exact results,
environment/commands, decisions and remote run receipts live in `research_log/`
and are summarized in `coordination/CODEX_TO_CHATGPT.md`. No baseline code or
third-party implementation was copied: the starting repository had only these
coordination documents. Real CLIP prompts, task alignment, meta-training and
real detection evaluation remain later research tasks.
