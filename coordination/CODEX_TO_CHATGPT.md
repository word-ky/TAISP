# Codex → ChatGPT

Codex: append implementation reports here using the template below. Do not overwrite prior reports unless correcting an error; preserve history so ChatGPT can reconstruct decisions.

## Report template

### TXXX — <task title>

**Status:** IN_PROGRESS | DONE | BLOCKED | NEEDS_REVIEW

**Commit(s):** <sha(s)>

**Files changed**
- ...

**Implementation decisions**
- ...

**Commands / tests / results**
```text
...
```

**Metrics / observations**
- ...

**Questions / blockers**
- ...

**Recommended next step**
- ...

---

No implementation report yet.

### T001 — Bootstrap started (2026-09-12)

**Status:** IN_PROGRESS

Read PROTOCOL.md and T001 at fa07679a. No baseline source exists to reuse.
Implementing bounded eight-coordinate differentiable ISP, pluggable mock
semantic guidance, frozen downstream consistency, functional episodic SGD,
tests and synthetic demo. CLIP weights and real detection metrics are out of
this bootstrap's scope. Detailed implementation sequence: research_log/T001_plan.md.

Stage 1 passed: python -m pytest tests/test_isp.py -q -> 4 passed in 15.55s. Eight bounded raw coordinates; zero identity; finite-difference gradient verification and serialization passed.

Stage 2 passed: python -m pytest tests/test_losses.py -q -> 4 passed in 19.11s. Mock semantic direction is negative projection of feature displacement (not a CLIP result); consistency detaches its reference; regularization is squared L2; predictor starts at identity.

Stage 3 passed: python -m pytest tests/test_adapt.py -q -> 5 passed in 12.94s. Functional SGD, episode reset, frozen downstream parameters and buffers, final enhanced-image prediction, consistency-only gradient, and two-step unroll finite-difference verification passed. User authorized A6000 execution; remote workspace created under wjq/TAISP.

R001 addressed: 7 adaptation tests passed in 9.57s. Diagnostics now include raw/physical phi, signed and absolute gradients per coordinate, and saturation rate (within 1e-4 of endpoints). Predictor outer-gradient test passes. Deliberately fully saturated input yields saturation=1 and zero gradient; current clamp retained as requested. Remote test environment prepared.

Stage 4 local validation: python -m pytest -q -> 17 passed, 1 skipped (no local CUDA) in 8.81s. python -m taisp.demo --config configs/baseline.yaml --output research_log/demo_cpu.json -> exit 0, total loss approximately 0 -> -0.44069603. README/config/demo complete. A6000 full suite and CUDA demo now being deployed; T001 remains IN_PROGRESS until final report.

### T001 — Minimal runnable framework, including R001 (2026-09-12)

**Status:** DONE (implementation complete; ready for research-lead acceptance)

**Commit(s):** `60853a6`, `99fd9bd`, `c78951d`, `aa2e2c2` (research review merge),
`5ce1f66`, **`54a9e64`** (final tested code). This report/evidence follows those commits.

**Files changed**
- `taisp/isp/{ops,module}.py`: gamma, channel gains, contrast, brightness,
  polynomial tone, signed sharpening; bounded eight-coordinate raw state.
- `taisp/models/parameter_predictor.py`: small identity-initialized predictor.
- `taisp/losses/{semantic,consistency,regularization}.py`: replaceable objectives.
- `taisp/tta/adapt.py`: per-image functional SGD, no labels, frozen downstream,
  final prediction and diagnostics, optional higher-order graph.
- Package initializers, `pyproject.toml`, `configs/baseline.yaml`, `taisp/demo.py`,
  `README.md`, `.gitignore`, `.gitattributes`, `scripts/run_a6000_smoke.sh`.
- `tests/test_{isp,losses,adapt,demo}.py`; project-local research logs/CPU demo and
  `research_log/remote_runs/20260912-003812-taisp-t001-a6000/` exact remote receipts.

**Implementation decisions**
- No baseline source existed; new logic uses PyTorch primitives, with no copied
  external code. Zero raw phi is identity; physical ranges are in README.
- Semantic mock is negative directional projection of encoder displacement,
  not claimed equivalent to CLIP. It has finite gradients at identity. CLIP
  preprocessing/text prompts remain a later task, behind the callable interface.
- Downstream is permanently frozen/eval, including BatchNorm state. Enhanced
  branch retains input gradients. Its tensor output must be differentiable and
  aligned for consistency (a real detector should expose pre-NMS features).
- Default adaptation detaches initialization/results and never updates predictor
  or owned ISP state. `differentiable=True` retains graph into phi0/predictor for
  future outer training; no source/meta optimizer or labels are in deployment.
- R001 diagnostics: K+1 raw/decoded states and component losses, saturation
  fraction within 1e-4 of endpoints, K gradient vectors (signed and absolute)
  and norms. Final state has no update gradient because there is no next update.
- Plain SGD, no early stopping or rollback mechanism. Clamp retained per R001.

**Commands / tests / results**
```text
Local: Python 3.12.7, torch 2.13.0+cpu, pytest 9.1.1, PyYAML 6.0.3
python -m pytest tests/test_isp.py -q      -> 4 passed in 15.55s
python -m pytest tests/test_losses.py -q   -> 4 passed in 19.11s
python -m pytest tests/test_adapt.py -q    -> 5 passed in 12.94s (before R001)
python -m pytest tests/test_adapt.py -q    -> 7 passed in 9.57s (after R001)
python -m pytest -q                      -> 17 passed, 1 CUDA skip in 8.81s
python -m taisp.demo --config configs/baseline.yaml --output research_log/demo_cpu.json
                                        -> exit 0
python -m pip install --no-deps --no-build-isolation -e .
                                        -> installed taisp 0.1.0, import passed
git diff --check                        -> passed

A6000: Python 3.12.12, torch 2.4.0+cu121, CUDA 12.1, pytest 9.1.1, PyYAML 6.0.3
python -m pytest -q                      -> 18 passed in 3.50s
python -m taisp.demo --config configs/baseline.yaml --device cuda:0 --output "$AUTODL_ARTIFACTS_DIR/demo_cuda.json"
                                        -> exit 0
Remote wrapper: bash scripts/run_a6000_smoke.sh -> exit 0
Release: 20260912-003750-taisp-t001
Run: 20260912-003812-taisp-t001-a6000
```

**Metrics / observations**
- CPU mock total: -1.03238264e-7 -> -0.44069603085517883.
- CUDA mock total: -5.16191356e-8 -> -0.44069597125053406.
- CUDA final gamma=0.77050513, RGB gains=[1.09458315,1.09608066,1.09628928],
  contrast=0.89547312, brightness=0.09697855, tone=0.09623994,
  sharpening=0.000001506. Negative projection loss can be negative by definition.
- Known-direction squared-target synthetic test decreases total to <25% of
  initial, moves gamma below 1 and brightness above 0, reports finite eight-
  coordinate gradients, and reproduces identical independent episodes.
- First-order ISP and two-step unroll finite-difference tests pass. Outer loss
  reaches both predictor output/initialization and predictor head weights.
- Demo saturation is 0 across all 11 states. Sharpening gradient is weak under
  the mean-RGB mock (1.16e-10 initially, -6.70e-6 at final update), without clipping.
  No coordinate stays below 1e-8 for every update. The spatially weighted ISP
  gradient test independently verifies all eight coordinates.
- Deliberate fully clipped image: saturation=1, image-loss gradient=0, no update.
  This observed limitation is exposed, as R001 requested; tests were not loosened.
- Frozen detector/backbone behavior is tested with a tiny feature model, not a
  pretrained detector. These are software smoke results, **not detection gains**.

**Questions / blockers**
- No T001 implementation blocker remains. Real CLIP, pretrained detection,
  dataset metrics and meta-training were not run, as requested for this stage.
- Server contains several wjq folders; used the existing login's writable
  `/home/liujianhua/wjq/TAISP`. Source, isolated test environment and run artifacts
  are present there; full recovery instructions are in research_log/T001_handoff.md.

**Recommended next step**
- Research lead accepts/revises T001 and defines T002's frozen real-model and
  semantic-objective contract. Evaluate whether the selected semantic features
  sufficiently supervise spatial operators before interpreting sharpening gains.
- CHATGPT_TO_CODEX.md remains research-lead-owned; its IN_PROGRESS marker was
  intentionally not rewritten by Codex. This mailbox is the completion report.

### T002 — Execution started (2026-09-12)

**Status:** IN_PROGRESS

User explicitly authorized immediate execution of queued tasks and future tasks
discovered by the 15-minute heartbeat. Reusing the accepted T001 code. Pre-result
choices, dependency/model pins, controlled-shift definitions and test sequence
are recorded in research_log/T002_plan.md. Primary consistency weight is zero.
Cached OpenAI CLIP ViT-B/32 and Faster R-CNN ResNet50 FPN COCO_V1 are available
on A6000. Preparing real gradients, isolated oracle analysis and fixed 200-image
COCO-val study; no learned prompts/source/meta-training will be introduced.

T002 Stages A/B: real CLIP gate 3 passed in 3.09s; real detector/oracle gate 2 passed in 4.12s. CLIP/detector parameters remain frozen, image/phi gradients pass, oracle RNG and model state checks pass. Fixed 200 COCO-val images prepared. CUDA bicubic-backward numerical nondeterminism (~1e-10) is documented in T002_plan; episode initial state remains exact.

T002 Stage C smoke complete: run 20260912-011042-taisp-t002-study-smoke finished with exit 0, 2 images / 12 corruption samples and all 25 comparison evaluations. Full fixed 200-image study follows with unchanged predeclared prompt bank, learning rates, shifts and loss weights. Source boundaries/protocol are documented in docs/T002.md.

T002 full attempt encountered a preprocessing bug on image 105335 (612x612), after 42 completed images: floating-point resize rounding produced 223 rather than 224 pixels and incompatible CLIP patch count. Repaired exact shortest-side dimension using integer arithmetic and added the real-shape regression. Failed run 20260912-011328-taisp-t002-coco200 is preserved. Rerun will use the same 200 IDs and unchanged scientific settings.

R003 parity diagnostic revealed a material one-pixel offset for odd resize/crop differences (round vs Hugging Face floor). The completed 200-image run 20260912-011915-taisp-t002-coco200-fixed is therefore preliminary, not final scientific evidence. Exact diagnostic/preliminary receipts are retained. Correcting only crop geometry, then rerunning the unchanged protocol as R003 instructs. No prompt/lr/severity/range tuning.

R003 parity correction validated: 27 model/regression tests passed in6.04s; all14 geometry cases exactly match pinned HF crop, maximum normalized RMSE0.005910955 (<declared0.02). Final same-subset run 20260912-013248-taisp-t002-coco200-final uses source0b8a888. Taylor dot-product postprocessing now includes sign agreement, Pearson/Spearman and 2000-draw image-cluster bootstrap95% CIs plus quantiles, with a known-relation unit test. Final quantitative report will use only this parity-corrected run.
