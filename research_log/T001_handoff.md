# T001 handoff — 2026-09-12, Asia/Shanghai

Status: DONE (implementation); research lead's acceptance is pending.
Read this file, progress.md and coordination mailboxes on resume.

## Delivered and tested

Eight-coordinate bounded differentiable ISP, optional identity predictor,
mock semantic projection, frozen downstream feature consistency, squared-state
regularization, functional episodic SGD with optional differentiable unroll,
R001 saturation/coordinate diagnostics, tests, YAML config, README and demo.

Code revision tested on A6000: **54a9e64**.
Commits: 60853a6 (ISP), 99fd9bd (losses/predictor), c78951d (adaptation),
aa2e2c2 (merge research R001), 5ce1f66 (R001 diagnostics), 54a9e64 (demo/docs).
Only evidence/report files change after this tested code revision.

## Verification

- Windows Python 3.12.7 / torch 2.13.0+cpu / pytest 9.1.1 / PyYAML 6.0.3.
- `python -m pytest -q`: **17 passed, 1 skipped in 8.81s**. CUDA skipped locally.
- `python -m taisp.demo --config configs/baseline.yaml --output research_log/demo_cpu.json`:
  exit 0; total loss -1.0323826415969961e-7 -> -0.44069603085517883.
- `python -m pip install --no-deps --no-build-isolation -e .`: installed taisp 0.1.0;
  package import resolves to this repository.
- A6000 Python 3.12.12 / torch 2.4.0+cu121 / CUDA 12.1 / pytest 9.1.1 / PyYAML 6.0.3.
- `python -m pytest -q`: **18 passed in 3.50s**, including CUDA parity.
- `python -m taisp.demo --config configs/baseline.yaml --device cuda:0 --output "$AUTODL_ARTIFACTS_DIR/demo_cuda.json"`:
  exit 0; total loss -5.1619135632563484e-8 -> -0.44069597125053406.
- `git diff --check`: passed.

CUDA final physical parameters: gamma 0.77050513, RGB gains
[1.09458315, 1.09608066, 1.09628928], contrast 0.89547312,
brightness 0.09697855, tone 0.09623994, sharpening 0.000001506.
All 11 demo states have saturation rate 0. No coordinate is below 1e-8 on
every update. Sharpening is weakly supervised by this mean-RGB mock: initial
gradient 1.16e-10, final update gradient -6.70e-6. This is not caused by clipping.
A deliberate saturated test reproduces saturation=1 and image-loss gradient=0;
the clamp is retained per R001. No tolerance was loosened.

## Remote execution / recovery

- Confirmed two RTX A6000 GPUs (49140 MiB each); used GPU 0 for a seconds-long smoke.
- Workspace: `/home/liujianhua/wjq/TAISP` (writable wjq under the existing login).
- Venv: `.venv`, using system packages from the existing Python 3.12 PyTorch
  environment; only pytest/PyYAML and pytest dependencies installed in this venv.
- Release: `20260912-003750-taisp-t001`.
- Run: `20260912-003812-taisp-t001-a6000`; exit 0, finished 00:38:26 +08:00.
- Current code: `current` -> `releases/20260912-003750-taisp-t001`.
- Remote run receipt: `runs/20260912-003812-taisp-t001-a6000/`.
- Local identical receipt: `research_log/remote_runs/20260912-003812-taisp-t001-a6000/`.
- Receipt files: train.log, meta.json, run.sh, artifacts/environment.txt,
  artifacts/pytest.txt, artifacts/demo_cuda.json.
- Existing workflow root: `D:\work\claude-autodl\autodl-workflow-clean`.
  Set `AUTODL_CONFIG_PATH` to this project's ignored `.autodl/config.json` before
  invoking its PowerShell deploy/run/log tools. Do not use its unrelated default
  remote_base. Connection config is local-only and is not committed.
- Deployed with `autodl-deploy.ps1 -Tag taisp-t001 -Source <this project>`;
  started with `autodl-run.ps1 -Name taisp-t001-a6000 -Cmd
  'export PATH=/home/liujianhua/wjq/TAISP/.venv/bin:$PATH; export TAISP_SOURCE_REVISION=54a9e64; bash scripts/run_a6000_smoke.sh'`.
- No job remains running. No datasets, weights or real detection results generated.

## Next action

Research lead reviews T001/R001 using the full mailbox report and actual code.
Await the next research task before adding CLIP, real detector adapters,
task alignment, source training, or meta-training. Do not infer benchmark gains
from the mock objective. The research lead owns CHATGPT_TO_CODEX.md status updates.
