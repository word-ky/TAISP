#!/usr/bin/env bash
set -euo pipefail

# Invoke through the existing AutoDL workflow with TAISP's venv on PATH.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
python -c 'import os,sys,torch,pytest,yaml; print("source_revision",os.environ["TAISP_SOURCE_REVISION"]); print("python",sys.version); print("torch",torch.__version__,"cuda",torch.version.cuda); print("pytest",pytest.__version__,"yaml",yaml.__version__); print("gpu",torch.cuda.get_device_name(0))' | tee "$AUTODL_ARTIFACTS_DIR/environment.txt"
python -m pytest -q | tee "$AUTODL_ARTIFACTS_DIR/pytest.txt"
python -m taisp.demo --config configs/baseline.yaml --device cuda:0 --output "$AUTODL_ARTIFACTS_DIR/demo_cuda.json" > /dev/null
python -c 'import json,os; from pathlib import Path; r=json.loads((Path(os.environ["AUTODL_ARTIFACTS_DIR"])/"demo_cuda.json").read_text()); print({k:r[k] for k in ["loss_before","loss_after","physical_after","repeatedly_near_zero_gradient_coordinates"]})'
