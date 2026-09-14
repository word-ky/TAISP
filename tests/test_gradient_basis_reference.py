import subprocess
import sys


def test_invalid_lock_rejected_before_annotation_or_oracle(tmp_path):
    code = """
import json,sys
from pathlib import Path
from taisp.analysis.gradient_basis_reference import verify_inputs
p=Path(sys.argv[1]);lock=p/'lock.json'
lock.write_text(json.dumps({'candidate_commit':'', 'files':{}}))
try:verify_inputs(p/'absent-cohort',p,lock,p/'absent-permutations')
except AssertionError:pass
else:raise RuntimeError('incomplete lock accepted')
assert 'taisp.analysis.oracle' not in sys.modules
assert 'taisp.analysis.common_jacobian' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', code, str(tmp_path)], check=True)
