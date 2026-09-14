import ast
import inspect
import math
import subprocess
import sys

import pytest
from taisp.analysis.gradient_state_features import features


def test_exact21_order_and_epsilon():
    p=[3.,4.]+[0.]*6;c=[0.,2.]+[0.]*6
    v=features(p,c,7.,0.)
    assert len(v)==21
    assert v[:8]==pytest.approx([x/(5+1e-12) for x in p])
    assert v[8:16]==pytest.approx([x/(2+1e-12) for x in c])
    assert v[16:]==pytest.approx([math.log(5+1e-12),math.log(2+1e-12),8/((5+1e-12)*(2+1e-12)),7,0])


def test_zero_gradients_retained():
    v=features([0.]*8,[0.]*8,0.,0.)
    assert v[:16]==[0.]*16 and v[16:18]==[math.log(1e-12)]*2
    assert all(math.isfinite(x) for x in v)


def test_candidate_imports_no_annotation_reference_transitively():
    code="import sys; import taisp.analysis.gradient_state_candidates; assert not [k for k in sys.modules if k.startswith('taisp.') and any(s in k for s in ('oracle','source_meta','common_jacobian','_reference','memory'))]"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
