import subprocess
import sys
import numpy as np
from taisp.analysis.gradient_regime_router import fit_router, assign, units


def test_two_opposite_regimes_and_lowest_index_ties():
    x=np.zeros((360,8));x[:180,0]=2.;x[180:,0]=-3.
    r=fit_router(x)
    assert r['status']=='router_complete' and r['counts']==[180,180]
    assert r['initial_indices']==[0,180] and r['iterations']==2
    assert r['assignments']==[0]*180+[1]*180
    assert assign([[0.]*8,[0.,1.]+[0.]*6],r['centroids']).tolist()==[0,0]


def test_zero_preserved_and_minimum_size_boundary():
    x=np.zeros((360,8));x[:288,0]=1.;x[288:,0]=-1.
    assert fit_router(x)['counts']==[288,72]
    assert fit_router(x)['status']=='router_complete'
    x[288,0]=1.;assert fit_router(x)['status']=='FAIL_ROUTER_COLLAPSE'
    assert fit_router(np.zeros((360,8)))['status']=='FAIL_ROUTER_COLLAPSE'
    u,z=units([[1e-13]+[0.]*7,[0.]*8]);assert not z[0] and z[1]
    assert u[0,0]==1e-13/(1e-13+1e-12)


def test_first_nonzero_and_cosine_initialization_ignore_magnitude():
    x=np.zeros((360,8));x[1:181,0]=1.;x[181:,0]=-1e-14
    r=fit_router(x)
    assert r['initial_indices']==[1,181] and r['counts']==[181,179]
    assert r['zero_count']==1 and r['assignments'][0]==0
    assert r==fit_router(x)


def test_router_import_has_no_detector_or_reference_dependency():
    # Package __init__ imports ISP/torch; router imports no detector or task oracle.
    code="import sys; import taisp.analysis.gradient_regime_router; assert not any(k.startswith('taisp.models') for k in sys.modules); assert not any(k.startswith('taisp.analysis.') and any(s in k for s in ('oracle','reference','source_meta')) for k in sys.modules)"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
