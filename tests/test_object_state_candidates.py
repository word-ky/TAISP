import subprocess
import sys
from types import SimpleNamespace
import torch
from taisp.analysis import object_state_candidates as o
from taisp.analysis.orthogonal_candidates import candidate


def test_weighted_normalized_roi_pool_and_empty():
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    z=torch.zeros(2,1024,device=device);z[0,0]=3.;z[1,1]=4.
    d=o.pool_features(z,z.new_tensor([.6,.8]))
    torch.testing.assert_close(d[:2],z.new_tensor([.6,.8]))
    assert torch.count_nonzero(d[2:])==0
    assert not o.pool_features(z[:0],z.new_zeros(0)).any()


def test_same_support_boxes_no_class_feature_or_detector_resample(monkeypatch):
    assert o.hard_candidate is candidate
    image=torch.ones(1,3,4,5);calls=[]
    isp=lambda x,p:x
    def roi(source,x,boxes):
        calls.append(boxes.clone());z=x.new_zeros(2,1024);z[0,0]=1.;z[1,1]=1.
        return z,x.new_zeros(2,91)
    monkeypatch.setattr(o,'fixed_roi_representation',roi)
    support={'boxes':[[0.,0.,1.,1.],[1.,1.,2.,2.]],'scores':[.6,.8],'labels':[1,2]}
    a=o.object_state(None,isp,image,support);support['labels']=[50,80]
    b=o.object_state(None,isp,image,support)
    assert a==b and a['support_count']==2 and len(a['roi_feature_sha256'])==2
    torch.testing.assert_close(calls[0],torch.tensor(support['boxes']))
    e=o.object_state(None,isp,image,{'boxes':[],'scores':[],'labels':[]})
    assert len(calls)==2 and e['d_obj']==[0.]*1024 and e['mean_score']==0


def test_candidate_import_boundary():
    code="import sys; import taisp.analysis.object_state_candidates; assert not [k for k in sys.modules if k.startswith('taisp.') and any(s in k for s in ('annotation','oracle','reference','source_meta','common_jacobian','clip_semantic','memory','native_task_signal','soft_pseudo'))]"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
