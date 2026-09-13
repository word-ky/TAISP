import json
import torch
from torch import nn
from taisp import DifferentiableISP
from taisp.tta.spatial_dose import adapt_spatial_dose
from taisp.analysis.spatial_dose_study import episode_receipt,edge_smoke,spatial_summary
from test_spatial_dose import UniformLoss


def test_frozen_performance_gate_rejects_small_delta_replication_and_clean_drop():
    import copy
    from taisp.analysis.spatial_dose_study import spatial_advancement
    from taisp.analysis.run_t018a import CASE_NAMES
    candidate='spatial_dose_ours'
    metrics={g:{f'{c}_{m}':{k:.3+(.002 if m==candidate else 0) for k in ('AP','AP50','AP75')}
                 for c in CASE_NAMES for m in ('no_adapt','current_ours',candidate)} for g in ['aggregate']+[f'block{i}' for i in range(4)]}
    assert spatial_advancement(metrics,True)['gate']['passed']
    assert not spatial_advancement(metrics,False)['gate']['passed']
    for mode in ('small','blocks','conditions','clean'):
        v=copy.deepcopy(metrics)
        if mode=='small':
            for c in CASE_NAMES[:-1]:v['aggregate'][f'{c}_{candidate}']['AP']=.3009
        elif mode=='blocks':
            for g in ('block0','block1'):
                for c in CASE_NAMES[:-1]:v[g][f'{c}_{candidate}']['AP']=.299
        elif mode=='conditions':
            for c in CASE_NAMES[:3]:v['aggregate'][f'{c}_{candidate}']['AP']=.299
        else:v['aggregate'][f'clean_s0_{candidate}']['AP']=.2989
        assert not spatial_advancement(v,True)['gate']['passed']



def test_spatial_runtime_receipts_and_fixed_group_summary():
    x=torch.full((1,3,4,6),.4,dtype=torch.double);isp=DifferentiableISP().double()
    mask=torch.zeros(1,1,4,6,dtype=torch.double);mask[:,:,:,:3]=1
    a=adapt_spatial_dose(x,isp,UniformLoss(),UniformLoss(),mask)
    receipt=episode_receipt(x,isp,mask,a)
    assert all(receipt['checks'].values()) and len(receipt['processed_images'])==4
    rows=[dict(method='spatial_dose_ours',case=c,block=0,updated=True,spatial=receipt,diagnostics=a.diagnostics) for c in ['clean_s0','gamma_s1']]
    summary=spatial_summary(rows)
    assert summary['overall']['episodes']==2 and summary['block0']['mask_area']['mean']==.5
    assert summary['clean']['steps']['0']['phi_obj_norm']['max']==0


def test_full_empty_masks_and_empty_support_smoke_receipt(tmp_path):
    x=torch.full((1,3,4,6),.4,dtype=torch.double);isp=DifferentiableISP().double()
    source=nn.Linear(2,2).double().eval().requires_grad_(False)
    selected=dict(boxes=x.new_tensor([[0,0,3,3]]),labels=torch.tensor([1]),scores=x.new_tensor([.9]))
    path=tmp_path/'edges.json'
    edge_smoke(x,isp,source,UniformLoss(),UniformLoss(),selected,path,dict(semantic_steps=3,semantic_lr=.1,radius_eps=1e-12))
    r=json.loads(path.read_text());assert r['passed'] and r['evaluations']==0 and len(r['records'])==3
    assert all(all(v['checks'].values()) for v in r['records'])
