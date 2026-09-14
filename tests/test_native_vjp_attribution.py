import subprocess
import sys
import torch
from taisp.analysis import native_vjp_attribution as a
from taisp.analysis.native_task_signal import KEYS


def test_one_engine_multi_output_and_separate_orders():
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    y=torch.rand(1,3,12,13,device=device,requires_grad=True)
    parts={k:(i+1)*y.square().mean() for i,k in enumerate(KEYS)}
    cs,canonical,reverse=a.graph_cotangents(sum(parts.values()),parts,y)
    assert set(cs)=={'sum','multi','sep','sep_rev'}
    assert cs['sep'].dtype==cs['sep_rev'].dtype==torch.float64
    for k in cs:torch.testing.assert_close(cs[k].double(),cs['sum'].double(),rtol=1e-6,atol=1e-8)
    for k in KEYS:torch.testing.assert_close(canonical[k],reverse[k])


def test_ordered_pair_dispersion_and_zero_convention():
    d=a.dispersion([[1.,0.],[2.,0.],[1.,0.],[2.,0.],[1.,0.]])
    assert d=={'max_pairwise_relative_l2':1.,'min_pairwise_cosine':1.}
    assert a.compare([0.],[0.])=={'relative_l2':0.,'cosine':1.}


def test_exact_replacement_gate_ignores_old_diagnostic_failure():
    episodes=[]
    for i in range(8):
        reps=[]
        for j in range(5):
            reps.append({'vectors':{'sum':[1.,0.]},'g_hard':[1.,0.],
                'projected':{k:{'relative_l2':v} for k,v in [('multi',.0001),('sep',.0002),('sep_rev',.0003)]},
                'image_space':{k:{'relative_l2':.0001} for k in ['multi','sep','sep_rev']},
                'direct_phi':{'relative_l2':1e-6,'cosine':1.},'reverse_order_relative':0.,
                'old_additivity':{'passed':False},'integrity_passed':True})
        episodes.append({'episode_index':i,'image_id':i//2,'case':'clean_s0','support_match':True,'repetitions':reps})
    assert a.summarize(episodes)['passed']
    episodes[0]['repetitions'][0]['direct_phi']['relative_l2']=1.01e-5
    s=a.summarize(episodes)
    assert not s['passed'] and s['status']=='BLOCKED' and not s['gates']['direct_phi_parity_all8']
    assert not s['GT_reference_authorized']


def test_deterministic_probe_restores_setting_after_exception(monkeypatch):
    old=torch.are_deterministic_algorithms_enabled()
    monkeypatch.setattr(a,'state_hash',lambda model:'fixed');monkeypatch.setattr(a,'isolated',lambda model:True)
    def fail(*args):raise RuntimeError('nondeterministic test operator')
    monkeypatch.setattr(a,'pseudo_native_loss',fail)
    result=a.deterministic_probe(None,lambda x,p:x,torch.ones(1,3,2,2),[])
    assert not result['succeeded'] and result['restored'] and result['non_gating']
    assert torch.are_deterministic_algorithms_enabled()==old


def test_no_annotation_reference_imports():
    code="import sys; import taisp.analysis.native_vjp_attribution; assert not [k for k in sys.modules if k.startswith('taisp.') and any(x in k for x in ('oracle','reference','source_meta','clip_semantic','memory'))]"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
