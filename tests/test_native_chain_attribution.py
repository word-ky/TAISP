import copy
import subprocess
import sys
import torch
from taisp.analysis import native_chain_attribution as a
from taisp.analysis.native_task_signal import KEYS
from taisp.isp import DifferentiableISP


def test_same_graph_scalar_calls_and_exact_cotangent_chain():
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(7)
    x=torch.rand(1,3,16,17,device=device)*.8+.1
    isp=DifferentiableISP().to(device);phi=x.new_zeros(8,requires_grad=True);y=isp(x,phi)
    parts={k:(i+1)*y.square().mean() for i,k in enumerate(KEYS)}
    cs,canonical,reverse=a.graph_cotangents(sum(parts.values()),parts,y)
    refs,_=a.common_jvp(x,isp,x.new_ones(1,1,*x.shape[-2:]),cs)
    direct=torch.autograd.grad(y,phi,grad_outputs=cs['sum_a'])[0].tolist()
    assert a.chain_parity(refs['sum_a']['global'],direct,cs['sum_a'])['passed']
    for k in cs:torch.testing.assert_close(cs[k].double(),cs['sum_a'].double(),rtol=1e-6,atol=1e-8)
    for k in KEYS:torch.testing.assert_close(canonical[k],reverse[k])
    assert phi.grad is None and isp.phi.grad is None


def test_symmetric_relative_and_zero_chain():
    assert a.compare([1.],[2.])==a.compare([2.],[1.])=={'relative_l2':.5,'cosine':1.}
    assert a.dispersion([[1.],[2.]])['max_pairwise_relative_l2']==.5
    assert a.chain_parity([0.],[0.],torch.zeros(1))['passed']
    assert not a.chain_parity([0.],[0.],torch.ones(1))['passed']


def episodes():
    r={'g_native':[1.,0.],'chain_parity':{'passed':True,'relative_l2':1e-6,'cosine':1.},
       'scalar_self_noise':.0003,'scalar_min_cosine':1.,'integrity_passed':True,
       'image_space':{k:{'relative_l2':.0006} for k in ['multi','sep','rev']}}
    return [{'episode_index':i,'image_id':i//2,'case':'clean_s0','support_match':True,'repetitions':[copy.deepcopy(r) for j in range(5)]} for i in range(8)]


def test_frozen_four_gates_and_seven_of_eight_rule():
    es=episodes();assert a.summarize(es)['passed']
    for r in es[0]['repetitions']:r['image_space']['rev']['relative_l2']=.00062
    assert a.summarize(es)['passed']
    for r in es[1]['repetitions']:r['image_space']['multi']['relative_l2']=.00062
    assert a.summarize(es)['first_failed_gate']=='gate4'
    es=episodes();es[0]['repetitions'][0]['chain_parity']['passed']=False
    assert a.summarize(es)['first_failed_gate']=='gate2'
    es=episodes();es[0]['repetitions'][0]['g_native']=[1.,.01]
    assert a.summarize(es)['first_failed_gate']=='gate1'
    es=episodes();es[0]['repetitions'][0]['scalar_min_cosine']=.99998
    assert a.summarize(es)['first_failed_gate']=='gate3'
    es=episodes();es[0]['repetitions'][0]['image_space']['sep']['relative_l2']=.00201
    assert a.summarize(es)['first_failed_gate']=='gate4'
    es=episodes();es[0]['repetitions'][0]['integrity_passed']=False
    assert a.summarize(es)['first_failed_gate']=='gate4'


def test_annotation_free_import_boundary():
    code="import sys; import taisp.analysis.native_chain_attribution; assert not [k for k in sys.modules if k.startswith('taisp.') and any(x in k for x in ('oracle','reference','source_meta','clip_semantic','memory'))]"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)
