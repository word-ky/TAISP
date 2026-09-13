import torch
from torch.nn import functional as F

from scripts.prepare_t026a_memory import select_instances
from taisp.analysis.roi_memory_ops import memory_health,retrieve_anchors


def test_memory_instance_selection_deterministic_disjoint_balanced():
    anns=[{'id':i,'image_id':i,'category_id':1 if i<30 else 2,'bbox':[1,2,3,4]} for i in range(60)]
    a=select_instances(anns,[1,2],set(range(60)),{0,30})
    b=select_instances(anns[::-1],[2,1],set(range(60)),{0,30})
    assert a==b and len(a)==32 and not {x['image_id'] for x in a}&{0,30}
    assert [x['class_rank'] for x in a]==list(range(16))*2


def test_balanced_memory_health_finite_normalized():
    x=F.normalize(torch.randn(1280,1024),dim=-1);c=torch.arange(80).repeat_interleave(16)
    assert memory_health(x,c)['passed']
    x[0]=0
    assert not memory_health(x,c)['passed']


def test_retrieval_class_only_top4_ties_and_detached_anchor():
    # Class1 entries tie; class2 is even more similar but cannot be retrieved.
    features=torch.cat([torch.tensor([[.6,.8]]).repeat(16,1),torch.tensor([[1.,0.]]).repeat(16,1)])
    classes=torch.tensor([1]*16+[2]*16)
    query=torch.tensor([[1.,0.]],requires_grad=True)
    a,ids,sims=retrieve_anchors(query,torch.tensor([1]),features,classes)
    assert ids.tolist()==[[0,1,2,3]] and torch.allclose(sims,torch.full((1,4),.6))
    assert torch.allclose(a,torch.tensor([[.6,.8]])) and not a.requires_grad
