import torch

from taisp import DifferentiableISP
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector_signal import flip_boxes,select_predictions
from taisp.models.flip_consensus import flip_consensus
from taisp.tta.trust_radius import adapt_clip_radius
from test_native_pseudo_target import ToyClip


def pred(boxes,scores,labels=None):
    return dict(boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
                scores=torch.tensor(scores),labels=torch.tensor(labels if labels is not None else [1]*len(scores)))


def flipped(p,width=100):
    return {**p,'boxes':flip_boxes(p['boxes'],width)}


def test_horizontal_inversion_score_iou_boundaries_and_class_eligibility():
    a=pred([[0,0,4,4],[10,0,14,4],[20,0,24,4]],[.5,.8,.9],[1,2,3])
    b=pred([[1,0,5,4],[10,0,14,4],[20,0,24,4]],[.5,.9,.4999],[1,3,3])
    torch.testing.assert_close(flip_boxes(flip_boxes(a['boxes'],100),100),a['boxes'],rtol=0,atol=0)
    support,receipt=flip_consensus(a,flipped(b),100)
    assert receipt['retained_count']==1 and receipt['retained_pairs'][0]['original_index']==0
    assert torch.equal(support['boxes'],a['boxes'][:1]) and support['scores'].item()==.5
    b['boxes'][0,0]=1.01
    assert flip_consensus(a,flipped(b),100)[1]['retained_count']==0


def test_geometric_confidence_priority_not_iou_priority_and_one_to_one():
    a=pred([[0,0,4,4],[1,0,5,4]],[.6,.9])
    b=pred([[0,0,4,4]],[.9])
    support,r=flip_consensus(a,flipped(b),100)
    assert [p['original_index'] for p in r['retained_pairs']]==[1]
    assert torch.equal(support['boxes'],a['boxes'][1:])
    assert torch.equal(support['scores'],a['scores'][1:])


def test_pair_ties_use_original_detection_indices_then_flip_indices():
    a=pred([[0,0,4,4]]*3,[.8,.8,.8]);b=pred([[0,0,4,4]]*3,[.8,.8,.8])
    _,r=flip_consensus(a,flipped(b),100)
    assert [(p['original_index'],p['flip_index']) for p in r['retained_pairs']]==[(0,0),(1,1),(2,2)]
    assert r==flip_consensus(a,flipped(b),100)[1]


def test_matching_precedes_top20_and_retains_original_scores_detached():
    boxes=[[3*i,0,3*i+2,2] for i in range(25)]
    a=pred(boxes,[.99-.001*i for i in range(25)]);b=pred(boxes,[.5]*20+[.99]*5)
    a['boxes'].requires_grad_();a['scores'].requires_grad_()
    support,r=flip_consensus(a,flipped(b),100)
    assert r['matched_count']==25 and r['retained_count']==20
    assert [p['original_index'] for p in r['retained_pairs'][:5]]==[20,21,22,23,24]
    indices=torch.tensor([p['original_index'] for p in r['retained_pairs']])
    assert torch.equal(support['scores'],a['scores'][indices])
    assert all(not v.requires_grad for v in support.values())
    loss=DetectorNativeLoss(None,{'base':support},'det_pseudo')
    torch.testing.assert_close(loss.weights,support['scores']/support['scores'].sum(),rtol=0,atol=0)
    assert not loss.boxes.requires_grad and not loss.weights.requires_grad


def test_empty_consensus_exact_identity_no_fallback_and_baseline_unmodified():
    a=pred([[0,0,4,4]],[.9]);b=pred([[20,0,24,4]],[.9])
    baseline=select_predictions(a)
    support,r=flip_consensus(a,flipped(b),100)
    assert r['zero_consensus'] and r['retention_fraction']==0 and len(baseline['boxes'])==1
    loss=DetectorNativeLoss(None,{'base':support},'det_pseudo')
    x=torch.rand(1,3,11,13);isp=DifferentiableISP()
    result=adapt_clip_radius(x,isp,loss,ToyClip())
    assert torch.equal(result.phi,torch.zeros(8)) and torch.equal(result.enhanced,isp(x,torch.zeros(8)))
    assert all(d['support_count']==0 and d['gradient_norm']==0 for d in result.diagnostics)
    assert all(torch.equal(v,select_predictions(a)[k]) for k,v in baseline.items())
    empty=pred([],[])
    assert flip_consensus(empty,empty,100)[1]['retention_fraction'] is None


def test_frozen_performance_gate_rejects_small_delta_replication_and_clean_drop():
    import copy
    from taisp.analysis.flip_consensus_study import consensus_advancement
    from taisp.analysis.run_t018a import CASE_NAMES
    candidate='flip_consensus_ours'
    metrics={g:{f'{c}_{m}':{k:.3+(.002 if m==candidate else 0) for k in ('AP','AP50','AP75')}
                 for c in CASE_NAMES for m in ('no_adapt','current_ours',candidate)} for g in ['aggregate']+[f'block{i}' for i in range(4)]}
    assert consensus_advancement(metrics,True)['gate']['passed']
    assert not consensus_advancement(metrics,False)['gate']['passed']
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
        assert not consensus_advancement(v,True)['gate']['passed']


def test_support_diagnostics_keep_zero_denominators_and_both_count_conventions():
    from taisp.analysis.flip_consensus_study import support_summary
    rows=[]
    for i in range(2):
        for case in ('clean_s0','gamma_s1'):
            rows.append(dict(method='flip_consensus_ours',block=i,case=case,
                consensus=dict(original_eligible_count=30*i,current_top20_count=20*i,
                    flip_eligible_count=30*i,matched_count=25*i,retained_count=20*i,
                    retention_fraction=2/3 if i else None,zero_consensus=not i),
                consensus_setup_seconds=.03,adapt_seconds=.3,deploy_seconds=.36,phi3_norm=.02*i,updated=bool(i)))
    result=support_summary(rows)
    assert result['overall']['undefined_retention_count']==2
    assert result['overall']['mean_retention_fraction_defined']==2/3
    assert result['overall']['zero_consensus_fraction']==.5
    assert result['overall']['original_eligible_count']['mean']==15
    assert result['overall']['current_top20_count']['mean']==10
    assert result['block0']['mean_retention_fraction_defined'] is None
