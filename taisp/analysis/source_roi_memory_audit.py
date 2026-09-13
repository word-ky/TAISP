"""T026-A audit candidate: predicted-class retrieval from frozen clean memory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from .roi_equivariance import (sha,write,setup,images,common_jvp,isolated,state_hash,
    support_mask,mask_receipt,select_predictions,fixed_roi_representation)
from .roi_memory_ops import retrieve_anchors,tensor_sha


NAME='class_conditional_clean_roi_memory'


def candidate_cotangent(detector,image,boxes,labels,features,memory_classes):
    if not len(boxes):
        return torch.zeros_like(image),image.new_empty(0,1024),{'loss':0.,'per_object':[],
            'retrieval_ids':[],'similarities':[],'query_hashes':[],'anchor_hashes':[],
            'predicted_classes':[],'empty_support':True,'anchor_pairwise_mean_cosine':None,
            'anchor_pairwise_min_cosine':None}
    with torch.no_grad():
        query,_=fixed_roi_representation(detector,image,boxes)
        anchors,ids,sims=retrieve_anchors(query,labels,features,memory_classes)
    y=image.detach().requires_grad_(True)
    z,_=fixed_roi_representation(detector,y,boxes)
    per=1-(F.normalize(z,dim=-1,eps=1e-12)*anchors).sum(-1)
    loss=per.mean();c=torch.autograd.grad(loss,y)[0].detach()
    assert torch.isfinite(c).all() and torch.isfinite(loss)
    pair=anchors@anchors.T
    pairs=pair[torch.triu(torch.ones_like(pair,dtype=torch.bool),diagonal=1)]
    return c,anchors,{'loss':loss.item(),'per_object':per.detach().cpu().tolist(),
        'retrieval_ids':ids.cpu().tolist(),'similarities':sims.cpu().tolist(),
        'query_hashes':[tensor_sha(q) for q in query],'anchor_hashes':[tensor_sha(a) for a in anchors],
        'anchor_norms':anchors.double().norm(dim=-1).cpu().tolist(),'predicted_classes':labels.cpu().tolist(),
        'empty_support':False,'anchor_pairwise_mean_cosine':pairs.mean().item() if len(pairs) else None,
        'anchor_pairwise_min_cosine':pairs.min().item() if len(pairs) else None}


def run(manifest_path,memory_root,memory_lock,output):
    lock=json.loads(memory_lock.read_text())
    assert lock['memory_commit']
    for name,digest in lock['files'].items():assert sha(memory_root/name)==digest,name
    health=json.loads((memory_root/'health.json').read_text());assert health['passed'] and health['source_audit_overlap']==0
    memory=torch.load(memory_root/'memory.pt',map_location='cuda:0',weights_only=True)
    features,classes=memory['features'],memory['classes']
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    source,isp,env=setup()
    env.update({'candidate':NAME,'memory_commit':lock['memory_commit'],'memory_lock_sha256':sha(memory_lock),
        'memory_sha256':sha(memory_root/'memory.pt'),'image_manifest_sha256':sha(manifest_path),
        'source_assumption':'offline source-GT-derived frozen memory; audit path label-free'})
    write(output/'environment.json',env)
    rows=[]
    for index,(info,case,image) in enumerate(images(json.loads(manifest_path.read_text()))):
        with torch.no_grad():selected=select_predictions(source(image)[0],.5,20)
        mask,rectangles=support_mask(image,selected['boxes'])
        identity=isp(image,image.new_zeros(8)).detach()
        c,anchors,obj=candidate_cotangent(source,identity,selected['boxes'],selected['labels'],features,classes)
        refs,jacobian=common_jvp(image,isp,mask,{NAME:c})
        supports={k:v.cpu().tolist() for k,v in selected.items()}
        torch.save(anchors.cpu(),output/f'anchors_{index:02d}.pt')
        row={'episode_index':index,'image_id':info['image_id'],'case':case,'block':info['block'],
            'support_count':len(selected['boxes']),'supports':supports,
            'supports_sha256':hashlib.sha256(json.dumps(supports,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'mask':mask_receipt(mask,rectangles),'objectives':{NAME:obj},'gradients':refs,'jacobian':jacobian,
            'candidate_gradient_norm':torch.tensor(refs[NAME]['object'],dtype=torch.float64).norm().item(),
            'cotangent_l2':c.double().norm().item(),'cotangent_sha256':tensor_sha(c),
            'anchors_file_sha256':sha(output/f'anchors_{index:02d}.pt'),
            'isolation':isolated(source) and isp.phi.grad is None and not bool(isp.phi.any())}
        assert row['isolation']
        write(output/f'record_{index:02d}.json',row);rows.append(row)
        print(json.dumps({'stage':'candidate','episode':index,'supports':len(selected['boxes'])}),flush=True)
    assert len(rows)==48 and state_hash(source)==env['source_state_sha256']
    forbidden=[k for k in sys.modules if k.startswith('taisp.') and any(s in k for s in ('oracle','source_meta','common_jacobian','_reference'))]
    assert not forbidden,forbidden
    write(output/'records.json',rows)
    write(output/'completion.json',{'status':'candidate_complete','episodes':48,'source_hash_after':state_hash(source),
        'frozen_eval_grad_none':isolated(source),'forbidden_modules_loaded':forbidden,'seconds':time.perf_counter()-started})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--memory-root',type=Path,required=True);p.add_argument('--memory-lock',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    run(a.manifest,a.memory_root,a.memory_lock,a.output)
