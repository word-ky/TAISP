"""T026-A offline source-label-derived clean memory, never audit annotations."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from .roi_equivariance import sha,write,setup,state_hash,isolated,fixed_roi_representation
from .roi_memory_ops import tensor_sha,memory_health


def run(manifest_path,output):
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();source,isp,env=setup()
    manifest=json.loads(manifest_path.read_text())
    env.update({'memory_manifest_sha256':sha(manifest_path),'source_labels':'GT ROI/class used for offline frozen memory only'})
    write(output/'environment.json',env)
    entries=manifest['entries'];features=torch.empty(1280,1024,device='cuda:0')
    with torch.no_grad():
        for j,i in enumerate(manifest['image_ids']):
            chosen=[e for e in entries if e['image_id']==i]
            p=Path(chosen[0]['path']);assert sha(p)==chosen[0]['jpeg_sha256']
            with Image.open(p) as im:image=pil_to_tensor(im.convert('RGB')).float().div(255).unsqueeze(0).cuda()
            boxes=image.new_tensor([e['box'] for e in chosen])
            z,_=fixed_roi_representation(source,image,boxes)
            assert torch.isfinite(z).all() and (z.norm(dim=-1)>1e-12).all()
            features[[e['entry_id'] for e in chosen]]=F.normalize(z,dim=-1,eps=1e-12)
            if j%100==0:print(json.dumps({'stage':'memory','images_done':j+1,'images_total':len(manifest['image_ids'])}),flush=True)
    classes=torch.tensor([e['class_id'] for e in entries],device='cuda:0')
    health=memory_health(features,classes)
    health.update({'source_audit_overlap':manifest['source_audit_overlap'],'source_excluded_overlap':manifest['source_excluded_overlap']})
    assert health['passed'] and health['source_audit_overlap']==health['source_excluded_overlap']==0
    assert isolated(source) and state_hash(source)==env['source_state_sha256']
    torch.save({'features':features.cpu(),'classes':classes.cpu()},output/'memory.pt')
    write(output/'entries.json',[{**e,'feature_sha256':tensor_sha(features[e['entry_id']])} for e in entries])
    write(output/'health.json',health)
    write(output/'completion.json',{'status':'memory_complete','entries':1280,'source_images':len(manifest['image_ids']),
        'source_hash_after':state_hash(source),'frozen_eval_grad_none':isolated(source),'seconds':time.perf_counter()-started})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.manifest,a.output)
