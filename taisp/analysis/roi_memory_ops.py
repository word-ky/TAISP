"""Pure frozen feature-memory operations; no annotations, models or oracle IO."""
import hashlib

import torch
from torch.nn import functional as F


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def memory_health(features,classes):
    unique,counts=classes.unique(sorted=True,return_counts=True)
    norms=features.double().norm(dim=-1)
    return {'entries':len(features),'dimension':features.shape[-1],
        'class_counts':{str(c):n for c,n in zip(unique.cpu().tolist(),counts.cpu().tolist())},
        'finite':bool(torch.isfinite(features).all()),'max_norm_error':(norms-1).abs().max().item(),
        'passed':features.shape==(1280,1024) and len(unique)==80 and bool((counts==16).all())
                 and bool(torch.isfinite(features).all()) and bool(((norms-1).abs()<=1e-5).all())}


def retrieve_anchors(query,predicted_classes,features,memory_classes):
    query=F.normalize(query.detach(),dim=-1,eps=1e-12)
    anchors,ids,similarities=[],[],[]
    for q,c in zip(query,predicted_classes):
        members=torch.where(memory_classes==c)[0]
        assert len(members)==16, 'Predicted class missing balanced memory; no fallback.'
        sim=features[members]@q
        selected=torch.argsort(sim,descending=True,stable=True)[:4]
        selected_ids=members[selected]
        anchor=F.normalize(features[selected_ids].mean(0),dim=0,eps=1e-12)
        assert torch.isfinite(anchor).all() and abs(anchor.double().norm().item()-1)<=1e-5
        anchors.append(anchor);ids.append(selected_ids);similarities.append(sim[selected])
    return torch.stack(anchors).detach(),torch.stack(ids).detach(),torch.stack(similarities).detach()
