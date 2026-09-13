"""R033: fixed flip stability filtering, original confidence weights preserved."""
import math

import torch
from torchvision.ops import box_iou

from .detector_signal import flip_boxes


@torch.no_grad()
def flip_consensus(original, flipped, width):
    original_indices = torch.where(original['scores'] >= .50)[0]
    flip_indices = torch.where(flipped['scores'] >= .50)[0]
    a = {k:original[k][original_indices] for k in ('boxes','scores','labels')}
    b = {k:flipped[k][flip_indices] for k in ('boxes','scores','labels')}
    mapped = flip_boxes(b['boxes'],width)
    ious = box_iou(a['boxes'],mapped)
    eligible = torch.nonzero((a['labels'][:,None]==b['labels'][None,:]) & (ious>=.60)).cpu().tolist()
    oi,fi = original_indices.cpu().tolist(),flip_indices.cpu().tolist()
    sa,sb = a['scores'].cpu().tolist(),b['scores'].cpu().tolist()
    iou_values = ious.cpu().tolist()
    candidates = [(math.sqrt(sa[i]*sb[j]),oi[i],fi[j],i,j) for i,j in eligible]
    candidates.sort(key=lambda p:(-p[0],p[1],p[2]))
    used_a,used_b,matches = set(),set(),[]
    for confidence,original_index,flip_index,i,j in candidates:
        if original_index in used_a or flip_index in used_b:
            continue
        used_a.add(original_index);used_b.add(flip_index)
        matches.append(dict(original_index=original_index,flip_index=flip_index,
                            geometric_confidence=confidence,iou=iou_values[i][j]))
    retained = matches[:20]
    selected = torch.tensor([p['original_index'] for p in retained],device=original['boxes'].device,dtype=torch.long)
    support = {k:original[k][selected].detach().clone() for k in ('boxes','scores','labels')}
    receipt = dict(original_indices=oi,flip_indices=fi,
        original_eligible={k:v.cpu().tolist() for k,v in a.items()},
        flip_eligible={k:v.cpu().tolist() for k,v in b.items()},mapped_flip_boxes=mapped.cpu().tolist(),
        eligible_pair_count=len(candidates),matched_count=len(matches),matches=matches,retained_pairs=retained,
        original_eligible_count=len(oi),flip_eligible_count=len(fi),retained_count=len(retained),
        retention_fraction=len(retained)/len(oi) if oi else None,zero_consensus=len(retained)==0)
    return support,receipt
