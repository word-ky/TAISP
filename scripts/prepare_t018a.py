"""Precommit 100 disjoint, readable train2017 images; no model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from prepare_t013b import sha256


def prepare(project, output, *, count=100, block_size=25, seed=20260918, additional_source=None):
    hpath=project/'research_log/T013H_train_cohort.json'
    apath=project/'research_log/T014A_source_manifest.json'
    ledgerpath=project/'research_log/T013H/exclusion_ledger.json'
    h, a, ledger=[json.loads(p.read_text()) for p in (hpath,apath,ledgerpath)]
    annotations, val=Path(h['annotation_file']),Path(h['val_annotation_file'])
    images=Path(h['images_root'])
    assert sha256(annotations)==h['annotation_sha256']
    assert sha256(val)==h['val_annotation_sha256']
    val_ids={r['id'] for r in json.loads(val.read_text())['images']}
    prior=set(h['image_ids'])|set(h['prior_source_ids'])|set(a['image_ids'])|set(ledger['prior_source_ids'])
    provenance=[hpath,apath,ledgerpath]
    if additional_source is not None:
        path=project/additional_source
        prior.update(json.loads(path.read_text())['image_ids'])
        provenance.append(path)
    excluded=prior|val_ids|set(ledger['evaluation_ids'])
    data=json.loads(annotations.read_text())
    valid={r['image_id'] for r in data['annotations'] if not r.get('iscrowd',0)
           and r['bbox'][2]>0 and r['bbox'][3]>0 and r.get('area',r['bbox'][2]*r['bbox'][3])>0}
    infos={r['id']:r for r in data['images']}
    available=[i for i in valid if os.access(images/infos[i]['file_name'],os.R_OK)
               and (images/infos[i]['file_name']).is_file()]
    eligible=[i for i in available if i not in excluded]
    key=lambda i:(hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest(),i)
    ids=sorted(eligible,key=key)[:count]
    assert len(ids)==count and not set(ids)&excluded
    records=[{'image_id':i,'path':str(images/infos[i]['file_name']),'file_name':infos[i]['file_name'],
              'sha256':sha256(images/infos[i]['file_name']),'width':infos[i]['width'],'height':infos[i]['height'],
              'block':j//block_size,'selection_hash':key(i)[0]} for j,i in enumerate(ids)]
    result={'split':'train2017','seed':seed,'image_ids':ids,'images':records,
            'replication_blocks':{f'block{i}':ids[block_size*i:block_size*(i+1)] for i in range(count//block_size)},
            'annotation_file':str(annotations),'annotation_sha256':sha256(annotations),
            'val_annotation_sha256':sha256(val),'prior_source_ids':sorted(prior),'val_excluded_count':len(val_ids),
            'evaluation_excluded_count':len(ledger['evaluation_ids']),'overlap_prior_source':0,'overlap_val':0,
            'available_valid_count':len(available),'eligible_count':len(eligible),
            'provenance_sha256':{str(p.relative_to(project)):sha256(p) for p in provenance},
            'selection':f"first{count} sorted by (sha256('{seed}:'+decimalID),ID) after frozen exclusions",
            'episode_order':'selected image order; gamma_s1/s2,contrast_s1/s2,color_cast_s1/s2,clean_s0'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('image_ids','eligible_count','prior_source_ids')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=100)
    p.add_argument('--block-size',type=int,default=25)
    p.add_argument('--seed',type=int,default=20260918)
    p.add_argument('--additional-source',type=Path)
    args=p.parse_args()
    prepare(args.project,args.output,count=args.count,block_size=args.block_size,seed=args.seed,
            additional_source=args.additional_source)
