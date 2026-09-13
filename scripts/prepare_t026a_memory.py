"""Select balanced clean-source instances using the precommitted T026-A order."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def select_instances(annotations, categories, available, excluded):
    result=[]
    for category in sorted(categories):
        eligible=[a for a in annotations if a['category_id']==category and a['image_id'] in available
                  and a['image_id'] not in excluded and not a.get('iscrowd',0)
                  and a['bbox'][2]>0 and a['bbox'][3]>0 and a.get('area',1)>0]
        key=lambda a:(hashlib.sha256(f"T026A:20260926:{a['id']}".encode()).hexdigest(),a['id'])
        chosen=sorted(eligible,key=key)[:16]
        assert len(chosen)==16,(category,len(chosen))
        result.extend({**a,'class_rank':rank,'selection_hash':key(a)[0]} for rank,a in enumerate(chosen))
    return result


def main(cohort,ledger,output):
    read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    c,l=read(cohort),read(ledger)
    annpath=Path(c['annotation_file']);assert sha(annpath)==c['annotation_sha256']
    data=read(annpath);infos={i['id']:i for i in data['images']}
    images=Path(c['images'][0]['path']).parent
    excluded=set(l['memory_excluded_ids'])
    available={i for i,v in infos.items() if i not in excluded and os.access(images/v['file_name'],os.R_OK)
               and (images/v['file_name']).is_file()}
    chosen=select_instances(data['annotations'],[c['id'] for c in data['categories']],available,excluded)
    rows=[]
    for index,a in enumerate(chosen):
        info=infos[a['image_id']];p=images/info['file_name'];x,y,w,h=a['bbox']
        rows.append({'entry_id':index,'class_id':a['category_id'],'annotation_id':a['id'],
            'image_id':a['image_id'],'path':str(p),'jpeg_sha256':sha(p),'box':[x,y,x+w,y+h],
            'class_rank':a['class_rank'],'selection_hash':a['selection_hash']})
    v={'entries':rows,'image_ids':sorted({r['image_id'] for r in rows}),'categories':sorted(c['id'] for c in data['categories']),
       'cohort_sha256':sha(cohort),'exclusion_ledger_sha256':sha(ledger),'annotation_sha256':sha(annpath),
       'source_audit_overlap':len({r['image_id'] for r in rows}&set(c['image_ids'])),
       'source_excluded_overlap':len({r['image_id'] for r in rows}&excluded)}
    assert len(rows)==1280 and v['source_audit_overlap']==v['source_excluded_overlap']==0
    output.write_text(json.dumps(v,indent=2)+'\n')
    print(json.dumps({'entries':len(rows),'images':len(v['image_ids']),'manifest_sha256':sha(output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True);p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    main(a.cohort,a.ledger,a.output)
