"""R050 router-locked train references; two experts locked before holdout."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .orthogonal_candidates import rng_state
from .orthogonal_reference import verify_inputs
from .orthogonal_transport import fit_vectors,evaluate,summarize,describe,coordinate_stats
from .gradient_regime_router import assign
from .roi_equivariance import setup,images,sha,write,common_jvp,state_hash,isolated


def locked_inputs(cohort_path,candidate_root,lock_path,router_path,router_lock_path,phase,map_root,map_lock_path):
    # Reuse the complete R049 candidate/cohort checks; this computes no gradients.
    cs,train_manifest,_,preflight=verify_inputs(cohort_path,candidate_root,lock_path,'train')
    rl=json.loads(router_lock_path.read_text());router=json.loads(router_path.read_text())
    assert rl['router_commit'] and sha(router_path)==rl['router_sha256']
    assert router['candidate_lock_sha256']==rl['candidate_lock_sha256']==sha(lock_path)
    assert router['status']=='router_complete' and min(router['counts'])>=72
    assert router['task_gradients_computed']==0 and not router['annotations_loaded']
    assert router['train_episode_indices']==[r['episode_index'] for r in cs]
    assert len(router['assignments'])==360
    assert [router['assignments'].count(k) for k in range(2)]==router['counts']
    maps=None;routed=None
    if phase=='train':
        manifest=train_manifest;routes=router['assignments']
    else:
        ml=json.loads(map_lock_path.read_text());assert ml['map_commit']
        assert ml['router_lock_sha256']==sha(router_lock_path)
        for f,h in ml['files'].items():assert sha(map_root/f)==h,f
        maps=json.loads((map_root/'maps.json').read_text());tr=json.loads((map_root/'records.json').read_text())
        assert maps['router_sha256']==sha(router_path) and maps['train_reference_sha256']==sha(map_root/'records.json')
        assert maps['holdout_task_gradients_computed']==0 and maps['svd_calls']==2
        assert len(tr)==360 and all(r['partition']=='train' for r in tr)
        assert [r['route'] for r in tr]==router['assignments']
        assert {r['image_id'] for r in tr}=={im['image_id'] for im in train_manifest['images']}
        full=json.loads(cohort_path.read_text());all_cs=json.loads((candidate_root/'records.json').read_text())
        chosen=[(im,cc) for im,cc in zip(full['images'],full['corrupted_cases']) if im['partition']=='holdout']
        manifest={**full,'images':[im for im,cc in chosen],'corrupted_cases':[cc for im,cc in chosen]}
        cs=[r for r in all_cs if r['partition']=='holdout']
        routes=assign([r['g_hard'] for r in cs],router['centroids']).tolist()
        routed=[{'episode_index':r['episode_index'],'route':k,
                 'g_route':(np.asarray(maps['experts'][k]['R'])@np.asarray(r['g_hard'])).tolist()}
                for r,k in zip(cs,routes)]
        preflight['map_lock']={'map_commit':ml['map_commit'],'maps_sha256':sha(map_root/'maps.json'),
                               'map_lock_sha256':sha(map_lock_path)}
    preflight.update(phase=phase,router_commit=rl['router_commit'],router_sha256=sha(router_path),
                     router_lock_sha256=sha(router_lock_path),routes_fixed_before_oracle=True)
    return cs,manifest,routes,maps,routed,preflight


def summary(rows):
    s=summarize(rows)
    s['gates']['S_route_overall']=s['gates'].pop('S_cal_overall')
    s['gates']['S_route_corrupt']=s['gates'].pop('S_cal_corrupt')
    s['gates']['holdout_route_coverage']=all(sum(r['route']==k for r in rows)>=12 for k in range(2))
    s['passed']=all(s['gates'].values())
    s['disposition']='two_regime_capacity_pending_review' if s['passed'] else 'close_exact_K2_two_O8_family'
    s['clusters']={}
    for k in range(2):
        rs=[r for r in rows if r['route']==k]
        s['clusters'][str(k)]={'n':len(rs),'distributions':{key:describe([r[key] for r in rs]) for key in
            ['S_hard','S_route','Delta','cos_hard_task','cos_cal_task','norm_hard','norm_cal']},
            'coordinates':{key:coordinate_stats(rs,key) for key in ['g_hard','g_route']} if rs else {}}
    groups={'overall':rows,'clean':[r for r in rows if r['case']=='clean_s0'],
            'corrupt':[r for r in rows if r['case']!='clean_s0'],
            **{c:[r for r in rows if r['case']==c] for c in sorted({r['case'] for r in rows})},
            **{'block'+str(k):[r for r in rows if r['block']==k] for k in range(4)}}
    s['route_counts']={g:[sum(r['route']==k for r in rs) for k in range(2)] for g,rs in groups.items()}
    return s


def run(cohort_path,candidate_root,lock_path,router_path,router_lock_path,phase,output,map_root=None,map_lock_path=None):
    candidates,manifest,routes,maps,routed,preflight=locked_inputs(cohort_path,candidate_root,lock_path,
        router_path,router_lock_path,phase,map_root,map_lock_path)
    output.mkdir(parents=True,exist_ok=False);write(output/'preflight.json',preflight)
    if routed is not None:write(output/'holdout_routes_before_oracle.json',routed)
    from .oracle import detector_task_loss
    from .common_jacobian import reference_checks
    # R049 task-gradient loop is inserted unchanged except route/score receipts.

    started=time.perf_counter();source,isp,env=setup()
    assert env['source_state_sha256']==preflight['source_state_sha256']
    env.update(phase=phase,candidate_lock_sha256=sha(lock_path),cohort_sha256=sha(cohort_path),map_lock=preflight['map_lock'])
    annotation=manifest['partition_annotations'][phase];assert sha(annotation['path'])==annotation['sha256']
    data=json.loads(Path(annotation['path']).read_text());ids={im['image_id'] for im in manifest['images']}
    assert {im['id'] for im in data['images']}==ids==set(annotation['image_ids'])
    assert all(a['image_id'] in ids for a in data['annotations'])
    env['annotation_path']=annotation['path'];env['annotation_sha256']=annotation['sha256'];write(output/'environment.json',env)
    anns={i:[] for i in ids}
    for a in data['annotations']:
        if not a.get('iscrowd',0) and a['bbox'][2]>0 and a['bbox'][3]>0 and a.get('area',1)>0:anns[a['image_id']].append(a)
    del data
    rows=[]
    for index,(info,case,image) in enumerate(images(manifest)):
        c=candidates[index];assert (c['image_id'],c['case'])==(info['image_id'],case)
        aa=anns[info['image_id']];boxes=[[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in aa]
        targets=[{'boxes':image.new_tensor(boxes).reshape(-1,4),'labels':torch.tensor([a['category_id'] for a in aa],device=image.device)}]
        before=state_hash(source);rb=rng_state(image.device)
        y=isp(image,image.new_zeros(8)).detach().requires_grad_(True)
        loss,parts=detector_task_loss(source,y,targets,seed=20260913);ct=torch.autograd.grad(loss,y)[0].detach()
        assert torch.isfinite(loss) and torch.isfinite(ct).all()
        refs,jac=common_jvp(image,isp,image.new_ones(1,1,*image.shape[-2:]),{'task':ct})
        r=refs['task'];t=r['global'];phi=image.new_zeros(8,requires_grad=True)
        direct=torch.autograd.grad((isp(image,phi)*ct).sum(),phi)[0].cpu().tolist();checks=reference_checks(t,r['object'],r['background'],direct)
        ra=rng_state(image.device);after=state_hash(source)
        row={'episode_index':c['episode_index'],'image_id':info['image_id'],'case':case,'block':info['block'],'partition':phase,
            'g_hard':c['g_hard'],'task_gradient':t,'task_loss':loss.item(),'task_components':{k:v.item() for k,v in parts.items()},
            'jacobian':jac,'reference_checks':checks,'state_before':before,'state_after':after,'rng':{'before':rb,'after':ra,'restored':rb==ra},
            'candidate_recomputed':False,'integrity_passed':before==after==env['source_state_sha256'] and rb==ra and isolated(source)
                and isp.phi.grad is None and not bool(isp.phi.any()) and all(v['passed'] for v in checks.values()) and all(math.isfinite(v) for v in t)}
        row['route']=routes[index]
        if phase=='holdout':
            ev=evaluate(t,c['g_hard'],maps['experts'][routes[index]]['R'])
            assert ev['g_cal']==routed[index]['g_route']
            row.update(ev);row.update(g_route=ev['g_cal'],S_route=ev['S_cal'])
        write(output/f'record_{index:03d}.json',row);assert row['integrity_passed'],'Reference integrity blocker'
        rows.append(row);print(json.dumps({'stage':phase,'episode':index}),flush=True)
    assert len(rows)==(360 if phase=='train' else 120) and sha(candidate_root/'records.json')==preflight['candidate_records_sha256']
    write(output/'records.json',rows)
    if phase=='train':
        experts=[]
        for k in range(2):
            rs=[r for r in rows if r['route']==k]
            experts.append(fit_vectors([r['g_hard'] for r in rs],[r['task_gradient'] for r in rs]))
        write(output/'maps.json',{'experts':experts,'svd_calls':2,'router_sha256':sha(router_path),
            'router_lock_sha256':sha(router_lock_path),'train_reference_sha256':sha(output/'records.json'),
            'candidate_lock_sha256':sha(lock_path),'holdout_task_gradients_computed':0,
            'source_revision':env['source_revision'],'code_sha256':sha(Path(__file__)),
            'fit_code_sha256':sha(Path(__file__).with_name('orthogonal_transport.py'))})
    else:write(output/'summary.json',summary(rows))
    write(output/'completion.json',{'status':'two_maps_complete' if phase=='train' else 'NEEDS_REVIEW',
        'episodes':len(rows),'phase':phase,'seconds':time.perf_counter()-started,
        'source_hash_after':state_hash(source),'candidate_recomputed':False,'AP_calls':0})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cohort','candidate-root','candidate-lock','router','router-lock','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=['train','holdout'],required=True)
    p.add_argument('--map-root',type=Path);p.add_argument('--map-lock',type=Path);a=p.parse_args()
    run(a.cohort,a.candidate_root,a.candidate_lock,a.router,a.router_lock,a.phase,a.output,a.map_root,a.map_lock)
