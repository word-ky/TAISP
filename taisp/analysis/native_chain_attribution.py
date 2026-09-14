"""R047 zero-GT same-graph scalar self-noise and same-cotangent ISP parity."""
import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time
import torch
from taisp.losses.detector_native import DetectorNativeLoss
from .native_task_signal import KEYS,pseudo_targets,pseudo_native_loss
from .native_vjp_attribution import digest,rng_state,norm
from .pseudo_native_candidates import component_sum_check
from .roi_equivariance import setup,images,select_predictions,common_jvp,sha,write,isolated,state_hash


def compare(a,b):
    na,nb=norm(a),norm(b)
    return {'relative_l2':norm([x-y for x,y in zip(a,b)])/max(na,nb,1e-12),
            'cosine':math.fsum(x*y for x,y in zip(a,b))/(na*nb) if na and nb else (1. if na==nb==0 else 0.)}


def tensor_compare(a,b):
    a,b=a.detach().double(),b.detach().double();na,nb=a.norm().item(),b.norm().item()
    return {'relative_l2':(a-b).norm().item()/max(na,nb,1e-12),
            'cosine':(a*b).sum().item()/(na*nb) if na and nb else (1. if na==nb==0 else 0.)}


def dispersion(vectors):
    pairs=[compare(a,b) for a,b in itertools.combinations(vectors,2)]
    return {'max_pairwise_relative_l2':max(p['relative_l2'] for p in pairs),
            'min_pairwise_cosine':min(p['cosine'] for p in pairs)}


def graph_cotangents(total,parts,y):
    # Keep the ISP graph too: the final chain VJP uses this exact y and phi.
    cs={k:torch.autograd.grad(total,y,retain_graph=True)[0].detach() for k in ['sum_a','sum_b','sum_c']}
    outputs=tuple(parts[k] for k in KEYS)
    cs['multi']=torch.autograd.grad(outputs,y,grad_outputs=tuple(torch.ones_like(v) for v in outputs),retain_graph=True)[0].detach()
    canonical={k:torch.autograd.grad(parts[k],y,retain_graph=True)[0].detach() for k in KEYS}
    reverse={k:torch.autograd.grad(parts[k],y,retain_graph=True)[0].detach() for k in reversed(KEYS)}
    cs['sep']=sum((canonical[k].double() for k in KEYS),torch.zeros_like(y,dtype=torch.float64))
    cs['rev']=sum((reverse[k].double() for k in reversed(KEYS)),torch.zeros_like(y,dtype=torch.float64))
    return cs,canonical,reverse


def chain_parity(g,direct,cotangent):
    result=compare(g,direct)
    result['finite']=all(math.isfinite(v) for v in g+direct)
    result['nonzero_required']=bool(cotangent.count_nonzero())
    result['nonzero_ok']=not result['nonzero_required'] or (norm(g)>0 and norm(direct)>0)
    result['passed']=result['finite'] and result['nonzero_ok'] and result['relative_l2']<=1e-5 and result['cosine']>=.999999
    return result


def repetition(source,isp,image,hard,targets,output,episode,rep):
    before=state_hash(source);target_before=digest({k:v.cpu().tolist() for k,v in targets[0].items()})
    rng_before=rng_state(image.device)
    phi=image.new_zeros(8,requires_grad=True);y=isp(image,phi)
    total,parts=pseudo_native_loss(source,y,targets)
    losses={k:parts[k].item() for k in KEYS};losses['total']=total.item()
    cs,canonical,reverse=graph_cotangents(total,parts,y)
    rng_graph=rng_state(image.device)
    allcs={**cs,**{'canonical_'+k:v for k,v in canonical.items()},**{'reverse_'+k:v for k,v in reverse.items()}}
    assert all(torch.isfinite(v).all() for v in allcs.values())
    refs,jac=common_jvp(image,isp,image.new_ones(1,1,*image.shape[-2:]),allcs)
    vectors={k:v['global'] for k,v in refs.items()}
    # No detector call: the very same detached c_sum_a contracts the original ISP graph.
    direct=torch.autograd.grad(y,phi,grad_outputs=cs['sum_a'])[0].detach().cpu().tolist()
    parity=chain_parity(vectors['sum_a'],direct,cs['sum_a'])
    scalar_pairs={a+'_'+b:tensor_compare(cs[a],cs[b]) for a,b in itertools.combinations(['sum_a','sum_b','sum_c'],2)}
    rng_after=rng_state(image.device);after=state_hash(source)
    old=component_sum_check({'native':{'gradient':vectors['sum_a']},**{k:{'gradient':vectors['canonical_'+k]} for k in KEYS}})
    data_path=output/f'cotangents_{episode:02d}_{rep:02d}.pt'
    torch.save({k:v.cpu() for k,v in cs.items()},data_path)
    row={'episode_index':episode,'repetition':rep,'losses':losses,'vectors':vectors,
        'g_native':vectors['sum_a'],'g_isp_vjp':direct,'chain_parity':parity,
        'scalar_pairs':scalar_pairs,'scalar_self_noise':statistics.median(v['relative_l2'] for v in scalar_pairs.values()),
        'scalar_min_cosine':min(v['cosine'] for v in scalar_pairs.values()),
        'image_space':{k:tensor_compare(cs['sum_a'],cs[k]) for k in ['multi','sep','rev']},
        'projected':{k:compare(vectors['sum_a'],vectors[k]) for k in ['multi','sep','rev']},
        'cotangent_receipts':{k:{'norm':v.double().norm().item(),'dtype':str(v.dtype),'shape':list(v.shape),
            'sha256':__import__('hashlib').sha256(v.cpu().contiguous().numpy().tobytes()).hexdigest()} for k,v in cs.items()},
        'old_additivity':old,'jacobian':jac,'rng':{'before':rng_before,'after_graph':rng_graph,'after':rng_after,
            'restored':rng_before==rng_graph==rng_after},'state_before':before,'state_after':after,
        'target_sha256':target_before,'target_unchanged':target_before==digest({k:v.cpu().tolist() for k,v in targets[0].items()}),
        'isolation':before==after and isolated(source) and isp.phi.grad is None and not bool(isp.phi.any()),
        'cotangent_file':data_path.name,'cotangent_sha256':sha(data_path)}
    assert all(math.isfinite(v) for g in vectors.values() for v in g) and all(math.isfinite(v) for v in losses.values())
    row['integrity_passed']=row['isolation'] and row['target_unchanged'] and row['rng']['restored']
    write(output/f'record_{episode:02d}_{rep:02d}.json',row)
    assert row['integrity_passed'],'State/RNG/target blocker'
    return row


def summarize(episodes):
    rows=[]
    for e in episodes:
        reps=e['repetitions'];n=statistics.median(r['scalar_self_noise'] for r in reps)
        forms={k:statistics.median(r['image_space'][k]['relative_l2'] for r in reps) for k in ['multi','sep','rev']}
        native=dispersion([r['g_native'] for r in reps])
        rows.append({'episode_index':e['episode_index'],'image_id':e['image_id'],'case':e['case'],'native':native,
            'gate1':native['min_pairwise_cosine']>=.99999 and native['max_pairwise_relative_l2']<=.002,
            'gate2':all(r['chain_parity']['passed'] for r in reps),
            'max_chain_relative':max(r['chain_parity']['relative_l2'] for r in reps),
            'min_chain_cosine':min(r['chain_parity']['cosine'] for r in reps),
            'scalar_self_noise_median':n,'scalar_min_cosine':min(r['scalar_min_cosine'] for r in reps),
            'gate3':all(math.isfinite(r['scalar_self_noise']) and r['scalar_min_cosine']>=.99999 for r in reps) and n<=.002,
            'formulation_medians':forms,'noise_bound':2*n+1e-5,'noise_explained':all(v<=2*n+1e-5 for v in forms.values()),
            'max_formulation_relative':max(r['image_space'][k]['relative_l2'] for r in reps for k in forms),
            'integrity_passed':len(reps)==5 and e['support_match'] and all(r['integrity_passed'] for r in reps)})
    gates={f'gate{i}':len(rows)==8 and all(r[f'gate{i}'] for r in rows) for i in [1,2,3]}
    gates['gate4']=len(rows)==8 and sum(r['noise_explained'] for r in rows)>=7 and all(r['max_formulation_relative']<=.002 and r['integrity_passed'] for r in rows)
    return {'episodes':rows,'gates':gates,'passed':all(gates.values()),'first_failed_gate':next((k for k,v in gates.items() if not v),None),
        'status':'NEEDS_REVIEW' if all(gates.values()) else 'BLOCKED','GT_reference_authorized':False}


def run(manifest_path,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();source,isp,env=setup()
    env.update(manifest_sha256=sha(manifest_path),native_sampling_seed=20260930,repetitions=5)
    write(output/'environment.json',env);episodes=[]
    for index,(info,case,image) in enumerate(images(json.loads(manifest_path.read_text()))):
        with torch.no_grad():support=select_predictions(source(image)[0],.5,20)
        hard=DetectorNativeLoss(source,{'base':support},'det_pseudo');targets=pseudo_targets(hard)
        supports={k:v.cpu().tolist() for k,v in support.items()}
        episode={'episode_index':index,'image_id':info['image_id'],'case':case,'supports':supports,
            'supports_sha256':digest(supports),'pseudo_targets':{k:v.cpu().tolist() for k,v in targets[0].items()},
            'support_match':torch.equal(hard.boxes,targets[0]['boxes']) and torch.equal(hard.labels,targets[0]['labels']),
            'repetitions':[]}
        for rep in range(5):
            episode['repetitions'].append(repetition(source,isp,image,hard,targets,output,index,rep))
            print(json.dumps({'episode':index,'repetition':rep,'stage':'diagnostic'}),flush=True)
        write(output/f'episode_{index:02d}.json',episode);episodes.append(episode)
    forbidden=[k for k in sys.modules if k.startswith('taisp.') and any(x in k for x in ('oracle','reference','source_meta','clip_semantic','memory'))]
    assert not forbidden and len(episodes)==8 and state_hash(source)==env['source_state_sha256']
    summary=summarize(episodes);write(output/'summary.json',summary)
    write(output/'completion.json',{'status':summary['status'],'episodes':8,'repetitions':40,'seconds':time.perf_counter()-started,
        'source_hash_after':state_hash(source),'forbidden_modules_loaded':forbidden,'AP_calls':0,'GT_loaded':False})
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.manifest,a.output)
