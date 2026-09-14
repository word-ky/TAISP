"""R046 zero-GT scalar/multi-output/separate VJP attribution; analysis only."""
import argparse
import hashlib
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
from .pseudo_native_candidates import component_sum_check
from .roi_equivariance import setup,images,select_predictions,common_jvp,sha,write,isolated,state_hash


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def norm(a):return math.sqrt(math.fsum(x*x for x in a))


def compare(a,b):
    na,nb=norm(a),norm(b)
    return {'relative_l2':norm([x-y for x,y in zip(a,b)])/max(na,1e-12),
            'cosine':math.fsum(x*y for x,y in zip(a,b))/(na*nb) if na and nb else (1. if na==nb==0 else 0.)}


def tensor_compare(a,b):
    a,b=a.detach().double(),b.detach().double();na,nb=a.norm().item(),b.norm().item()
    return {'relative_l2':(a-b).norm().item()/max(na,1e-12),
            'cosine':(a*b).sum().item()/(na*nb) if na and nb else (1. if na==nb==0 else 0.)}


def dispersion(vectors):
    pairs=[compare(a,b) for i,a in enumerate(vectors) for j,b in enumerate(vectors) if i!=j]
    return {'max_pairwise_relative_l2':max(p['relative_l2'] for p in pairs),
            'min_pairwise_cosine':min(p['cosine'] for p in pairs)}


def rng_state(device):
    values={'cpu':torch.random.get_rng_state()}
    if device.type=='cuda':values['cuda']=torch.cuda.get_rng_state(device)
    return {k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in values.items()}


def graph_cotangents(total,parts,y):
    csum=torch.autograd.grad(total,y,retain_graph=True)[0].detach()
    outputs=tuple(parts[k] for k in KEYS)
    cmulti=torch.autograd.grad(outputs,y,grad_outputs=tuple(torch.ones_like(v) for v in outputs),retain_graph=True)[0].detach()
    canonical={k:torch.autograd.grad(parts[k],y,retain_graph=True)[0].detach() for k in KEYS}
    reverse={}
    for i,k in enumerate(reversed(KEYS)):
        reverse[k]=torch.autograd.grad(parts[k],y,retain_graph=i<len(KEYS)-1)[0].detach()
    sep=sum((canonical[k].double() for k in KEYS),torch.zeros_like(y,dtype=torch.float64))
    sep_rev=sum((reverse[k].double() for k in reversed(KEYS)),torch.zeros_like(y,dtype=torch.float64))
    return {'sum':csum,'multi':cmulti,'sep':sep,'sep_rev':sep_rev},canonical,reverse


def repetition(source,isp,image,hard,targets,output,episode,rep):
    before=state_hash(source);target_before=digest({k:v.cpu().tolist() for k,v in targets[0].items()})
    rng_before=rng_state(image.device)
    y=isp(image,image.new_zeros(8)).detach().requires_grad_(True)
    total,parts=pseudo_native_loss(source,y,targets)
    losses={k:parts[k].item() for k in KEYS};losses['total']=total.item()
    cotangents,canonical,reverse=graph_cotangents(total,parts,y)
    rng_graph=rng_state(image.device)
    allcs={**cotangents,**{'canonical_'+k:v for k,v in canonical.items()},**{'reverse_'+k:v for k,v in reverse.items()}}
    assert all(torch.isfinite(v).all() for v in allcs.values())
    refs,jac=common_jvp(image,isp,image.new_ones(1,1,*image.shape[-2:]),allcs)
    vectors={k:v['global'] for k,v in refs.items()}
    del allcs,canonical,reverse
    # Independently differentiate a fresh native forward all the way to phi.
    phi=image.new_zeros(8,requires_grad=True)
    fresh,fresh_parts=pseudo_native_loss(source,isp(image,phi),targets)
    direct=torch.autograd.grad(fresh,phi)[0].detach().cpu().tolist()
    rng_direct=rng_state(image.device)
    yh=isp(image,image.new_zeros(8)).detach().requires_grad_(True)
    lh=hard(image,yh);ch=torch.autograd.grad(lh,yh)[0].detach()
    hr,hj=common_jvp(image,isp,image.new_ones(1,1,*image.shape[-2:]),{'hard':ch})
    rng_after=rng_state(image.device);after=state_hash(source)
    old=component_sum_check({'native':{'gradient':vectors['sum']},**{k:{'gradient':vectors['canonical_'+k]} for k in KEYS}})
    data_path=output/f'cotangents_{episode:02d}_{rep:02d}.pt'
    torch.save({k:v.detach().cpu() for k,v in cotangents.items()},data_path)
    row={'episode_index':episode,'repetition':rep,'losses':losses,
        'fresh_direct_losses':{k:v.item() for k,v in fresh_parts.items()},'vectors':vectors,
        'g_phi_direct':direct,'g_hard':hr['hard']['global'],'hard_loss':lh.item(),
        'projected':{k:compare(vectors['sum'],vectors[k]) for k in ['multi','sep','sep_rev']},
        'image_space':{k:tensor_compare(cotangents['sum'],cotangents[k]) for k in ['multi','sep','sep_rev']},
        'reverse_order_relative':norm([a-b for a,b in zip(vectors['sep'],vectors['sep_rev'])])/max(norm(vectors['sum']),1e-12),
        'image_reverse_order_relative':(cotangents['sep']-cotangents['sep_rev']).norm().item()/max(cotangents['sum'].double().norm().item(),1e-12),
        'direct_phi':compare(vectors['sum'],direct),'old_additivity':old,'jacobian':jac,'hard_jacobian':hj,
        'rng':{'before':rng_before,'after_graph':rng_graph,'after_direct':rng_direct,'after':rng_after,
               'restored':rng_before==rng_graph==rng_direct==rng_after},
        'state_before':before,'state_after':after,'target_sha256':target_before,
        'target_unchanged':target_before==digest({k:v.cpu().tolist() for k,v in targets[0].items()}),
        'isolation':before==after and isolated(source) and isp.phi.grad is None and not bool(isp.phi.any()),
        'cotangent_file':data_path.name,'cotangent_sha256':sha(data_path)}
    assert all(math.isfinite(x) for v in vectors.values() for x in v) and all(math.isfinite(x) for x in direct)
    row['integrity_passed']=row['isolation'] and row['target_unchanged'] and row['rng']['restored']
    write(output/f'record_{episode:02d}_{rep:02d}.json',row)
    assert row['integrity_passed'],'State/RNG/target blocker'
    return row


def deterministic_probe(source,isp,image,targets):
    old=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    before=state_hash(source);rng_before=rng_state(image.device)
    result={'non_gating':True,'original_setting':old,'original_warn_only':warn}
    try:
        torch.use_deterministic_algorithms(True)
        phi=image.new_zeros(8,requires_grad=True)
        loss,_=pseudo_native_loss(source,isp(image,phi),targets)
        g=torch.autograd.grad(loss,phi)[0]
        result.update(succeeded=True,gradient=g.detach().cpu().tolist(),loss=loss.item())
    except Exception as error:
        result.update(succeeded=False,error_type=type(error).__name__,error=str(error))
    finally:
        torch.use_deterministic_algorithms(old,warn_only=warn)
    result.update(restored=torch.are_deterministic_algorithms_enabled()==old and torch.is_deterministic_algorithms_warn_only_enabled()==warn,
                  frozen_state_unchanged=state_hash(source)==before and isolated(source),rng_restored=rng_before==rng_state(image.device))
    return result


def summarize(episodes):
    rows=[]
    for episode in episodes:
        reps=episode['repetitions'];native=dispersion([r['vectors']['sum'] for r in reps]);hard=dispersion([r['g_hard'] for r in reps])
        med={k:statistics.median(r['projected'][k]['relative_l2'] for r in reps) for k in ['multi','sep','sep_rev']}
        row={'episode_index':episode['episode_index'],'image_id':episode['image_id'],'case':episode['case'],
            'native':native,'hard':hard,'median_projected_relative_l2':med,
            'median_image_relative_l2':{k:statistics.median(r['image_space'][k]['relative_l2'] for r in reps) for k in ['multi','sep','sep_rev']},
            'direct_phi_max_relative_l2':max(r['direct_phi']['relative_l2'] for r in reps),
            'direct_phi_min_cosine':min(r['direct_phi']['cosine'] for r in reps),
            'reverse_order_material_reps':sum(r['reverse_order_relative']>1e-5 for r in reps),
            'multi_no_worse':med['multi']<=med['sep'] and med['multi']<=med['sep_rev'],
            'integrity_passed':episode['support_match'] and all(r['integrity_passed'] for r in reps)}
        rows.append(row)
    flags={'native_cosine_all8':all(r['native']['min_pairwise_cosine']>=.99999 for r in rows),
        'native_dispersion_all8':all(r['native']['max_pairwise_relative_l2']<=.002 for r in rows),
        'direct_phi_parity_all8':all(r['direct_phi_max_relative_l2']<=1e-5 and r['direct_phi_min_cosine']>=.999999 for r in rows),
        'integrity_all8':len(rows)==8 and all(r['integrity_passed'] for r in rows),
        'multi_no_worse_atleast7':sum(r['multi_no_worse'] for r in rows)>=7,
        'multi_median_bound_all8':all(r['median_projected_relative_l2']['multi']<=.002 for r in rows)}
    return {'episodes':rows,'gates':flags,'passed':all(flags.values()),
            'status':'NEEDS_REVIEW' if all(flags.values()) else 'BLOCKED','GT_reference_authorized':False}


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
        if index==0:
            probe=deterministic_probe(source,isp,image,targets);write(output/'deterministic_probe.json',probe)
            assert probe['restored'] and probe['frozen_state_unchanged'] and probe['rng_restored']
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
