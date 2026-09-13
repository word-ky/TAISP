"""T017-A frozen native task-component differential attribution on A6000."""
import argparse
import hashlib
import itertools
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from taisp import DifferentiableISP
from taisp.models.detector import load_detector
from .common_jacobian import common_reference, reference_checks
from .deterministic_replay import state_hash
from .differential_subspace import cosine, decompose, describe, write_json
from .oracle import detector_task_loss
from .source_meta_smoke import load_episodes, frozen_unchanged
from .spatial_action import EPS, support_mask

KEYS = ('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg')
REGIONS = ('global', 'object', 'background')


def component_reference(image, isp, mask, loss_fn):
    phi = image.new_zeros(8, requires_grad=True)
    y = isp(image, phi)
    total, losses = loss_fn(y)
    assert set(losses) == set(KEYS), f'Unexpected component keys: {list(losses)}'
    cotangents = {k: torch.autograd.grad(losses[k], y, retain_graph=True)[0].detach() for k in KEYS}
    cotangents['total'], direct = torch.autograd.grad(total, (y, phi))
    assert all(torch.isfinite(v).all() for v in cotangents.values())
    refs, jacobian = common_reference(image, isp, mask, cotangents)
    return {'losses': {k: float(v.detach()) for k, v in losses.items()}, 'total_loss': float(total.detach()),
            'references': refs, 'direct_global': direct.detach().cpu().tolist(), 'jacobian': jacobian}


def sum_check(parts, total, atol=1e-12, rtol=1e-10):
    a, total = np.asarray(parts, dtype=np.float64), np.asarray(total, dtype=np.float64)
    summed = a.sum(axis=0)
    error, bound = np.abs(summed-total), atol + rtol*np.abs(a).sum(axis=0)
    return {'sum': summed.tolist(), 'reference': total.tolist(), 'errors': error.tolist(), 'bounds': bound.tolist(),
            'max_absolute_error': float(np.max(error)), 'max_error_over_bound': float(np.max(error/bound)),
            'passed': bool(np.all(error <= bound))}


def parity(value, reference):
    return reference_checks(reference, reference, np.zeros_like(reference), value)['parity']


def numerical_checks(refs, direct, saved):
    closure = {r: sum_check([refs[k][r] for k in KEYS], refs['total'][r]) for r in REGIONS}
    partition = {k: reference_checks(v['global'], v['object'], v['background'], v['global'])['partition']
                 for k, v in refs.items()}
    inherited = {}
    for r in REGIONS:
        inherited[r] = {'component_sum': parity(closure[r]['sum'], saved[r]),
                        'independent_total': parity(refs['total'][r], saved[r])}
    inherited['regional_concat'] = {
        'component_sum': parity(np.r_[closure['object']['sum'], closure['background']['sum']],
                                np.r_[saved['object'], saved['background']]),
        'independent_total': parity(np.r_[refs['total']['object'], refs['total']['background']],
                                    np.r_[saved['object'], saved['background']])}
    direct_check = parity(direct, refs['total']['global'])
    return {'component_closure': closure, 'partitions': partition, 'saved_A1_parity': inherited,
            'direct_global_parity': direct_check,
            'passed': all(c['passed'] for c in closure.values()) and all(c['passed'] for c in partition.values())
                      and all(c['passed'] for r in inherited.values() for c in r.values()) and direct_check['passed']}


def attribution(refs, pseudo):
    parts = {k: decompose(refs[k]['object'], refs[k]['background'])[0] for k in KEYS}
    ds = {k: np.asarray(v['d']) for k, v in parts.items()}
    total = sum(ds.values())
    ds['loc'] = ds['loss_box_reg'] + ds['loss_rpn_box_reg']
    ds['conf'] = ds['loss_classifier'] + ds['loss_objectness']
    energy, norm = float(total@total), float(np.linalg.norm(total))
    dp, pnorm = np.asarray(pseudo['pseudo']['d']), pseudo['pseudo']['norm']
    metrics, components = {}, {}
    for k, d in ds.items():
        a = float(total@d)/(energy+EPS)
        c = 2*float(d@dp)/(pnorm+EPS)
        metrics.update({k+'/A': a, k+'/norm_ratio': float(np.linalg.norm(d))/(norm+EPS),
                        k+'/norm': float(np.linalg.norm(d)), k+'/pseudo_cosine': cosine(dp, d), k+'/C_diff': c})
        components[k] = {'d': d.tolist(), 's': parts[k]['s'] if k in parts else
                         (np.asarray(parts['loss_box_reg']['s'])+parts['loss_rpn_box_reg']['s']).tolist() if k=='loc' else
                         (np.asarray(parts['loss_classifier']['s'])+parts['loss_objectness']['s']).tolist()}
    for a, b in itertools.combinations(KEYS, 2):
        metrics[f'cosine/{a}/{b}'] = cosine(ds[a], ds[b])
    current_c = 2*float(total@dp)/(pnorm+EPS)
    saved_c = pseudo['metrics']['C_diff']
    metrics.update({'total_diff_norm': norm, 'C_diff_current': current_c, 'C_diff_saved': saved_c})
    ac = [metrics[k+'/A'] for k in KEYS]
    ag = [metrics[k+'/A'] for k in ('loc', 'conf')]
    eps_deficit = EPS/(energy+EPS)
    checks = {'A_components': sum_check(ac, 1., atol=eps_deficit+1e-12),
              'A_groups': sum_check(ag, 1., atol=eps_deficit+1e-12),
              'C_components': sum_check([metrics[k+'/C_diff'] for k in KEYS], current_c),
              'C_groups': sum_check([metrics[k+'/C_diff'] for k in ('loc', 'conf')], current_c),
              'C_saved': sum_check([metrics[k+'/C_diff'] for k in KEYS], saved_c, atol=1e-7, rtol=1e-5)}
    return {'metrics': metrics, 'components': components, 'd_total': total.tolist(),
            'checks': checks, 'EPS_attribution_deficit': eps_deficit,
            'zero_norm': {'total_diff': norm==0, 'pseudo_diff': np.linalg.norm(dp)==0},
            'diff_dot_sign': int(np.sign(current_c)), 'passed': all(c['passed'] for c in checks.values())}


def triage(scopes):
    dominance, utility = {}, {}
    for group in ('loc', 'conf'):
        a, c = group+'/A', group+'/C_diff'
        dominance[group] = {'overall_median_above_half': scopes['overall']['metrics'][a]['median']>.5,
                            'corrupted_median_above_half': scopes['corrupted']['metrics'][a]['median']>.5,
                            'at_least3_blocks_above_half': sum(scopes[f'block{i}']['metrics'][a]['median']>.5 for i in range(4))>=3}
        utility[group] = {'overall_positive_ge20': scopes['overall']['metrics'][c]['positive_count']>=20,
                          'corrupted_positive_ge10': scopes['corrupted']['metrics'][c]['positive_count']>=10,
                          'overall_median_positive': scopes['overall']['metrics'][c]['median']>0,
                          'at_least3_positive_blocks': sum(scopes[f'block{i}']['metrics'][c]['median']>0 for i in range(4))>=3}
    dominant = next((g for g in ('loc', 'conf') if all(dominance[g].values())), None)
    useful = all(utility[dominant].values()) if dominant else None
    decision = ('mixed_heterogeneous_no_single_component_rescue' if not dominant else
                'other_component_or_interaction_review' if useful else
                'localization_objective_design_review' if dominant=='loc' else 'confidence_objectness_design_review')
    return {'dominance_flags': dominance, 'utility_flags': utility, 'dominant': dominant,
            'dominant_pseudo_useful': useful, 'decision': decision, 'stop_for_research_review': True}


def summarize(rows):
    groups = {'overall': rows, 'clean': [r for r in rows if r['case']=='clean_s0'],
              'corrupted': [r for r in rows if r['case']!='clean_s0']}
    groups.update({f'block{i}': [r for r in rows if r['block']==i] for i in range(4)})
    groups.update({'case/'+c: [r for r in rows if r['case']==c] for c in sorted({r['case'] for r in rows})})
    scopes = {k: describe(v) for k, v in groups.items()}
    return {'scopes': scopes, 'triage': triage(scopes)}


def run(prior_root, manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    paths = {k: prior_root/v['path'] for k,v in manifest.items()}
    for k,p in paths.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[k]['sha256']
    old = json.loads(paths['records.json'].read_text())['records']
    pseudo = json.loads(paths['t015_records'].read_text())['records']
    supports = json.loads(paths['supports.json'].read_text())
    cohort = json.loads(paths['cohort.json'].read_text())
    assert len(old)==len(pseudo)==len(supports)==32
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260913)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark=False
    source, isp = load_detector('cuda:0'), DifferentiableISP().to('cuda:0')
    source_hash=state_hash(source)
    assert source_hash=='73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf'
    weight_hash=hashlib.sha256((Path(torch.hub.get_dir())/'checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth').read_bytes()).hexdigest()
    assert weight_hash=='258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384'
    states=[{k:v.clone() for k,v in source.state_dict().items()}]
    assert frozen_unchanged((source,),states)
    write_json(output/'environment.json', {'inputs': manifest, 'source_revision': os.environ.get('TAISP_SOURCE_REVISION'),
               'source_weight_sha256': weight_hash, 'source_state_sha256': source_hash,
               'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
               'gpu': torch.cuda.get_device_name(0), 'seed': 20260913, 'jacobian_dtype': 'float32',
               'reduction_dtype': 'float64', 'keys': KEYS, 'deterministic_algorithms': False,
               'cudnn_benchmark': False, 'cublas_workspace_config': None, 'optimizer_steps': 0, 'clip_calls': 0})
    episodes=load_episodes(cohort,'cuda:0')
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    rows=[]
    for i,ep in enumerate(episodes):
        assert ep['image_id']==old[i]['image_id']==pseudo[i]['image_id']==supports[i]['image_id']
        assert ep['case']==old[i]['case']==pseudo[i]['case']==supports[i]['case']
        mask,rectangles=support_mask(ep['image'],torch.tensor(supports[i]['boxes'],device='cuda:0'))
        mask_hash=hashlib.sha256(mask.to(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        assert mask_hash==old[i]['mask_uint8_sha256'] and rectangles==old[i]['mask_rectangles']
        row={k:old[i][k] for k in ('episode_index','image_id','case','block','support_count','mask_area_fraction','mask_uint8_sha256')}
        row.update(component_reference(ep['image'],isp,mask,lambda y:detector_task_loss(source,y,ep['targets'],seed=20260913)))
        row['numerical_checks']=numerical_checks(row['references'],row['direct_global'],old[i]['task']['reference'])
        row['isolation']={'source_frozen_eval_grad_none':frozen_unchanged((source,),states),
                          'source_hash_unchanged':state_hash(source)==source_hash,
                          'isp_identity_grad_none':bool(torch.count_nonzero(isp.phi)==0 and isp.phi.grad is None)}
        row['numerical_passed']=row['numerical_checks']['passed'] and all(row['isolation'].values()) and all(c['passed'] for c in row['jacobian']['primal_identity_checks'])
        write_json(output/f'record_{i:02d}.json',row)
        rows.append(row)
        if not row['numerical_passed']:
            write_json(output/'blocker.json',{'stage':'component_numerical_closure','episode_index':i,
                       'collected_records':len(rows),'detector_forwards':len(rows),'image_cotangent_requests':5*len(rows),
                       'ISP_JVP_columns':8*len(rows),'scientific_stage':'NOT_REACHED','elapsed_seconds':time.perf_counter()-started,
                       'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
            raise RuntimeError('Component closure/parity/isolation blocker; no tolerance change or rerun')
        print(json.dumps({'episode':i+1,'of':32,'numerical_passed':True}),flush=True)
    for i,row in enumerate(rows):
        row['analysis']=attribution(row['references'],pseudo[i]['analysis'])
        row['analysis']['metrics'].update({'value/'+k:v for k,v in row['losses'].items()})
        write_json(output/f'attribution_{i:02d}.json',row['analysis'])
        assert row['analysis']['passed'], 'Attribution closure blocker; no post-outcome tolerance change'
    result=summarize(rows)
    write_json(output/'records.json',{'records':rows})
    write_json(output/'summary.json',result)
    write_json(output/'completion.json',{'status':'completed','episodes':32,'detector_forwards':32,'image_cotangent_requests':160,
               'ISP_JVP_columns':256,'optimizer_steps':0,'clip_calls':0,'pseudo_calls':0,'triage':result['triage'],
               'elapsed_seconds':time.perf_counter()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()})
    print(json.dumps(result['triage']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prior-root','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    run(args.prior_root,args.manifest,args.output)
