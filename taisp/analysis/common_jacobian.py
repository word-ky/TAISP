"""R025 common-ISP JVP reference, fixed numerical replay and conditional R024 audit."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from taisp import DifferentiableISP
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector import load_detector
from .deterministic_replay import state_hash
from .oracle import detector_task_loss
from .predictor_factorization import write_json
from .source_meta_smoke import load_episodes, frozen_unchanged
from .spatial_action import support_mask, identity_gradients, tensor_check, geometry, summarize


def reference_checks(global_ref, obj, bg, direct):
    gr,go,gb,gd = [np.asarray(v,dtype=np.float64) for v in (global_ref,obj,bg,direct)]
    error = np.abs(gr-(go+gb))
    bound = 1e-12+1e-10*(np.abs(go)+np.abs(gb))
    nr,nd = np.linalg.norm(gr),np.linalg.norm(gd)
    cosine = float(gr@gd/(nr*nd)) if nr and nd else (1. if nr==nd==0 else 0.)
    relative = float(np.linalg.norm(gr-gd)/max(nr,1e-12))
    return {'partition':{'passed':bool(np.all(error<=bound)),'absolute_errors':error.tolist(),'bounds':bound.tolist(),
                         'max_absolute_error':float(error.max()),'max_error_over_bound':float(np.max(error/bound))},
            'parity':{'passed':cosine>=.999999 and relative<=1e-5,'cosine':cosine,'relative_l2_error':relative,
                      'max_absolute_error':float(np.max(np.abs(gr-gd))),'min_cosine':.999999,'max_relative_l2':1e-5}}


def common_reference(image, isp, mask, cotangents):
    x,m = image.detach(),mask.detach().double()
    cs = {name:c.detach().double() for name,c in cotangents.items()}
    columns,primals = [],[]
    refs = {name:{'global':[],'object':[],'background':[]} for name in cs}
    phi = x.new_zeros(8)
    for k in range(8):
        direction = x.new_zeros(8)
        direction[k]=1
        primal,column = torch.func.jvp(lambda p:isp(x,p),(phi,),(direction,))
        assert torch.isfinite(column).all(), 'Non-finite ISP JVP column; stop.'
        jd = column.detach().double()
        columns.append(jd.norm().item())
        primals.append(tensor_check(primal,x,atol=2e-7,rtol=1e-6))
        for name,c in cs.items():
            # All products and sums are float64; both regions are reduced independently.
            refs[name]['global'].append((c*jd).sum().item())
            refs[name]['object'].append((c*m*jd).sum().item())
            refs[name]['background'].append((c*(1-m)*jd).sum().item())
    return refs,{'column_l2_norms':columns,'primal_identity_checks':primals,'jacobian_dtype':str(image.dtype),
                 'reduction_dtype':'torch.float64','reduction_device':str(image.device),'columns':8,
                 'cotangent_sha256':{name:hashlib.sha256(c.detach().cpu().numpy().tobytes()).hexdigest() for name,c in cotangents.items()}}


def numerical_pass(row):
    return (all(row['isolation'].values()) and all(c['passed'] for c in row['jacobian']['primal_identity_checks'])
            and all(row[name]['reference_checks'][check]['passed'] for name in ('task','pseudo') for check in ('partition','parity'))
            and all(row[name]['reverse']['checks'][check]['passed'] for name in ('task','pseudo')
                    for check in ('identity_image','regional_global_output')))


def collect(episode, saved, index, source, isp, states, source_hash):
    device = episode['image'].device
    assert (episode['image_id'],episode['case'])==(saved['image_id'],saved['case'])
    selected = {k:torch.tensor(saved[k],device=device,dtype=torch.long if k=='labels' else torch.float32)
                for k in ('boxes','scores','labels')}
    mask,rectangles = support_mask(episode['image'],selected['boxes'])
    native = DetectorNativeLoss(source,{'base':selected},'det_pseudo')
    task,ct = identity_gradients(episode['image'],isp,mask,
        lambda y:detector_task_loss(source,y,episode['targets'],seed=20260913),return_cotangent=True)
    pseudo,cp = identity_gradients(episode['image'],isp,mask,
        lambda y:(native(episode['image'],y),{}),return_cotangent=True)
    refs,jacobian = common_reference(episode['image'],isp,mask,{'task':ct,'pseudo':cp})
    current_hash = state_hash(source)
    row = {'episode_index':index,'parent_episode_index':saved['episode_index'],'image_id':episode['image_id'],
           'case':episode['case'],'block':index//8,'support_count':len(selected['boxes']),
           'mask_shape':list(mask.shape),'mask_rectangles':rectangles,'mask_area_fraction':mask.double().mean().item(),
           'mask_uint8_sha256':hashlib.sha256(mask.to(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
           'jacobian':jacobian,'source_state_sha256':current_hash,
           'isolation':{'source_unchanged_frozen_eval_grad_none':frozen_unchanged((source,),states),
                        'source_hash_unchanged':current_hash==source_hash,
                        'isp_unchanged_grad_none':bool(torch.count_nonzero(isp.phi)==0 and isp.phi.grad is None)}}
    for name,reverse in (('task',task),('pseudo',pseudo)):
        row[name] = {'reverse':reverse,'reference':refs[name],
                     'reference_checks':reference_checks(refs[name]['global'],refs[name]['object'],refs[name]['background'],reverse['global'])}
    row['numerical_passed'] = numerical_pass(row)
    return row


def run(manifest_path, prior_root, old_run, old_blocker, output):
    pins = {'supports.json':'bf8f688741fcb4956c7ed40f52cc5f0a63d3030cad3ada968f1457a38c2cf747',
            'environment.json':'d0635da6c52a1f41a86181ace053c6c6fe1f38f342530b6fd9d806e8ee98cc42',
            'cohort.json':'99f15bc4329b8431a6718bc1c2ae1ef3fa6221db0869a7cc593ef95bdca2d357'}
    frozen_files = {manifest_path:'a4b250a877ae1171cd6f30899f531ba2645cb37ff90abf1a26b8de48d6aa558e',
                    old_run/'artifacts/audit/record_06.json':'ffb8e0561043d6e5d1e9bbdd466fe53bc9b8192912660076a837ccdade52fcae',
                    old_blocker:'81c8a815f12c6d7026d2896db358888820896074a758fc0500d2a94d835c79c6',
                    **{prior_root/name:pin for name,pin in pins.items()}}
    for path,pin in frozen_files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pin,str(path)
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None
    output.mkdir(parents=True,exist_ok=True)
    (output/'debug').mkdir()
    started = time.perf_counter()
    torch.manual_seed(20260913)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    device = 'cuda:0'
    source,isp = load_detector(device),DifferentiableISP().to(device)
    source_hash = state_hash(source)
    assert source_hash=='73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf'
    weight_hash = hashlib.sha256((Path(torch.hub.get_dir())/'checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth').read_bytes()).hexdigest()
    assert weight_hash=='258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384'
    states = [{k:v.clone() for k,v in source.state_dict().items()}]
    assert frozen_unchanged((source,),states)
    write_json(output/'environment.json',{'source_revision':os.environ.get('TAISP_SOURCE_REVISION'),
        'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),
        'cuda':torch.version.cuda,'seed':20260913,'source_weight_sha256':weight_hash,'source_state_sha256':source_hash,
        'input_sha256':{str(p):h for p,h in frozen_files.items()},'jacobian':'torch.func.jvp, float32 existing ISP',
        'reduction':'float64 independent global/object/background','deterministic_algorithms':False,
        'cudnn_benchmark':False,'cublas_workspace_config':None,'clip_calls':0,'optimizer_steps':0})
    shutil.copyfile(manifest_path,output/'cohort.json')
    manifest = json.loads(manifest_path.read_text())
    supports = json.loads((prior_root/'supports.json').read_text())
    selected = [supports[j] for i in manifest['parent_image_indices'] for j in (2*i,2*i+1)]
    write_json(output/'supports.json',selected)
    episodes = load_episodes(manifest,device)
    assert len(episodes)==32
    torch.cuda.reset_peak_memory_stats()
    debug = []
    for index in (0,5,6):
        row = collect(episodes[index],selected[index],index,source,isp,states,source_hash)
        write_json(output/'debug'/f'record_{index:02d}.json',row)
        debug.append(row)
        stage = {'indices':[0,5,6],'collected_indices':[r['episode_index'] for r in debug],
                 'passed':len(debug)==3 and all(r['numerical_passed'] for r in debug),
                 'all_collected_passed':all(r['numerical_passed'] for r in debug),'records':debug}
        write_json(output/'stage_c.json',stage)
        assert row['numerical_passed'], 'Stage C numerical blocker; no full rerun or threshold/precision sweep.'
        print(json.dumps({'stage':'debug','episode_index':index,'numerical_passed':True}),flush=True)
    assert stage['passed']
    (output/'full').mkdir()
    rows = []
    for index,episode in enumerate(episodes):
        row = collect(episode,selected[index],index,source,isp,states,source_hash)
        write_json(output/'full'/f'record_{index:02d}.json',row)
        assert row['numerical_passed'], 'Full cohort numerical blocker; keep all receipts, no tolerance change.'
        rows.append(row)
        print(json.dumps({'stage':'full','episode':index+1,'of':32,'numerical_passed':True}),flush=True)
    for row in rows:
        row['geometry'] = geometry(row['task']['reference']['object'],row['task']['reference']['background'],
                                   row['pseudo']['reference']['object'],row['pseudo']['reference']['background'])
    write_json(output/'records.json',{'records':rows})
    summary = summarize(rows)
    write_json(output/'summary.json',summary)
    for path,pin in frozen_files.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==pin,str(path)
    write_json(output/'completion.json',{'status':'completed','source_revision':os.environ.get('TAISP_SOURCE_REVISION'),
        'stage_c_passed':True,'full_records':32,'debug_records':3,'source_loss_calls':70,'ISP_JVP_columns':280,
        'old_receipts_unchanged':True,'all_numerical_and_isolation_passed':True,'gate':summary['gate'],
        'elapsed_seconds':time.perf_counter()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'optimizer_steps':0,'clip_calls':0})
    print(json.dumps({'gate':summary['gate'],'overall':summary['scopes']['overall']}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','prior-root','old-run','old-blocker','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    run(args.manifest,args.prior_root,args.old_run,args.old_blocker,args.output)
