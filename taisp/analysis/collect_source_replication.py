"""T013-H exactly one frozen source record per precommitted episode."""
import argparse
import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch

from taisp import DifferentiableISP
from taisp.losses.clip_semantic import load_clip_guidance
from taisp.losses.detector_native import DetectorNativeLoss
from taisp.models.detector import load_detector
from taisp.models.detector_signal import select_predictions
from taisp.models.parameter_predictor import ParameterPredictor
from .deterministic_replay import state_hash
from .gradient_conflict import evaluate, write_json
from .source_meta_smoke import load_episodes, frozen_unchanged, group_statistics
from .run_t002 import environment_metadata


def run(manifest_path, checkpoint, output):
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == '99f15bc4329b8431a6718bc1c2ae1ef3fa6221db0869a7cc593ef95bdca2d357'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == 'a0eb1023150395de2a882d564dc28887a9f80065590ad77e92267896b313e120'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') is None
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.manual_seed(20260913)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    device = 'cuda:0'
    source, clip = load_detector(device), load_clip_guidance(device, local_files_only=True)
    clip.eval()
    isp, original = DifferentiableISP().to(device), ParameterPredictor().to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    assert all(torch.equal(v,saved[k]) for k,v in original.state_dict().items())
    assert torch.count_nonzero(original.head.weight) == torch.count_nonzero(original.head.bias) == 0
    original_hash = state_hash(original)
    models = (source,clip)
    states = [{k:v.clone() for k,v in m.state_dict().items()} for m in models]
    assert frozen_unchanged(models,states)
    meta = environment_metadata({'seed':20260913,'inner_steps':3,'inner_lr':.1,'norm_epsilon':1e-12,
                                 'outer_coefficient_algebra_only':.001,'optimizer_steps':0,'records':64})
    assert meta['detector_sha256'] == '258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384'
    assert meta['clip_sha256'] == 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
    meta.update(interpretation='T013-H source-only primary records; no training or target/AP.',
                source_state_sha256=state_hash(source),clip_state_sha256=state_hash(clip),
                predictor_state_sha256=original_hash,initial_isolation_passed=True,
                deterministic_algorithms=False,cudnn_benchmark=False,cublas_workspace_config=None,
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    write_json(output/'environment.json',meta)
    shutil.copyfile(manifest_path,output/'cohort.json')
    shutil.copyfile(checkpoint,output/'original_predictor.pt')
    episodes = load_episodes(json.loads(manifest_path.read_text()),device)
    assert len(episodes) == 64
    records, supports = [], []
    torch.cuda.reset_peak_memory_stats()
    for index, episode in enumerate(episodes):
        with torch.no_grad():
            selected = select_predictions(source(episode['image'])[0],threshold=.5,topk=20)
        episode['native'] = DetectorNativeLoss(source,{'base':selected},'det_pseudo')
        supports.append({'episode_index':index,'image_id':episode['image_id'],'case':episode['case'],
                         **{k:v.cpu().tolist() for k,v in selected.items()}})
        write_json(output/'supports.json',supports)
        predictor = copy.deepcopy(original)
        features = []
        hook = predictor.features.register_forward_hook(lambda module,inputs,value: features.append(value.detach().cpu()))
        rows,_,_ = evaluate(predictor,isp,source,clip,[episode],gradients=True)
        hook.remove()
        assert len(features) == 1
        row = rows[0]
        row.update(episode_index=index,block=index//16,features=features[0].squeeze(0).tolist(),
                   empty_support=row['support_count']==0)
        row['isolation'] = {'frozen_models_unchanged':frozen_unchanged(models,states),
                            'predictor_unchanged':state_hash(predictor)==original_hash,
                            'original_unchanged':state_hash(original)==original_hash,
                            'predictor_grad_none':all(p.grad is None for p in predictor.parameters()),
                            'isp_unchanged':bool(torch.count_nonzero(isp.phi)==0 and isp.phi.grad is None)}
        write_json(output/f'record_{index:02d}.json',row)
        assert all(row['isolation'].values())
        records.append(row)
        print(json.dumps({'episode':index+1,'of':64,'image_id':row['image_id'],'case':row['case'],
                          'support_count':row['support_count'],'isolation_passed':True}),flush=True)
        del predictor
    write_json(output/'records.json',{'records':records})
    write_json(output/'completion.json',{'status':'completed','source_revision':os.environ.get('TAISP_SOURCE_REVISION'),
               'episode_count':len(records),'optimizer_steps':0,'primary_passes_per_episode':1,
               'all_isolation_passed':True,'clean':group_statistics(records,True),'corrupted':group_statistics(records,False),
               'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.perf_counter()-started})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    run(args.manifest,args.checkpoint,args.output)
