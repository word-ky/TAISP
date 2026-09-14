"""R051 label-free representation and pre-oracle corrected-direction stages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from .object_state_math import fit_representation,transform,corrected


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def vector_sha(x):return hashlib.sha256(np.asarray(x,dtype=np.float64).tobytes()).hexdigest()
def seal(root):write(root/'sha256.json',{p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file()})


def candidates(root,lock_path):
    lock=read(lock_path);assert lock['candidate_commit'] and len(lock['files'])==603
    assert sha(root/'sha256.json')==lock['manifest_sha256']
    for f,h in lock['files'].items():assert sha(root/f)==h,f
    rows=read(root/'records.json');env=read(root/'environment.json');done=read(root/'completion.json')
    assert len(rows)==600 and [r['episode_index'] for r in rows]==list(range(600))
    assert env['source_revision']==lock['candidate_code_commit'][:7]
    assert env['manifest_sha256']==lock['candidate_images_sha256']
    assert env['source_state_sha256']==done['source_hash_after']==lock['source_state_sha256']
    assert not done['GT_loaded'] and not done['forbidden_modules_loaded']
    for r in rows:
        assert r['integrity_passed'] and r['rng']['restored'] and r['hard']['parity']['passed']
        assert r['g_hard']==r['hard']['gradient'] and r['descriptor_support_sha256']==r['hard']['supports_sha256']
        assert len(r['d_obj'])==1024 and np.isfinite(r['d_obj']).all()
        assert hashlib.sha256(np.asarray(r['d_obj'],dtype=np.float32).tobytes()).hexdigest()==r['descriptor_sha256']
    assert [r['partition'] for r in rows]==['train']*480+['holdout']*120
    assert not {r['image_id'] for r in rows[:480]}&{r['image_id'] for r in rows[480:]}
    return rows,lock


def locked_representation(root,lock_path,candidate_lock_path):
    lock=read(lock_path);assert lock['representation_commit']
    for f,h in lock['files'].items():assert sha(root/f)==h,f
    rep=read(root/'representation.json');states=read(root/'states.json')
    assert rep['status']=='representation_complete' and rep['candidate_lock_sha256']==sha(candidate_lock_path)
    assert len(states)==600 and rep['train_episode_indices']==list(range(480))
    assert all(vector_sha(s['h_std'])==s['h_std_sha256'] for s in states)
    assert rep['task_gradients_computed']==0 and not rep['annotations_loaded']
    return rep,states,lock


def no_oracle():
    bad=[k for k in sys.modules if k.startswith('taisp.') and any(s in k for s in ('oracle','reference','source_meta','annotation'))]
    assert not bad,bad


def representation_stage(candidate_root,candidate_lock,output):
    no_oracle();rows,lock=candidates(candidate_root,candidate_lock);tr=rows[:480]
    output.mkdir(parents=True,exist_ok=False)
    rep=fit_representation([r['d_obj'] for r in tr],[r['g_hard'] for r in tr],
        [r['support_count'] for r in tr],[r['mean_score'] for r in tr])
    rep.update(candidate_lock_sha256=sha(candidate_lock),train_episode_indices=[r['episode_index'] for r in tr],
        annotations_loaded=False,task_gradients_computed=0,code_sha256=sha(Path(__file__)),
        math_code_sha256=sha(Path(__file__).with_name('object_state_math.py')),
        source_revision=os.environ.get('TAISP_SOURCE_REVISION'),numpy_version=np.__version__)
    write(output/'representation.json',rep)
    if rep['status']=='representation_complete':
        z=transform([r['d_obj'] for r in rows],[r['g_hard'] for r in rows],
            [r['support_count'] for r in rows],[r['mean_score'] for r in rows],rep)
        states=[{'episode_index':r['episode_index'],'partition':r['partition'],'h_std':h.tolist(),
                 'h_std_sha256':vector_sha(h)} for r,h in zip(rows,z)]
        write(output/'states.json',states)
    no_oracle();write(output/'completion.json',{'status':rep['status'],'train':480,'holdout':120,
        'GT_loaded':False,'task_gradients_computed':0,'AP_calls':0});seal(output)
    print(json.dumps({'status':rep['status'],'rank':rep['pca_rank']}))


def directions_stage(candidate_root,candidate_lock,representation_root,representation_lock,model_root,model_lock,output):
    no_oracle();rows,_=candidates(candidate_root,candidate_lock)
    rep,states,_=locked_representation(representation_root,representation_lock,candidate_lock)
    ml=read(model_lock);assert ml['model_commit']
    for f,h in ml['files'].items():assert sha(model_root/f)==h,f
    model=read(model_root/'model.json')
    assert model['representation_lock_sha256']==sha(representation_lock)
    assert model['candidate_lock_sha256']==sha(candidate_lock) and model['holdout_task_gradients_computed']==0
    assert model['train_reference_sha256']==sha(model_root/'records.json')
    hold=rows[480:];z=[r['h_std'] for r in states[480:]]
    out=corrected(z,[r['g_hard'] for r in hold],model)
    result=[{k:r[k] for k in ['episode_index','image_id','case','block','partition','g_hard','support_count']} for r in hold]
    for i,r in enumerate(result):r.update({k:v[i] for k,v in out.items()});r['u_obj_sha256']=vector_sha(r['u_obj'])
    output.mkdir(parents=True,exist_ok=False);write(output/'directions.json',result)
    no_oracle();write(output/'completion.json',{'status':'directions_complete','episodes':120,'GT_loaded':False,
        'task_gradients_computed':0,'candidate_lock_sha256':sha(candidate_lock),
        'representation_lock_sha256':sha(representation_lock),'model_commit':ml['model_commit'],
        'model_lock_sha256':sha(model_lock),'model_sha256':sha(model_root/'model.json'),'AP_calls':0})
    seal(output);print(json.dumps({'status':'directions_complete','episodes':120,'abstentions':sum(out['abstain'])}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['representation','directions'])
    for name in ['candidate-root','candidate-lock','output']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['representation-root','representation-lock','model-root','model-lock']:p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.phase=='representation':representation_stage(a.candidate_root,a.candidate_lock,a.output)
    else:directions_stage(a.candidate_root,a.candidate_lock,a.representation_root,a.representation_lock,a.model_root,a.model_lock,a.output)
