"""R043 fixed train-only affine SVD fit and pair-preserving nulls, CPU float64."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design_fit(x):
    x=np.asarray(x,dtype=np.float64)
    mean=x.mean(0);std=x.std(0,ddof=0);active=std!=0
    z=np.zeros_like(x);z[:,active]=(x[:,active]-mean[active])/std[active]
    z=np.column_stack([z,np.ones(len(z))])
    u,s,vt=np.linalg.svd(z,full_matrices=False)
    tol=np.finfo(np.float64).eps*max(z.shape)*s[0];keep=s>tol
    inverse=(vt[keep].T/s[keep])@u[:,keep].T
    return {'mean':mean,'std':std,'active':active,'z':z,'pinv':inverse,'singular_values':s,
            'tol':float(tol),'rank':int(keep.sum()),'effective_condition':float(s[0]/s[keep][-1])}


def transform(x, fit):
    x=np.asarray(x,dtype=np.float64);z=np.zeros_like(x);a=fit['active']
    z[:,a]=(x[:,a]-fit['mean'][a])/fit['std'][a]
    return np.column_stack([z,np.ones(len(z))])


def average_ranks(x):
    x=np.asarray(x);order=np.argsort(x,kind='stable');r=np.zeros(len(x));i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        r[order[i:j]]=(i+j-1)/2+1;i=j
    return r


def auc(y, score):
    positive=np.asarray(y)>0;n=int(positive.sum());m=len(positive)-n
    if not n or not m:return None
    return float((average_ranks(score)[positive].sum()-n*(n+1)/2)/(n*m))


def spearman(x,y):
    a,b=average_ranks(x),average_ranks(y);a-=a.mean();b-=b.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den else None


def describe(values):
    a=np.asarray(values,dtype=np.float64)
    if not len(a):return {'n':0,'positive_fraction':None,'mean':None,'median':None,'quantiles':None}
    return {'n':len(a),'positive_fraction':float((a>0).mean()),'mean':float(a.mean()),
            'median':float(np.median(a)),'quantiles':dict(zip(['0','.25','.5','.75','1'],np.quantile(a,[0,.25,.5,.75,1]).tolist()))}


def split_indices(rows):
    train=[i for i,r in enumerate(rows) if r['partition']=='train']
    holdout=[i for i,r in enumerate(rows) if r['partition']=='holdout']
    assert not {rows[i]['image_id'] for i in train}&{rows[i]['image_id'] for i in holdout}
    return train,holdout


def pair_permutations(families, count=128, seed=20260928):
    rng=np.random.Generator(np.random.PCG64(seed));families=np.asarray(families)
    for _ in range(count):
        order=np.arange(len(families))
        for family in sorted(set(families)):
            indices=np.flatnonzero(families==family);order[indices]=rng.permutation(indices)
        yield order


def metrics(rows):
    utility=[r['S_orig'] for r in rows];trusted=[r['S_orig'] for r in rows if r['score']>0]
    untrusted=[r['S_orig'] for r in rows if r['score']<=0];a,t,u=describe(utility),describe(trusted),describe(untrusted)
    return {'n':len(rows),'prevalence':a['positive_fraction'],'AUROC':auc([r['y'] for r in rows],[r['score'] for r in rows]),
            'Spearman':spearman([r['score'] for r in rows],utility),'trusted_coverage':len(trusted)/len(rows),
            'trusted':t,'untrusted':u,'all_utility':a,
            'precision_gain':t['positive_fraction']-a['positive_fraction'] if trusted else None,
            'nonzero_pseudo':sum(r['norm_pseudo']!=0 for r in rows),
            'nonzero_clip':sum(r['norm_clip']!=0 for r in rows),
            'integrity_passed':all(r['integrity_passed'] for r in rows)}


def run(candidate_root, reference_root, output):
    for folder in [candidate_root,reference_root]:
        for f,h in json.loads((folder/'sha256.json').read_text()).items():assert sha(folder/f)==h
    candidates=json.loads((candidate_root/'records.json').read_text())
    references=json.loads((reference_root/'records.json').read_text())
    assert len(candidates)==len(references)==360
    for c,r in zip(candidates,references,strict=True):
        assert (c['image_id'],c['case'],c['partition'])==(r['image_id'],r['case'],r['partition'])
    train,holdout=split_indices(candidates);assert len(train)==240 and len(holdout)==120
    x=np.asarray([r['features'] for r in candidates],dtype=np.float64)
    y_train=np.asarray([references[i]['y'] for i in train],dtype=np.float64)
    fit=design_fit(x[train]);w=fit['pinv']@y_train
    output.mkdir(parents=True,exist_ok=False)
    # Save literal estimator before applying it or inspecting holdout labels.
    estimator={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in fit.items() if k not in ['pinv','z']}
    estimator.update(weights=w.tolist(),train_episode_indices=train,threshold=0.,std_ddof=0,
                     reconstruction_relative_error=float(np.linalg.norm(fit['z']@fit['pinv']@fit['z']-fit['z'])/np.linalg.norm(fit['z'])),
                     numpy_version=np.__version__,device='CPU float64')
    write(output/'estimator.json',estimator)
    z_holdout=transform(x[holdout],fit);scores=z_holdout@w
    rows=[dict(references[i],score=float(score),norm_pseudo=candidates[i]['norm_pseudo'],norm_clip=candidates[i]['norm_clip']) for i,score in zip(holdout,scores)]
    families=[candidates[train[i+1]]['case'] for i in range(0,len(train),2)]
    for i in range(0,len(train),2):
        assert candidates[train[i]]['image_id']==candidates[train[i+1]]['image_id'] and candidates[train[i]]['case']=='clean_s0'
    null=[];pair_labels=y_train.reshape(-1,2);truth=np.asarray([r['y'] for r in rows]);corrupt=np.asarray([r['case']!='clean_s0' for r in rows])
    for k,order in enumerate(pair_permutations(families)):
        nw=fit['pinv']@pair_labels[order].reshape(-1);ns=z_holdout@nw
        null.append({'index':k,'pair_permutation':order.tolist(),'weights':nw.tolist(),'holdout_scores':ns.tolist(),
                     'overall_AUROC':auc(truth,ns),'corrupt_AUROC':auc(truth[corrupt],ns[corrupt])})
    overall=metrics(rows);corrupted=metrics([r for r in rows if r['case']!='clean_s0'])
    blocks={str(i):metrics([r for r in rows if r['block']==i]) for i in range(4)}
    conditions={case:metrics([r for r in rows if r['case']==case]) for case in sorted({r['case'] for r in rows})}
    null_summary={}
    for name,m in [('overall',overall),('corrupt',corrupted)]:
        values=[r[name+'_AUROC'] for r in null];observed=m['AUROC'];valid=observed is not None and all(v is not None for v in values)
        null_summary[name]={'observed':observed,'q95':float(np.quantile(values,.95)) if valid else None,
            'percentile_strict':sum(v<observed for v in values)/128 if valid else None,
            'corrected_one_sided_tail':(1+sum(v>=observed for v in values))/129 if valid else None,
            'passed':valid and observed>float(np.quantile(values,.95))}
    gates={}
    for name,m,cut in [('overall',overall,.70),('corrupt',corrupted,.65)]:
        gates.update({name+'_AUROC':m['AUROC'] is not None and m['AUROC']>=cut,
            name+'_coverage':.25<=m['trusted_coverage']<=.80,
            name+'_precision_gain':m['precision_gain'] is not None and m['precision_gain']>=.10,
            name+'_trusted_median':m['trusted']['median'] is not None and m['trusted']['median']>0,
            name+'_null':null_summary[name]['passed']})
    gates['positive_blocks']=sum(m['precision_gain'] is not None and m['precision_gain']>0 for m in blocks.values())>=3
    gates['integrity']=all(r['integrity_passed'] for r in references)
    summary={'status':'NEEDS_REVIEW','overall':overall,'corrupt':corrupted,'conditions':conditions,'blocks':blocks,
        'null':null_summary,'gates':gates,'passed':all(gates.values()),'AP_calls':0,'runtime_authorized':False,
        'disposition':'source_reliability_capacity_pending_review' if all(gates.values()) else 'close_fixed_affine_gradient_state_representation',
        'estimator_sha256':sha(output/'estimator.json'),'candidate_records_sha256':sha(candidate_root/'records.json'),
        'reference_records_sha256':sha(reference_root/'records.json')}
    write(output/'holdout_records.json',rows);write(output/'null_records.json',null);write(output/'summary.json',summary)
    write(output/'sha256.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidate-root','reference-root','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();run(a.candidate_root,a.reference_root,a.output)
