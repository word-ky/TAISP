"""R051 frozen PCA16, 27-D state, affine tangent correction and score algebra."""
import numpy as np
from .orthogonal_transport import describe,coordinate_stats,summarize,cosine

EPS=1e-12


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return x/(np.linalg.norm(x,axis=-1,keepdims=True)+EPS)


def raw_state(descriptors,hard,counts,scores,representation):
    d=np.asarray(descriptors,dtype=np.float64);g=np.asarray(hard,dtype=np.float64)
    pc=(d-np.asarray(representation['pca_mean']))@np.asarray(representation['pca_basis']).T
    return np.column_stack([pc,unit(g),np.log(np.linalg.norm(g,axis=1)+EPS),np.asarray(counts)/20.,scores])


def transform(descriptors,hard,counts,scores,representation):
    x=raw_state(descriptors,hard,counts,scores,representation)
    z=(x-np.asarray(representation['state_mean']))/np.asarray(representation['state_scale'])
    z[:,representation['constant_dimensions']]=0
    assert np.isfinite(z).all()
    return z


def fit_representation(descriptors,hard,counts,scores):
    d=np.asarray(descriptors,dtype=np.float64);mean=d.mean(0);centered=d-mean
    _,s,vt=np.linalg.svd(centered,full_matrices=False)
    tol=np.finfo(np.float64).eps*max(centered.shape)*s[0]
    rank=int((s>tol).sum())
    result={'status':'representation_complete' if rank>=16 else 'FAIL_REPRESENTATION_COLLAPSE',
            'pca_mean':mean.tolist(),'pca_singular_values':s.tolist(),'pca_rank':rank,'pca_rank_tolerance':float(tol),
            'pca_svd_calls':1,'pca_dimension':16,'dtype':'float64','device':'cpu','train_count':len(d)}
    if rank<16:return result
    basis=vt[:16].copy();indices=np.argmax(np.abs(basis),axis=1)
    signs=np.where(basis[np.arange(16),indices]<0,-1.,1.);basis*=signs[:,None]
    result.update(pca_basis=basis.tolist(),sign_loading_indices=indices.tolist(),
                  sign_convention='largest absolute loading, lowest index tie, positive')
    h=raw_state(d,hard,counts,scores,result);std=h.std(0);constant=np.flatnonzero(std<=EPS)
    scale=std.copy();scale[constant]=1.
    result.update(state_mean=h.mean(0).tolist(),state_std=std.tolist(),state_scale=scale.tolist(),
                  constant_dimensions=constant.tolist(),state_dimension=27,standardization_ddof=0)
    return result


def tangent_target(hard,task):
    g,t=np.asarray(hard,dtype=np.float64),np.asarray(task,dtype=np.float64)
    uh,ut=unit(g),unit(t)
    zero=(np.linalg.norm(g,axis=1)==0)|(np.linalg.norm(t,axis=1)==0)
    target=ut-(ut*uh).sum(1,keepdims=True)*uh;target[zero]=0
    return target,zero


def fit_affine(states,hard,task):
    h=np.asarray(states,dtype=np.float64);target,zero=tangent_target(hard,task)
    x=np.column_stack([np.ones(len(h)),h]);u,s,vt=np.linalg.svd(x,full_matrices=False)
    tol=np.finfo(np.float64).eps*max(x.shape)*s[0];keep=s>tol
    coef=vt[keep].T@((u[:,keep].T@target)/s[keep,None])
    assert np.isfinite(coef).all()
    return {'b':coef[0].tolist(),'B':coef[1:].T.tolist(),'rank':int(keep.sum()),
            'singular_values':s.tolist(),'rank_tolerance':float(tol),'svd_calls':1,'train_count':len(h),
            'zero_target_count':int(zero.sum()),'zero_hard_count':int((np.linalg.norm(hard,axis=1)==0).sum()),
            'zero_task_count':int((np.linalg.norm(task,axis=1)==0).sum()),
            'condition_effective':float(s[0]/s[keep][-1]),
            'condition_full':float(s[0]/s[-1]) if keep.all() else None,
            'coefficient_frobenius':float(np.linalg.norm(coef)),'B_frobenius':float(np.linalg.norm(coef[1:])),
            'b_l2':float(np.linalg.norm(coef[0])),'training_squared_residual':float(np.sum((x@coef-target)**2)),
            'dtype':'float64','device':'cpu','target':target.tolist(),'train_r_hat':(x@coef).tolist()}


def corrected(states,hard,model):
    h,g=np.asarray(states,dtype=np.float64),np.asarray(hard,dtype=np.float64)
    uh=unit(g);rhat=np.asarray(model['b'])+h@np.asarray(model['B']).T
    perp=rhat-(rhat*uh).sum(1,keepdims=True)*uh
    v=uh+perp;out=unit(v);zero=np.linalg.norm(g,axis=1)==0;out[zero]=0
    assert np.isfinite(out).all() and np.isfinite(rhat).all()
    return {'u_obj':out.tolist(),'r_hat':rhat.tolist(),'r_perp':perp.tolist(),'abstain':zero.tolist(),
            'projection_dot':(perp*uh).sum(1).tolist(),'u_obj_norm':np.linalg.norm(out,axis=1).tolist()}


def evaluate(task,hard,u_obj):
    t,g,u=[np.asarray(v,dtype=np.float64) for v in [task,hard,u_obj]]
    sh=float(t@g/(np.linalg.norm(g)+EPS));so=float(t@u)
    return {'S_hard':sh,'S_obj':so,'Delta':so-sh,'cos_hard_task':cosine(g.tolist(),t.tolist()),
            'cos_obj_task':cosine(u.tolist(),t.tolist()),'norm_hard':float(np.linalg.norm(g)),
            'norm_obj':float(np.linalg.norm(u)),'abstain':bool(np.linalg.norm(g)==0)}


def summary(rows):
    aliases=[{**r,'S_cal':r['S_obj'],'cos_cal_task':r['cos_obj_task'],'norm_cal':r['norm_obj'],'g_cal':r['u_obj']} for r in rows]
    s=summarize(aliases)
    s['gates'].pop('S_cal_overall');s['gates']['S_obj_overall']=sum(r['S_obj']>0 for r in rows)>=82
    s['gates']['S_obj_corrupt']=s['gates'].pop('S_cal_corrupt')
    s['gates']['finite_zero_integrity']=all(np.isfinite(r['u_obj']).all() and (not r['abstain'] or (r['S_obj']==0 and r['Delta']==0)) for r in rows)
    s['passed']=all(s['gates'].values());s['abstentions']=sum(r['abstain'] for r in rows)
    s['disposition']='object_state_tangent_capacity_pending_review' if s['passed'] else 'close_exact_pooled_ROI_PCA16_affine_tangent_family'
    s['support_subsets']={}
    for name,rs in [('empty',[r for r in rows if r['support_count']==0]),('nonempty',[r for r in rows if r['support_count']>0])]:
        s['support_subsets'][name]={'n':len(rs),'distributions':{k:describe([r[k] for r in rs]) for k in
            ['S_hard','S_obj','Delta','cos_hard_task','cos_obj_task','norm_hard','norm_obj']},
            'coordinates':{k:coordinate_stats(rs,k) for k in ['g_hard','u_obj']} if rs else {}}
    return s
