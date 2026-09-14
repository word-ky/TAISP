"""R049 one source-only O(8) Procrustes fit and frozen holdout algebra."""
import math
import statistics
import numpy as np

EPS=1e-12


def norm(g):return math.sqrt(math.fsum(v*v for v in g))


def score(t,g):return math.fsum(a*b for a,b in zip(t,g))/(norm(g)+EPS)


def cosine(a,b):
    na,nb=norm(a),norm(b)
    return math.fsum(x*y for x,y in zip(a,b))/(na*nb) if na and nb else None


def fit_vectors(h,t):
    h,t=np.asarray(h,dtype=np.float64),np.asarray(t,dtype=np.float64)
    nh,nt=np.linalg.norm(h,axis=1),np.linalg.norm(t,axis=1)
    uh=np.zeros_like(h);ut=np.zeros_like(t)
    uh[nh>EPS]=h[nh>EPS]/nh[nh>EPS,None];ut[nt>EPS]=t[nt>EPS]/nt[nt>EPS,None]
    c=ut.T@uh
    u,s,vt=np.linalg.svd(c,full_matrices=True)
    r=u@vt
    error=float(np.max(np.abs(r.T@r-np.eye(8))))
    assert np.isfinite(r).all() and error<=1e-10
    return {'R':r.tolist(),'C':c.tolist(),'singular_values':s.tolist(),'determinant':float(np.linalg.det(r)),
        'orthogonality_max_abs':error,'train_episode_count':len(h),'zero_unit_hard':int((nh<=EPS).sum()),
        'zero_unit_task':int((nt<=EPS).sum()),'unit_fit_squared_residual':float(np.sum((uh@r.T-ut)**2)),
        'svd_calls':1,'full_orthogonal_group':True,'dtype':'float64','device':'cpu','epsilon':EPS}


def evaluate(t,h,r):
    cal=(np.asarray(r,dtype=np.float64)@np.asarray(h,dtype=np.float64)).tolist()
    error=abs(norm(cal)-norm(h));assert error<=1e-10*max(norm(h),1.)
    sh,sc=score(t,h),score(t,cal)
    return {'g_cal':cal,'g_hard':h,'S_hard':sh,'S_cal':sc,'Delta':sc-sh,'norm_hard':norm(h),'norm_cal':norm(cal),
        'norm_preservation_error':error,'cos_hard_task':cosine(h,t),'cos_cal_task':cosine(cal,t)}


def describe(values):
    v=[x for x in values if x is not None]
    if not v:return {'n':0,'positive':0,'mean':None,'median':None,'quantiles':None}
    return {'n':len(v),'positive':sum(x>0 for x in v),'mean':statistics.mean(v),'median':statistics.median(v),
        'quantiles':np.quantile(v,[0,.25,.5,.75,1]).tolist()}


def coordinate_stats(rows,key):
    x=np.asarray([r[key] for r in rows],dtype=np.float64);energy=(x*x).sum(0);total=energy.sum()
    share=energy/total if total else np.zeros(8)
    return {'mean':x.mean(0).tolist(),'std':x.std(0).tolist(),'mean_abs':np.abs(x).mean(0).tolist(),
        'rms':np.sqrt((x*x).mean(0)).tolist(),'energy_share':share.tolist(),'max_energy_share':float(share.max()),
        'max_energy_coordinate':int(share.argmax()) if total else None}


def summarize(rows):
    def group(rs):
        return {'n':len(rs),'distributions':{k:describe([r[k] for r in rs]) for k in ['S_hard','S_cal','Delta','cos_hard_task','cos_cal_task','norm_hard','norm_cal']},
            'zero_hard':sum(r['norm_hard']==0 for r in rs),'zero_cal':sum(r['norm_cal']==0 for r in rs),
            'coordinates':{k:coordinate_stats(rs,k) for k in ['g_hard','g_cal']}}
    a,b,c=[group(rs) for rs in [rows,[r for r in rows if r['case']!='clean_s0'],[r for r in rows if r['case']=='clean_s0']]]
    blocks={str(i):group([r for r in rows if r['block']==i]) for i in range(4)}
    conditions={case:group([r for r in rows if r['case']==case]) for case in sorted({r['case'] for r in rows})}
    da,db,dc=[g['distributions'] for g in [a,b,c]]
    gates={'S_cal_overall':da['S_cal']['positive']>=80,'S_cal_corrupt':db['S_cal']['positive']>=45,
        'Delta_overall':da['Delta']['positive']>=72,'Delta_corrupt':db['Delta']['positive']>=36,
        'median_overall':da['Delta']['median']>0,'median_corrupt':db['Delta']['median']>0,
        'mean_overall':da['Delta']['mean']>0,'mean_corrupt':db['Delta']['mean']>0,
        'positive_blocks':sum(g['distributions']['Delta']['median']>0 for g in blocks.values())>=3,
        'median_clean':dc['Delta']['median']>=0,'integrity':len(rows)==120 and all(r['integrity_passed'] for r in rows)}
    return {'passed':all(gates.values()),'gates':gates,'overall':a,'corrupt':b,'clean':c,'conditions':conditions,'blocks':blocks,
        'status':'NEEDS_REVIEW','disposition':'orthogonal_capacity_pending_review' if all(gates.values()) else 'close_exact_global_orthogonal_family',
        'AP_calls':0,'runtime_authorized':False}
