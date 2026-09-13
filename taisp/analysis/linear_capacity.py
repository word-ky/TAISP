"""T013-I frozen-array SVD capacity audit and matched image-pair permutation null."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .predictor_factorization import factorize, roundoff_check, write_json
from .source_replication import cosine, fold_indices

PINS = {
    'artifacts/collection/records.json': 'ac3d462521ad84e71af298eb57ae91b732f97e74fcf5336de91ab0967c41c2ab',
    'artifacts/collection/cohort.json': '99f15bc4329b8431a6718bc1c2ae1ef3fa6221db0869a7cc593ef95bdca2d357',
    'artifacts/algebra/audit.json': 'ef31b0394eca3e1d4ee773380b65f2a1a11f37620b67d05c3bb77b2da6ba0a72',
}


def pair_rows(order):
    return (2*np.asarray(order)[:,None]+np.array([0,1])).ravel()


def permutation_schedule():
    rng = np.random.default_rng(20260913)
    return [[rng.permutation(24).tolist() for _ in range(4)] for _ in range(128)]


def fit(h, g, pair_order=None):
    mu, gbar = h.mean(0), g.mean(0)
    x, y = h-mu, g-gbar
    if pair_order is not None:
        y = y[pair_rows(pair_order)]
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    tol = np.finfo(np.float64).eps*max(x.shape)*s[0]
    keep = s > tol
    coef = (vt[keep].T/s[keep])@(u[:,keep].T@y)
    assert np.isfinite(coef).all(), 'Non-finite SVD coefficients; stop, no regularization.'
    return {'mu':mu.tolist(), 'g_bar':gbar.tolist(), 'B':coef.T.tolist(),
            'singular_values':s.tolist(), 'rank_tolerance':float(tol), 'numerical_rank':int(keep.sum()),
            'rank_deficient':bool(keep.sum()<min(x.shape)), 'coefficient_frobenius_norm':float(np.linalg.norm(coef)),
            'condition_all_singular_values':float(s[0]/s[-1]) if s[-1]>0 else None,
            'condition_retained_subspace':float(s[0]/s[keep][-1]) if keep.any() else None,
            'training_residual_sse':float(np.sum((y-x@coef)**2))}


def cross_validate(h, g, permutations=None):
    residual, predicted, ghat = np.empty_like(g), np.empty_like(g), np.empty_like(g)
    folds = []
    for f,(train,test) in enumerate(fold_indices()):
        order = permutations[f] if permutations is not None else None
        fitted = fit(h[train],g[train],order)
        mu,gbar,b = (np.array(fitted[k]) for k in ('mu','g_bar','B'))
        residual[test] = g[test]-gbar
        predicted[test] = (h[test]-mu)@b.T
        ghat[test] = gbar+predicted[test]
        assert np.isfinite(predicted[test]).all() and np.isfinite(ghat[test]).all(), 'Non-finite prediction; stop.'
        folds.append({'fold':f,'train_indices':train.tolist(),'test_indices':test.tolist(),
                      'gradient_pair_order':order,'gradient_row_indices':train[pair_rows(order)].tolist() if order is not None else train.tolist(),
                      **fitted})
    return {'folds':folds, 'true_residual':residual.tolist(), 'predicted_residual':predicted.tolist(),
            'predicted_gradient':ghat.tolist()}


def metrics(g, residual, predicted, ghat):
    dots = np.sum(residual*predicted,axis=1)
    sse = float(np.sum((residual-predicted)**2))
    energy = float(np.sum(residual**2))
    residual_cos = cosine(residual,predicted)
    gradient_cos = cosine(g,ghat)
    changes = -.001*np.sum(g*ghat,axis=1)
    norms = {'true_residual':np.linalg.norm(residual,axis=1),
             'predicted_residual':np.linalg.norm(predicted,axis=1), 'predicted_gradient':np.linalg.norm(ghat,axis=1)}
    return {'n':len(g), 'residual_sse':sse, 'residual_energy':energy,
            'R2_residual':1-sse/energy if energy else None,
            'median_residual_cosine':float(np.median(residual_cos)), 'positive_residual_dot_count':int(np.sum(dots>0)),
            'median_full_gradient_cosine':float(np.median(gradient_cos)),
            'negative_first_order_count':int(np.sum(changes<0)), 'first_order_sum':float(changes.sum()),
            'first_order_mean':float(changes.mean()), 'first_order_median':float(np.median(changes)),
            'first_order_mean_absolute':float(np.mean(np.abs(changes))),
            'norms':{k:{'mean':float(v.mean()),'median':float(np.median(v)),'max':float(v.max())} for k,v in norms.items()}}


def scopes(labels):
    indices = np.arange(64)
    result = {'overall':indices,'clean':indices[::2],'corrupted':indices[1::2]}
    for f,(_,test) in enumerate(fold_indices()):
        result[f'fold{f}'] = test
        result[f'fold{f}/clean'] = test[::2]
        result[f'fold{f}/corrupted'] = test[1::2]
    for case in sorted({r['case'] for r in labels}):
        result['case/'+case] = np.array([i for i,r in enumerate(labels) if r['case']==case])
    for family in sorted({r['case'].rsplit('_s',1)[0] for r in labels}):
        result['family/'+family] = np.array([i for i,r in enumerate(labels) if r['case'].rsplit('_s',1)[0]==family])
    return result


def summaries(g, residual, predictions, labels):
    return {name:{method:metrics(g[idx],residual[idx],rhat[idx],ghat[idx])
                  for method,(rhat,ghat) in predictions.items()} for name,idx in scopes(labels).items()}


def null_comparison(observed, values):
    values = np.array(values,dtype=np.float64)
    return {'observed':observed,'values':values.tolist(),
            'empirical_percentile_strict':float(100*np.mean(values<observed)),
            'ties':int(np.sum(values==observed)), 'null_at_least_observed':int(np.sum(values>=observed)),
            'corrected_one_sided_tail':float((1+np.sum(values>=observed))/129),
            'null_95th_percentile_linear':float(np.quantile(values,.95,method='linear'))}


def gate(summary, comparisons):
    overall = summary['overall']['linear']
    positive_folds = sum(summary[f'fold{i}']['linear']['R2_residual'] is not None
                         and summary[f'fold{i}']['linear']['R2_residual']>0 for i in range(4))
    flags = {'pooled_R2_at_least_point10':overall['R2_residual'] is not None and overall['R2_residual']>=.10,
             'at_least_three_positive_R2_folds':positive_folds>=3,
             'median_residual_cosine_at_least_point20':overall['median_residual_cosine']>=.20,
             'positive_dot_at_least44':overall['positive_residual_dot_count']>=44,
             'clean_positive_dot_at_least20':summary['clean']['linear']['positive_residual_dot_count']>=20,
             'corrupted_positive_dot_at_least20':summary['corrupted']['linear']['positive_residual_dot_count']>=20,
             'R2_above_null95':overall['R2_residual'] is not None and overall['R2_residual']>comparisons['R2_residual']['null_95th_percentile_linear'],
             'cosine_above_null95':overall['median_residual_cosine']>comparisons['median_residual_cosine']['null_95th_percentile_linear']}
    return {'flags':flags,'positive_R2_folds':positive_folds,'passed':all(flags.values()),
            'decision':'linear_capacity_supported_for_later_dynamics_review' if all(flags.values()) else 'current_linear_initialization_branch_not_supported_request_research_pivot',
            'T013H_utility_remains_failed':True,'stop_for_research_review':True}


def run(prior_root, manifest_path, output):
    started = time.perf_counter()
    manifest = json.loads(manifest_path.read_text())
    hashes = {r['path']:r['sha256'] for r in manifest['files']}
    for name,pin in PINS.items():
        assert hashes[name] == pin == hashlib.sha256((prior_root/name).read_bytes()).hexdigest(), name
    rows = json.loads((prior_root/'artifacts/collection/records.json').read_text())['records']
    cohort = json.loads((prior_root/'artifacts/collection/cohort.json').read_text())
    prior = json.loads((prior_root/'artifacts/algebra/audit.json').read_text())
    assert len(rows)==64 and [r['image_id'] for r in rows[::2]]==cohort['image_ids']
    assert all(r['block']==i//16 and r['image_id']==cohort['image_ids'][i//2]
               and r['case']==('clean_s0' if i%2==0 else cohort['corrupted_cases'][i//2]) for i,r in enumerate(rows))
    h = np.array([r['features'] for r in rows],dtype=np.float64)
    g = np.array([r['phi0_gradient'] for r in rows],dtype=np.float64)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'provenance.json',{'input_sha256':PINS,'artifact_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               'source_revision':os.environ.get('TAISP_SOURCE_REVISION'),'python':platform.python_version(),'numpy':np.__version__,
               'dtype':'float64','device':'CPU','model_calls':0,'optimizer_steps':0,'new_records':0,
               'seed':20260913,'permutations':128,'quantile_method':'linear','tol':'eps64 * max(X.shape) * s_max'})
    observed = cross_validate(h,g)
    residual = np.array(observed['true_residual'])
    cov,common = np.empty_like(g),np.empty_like(g)
    for f,(train,test) in enumerate(fold_indices()):
        mu,gbar,_,c = factorize(h[train],g[train])
        assert test.tolist()==prior['crossfit']['folds'][f]['test_indices']
        assert train.tolist()==prior['crossfit']['folds'][f]['train_indices']
        cov[test],common[test] = (h[test]-mu)@c.T,gbar
    check = roundoff_check(-.001*cov,np.array([r['directions']['cov']['delta'] for r in prior['crossfit']['records']]))
    write_json(output/'covariance_reference_check.json',check)
    assert check['passed'], 'Inherited covariance reference mismatch; do not change tolerance.'
    predictions = {'linear':(np.array(observed['predicted_residual']),np.array(observed['predicted_gradient'])),
                   'common':(np.zeros_like(g),common),'covariance':(cov,cov)}
    summary = summaries(g,residual,predictions,rows)
    per_episode = []
    for i,row in enumerate(rows):
        item = {'episode_index':i,'image_id':row['image_id'],'case':row['case'],'fold':i//16,
                'h':h[i].tolist(),'g':g[i].tolist(),'true_residual':residual[i].tolist(),'methods':{}}
        for name,(rhat,ghat) in predictions.items():
            item['methods'][name] = {'predicted_residual':rhat[i].tolist(),'predicted_gradient':ghat[i].tolist(),
                'residual_dot':float(residual[i]@rhat[i]),'residual_cosine':float(cosine(residual[i],rhat[i])),
                'full_gradient_cosine':float(cosine(g[i],ghat[i])),'residual_sse':float(np.sum((residual[i]-rhat[i])**2)),
                'residual_energy':float(np.sum(residual[i]**2)),'delta':(-.001*ghat[i]).tolist(),
                'g_dot_delta':float(g[i]@(-.001*ghat[i])),
                'norm_r':float(np.linalg.norm(residual[i])),'norm_r_hat':float(np.linalg.norm(rhat[i])),
                'norm_g_hat':float(np.linalg.norm(ghat[i]))}
        per_episode.append(item)
    write_json(output/'observed.json',{'folds':observed['folds'],'episodes':per_episode,'summaries':summary,
               'covariance_reference':'Residual estimate C(h-mu); descent direction is this residual alone, without gbar, as R022.'})
    schedule = permutation_schedule()
    write_json(output/'permutation_schedule.json',schedule)
    null = []
    for index,permutations in enumerate(schedule):
        fitted = cross_validate(h,g,permutations)
        assert fitted['true_residual']==observed['true_residual']
        stats = metrics(g,residual,np.array(fitted['predicted_residual']),np.array(fitted['predicted_gradient']))
        # True residuals and inputs are shared with observed.json; retain every fitted prediction and coefficient.
        del fitted['true_residual']
        write_json(output/f'permutation_{index:03d}.json',{'permutation':index,**fitted,'pooled_metrics':stats})
        null.append(stats)
    fields = ('R2_residual','median_residual_cosine','positive_residual_dot_count')
    comparisons = {key:null_comparison(summary['overall']['linear'][key],[n[key] for n in null]) for key in fields}
    result = {'comparisons':comparisons,'gate':gate(summary,comparisons),'summaries':summary,
              'elapsed_seconds':time.perf_counter()-started,'covariance_reference_reconstruction':check}
    write_json(output/'summary.json',result)
    write_json(output/'completion.json',{'status':'completed','gate':result['gate'],'permutations':len(null),
               'heldout_records':64,'svd_fits':4+4*len(null),'model_calls':0,'optimizer_steps':0,
               'elapsed_seconds':result['elapsed_seconds'],'source_revision':os.environ.get('TAISP_SOURCE_REVISION')})
    print(json.dumps({'gate':result['gate'],'linear_pooled':summary['overall']['linear']}),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prior-root','artifact-manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args = p.parse_args()
    run(args.prior_root,args.artifact_manifest,args.output)
