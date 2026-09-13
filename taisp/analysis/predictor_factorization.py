"""T013-G offline saved-array factorization; no models or optimizer execution."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

FILE_PINS = {
    'receipt.json':'7c35fb9e5f7b1c923c06f625ddf806d443e763b56aa52d8027594f0b95cce943',
    'original_predictor.pt':'a0eb1023150395de2a882d564dc28887a9f80065590ad77e92267896b313e120',
    'joint_one_step_predictor.pt':'fec4271dcfda344023521c17070d117d08082664fdb6ddd6934adb47e727c634',
}
ARRAY_PINS = {
    'features':'4ffe547ee5d3373a6c5258a1fa89582dea6d04389eb8b82aef5a37f53af066bf',
    'phi0_gradients':'ba330697a73af6b2e1017944f0446de85039f4406850b3b5dd28f842f42df90d',
    'head_weight_gradients':'ebf8700a86f71e250258d138c19616621797639c3e2b8fc6b99d5be81c82a901',
    'head_bias_gradients':'ba330697a73af6b2e1017944f0446de85039f4406850b3b5dd28f842f42df90d',
    'saved_conditioning_outputs':'e9d159188699e2a25c52fda856b178fd9af3a2de0457ef05b2c9b9122e4586c9',
    'saved_episode_phi0':'174b8e0d3bc034bdd18500a41e1508ee9e0455458ec997744aa277e96444fa9c',
}


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def distances(x):
    return np.linalg.norm(x[:,None,:]-x[None,:,:],axis=-1)


def feature_statistics(h):
    mu = h.mean(0)
    z = h-mu
    s = np.linalg.svd(z,compute_uv=False)
    d = distances(h)
    norms = np.linalg.norm(h,axis=1)
    cosine = (h@h.T)/(norms[:,None]*norms[None,:])
    pairs = []
    for i in range(len(h)//2):
        clean,corrupt = 2*i,2*i+1
        ordinary = [d[clean,2*j] for j in range(len(h)//2) if i != j]
        nearest = int(np.argmin(d[corrupt,::2]))
        pairs.append({'clean_index':clean,'corrupt_index':corrupt,'condition_distance':float(d[clean,corrupt]),
                      'other_clean_distances':ordinary,'median_other_clean_distance':float(np.median(ordinary)),
                      'rho':float(d[clean,corrupt]/np.median(ordinary)),
                      'nearest_clean_index':2*nearest,'nearest_clean_distance':float(d[corrupt,2*nearest]),
                      'nearest_clean_is_own_image':nearest == i})
    return {'H':h.tolist(),'mu':mu.tolist(),'Z':z.tolist(),'mu_norm':float(np.linalg.norm(mu)),
            'total_energy':float(np.sum(h*h)),'centered_energy':float(np.sum(z*z)),
            'centered_fraction':float(np.sum(z*z)/np.sum(h*h)),
            'centered_singular_values':s.tolist(),'participation_effective_rank':float(np.sum(s*s)**2/np.sum(s**4)),
            'euclidean_distances':d.tolist(),'cosine_matrix':cosine.tolist(),'pairs':pairs,
            'median_rho':float(np.median([p['rho'] for p in pairs]))}


def factorize(h,g):
    mu,gbar = h.mean(0),g.mean(0)
    a = np.outer(gbar,mu)
    c = np.einsum('ni,nj->ij',g-gbar,h-mu)/len(h)
    return mu,gbar,a,c


def output_statistics(x):
    centered = x-x.mean(0)
    d = distances(x)
    return {'outputs':x.tolist(),'mean':x.mean(0).tolist(),'total_energy':float(np.sum(x*x)),
            'centered_energy':float(np.sum(centered*centered)),
            'same_image_distances':[float(d[i,i+1]) for i in range(0,len(x),2)],'pairwise_distances':d.tolist()}


def roundoff_check(calculated,saved):
    error = np.abs(calculated-saved)
    scale = float(np.max(np.abs(saved)))
    bound = 8*np.finfo(np.float32).eps*(scale+np.abs(saved))
    return {'passed':bool(np.all(error <= bound)),'max_absolute_error':float(np.max(error)),
            'frobenius_error':float(np.sqrt(np.sum(error*error))),'reference_scale':scale,
            'max_elementwise_bound':float(np.max(bound)),
            'max_error_over_bound':float(np.max(error/bound)) if scale else None,
            'tolerance':'8 * eps_float32 * (max_abs_saved + abs_saved_element); fixed before outcomes'}


def triage(rho,rc,rc_out):
    if rho < .10:
        return 'representation_insensitive_on_this_microset'
    if rc < .10 and rc_out < .10:
        return 'zero_head_common_gradient_collapse_despite_input_variation'
    return 'mixed_common_term_domination'


def analyze(arrays,weight,bias):
    h,g = arrays['features'],arrays['phi0_gradients']
    mu,gbar,a,c = factorize(h,g)
    gw = np.einsum('ni,nj->nij',g,h)
    eta = 1e-3
    components = {'bias_common_gradient':np.broadcast_to(-eta*gbar,(8,8)),
                  'mean_feature':-eta*(h@a.T),'gradient_feature_covariance':-eta*(h@c.T)}
    full = sum(components.values())
    checks = {
        'per_episode_weight_outer_product':roundoff_check(gw,arrays['head_weight_gradients']),
        'per_episode_bias_identity':roundoff_check(g,arrays['head_bias_gradients']),
        'saved_mean_weight_gradient':roundoff_check(a+c,arrays['head_weight_gradients'].mean(0)),
        'saved_joint_checkpoint_weight':roundoff_check(-eta*(a+c),weight),
        'saved_joint_checkpoint_bias':roundoff_check(-eta*gbar,bias),
        'saved_conditioning_output_sum':roundoff_check(full,arrays['saved_conditioning_outputs']),
        'saved_executed_phi0_sum':roundoff_check(full,arrays['saved_episode_phi0']),
    }
    an,cn = np.linalg.norm(a),np.linalg.norm(c)
    ah,ch = h@a.T,h@c.T
    ac,cc = ah-ah.mean(0),ch-ch.mean(0)
    rc,rc_out = float(cn/(an+cn)),float(np.linalg.norm(cc)/(np.linalg.norm(ac)+np.linalg.norm(cc)))
    cross = []
    names = list(components)
    for i,n in enumerate(names):
        for m in names[i+1:]:
            x,y = components[n],components[m]
            cross.append({'components':[n,m],'raw_cross_energy':float(2*np.sum(x*y)),
                          'centered_cross_energy':float(2*np.sum((x-x.mean(0))*(y-y.mean(0))))})
    features = feature_statistics(h)
    result = {'features':features,'G':g.tolist(),'g_bar':gbar.tolist(),'A':a.tolist(),'C':c.tolist(),
              'calculated_per_episode_weight_gradients':gw.tolist(),
              'saved_per_episode_weight_gradients':arrays['head_weight_gradients'].tolist(),
              'saved_per_episode_bias_gradients':arrays['head_bias_gradients'].tolist(),
              'G_W':(a+c).tolist(),'A_frobenius_norm':float(an),'C_frobenius_norm':float(cn),
              'A_C_cosine':float(np.sum(a*c)/(an*cn)),'A_C_cross_energy':float(2*np.sum(a*c)),
              'r_C':rc,'r_C_out':rc_out,'checks':checks,
              'float64_factorization_max_error':float(np.max(np.abs(gw.mean(0)-(a+c)))),
              'components':{n:output_statistics(v) for n,v in components.items()},'cross_terms':cross,
              'full_output':output_statistics(full),'saved_conditioning_outputs':arrays['saved_conditioning_outputs'].tolist(),
              'saved_executed_phi0':arrays['saved_episode_phi0'].tolist()}
    if all(v['passed'] for v in checks.values()):
        result['counterfactuals'] = {'common_only':output_statistics(components['bias_common_gradient']+components['mean_feature']),
                                    'covariance_only':output_statistics(-eta*((h-mu)@c.T))}
        result['triage'] = triage(features['median_rho'],rc,rc_out)
    else:
        result['triage'] = 'STOP_RECONSTRUCTION_FAILURE_NO_INTERPRETATION'
    return result


def run(prior_root,output):
    started = time.perf_counter()
    for name,expected in FILE_PINS.items():
        assert hashlib.sha256((prior_root/name).read_bytes()).hexdigest() == expected,name
    prior = json.loads((prior_root/'receipt.json').read_text())
    rows = prior['gradient_audit']['episodes']
    joint = prior['probes']['joint']
    arrays = {'features':joint['conditioning']['features'],'phi0_gradients':[r['phi0_gradient'] for r in rows],
              'head_weight_gradients':[r['head_weight_gradient'] for r in rows],
              'head_bias_gradients':[r['head_bias_gradient'] for r in rows],
              'saved_conditioning_outputs':joint['conditioning']['outputs'],
              'saved_episode_phi0':[r['phi0'] for r in joint['episodes']]}
    for name,value in arrays.items():
        raw = json.dumps(value,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
        assert hashlib.sha256(raw).hexdigest() == ARRAY_PINS[name],name
    # Deserialize small saved tensors only. No model is constructed or called.
    import torch
    original = torch.load(prior_root/'original_predictor.pt',map_location='cpu',weights_only=True)
    checkpoint = torch.load(prior_root/'joint_one_step_predictor.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,checkpoint[k]) for k,v in original.items() if k.startswith('features.'))
    assert torch.count_nonzero(original['head.weight']) == 0 and torch.count_nonzero(original['head.bias']) == 0
    result = analyze({k:np.array(v,dtype=np.float64) for k,v in arrays.items()},
                     checkpoint['head.weight'].double().numpy(),checkpoint['head.bias'].double().numpy())
    result['episodes'] = [{'image_id':r['image_id'],'case':r['case']} for r in rows]
    result['provenance'] = {'file_sha256':FILE_PINS,'array_sha256':ARRAY_PINS,'original_joint_trunk_identical':True,
                            'source_revision':os.environ.get('TAISP_SOURCE_REVISION'),'python':platform.python_version(),
                            'numpy':np.__version__,'algebra_dtype':'float64','algebra_device':'CPU',
                            'model_executions':0,'optimizer_steps':0,'elapsed_seconds':time.perf_counter()-started}
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'audit.json',result)
    assert all(v['passed'] for v in result['checks'].values()),'Saved artifact factorization mismatch; see audit.json. Do not relax tolerance.'
    write_json(output/'completion.json',{'status':'completed','triage':result['triage'],
               'median_rho':result['features']['median_rho'],'r_C':result['r_C'],'r_C_out':result['r_C_out'],
               'all_reconstructions_passed':True,**result['provenance']})
    print(json.dumps({'triage':result['triage'],'median_rho':result['features']['median_rho'],
                      'r_C':result['r_C'],'r_C_out':result['r_C_out']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    run(args.prior_root,args.output)
