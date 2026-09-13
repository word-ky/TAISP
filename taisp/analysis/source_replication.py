"""T013-H saved-array float64 replication and image-block cross-fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .predictor_factorization import factorize, feature_statistics, output_statistics, roundoff_check, write_json


def cosine(x, y):
    denominator = np.linalg.norm(x, axis=-1)*np.linalg.norm(y, axis=-1)
    return np.divide(np.sum(x*y, axis=-1), denominator,
                     out=np.zeros_like(denominator), where=denominator != 0)


def statistics(h, g):
    mu, gbar, a, c = factorize(h, g)
    parts = {'bias_common_gradient': np.broadcast_to(-.001*gbar, g.shape),
             'mean_feature': -.001*h@a.T, 'gradient_feature_covariance': -.001*h@c.T}
    an, cn = np.linalg.norm(a), np.linalg.norm(c)
    ah, ch = (h-mu)@a.T, (h-mu)@c.T
    cross = []
    names = list(parts)
    for i, n in enumerate(names):
        for m in names[i+1:]:
            x, y = parts[n], parts[m]
            cross.append({'components': [n,m], 'raw_cross_energy': float(2*np.sum(x*y)),
                          'centered_cross_energy': float(2*np.sum((x-x.mean(0))*(y-y.mean(0))))})
    full = output_statistics(sum(parts.values()))
    full['centered_fraction'] = full['centered_energy']/full['total_energy']
    return {'features': feature_statistics(h), 'g_bar': gbar.tolist(), 'A': a.tolist(), 'C': c.tolist(),
            'A_frobenius_norm': float(an), 'C_frobenius_norm': float(cn),
            'A_C_cosine': float(cosine(a.ravel(), c.ravel())), 'A_C_cross_energy': float(2*np.sum(a*c)),
            'r_C': float(cn/(an+cn)), 'r_C_out': float(np.linalg.norm(ch)/(np.linalg.norm(ah)+np.linalg.norm(ch))),
            'components': {k: output_statistics(v) for k,v in parts.items()}, 'cross_terms': cross,
            'full_output': full, 'clean_corrupted_gradient_cosine': float(cosine(g[::2].mean(0), g[1::2].mean(0)))}


def fold_indices():
    indices = np.arange(64)
    return [(indices[indices//16 != fold], indices[indices//16 == fold]) for fold in range(4)]


def crossfit(h, g):
    folds, records = [], []
    for fold, (train, test) in enumerate(fold_indices()):
        mu, gbar, a, c = factorize(h[train], g[train])
        deltas = {'full': -.001*(gbar+h[test]@(a+c).T), 'common': -.001*(gbar+h[test]@a.T),
                  'cov': -.001*(h[test]-mu)@c.T}
        folds.append({'fold': fold, 'train_indices': train.tolist(), 'test_indices': test.tolist(),
                      'mu': mu.tolist(), 'g_bar': gbar.tolist(), 'A': a.tolist(), 'C': c.tolist(),
                      'perturbations': {name: output_statistics(v) for name,v in deltas.items()}})
        for j, index in enumerate(test):
            records.append({'episode_index': int(index), 'fold': fold, 'clean': bool(index%2 == 0),
                            'directions': {name: {'delta': v[j].tolist(), 'g_dot_delta': float(g[index]@v[j]),
                            'cosine_to_negative_g': float(cosine(v[j], -g[index])), 'norm': float(np.linalg.norm(v[j])),
                            'same_image_separation': float(np.linalg.norm(v[j]-v[j^1]))} for name,v in deltas.items()}})
    return {'folds': folds, 'records': records}


def utility(records):
    chosen = [r['directions']['cov'] for r in records]
    total = sum(r['g_dot_delta'] < 0 for r in chosen)
    clean = sum(r['directions']['cov']['g_dot_delta'] < 0 for r in records if r['clean'])
    corrupt = total-clean
    median = float(np.median([r['cosine_to_negative_g'] for r in chosen]))
    return {'negative_count': total, 'clean_negative_count': clean, 'corrupted_negative_count': corrupt,
            'median_cosine': median, 'passed': total >= 40 and clean >= 18 and corrupt >= 18 and median > 0}


def decision(overall, blocks, utility_result):
    sensitivity = overall['features']['median_rho'] >= .10
    covariance = overall['r_C'] >= .10 or overall['r_C_out'] >= .10
    small_blocks = sum(b['full_output']['centered_fraction'] < .10 for b in blocks)
    common = overall['full_output']['centered_fraction'] < .05 and small_blocks >= 3
    structural = sensitivity and covariance and common
    return {'condition_sensitivity': sensitivity, 'non_negligible_covariance': covariance,
            'blocks_below_10_percent': small_blocks, 'common_mode_replication': common,
            'structural_replication': structural, 'crossfit_utility': utility_result['passed'],
            'both_pass': structural and utility_result['passed'], 'stop_for_research_review': True}


def run(records_path, output):
    records = json.loads(records_path.read_text())['records']
    assert len(records) == 64
    h = np.array([r['features'] for r in records], dtype=np.float64)
    g = np.array([r['phi0_gradient'] for r in records], dtype=np.float64)
    gw = np.array([r['head_weight_gradient'] for r in records], dtype=np.float64)
    gb = np.array([r['head_bias_gradient'] for r in records], dtype=np.float64)
    _, _, a, c = factorize(h, g)
    checks = {'per_episode_weight_outer_product': roundoff_check(np.einsum('ni,nj->nij',g,h),gw),
              'per_episode_bias_identity': roundoff_check(g,gb),
              'mean_weight_gradient': roundoff_check(a+c,gw.mean(0))}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'reconstruction.json', checks)
    assert all(v['passed'] for v in checks.values()), 'Saved gradient mismatch; no tolerance change.'
    overall = statistics(h,g)
    blocks = [statistics(h[i:i+16],g[i:i+16]) for i in range(0,64,16)]
    fitted = crossfit(h,g)
    util = utility(fitted['records'])
    result = {'overall': overall, 'blocks': blocks, 'crossfit': fitted, 'utility': util,
              'decision': decision(overall,blocks,util), 'reconstruction': checks,
              'episodes': [{'image_id':r['image_id'],'case':r['case']} for r in records],
              'provenance': {'records_sha256': hashlib.sha256(records_path.read_bytes()).hexdigest(),
                             'dtype': 'float64', 'device': 'CPU', 'optimizer_steps': 0, 'counterfactual_model_calls': 0}}
    write_json(output/'audit.json', result)
    write_json(output/'completion.json', {'status':'completed', 'decision':result['decision'], 'utility':util})
    print(json.dumps({'decision':result['decision'],'utility':util}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.records,args.output)
