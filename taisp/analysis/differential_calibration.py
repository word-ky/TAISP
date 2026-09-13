"""T016-A: cross-fitted diagonal calibration of frozen differential gradients."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .differential_subspace import cosine, describe, write_json
from .linear_capacity import pair_rows
from .spatial_action import EPS


def folds():
    indices = np.arange(32)
    return [(indices[indices // 8 != f], indices[indices // 8 == f]) for f in range(4)]


def fit(dp, dt):
    numerator = np.sum(dp * dt, axis=0)
    denominator = np.sum(dp**2, axis=0) + EPS
    return {'a': (numerator / denominator).tolist(), 'numerator': numerator.tolist(),
            'denominator': denominator.tolist()}


def crossfit(dp, dt, permutations=None):
    prediction = np.empty_like(dp)
    receipts = []
    for f, (train, test) in enumerate(folds()):
        targets = train if permutations is None else train[pair_rows(permutations[f])]
        fitted = fit(dp[train], dt[targets])
        prediction[test] = dp[test] * np.asarray(fitted['a'])
        receipts.append({'fold': f, 'train_indices': train.tolist(), 'test_indices': test.tolist(),
                         'target_indices': targets.tolist(), **fitted})
    return prediction, receipts


def evaluate(st, dt, sp, dp, predicted, raw_d):
    tdiff, pdiff = np.r_[dt, -dt], np.r_[dp, -dp]
    pshared, t = np.r_[sp, sp], np.r_[st + dt, st - dt]
    p = pshared + pdiff
    qdiff = np.r_[predicted, -predicted]
    nd, nqdiff, npfull = (float(np.linalg.norm(x)) for x in (pdiff, qdiff, p))
    qnm = nd * qdiff / (nqdiff + EPS)
    q = pshared + qnm
    n_nm, nq = float(np.linalg.norm(qnm)), float(np.linalg.norm(q))
    diff_error, full_error = abs(n_nm - nd), abs(nq - npfull)
    diff_bound, full_bound = 1e-10 + 1e-8 * nd, 1e-10 + 1e-8 * npfull
    norm_check = {'diff_absolute_error': diff_error, 'full_absolute_error': full_error,
                  'diff_bound': diff_bound, 'full_bound': full_bound,
                  'diff_error_over_bound': diff_error / diff_bound,
                  'full_error_over_bound': full_error / full_bound,
                  'EPS_diff_shrinkage': nd * EPS / (nqdiff + EPS),
                  'passed': diff_error <= diff_bound and full_error <= full_bound}
    raw_dot, cal_dot = float(dt @ dp), float(dt @ predicted)
    c_raw = float(tdiff @ pdiff) / (npfull + EPS)
    c_cal = float(tdiff @ qnm) / (nq + EPS)
    d_cal = float(t @ q) / (nq + EPS)
    metrics = {'cos_diff_raw': cosine(dt, dp), 'cos_diff_cal': cosine(dt, predicted),
               'diff_dot_raw': raw_dot, 'diff_dot_cal': cal_dot,
               'C_diff_raw': c_raw, 'C_diff_cal': c_cal, 'D_spatial_raw': raw_d,
               'D_cal': d_cal, 'Delta_D_cal': d_cal - raw_d,
               'raw_diff_norm': nd, 'cal_diff_norm': n_nm,
               'raw_full_norm': npfull, 'cal_full_norm': nq,
               'diff_norm_error': diff_error, 'full_norm_error': full_error}
    return {'metrics': metrics, 'q_diff_nm': qnm.tolist(), 'q': q.tolist(),
            'predicted_d': predicted.tolist(), 'norm_check': norm_check,
            'raw_diff_dot_sign': int(np.sign(raw_dot)), 'diff_dot_sign': int(np.sign(cal_dot)),
            'zero_norm': {'raw_cosine': cosine(dt, dp) is None,
                          'cal_cosine': cosine(dt, predicted) is None}}


def summary(rows):
    groups = {'overall': rows, 'clean': [r for r in rows if r['case'] == 'clean_s0'],
              'corrupted': [r for r in rows if r['case'] != 'clean_s0']}
    groups.update({f'block{i}': [r for r in rows if r['block'] == i] for i in range(4)})
    groups.update({'case/' + case: [r for r in rows if r['case'] == case]
                   for case in sorted({r['case'] for r in rows})})
    return {name: describe(selected) for name, selected in groups.items()}


def pooled(scopes):
    m = scopes['overall']['metrics']
    return {'median_Delta_D_cal': m['Delta_D_cal']['median'],
            'positive_Delta_D_cal': m['Delta_D_cal']['positive_count'],
            'positive_C_diff_cal': m['C_diff_cal']['positive_count'],
            'median_cos_diff_cal': m['cos_diff_cal']['median']}


def null_comparison(observed, values):
    v = np.asarray(values, dtype=np.float64)
    return {'observed': observed, 'values': values,
            'null_95th_percentile_linear': float(np.quantile(v, .95, method='linear')),
            'null_at_least_observed': int(np.sum(v >= observed)), 'ties': int(np.sum(v == observed)),
            'empirical_percentile_strict': float(100 * np.mean(v < observed)),
            'corrected_one_sided_tail': float((1 + np.sum(v >= observed)) / (1 + len(v)))}


def gate(scopes, comparisons):
    flags, blocks = {}, {}
    for metric in ('C_diff_cal', 'Delta_D_cal'):
        overall, corrupt = scopes['overall']['metrics'][metric], scopes['corrupted']['metrics'][metric]
        blocks[metric] = sum(scopes[f'block{i}']['metrics'][metric]['median'] > 0 for i in range(4))
        flags.update({metric + '_overall_positive_ge20': overall['positive_count'] >= 20,
                      metric + '_corrupted_positive_ge10': corrupt['positive_count'] >= 10,
                      metric + '_overall_median_positive': overall['median'] > 0,
                      metric + '_at_least3_positive_block_medians': blocks[metric] >= 3})
    for key in ('median_Delta_D_cal', 'positive_Delta_D_cal'):
        flags[key + '_above_null95'] = comparisons[key]['observed'] > comparisons[key]['null_95th_percentile_linear']
    passed = all(flags.values())
    return {'flags': flags, 'positive_block_medians': blocks, 'passed': passed,
            'decision': 'replicate_exact_diagonal_form_on_future_precommitted_source_cohort' if passed
                        else 'close_diagonal_source_calibration_rescue',
            'stop_for_research_review': True}


def calibrate(rows, permutations=None):
    arrays = {name: np.asarray([r['analysis'][obj][part] for r in rows], dtype=np.float64)
              for name, obj, part in [('st', 'task', 's'), ('dt', 'task', 'd'),
                                      ('sp', 'pseudo', 's'), ('dp', 'pseudo', 'd')]}
    prediction, fits = crossfit(arrays['dp'], arrays['dt'], permutations)
    result = []
    for i, r in enumerate(rows):
        out = {k: r[k] for k in ('episode_index', 'image_id', 'case', 'block', 'support_count', 'mask_area_fraction')}
        out['analysis'] = evaluate(*(arrays[k][i] for k in ('st', 'dt', 'sp', 'dp')),
                                   prediction[i], r['analysis']['metrics']['D_spatial_saved'])
        result.append(out)
    return {'folds': fits, 'records': result, 'scopes': summary(result),
            'all_norm_checks_passed': all(r['analysis']['norm_check']['passed'] for r in result)}


def coefficient_stability(fitted):
    values = np.asarray([f['a'] for f in fitted])
    return [{'coordinate': k, 'values': values[:, k].tolist(),
             'signs': np.sign(values[:, k]).astype(int).tolist(),
             'mean': float(values[:, k].mean()), 'std_population': float(values[:, k].std()),
             'min': float(values[:, k].min()), 'max': float(values[:, k].max()),
             'positive_folds': int(np.sum(values[:, k] > 0)), 'negative_folds': int(np.sum(values[:, k] < 0)),
             'zero_folds': int(np.sum(values[:, k] == 0))} for k in range(8)]


def run(prior_root, manifest_path, schedule_path, output):
    started = time.perf_counter()
    manifest = json.loads(manifest_path.read_text())
    for receipt in manifest.values():
        assert hashlib.sha256((prior_root / receipt['path']).read_bytes()).hexdigest() == receipt['sha256']
    rows = json.loads((prior_root / manifest['t015_records']['path']).read_text())['records']
    assert len(rows) == 32 and [r['episode_index'] for r in rows] == list(range(32))
    for i in range(16):
        x, y = rows[2*i:2*i+2]
        assert x['image_id'] == y['image_id'] and x['case'] == 'clean_s0' and y['case'] != 'clean_s0'
        assert x['block'] == y['block'] == i // 4
    schedule = json.loads(schedule_path.read_text())
    rng = np.random.default_rng(20260913)
    assert schedule['seed'] == 20260913
    assert schedule['permutations'] == [[rng.permutation(12).tolist() for _ in range(4)] for _ in range(256)]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'provenance.json', {'inputs': manifest, 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               'schedule_sha256': hashlib.sha256(schedule_path.read_bytes()).hexdigest(),
               'source_revision': os.environ.get('TAISP_SOURCE_REVISION'),
               'python': platform.python_version(), 'numpy': np.__version__, 'dtype': 'float64', 'device': 'cpu',
               'EPS': EPS, 'norm_atol': 1e-10, 'norm_rtol': 1e-8, 'seed': 20260913,
               'model_calls': 0, 'optimizer_steps': 0})
    write_json(output / 'permutation_schedule.json', schedule)
    observed = calibrate(rows)
    observed['coefficient_stability'] = coefficient_stability(observed['folds'])
    write_json(output / 'observed.json', observed)
    assert observed['all_norm_checks_passed'], 'Observed norm preservation blocker; no tolerance change'
    null = []
    for index, permutations in enumerate(schedule['permutations']):
        fitted = calibrate(rows, permutations)
        values = pooled(fitted['scopes'])
        # Full null coefficients, training mappings and heldout vectors are retained.
        del fitted['scopes']
        write_json(output / f'permutation_{index:03d}.json', {'index': index, **fitted, 'pooled': values})
        assert fitted['all_norm_checks_passed'], f'Null {index} norm preservation blocker; no tolerance change'
        null.append(values)
    primary = pooled(observed['scopes'])
    comparisons = {k: null_comparison(v, [n[k] for n in null]) for k, v in primary.items()}
    result = {'scopes': observed['scopes'], 'comparisons': comparisons,
              'gate': gate(observed['scopes'], comparisons), 'coefficient_stability': observed['coefficient_stability']}
    write_json(output / 'summary.json', result)
    write_json(output / 'completion.json', {'status': 'completed', 'gate': result['gate'],
               'permutations': len(null), 'closed_form_fits': 4 + 4 * len(null), 'heldout_primary_records': 32,
               'model_calls': 0, 'optimizer_steps': 0, 'all_norm_checks_passed': True,
               'elapsed_seconds': time.perf_counter() - started})
    print(json.dumps({'gate': result['gate'], 'pooled': primary, 'comparisons':
                      {k: {n: v for n, v in c.items() if n != 'values'} for k, c in comparisons.items()}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prior-root', 'manifest', 'schedule', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.prior_root, args.manifest, args.schedule, args.output)
