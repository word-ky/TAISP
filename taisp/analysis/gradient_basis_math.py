"""R052 float64 span ceilings and frozen paired permutation null; no models."""
import numpy as np


def unit_candidate(g):
    g = np.asarray(g, dtype=np.float64)
    n = np.linalg.norm(g)
    return g/n if n > 0 else np.zeros_like(g)


def span(columns):
    matrix = np.column_stack([unit_candidate(g) for g in columns])
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    tol = np.finfo(np.float64).eps*max(matrix.shape)*s[0]
    rank = int(np.count_nonzero(s > tol))
    return {'Q': u[:, :rank].tolist(), 'singular_values': s.tolist(), 'rank': rank, 'tolerance': float(tol)}


def basis(hard, clip, native):
    return {'u_hard': unit_candidate(hard).tolist(), 'HCN': span([hard, clip, native]),
            'HC': span([hard, clip]), 'HN': span([hard, native])}


def scores(b, task):
    t = np.asarray(task, dtype=np.float64)
    u = t/(np.linalg.norm(t)+1e-12)
    result = {'C_H': float(abs(np.asarray(b['u_hard'])@u))}
    for key in ['HC', 'HN', 'HCN']:
        result['C_'+key] = float(np.linalg.norm(np.asarray(b[key]['Q']).T@u))
    result['E'] = result['C_HCN']-result['C_H']
    return result


def groups(rows):
    return {'overall': list(range(len(rows))),
            'clean': [i for i, r in enumerate(rows) if r['case'] == 'clean_s0'],
            'corrupt': [i for i, r in enumerate(rows) if r['case'] != 'clean_s0'],
            **{'family_'+f: [i for i, r in enumerate(rows) if r['family'] == f]
               for f in ['gamma_s2', 'contrast_s2', 'color_cast_s2']},
            **{'block'+str(b): [i for i, r in enumerate(rows) if r['block'] == b] for b in range(4)}}


def advancement(observed, null95, rank_count, count, integrity):
    gates = {}
    for name in ['overall', 'corrupt']:
        hcn, extra = observed[name]['C_HCN']['median'], observed[name]['E']['median']
        gates[name+'_HCN_ge_070'] = hcn >= .70
        gates[name+'_HCN_above_null95'] = hcn > null95[name]['C_HCN']
        gates[name+'_E_ge_010'] = extra >= .10
        gates[name+'_E_above_null95'] = extra > null95[name]['E']
    blocks = [observed['block'+str(b)]['E']['median'] > max(.05, null95['block'+str(b)]['E']) for b in range(4)]
    gates['three_blocks'] = sum(blocks) >= 3
    gates['rank_ge2_90percent'] = rank_count >= .9*count
    gates['integrity'] = bool(integrity)
    return {'gates': gates, 'block_pass': blocks, 'blocks_passed': sum(blocks),
            'basis_collapse': not gates['rank_ge2_90percent'],
            'decision': 'BLOCKED' if not integrity else ('PASS' if all(gates.values()) else 'FAIL')}


def analyze(rows, pair_permutations):
    assert len(rows) == 240 and len(pair_permutations) == 256
    bs = [r['basis'] for r in rows]
    tasks = np.asarray([r['task_gradient'] for r in rows], dtype=np.float64)
    uts = tasks/(np.linalg.norm(tasks, axis=1, keepdims=True)+1e-12)
    hard = np.asarray([b['u_hard'] for b in bs])
    # Zero-padded Q preserves exactly the SVD subspace and its empty-span behavior.
    q = np.zeros((240, 8, 3), dtype=np.float64)
    for i, b in enumerate(bs):
        rank = b['HCN']['rank']
        q[i, :, :rank] = np.asarray(b['HCN']['Q'])
    null = {k: [] for k in ['C_HCN', 'C_H', 'E']}
    for pair_map in pair_permutations:
        assert len(pair_map) == 120 and sorted(pair_map) == list(range(120))
        for target, source in enumerate(pair_map):
            for parity in [0, 1]:
                a, b = rows[2*target+parity], rows[2*source+parity]
                assert (a['block'], a['family'], a['case']) == (b['block'], b['family'], b['case'])
        indices = np.asarray([[2*p, 2*p+1] for p in pair_map]).reshape(-1)
        u = uts[indices]
        ch = np.abs(np.einsum('ni,ni->n', hard, u))
        hcn = np.linalg.norm(np.einsum('nij,ni->nj', q, u), axis=1)
        for k, v in [('C_HCN', hcn), ('C_H', ch), ('E', hcn-ch)]:
            null[k].append(v.tolist())
    masks = groups(rows)
    observed, null_medians, null95 = {}, {}, {}
    for name, indices in masks.items():
        observed[name] = {'count': len(indices)}
        for k in ['C_H', 'C_HC', 'C_HN', 'C_HCN', 'E']:
            v = [rows[i][k] for i in indices]
            observed[name][k] = {'median': float(np.median(v)), 'mean': float(np.mean(v)), 'min': float(np.min(v)), 'max': float(np.max(v))}
        null_medians[name] = {k: np.median(np.asarray(v)[:, indices], axis=1).tolist() for k, v in null.items()}
        null95[name] = {k: float(np.quantile(v, .95, method='linear')) for k, v in null_medians[name].items()}
    rank_count = sum(b['HCN']['rank'] >= 2 for b in bs)
    summary = {'observed': observed, 'null95': null95, 'null_medians': null_medians,
               'rank_histogram': {str(k): sum(b['HCN']['rank'] == k for b in bs) for k in range(4)},
               'rank_ge2_count': rank_count, 'zero_task_count': int(np.count_nonzero(np.linalg.norm(tasks, axis=1) == 0)),
               'numpy_version': np.__version__, 'permutations': 256, 'quantile_method': 'linear'}
    summary.update(advancement(observed, null95, rank_count, len(rows), all(r['integrity_passed'] for r in rows)))
    return summary, null
