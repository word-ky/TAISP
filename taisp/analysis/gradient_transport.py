"""R032 source-only closed-form fitting and held-out diagnostics; no optimizer."""
import itertools
import statistics

import torch


def fit_folds(records, *, eps=1e-12, device='cpu'):
    """Rows are image-condition pairs; all conditions of an image share one fold."""
    by_image = {}
    for row in records:
        previous = by_image.setdefault(row['image_id'], row['block'])
        assert previous == row['block'], 'image conditions split across folds'
    fits = {}
    for block in sorted(set(by_image.values())):
        train = [i for i, r in enumerate(records) if r['block'] != block]
        held = [i for i, r in enumerate(records) if r['block'] == block]
        p = torch.tensor([records[i]['pseudo_gradient'] for i in train], dtype=torch.float64, device=device)
        t = torch.tensor([records[i]['task_gradient'] for i in train], dtype=torch.float64, device=device)
        pn, tn = p.norm(dim=1, keepdim=True), t.norm(dim=1, keepdim=True)
        p, t = p/(pn+eps), t/(tn+eps)
        u, s, vh = torch.linalg.svd(p.T @ t)
        q = u @ vh
        error = (q.T @ q-torch.eye(8, device=device, dtype=q.dtype)).abs().max().item()
        assert error <= 1e-10 and torch.isfinite(q).all()
        fits[str(block)] = {'Q': q.cpu().tolist(), 'singular_values': s.cpu().tolist(),
            'determinant': torch.linalg.det(q).item(), 'orthogonality_max_abs': error,
            'fit_squared_residual': (p @ q-t).square().sum().item(),
            'train_indices': train, 'heldout_indices': held,
            'train_image_ids': sorted({records[i]['image_id'] for i in train}),
            'heldout_image_ids': sorted({records[i]['image_id'] for i in held}),
            'zero_pseudo_rows': int((pn == 0).sum()), 'zero_task_rows': int((tn == 0).sum()),
            'excluded_rows': 0}
    return fits


def heldout_alignment(records, fits):
    rows = []
    for r in records:
        f = fits[str(r['block'])]
        assert r['image_id'] in f['heldout_image_ids'] and r['image_id'] not in f['train_image_ids']
        p, t = [torch.tensor(r[k], dtype=torch.float64) for k in ('pseudo_gradient', 'task_gradient')]
        q = torch.tensor(f['Q'], dtype=torch.float64)
        raw = transported = None
        if p.norm() > 0 and t.norm() > 0:
            raw = (p @ t/(p.norm()*t.norm())).item()
            g = p @ q
            transported = (g @ t/(g.norm()*t.norm())).item()
        rows.append({k: r[k] for k in ('image_id', 'block', 'case')} |
                    {'raw_cosine': raw, 'transported_cosine': transported,
                     'improved': transported > raw if raw is not None else None})
    groups = {'overall': rows, 'corrupted': [r for r in rows if r['case'] != 'clean_s0'],
              'clean': [r for r in rows if r['case'] == 'clean_s0']}
    groups.update({f'block{b}': [r for r in rows if r['block'] == b] for b in sorted({r['block'] for r in rows})})
    groups.update({c: [r for r in rows if r['case'] == c] for c in sorted({r['case'] for r in rows})})
    summary = {}
    for name, selected in groups.items():
        valid = [r for r in selected if r['raw_cosine'] is not None]
        summary[name] = {'episodes': len(selected), 'defined': len(valid), 'undefined_zero': len(selected)-len(valid),
            'improved_fraction_defined': statistics.mean(r['improved'] for r in valid) if valid else None,
            'improved_count': sum(r['improved'] for r in valid)}
        for key in ('raw_cosine', 'transported_cosine'):
            summary[name][key] = {s: fn(r[key] for r in valid) if valid else None
                                 for s, fn in [('mean', statistics.mean), ('median', statistics.median)]}
    similarities = []
    for a, b in itertools.combinations(fits, 2):
        qa, qb = [torch.tensor(fits[k]['Q'], dtype=torch.float64) for k in (a, b)]
        similarities.append({'fold_a': a, 'fold_b': b, 'frobenius_distance': (qa-qb).norm().item(),
                             'frobenius_cosine': ((qa*qb).sum()/(qa.norm()*qb.norm())).item()})
    return {'rows': rows, 'groups': summary, 'matrix_similarities': similarities}


def transport_advancement(metrics, isolation_passed):
    from .run_t018a import advancement, CASE_NAMES
    candidate = 'grad_transport_ours'
    mapped = {g: {k.replace(candidate, 'nativePT_ours'): v for k, v in values.items()} for g, values in metrics.items()}
    result = advancement(mapped)
    for g, values in metrics.items():
        group = result['groups'][g]
        for key in ('macro_AP', 'clean_AP'):
            group[key][candidate] = group[key].pop('nativePT_ours')
        group['clean_delta_vs_raw'] = group['clean_AP'][candidate]-group['clean_AP']['no_adapt']
        group['additional_macro_metrics'] = {k: {m: 100*statistics.mean(values[f'{c}_{m}'][k] for c in CASE_NAMES[:-1])
            for m in ('no_adapt', 'current_ours', candidate)} for k in ('AP50', 'AP75')}
        group['additional_macro_deltas'] = {k: v[candidate]-v['current_ours'] for k,v in group['additional_macro_metrics'].items()}
    gate = result['gate']
    flags = gate['flags']
    flags.pop('corruption_macro_above_current'); flags.pop('corruption_macro_delta_at_least_point10')
    flags['corruption_macro_delta_at_least_point15'] = result['groups']['aggregate']['candidate_minus_current'] >= .15
    flags['no_leakage_isolation_norm_or_reproducibility_blocker'] = isolation_passed
    gate['passed'] = all(flags.values())
    gate['decision'] = 'development_candidate_pending_confirmation' if gate['passed'] else 'close_fixed_global_linear_gradient_transport'
    return result
