import numpy as np
from taisp.analysis.gradient_basis_math import basis, scores, analyze, advancement


def test_orthogonal_span_unsigned_hard_and_zeros():
    e = np.eye(8)
    b = basis(e[0], e[1], e[2])
    s = scores(b, -e[0]+e[1])
    assert b['HCN']['rank'] == 3
    np.testing.assert_allclose([s['C_H'], s['C_HC'], s['C_HN'], s['C_HCN']], [1/np.sqrt(2), 1, 1/np.sqrt(2), 1], atol=1e-11)
    z = basis(np.zeros(8), np.zeros(8), np.zeros(8))
    assert z['HCN']['rank'] == 0 and all(v == 0 for v in scores(z, e[0]).values())
    assert all(v == 0 for v in scores(b, np.zeros(8)).values())
    assert basis(e[0], 2*e[0], -e[0])['HCN']['rank'] == 1


def test_paired_null_identity_and_stratified_generation():
    import importlib.util
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from prepare_t034a import permutations, FAMILIES
    records = [{'block': i//30, 'family': FAMILIES[i%3]} for i in range(120)]
    perms = permutations(records)
    assert perms == permutations(records)
    rows = []
    e = np.eye(8)
    for i, record in enumerate(records):
        for case in ['clean_s0', record['family']]:
            b = basis(e[0], e[1], e[2]); task = e[(i//3)%8]
            rows.append({**record, 'case': case, 'basis': b, 'task_gradient': task.tolist(), 'integrity_passed': True, **scores(b, task)})
    summary, null = analyze(rows, [list(range(120))]*256)
    np.testing.assert_allclose(null['E'][0], [r['E'] for r in rows], atol=1e-15)
    assert not summary['gates']['overall_HCN_above_null95']
    analyze(rows, perms)


def test_gate_conjunction_strict_null_and_rank_boundary():
    observed = {k: {'C_HCN': {'median': .70}, 'E': {'median': .10}} for k in ['overall', 'corrupt', 'block0', 'block1', 'block2', 'block3']}
    null = {k: {'C_HCN': .69, 'E': .09} for k in observed}
    assert advancement(observed, null, 216, 240, True)['decision'] == 'PASS'
    assert advancement(observed, null, 215, 240, True)['basis_collapse']
    null['overall']['E'] = .10
    assert advancement(observed, null, 240, 240, True)['decision'] == 'FAIL'
    assert advancement(observed, null, 240, 240, False)['decision'] == 'BLOCKED'
