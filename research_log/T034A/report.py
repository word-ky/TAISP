"""Render R052 report solely from locked candidate and reference artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main(project, candidate_run, reference_run):
    out = project/'research_log/T034A'
    cr = project/'research_log/remote_runs'/candidate_run
    rr = project/'research_log/remote_runs'/reference_run
    croot, rroot = cr/'artifacts/candidates', rr/'artifacts/reference'
    verified = {}
    for name, root in [('candidate', croot), ('reference', rroot)]:
        manifest = read(root/'sha256.json')
        for f, h in manifest.items():
            assert sha(root/f) == h, f
        verified[name] = len(manifest)
    cs, rows, summary = read(croot/'records.json'), read(rroot/'records.json'), read(rroot/'summary.json')
    lock = read(project/'research_log/T034A_candidate_lock.json')
    assert sha(croot/'records.json') == lock['files']['records.json']
    assert len(cs) == len(rows) == 240 and all(r['integrity_passed'] for r in cs+rows)
    assert all(not r['candidate_recomputed'] for r in rows)
    pins = read(project/'research_log/T034A_all_code_pins.json')
    for f, h in pins.items():
        assert hashlib.sha256((project/f).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == h, f
    diagnostics = {}
    for group in summary['observed']:
        if group == 'overall': chosen = list(range(240))
        elif group == 'clean': chosen = [i for i,r in enumerate(rows) if r['case']=='clean_s0']
        elif group == 'corrupt': chosen = [i for i,r in enumerate(rows) if r['case']!='clean_s0']
        elif group.startswith('block'): chosen = [i for i,r in enumerate(rows) if r['block']==int(group[-1])]
        else: chosen = [i for i,r in enumerate(rows) if r['family']==group.removeprefix('family_')]
        pair = {}
        for k in ['hard_clip', 'hard_native', 'clip_native']:
            values = [cs[i]['pairwise_cosines'][k] for i in chosen if cs[i]['pairwise_cosines'][k] is not None]
            pair[k] = {'nonzero_pairs': len(values), 'median': statistics.median(values) if values else None,
                       'mean': statistics.mean(values) if values else None, 'values': values}
        diagnostics[group] = {'pairs': pair, 'zeros': {k:sum(cs[i]['zero_flags'][k] for i in chosen) for k in ['hard','clip','native']},
            'rank_histogram': {str(k):sum(rows[i]['basis']['HCN']['rank']==k for i in chosen) for k in range(4)}}
    (out/'pairwise_diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n', encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    fields = ['episode_index','image_id','case','family','block','C_H','C_HC','C_HN','C_HCN','E','integrity_passed']
    with (out/'per_episode.tsv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    ce, re = read(croot/'environment.json'), read(rroot/'environment.json')
    cc, rc = read(croot/'completion.json'), read(rroot/'completion.json')
    lines = ['# T034-A / R052 — '+summary['decision']+'; NEEDS_REVIEW', '',
        'Oracle ceiling audit only. All240 H/C/N candidate records were committed before source annotations and task gradients. No candidate regeneration, learned mixer, finite ISP step, AP or deployment change.', '',
        '## Frozen gate', '', '| Check | Result |', '|---|---|']
    for k,v in summary['gates'].items(): lines.append(f'| {k} | {v} |')
    lines += ['',f"Rank histogram: {summary['rank_histogram']}; rank>=2: {summary['rank_ge2_count']}/240; BASIS_COLLAPSE={summary['basis_collapse']}; zero tasks={summary['zero_task_count']}; blocks passed={summary['blocks_passed']}/4.", '',
              '## Observed medians and matched null', '', '| Group | N | C_H | C_HC | C_HN | C_HCN | HCN null95 | E | E null95 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k,v in summary['observed'].items():
        n=summary['null95'][k]
        lines.append(f"| {k} | {v['count']} | {v['C_H']['median']:.9f} | {v['C_HC']['median']:.9f} | {v['C_HN']['median']:.9f} | {v['C_HCN']['median']:.9f} | {n['C_HCN']:.9f} | {v['E']['median']:.9f} | {n['E']:.9f} |")
    lines += ['', 'Means/min/max for all nested ceilings and E, all256 null-median distributions for every group, all256x240 null H/HCN/E values, per-episode SVD bases/spectra/rank/tolerance and task vectors are retained in summary and raw records. Exact nonzero candidate normalization, epsilon task normalization, float64 SVD and linear null quantiles follow the precommit.', '',
              '## Pairwise candidate cosine medians and zeros', '', '| Group | H-C | H-N | C-N | Zero H/C/N |', '|---|---:|---:|---:|---|']
    for k,d in diagnostics.items():
        med = [d['pairs'][p]['median'] for p in ['hard_clip','hard_native','clip_native']]
        lines.append(f"| {k} | {med[0]} | {med[1]} | {med[2]} | {d['zeros']} |")
    lines += ['', '## Integrity and execution', '',
        f"Cohort plan commit d14a1f31f63ee700eb0ab9432ed9a71d401351c0; candidate code {ce['source_revision']}; reference code committed5fc938974506cb03d7d3f503d579ebd3ff43b850 before reference reveal.",
        f"Candidate commit {lock['candidate_commit']}; lock SHA {sha(project/'research_log/T034A_candidate_lock.json')}; reference process source revision {re['source_revision']}.",
        f"Candidate run {candidate_run}: {cc['seconds']}s; reference run {reference_run}: {rc['seconds']}s. Model device {ce['device']} / {ce['gpu']}; CUDA {ce['cuda']}; torch {ce['torch']}. CPU used for fixed NumPy SVD/permutation statistics only.",
        f"Every240 candidate and240 reference state/RNG/JVP check passed. Raw verified files {verified}; {len(pins)} protected/new code pins unchanged. Detector state {ce['source_state_sha256']}; CLIP state {ce['clip_state_sha256']}; CLIP weight {ce['clip_weight_sha256']}.",
        'Native literal fullscalar is authoritative. Component-additivity is retained as a non-gating R047 diagnostic. Support boxes/labels/weights, pseudo targets, source/CLIP state and RNG receipts, zero flags and direct-reverse comparisons retained.',
        f"Candidate records SHA {sha(croot/'records.json')}; reference records SHA {sha(rroot/'records.json')}; null raw SHA {sha(rroot/'null_episode_scores.json')}.", '',
        '## Validation', '', 'Donor baseline:11passed1skipped5.75s. Candidate+native focused:9passed8.53s. Span/null math:3passed2.03s. Reference order+math:4passed4.75s. Full suite receipt:research_log/T034A_full_tests.txt. Opt-in real-model tests skipped by default; actual240 candidate and240 reference episodes are the real CUDA integration.', '',
        '## Decision boundary', '']
    if summary['decision']=='PASS':
        lines.append('Full frozen conjunction passes. This supports only sample-specific task-direction information in the existing H/C/N span. Stop NEEDS_REVIEW before coefficient learning or AP.')
    else:
        lines.append('Full frozen conjunction fails. Close the exact old hard/CLIP/native gradient-basis mixing direction. No basis expansion, nonlinear mixer, threshold changes or outcome-driven rescue. Stop NEEDS_REVIEW for a new research instruction.')
    lines += ['', 'Operational receipts: one transient SSH log-read disconnect; status confirmed the same candidate process continued, no restart. Preparer import precedence repaired before cohort generation. No numerical protocol change or deleted recovery artifact.', '']
    (project/'research_log/T034A_report.md').write_text('\n'.join(lines), encoding='utf-8')
    paths = set(p for p in project.glob('research_log/T034A*') if p.is_file())
    paths.update(p for p in out.rglob('*') if p.is_file())
    paths.update(p for root in [cr,rr] for p in root.rglob('*') if p.is_file())
    paths.update(project/f for f in pins)
    manifest = {str(p.relative_to(project)).replace('\\','/'):sha(p) for p in sorted(paths) if p.name!='artifact_manifest.json'}
    (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'decision':summary['decision'], 'verified_raw':verified, 'artifacts':len(manifest), 'report_sha256':sha(project/'research_log/T034A_report.md')}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True);p.add_argument('--candidate-run',required=True);p.add_argument('--reference-run',required=True)
    a=p.parse_args();main(a.project,a.candidate_run,a.reference_run)
