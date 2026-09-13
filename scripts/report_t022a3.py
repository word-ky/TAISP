"""R037 offline complete comparison tables and artifact manifest."""
import argparse
import hashlib
import json
import statistics as st
from pathlib import Path
from report_t022a2 import aggregate


def render(project,run,prepare):
    raw=project/'research_log/remote_runs'/run;root=raw/'artifacts/confirmation'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    complete=read(root/'completion.json');tuples=read(root/'tuple_summary.json')
    assert complete['episodes']==80 and complete['evaluations']==0 and len(tuples)==8
    supports=read(project/'research_log/remote_runs'/prepare/'artifacts/prepare/supports.json')['rows']
    out=project/'research_log/T022A3';out.mkdir(exist_ok=True)
    tables='# T022-A3 complete output/state/update dispersion and latency\n\nPrimary metric: maximum pairwise relative L2 of final images; denominator is first member norm, floor1e-12. All10pairs per group. State/update dispersion is diagnostic only.\n\n'
    tables+='| ID | Condition | Current d | Spatial d | Ratio | Within2+floor | Above5+floor |\n| --- | --- | --- | --- | --- | --- | --- |\n'
    for r in tuples:tables+=f"| {r['image_id']} | {r['case']} | {r['d_cur']:.12g} | {r['d_sp']:.12g} | {r['ratio']} | {r['within2']} | {r['above5']} |\n"
    primary=tables
    tables+='\n| Tuple | Method | Field | Exact pairs | Max abs | Max relative L2 | Min cosine |\n| --- | --- | --- | --- | --- | --- | --- |\n'
    summaries={};latency=[];all_receipts=[]
    for i,t in enumerate(tuples):
        for method in ['current_ours','spatial_dose_ours']:
            ps=read(root/f't{i}_{method}_pairs.json');rs=read(root/f't{i}_{method}_receipts.json');assert len(rs)==5 and len(ps)==10
            assert all(all(r['isolation'].values()) and r['support_sha256']==supports[i]['support_sha256'] and r['mask_sha256']==supports[i]['mask_sha256'] for r in rs)
            assert all(len(r['diagnostics'])==4 for r in rs)
            a=aggregate(ps);summaries[f'{i}/{method}']=a;all_receipts+=rs
            for k,v in a.items():tables+=f"| {i}:{t['image_id']}/{t['case']} | {method} | {k} | {v['exact_pairs']}/10 | {v['max_absolute']:.12g} | {v['max_relative_l2']:.12g} | {v['minimum_cosine']} |\n"
            latency.append(dict(tuple=i,image_id=t['image_id'],case=t['case'],method=method,mean_seconds=st.mean(r['seconds'] for r in rs),median_seconds=st.median(r['seconds'] for r in rs),zero_updates=sum(not r['updated'] for r in rs),support_count=len(supports[i]['support']['boxes']),mask_area=supports[i]['mask_area']))
    tables+='\n| Tuple | Method | Mean s | Median s | Zero updates /5 | Supports | Mask area |\n| --- | --- | --- | --- | --- | --- | --- |\n'
    for r in latency:tables+=f"| {r['tuple']}:{r['image_id']}/{r['case']} | {r['method']} | {r['mean_seconds']:.9g} | {r['median_seconds']:.9g} | {r['zero_updates']}/5 | {r['support_count']} | {r['mask_area']} |\n"
    (out/'complete_tables.md').write_text(tables,encoding='utf-8');(out/'primary_table.md').write_text(primary,encoding='utf-8')
    summary=dict(completion=complete,tuples=tuples,dispersion=summaries,latency=latency,all80isolation_and_hash_receipts_passed=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    files=[]
    for name in [prepare,run]:
        for p in sorted((project/'research_log/remote_runs'/name).rglob('*')):
            if p.is_file():files.append(dict(path=p.relative_to(project).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    manifest=dict(file_count=len(files),total_bytes=sum(r['bytes'] for r in files),files=files,archives=dict(prepare='016981867c1d2168006b830528a5b3e4994d978777c14a97576567de216b9fdf',confirmation='fb9296465cc53ebb42df92e5594c04361f4d70034342424b03484cb5bc25c2ab'))
    (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return summary,manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,default=Path('.'));p.add_argument('--run',required=True);p.add_argument('--prepare',required=True)
    a=p.parse_args();render(a.project,a.run,a.prepare)
