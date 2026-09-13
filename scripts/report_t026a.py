"""T026-A report from pinned memory/candidate/reference artifacts, stdlib only."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    root=Path(__file__).resolve().parents[1];out=root/'research_log/T026A'
    runs={'memory':'20260914-065856-taisp-t026a-memory1280','candidates':'20260914-070245-taisp-t026a-candidates48',
          'reference':'20260914-070746-taisp-t026a-reference48'}
    dirs={k:root/f'research_log/remote_runs/{v}/artifacts/{k}' for k,v in runs.items()}
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in dirs.values():
        for f,h in read(folder/'sha256.json').items():assert sha(folder/f)==h
    cs=read(dirs['candidates']/'records.json');rs=read(dirs['reference']/'records.json');s=read(dirs['reference']/'summary.json')
    health=read(dirs['memory']/'health.json');name=s['candidate'];table=[]
    for c,r in zip(cs,rs,strict=True):
        assert (c['image_id'],c['case'])==(r['image_id'],r['case'])
        obj=c['objectives'][name];m=r['metrics'][name];g=c['gradients'][name]['object'];t=r['reference_gradients']['task']['object']
        assert math.fsum(a*b for a,b in zip(t,g))/(math.sqrt(math.fsum(v*v for v in g))+1e-12)==m['S_mem']
        similarities=r['retrieval']['cosines']
        table.append({'episode':r['episode_index'],'image_id':r['image_id'],'case':r['case'],'block':r['block'],
            'supports':r['support_count'],'mask_area':r['mask']['area_fraction'],'loss':obj['loss'],
            'per_object_min':min(obj['per_object']) if obj['per_object'] else None,
            'per_object_median':statistics.median(obj['per_object']) if obj['per_object'] else None,
            'per_object_max':max(obj['per_object']) if obj['per_object'] else None,
            **{k:v for k,v in m.items() if k!='norms'},**{f'norm_{k}':v for k,v in m['norms'].items()},
            'retrieval_cosine_mean':statistics.mean(similarities) if similarities else None,
            'retrieval_cosine_min':min(similarities) if similarities else None,
            'distinct_memory_entries':len(r['retrieval']['entry_ids']),'distinct_classes':len(r['retrieval']['classes']),
            **{k:v for k,v in r['retrieval'].items() if k not in ('cosines','classes','entry_ids')}})
    with (out/'per_episode.tsv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]),delimiter='\t');writer.writeheader();writer.writerows(table)
    (out/'summary.json').write_bytes((dirs['reference']/'summary.json').read_bytes())
    plan=read(root/'research_log/T026A_plan.json');pins={**plan['protected_modules_LF'],**plan['inherited_analysis_LF']}
    local={p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h for p,h in pins.items()}
    assert all(local.values()) and all(read(out/'remote_code_checks.json').values())
    integrity={'local_protected_and_inherited':local,'remote_checks':read(out/'remote_code_checks.json'),
        'all48_integrity':all(r['integrity_passed'] for r in rs),'memory_health':health,
        'max_reference_relative_l2':max(r['reference_checks'][k]['parity']['relative_l2_error'] for r in rs for k in ('task','pseudo')),
        'max_partition_absolute_error':max(r['reference_checks'][k]['partition']['max_absolute_error'] for r in rs for k in ('task','pseudo')),
        'empty_supports':sum(r['support_count']==0 for r in rs),'minimum_candidate_norm':min(r['norm_candidate'] for r in table)}
    (out/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n',encoding='utf-8')
    a,b=s['overall'],s['corrupt'];nb=sum(v['median_Delta_global']>0 for v in s['blocks'].values())
    timings={k:read(p/'completion.json')['seconds'] for k,p in dirs.items()}
    report=f'''# T026-A — NEEDS_REVIEW; clean-source ROI memory objective FAIL

R041/d5a9d7a executed. Close this exact class-conditional clean-source nearest-memory ROI feature objective. No nomination/AP/K-step/runtime/FCOS/SSD/training or protected-method change.

## Source assumption, cohort and ordering

This candidate explicitly uses an offline source-GT-derived feature memory. It is not source-free/training-free in the strongest sense specified by R041. Detector weights remain frozen, with no source fitting/projector/prototype training; audit-time support and retrieval use only predicted classes.

Plan/cohort `a8610b6` precommitted source-instance hash ordering. Source manifest/code `cfcb895`; complete memory committed/pushed `2befc7f` BEFORE candidate outcomes. Candidate code `051d7ee`; all48 candidate records/anchors committed/pushed `b4c1c04` BEFORE reference code/run `fb8de70`. Separate candidate interpreter loaded no oracle/reference/audit annotations. Separate source-cohort preparation uses inherited GT-valid-box/readable-image eligibility, disclosed before outcomes. Source GT instances are used only to construct the frozen memory; audit GT enters only the post-lock reference process.

24 brand-new train2017 audit images, seed20260926, four6imageblocks,24clean plus8each gamma_s2/contrast_s2/color_cast_s2. Audit excludes1476previous source/audit/debug IDs and5000val IDs; memory additionally excludes24audit images,6500total. From readable clean training JPEGs, each class selects first16 noncrowd positive-area instances by sha256('T026A:20260926:'+annotationID), annotationID tie-break.1280 entries from1187distinctsourceimages, zero audit/prior/val overlap. No selection by model outcomes. Minimum available class count was23; no class substitutions or source downloads.

Source manifest SHA256 `{sha(root/'research_log/T026A_memory_manifest.json')}`. Complete memory.pt SHA256 `{sha(dirs['memory']/'memory.pt')}`. Cohort SHA256 `{sha(root/'research_log/T026A_train_cohort.json')}`. All80classes have16 finite1024D unit vectors; max norm error={health['max_norm_error']}. No learned normalization/projection.

## Fixed objective and reference

Original-view FasterRCNN supports score>=.50, stable descending top20, original boxes/classes/scores frozen. Same binary support union mask; background identity and object8D identity analysis only. Query and current features use unchanged1024D box_head output before predictor. Same-predicted-class cosine top4, stable memory-index ties; anchor is detached normalize(mean4 normalized memory features). Single unweighted mean ROI cosine distance. No confidence weighting, cross-class retrieval, CLIP/pseudo/flip/exposure/geometry blend, fallback anchors or adaptation step.

Shared accepted common8-column ISP JVP; float32 image/model/ISP and independent float64 global/object/background reductions. Full source reference remains unchanged native four-loss sum at seed20260913 and currentdet_pseudo, eps1e-12. Exact supports, mask, selected IDs/similarities, query/anchor hashes, actual anchor tensors, per-object losses, cotangent diagnostics, JVP column norms/identity checks and8D gradients are retained. Memory and candidate hashes are checked before audit annotation loading.

## Frozen gate

|Metric|Observed|Required|
|---|---:|---:|
|S_mem>0 overall|{a['S_mem_positive']}/48|>=30/48|
|S_mem>0 corrupt|{b['S_mem_positive']}/24|>=15/24|
|Delta_global>0 overall|{a['Delta_global_positive']}/48|>=30/48|
|Delta_global>0 corrupt|{b['Delta_global_positive']}/24|>=15/24|
|MedianDelta overall|{a['median_Delta_global']}|>0|
|MedianDelta corrupt|{b['median_Delta_global']}|>0|
|Positiveblockmedians|{nb}/4|>=3/4|
|Memory health/integrity|PASS|PASS|
|Conjunction|FAIL|allconditions|

S_mem=dot(t_o,g)/(norm(g)+eps); Delta_global subtracts dot(t_s,p_s)/(norm(p_s)+eps). Positive medians, retrieval similarity or Delta_obj do not rescue the failed counts/block requirement. Overall meanDelta={a['distributions']['Delta_global']['mean']}; corrupted meanDelta={b['distributions']['Delta_global']['mean']}.

## Retrieval and distribution diagnostics

Across all48 episodes,{a['distinct_memory_entries']}distinctmemory entries from{a['distinct_classes']}predicted classes were used; corrupted subset{b['distinct_memory_entries']}entries/{b['distinct_classes']}classes. Retrieval cosine overall mean/median={a['distributions']['retrieval_cosine']['mean']}/{a['distributions']['retrieval_cosine']['median']}; corrupted={b['distributions']['retrieval_cosine']['mean']}/{b['distributions']['retrieval_cosine']['median']}.

`T026A/summary.json` reports overall,clean,corrupt,eachcorruption and all4blocks: positive counts/fractions,means,medians,linear quantiles[0,.25,.5,.75,1] for S_mem/Delta_global/Delta_obj; gradient norms/near-zero frequency; retrieval similarities, distinct entries/classes, unique-anchor counts and pairwise anchor-cosine diversity. Per-episode values/norms/cosines/mask/support/anchor-diversity/loss statistics are in per_episode.tsv. Undefined pairwise diversity for fewer than2supports is null, without excluding episodes.

## Integrity, validation and execution

All48reference integrity checks passed. Maxreverse/JVPrelativeL2={integrity['max_reference_relative_l2']}; partitionerror={integrity['max_partition_absolute_error']}. Empty supports={integrity['empty_supports']};zero/nearzero candidategradients={a['zero_gradients']}/{a['near_zero_gradients']};minimumgradientnorm={integrity['minimum_candidate_norm']}. Prospective nearzero convention remains norm<=1e-12, systematic if>=2distinct nonempty images; none occurred.

Source before/after every stage:73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf. Frozen/eval/parameter-grad-None and ISP-state checks passed.23protected+2inherited modules unchanged locally/remotely; all8new source/test pins match remote release. Tests: baseline18pass3.90s; memoryincrement13pass2.89s; candidate24pass3.83s; finalfocused26pass3.89s; full220pass11skip4warnings10.50s. Existing NVML/protobuf warnings retained, no driver/environment change.

All model-bearing execution on NVIDIA RTX A6000 CUDA12.1: memory `{runs['memory']}` {timings['memory']}s; candidate `{runs['candidates']}` {timings['candidates']}s; reference `{runs['reference']}` {timings['reference']}s; all exit0. CPU only manifests/synthetic checks/lightweight aggregation. Aborted launcher065756 left meta.json only and no tmux/run.sh; confirmed before successful065856 launch. Later SCPtimeout recovered through existingworkflow retry; no method change or repeated outcome run.

## Research-review numeric discrepancy and handoff

R041 quotes T025 figures that differ from authoritative `a268f90` receipts. Actual T025 S_geom28/48overall,15/24corrupt;Delta23/48,14/24;medians-.019543044761259266overall,+.009389868326732737corrupt;2/4blocks. R041 quotes27/48,16/24 and25/48,12/24 with different medians. Both concludeFAIL; T025 rawdata unchanged, no rerun, research-owned files not edited. T026 was executed as its independent newly authorized hypothesis.

Preserve all memory/48candidate/48reference raw artifacts under the three remote_runs roots, source-instance/JPEG/exclusion manifests, codepins and commit locks. Future fresh cohorts MUST use `research_log/T026A_source_exclusion_manifest.json` as the cumulative additional_source: includes all1187memoryimages+24audit and1476prior IDs,2687source IDs total, plusval5000 exclusion. Do not use only T026A_train_cohort, which omits memory images.

Stop NEEDS_REVIEW; close this exact clean-source nearest-memory ROI objective. No top-k/memory-size/layer/projection/source-selection/weights/support/mask/K/LR/blend rescue and no T026-B/AP/new family without a new explicit research task.
'''
    (root/'research_log/T026A_report.md').write_text(report,encoding='utf-8')


if __name__=='__main__':
    main()
