"""Build T025-A report/tables from pinned CUDA receipts; stdlib only."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'research_log/T025A'
    crun='20260914-052059-taisp-t025a-candidates48'
    rrun='20260914-052527-taisp-t025a-reference48'
    cp=root/f'research_log/remote_runs/{crun}/artifacts/candidates'
    rp=root/f'research_log/remote_runs/{rrun}/artifacts/reference'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    for folder in (cp,rp):
        for name,digest in read(folder/'sha256.json').items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
    cs,rs,s=read(cp/'records.json'),read(rp/'records.json'),read(rp/'summary.json')
    name=s['candidate'];table=[]
    for c,r in zip(cs,rs,strict=True):
        assert (c['image_id'],c['case'])==(r['image_id'],r['case'])
        m=r['metrics'][name];obj=c['objectives'][name]
        assert obj['class_indices']==c['supports']['labels'] and c['roi_order']==list(range(c['support_count']))
        g=c['gradients'][name]['object'];t=r['reference_gradients']['task']['object']
        assert math.fsum(a*b for a,b in zip(t,g))/(math.sqrt(math.fsum(v*v for v in g))+1e-12)==m['S_geom']
        delta=[a-b for hi,lo in zip(obj['delta_hi'],obj['delta_lo']) for a,b in zip(hi,lo)]
        row={'episode':r['episode_index'],'image_id':r['image_id'],'case':r['case'],'block':r['block'],
            'supports':r['support_count'],'mask_area':r['mask']['area_fraction'],'geometry_loss':obj['loss'],
            'per_object_loss_min':min(obj['per_object']) if delta else None,
            'per_object_loss_median':statistics.median(obj['per_object']) if delta else None,
            'per_object_loss_max':max(obj['per_object']) if delta else None,
            'delta_disagreement_l2':math.sqrt(math.fsum(v*v for v in delta)),
            'cotangent_l2':c['cotangent']['l2'],
            **{k:v for k,v in m.items() if not isinstance(v,dict)},
            **{f'norm_{k}':v for k,v in m['norms'].items()},
            **{f'{n}_{k}':v for n in ('roi_box_diagnostic','localization_diagnostic') for k,v in m[n].items()}}
        table.append(row)
    with (out/'per_episode.tsv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]),delimiter='\t');writer.writeheader();writer.writerows(table)
    plan=read(root/'research_log/T025A_plan.json')
    pins={**plan['protected_modules_LF'],**plan['inherited_analysis_LF']}
    checks={p:hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h for p,h in pins.items()}
    assert all(checks.values()) and all(read(out/'remote_code_checks.json').values())
    integrity={'local_protected_and_inherited':checks,'remote_checks':read(out/'remote_code_checks.json'),
        'all48_integrity':all(r['integrity_passed'] for r in rs),
        'max_reference_relative_l2':max(r['reference_checks'][k]['parity']['relative_l2_error'] for r in rs for k in ('task','pseudo')),
        'max_partition_absolute_error':max(r['reference_checks'][k]['partition']['max_absolute_error'] for r in rs for k in ('task','pseudo')),
        'empty_supports':sum(r['support_count']==0 for r in rs),
        'minimum_candidate_gradient_norm':min(r['norm_candidate'] for r in table),
        'candidate_commit':'ef14e99','reference_code_commit':'2cd7415'}
    (out/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n',encoding='utf-8')
    (out/'summary.json').write_bytes((rp/'summary.json').read_bytes())
    a,b=s['overall'],s['corrupt'];positive_blocks=sum(v['median_Delta_global']>0 for v in s['blocks'].values())
    report=f'''# T025-A — NEEDS_REVIEW; fixed geometry objective FAIL

R040/0b6e3ca executed. The fixed `roi_bbox_exposure_stability` candidate fails the original conjunction. Close this exact exposure-pair ROI box-geometry objective. No nomination, AP, K-step runtime, FCOS/SSD, source/meta/predictor training or protected-method change.

## Protocol and ordering

Plan/cohort `65416d5`; candidate code `8738f05`; all48 candidate receipts committed and pushed `ef14e99` before reference code/run `2cd7415`. Separate candidate interpreter loaded no GT/oracle/reference module. Only the inherited cohort-preparation process read annotations for noncrowd valid-box/readable-image eligibility; candidate input contained image metadata and conditions only.

24 fresh train2017 images, seed20260925, excluding1452 prior source/development/debug/audit IDs plus all5000val IDs (6452total). Four consecutive6imageblocks.48episodes=24clean+8gamma_s2+8contrast_s2+8color_cast_s2. Ordered IDs/JPEG hashes/full exclusion ledger are committed.

Original-view FasterRCNN score>=.50/stable descending top20 supports were frozen. Same binary union mask; background identity and object8D identity as analysis variable only. Exposure views are exactly a*y/(1+(a-1)*y), factors1.2 and1/1.2, no clamp/randomness. Raw predictor regression layout(N,91,4) uses original ROI index/frozen predicted class. Per-object SmoothL1(beta1) sums four coordinates; objective means supports. No decoding/NMS/rematching/weights/feature-logit blend.

Accepted common8-column ISP JVP: float32 image/model/ISP, independent float64 global/object/background reductions onCUDA. Same unmodified helper for candidates/reference, equality to A1 checked. Source full-task oracle is unchanged native four-loss sum at seed20260913; current pseudo is unchanged det_pseudo. Additional ROI box-regression and combined RPN+ROI localization gradients come from the same oracle forward as diagnostics only.

## Frozen gate

|Metric|Observed|Required|
|---|---:|---:|
|S_geom>0 overall|{a['S_geom_positive']}/48|>=30/48|
|S_geom>0 corrupt|{b['S_geom_positive']}/24|>=15/24|
|Delta_global>0 overall|{a['Delta_global_positive']}/48|>=30/48|
|Delta_global>0 corrupt|{b['Delta_global_positive']}/24|>=15/24|
|Median Delta_global overall|{a['median_Delta_global']}|>0|
|Median Delta_global corrupt|{b['median_Delta_global']}|>0|
|Positive block medians|{positive_blocks}/4|>=3/4|
|Conjunction|FAIL|all conditions|

S_geom=dot(t_o,g)/(norm(g)+1e-12); Delta_global subtracts dot(t_s,p_s)/(norm(p_s)+1e-12). Overall meanDelta={a['mean_Delta_global']}; corrupted meanDelta={b['mean_Delta_global']}. Positive corrupted median does not substitute for the failed overall/count/block conditions. No post-outcome change was made.

ROI-box diagnostic alignment is positive on{a['roi_box_positive']}/48overall and{b['roi_box_positive']}/24corrupt; combined localization positivity is{a['localization_positive']}/48 and{b['localization_positive']}/24. These do not alter the full-task gate. Exact condition/block summaries, object-current diagnostics, cosines, all gradient norms, mask/support strata values and regression-delta disagreement are retained in summary.json/per_episode.tsv and raw records.

## Integrity and tests

All48 reference integrity checks passed. Max global reverse/JVP relativeL2={integrity['max_reference_relative_l2']}; max partitionerror={integrity['max_partition_absolute_error']}. Empty supports={integrity['empty_supports']}. Zero/near-zero candidategradients={a['zero_gradients']}/{a['near_zero_gradients']}; minimum norm={integrity['minimum_candidate_gradient_norm']}. Before outcomes, systematic near-zero was specified as nonempty norm<=1e-12 on at least2distinctimages; none occurred.

Source state before/after both stages:73eed6eae3ab74a76539b3f76ff544ff19f7e9e06a6d7e20131ee4ece4751ecf. Detector frozen/eval/parameter-grad-None, ISP state unchanged. All23protected modules and2inherited analysis helpers unchanged locally/remotely; all4new source/test pins verified on remote. Class slice and ROI ordering passed synthetic and real FasterRCNN CUDA smoke on a generated tensor (no extra COCO image).

Baseline18tests passed3.91s; candidate increment24passed6.04s; final focused30passed6.13s including cached real-model CUDA smoke. Full regression212passed/11skipped/4warnings in10.56s. Existing NVML and protobuf warnings retained; CUDA succeeds, no driver/environment changes.

Candidate run `{crun}`: {read(cp/'completion.json')['seconds']}s; reference `{rrun}`: {read(rp/'completion.json')['seconds']}s, both exit0. NVIDIA RTX A6000/CUDA12.1. CPU only for manifests, synthetic unit checks and lightweight aggregation.

## Artifacts and disposition

`research_log/T025A/` contains per_episode.tsv, exact summary.json, integrity and code-hash checks. Raw48candidate records retain supports/classes/order, mask rectangles/hash, both Nx4exposure deltas, per-object losses, cotangent shape/norm/hash, eight JVP column norms/identity checks and final global/object/background8D gradients. Raw48reference records retain unchanged task/pseudo plus diagnostic gradients, numerical checks and all metrics. Candidate and reference roots are `research_log/remote_runs/{crun}/artifacts/candidates` and `research_log/remote_runs/{rrun}/artifacts/reference`. Plans, JPEG/exclusion pins, pre-reference candidate commit/hash lock and tests are retained.

Stop NEEDS_REVIEW. Close this exact exposure-pair ROI box-geometry objective per R040. Do not tune exposure/beta/views/layer/decoding/classes/support/mask/K/LR/corruptions, implement T025-B or start a clean-source anchor/memory family without a new explicit research task.
'''
    (root/'research_log/T025A_report.md').write_text(report,encoding='utf-8')


if __name__=='__main__':
    main()
