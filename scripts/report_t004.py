"""T004 paired representation, norm-matched and predicted-patch diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.analyze_t002_linear import describe
from scripts.report_t003 import summarize_variant, paired_delta, interval


def norm_rows(rows):
    return [{**r,'g_sem':r['norm_matched']['gradient'],
             'det_loss_delta_sem1':r['norm_matched']['det_loss_delta']} for r in rows]


def spatial_summary(rows):
    support=np.array([r['region']['support_fraction'] for r in rows])
    result={'predicted_object_support_mean':float(support.mean()),
            'uniform_region_fallback_fraction':float(np.mean([r['region']['uniform_fallback'] for r in rows])),
            'selected_boxes_mean':float(np.mean([r['region']['selected_count'] for r in rows])),
            'visible_boxes_mean':float(np.mean([r['region']['visible_count'] for r in rows]))}
    if rows[0]['patch_weights'] is not None:
        result.update(effective_patch_count_mean=float(np.mean([r['effective_patch_count'] for r in rows])),
                      weighted_patch_fraction_mean=float(np.mean([r['weighted_patch_fraction'] for r in rows])))
        go=np.array([r['partition']['g_object'] for r in rows])
        gb=np.array([r['partition']['g_background'] for r in rows])
        gd=np.array([r['g_det'] for r in rows])
        result.update(object_gradient_mean=go.mean(0).tolist(),background_gradient_mean=gb.mean(0).tolist(),
                      object_gradient_norm_mean=float(np.linalg.norm(go,axis=1).mean()),
                      background_gradient_norm_mean=float(np.linalg.norm(gb,axis=1).mean()),
                      object_detector_dot_mean=float((go*gd).sum(1).mean()),
                      background_detector_dot_mean=float((gb*gd).sum(1).mean()),
                      partition_gradient_residual_max=max(r['partition']['sum_gradient_residual_norm'] for r in rows))
        for step in ('step1','step3'):
            result[step]={}
            for field in ('object_weighted_sum','background_weighted_sum','object_unweighted_mean','background_unweighted_mean'):
                values=[r['partition'][step][field] for r in rows if r['partition'][step][field] is not None]
                result[step][field]={'count':len(values),'mean':float(np.mean(values)) if values else None}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('study',type=Path)
    args=p.parse_args()
    study=args.study
    rows=[json.loads(s) for s in (study/'samples.jsonl').read_text().splitlines()]
    env=json.loads((study/'environment.json').read_text())
    variants=env['config']['variants']
    cases=list(json.loads((study/'summary.json').read_text())[variants[0]])
    metrics=json.loads((study/'metrics.json').read_text())
    baseline=json.loads((study/'baseline_metrics.json').read_text())
    groups={}
    for case in ['overall']+cases:
        chosen=[r for r in rows if case=='overall' or f"{r['family']}_s{r['severity']}"==case]
        ref=[r for r in chosen if r['variant']=='global_generic']
        groups[case]={}
        for v in variants:
            vr=[r for r in chosen if r['variant']==v]
            g=summarize_variant(vr,ref,env['config']['semantic_lr'])
            nr=norm_rows(vr)
            taylor,_=describe(nr,env['config']['semantic_lr'])
            g['norm_matched']={'benefit_fraction':float(np.mean([r['det_loss_delta_sem1']<0 for r in nr])),
                'loss_delta_mean':float(np.mean([r['det_loss_delta_sem1'] for r in nr])),
                'paired_benefit_delta':paired_delta(nr,ref,lambda r:float(r['det_loss_delta_sem1']<0)),
                'paired_loss_delta':paired_delta(nr,ref,lambda r:r['det_loss_delta_sem1']),
                'zero_gradients':sum(r['norm_matched']['zero_gradient'] for r in vr),
                'max_norm_target_error':max(abs(r['norm_matched']['gradient_norm']-r['norm_matched']['target_norm']) for r in vr),
                'taylor':taylor}
            g['spatial']=spatial_summary(vr)
            g['deploy_seconds3_mean']=float(np.mean([r['deploy_seconds_3'] for r in vr]))
            g['region_setup_seconds_mean']=float(np.mean([r['region_setup_seconds'] for r in vr]))
            g['paired_deploy_seconds3']=paired_delta(vr,ref,lambda r:r['deploy_seconds_3'])
            if case!='overall':
                g['AP_delta_vs_global_generic']={f'step{k}':100*(metrics[f'{case}_{v}{k}']['AP']-metrics[f'{case}_global_generic{k}']['AP']) for k in (1,3)}
            if v.startswith('region_'):
                uniform=[r for r in chosen if r['variant']==v.replace('region_','patch_')]
                g['paired_region_vs_uniform']={'cosine':paired_delta(vr,uniform,lambda r:r['gradient_cosine']),
                    'benefit':paired_delta(vr,uniform,lambda r:float(r['det_loss_delta_sem1']<0)),
                    'loss_delta':paired_delta(vr,uniform,lambda r:r['det_loss_delta_sem1'])}
            groups[case][v]=g
        print(f'Analyzed {case}',flush=True)
    payload={'groups':groups,'source_revision':env['source_revision'],
             'bootstrap':'2000 image-cluster percentile draws seed20260912,paired;exploratory,no multiplicity adjustment',
             'norm_reference':'contemporaneous global_generic semantic-gradient norm; no labels in scaling'}
    (study/'analysis.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    lines=['# T004 spatial CLIP fixed-subset results','',
        f"Source {env['source_revision']};{len(env['evaluated_image_ids'])} images/{len(rows)} observations;smoke={env['smoke_limit']}.",'',
        'Same text banks, last-layer post-LN/projected normalized7x7 tokens, global8D ISP,lr0.1,K3. '
        'Detector regions come only from original inference:score>=0.5,top20,overlap weights,uniform fallback. '
        'Oracle family directions and annotated losses exist only in analysis. Patch tokens remain contextualized '
        'by global self-attention; this tests token readout, not independent local receptive fields.','',
        '## AP1 / AP3 (0–100 subset points)','',
        '| Case | Corrupted | Global generic | Global oracle | Uniform generic | Uniform oracle | Region generic | Region oracle |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        cells=[f"{100*metrics[f'{c}_{v}1']['AP']:.3f} / {100*metrics[f'{c}_{v}3']['AP']:.3f}" for v in variants]
        lines.append(f"| {c} | {100*baseline[c+'_corrupted']['AP']:.3f} | "+' | '.join(cells)+' |')
    lines+=['','## Paired AP differences versus global-generic','',
        'Same fixed image IDs; subset AP differences, no AP bootstrap. AP is not a mean of per-image AP.','',
        '| Case | Variant | ΔAP1 | ΔAP3 |','|---|---|---:|---:|']
    for c in cases:
        for v,g in groups[c].items():
            d=g['AP_delta_vs_global_generic']
            lines.append(f"| {c} | {v} | {d['step1']:+.3f} | {d['step3']:+.3f} |")
    lines+=['','## Primary raw-gradient and paired task behavior','',
        '95% image-cluster percentile intervals:2000 paired draws,seed20260912. All six conditions remain '
        'together for each image overall. Exploratory intervals, no multiplicity adjustment.','',
        '| Group | Variant | Gradient norm | Mean / median cosine | Positive | Loss benefit | Δcosine [95% CI] | Δbenefit pp [95% CI] | Δloss [95% CI] |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for c,vg in groups.items():
        for v,g in vg.items():
            d=g['paired_vs_generic']
            lines.append(f"| {c} | {v} | {g['g_sem_norm']['mean']:.5f} | {g['cosine_mean']:.4f} / {g['cosine_median']:.4f} | "
                f"{100*g['positive_fraction']:.1f}% | {100*g['benefit_fraction']:.1f}% | {interval(d['cosine'])} | "
                f"{interval(d['benefit_rate'],100)} | {interval(d['loss_delta1'])} |")
    lines+=['','## Direction-only norm-matched one-step diagnostic','',
        'Each gradient is rescaled to that image/case global-generic gradient norm. Primary fixed-lr results '
        'above are unchanged. Scaling uses no annotated gradient or label.','',
        '| Group | Variant | Benefit | Mean detector Δloss | Δbenefit pp [95% CI] | Δloss [95% CI] | Taylor sign match | Taylor Spearman |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for c,vg in groups.items():
        for v,g in vg.items():
            d=g['norm_matched']
            lines.append(f"| {c} | {v} | {100*d['benefit_fraction']:.1f}% | {d['loss_delta_mean']:.6f} | "
                f"{interval(d['paired_benefit_delta'],100)} | {interval(d['paired_loss_delta'])} | "
                f"{100*d['taylor']['sign_agreement']['estimate']:.1f}% | {d['taylor']['spearman_linear_observed']['estimate']:.4f} |")
    lines+=['','## Region weighting versus uniform with the same text direction','',
        '| Group | Variant | Δcosine [95% CI] | Δbenefit pp [95% CI] | Δloss [95% CI] |',
        '|---|---|---:|---:|---:|']
    for c,vg in groups.items():
        for v,g in vg.items():
            if 'paired_region_vs_uniform' in g:
                d=g['paired_region_vs_uniform']
                lines.append(f"| {c} | {v} | {interval(d['cosine'])} | {interval(d['benefit'],100)} | {interval(d['loss_delta'])} |")
    lines+=['','## Patch support and contribution','',
        'Object support is any positive predicted-box overlap in the CLIP crop, not GT objects. Region '
        'background loss is zero by construction except empty-region uniform fallback. Gradient partitions '
        'are recomputed for analysis, so numerical residuals are retained instead of assuming bitwise equality.','',
        '| Group | Variant | Object support | Fallback | Effective patches | Weighted fraction | Object / background grad norm | Object / background loss3 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for c,vg in groups.items():
        for v,g in vg.items():
            if v.startswith('global_'):continue
            d=g['spatial']
            lines.append(f"| {c} | {v} | {100*d['predicted_object_support_mean']:.1f}% | {100*d['uniform_region_fallback_fraction']:.1f}% | "
                f"{d['effective_patch_count_mean']:.2f} | {100*d['weighted_patch_fraction_mean']:.1f}% | "
                f"{d['object_gradient_norm_mean']:.5f} / {d['background_gradient_norm_mean']:.5f} | "
                f"{d['step3']['object_weighted_sum']['mean']:.5f} / {d['step3']['background_weighted_sum']['mean']:.5f} |")
    lines+=['','## Loss, saturation and measured runtime','',
        'Deploy latency adds one original detector inference/map for region variants; the same inference is '
        'used only as an excluded analysis diagnostic for global/uniform variants. All adaptation timing '
        'includes diagnostics/synchronization and excludes annotated loss/AP/norm-matched/partition work.','',
        '| Group | Variant | Own loss1 / loss3 | Own loss3 decreases | Sat0 / Sat1 / Sat3 | Adapt / deploy seconds3 | Peak MiB |',
        '|---|---|---:|---:|---:|---:|---:|']
    for c,vg in groups.items():
        for v,g in vg.items():
            sat=' / '.join(f'{100*g[k]["mean"]:.2f}%' for k in ('saturation_before','saturation_1','saturation_3'))
            lines.append(f"| {c} | {v} | {g['semantic_loss_1']['mean']:.5f} / {g['semantic_loss_3']['mean']:.5f} | "
                f"{100*g['own_semantic_decrease3_fraction']:.1f}% | {sat} | {g['adapt_seconds_3']['mean']:.4f} / "
                f"{g['deploy_seconds3_mean']:.4f} | {g['peak_allocated_mb']['mean']:.1f} |")
    lines+=['','Full per-coordinate energy/signed contributions, trajectories, paired positive/saturation/runtime '
        'intervals, Taylor distributions, object/background projections and partition errors are in analysis.json '
        'and raw samples.jsonl. No negative family or sample is excluded.','',
        '![Three-step AP change against corrupted input](ap_change.png)','']
    (study/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    changes=np.array([[100*(metrics[f'{c}_{v}3']['AP']-baseline[c+'_corrupted']['AP']) for c in cases] for v in variants])
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    bound=max(abs(changes).max(),.1)
    im=ax.imshow(changes,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
    ax.set_xticks(range(len(cases)),cases,rotation=20,ha='right')
    ax.set_yticks(range(len(variants)),variants)
    for (i,j),value in np.ndenumerate(changes):
        ax.text(j,i,f'{value:+.3f}',ha='center',va='center',color='white' if abs(value)>.65*bound else 'black')
    ax.set_title('T004 · three-step AP change from corrupted input')
    fig.colorbar(im,ax=ax,label='COCO subset AP points')
    fig.savefig(study/'ap_change.png',dpi=180)
    fig.savefig(study/'ap_change.pdf')
    plt.close(fig)
    print(study/'results.md')


if __name__=='__main__':main()
