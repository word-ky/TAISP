"""Frozen R034 gate and R035 runtime receipts; no method changes."""
import statistics as st


def spatial_advancement(metrics, isolation_passed):
    from .run_t018a import advancement, CASE_NAMES
    candidate='spatial_dose_ours'
    mapped={g:{k.replace(candidate,'nativePT_ours'):v for k,v in values.items()} for g,values in metrics.items()}
    result=advancement(mapped)
    for g,values in metrics.items():
        group=result['groups'][g]
        for key in ('macro_AP','clean_AP'):
            group[key][candidate]=group[key].pop('nativePT_ours')
        group['clean_delta_vs_raw']=group['clean_AP'][candidate]-group['clean_AP']['no_adapt']
        group['additional_macro_metrics']={k:{m:100*st.mean(values[f'{c}_{m}'][k] for c in CASE_NAMES[:-1])
            for m in ('no_adapt','current_ours',candidate)} for k in ('AP50','AP75')}
        group['additional_macro_deltas']={k:v[candidate]-v['current_ours'] for k,v in group['additional_macro_metrics'].items()}
    gate=result['gate'];gate['flags'].pop('corruption_macro_above_current')
    gate['flags']['no_parity_isolation_or_contamination_blocker']=isolation_passed
    gate['passed']=all(gate['flags'].values())
    gate['decision']='development_candidate_pending_confirmation' if gate['passed'] else 'close_direction_locked_two_region_dose'
    return result



def episode_receipt(image,isp,mask,result):
    import torch
    from taisp.isp.spatial import compose
    ds=result.diagnostics
    checks=dict(reset_zero=bool(torch.count_nonzero(result.phi0)==0),
        three_updates=len(ds)==4 and sum(d['update_applied'] for d in ds)==3,
        fixed_support_count=all(d['support_count']==ds[0]['support_count'] for d in ds),
        dose_bounds=all(-1-1e-6<=d['c']<=1+1e-6 and all(.5-1e-6<=m<=1.5+1e-6 for m in d['multipliers']) for d in ds),
        mean_one=all(abs(sum(d['multipliers'])/2-1)<=1e-6 for d in ds),
        zero_common_no_update=all(not d['zero_common_pseudo'] or all(v==0 for a in d['state_delta'] for v in a) for d in ds))
    images=[]
    with torch.no_grad():
        for d in ds:
            state=image.new_tensor(d['phi']);y=compose(image,isp,mask,*state)
            images.append(dict(step=d['step'],minimum=y.min().item(),maximum=y.max().item(),mean=y.mean().item(),
                input_rms_difference=(y-image).square().mean().sqrt().item(),finite=bool(torch.isfinite(y).all())))
        checks['processed_images_finite']=all(d['finite'] for d in images)
        checks['final_image_matches_states']=torch.equal(y,result.enhanced)
    return dict(mask_area=mask.double().mean().item(),checks=checks,processed_images=images)


def edge_smoke(image,isp,source,loss,clip,selected,path,config):
    import torch
    from taisp.tta.spatial_dose import adapt_spatial_dose
    from taisp.losses.detector_native import DetectorNativeLoss
    from .deterministic_replay import state_hash
    from .differential_subspace import write_json
    before=(state_hash(source),state_hash(clip));records=[]
    empty={k:v[:0] for k,v in selected.items()}
    for name,value,inactive in [('empty_mask',0.,0),('full_mask',1.,1),('empty_support',0.,0)]:
        mask=image.new_full((1,1,*image.shape[-2:]),value)
        objective=DetectorNativeLoss(source,{'base':empty},'det_pseudo') if name=='empty_support' else loss
        def run():return adapt_spatial_dose(image,isp,objective,clip,mask,steps=config['semantic_steps'],lr=config['semantic_lr'],eps=config['radius_eps'])
        a,b=run(),run();receipt=episode_receipt(image,isp,mask,a)
        keys=('object_gradient','clip_object_gradient') if inactive==0 else ('background_gradient','clip_background_gradient')
        receipt['checks']['inactive_gradients_zero']=all(v==0 for d in a.diagnostics for key in keys for v in d[key])
        receipt['checks']['repeat_deterministic']=torch.equal(a.phi,b.phi) and torch.equal(a.enhanced,b.enhanced)
        if name=='empty_support':receipt['checks']['empty_support_identity']=bool(torch.count_nonzero(a.phi)==0)
        records.append(dict(case=name,diagnostics=a.diagnostics,repeat_diagnostics=b.diagnostics,**receipt))
    checks=dict(source_hash_unchanged=state_hash(source)==before[0],clip_hash_unchanged=state_hash(clip)==before[1],
        frozen_eval_grad_none=all(not m.training for model in (source,clip) for m in model.modules()) and all(not p.requires_grad and p.grad is None for model in (source,clip) for p in model.parameters()),
        isp_unchanged=bool(torch.count_nonzero(isp.phi)==0 and isp.phi.grad is None))
    passed=all(checks.values()) and all(all(r['checks'].values()) for r in records)
    write_json(path,dict(records=records,isolation=checks,passed=passed,evaluations=0))
    assert passed,'R035 runtime edge blocker; stop without AP'


def spatial_summary(rows):
    rows=[r for r in rows if r['method']=='spatial_dose_ours']
    groups={'overall':rows,'clean':[r for r in rows if r['case']=='clean_s0'],'corrupted':[r for r in rows if r['case']!='clean_s0']}
    groups.update({f'block{b}':[r for r in rows if r['block']==b] for b in sorted({r['block'] for r in rows})})
    output={}
    for name,rs in groups.items():
        if not rs:continue
        d=dict(episodes=len(rs),update_fraction=st.mean(r['updated'] for r in rs))
        def stats(v):return dict(mean=st.mean(v),median=st.median(v),min=min(v),max=max(v))
        d['mask_area']=stats([r['spatial']['mask_area'] for r in rs]);d['steps']={}
        for step in range(4):
            ds=[r['diagnostics'][step] for r in rs]
            d['steps'][str(step)]={k:stats([a[k] for a in ds]) for k in ('c','common_pseudo_norm','common_clip_norm','phi_obj_norm','phi_bg_norm','phi_difference_norm')}
            d['steps'][str(step)]['multipliers']=[stats([a['multipliers'][i] for a in ds]) for i in range(2)]
        output[name]=d
    return output
