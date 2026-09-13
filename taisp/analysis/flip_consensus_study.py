"""Frozen R033 performance decision and support diagnostics."""
import statistics as st


def consensus_advancement(metrics, isolation_passed):
    from .run_t018a import advancement, CASE_NAMES
    candidate='flip_consensus_ours'
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
    gate['flags']['no_support_matching_isolation_or_contamination_blocker']=isolation_passed
    gate['passed']=all(gate['flags'].values())
    gate['decision']='development_candidate_pending_confirmation' if gate['passed'] else 'close_fixed_horizontal_flip_support_filter'
    return result


def support_summary(rows):
    rows=[r for r in rows if r['method']=='flip_consensus_ours']
    groups={'overall':rows,'clean':[r for r in rows if r['case']=='clean_s0'],
            'corrupted':[r for r in rows if r['case']!='clean_s0']}
    groups.update({f'block{b}':[r for r in rows if r['block']==b] for b in sorted({r['block'] for r in rows})})
    groups.update({c:[r for r in rows if r['case']==c] for c in sorted({r['case'] for r in rows})})
    output={}
    for name,chosen in groups.items():
        receipts=[r['consensus'] for r in chosen]
        valid=[r['retention_fraction'] for r in receipts if r['retention_fraction'] is not None]
        output[name]={'episodes':len(chosen),'zero_consensus_count':sum(r['zero_consensus'] for r in receipts),
            'zero_consensus_fraction':st.mean(r['zero_consensus'] for r in receipts),
            'undefined_retention_count':len(chosen)-len(valid),
            'mean_retention_fraction_defined':st.mean(valid) if valid else None,
            'median_retention_fraction_defined':st.median(valid) if valid else None}
        for key in ('original_eligible_count','current_top20_count','flip_eligible_count','matched_count','retained_count'):
            values=[r[key] for r in receipts]
            output[name][key]={'mean':st.mean(values),'median':st.median(values),'min':min(values),'max':max(values)}
        for key in ('consensus_setup_seconds','adapt_seconds','deploy_seconds','phi3_norm'):
            values=[r[key] for r in chosen]
            output[name][key]={'mean':st.mean(values),'median':st.median(values)}
        output[name]['update_fraction']=st.mean(r['updated'] for r in chosen)
    return output
