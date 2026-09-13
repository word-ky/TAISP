"""Frozen R031 developmental eligibility and candidate selection."""
import statistics

CANDIDATES = ('native_cls','native_conf','native_roi','native_conf_roi')


def component_selection(metrics, isolation_passed):
    from .run_t018a import CASE_NAMES, advancement
    candidates={}
    for candidate in CANDIDATES:
        renamed={g:{f'{c}_{dest}':v[f'{c}_{src}'] for c in CASE_NAMES
                    for src,dest in [('no_adapt','no_adapt'),('current_ours','current_ours'),(candidate,'nativePT_ours')]}
                 for g,v in metrics.items()}
        prior=advancement(renamed)
        groups=prior['groups']
        for g,v in groups.items():
            for field in ('macro_AP','clean_AP'):
                v[field][candidate]=v[field].pop('nativePT_ours')
            v['clean_delta_vs_raw']=v['clean_AP'][candidate]-v['clean_AP']['no_adapt']
            v['additional_macro_metrics']={k:{m:100*statistics.mean(metrics[g][f'{c}_{m}'][k] for c in CASE_NAMES[:-1])
                                             for m in ('no_adapt','current_ours',candidate)} for k in ('AP50','AP75')}
            v['additional_macro_deltas']={k:x[candidate]-x['current_ours'] for k,x in v['additional_macro_metrics'].items()}
        flags={k:v for k,v in prior['gate']['flags'].items() if k!='corruption_macro_above_current'}
        flags['no_isolation_or_reproducibility_blocker']=isolation_passed
        candidates[candidate]={'groups':groups,'flags':flags,'eligible':all(flags.values()),
                               'positive_blocks':prior['gate']['positive_blocks']}
    eligible=[c for c,v in candidates.items() if v['eligible']]
    near_best=[]
    selected=None
    if eligible:
        best=max(candidates[c]['groups']['aggregate']['macro_AP'][c] for c in eligible)
        near_best=[c for c in eligible if best-candidates[c]['groups']['aggregate']['macro_AP'][c]<=.01]
        selected=max(near_best,key=lambda c:(candidates[c]['groups']['aggregate']['additional_macro_metrics']['AP75'][c],
                                            candidates[c]['positive_blocks']))
    return {'candidates':candidates,'gate':{'eligible_candidates':eligible,'within_point01_of_best':near_best,
            'selected':selected,'passed':selected is not None,'decision':'development_candidate_pending_confirmation' if selected else
            'close_fixed_component_subset_branch','stop_for_research_review':True}}
