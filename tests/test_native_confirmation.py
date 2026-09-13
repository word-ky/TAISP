import copy

from taisp.analysis.run_t018a import CASE_NAMES, METHODS, confirmation


def example():
    return {g:{f'{c}_{m}':{'AP':.3+(.002 if m=='nativePT_ours' else 0),
                            'AP50':.5,'AP75':.3-(.001 if m=='nativePT_ours' else 0)}
               for c in CASE_NAMES for m in METHODS}
            for g in ['aggregate']+[f'block{i}' for i in range(5)]}


def test_five_block_confirmation_and_AP75_is_only_diagnostic():
    metrics=example()
    result=confirmation(metrics,True)
    assert result['gate']['passed'] and result['gate']['positive_blocks']==5
    assert not result['AP75_nonnegative_diagnostic']
    for c in CASE_NAMES[:-1]:
        metrics['block4'][f'{c}_nativePT_ours']['AP']=.299
    assert confirmation(metrics,True)['gate']['passed']
    for c in CASE_NAMES[:-1]:
        metrics['block3'][f'{c}_nativePT_ours']['AP']=.299
    assert not confirmation(metrics,True)['gate']['passed']


def test_confirmation_rejects_small_AP_gain_clean_drop_and_isolation_failure():
    metrics=example()
    assert not confirmation(metrics,False)['gate']['passed']
    small=copy.deepcopy(metrics)
    for c in CASE_NAMES[:-1]:
        small['aggregate'][f'{c}_nativePT_ours']['AP']=.3009
    assert not confirmation(small,True)['gate']['passed']
    metrics['aggregate']['clean_s0_nativePT_ours']['AP']=.2989
    assert not confirmation(metrics,True)['gate']['passed']
