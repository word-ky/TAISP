"""Fixed R014 interpretation and descriptive paired-dose ratios."""


def half_dose_criteria(target_macros, positive_blocks, above_control_medians,
                       above_block_medians, clean_deltas, clean_half_phi, clean_full_phi):
    return {
        'both_target_macros_positive': min(target_macros) > 0,
        'both_targets_four_positive_blocks': min(positive_blocks) >= 4,
        'both_targets_above_control_medians': all(above_control_medians) and min(above_block_medians) >= 4,
        'all_clean_AP_within_bound': min(clean_deltas) >= -.1,
        'clean_phi_at_most_65_percent': clean_half_phi <= .65*clean_full_phi,
    }


def ratio_summary(numerator, denominator):
    from scripts.report_t005 import distribution
    paired = list(zip(numerator, denominator))
    ratios = [a/b for a, b in paired if b > 0]
    return {'positive_denominator_count': len(ratios),
            'both_zero_count': sum(a == b == 0 for a, b in paired),
            'nonzero_numerator_zero_denominator_count': sum(a != 0 and b == 0 for a, b in paired),
            'ratio': distribution(ratios) if ratios else None}
