"""
Cross-model grammar consensus analysis.

Determines whether different model architectures agree on grammar rules.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import spearmanr
from collections import Counter


def compute_grammar_consensus(rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each motif pair in each enhancer, compute cross-model agreement.

    Agreement measured by:
    1. Correlation of spacing profiles across models
    2. Agreement on optimal spacing (+/-2bp)
    3. Agreement on optimal orientation

    Args:
        rules_df: Grammar rules database from rule extraction

    Returns:
        DataFrame with consensus scores per (enhancer, motif_pair)
    """
    results = []

    for (seq_id, pair), group in rules_df.groupby(['seq_id', 'pair']):
        if len(group) < 2:
            continue

        models = group['model'].tolist()
        n_models = len(models)

        # 1. Spacing profile correlations
        spacing_correlations = []
        for m1, m2 in combinations(range(n_models), 2):
            p1 = np.array(group.iloc[m1]['spacing_profile'])
            p2 = np.array(group.iloc[m2]['spacing_profile'])
            if len(p1) == len(p2) and len(p1) > 2:
                r, _ = spearmanr(p1, p2)
                if not np.isnan(r):
                    spacing_correlations.append(r)

        mean_spacing_corr = np.mean(spacing_correlations) if spacing_correlations else 0

        # 2. Optimal spacing agreement (+/-2bp)
        optimal_spacings = group['optimal_spacing'].values
        spacing_range = float(optimal_spacings.max() - optimal_spacings.min())

        spacing_agreement = 0
        n_pairs = 0
        for s1, s2 in combinations(optimal_spacings, 2):
            n_pairs += 1
            if abs(s1 - s2) <= 2:
                spacing_agreement += 1
        spacing_agreement = spacing_agreement / max(n_pairs, 1)

        # 3. Orientation agreement
        optimal_orientations = group['optimal_orientation'].values
        orient_counts = Counter(optimal_orientations)
        mode_count = orient_counts.most_common(1)[0][1]
        orientation_agreement = mode_count / n_models

        # Overall consensus
        consensus_score = (mean_spacing_corr + spacing_agreement + orientation_agreement) / 3

        results.append({
            'seq_id': seq_id,
            'pair': pair,
            'n_models': n_models,
            'mean_spacing_correlation': float(mean_spacing_corr),
            'spacing_agreement': float(spacing_agreement),
            'spacing_range': float(spacing_range),
            'orientation_agreement': float(orientation_agreement),
            'consensus_score': float(consensus_score),
            'models_tested': ','.join(models)
        })

    return pd.DataFrame(results)


def compute_global_consensus(consensus_df: pd.DataFrame) -> dict:
    """Compute global statistics across all grammar rules."""
    if len(consensus_df) == 0:
        return {'error': 'No consensus data available'}

    return {
        'mean_consensus': float(consensus_df['consensus_score'].mean()),
        'median_consensus': float(consensus_df['consensus_score'].median()),
        'frac_high_consensus': float((consensus_df['consensus_score'] > 0.7).mean()),
        'frac_contested': float((consensus_df['consensus_score'] < 0.3).mean()),
        'mean_spacing_correlation': float(consensus_df['mean_spacing_correlation'].mean()),
        'mean_orientation_agreement': float(consensus_df['orientation_agreement'].mean()),
        'n_rules_total': len(consensus_df),
        'n_enhancers': int(consensus_df['seq_id'].nunique()),
    }
