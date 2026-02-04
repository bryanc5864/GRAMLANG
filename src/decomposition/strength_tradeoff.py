"""Grammar-strength interaction analysis."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_grammar_strength_tradeoff(
    rules_df: pd.DataFrame,
    motif_hits: pd.DataFrame,
) -> dict:
    """
    Test: do weak motifs need more precise grammar?

    Returns:
        Dict with correlation and quartile analysis
    """
    # Merge rules with motif scores
    merged = rules_df.copy()
    avg_scores = motif_hits.groupby(['seq_id', 'motif_name'])['score'].mean().reset_index()

    # Get mean motif score per rule pair
    merged_a = merged.merge(
        avg_scores.rename(columns={'motif_name': 'motif_a', 'score': 'motif_a_score'}),
        on=['seq_id', 'motif_a'], how='left'
    )

    if 'motif_a_score' not in merged_a.columns or merged_a['motif_a_score'].isna().all():
        return {'error': 'No motif scores available for merge'}

    merged_a['mean_strength'] = merged_a['motif_a_score'].fillna(0)

    # Quartile analysis
    valid = merged_a.dropna(subset=['mean_strength', 'spacing_sensitivity'])
    if len(valid) < 20:
        return {'error': f'Too few valid rules ({len(valid)})'}

    try:
        valid['strength_quartile'] = pd.qcut(
            valid['mean_strength'], 4,
            labels=['Q1_weak', 'Q2', 'Q3', 'Q4_strong'],
            duplicates='drop'
        )
    except ValueError:
        valid['strength_quartile'] = pd.cut(
            valid['mean_strength'], 4,
            labels=['Q1_weak', 'Q2', 'Q3', 'Q4_strong']
        )

    quartile_stats = valid.groupby('strength_quartile').agg(
        mean_spacing_sens=('spacing_sensitivity', 'mean'),
        mean_orient_sens=('orientation_sensitivity', 'mean'),
        mean_fold_change=('fold_change', 'mean'),
        n_rules=('pair', 'count')
    ).to_dict('index')

    # Correlation
    corr, pval = spearmanr(
        valid['mean_strength'], valid['spacing_sensitivity']
    )

    return {
        'strength_grammar_correlation': float(corr),
        'strength_grammar_pval': float(pval),
        'quartile_stats': quartile_stats,
        'n_rules': len(valid),
        'interpretation': 'Negative correlation = weak sites need more grammar' if corr < 0 else 'No strong tradeoff detected'
    }
