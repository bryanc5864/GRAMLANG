"""
Experiment G: Counterfactual Grammar Potential.

Measures the maximum possible grammar effect for each enhancer:
- Grammar potential: max(shuffles) - min(shuffles)
- Grammar utilization: how much of the potential the natural arrangement exploits
- Optimization headroom: how much expression could improve by rearranging
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from src.perturbation.vocabulary_preserving import generate_vocabulary_preserving_shuffles


def compute_grammar_potential(
    dataset: pd.DataFrame,
    model,
    motif_hits: pd.DataFrame,
    n_shuffles: int = 200,
    min_motifs: int = 2,
    cell_type: str = None,
    max_enhancers: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute grammar potential for each enhancer.

    For each enhancer, generates many shuffles and measures:
    - grammar_potential: max(pred) - min(pred) across shuffles
    - grammar_utilization: (original - min) / (max - min)
    - optimization_headroom: max(pred) - original
    - percentile_rank: where does original land in shuffle distribution

    Returns DataFrame with one row per enhancer.
    """
    rng = np.random.default_rng(seed)
    results = []

    # Filter to sequences with enough motifs
    seq_with_motifs = motif_hits.groupby('seq_id').size()
    eligible = seq_with_motifs[seq_with_motifs >= min_motifs].index
    eligible_df = dataset[dataset['seq_id'].isin(eligible)]

    if len(eligible_df) > max_enhancers:
        eligible_df = eligible_df.sample(max_enhancers, random_state=seed)

    for idx, row in eligible_df.iterrows():
        seq_id = str(row['seq_id'])
        seq = row['sequence']
        expr = row.get('expression', np.nan)

        seq_motifs = motif_hits[motif_hits['seq_id'] == seq_id]
        if len(seq_motifs) < min_motifs:
            continue

        annotation = {
            'sequence': seq,
            'motifs': seq_motifs.to_dict('records'),
        }

        # Generate shuffles using the function-based API
        try:
            shuffled_seqs = generate_vocabulary_preserving_shuffles(
                seq, annotation, n_shuffles=n_shuffles,
                seed=int(rng.integers(1e9)),
            )
        except Exception:
            continue

        if len(shuffled_seqs) < 10:
            continue

        # Predict expression for original + shuffles
        all_seqs = [seq] + shuffled_seqs
        preds = model.predict_expression(all_seqs, cell_type=cell_type)

        original_pred = float(preds[0])
        shuffle_preds = preds[1:]

        # Compute metrics
        max_pred = float(shuffle_preds.max())
        min_pred = float(shuffle_preds.min())
        mean_pred = float(shuffle_preds.mean())
        std_pred = float(shuffle_preds.std())

        potential = max_pred - min_pred
        if potential > 0:
            utilization = (original_pred - min_pred) / potential
        else:
            utilization = 0.5

        headroom = max_pred - original_pred
        percentile = float(np.mean(shuffle_preds <= original_pred))

        results.append({
            'seq_id': seq_id,
            'expression': expr,
            'original_pred': original_pred,
            'shuffle_mean': mean_pred,
            'shuffle_std': std_pred,
            'shuffle_min': min_pred,
            'shuffle_max': max_pred,
            'grammar_potential': float(potential),
            'grammar_utilization': float(utilization),
            'optimization_headroom': float(headroom),
            'percentile_rank': percentile,
            'n_shuffles': len(shuffled_seqs),
            'n_motifs': len(seq_motifs),
        })

    return pd.DataFrame(results)


def summarize_grammar_potential(results_df: pd.DataFrame) -> Dict:
    """Summarize grammar potential analysis."""
    if len(results_df) == 0:
        return {'error': 'No results'}

    return {
        'n_enhancers': len(results_df),
        'mean_potential': float(results_df['grammar_potential'].mean()),
        'median_potential': float(results_df['grammar_potential'].median()),
        'max_potential': float(results_df['grammar_potential'].max()),
        'mean_utilization': float(results_df['grammar_utilization'].mean()),
        'median_utilization': float(results_df['grammar_utilization'].median()),
        'mean_headroom': float(results_df['optimization_headroom'].mean()),
        'mean_percentile': float(results_df['percentile_rank'].mean()),
        'frac_above_median': float((results_df['percentile_rank'] > 0.5).mean()),
        'frac_in_top_10pct': float((results_df['percentile_rank'] > 0.9).mean()),
        'potential_vs_n_motifs_corr': float(
            results_df['grammar_potential'].corr(results_df['n_motifs'])
        ),
        'potential_percentiles': {
            'p25': float(results_df['grammar_potential'].quantile(0.25)),
            'p50': float(results_df['grammar_potential'].quantile(0.50)),
            'p75': float(results_df['grammar_potential'].quantile(0.75)),
            'p95': float(results_df['grammar_potential'].quantile(0.95)),
        },
    }
