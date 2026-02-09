"""
Grammar Sensitivity Index (GSI) computation.

The GSI quantifies how much arrangement (syntax) matters for each
enhancer by comparing expression predictions across vocabulary-preserving
shuffles.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats
from tqdm import tqdm

from src.perturbation.vocabulary_preserving import generate_vocabulary_preserving_shuffles


def compute_gsi(
    sequence: str,
    motif_annotations: dict,
    model,
    n_shuffles: int = 100,
    cell_type: str = None,
    seed: Optional[int] = None
) -> dict:
    """
    Compute Grammar Sensitivity Index for a single enhancer.

    GSI = CV(expression across shuffles) = std / |mean|
    High GSI = grammar-sensitive (arrangement matters)
    Low GSI = billboard-like (arrangement doesn't matter)

    Args:
        sequence: Original enhancer DNA sequence
        motif_annotations: From MotifScanner.annotate_enhancer()
        model: GrammarModel instance
        n_shuffles: Number of vocabulary-preserving shuffles
        cell_type: Cell type for expression prediction
        seed: Random seed for reproducibility

    Returns:
        Dict with GSI metrics
    """
    # Original expression
    original_expr = model.predict_expression([sequence], cell_type=cell_type)[0]

    # Generate vocabulary-preserving shuffles
    shuffles = generate_vocabulary_preserving_shuffles(
        sequence, motif_annotations, n_shuffles=n_shuffles, seed=seed
    )

    # Predict expression for all shuffles
    shuffle_exprs = model.predict_expression(shuffles, cell_type=cell_type)

    # Compute GSI
    shuffle_mean = np.mean(shuffle_exprs)
    shuffle_std = np.std(shuffle_exprs)

    if abs(shuffle_mean) > 1e-10:
        gsi = shuffle_std / abs(shuffle_mean)
    else:
        gsi = 0.0

    # Normalized GSI
    total_var = np.var(np.concatenate([[original_expr], shuffle_exprs]))
    gsi_normalized = np.var(shuffle_exprs) / max(total_var, 1e-10)

    # Disruption metrics
    if abs(original_expr) > 1e-10:
        max_disruption = (original_expr - np.min(shuffle_exprs)) / abs(original_expr)
        mean_disruption = (original_expr - np.mean(shuffle_exprs)) / abs(original_expr)
    else:
        max_disruption = 0.0
        mean_disruption = 0.0

    # P-value: permutation-based test
    # Models are deterministic at inference, so repeated predictions of the same
    # sequence give identical results (noise_var = 0). Instead of wasting 20
    # redundant forward passes, use a permutation p-value: where does the
    # original expression rank among the shuffled expressions?
    n_above = np.sum(np.abs(shuffle_exprs - shuffle_mean) >= np.abs(original_expr - shuffle_mean))
    p_value = float(n_above / len(shuffle_exprs)) if len(shuffle_exprs) > 0 else 1.0

    # --- v3 metrics ---
    # Grammar Effect Size (GES): robust z-score using median/MAD
    shuffle_median = np.median(shuffle_exprs)
    shuffle_mad = np.median(np.abs(shuffle_exprs - shuffle_median))
    if shuffle_mad > 1e-10:
        ges = abs(original_expr - shuffle_median) / (shuffle_mad * 1.4826)  # 1.4826 scales MAD to σ
    else:
        ges = abs(original_expr - shuffle_median) / max(shuffle_std, 1e-10)

    # Grammar Practical Effect (GPE): dynamic range relative to median
    gpe = (np.max(shuffle_exprs) - np.min(shuffle_exprs)) / max(abs(shuffle_median), 1e-10)

    # Robust GSI: stabilize denominator
    gsi_robust = shuffle_std / max(abs(shuffle_mean), shuffle_std * 0.1, 1e-10)

    # z-score (standard, for p-value computation)
    z_score = abs(original_expr - shuffle_mean) / max(shuffle_std, 1e-10)
    # Two-sided p-value from z-score
    from scipy.stats import norm
    p_value_zscore = float(2 * (1 - norm.cdf(z_score)))

    return {
        'original_expression': float(original_expr),
        'shuffle_mean': float(shuffle_mean),
        'shuffle_std': float(shuffle_std),
        'shuffle_median': float(shuffle_median),
        'shuffle_mad': float(shuffle_mad),
        'shuffle_expressions': shuffle_exprs.tolist(),
        'gsi': float(gsi),
        'gsi_robust': float(gsi_robust),
        'gsi_normalized': float(gsi_normalized),
        'ges': float(ges),
        'gpe': float(gpe),
        'z_score': float(z_score),
        'max_disruption': float(max_disruption),
        'mean_disruption': float(mean_disruption),
        'n_shuffles': n_shuffles,
        'p_value': float(p_value),
        'p_value_zscore': float(p_value_zscore),
    }


def run_gsi_census(
    dataset: pd.DataFrame,
    model,
    motif_hits: pd.DataFrame,
    n_shuffles: int = 100,
    min_motifs: int = 2,
    cell_type: str = None,
    max_enhancers: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run GSI computation across enhancers for a single model.

    Args:
        dataset: Preprocessed MPRA DataFrame with 'sequence', 'seq_id', 'n_motifs'
        model: GrammarModel instance
        motif_hits: DataFrame of motif scanning results
        n_shuffles: Shuffles per enhancer
        min_motifs: Minimum motifs to include
        cell_type: Cell type context
        max_enhancers: Cap on number of enhancers (for speed)
        seed: Random seed

    Returns:
        DataFrame with GSI for each enhancer
    """
    eligible = dataset[dataset['n_motifs'] >= min_motifs].copy()
    if max_enhancers and len(eligible) > max_enhancers:
        eligible = eligible.sample(n=max_enhancers, random_state=seed)

    print(f"  Computing GSI for {len(eligible)} enhancers (min_motifs={min_motifs})")

    results = []
    for idx, row in tqdm(eligible.iterrows(), total=len(eligible), desc="GSI"):
        seq = row['sequence']
        seq_id = str(row['seq_id'])

        seq_motifs = motif_hits[motif_hits['seq_id'] == seq_id]
        annotation = {
            'sequence': seq,
            'motifs': seq_motifs.to_dict('records'),
            'motif_count': len(seq_motifs),
            'motif_names': list(seq_motifs['motif_name'].unique()) if len(seq_motifs) > 0 else []
        }

        try:
            gsi_result = compute_gsi(
                seq, annotation, model,
                n_shuffles=n_shuffles,
                cell_type=cell_type,
                seed=seed + idx
            )

            results.append({
                'seq_id': seq_id,
                'model': model.name,
                'gsi': gsi_result['gsi'],
                'gsi_robust': gsi_result.get('gsi_robust', gsi_result['gsi']),
                'gsi_normalized': gsi_result['gsi_normalized'],
                'ges': gsi_result.get('ges', np.nan),
                'gpe': gsi_result.get('gpe', np.nan),
                'z_score': gsi_result.get('z_score', np.nan),
                'max_disruption': gsi_result['max_disruption'],
                'mean_disruption': gsi_result['mean_disruption'],
                'p_value': gsi_result['p_value'],
                'p_value_zscore': gsi_result.get('p_value_zscore', np.nan),
                'original_expression': gsi_result['original_expression'],
                'shuffle_mean': gsi_result['shuffle_mean'],
                'shuffle_std': gsi_result['shuffle_std'],
                'shuffle_median': gsi_result.get('shuffle_median', np.nan),
                'shuffle_mad': gsi_result.get('shuffle_mad', np.nan),
                'mpra_expression': row.get('expression', np.nan),
                'n_motifs': row.get('n_motifs', 0),
                'motif_density': row.get('motif_density', 0),
            })
        except Exception as e:
            print(f"  Error for {seq_id}: {e}")
            continue

    return pd.DataFrame(results)


def compute_grammar_information(
    original_expression: float,
    shuffle_expressions: np.ndarray,
    n_bins: int = 20
) -> dict:
    """
    Compute the information content of grammar for a single enhancer.

    Args:
        original_expression: Expression of natural arrangement
        shuffle_expressions: Expressions of vocabulary-matched shuffles
        n_bins: Bins for entropy estimation

    Returns:
        Dict with entropy, percentile, specificity, bits of grammar
    """
    from scipy.stats import entropy as scipy_entropy

    all_expr = np.concatenate([[original_expression], shuffle_expressions])
    bins = np.linspace(all_expr.min() - 1e-10, all_expr.max() + 1e-10, n_bins + 1)

    # Entropy of shuffle distribution
    shuffle_hist, _ = np.histogram(shuffle_expressions, bins=bins)
    shuffle_hist = shuffle_hist / max(shuffle_hist.sum(), 1)
    shuffle_hist = shuffle_hist[shuffle_hist > 0]
    h_shuffles = scipy_entropy(shuffle_hist, base=2)

    # Percentile of original in shuffle distribution
    percentile = float(np.mean(shuffle_expressions <= original_expression))

    # Grammar specificity
    shuffle_std = np.std(shuffle_expressions)
    if shuffle_std > 0:
        grammar_specificity = abs(original_expression - np.mean(shuffle_expressions)) / shuffle_std
    else:
        grammar_specificity = 0.0

    # Bits of grammar (approximate)
    bits = h_shuffles * (1 - 1 / max(grammar_specificity + 1, 1))

    return {
        'entropy_shuffles': float(h_shuffles),
        'percentile_original': percentile,
        'grammar_specificity': float(grammar_specificity),
        'bits_of_grammar': float(bits),
    }
