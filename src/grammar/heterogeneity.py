"""
Experiment F: Grammar Heterogeneity Analysis.

Identifies grammar-rich enhancers and characterizes what makes them special.
Shifts the narrative from 'grammar is weak on average' to 'grammar is a
property of specific regulatory architectures.'
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score

from src.utils.sequence import gc_content


def analyze_grammar_heterogeneity(
    gsi_results: pd.DataFrame,
    dataset: pd.DataFrame,
    motif_hits: pd.DataFrame,
    top_percentile: float = 0.05,
) -> Dict:
    """
    Analyze grammar heterogeneity: what distinguishes grammar-rich enhancers?

    Args:
        gsi_results: GSI measurements with seq_id, gsi, dataset columns
        dataset: Processed dataset with sequences
        motif_hits: Motif annotations
        top_percentile: Fraction for grammar-rich cutoff (default: top 5%)

    Returns:
        Dict with heterogeneity analysis results
    """
    # Aggregate GSI per enhancer (mean across models)
    gsi_per_seq = gsi_results.groupby('seq_id').agg(
        mean_gsi=('gsi', 'mean'),
        max_gsi=('gsi', 'max'),
        std_gsi=('gsi', 'std'),
        n_models=('model', 'nunique'),
    ).reset_index()

    # Merge with dataset features
    merged = gsi_per_seq.merge(dataset[['seq_id', 'sequence', 'expression']], on='seq_id', how='inner')

    if len(merged) < 20:
        return {'error': 'Too few sequences for heterogeneity analysis'}

    # Compute sequence features
    features = _compute_sequence_features(merged, motif_hits)

    # Classify grammar-rich vs grammar-poor
    gsi_threshold = merged['mean_gsi'].quantile(1 - top_percentile)
    merged['grammar_rich'] = merged['mean_gsi'] >= gsi_threshold
    n_rich = merged['grammar_rich'].sum()
    n_poor = len(merged) - n_rich

    # Compare features between grammar-rich and grammar-poor
    feature_comparison = {}
    feature_cols = [c for c in features.columns if c not in ['seq_id']]
    feature_df = features.merge(merged[['seq_id', 'mean_gsi', 'grammar_rich']], on='seq_id')

    for col in feature_cols:
        rich_vals = feature_df[feature_df['grammar_rich']][col].dropna()
        poor_vals = feature_df[~feature_df['grammar_rich']][col].dropna()
        if len(rich_vals) > 0 and len(poor_vals) > 0:
            from scipy.stats import mannwhitneyu
            try:
                stat, pval = mannwhitneyu(rich_vals, poor_vals, alternative='two-sided')
                feature_comparison[col] = {
                    'rich_mean': float(rich_vals.mean()),
                    'poor_mean': float(poor_vals.mean()),
                    'difference': float(rich_vals.mean() - poor_vals.mean()),
                    'mannwhitney_p': float(pval),
                }
            except Exception:
                pass

    # Train random forest to predict GSI from sequence features
    X = features.merge(merged[['seq_id', 'mean_gsi']], on='seq_id')
    y = X['mean_gsi'].values
    X_features = X[feature_cols].fillna(0).values

    if len(X_features) >= 50:
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        cv_scores = cross_val_score(rf, X_features, y, cv=5, scoring='r2')
        rf.fit(X_features, y)
        importances = dict(zip(feature_cols, rf.feature_importances_))
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    else:
        cv_scores = np.array([0])
        importances = {}

    # Grammar type clustering among rich enhancers
    grammar_types = _classify_grammar_types(
        merged[merged['grammar_rich']], gsi_results, motif_hits
    )

    # Distribution statistics
    gsi_values = merged['mean_gsi'].values
    percentiles = {
        'p5': float(np.percentile(gsi_values, 5)),
        'p25': float(np.percentile(gsi_values, 25)),
        'p50': float(np.percentile(gsi_values, 50)),
        'p75': float(np.percentile(gsi_values, 75)),
        'p95': float(np.percentile(gsi_values, 95)),
        'p99': float(np.percentile(gsi_values, 99)),
    }

    return {
        'n_sequences': len(merged),
        'n_grammar_rich': int(n_rich),
        'n_grammar_poor': int(n_poor),
        'gsi_threshold': float(gsi_threshold),
        'gsi_distribution': percentiles,
        'gsi_mean': float(gsi_values.mean()),
        'gsi_std': float(gsi_values.std()),
        'gsi_skewness': float(pd.Series(gsi_values).skew()),
        'gsi_kurtosis': float(pd.Series(gsi_values).kurtosis()),
        'feature_comparison': feature_comparison,
        'predictor_r2_cv': float(cv_scores.mean()),
        'predictor_r2_std': float(cv_scores.std()),
        'feature_importances': {k: float(v) for k, v in list(importances.items())[:15]},
        'grammar_types': grammar_types,
    }


def _compute_sequence_features(merged: pd.DataFrame, motif_hits: pd.DataFrame) -> pd.DataFrame:
    """Compute sequence-level features for heterogeneity analysis."""
    features = []

    for _, row in merged.iterrows():
        seq = row['sequence']
        seq_id = row['seq_id']
        seq_hits = motif_hits[motif_hits['seq_id'] == seq_id]

        # Basic features
        feat = {'seq_id': seq_id}
        feat['seq_length'] = len(seq)
        feat['gc_content'] = gc_content(seq)
        feat['expression'] = row.get('expression', np.nan)

        # Motif features
        feat['n_motifs'] = len(seq_hits)
        feat['n_unique_motifs'] = seq_hits['motif_name'].nunique() if len(seq_hits) > 0 else 0
        feat['motif_density'] = len(seq_hits) / max(len(seq), 1) * 100

        # Motif coverage
        if len(seq_hits) > 0 and 'start' in seq_hits.columns and 'end' in seq_hits.columns:
            covered = np.zeros(len(seq))
            for _, hit in seq_hits.iterrows():
                s, e = int(hit['start']), int(hit['end'])
                covered[s:e] = 1
            feat['motif_coverage'] = float(covered.mean())

            # Motif clustering: mean distance between consecutive motifs
            starts = sorted(seq_hits['start'].values)
            if len(starts) > 1:
                distances = np.diff(starts)
                feat['mean_motif_distance'] = float(distances.mean())
                feat['min_motif_distance'] = float(distances.min())
                feat['motif_clustering'] = float(distances.std())
            else:
                feat['mean_motif_distance'] = float(len(seq))
                feat['min_motif_distance'] = float(len(seq))
                feat['motif_clustering'] = 0
        else:
            feat['motif_coverage'] = 0
            feat['mean_motif_distance'] = float(len(seq))
            feat['min_motif_distance'] = float(len(seq))
            feat['motif_clustering'] = 0

        # Dinucleotide frequencies
        dinucs = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
                  'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
        for di in dinucs:
            count = sum(1 for i in range(len(seq)-1) if seq[i:i+2].upper() == di)
            feat[f'dinuc_{di}'] = count / max(len(seq) - 1, 1)

        # CpG ratio
        obs_cpg = feat.get('dinuc_CG', 0)
        c_freq = sum(1 for c in seq.upper() if c == 'C') / max(len(seq), 1)
        g_freq = sum(1 for c in seq.upper() if c == 'G') / max(len(seq), 1)
        expected_cpg = c_freq * g_freq
        feat['cpg_ratio'] = obs_cpg / max(expected_cpg, 1e-10)

        features.append(feat)

    return pd.DataFrame(features)


def _classify_grammar_types(
    rich_df: pd.DataFrame,
    gsi_results: pd.DataFrame,
    motif_hits: pd.DataFrame
) -> Dict:
    """Classify grammar-rich enhancers into grammar type categories."""
    # For now, classify by motif count and density patterns
    types = {
        'high_density': 0,  # Many motifs, tightly packed
        'moderate_density': 0,  # Moderate motif count
        'sparse': 0,  # Few motifs but strong grammar
    }

    for _, row in rich_df.iterrows():
        seq_id = row['seq_id']
        n_motifs = len(motif_hits[motif_hits['seq_id'] == seq_id])
        seq_len = len(row['sequence']) if 'sequence' in row.index else 200
        density = n_motifs / seq_len * 100

        if density > 5:
            types['high_density'] += 1
        elif n_motifs >= 4:
            types['moderate_density'] += 1
        else:
            types['sparse'] += 1

    return types
