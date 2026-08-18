"""
Grammar completeness: how much of expression can vocabulary alone explain,
versus vocabulary plus grammar, versus the full model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr


def compute_grammar_completeness(
    dataset: pd.DataFrame,
    motif_hits: pd.DataFrame,
    grammar_features: pd.DataFrame,
    model,
    cell_type: str = None,
    replicate_r2: float = 0.85,
) -> dict:
    """
    R2 at four nested levels: motif counts, + simple grammar stats, + full
    grammar features, and the pretrained model prediction. The MPRA replicate
    ceiling is the reference the levels are read against.
    """
    if 'expression' not in dataset.columns:
        return {'error': 'No expression column in dataset'}

    y = dataset['expression'].values
    valid = np.isfinite(y)
    y = y[valid]

    if len(y) < 50:
        return {'error': f'Too few samples ({len(y)})'}

    # vocabulary
    X_vocab = _build_vocabulary_features(dataset[valid], motif_hits)
    r2_vocab = _cv_r2(X_vocab, y)

    # + simple grammar
    X_simple = _build_simple_grammar_features(dataset[valid], motif_hits, grammar_features)
    r2_simple = _cv_r2(X_simple, y)

    # + full grammar
    X_full = _build_full_grammar_features(dataset[valid], motif_hits, grammar_features)
    r2_full = _cv_r2(X_full, y)

    # full model
    seqs = dataset[valid]['sequence'].tolist()
    preds = model.predict_expression(seqs, cell_type=cell_type)
    if np.std(preds) > 0 and np.std(y) > 0:
        r, _ = pearsonr(preds, y)
        r2_model = float(r ** 2)
    else:
        r2_model = 0.0

    grammar_contribution = (r2_full - r2_vocab) / max(replicate_r2 - r2_vocab, 0.01)
    grammar_gap = r2_model - r2_full

    return {
        'vocabulary_r2': r2_vocab,
        'vocab_plus_simple_grammar_r2': r2_simple,
        'vocab_plus_full_grammar_r2': r2_full,
        'full_model_r2': r2_model,
        'replicate_r2': replicate_r2,
        'grammar_contribution': float(grammar_contribution),
        'grammar_gap': float(grammar_gap),
        'grammar_completeness': float(r2_full / max(replicate_r2, 0.01)),
        'n_samples': len(y),
    }


def _build_vocabulary_features(dataset, motif_hits):
    """Motif count features only."""
    counts = motif_hits.groupby(['seq_id', 'motif_name']).size().unstack(fill_value=0)
    features = dataset[['seq_id']].merge(counts, on='seq_id', how='left').fillna(0)
    return features.drop('seq_id', axis=1).values


def _build_simple_grammar_features(dataset, motif_hits, grammar_features):
    """Vocabulary + basic grammar statistics."""
    vocab = _build_vocabulary_features(dataset, motif_hits)

    if grammar_features is not None and len(grammar_features) > 0:
        grammar_cols = ['spacing_sensitivity', 'orientation_sensitivity',
                        'helical_phase_score', 'fold_change']
        available = [c for c in grammar_cols if c in grammar_features.columns]

        if available:
            gf_stats = grammar_features.groupby('seq_id')[available].mean().reset_index()
            gf_array = dataset[['seq_id']].merge(
                gf_stats, on='seq_id', how='left'
            ).fillna(0).drop('seq_id', axis=1).values
            return np.hstack([vocab, gf_array])

    return vocab


def _build_full_grammar_features(dataset, motif_hits, grammar_features):
    """Full grammar features. Same as simple for now."""
    return _build_simple_grammar_features(dataset, motif_hits, grammar_features)


def _cv_r2(X, y, n_splits=5):
    """Cross-validated R^2 with gradient boosting."""
    if X.shape[1] == 0 or len(X) < 20:
        return 0.0

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        random_state=42, subsample=0.8
    )
    scores = cross_val_score(model, X, y, cv=n_splits, scoring='r2')
    return float(max(scores.mean(), 0))
