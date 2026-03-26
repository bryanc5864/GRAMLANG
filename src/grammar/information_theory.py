"""
Experiment E: Information-Theoretic Decomposition.

Decomposes expression variance into vocabulary vs grammar components
using mutual information estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def compute_information_decomposition(
    dataset: pd.DataFrame,
    motif_hits: pd.DataFrame,
    model,
    cell_type: str = None,
    n_shuffles: int = 50,
    max_enhancers: int = 500,
    seed: int = 42,
) -> Dict:
    """
    Information-theoretic decomposition of expression into vocabulary + grammar.

    Computes:
    - I(Expression; Vocabulary): how much motif presence/absence explains expression
    - I(Expression; Arrangement | Vocabulary): grammar information controlling for vocabulary
    - R² decomposition as a proxy for mutual information

    Uses R² decomposition rather than raw MI estimation (which is noisy
    for continuous variables in finite samples).
    """
    rng = np.random.default_rng(seed)

    # Build vocabulary feature matrix (bag of motifs)
    vocab_features, seq_ids = _build_vocabulary_features(dataset, motif_hits)

    if len(vocab_features) < 50:
        return {'error': 'Too few sequences for information decomposition'}

    # Get expressions
    expr_df = dataset[dataset['seq_id'].isin(seq_ids)].set_index('seq_id')
    expressions = expr_df.loc[seq_ids, 'expression'].values.astype(float)

    # Limit sample size
    if len(expressions) > max_enhancers:
        idx = rng.choice(len(expressions), max_enhancers, replace=False)
        vocab_features = vocab_features[idx]
        expressions = expressions[idx]
        seq_ids = [seq_ids[i] for i in idx]

    # 1. R²(Expression ~ Vocabulary)
    scaler = StandardScaler()
    X_vocab = scaler.fit_transform(vocab_features)

    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import GradientBoostingRegressor

    # Use gradient boosting for nonlinear vocabulary -> expression mapping
    gb_vocab = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed
    )
    r2_vocab_scores = cross_val_score(gb_vocab, X_vocab, expressions, cv=5, scoring='r2')
    r2_vocab = float(max(r2_vocab_scores.mean(), 0))

    # Linear version for comparison
    lr = LinearRegression()
    r2_vocab_linear_scores = cross_val_score(lr, X_vocab, expressions, cv=5, scoring='r2')
    r2_vocab_linear = float(max(r2_vocab_linear_scores.mean(), 0))

    # 2. Build pairwise grammar features
    grammar_features = _build_pairwise_grammar_features(
        seq_ids, motif_hits, dataset
    )

    if grammar_features is not None and grammar_features.shape[1] > 0:
        X_grammar = scaler.fit_transform(grammar_features)

        # R²(Expression ~ Vocabulary + Grammar)
        X_combined = np.hstack([X_vocab, X_grammar])
        gb_combined = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed
        )
        r2_combined_scores = cross_val_score(gb_combined, X_combined, expressions, cv=5, scoring='r2')
        r2_combined = float(max(r2_combined_scores.mean(), 0))

        # R²(Expression ~ Grammar only)
        gb_grammar = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=seed
        )
        r2_grammar_scores = cross_val_score(gb_grammar, X_grammar, expressions, cv=5, scoring='r2')
        r2_grammar_only = float(max(r2_grammar_scores.mean(), 0))
    else:
        r2_combined = r2_vocab
        r2_grammar_only = 0

    # 3. Compute conditional grammar information
    # I(Expression; Grammar | Vocabulary) ≈ R²(combined) - R²(vocabulary)
    grammar_information = max(r2_combined - r2_vocab, 0)

    # 4. Model predictions for upper bound
    try:
        # Get model predictions for a subset
        subset_seqs = dataset[dataset['seq_id'].isin(seq_ids[:min(200, len(seq_ids))])]['sequence'].tolist()
        if subset_seqs:
            model_preds = model.predict_expression(subset_seqs, cell_type=cell_type)
            model_r2 = 1 - np.sum((expressions[:len(model_preds)] - model_preds)**2) / np.sum((expressions[:len(model_preds)] - expressions[:len(model_preds)].mean())**2)
            model_r2 = float(max(model_r2, 0))
        else:
            model_r2 = 0
    except Exception:
        model_r2 = 0

    return {
        'n_sequences': len(expressions),
        'n_vocab_features': int(vocab_features.shape[1]),
        'n_grammar_features': int(grammar_features.shape[1]) if grammar_features is not None else 0,
        'r2_vocabulary_gb': r2_vocab,
        'r2_vocabulary_linear': r2_vocab_linear,
        'r2_grammar_only': r2_grammar_only,
        'r2_vocab_plus_grammar': r2_combined,
        'grammar_information': grammar_information,
        'model_r2': model_r2,
        'vocabulary_fraction': r2_vocab / max(r2_combined, 1e-10),
        'grammar_fraction': grammar_information / max(r2_combined, 1e-10),
        'unexplained_fraction': max(1 - r2_combined, 0),
    }


def _build_vocabulary_features(
    dataset: pd.DataFrame, motif_hits: pd.DataFrame
) -> Tuple[np.ndarray, List[str]]:
    """Build bag-of-motifs feature matrix."""
    # Get unique motif names
    motif_names = sorted(motif_hits['motif_name'].unique())
    motif_to_idx = {m: i for i, m in enumerate(motif_names)}

    seq_ids = []
    features = []

    for seq_id in dataset['seq_id'].unique():
        hits = motif_hits[motif_hits['seq_id'] == seq_id]
        if len(hits) < 2:
            continue

        # Binary presence/absence + count features
        presence = np.zeros(len(motif_names))
        counts = np.zeros(len(motif_names))
        for _, hit in hits.iterrows():
            idx = motif_to_idx.get(hit['motif_name'])
            if idx is not None:
                presence[idx] = 1
                counts[idx] += 1

        feature_vec = np.concatenate([presence, counts])
        features.append(feature_vec)
        seq_ids.append(seq_id)

    if not features:
        return np.zeros((0, 0)), []

    return np.array(features), seq_ids


def _build_pairwise_grammar_features(
    seq_ids: List[str], motif_hits: pd.DataFrame, dataset: pd.DataFrame
) -> np.ndarray:
    """Build pairwise grammar features (spacing, orientation between motif pairs)."""
    features = []

    for seq_id in seq_ids:
        hits = motif_hits[motif_hits['seq_id'] == seq_id].sort_values('start')
        if len(hits) < 2:
            features.append(np.zeros(6))  # Fixed-size feature
            continue

        # Compute pairwise features for up to first 5 motifs
        spacings = []
        overlaps = []
        for i in range(min(len(hits) - 1, 5)):
            h1 = hits.iloc[i]
            h2 = hits.iloc[i + 1]
            sp = max(0, int(h2['start']) - int(h1['end']))
            spacings.append(sp)
            overlaps.append(1 if int(h2['start']) < int(h1['end']) else 0)

        feat = [
            np.mean(spacings) if spacings else 0,
            np.std(spacings) if len(spacings) > 1 else 0,
            np.min(spacings) if spacings else 0,
            np.max(spacings) if spacings else 0,
            np.mean(overlaps) if overlaps else 0,
            len(hits),
        ]
        features.append(feat)

    return np.array(features) if features else None
