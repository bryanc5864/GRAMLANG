"""
ANOVA-Based Variance Decomposition.

Unified framework that decomposes expression variance into:
- Vocabulary (motif presence/absence)
- Pairwise grammar (motif pair interactions)
- Higher-order grammar (k >= 3 interactions)
- Residual (noise)

Replaces ad-hoc metrics with a single interpretable framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler


def compute_anova_decomposition(
    dataset: pd.DataFrame,
    motif_hits: pd.DataFrame,
    model=None,
    cell_type: str = None,
    max_sequences: int = 2000,
    seed: int = 42,
) -> Dict:
    """
    ANOVA-like variance decomposition of expression.

    Fits models of increasing complexity and measures the variance
    explained at each level:
    1. Vocabulary only (motif presence/absence)
    2. + Pairwise grammar (motif pair features)
    3. + Higher-order grammar (triplet features)
    4. Full model predictions (upper bound)

    Returns eta-squared for each level.
    """
    rng = np.random.default_rng(seed)

    # Build feature matrices
    seq_ids, y, X_vocab, X_pairwise, X_higher = _build_feature_hierarchy(
        dataset, motif_hits, max_sequences, rng
    )

    if len(y) < 50:
        return {'error': 'Too few sequences for ANOVA decomposition'}

    scaler = StandardScaler()

    # Level 1: Vocabulary only
    X1 = scaler.fit_transform(X_vocab)
    r2_vocab = _robust_cv_r2(X1, y, seed)

    # Level 2: Vocabulary + Pairwise
    X2 = scaler.fit_transform(np.hstack([X_vocab, X_pairwise]))
    r2_pairwise = _robust_cv_r2(X2, y, seed)

    # Level 3: Vocabulary + Pairwise + Higher
    if X_higher.shape[1] > 0:
        X3 = scaler.fit_transform(np.hstack([X_vocab, X_pairwise, X_higher]))
        r2_higher = _robust_cv_r2(X3, y, seed)
    else:
        r2_higher = r2_pairwise

    # Level 4: Full model predictions (if model provided)
    r2_model = 0
    if model is not None:
        try:
            seqs = dataset[dataset['seq_id'].isin(seq_ids)]['sequence'].tolist()
            if len(seqs) > 0:
                preds = model.predict_expression(seqs[:len(y)], cell_type=cell_type)
                ss_res = np.sum((y[:len(preds)] - preds) ** 2)
                ss_tot = np.sum((y[:len(preds)] - y[:len(preds)].mean()) ** 2)
                r2_model = max(1 - ss_res / max(ss_tot, 1e-10), 0)
        except Exception:
            r2_model = 0

    # Compute eta-squared (variance fraction at each level)
    ss_total = np.sum((y - y.mean()) ** 2)

    eta2_vocab = max(r2_vocab, 0)
    eta2_pairwise = max(r2_pairwise - r2_vocab, 0)
    eta2_higher = max(r2_higher - r2_pairwise, 0)
    eta2_model_residual = max(r2_model - r2_higher, 0)
    eta2_noise = max(1 - r2_model, 0) if r2_model > 0 else max(1 - r2_higher, 0)

    return {
        'n_sequences': len(y),
        'n_vocab_features': int(X_vocab.shape[1]),
        'n_pairwise_features': int(X_pairwise.shape[1]),
        'n_higher_features': int(X_higher.shape[1]),
        'r2_vocabulary': float(r2_vocab),
        'r2_vocab_plus_pairwise': float(r2_pairwise),
        'r2_vocab_plus_all_grammar': float(r2_higher),
        'r2_full_model': float(r2_model),
        'eta2_vocabulary': float(eta2_vocab),
        'eta2_pairwise_grammar': float(eta2_pairwise),
        'eta2_higher_order_grammar': float(eta2_higher),
        'eta2_model_features': float(eta2_model_residual),
        'eta2_unexplained': float(eta2_noise),
        'grammar_total_eta2': float(eta2_pairwise + eta2_higher),
        'vocabulary_dominance': float(eta2_vocab / max(eta2_vocab + eta2_pairwise + eta2_higher, 1e-10)),
    }


def compute_power_analysis(
    effect_size: float,
    n_samples: int,
    alpha: float = 0.05,
) -> Dict:
    """
    Compute statistical power for detecting a given effect size.

    Uses the approximation for F-test power.
    """
    from scipy.stats import f as f_dist, ncf

    # For regression R²: F = (R²/k) / ((1-R²)/(n-k-1))
    k = 1  # Number of predictors (simplified)
    df1 = k
    df2 = n_samples - k - 1

    if df2 <= 0:
        return {'power': 0, 'mde': 0}

    # Non-centrality parameter
    lambda_nc = n_samples * effect_size / (1 - effect_size)

    # Critical F value
    f_crit = f_dist.ppf(1 - alpha, df1, df2)

    # Power = P(F > f_crit | H1)
    power = 1 - ncf.cdf(f_crit, df1, df2, lambda_nc)

    # Minimum detectable effect at 80% power
    from scipy.optimize import brentq
    try:
        def power_fn(r2):
            lam = n_samples * r2 / (1 - r2)
            return 1 - ncf.cdf(f_crit, df1, df2, lam) - 0.8
        mde = brentq(power_fn, 0.001, 0.999)
    except Exception:
        mde = 0

    return {
        'observed_effect': float(effect_size),
        'n_samples': n_samples,
        'alpha': alpha,
        'power': float(power),
        'minimum_detectable_effect_80pct': float(mde),
        'adequately_powered': power > 0.8,
    }


def _build_feature_hierarchy(
    dataset: pd.DataFrame,
    motif_hits: pd.DataFrame,
    max_sequences: int,
    rng: np.random.Generator,
) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build hierarchical feature matrices for ANOVA decomposition."""
    # Get unique motif names
    motif_names = sorted(motif_hits['motif_name'].unique())
    motif_to_idx = {m: i for i, m in enumerate(motif_names)}
    n_motifs = len(motif_names)

    seq_ids = []
    expressions = []
    vocab_features = []
    pairwise_features = []
    higher_features = []

    for seq_id in dataset['seq_id'].unique():
        hits = motif_hits[motif_hits['seq_id'] == seq_id]
        if len(hits) < 2:
            continue

        row = dataset[dataset['seq_id'] == seq_id].iloc[0]
        expr = row.get('expression')
        if expr is None or np.isnan(expr):
            continue

        seq_ids.append(seq_id)
        expressions.append(expr)

        # Vocabulary features: presence + count
        presence = np.zeros(n_motifs)
        counts = np.zeros(n_motifs)
        for _, hit in hits.iterrows():
            idx = motif_to_idx.get(hit['motif_name'])
            if idx is not None:
                presence[idx] = 1
                counts[idx] += 1
        vocab_features.append(np.concatenate([presence, counts]))

        # Pairwise features: spacing + co-occurrence
        hits_sorted = hits.sort_values('start')
        pair_spacing = []
        pair_cooccur = []
        for i in range(min(len(hits_sorted) - 1, 10)):
            for j in range(i + 1, min(len(hits_sorted), 10)):
                h1 = hits_sorted.iloc[i]
                h2 = hits_sorted.iloc[j]
                sp = max(0, int(h2['start']) - int(h1['end']))
                pair_spacing.append(sp)
                pair_cooccur.append(1)

        pairwise_feat = [
            np.mean(pair_spacing) if pair_spacing else 0,
            np.std(pair_spacing) if len(pair_spacing) > 1 else 0,
            np.min(pair_spacing) if pair_spacing else 0,
            np.max(pair_spacing) if pair_spacing else 0,
            len(pair_spacing),
        ]
        pairwise_features.append(pairwise_feat)

        # Higher-order: cluster/motif arrangement features
        positions = sorted(hits_sorted['start'].values)
        if len(positions) >= 3:
            # Triplet statistics
            diffs = np.diff(positions)
            higher_feat = [
                float(np.std(diffs)) if len(diffs) > 1 else 0,
                float(np.min(diffs)) if len(diffs) > 0 else 0,
                float(len(positions)),
                float(np.mean(positions)) / max(len(row.get('sequence', 'A')), 1),
            ]
        else:
            higher_feat = [0, 0, float(len(positions)), 0]
        higher_features.append(higher_feat)

    if not seq_ids:
        return [], np.array([]), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))

    # Subsample if needed
    y = np.array(expressions)
    X_v = np.array(vocab_features)
    X_p = np.array(pairwise_features)
    X_h = np.array(higher_features)

    if len(y) > max_sequences:
        idx = rng.choice(len(y), max_sequences, replace=False)
        seq_ids = [seq_ids[i] for i in idx]
        y = y[idx]
        X_v = X_v[idx]
        X_p = X_p[idx]
        X_h = X_h[idx]

    return seq_ids, y, X_v, X_p, X_h


def _robust_cv_r2(X, y, seed, cv=5):
    """Compute cross-validated R² using gradient boosting."""
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=seed, subsample=0.8
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return float(max(scores.mean(), 0))
