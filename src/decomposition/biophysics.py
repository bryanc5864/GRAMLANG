"""
Biophysics residual analysis.

Tests how much of grammar can be explained by biophysical features
(DNA shape, nucleosome affinity, GC content, dinucleotide frequencies).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from typing import Dict, List


def compute_biophysical_features(sequence: str) -> np.ndarray:
    """
    Compute a vector of biophysical features for a DNA sequence.

    Features (~35 dimensions):
    1. GC content (1)
    2. Dinucleotide frequencies (16)
    3. DNA shape approximations (16) - from dinucleotide step parameters
    4. Bendability proxy (1)
    5. Nucleosome affinity proxy (1)

    Args:
        sequence: DNA sequence

    Returns:
        Feature vector
    """
    seq = sequence.upper()
    n = len(seq)
    features = []

    # 1. GC content
    gc = (seq.count('G') + seq.count('C')) / max(n, 1)
    features.append(gc)

    # 2. Dinucleotide frequencies
    dinucs = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
              'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    total_dn = max(n - 1, 1)
    for dn in dinucs:
        count = sum(1 for i in range(n - 1) if seq[i:i+2] == dn)
        features.append(count / total_dn)

    # 3. DNA shape proxies (from dinucleotide step parameters)
    # Minor groove width approximation
    # AT-rich -> narrow minor groove; GC-rich -> wider
    mgw_params = {
        'AA': 3.26, 'AT': 3.30, 'TA': 3.80, 'TT': 3.26,
        'AC': 4.70, 'AG': 4.20, 'CA': 4.60, 'GA': 4.20,
        'CC': 5.00, 'CG': 5.50, 'GC': 5.20, 'GG': 5.00,
        'CT': 4.20, 'GT': 4.70, 'TC': 4.60, 'TG': 4.60,
    }
    mgw_values = [mgw_params.get(seq[i:i+2], 4.5) for i in range(n-1)]
    features.extend([np.mean(mgw_values), np.std(mgw_values),
                     np.min(mgw_values), np.max(mgw_values)])

    # Roll angle approximation
    roll_params = {
        'AA': -0.5, 'AT': -0.2, 'TA': 3.5, 'TT': -0.5,
        'AC': 0.5, 'AG': 0.3, 'CA': 1.0, 'GA': 0.3,
        'CC': 0.1, 'CG': 0.6, 'GC': 0.7, 'GG': 0.1,
        'CT': 0.3, 'GT': 0.5, 'TC': 1.0, 'TG': 1.0,
    }
    roll_values = [roll_params.get(seq[i:i+2], 0.5) for i in range(n-1)]
    features.extend([np.mean(roll_values), np.std(roll_values),
                     np.min(roll_values), np.max(roll_values)])

    # Propeller twist approximation
    prot_params = {
        'AA': -16.0, 'AT': -15.5, 'TA': -11.0, 'TT': -16.0,
        'AC': -13.0, 'AG': -14.0, 'CA': -13.5, 'GA': -14.0,
        'CC': -11.5, 'CG': -11.0, 'GC': -12.0, 'GG': -11.5,
        'CT': -14.0, 'GT': -13.0, 'TC': -13.5, 'TG': -13.5,
    }
    prot_values = [prot_params.get(seq[i:i+2], -13.0) for i in range(n-1)]
    features.extend([np.mean(prot_values), np.std(prot_values),
                     np.min(prot_values), np.max(prot_values)])

    # Helix twist approximation
    helt_params = {
        'AA': 35.6, 'AT': 31.5, 'TA': 36.0, 'TT': 35.6,
        'AC': 34.4, 'AG': 27.7, 'CA': 34.5, 'GA': 27.7,
        'CC': 33.7, 'CG': 29.8, 'GC': 40.0, 'GG': 33.7,
        'CT': 27.7, 'GT': 34.4, 'TC': 34.5, 'TG': 34.5,
    }
    helt_values = [helt_params.get(seq[i:i+2], 34.0) for i in range(n-1)]
    features.extend([np.mean(helt_values), np.std(helt_values),
                     np.min(helt_values), np.max(helt_values)])

    # 4. Bendability proxy (sum of trinucleotide bendability)
    # Simplified: use AT/GC alternation as proxy
    alternation = sum(1 for i in range(n-1) if
                      (seq[i] in 'AT' and seq[i+1] in 'GC') or
                      (seq[i] in 'GC' and seq[i+1] in 'AT'))
    features.append(alternation / max(n-1, 1))

    # 5. Nucleosome affinity proxy
    # GC-rich ~ higher nucleosome affinity, poly-A/T ~ lower
    poly_at_runs = 0
    max_run = 0
    current_run = 0
    for base in seq:
        if base in 'AT':
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            if current_run >= 4:
                poly_at_runs += 1
            current_run = 0
    features.append(max_run / max(n, 1))

    return np.array(features)


def get_biophysical_feature_names() -> List[str]:
    """Return names of all biophysical features."""
    names = ['gc_content']
    dinucs = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
              'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    names.extend([f'dinuc_{dn}' for dn in dinucs])
    for shape in ['MGW', 'Roll', 'ProT', 'HelT']:
        for stat in ['mean', 'std', 'min', 'max']:
            names.append(f'shape_{shape}_{stat}')
    names.append('bendability')
    names.append('nucleosome_poly_at')
    return names


def compute_biophysics_residual(
    enhancers: pd.DataFrame,
    grammar_effects: np.ndarray,
) -> dict:
    """
    Compute how much grammar variance biophysics explains.

    Args:
        enhancers: DataFrame with 'sequence' column
        grammar_effects: Array of grammar effect sizes (e.g., GSI values)

    Returns:
        Dict with biophysics R^2, feature importances
    """
    # Compute biophysical features for all sequences
    X = np.array([compute_biophysical_features(seq)
                   for seq in enhancers['sequence'].values])
    y = np.array(grammar_effects)

    # Remove NaN/inf
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]

    if len(X) < 20:
        return {'error': 'Too few valid samples', 'biophysics_r2': 0.0}

    # Fit model
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        random_state=42, subsample=0.8
    )
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    biophysics_r2 = float(scores.mean())

    # Feature importances
    model.fit(X, y)
    feature_names = get_biophysical_feature_names()
    importances = dict(zip(feature_names[:len(model.feature_importances_)],
                           model.feature_importances_.tolist()))

    # Sort by importance
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    return {
        'biophysics_r2': biophysics_r2,
        'biophysics_r2_std': float(scores.std()),
        'total_grammar_variance': float(np.var(y)),
        'residual_variance': float(np.var(y) * (1 - max(biophysics_r2, 0))),
        'biophysics_explained_fraction': max(biophysics_r2, 0),
        'feature_importances': importances,
        'n_samples': len(X),
        'n_features': X.shape[1],
    }
