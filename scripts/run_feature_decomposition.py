#!/usr/bin/env python3
"""
GRAMLANG v3: Feature Decomposition Analysis (Critical Gap 2)

Analyze what foundation models actually learn by decomposing their
predictions into interpretable features:

1. K-mer composition (1-mer to 6-mer frequencies)
2. GC content and dinucleotide statistics
3. DNA shape features (MGW, Roll, ProT, HelT)
4. Spacer-only features vs motif features

Goal: Quantify X% GC, Y% dinucleotide, Z% DNA shape, W% other
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from collections import Counter
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model
from src.models.expression_probes import load_probe
from src.utils.io import load_processed, save_json

RESULTS_DIR = Path('results/v3/feature_decomposition')
PROBES_DIR = Path('data/probes')


def compute_kmer_features(sequence, k_range=(1, 6)):
    """Compute k-mer frequency features for a sequence."""
    features = {}
    bases = 'ACGT'

    for k in range(k_range[0], k_range[1] + 1):
        # All possible k-mers
        all_kmers = [''.join(p) for p in product(bases, repeat=k)]
        kmer_counts = Counter(sequence[i:i+k] for i in range(len(sequence) - k + 1))
        total = sum(kmer_counts.values())

        for kmer in all_kmers:
            features[f'kmer_{kmer}'] = kmer_counts.get(kmer, 0) / max(total, 1)

    return features


def compute_gc_features(sequence):
    """Compute GC-related features."""
    gc = (sequence.count('G') + sequence.count('C')) / len(sequence)
    at = 1 - gc

    # GC in different windows
    window_size = 50
    gc_profile = []
    for i in range(0, len(sequence) - window_size + 1, window_size // 2):
        window = sequence[i:i+window_size]
        gc_profile.append((window.count('G') + window.count('C')) / len(window))

    return {
        'gc_content': gc,
        'at_content': at,
        'gc_variance': np.var(gc_profile) if gc_profile else 0,
        'gc_max': max(gc_profile) if gc_profile else gc,
        'gc_min': min(gc_profile) if gc_profile else gc,
    }


def compute_dinuc_features(sequence):
    """Compute dinucleotide frequencies."""
    dinucs = [''.join(p) for p in product('ACGT', repeat=2)]
    counts = Counter(sequence[i:i+2] for i in range(len(sequence) - 1))
    total = sum(counts.values())

    return {f'dinuc_{d}': counts.get(d, 0) / max(total, 1) for d in dinucs}


def compute_shape_features(sequence):
    """
    Compute DNA shape features using simple sequence-based approximations.

    Real DNA shape (MGW, Roll, ProT, HelT) requires lookup tables or
    external tools. Here we use dinucleotide-based proxies.
    """
    # Dinucleotide step parameters (simplified approximations)
    # Based on Rohs et al. structural studies
    roll_values = {
        'AA': 0.0, 'AT': -5.4, 'AG': 4.5, 'AC': 4.0,
        'TA': 2.5, 'TT': 0.0, 'TG': 4.0, 'TC': 4.5,
        'GA': -1.0, 'GT': 4.0, 'GG': 0.5, 'GC': -0.1,
        'CA': 4.0, 'CT': -1.0, 'CG': 1.2, 'CC': 0.5,
    }

    twist_values = {
        'AA': 35.6, 'AT': 31.5, 'AG': 32.0, 'AC': 34.4,
        'TA': 36.0, 'TT': 35.6, 'TG': 34.4, 'TC': 32.0,
        'GA': 36.9, 'GT': 34.4, 'GG': 33.6, 'GC': 40.0,
        'CA': 34.4, 'CT': 36.9, 'CG': 29.8, 'CC': 33.6,
    }

    rolls = []
    twists = []
    for i in range(len(sequence) - 1):
        dinuc = sequence[i:i+2]
        if dinuc in roll_values:
            rolls.append(roll_values[dinuc])
            twists.append(twist_values[dinuc])

    if not rolls:
        return {'roll_mean': 0, 'roll_std': 0, 'twist_mean': 36, 'twist_std': 0}

    return {
        'roll_mean': np.mean(rolls),
        'roll_std': np.std(rolls),
        'roll_min': np.min(rolls),
        'roll_max': np.max(rolls),
        'twist_mean': np.mean(twists),
        'twist_std': np.std(twists),
    }


def extract_all_features(sequence):
    """Extract all sequence features."""
    features = {}
    features.update(compute_gc_features(sequence))
    features.update(compute_dinuc_features(sequence))
    features.update(compute_shape_features(sequence))
    # Skip full k-mer for speed; use dinucs + trinucs only
    for k in [3]:
        kmer_feats = compute_kmer_features(sequence, k_range=(k, k))
        features.update(kmer_feats)
    return features


def swap_probe(model, model_name, ds_name, device='cuda'):
    """Swap expression probe on already-loaded model."""
    if model_name == 'enformer' or not hasattr(model, 'set_probe'):
        return
    for probe_name in [f'{model_name}_{ds_name}', f'{model_name}_vaishnav2022']:
        probe_path = PROBES_DIR / f'{probe_name}_probe.pt'
        if probe_path.exists():
            probe = load_probe(str(PROBES_DIR), probe_name, model.hidden_dim, device=device)
            model.set_probe(probe)
            print(f"  Loaded probe: {probe_name}")
            return
    print(f"  WARNING: No probe found for {model_name}/{ds_name}")


def run_feature_decomposition(dataset_name, model_name='dnabert2', n_samples=500, seed=42):
    """
    Decompose model predictions into interpretable features.
    """
    print(f"\n{'='*60}")
    print(f"FEATURE DECOMPOSITION — {dataset_name} / {model_name}")
    print(f"{'='*60}")

    # Load data
    dataset = load_processed(f'data/processed/{dataset_name}_processed.parquet')
    if len(dataset) > n_samples:
        dataset = dataset.sample(n=n_samples, random_state=seed)

    print(f"  Samples: {len(dataset)}")

    # Load model
    print(f"  Loading model {model_name}...")
    model = load_model(model_name, dataset_name='__dummy__')
    swap_probe(model, model_name, dataset_name)

    # Get model predictions
    sequences = dataset['sequence'].tolist()
    predictions = model.predict_expression(sequences)

    # Extract features for all sequences
    print("  Extracting sequence features...")
    feature_dicts = [extract_all_features(seq) for seq in sequences]
    feature_df = pd.DataFrame(feature_dicts)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df.values)
    y = predictions

    # Fit models to decompose variance
    print("  Fitting regression models...")
    results = {}

    # 1. GC content only
    gc_cols = [c for c in feature_df.columns if c.startswith('gc_') or c.startswith('at_')]
    if gc_cols:
        X_gc = scaler.fit_transform(feature_df[gc_cols].values)
        scores_gc = cross_val_score(Ridge(alpha=1.0), X_gc, y, cv=5, scoring='r2')
        results['gc_only'] = {
            'mean_r2': float(np.mean(scores_gc)),
            'std_r2': float(np.std(scores_gc)),
            'features': gc_cols,
        }
        print(f"    GC only: R² = {results['gc_only']['mean_r2']:.4f}")

    # 2. Dinucleotide frequencies
    dinuc_cols = [c for c in feature_df.columns if c.startswith('dinuc_')]
    if dinuc_cols:
        X_dinuc = scaler.fit_transform(feature_df[dinuc_cols].values)
        scores_dinuc = cross_val_score(Ridge(alpha=1.0), X_dinuc, y, cv=5, scoring='r2')
        results['dinuc_only'] = {
            'mean_r2': float(np.mean(scores_dinuc)),
            'std_r2': float(np.std(scores_dinuc)),
            'n_features': len(dinuc_cols),
        }
        print(f"    Dinuc only: R² = {results['dinuc_only']['mean_r2']:.4f}")

    # 3. DNA shape
    shape_cols = [c for c in feature_df.columns if c.startswith('roll_') or c.startswith('twist_')]
    if shape_cols:
        X_shape = scaler.fit_transform(feature_df[shape_cols].values)
        scores_shape = cross_val_score(Ridge(alpha=1.0), X_shape, y, cv=5, scoring='r2')
        results['shape_only'] = {
            'mean_r2': float(np.mean(scores_shape)),
            'std_r2': float(np.std(scores_shape)),
            'features': shape_cols,
        }
        print(f"    DNA shape only: R² = {results['shape_only']['mean_r2']:.4f}")

    # 4. Trinucleotide (k=3) frequencies
    trinuc_cols = [c for c in feature_df.columns if c.startswith('kmer_') and len(c.split('_')[1]) == 3]
    if trinuc_cols:
        X_trinuc = scaler.fit_transform(feature_df[trinuc_cols].values)
        scores_trinuc = cross_val_score(Ridge(alpha=1.0), X_trinuc, y, cv=5, scoring='r2')
        results['trinuc_only'] = {
            'mean_r2': float(np.mean(scores_trinuc)),
            'std_r2': float(np.std(scores_trinuc)),
            'n_features': len(trinuc_cols),
        }
        print(f"    Trinuc only: R² = {results['trinuc_only']['mean_r2']:.4f}")

    # 5. All sequence features combined
    scores_all = cross_val_score(Ridge(alpha=1.0), X, y, cv=5, scoring='r2')
    results['all_features'] = {
        'mean_r2': float(np.mean(scores_all)),
        'std_r2': float(np.std(scores_all)),
        'n_features': X.shape[1],
    }
    print(f"    All features: R² = {results['all_features']['mean_r2']:.4f}")

    # 6. Feature importances using Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    importances = dict(zip(feature_df.columns, rf.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:20]
    results['top_features'] = [(f, float(v)) for f, v in top_features]

    print(f"\n  Top 5 features:")
    for feat, imp in top_features[:5]:
        print(f"    {feat}: {imp:.4f}")

    # Correlation analysis
    print("\n  Feature-prediction correlations:")
    correlations = {}
    for col in ['gc_content', 'dinuc_CG', 'dinuc_GC', 'roll_mean', 'twist_std']:
        if col in feature_df.columns:
            r, p = stats.spearmanr(feature_df[col], y)
            correlations[col] = {'r': float(r), 'p': float(p)}
            print(f"    {col}: r = {r:.3f}")
    results['correlations'] = correlations

    # Summary
    results['dataset'] = dataset_name
    results['model'] = model_name
    results['n_samples'] = len(dataset)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(results, RESULTS_DIR / f'{dataset_name}_{model_name}_decomposition.json')
    print(f"\n  Saved to {RESULTS_DIR}/")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Feature Decomposition Analysis')
    parser.add_argument('--datasets', type=str, default='agarwal,jores')
    parser.add_argument('--model', type=str, default='dnabert2')
    parser.add_argument('--n-samples', type=int, default=500)
    args = parser.parse_args()

    datasets = args.datasets.split(',')
    all_results = {}

    for ds in datasets:
        results = run_feature_decomposition(ds, model_name=args.model, n_samples=args.n_samples)
        all_results[ds] = results

    # Summary
    print(f"\n{'='*60}")
    print("FEATURE DECOMPOSITION SUMMARY")
    print(f"{'='*60}")

    for ds, results in all_results.items():
        print(f"\n{ds}:")
        for key in ['gc_only', 'dinuc_only', 'shape_only', 'all_features']:
            if key in results:
                print(f"  {key}: R² = {results[key]['mean_r2']:.4f}")

    save_json(all_results, RESULTS_DIR / 'feature_decomposition_summary.json')


if __name__ == '__main__':
    main()
