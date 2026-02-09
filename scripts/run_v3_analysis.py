#!/usr/bin/env python3
"""
GRAMLANG v3: Extension Analysis Pipeline

Runs the critical P0 and P1 experiments from EXTENSION_PLAN.md:
  P0.3: Power analysis with 1000 shuffles
  P1.1: Factorial shuffle decomposition (position vs orientation vs spacer)
  P1.2: Bag-of-motifs baseline comparison
  P1.3: Unexplained variance decomposition

Usage:
    python scripts/run_v3_analysis.py [--phase P0.3|P1.1|P1.2|P1.3|all]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model
from src.perturbation.motif_scanner import MotifScanner
from src.perturbation.vocabulary_preserving import (
    generate_vocabulary_preserving_shuffles,
    generate_position_only_shuffles,
    generate_orientation_only_shuffles,
    generate_spacer_only_shuffles,
)
from src.grammar.sensitivity import compute_gsi
from src.utils.io import load_processed, save_json

# Configuration
DATASETS = ['agarwal', 'klein', 'de_almeida', 'vaishnav', 'jores']
MODELS = ['dnabert2', 'nt', 'hyenadna']
RESULTS_DIR = Path('results/v3')
V2_RESULTS_DIR = Path('results/v2')

DATASET_CELL_TYPES = {
    'agarwal': 'K562',
    'klein': 'HepG2',
    'de_almeida': 'neural',
    'vaishnav': 'yeast',
    'jores': 'plant',
}


def ensure_dirs():
    """Create output directories."""
    for d in ['power_analysis', 'factorial_decomposition', 'bom_baseline',
              'variance_decomposition']:
        (RESULTS_DIR / d).mkdir(parents=True, exist_ok=True)


PROBES_DIR = 'data/probes'


def swap_probe(model, model_name, ds_name, device='cuda'):
    """Swap expression probe on already-loaded model."""
    if model_name == 'enformer' or not hasattr(model, 'set_probe'):
        return
    from src.models.expression_probes import load_probe
    for probe_name in [f'{model_name}_{ds_name}', f'{model_name}_vaishnav',
                       f'{model_name}_vaishnav2022']:
        probe_path = os.path.join(PROBES_DIR, f'{probe_name}_probe.pt')
        if os.path.exists(probe_path):
            probe = load_probe(PROBES_DIR, probe_name, model.hidden_dim, device=device)
            model.set_probe(probe)
            print(f"  Loaded probe: {probe_name}")
            return
    print(f"  WARNING: No probe found for {model_name}/{ds_name}")


def load_data_and_model(dataset_name, model_name):
    """Load dataset, motif hits, and model."""
    dataset = load_processed(f'data/processed/{dataset_name}_processed.parquet')
    motif_hits = pd.read_parquet(f'data/processed/{dataset_name}_processed_motif_hits.parquet')
    # Load model without probe first, then swap in dataset-specific probe
    model = load_model(model_name, dataset_name='__dummy__')
    swap_probe(model, model_name, dataset_name)
    return dataset, motif_hits, model


def get_enhancer_sample(dataset, motif_hits, n=100, min_motifs=2, seed=42):
    """Get a sample of enhancers with sufficient motifs."""
    eligible = dataset[dataset['n_motifs'] >= min_motifs].copy()
    if len(eligible) > n:
        eligible = eligible.sample(n=n, random_state=seed)
    return eligible


def get_annotation(sequence, seq_id, motif_hits):
    """Build annotation dict for an enhancer."""
    seq_motifs = motif_hits[motif_hits['seq_id'] == str(seq_id)]
    return {
        'sequence': sequence,
        'motifs': seq_motifs.to_dict('records'),
        'motif_count': len(seq_motifs),
        'motif_names': list(seq_motifs['motif_name'].unique()) if len(seq_motifs) > 0 else []
    }


# ============================================================
# P0.3: Power Analysis with 1000 Shuffles
# ============================================================

def run_power_analysis(dataset_name='agarwal', model_name='dnabert2',
                       n_enhancers=100, max_shuffles=1000):
    """
    Run 1000 shuffles per enhancer and compute significance at
    100, 250, 500, 750, 1000 shuffle levels.
    """
    print(f"\n{'='*60}")
    print(f"P0.3: POWER ANALYSIS — {dataset_name} / {model_name}")
    print(f"{'='*60}")

    dataset, motif_hits, model = load_data_and_model(dataset_name, model_name)
    sample = get_enhancer_sample(dataset, motif_hits, n=n_enhancers)

    print(f"  Enhancers: {len(sample)}, Shuffles: {max_shuffles}")

    shuffle_checkpoints = [100, 250, 500, 750, 1000]
    all_results = []

    for idx, (_, row) in enumerate(sample.iterrows()):
        if idx % 10 == 0:
            print(f"  [{idx}/{len(sample)}] Processing enhancer {row['seq_id']}...")

        seq = row['sequence']
        seq_id = str(row['seq_id'])
        annotation = get_annotation(seq, seq_id, motif_hits)

        if annotation['motif_count'] < 2:
            continue

        # Run full 1000 shuffles
        try:
            result = compute_gsi(
                seq, annotation, model,
                n_shuffles=max_shuffles,
                seed=42 + idx
            )
        except Exception as e:
            print(f"  Error for {seq_id}: {e}")
            continue

        shuffle_exprs = np.array(result['shuffle_expressions'])
        original_expr = result['original_expression']

        # Compute metrics at each checkpoint
        enhancer_result = {
            'seq_id': seq_id,
            'original_expression': original_expr,
            'n_motifs': row.get('n_motifs', 0),
        }

        for n_shuf in shuffle_checkpoints:
            if n_shuf > len(shuffle_exprs):
                continue
            subset = shuffle_exprs[:n_shuf]
            sub_mean = np.mean(subset)
            sub_std = np.std(subset)
            sub_median = np.median(subset)
            sub_mad = np.median(np.abs(subset - sub_median))

            # GSI
            gsi = sub_std / max(abs(sub_mean), 1e-10)
            gsi_robust = sub_std / max(abs(sub_mean), sub_std * 0.1, 1e-10)

            # z-score and p-value
            z = abs(original_expr - sub_mean) / max(sub_std, 1e-10)
            p_zscore = 2 * (1 - stats.norm.cdf(z))

            # GES (robust z-score)
            ges = abs(original_expr - sub_median) / max(sub_mad * 1.4826, sub_std, 1e-10)

            # GPE
            gpe = (np.max(subset) - np.min(subset)) / max(abs(sub_median), 1e-10)

            enhancer_result[f'gsi_{n_shuf}'] = float(gsi)
            enhancer_result[f'gsi_robust_{n_shuf}'] = float(gsi_robust)
            enhancer_result[f'z_score_{n_shuf}'] = float(z)
            enhancer_result[f'p_zscore_{n_shuf}'] = float(p_zscore)
            enhancer_result[f'ges_{n_shuf}'] = float(ges)
            enhancer_result[f'gpe_{n_shuf}'] = float(gpe)
            enhancer_result[f'sig_{n_shuf}'] = p_zscore < 0.05

        all_results.append(enhancer_result)

    df = pd.DataFrame(all_results)

    # Summary statistics
    summary = {
        'dataset': dataset_name,
        'model': model_name,
        'n_enhancers': len(df),
        'max_shuffles': max_shuffles,
        'checkpoints': {},
    }

    for n_shuf in shuffle_checkpoints:
        col = f'sig_{n_shuf}'
        if col in df.columns:
            sig_rate = float(df[col].mean())
            median_z = float(df[f'z_score_{n_shuf}'].median())
            median_gsi = float(df[f'gsi_{n_shuf}'].median())
            median_ges = float(df[f'ges_{n_shuf}'].median())
            median_gpe = float(df[f'gpe_{n_shuf}'].median())
            summary['checkpoints'][str(n_shuf)] = {
                'significance_rate': sig_rate,
                'median_z_score': median_z,
                'median_gsi': median_gsi,
                'median_ges': median_ges,
                'median_gpe': median_gpe,
                'n_significant': int(df[col].sum()),
            }
            print(f"  {n_shuf} shuffles: {sig_rate:.1%} significant "
                  f"(median z={median_z:.3f}, median GES={median_ges:.3f})")

    # Save
    outdir = RESULTS_DIR / 'power_analysis'
    df.to_parquet(outdir / f'{dataset_name}_{model_name}_power.parquet')
    save_json(summary, outdir / f'{dataset_name}_{model_name}_power_summary.json')
    print(f"  Saved to {outdir}/")
    return summary


# ============================================================
# P1.1: Factorial Shuffle Decomposition
# ============================================================

def run_factorial_decomposition(dataset_name='agarwal', model_name='dnabert2',
                                 n_enhancers=100, n_shuffles=100):
    """
    Run 4 types of shuffles to decompose grammar sensitivity into:
    1. Position effect (motif order permuted, orientation & spacer fixed)
    2. Orientation effect (motif orientations flipped, position & spacer fixed)
    3. Spacer effect (spacer DNA reshuffled, motifs fixed)
    4. Full shuffle (all factors changed — original GSI approach)
    """
    print(f"\n{'='*60}")
    print(f"P1.1: FACTORIAL DECOMPOSITION — {dataset_name} / {model_name}")
    print(f"{'='*60}")

    dataset, motif_hits, model = load_data_and_model(dataset_name, model_name)
    sample = get_enhancer_sample(dataset, motif_hits, n=n_enhancers)

    print(f"  Enhancers: {len(sample)}, Shuffles per type: {n_shuffles}")

    shuffle_types = ['position', 'orientation', 'spacer', 'full']
    shuffle_funcs = {
        'position': generate_position_only_shuffles,
        'orientation': generate_orientation_only_shuffles,
        'spacer': generate_spacer_only_shuffles,
        'full': generate_vocabulary_preserving_shuffles,
    }

    all_results = []

    for idx, (_, row) in enumerate(sample.iterrows()):
        if idx % 20 == 0:
            print(f"  [{idx}/{len(sample)}] Processing enhancer {row['seq_id']}...")

        seq = row['sequence']
        seq_id = str(row['seq_id'])
        annotation = get_annotation(seq, seq_id, motif_hits)

        if annotation['motif_count'] < 2:
            continue

        # Original expression
        original_expr = float(model.predict_expression([seq])[0])

        enhancer_result = {
            'seq_id': seq_id,
            'original_expression': original_expr,
            'n_motifs': row.get('n_motifs', 0),
        }

        for stype in shuffle_types:
            try:
                shuffles = shuffle_funcs[stype](
                    seq, annotation, n_shuffles=n_shuffles, seed=42 + idx
                )
                exprs = model.predict_expression(shuffles)

                shuf_mean = float(np.mean(exprs))
                shuf_std = float(np.std(exprs))
                shuf_median = float(np.median(exprs))
                shuf_mad = float(np.median(np.abs(exprs - shuf_median)))

                # Effect size: how much does this factor change expression?
                z = abs(original_expr - shuf_mean) / max(shuf_std, 1e-10)
                variance = float(np.var(exprs))

                enhancer_result[f'{stype}_mean'] = shuf_mean
                enhancer_result[f'{stype}_std'] = shuf_std
                enhancer_result[f'{stype}_median'] = shuf_median
                enhancer_result[f'{stype}_mad'] = shuf_mad
                enhancer_result[f'{stype}_variance'] = variance
                enhancer_result[f'{stype}_z_score'] = float(z)
                enhancer_result[f'{stype}_gsi'] = shuf_std / max(abs(shuf_mean), 1e-10)

            except Exception as e:
                print(f"  Error {stype} for {seq_id}: {e}")
                for key in ['mean', 'std', 'median', 'mad', 'variance', 'z_score', 'gsi']:
                    enhancer_result[f'{stype}_{key}'] = np.nan

        all_results.append(enhancer_result)

    df = pd.DataFrame(all_results)

    # Compute variance fractions (relative to full shuffle variance)
    summary = {
        'dataset': dataset_name,
        'model': model_name,
        'n_enhancers': len(df),
        'n_shuffles': n_shuffles,
        'variance_decomposition': {},
        'effect_sizes': {},
    }

    for stype in shuffle_types:
        var_col = f'{stype}_variance'
        z_col = f'{stype}_z_score'
        gsi_col = f'{stype}_gsi'
        if var_col in df.columns:
            median_var = float(df[var_col].median())
            median_z = float(df[z_col].median())
            median_gsi = float(df[gsi_col].median())
            mean_var = float(df[var_col].mean())
            summary['variance_decomposition'][stype] = {
                'median_variance': median_var,
                'mean_variance': mean_var,
                'median_z_score': median_z,
                'median_gsi': median_gsi,
            }
            print(f"  {stype:12s}: median variance={median_var:.6f}, "
                  f"median z={median_z:.3f}, median GSI={median_gsi:.4f}")

    # Fraction of full variance explained by each factor
    full_var = df['full_variance'].values
    for stype in ['position', 'orientation', 'spacer']:
        var_col = f'{stype}_variance'
        if var_col in df.columns:
            ratios = df[var_col].values / np.maximum(full_var, 1e-20)
            summary['effect_sizes'][stype] = {
                'median_fraction_of_full': float(np.median(ratios)),
                'mean_fraction_of_full': float(np.mean(ratios)),
            }
            print(f"  {stype:12s} / full: median={np.median(ratios):.3f}, "
                  f"mean={np.mean(ratios):.3f}")

    # Interaction: full - (position + orientation + spacer)
    pos_var = df.get('position_variance', pd.Series(dtype=float)).fillna(0).values
    orient_var = df.get('orientation_variance', pd.Series(dtype=float)).fillna(0).values
    spacer_var = df.get('spacer_variance', pd.Series(dtype=float)).fillna(0).values
    interaction_var = full_var - pos_var - orient_var - spacer_var
    summary['effect_sizes']['interaction'] = {
        'median_fraction_of_full': float(np.median(interaction_var / np.maximum(full_var, 1e-20))),
        'mean_fraction_of_full': float(np.mean(interaction_var / np.maximum(full_var, 1e-20))),
    }

    # Save
    outdir = RESULTS_DIR / 'factorial_decomposition'
    df.to_parquet(outdir / f'{dataset_name}_{model_name}_factorial.parquet')
    save_json(summary, outdir / f'{dataset_name}_{model_name}_factorial_summary.json')
    print(f"  Saved to {outdir}/")
    return summary


# ============================================================
# P1.2: Bag-of-Motifs Baseline
# ============================================================

def run_bom_baseline(dataset_name='agarwal', n_enhancers=500, seed=42):
    """
    Compare expression prediction R²:
    1. Bag-of-Motifs (BOM): [count(motif_i) for each motif type]
    2. Grammar features: BOM + pairwise spacing/orientation stats
    3. Upper bound: DNABERT-2 probe R² (from probes)
    """
    print(f"\n{'='*60}")
    print(f"P1.2: BAG-OF-MOTIFS BASELINE — {dataset_name}")
    print(f"{'='*60}")

    dataset = load_processed(f'data/processed/{dataset_name}_processed.parquet')
    motif_hits = pd.read_parquet(f'data/processed/{dataset_name}_processed_motif_hits.parquet')

    # Filter to enhancers with motifs and expression data
    eligible = dataset[dataset['n_motifs'] >= 2].copy()
    if 'expression' not in eligible.columns:
        print(f"  No expression column in {dataset_name}, skipping")
        return None
    eligible = eligible.dropna(subset=['expression'])
    if len(eligible) > n_enhancers:
        eligible = eligible.sample(n=n_enhancers, random_state=seed)

    print(f"  Enhancers with expression: {len(eligible)}")

    # Build BOM feature matrix
    all_motif_names = sorted(motif_hits['motif_name'].unique())
    print(f"  Unique motif types: {len(all_motif_names)}")

    bom_features = []
    grammar_features = []
    expressions = []

    for _, row in eligible.iterrows():
        seq_id = str(row['seq_id'])
        seq_motifs = motif_hits[motif_hits['seq_id'] == seq_id]

        # BOM: motif counts
        counts = {name: 0 for name in all_motif_names}
        for _, m in seq_motifs.iterrows():
            name = m['motif_name']
            if name in counts:
                counts[name] += 1

        bom_row = [counts[name] for name in all_motif_names]
        bom_features.append(bom_row)

        # Grammar features: BOM + pairwise stats
        grammar_row = list(bom_row)  # start with BOM

        # Add simple grammar features
        if len(seq_motifs) >= 2:
            positions = seq_motifs['start'].values
            # Mean/std of inter-motif distances
            sorted_pos = np.sort(positions)
            dists = np.diff(sorted_pos)
            grammar_row.extend([
                float(np.mean(dists)) if len(dists) > 0 else 0,
                float(np.std(dists)) if len(dists) > 0 else 0,
                float(np.min(dists)) if len(dists) > 0 else 0,
                float(np.max(dists)) if len(dists) > 0 else 0,
                float(len(seq_motifs)),  # total motif count
                float(len(seq_motifs) / max(len(row['sequence']), 1)),  # density
            ])
            # Strand balance
            if 'strand' in seq_motifs.columns:
                plus_frac = (seq_motifs['strand'] == '+').mean()
                grammar_row.append(float(plus_frac))
            else:
                grammar_row.append(0.5)
        else:
            grammar_row.extend([0, 0, 0, 0, 0, 0, 0.5])

        grammar_features.append(grammar_row)
        expressions.append(row['expression'])

    X_bom = np.array(bom_features)
    X_grammar = np.array(grammar_features)
    y = np.array(expressions)

    if len(bom_features) < 20:
        print(f"  Too few enhancers ({len(bom_features)}), skipping")
        return None

    print(f"  BOM features: {X_bom.shape[1]}, Grammar features: {X_grammar.shape[1]}")

    # Cross-validated R²
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    # BOM only (Ridge regression — many motif features may be sparse)
    model_bom = Ridge(alpha=1.0)
    scores_bom = cross_val_score(model_bom, X_bom, y, cv=cv, scoring='r2')

    # Grammar features (BOM + spacing/orientation stats)
    model_grammar = Ridge(alpha=1.0)
    scores_grammar = cross_val_score(model_grammar, X_grammar, y, cv=cv, scoring='r2')

    # Random forest versions
    model_bom_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed)
    scores_bom_rf = cross_val_score(model_bom_rf, X_bom, y, cv=cv, scoring='r2')

    model_grammar_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=seed)
    scores_grammar_rf = cross_val_score(model_grammar_rf, X_grammar, y, cv=cv, scoring='r2')

    summary = {
        'dataset': dataset_name,
        'n_enhancers': len(eligible),
        'n_motif_types': len(all_motif_names),
        'bom_ridge': {
            'mean_r2': float(np.mean(scores_bom)),
            'std_r2': float(np.std(scores_bom)),
            'fold_r2': scores_bom.tolist(),
        },
        'grammar_ridge': {
            'mean_r2': float(np.mean(scores_grammar)),
            'std_r2': float(np.std(scores_grammar)),
            'fold_r2': scores_grammar.tolist(),
        },
        'bom_rf': {
            'mean_r2': float(np.mean(scores_bom_rf)),
            'std_r2': float(np.std(scores_bom_rf)),
            'fold_r2': scores_bom_rf.tolist(),
        },
        'grammar_rf': {
            'mean_r2': float(np.mean(scores_grammar_rf)),
            'std_r2': float(np.std(scores_grammar_rf)),
            'fold_r2': scores_grammar_rf.tolist(),
        },
        'grammar_increment_ridge': float(np.mean(scores_grammar) - np.mean(scores_bom)),
        'grammar_increment_rf': float(np.mean(scores_grammar_rf) - np.mean(scores_bom_rf)),
    }

    print(f"\n  Results:")
    print(f"\n  Results:")
    print(f"  BOM (Ridge):     R² = {np.mean(scores_bom):.4f} ± {np.std(scores_bom):.4f}")
    print(f"  Grammar (Ridge): R² = {np.mean(scores_grammar):.4f} ± {np.std(scores_grammar):.4f}")
    print(f"  BOM (RF):        R² = {np.mean(scores_bom_rf):.4f} ± {np.std(scores_bom_rf):.4f}")
    print(f"  Grammar (RF):    R² = {np.mean(scores_grammar_rf):.4f} ± {np.std(scores_grammar_rf):.4f}")
    print(f"  Grammar increment (Ridge): {summary['grammar_increment_ridge']:+.4f}")
    print(f"  Grammar increment (RF):    {summary['grammar_increment_rf']:+.4f}")

    outdir = RESULTS_DIR / 'bom_baseline'
    save_json(summary, outdir / f'{dataset_name}_bom_summary.json')
    print(f"  Saved to {outdir}/")
    return summary


# ============================================================
# P1.3: Unexplained Variance Decomposition
# ============================================================

def run_variance_decomposition(dataset_name='agarwal', model_name='dnabert2',
                                n_enhancers=500, seed=42):
    """
    Decompose expression variance using DL embeddings vs hand-crafted features.

    Compare R² from:
    1. Vocabulary features (motif counts)
    2. Grammar features (motif counts + spacing/orientation)
    3. DL embeddings (frozen model representations)
    4. Probe R² (upper bound from expression probe)

    The gap between (2) and (3) = "grammar info the DL captures but we don't"
    """
    print(f"\n{'='*60}")
    print(f"P1.3: VARIANCE DECOMPOSITION — {dataset_name} / {model_name}")
    print(f"{'='*60}")

    dataset = load_processed(f'data/processed/{dataset_name}_processed.parquet')
    motif_hits = pd.read_parquet(f'data/processed/{dataset_name}_processed_motif_hits.parquet')

    eligible = dataset[dataset['n_motifs'] >= 2].copy()
    if 'expression' not in eligible.columns:
        print(f"  No expression column in {dataset_name}, skipping")
        return None
    eligible = eligible.dropna(subset=['expression'])
    if len(eligible) > n_enhancers:
        eligible = eligible.sample(n=n_enhancers, random_state=seed)

    print(f"  Loading model {model_name} for embedding extraction...")
    model = load_model(model_name)

    # Build features
    all_motif_names = sorted(motif_hits['motif_name'].unique())
    sequences = []
    vocab_features = []
    grammar_features = []
    expressions = []

    for _, row in eligible.iterrows():
        seq_id = str(row['seq_id'])
        seq_motifs = motif_hits[motif_hits['seq_id'] == seq_id]
        sequences.append(row['sequence'])

        # Vocabulary features
        counts = {name: 0 for name in all_motif_names}
        for _, m in seq_motifs.iterrows():
            if m['motif_name'] in counts:
                counts[m['motif_name']] += 1
        vocab_features.append([counts[n] for n in all_motif_names])

        # Grammar features (vocab + arrangement stats)
        gram_row = [counts[n] for n in all_motif_names]
        if len(seq_motifs) >= 2:
            positions = np.sort(seq_motifs['start'].values)
            dists = np.diff(positions)
            gram_row.extend([
                float(np.mean(dists)) if len(dists) > 0 else 0,
                float(np.std(dists)) if len(dists) > 0 else 0,
                float(np.min(dists)) if len(dists) > 0 else 0,
                float(np.max(dists)) if len(dists) > 0 else 0,
                float(len(seq_motifs)),
                float(len(seq_motifs) / max(len(row['sequence']), 1)),
            ])
            if 'strand' in seq_motifs.columns:
                gram_row.append(float((seq_motifs['strand'] == '+').mean()))
            else:
                gram_row.append(0.5)
        else:
            gram_row.extend([0, 0, 0, 0, 0, 0, 0.5])
        grammar_features.append(gram_row)
        expressions.append(row['expression'])

    X_vocab = np.array(vocab_features)
    X_grammar = np.array(grammar_features)
    y = np.array(expressions)

    # Extract DL embeddings
    print(f"  Extracting embeddings for {len(sequences)} sequences...")
    try:
        embeddings = model.get_embeddings(sequences)
        if isinstance(embeddings, dict):
            # Use last layer
            layer_keys = sorted(embeddings.keys())
            X_embed = embeddings[layer_keys[-1]]
        else:
            X_embed = embeddings
        if hasattr(X_embed, 'numpy'):
            X_embed = X_embed.numpy()
        # Mean-pool if 3D (batch, seq_len, hidden)
        if X_embed.ndim == 3:
            X_embed = np.mean(X_embed, axis=1)
        print(f"  Embedding shape: {X_embed.shape}")
    except Exception as e:
        print(f"  Could not extract embeddings: {e}")
        print(f"  Falling back to probe predictions as embedding proxy...")
        # Use model predictions as a 1D embedding
        preds = model.predict_expression(sequences)
        X_embed = preds.reshape(-1, 1)

    # Cross-validated R²
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    scores_vocab = cross_val_score(Ridge(alpha=1.0), X_vocab, y, cv=cv, scoring='r2')
    scores_grammar = cross_val_score(Ridge(alpha=1.0), X_grammar, y, cv=cv, scoring='r2')
    scores_embed = cross_val_score(Ridge(alpha=1.0), X_embed, y, cv=cv, scoring='r2')

    # Combined: grammar + embeddings
    X_combined = np.hstack([X_grammar, X_embed])
    scores_combined = cross_val_score(Ridge(alpha=1.0), X_combined, y, cv=cv, scoring='r2')

    summary = {
        'dataset': dataset_name,
        'model': model_name,
        'n_enhancers': len(eligible),
        'n_vocab_features': X_vocab.shape[1],
        'n_grammar_features': X_grammar.shape[1],
        'n_embed_features': X_embed.shape[1],
        'vocab_r2': {'mean': float(np.mean(scores_vocab)), 'std': float(np.std(scores_vocab))},
        'grammar_r2': {'mean': float(np.mean(scores_grammar)), 'std': float(np.std(scores_grammar))},
        'embedding_r2': {'mean': float(np.mean(scores_embed)), 'std': float(np.std(scores_embed))},
        'combined_r2': {'mean': float(np.mean(scores_combined)), 'std': float(np.std(scores_combined))},
        'grammar_increment': float(np.mean(scores_grammar) - np.mean(scores_vocab)),
        'embedding_increment_over_grammar': float(np.mean(scores_embed) - np.mean(scores_grammar)),
        'unexplained_by_grammar': float(1 - np.mean(scores_grammar)),
        'unexplained_by_embedding': float(1 - np.mean(scores_embed)),
        'grammar_captures_fraction_of_embedding': float(
            np.mean(scores_grammar) / max(np.mean(scores_embed), 1e-10)
        ),
    }

    print(f"\n  Results:")
    print(f"  Vocabulary:   R² = {np.mean(scores_vocab):.4f} ± {np.std(scores_vocab):.4f}")
    print(f"  Grammar:      R² = {np.mean(scores_grammar):.4f} ± {np.std(scores_grammar):.4f}")
    print(f"  Embeddings:   R² = {np.mean(scores_embed):.4f} ± {np.std(scores_embed):.4f}")
    print(f"  Combined:     R² = {np.mean(scores_combined):.4f} ± {np.std(scores_combined):.4f}")
    print(f"  Grammar increment over vocab: {summary['grammar_increment']:+.4f}")
    print(f"  Embedding increment over grammar: {summary['embedding_increment_over_grammar']:+.4f}")
    print(f"  Grammar captures {summary['grammar_captures_fraction_of_embedding']:.1%} of embedding signal")

    outdir = RESULTS_DIR / 'variance_decomposition'
    save_json(summary, outdir / f'{dataset_name}_{model_name}_variance_summary.json')
    print(f"  Saved to {outdir}/")
    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='GRAMLANG v3 Extension Analysis')
    parser.add_argument('--phase', type=str, default='all',
                        choices=['P0.3', 'P1.1', 'P1.2', 'P1.3', 'all'],
                        help='Which analysis phase to run')
    parser.add_argument('--datasets', type=str, default='agarwal,jores,de_almeida',
                        help='Comma-separated dataset names')
    parser.add_argument('--models', type=str, default='dnabert2,nt,hyenadna',
                        help='Comma-separated model names')
    parser.add_argument('--n-enhancers', type=int, default=100,
                        help='Number of enhancers per analysis')
    parser.add_argument('--n-shuffles', type=int, default=100,
                        help='Number of shuffles (P1.1)')
    parser.add_argument('--max-shuffles', type=int, default=1000,
                        help='Max shuffles for power analysis (P0.3)')
    args = parser.parse_args()

    ensure_dirs()

    datasets = args.datasets.split(',')
    models = args.models.split(',')
    phases = ['P0.3', 'P1.1', 'P1.2', 'P1.3'] if args.phase == 'all' else [args.phase]

    start_time = time.time()
    all_summaries = {}

    for phase in phases:
        print(f"\n{'#'*60}")
        print(f"# PHASE {phase}")
        print(f"{'#'*60}")

        if phase == 'P0.3':
            # Power analysis: run on first dataset + first model (most important)
            for ds in datasets[:2]:  # Do 2 datasets
                for mdl in models[:1]:  # Primary model only for speed
                    key = f'power_{ds}_{mdl}'
                    all_summaries[key] = run_power_analysis(
                        ds, mdl, n_enhancers=args.n_enhancers,
                        max_shuffles=args.max_shuffles
                    )

        elif phase == 'P1.1':
            # Factorial decomposition: run on all datasets × primary model
            for ds in datasets:
                for mdl in models[:1]:  # Primary model for speed
                    key = f'factorial_{ds}_{mdl}'
                    all_summaries[key] = run_factorial_decomposition(
                        ds, mdl, n_enhancers=args.n_enhancers,
                        n_shuffles=args.n_shuffles
                    )

        elif phase == 'P1.2':
            # BOM baseline: all datasets (no model needed)
            for ds in datasets:
                key = f'bom_{ds}'
                all_summaries[key] = run_bom_baseline(
                    ds, n_enhancers=500
                )

        elif phase == 'P1.3':
            # Variance decomposition: all datasets × primary model
            for ds in datasets:
                for mdl in models[:1]:
                    key = f'variance_{ds}_{mdl}'
                    all_summaries[key] = run_variance_decomposition(
                        ds, mdl, n_enhancers=500
                    )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"ALL PHASES COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    # Save master summary
    save_json(all_summaries, RESULTS_DIR / 'v3_analysis_summary.json')
    print(f"Master summary saved to {RESULTS_DIR / 'v3_analysis_summary.json'}")


if __name__ == '__main__':
    main()
