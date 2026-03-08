#!/usr/bin/env python
"""
Train species-specific expression probes for ALL (model, dataset) pairs.

Critical fix: Previously, probes were trained only on Vaishnav 2022 (yeast)
and applied to human/plant datasets. This trains dataset-specific probes
so each model's expression predictions are calibrated to the target dataset.
"""

import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model
from src.models.expression_probes import (
    ExpressionProbe, train_expression_probe, save_probe, load_probe
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_DIR, 'data', 'embeddings_cache')
PROBES_DIR = os.path.join(PROJECT_DIR, 'data', 'probes')
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PROBES_DIR, exist_ok=True)

# Dataset name mapping: pipeline name -> raw parquet name
DATASET_MAP = {
    'vaishnav': 'vaishnav2022',
    'klein': 'klein2020',
    'agarwal': 'agarwal2023',
    'inoue': 'inoue2024',
    'jores': 'jores2021',
}

MODELS = ['dnabert2', 'nt', 'hyenadna']
MAX_SEQUENCES = 10000


def find_dataset_file(ds_name):
    """Find the parquet file for a dataset."""
    # Try processed file first (has motif annotations), then raw
    for suffix in ['_processed.parquet', '.parquet']:
        for prefix in [ds_name, DATASET_MAP.get(ds_name, ds_name)]:
            path = os.path.join(PROCESSED_DIR, f'{prefix}{suffix}')
            if os.path.exists(path):
                return path
    return None


def train_probe_for_pair(model_name, ds_name, device='cuda'):
    """Train expression probe for one (model, dataset) pair."""
    print(f"\n{'='*60}")
    print(f"Training probe: {model_name} on {ds_name}")
    print(f"{'='*60}")

    # Check if probe already exists
    probe_name = f'{model_name}_{ds_name}'
    probe_path = os.path.join(PROBES_DIR, f'{probe_name}_probe.pt')
    metrics_path = os.path.join(PROBES_DIR, f'{probe_name}_probe_metrics.json')
    if os.path.exists(probe_path) and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            existing = json.load(f)
        if existing.get('viable', False):
            print(f"  Probe already exists and is viable (r={existing['pearson_r']:.3f}), skipping")
            return existing

    # Find and load dataset
    data_path = find_dataset_file(ds_name)
    if data_path is None:
        # Try with the mapped name
        raw_name = DATASET_MAP.get(ds_name, ds_name)
        data_path = find_dataset_file(raw_name)

    if data_path is None:
        print(f"  Dataset {ds_name} not found, skipping")
        return None

    print(f"  Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    # Ensure required columns
    if 'sequence' not in df.columns or 'expression' not in df.columns:
        print(f"  Missing required columns (sequence, expression), skipping")
        print(f"  Available columns: {list(df.columns)}")
        return None

    # Subsample
    if len(df) > MAX_SEQUENCES:
        print(f"  Subsampling {len(df)} -> {MAX_SEQUENCES}")
        df = df.sample(MAX_SEQUENCES, random_state=42).reset_index(drop=True)

    sequences = df['sequence'].tolist()
    expressions = df['expression'].values.astype(np.float32)

    print(f"  Dataset: {len(sequences)} sequences")
    print(f"  Expression range: [{expressions.min():.3f}, {expressions.max():.3f}]")
    print(f"  Expression mean: {expressions.mean():.3f}, std: {expressions.std():.3f}")

    # Check for cached embeddings
    cache_file = os.path.join(CACHE_DIR, f'{model_name}_{ds_name}_embeddings.npz')

    if os.path.exists(cache_file):
        print(f"  Loading cached embeddings")
        data = np.load(cache_file)
        embeddings = data['embeddings']
    else:
        # Load model (without probe - we just need embeddings)
        print(f"  Loading model {model_name} for embedding extraction...")
        model = load_model(model_name, device=device)
        print(f"  Extracting embeddings...")
        embeddings = model.get_embeddings(sequences)
        print(f"  Embeddings shape: {embeddings.shape}")

        # Cache
        np.savez_compressed(cache_file, embeddings=embeddings, expressions=expressions)
        print(f"  Cached to {cache_file}")

        model.unload()

    # Train probe
    print(f"  Training expression probe (input_dim={embeddings.shape[1]})...")
    probe, metrics = train_expression_probe(
        embeddings, expressions,
        input_dim=embeddings.shape[1],
        hidden_dim=256,
        lr=1e-3,
        max_epochs=100,
        batch_size=256,
        patience=10,
        device=device,
    )

    result = {
        'model': model_name,
        'dataset': ds_name,
        'n_sequences': len(sequences),
        'embedding_dim': int(embeddings.shape[1]),
        'pearson_r': metrics['pearson_r'],
        'spearman_rho': metrics['spearman_rho'],
        'r_squared': metrics['r_squared'],
        'viable': metrics['pearson_r'] > 0.3,
        'best_epoch': metrics['best_epoch'],
        'date': datetime.now().isoformat(),
    }

    print(f"\n  Results:")
    print(f"    Pearson r:    {metrics['pearson_r']:.4f}")
    print(f"    Spearman rho: {metrics['spearman_rho']:.4f}")
    print(f"    R²:           {metrics['r_squared']:.4f}")
    print(f"    Viable:       {result['viable']}")

    # Save probe regardless of viability (even weak probes are better than cross-species)
    save_probe(probe, metrics, PROBES_DIR, probe_name)
    print(f"  Probe saved as {probe_name}")

    return result


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Models: {MODELS}")
    print(f"Datasets: {list(DATASET_MAP.keys())}")

    all_results = []

    for model_name in MODELS:
        for ds_name in DATASET_MAP.keys():
            try:
                result = train_probe_for_pair(model_name, ds_name, device=device)
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    'model': model_name,
                    'dataset': ds_name,
                    'error': str(e),
                })

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    print(f"\n\n{'='*60}")
    print("PROBE TRAINING SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        if 'error' in r:
            print(f"  {r['model']:12s} on {r['dataset']:12s}: ERROR - {r['error'][:60]}")
        else:
            status = "VIABLE" if r['viable'] else "WEAK"
            print(f"  {r['model']:12s} on {r['dataset']:12s}: "
                  f"r={r['pearson_r']:.3f}, R²={r['r_squared']:.3f} [{status}]")

    # Save summary
    results_path = os.path.join(PROJECT_DIR, 'results', 'probe_training_all_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
