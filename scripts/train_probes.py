#!/usr/bin/env python
"""
Train expression probes for foundation models on MPRA data.

For each (model, dataset) pair:
1. Extract embeddings (cached to disk)
2. Train 2-layer MLP probe
3. Evaluate with Pearson r, Spearman rho, R^2
4. Save probe if viable (r > 0.3)
"""

import os
import sys
import json
import argparse
import gc
import numpy as np
import pandas as pd
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model
from src.models.expression_probes import (
    ExpressionProbe, train_expression_probe,
    extract_and_cache_embeddings, save_probe, load_probe
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'embeddings_cache')
PROBES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'probes')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data', 'processed')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PROBES_DIR, exist_ok=True)


def train_probe_for_model(model_name, dataset_name, max_sequences=10000, device='cuda'):
    """Train expression probe for a single model on a single dataset."""
    print(f"\n{'='*60}")
    print(f"Training probe: {model_name} on {dataset_name}")
    print(f"{'='*60}")

    # Load processed data
    data_path = os.path.join(PROCESSED_DIR, f'{dataset_name}.parquet')
    if not os.path.exists(data_path):
        data_path = os.path.join(PROCESSED_DIR, f'{dataset_name}.csv.gz')
    if not os.path.exists(data_path):
        print(f"  Dataset {dataset_name} not found, skipping")
        return None

    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # Subsample if needed
    if len(df) > max_sequences:
        print(f"  Subsampling {len(df)} -> {max_sequences} sequences")
        df = df.sample(max_sequences, random_state=42).reset_index(drop=True)

    sequences = df['sequence'].tolist()
    expressions = df['expression'].values

    print(f"  Dataset: {len(sequences)} sequences")
    print(f"  Expression range: [{expressions.min():.3f}, {expressions.max():.3f}]")

    # Check for cached embeddings
    cache_file = os.path.join(CACHE_DIR, f'{model_name}_{dataset_name}_embeddings.npz')

    if os.path.exists(cache_file):
        print(f"  Loading cached embeddings from {cache_file}")
        data = np.load(cache_file)
        embeddings = data['embeddings']
    else:
        # Load model and extract embeddings
        print(f"  Loading model {model_name}...")
        model = load_model(model_name, device=device)

        print(f"  Extracting embeddings (batch processing)...")
        batch_size = 32 if model_name in ['dnabert2', 'hyenadna', 'caduceus'] else 16
        if model_name == 'enformer':
            batch_size = 2

        embeddings = model.get_embeddings(sequences)
        print(f"  Embeddings shape: {embeddings.shape}")

        # Cache embeddings
        np.savez_compressed(cache_file, embeddings=embeddings, expressions=expressions)
        print(f"  Cached embeddings to {cache_file}")

        # Unload model to free GPU memory
        model.unload()

    # Train probe
    print(f"  Training expression probe...")
    input_dim = embeddings.shape[1]

    probe, metrics = train_expression_probe(
        embeddings, expressions,
        input_dim=input_dim,
        hidden_dim=256,
        lr=1e-3,
        max_epochs=100,
        batch_size=256,
        patience=10,
    )

    # Results
    result = {
        'model': model_name,
        'dataset': dataset_name,
        'n_sequences': len(sequences),
        'embedding_dim': embeddings.shape[1],
        'pearson_r': metrics['pearson_r'],
        'spearman_rho': metrics['spearman_rho'],
        'r_squared': metrics['r_squared'],
        'viable': metrics['pearson_r'] > 0.3,
        'date': datetime.now().isoformat(),
    }

    print(f"\n  Results:")
    print(f"    Pearson r:    {metrics['pearson_r']:.4f}")
    print(f"    Spearman rho: {metrics['spearman_rho']:.4f}")
    print(f"    R^2:          {metrics['r_squared']:.4f}")
    print(f"    Viable:       {result['viable']}")

    # Save probe if viable
    if result['viable']:
        save_probe(probe, metrics, PROBES_DIR, f'{model_name}_{dataset_name}')
        result['probe_path'] = os.path.join(PROBES_DIR, f'{model_name}_{dataset_name}_probe.pt')
        print(f"  Probe saved to {PROBES_DIR}")
    else:
        print(f"  Probe NOT saved (r < 0.3)")

    return result


def main():
    parser = argparse.ArgumentParser(description='Train expression probes')
    parser.add_argument('--models', nargs='+',
                        default=['dnabert2', 'nt', 'hyenadna'],
                        help='Models to train probes for')
    parser.add_argument('--datasets', nargs='+',
                        default=['vaishnav2022', 'klein2020'],
                        help='Datasets to train on')
    parser.add_argument('--max-sequences', type=int, default=10000,
                        help='Max sequences per dataset')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    all_results = []

    for model_name in args.models:
        for dataset_name in args.datasets:
            try:
                result = train_probe_for_model(
                    model_name, dataset_name,
                    max_sequences=args.max_sequences,
                    device=args.device,
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'error': str(e),
                })

            # Clean up GPU memory between runs
            gc.collect()
            torch.cuda.empty_cache()

    # Save all results
    results_path = os.path.join(RESULTS_DIR, 'probe_training_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n\n{'='*60}")
    print("PROBE TRAINING SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        if 'error' in r:
            print(f"  {r['model']} on {r['dataset']}: ERROR - {r['error']}")
        else:
            status = "VIABLE" if r['viable'] else "NOT VIABLE"
            print(f"  {r['model']} on {r['dataset']}: r={r['pearson_r']:.3f}, "
                  f"rho={r['spearman_rho']:.3f}, R^2={r['r_squared']:.3f} [{status}]")

    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
