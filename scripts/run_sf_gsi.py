#!/usr/bin/env python3
"""
Compute Spacer-Factored Grammar Sensitivity Index (SF-GSI) on datasets.

Usage:
    python scripts/run_sf_gsi.py --dataset agarwal --model-path results/sfgn/agarwal_sfgn_best.pt
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sfgn import SFGN, SFGNConfig
from src.grammar.sf_gsi import compute_sf_gsi, compute_sf_gsi_with_sfgn


def load_dataset(dataset_name: str, data_dir: str = 'data/processed'):
    """Load dataset and motif annotations."""
    data_path = Path(data_dir) / f'{dataset_name}_processed.parquet'
    motif_path = Path(data_dir) / f'{dataset_name}_processed_motif_hits.parquet'

    df = pd.read_parquet(data_path)
    motif_df = pd.read_parquet(motif_path)

    motif_groups = motif_df.groupby('seq_id')

    motif_annotations = []
    for seq_id in df['seq_id']:
        if seq_id in motif_groups.groups:
            group = motif_groups.get_group(seq_id)
            motifs = []
            for _, row in group.iterrows():
                motifs.append({
                    'start': int(row['start']),
                    'end': int(row['end']),
                    'motif_name': row.get('motif_name', 'unknown'),
                    'strand': row.get('strand', '+'),
                })
            motif_annotations.append(motifs)
        else:
            motif_annotations.append([])

    return df, motif_annotations


def main():
    parser = argparse.ArgumentParser(description='Compute SF-GSI')
    parser.add_argument('--dataset', type=str, default='agarwal', help='Dataset name')
    parser.add_argument('--model-path', type=str, default=None, help='Path to SFGN model')
    parser.add_argument('--n-shuffles', type=int, default=50, help='Shuffles per sequence')
    parser.add_argument('--max-samples', type=int, default=200, help='Max sequences to analyze')
    parser.add_argument('--output-dir', type=str, default='results/sf_gsi', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading {args.dataset}...")
    df, motif_annotations = load_dataset(args.dataset)

    # Filter sequences with motifs
    valid_indices = [i for i, m in enumerate(motif_annotations) if len(m) >= 2]
    print(f"  {len(valid_indices)} sequences with >= 2 motifs")

    # Subsample
    if len(valid_indices) > args.max_samples:
        valid_indices = np.random.choice(valid_indices, args.max_samples, replace=False)
    print(f"  Analyzing {len(valid_indices)} sequences")

    sequences = [df.iloc[i]['sequence'] for i in valid_indices]
    motifs = [motif_annotations[i] for i in valid_indices]

    # Load or create model
    if args.model_path and Path(args.model_path).exists():
        print(f"Loading model from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location=device)
        config = checkpoint.get('config', SFGNConfig())
        model = SFGN(config, device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        use_sfgn = True
        print("  Using SFGN model for predictions")
    else:
        print("No SFGN model found, using foundation model directly...")
        # Use existing model loader
        from src.models.model_loader import load_model
        foundation = load_model('dnabert2', dataset_name=args.dataset)
        use_sfgn = False

        def predict_fn(seqs, mots):
            return foundation.predict_expression(seqs)

    results = {}

    if use_sfgn:
        # Method 1: SFGN built-in decomposition
        print("\nComputing SF-GSI with SFGN decomposition...")
        sfgn_result = compute_sf_gsi_with_sfgn(
            model, sequences, motifs, n_shuffles=args.n_shuffles
        )
        results['sfgn_decomposition'] = sfgn_result
        print(f"  GSI: {sfgn_result['gsi']:.4f}")
        print(f"  SF-GSI: {sfgn_result['sf_gsi']:.4f}")
        print(f"  Mean α: {sfgn_result['mean_alpha']:.4f}")
        print(f"  Spacer contribution: {sfgn_result['spacer_contribution']:.4f}")

        # Define predict function for SFGN
        def predict_fn(seqs, mots):
            model.eval()
            with torch.no_grad():
                output = model(seqs, mots)
                return output.prediction

    # Method 2: Motif-only shuffles
    print("\nComputing SF-GSI with motif-only method...")
    motif_only_result = compute_sf_gsi(
        sequences, motifs, predict_fn,
        n_shuffles=args.n_shuffles,
        method='motif_only',
        device=device
    )
    results['motif_only'] = {
        'gsi': motif_only_result.gsi,
        'sf_gsi': motif_only_result.sf_gsi,
        'spacer_contribution': motif_only_result.spacer_contribution,
        'grammar_pvalue': motif_only_result.grammar_pvalue,
        'total_variance': motif_only_result.total_variance,
        'grammar_variance': motif_only_result.grammar_variance,
    }
    print(f"  GSI: {motif_only_result.gsi:.4f}")
    print(f"  SF-GSI: {motif_only_result.sf_gsi:.4f}")
    print(f"  Spacer contribution: {motif_only_result.spacer_contribution:.2%}")

    # Method 3: Matched shuffles
    print("\nComputing SF-GSI with matched method...")
    matched_result = compute_sf_gsi(
        sequences, motifs, predict_fn,
        n_shuffles=args.n_shuffles,
        method='matched',
        device=device
    )
    results['matched'] = {
        'gsi': matched_result.gsi,
        'sf_gsi': matched_result.sf_gsi,
        'spacer_contribution': matched_result.spacer_contribution,
        'grammar_pvalue': matched_result.grammar_pvalue,
        'total_variance': matched_result.total_variance,
        'grammar_variance': matched_result.grammar_variance,
    }
    print(f"  GSI: {matched_result.gsi:.4f}")
    print(f"  SF-GSI: {matched_result.sf_gsi:.4f}")
    print(f"  Spacer contribution: {matched_result.spacer_contribution:.2%}")

    # Method 4: Regression
    print("\nComputing SF-GSI with regression method...")
    try:
        regression_result = compute_sf_gsi(
            sequences, motifs, predict_fn,
            n_shuffles=args.n_shuffles,
            method='regression',
            device=device
        )
        results['regression'] = {
            'gsi': regression_result.gsi,
            'sf_gsi': regression_result.sf_gsi,
            'spacer_contribution': regression_result.spacer_contribution,
            'grammar_pvalue': regression_result.grammar_pvalue,
            'total_variance': regression_result.total_variance,
            'grammar_variance': regression_result.grammar_variance,
        }
        print(f"  GSI: {regression_result.gsi:.4f}")
        print(f"  SF-GSI: {regression_result.sf_gsi:.4f}")
        print(f"  Spacer contribution: {regression_result.spacer_contribution:.2%}")
    except Exception as e:
        print(f"  Regression method failed: {e}")
        results['regression'] = {'error': str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Sequences analyzed: {len(sequences)}")
    print(f"Shuffles per sequence: {args.n_shuffles}")
    print()

    print("Method Comparison:")
    print("-" * 40)
    for method, res in results.items():
        if 'error' not in res:
            gsi = res.get('gsi', 0)
            sf_gsi = res.get('sf_gsi', 0)
            spacer = res.get('spacer_contribution', 0)
            print(f"  {method:20s}: GSI={gsi:.4f}, SF-GSI={sf_gsi:.4f}, Spacer={spacer:.2%}")

    # Save results (convert numpy types to Python floats)
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output_path = output_dir / f'{args.dataset}_sf_gsi.json'
    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable({
            'dataset': args.dataset,
            'n_sequences': len(sequences),
            'n_shuffles': args.n_shuffles,
            'results': results,
        }), f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
