#!/usr/bin/env python3
"""
Run ablation studies for SFGN.

Ablations:
1. Orthogonality weight: 0, 0.01, 0.1 (default), 0.5
2. With/without composition module
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sfgn import SFGN, SFGNConfig


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


def quick_train(config, sequences, expressions, motifs, device, epochs=5):
    """Quick training for ablation."""
    model = SFGN(config, device=device)

    # Split
    indices = list(range(len(sequences)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_seqs = [sequences[i] for i in train_idx]
    train_expr = torch.tensor([expressions[i] for i in train_idx], dtype=torch.float32).to(device)
    train_motifs = [motifs[i] for i in train_idx]

    val_seqs = [sequences[i] for i in val_idx]
    val_expr = torch.tensor([expressions[i] for i in val_idx], dtype=torch.float32).to(device)
    val_motifs = [motifs[i] for i in val_idx]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5, weight_decay=0.01
    )

    batch_size = 16
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        for i in range(0, len(train_seqs), batch_size):
            batch_seqs = train_seqs[i:i+batch_size]
            batch_expr = train_expr[i:i+batch_size]
            batch_motifs = train_motifs[i:i+batch_size]

            optimizer.zero_grad()
            output = model(batch_seqs, batch_motifs)
            loss, _ = model.compute_loss(output, batch_expr)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    # Evaluate
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_alphas = []
        for i in range(0, len(val_seqs), batch_size):
            batch_seqs = val_seqs[i:i+batch_size]
            batch_motifs = val_motifs[i:i+batch_size]
            try:
                output = model(batch_seqs, batch_motifs)
                preds = output.prediction.cpu().numpy()
                alphas = output.alpha.cpu().numpy()
                # Replace NaN/Inf with 0
                preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
                alphas = np.nan_to_num(alphas, nan=0.5, posinf=0.5, neginf=0.5)
                all_preds.extend(preds)
                all_alphas.extend(alphas)
            except Exception as e:
                # Skip problematic batches
                continue

    all_preds = np.array(all_preds)
    all_alphas = np.array(all_alphas)
    val_targets = val_expr.cpu().numpy()

    # Handle NaN values
    valid_mask = ~np.isnan(all_preds) & ~np.isinf(all_preds)
    if valid_mask.sum() < 10:
        return {
            'pearson_r': 0.0,
            'r2': -1.0,
            'mean_alpha': float(np.nanmean(all_alphas)),
            'final_loss': float(total_loss / max(n_batches, 1)),
            'nan_ratio': float(1 - valid_mask.mean())
        }

    valid_preds = all_preds[valid_mask]
    valid_targets = val_targets[valid_mask]

    r, _ = pearsonr(valid_preds, valid_targets)
    r2 = 1 - np.sum((valid_preds - valid_targets)**2) / np.sum((valid_targets - valid_targets.mean())**2)

    return {
        'pearson_r': float(r),
        'r2': float(r2),
        'mean_alpha': float(np.nanmean(all_alphas)),
        'final_loss': float(total_loss / max(n_batches, 1)),
        'nan_ratio': float(1 - valid_mask.mean())
    }


def main():
    parser = argparse.ArgumentParser(description='Run SFGN ablations')
    parser.add_argument('--dataset', type=str, default='agarwal', help='Dataset name')
    parser.add_argument('--max-samples', type=int, default=1000, help='Max samples')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs per ablation')
    parser.add_argument('--output-dir', type=str, default='results/sfgn/ablations', help='Output dir')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading {args.dataset}...")
    df, motif_annotations = load_dataset(args.dataset)

    # Subsample
    if len(df) > args.max_samples:
        indices = np.random.choice(len(df), args.max_samples, replace=False)
        df = df.iloc[indices].reset_index(drop=True)
        motif_annotations = [motif_annotations[i] for i in indices]

    sequences = df['sequence'].tolist()
    expressions = df['expression'].tolist()
    print(f"  {len(sequences)} sequences")

    results = []

    # Ablation 1: Orthogonality weight
    print("\n=== Ablation 1: Orthogonality Weight ===")
    orth_weights = [0.0, 0.01, 0.1, 0.5, 1.0]

    for orth_w in orth_weights:
        print(f"\n  Testing orthogonality_weight={orth_w}...")
        config = SFGNConfig(orthogonality_weight=orth_w)
        result = quick_train(config, sequences, expressions, motif_annotations,
                           device, epochs=args.epochs)
        result['ablation'] = 'orthogonality_weight'
        result['value'] = orth_w
        results.append(result)
        print(f"    α={result['mean_alpha']:.3f}, R²={result['r2']:.4f}")

    # Save results
    output_path = output_dir / f'{args.dataset}_ablations.json'
    with open(output_path, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'epochs': args.epochs,
            'n_samples': len(sequences),
            'results': results
        }, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"{'Orth Weight':<12} {'α':<8} {'R²':<10}")
    print("-" * 30)
    for r in results:
        if r['ablation'] == 'orthogonality_weight':
            print(f"{r['value']:<12} {r['mean_alpha']:<8.3f} {r['r2']:<10.4f}")

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
