#!/usr/bin/env python3
"""
Train Spacer-Factored Grammar Network (SFGN).

Usage:
    python scripts/train_sfgn.py --dataset agarwal --epochs 50 --batch-size 32
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sfgn import SFGN, SFGNConfig


class MPRADataset(Dataset):
    """Dataset for MPRA sequences with motif annotations."""

    def __init__(
        self,
        sequences: list,
        expressions: list,
        motif_annotations: list,
    ):
        self.sequences = sequences
        self.expressions = expressions
        self.motif_annotations = motif_annotations

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'expression': self.expressions[idx],
            'motifs': self.motif_annotations[idx],
        }


def collate_fn(batch):
    """Custom collate function for variable-length motif lists."""
    sequences = [item['sequence'] for item in batch]
    expressions = torch.tensor([item['expression'] for item in batch], dtype=torch.float32)
    motifs = [item['motifs'] for item in batch]
    return sequences, expressions, motifs


def load_dataset(dataset_name: str, data_dir: str = 'data/processed'):
    """Load dataset and motif annotations."""
    data_path = Path(data_dir) / f'{dataset_name}_processed.parquet'
    motif_path = Path(data_dir) / f'{dataset_name}_processed_motif_hits.parquet'

    print(f"Loading {dataset_name}...")
    df = pd.read_parquet(data_path)
    print(f"  {len(df)} sequences")

    # Load motif hits
    if motif_path.exists():
        motif_df = pd.read_parquet(motif_path)
        print(f"  {len(motif_df)} motif hits")

        # Group motifs by sequence
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
    else:
        print("  No motif annotations found, using empty lists")
        motif_annotations = [[] for _ in range(len(df))]

    return df, motif_annotations


def evaluate(model, dataloader, device):
    """Evaluate model on dataset."""
    model.eval()
    all_preds = []
    all_targets = []
    all_alphas = []
    all_betas = []

    with torch.no_grad():
        for sequences, expressions, motifs in dataloader:
            expressions = expressions.to(device)

            output = model(sequences, motifs)
            all_preds.extend(output.prediction.cpu().numpy())
            all_targets.extend(expressions.cpu().numpy())
            all_alphas.extend(output.alpha.cpu().numpy())
            all_betas.extend(output.beta.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    r, _ = pearsonr(all_preds, all_targets)
    rho, _ = spearmanr(all_preds, all_targets)
    r2 = 1 - np.sum((all_preds - all_targets) ** 2) / np.sum((all_targets - all_targets.mean()) ** 2)

    return {
        'mse': mse,
        'pearson_r': r,
        'spearman_rho': rho,
        'r2': r2,
        'mean_alpha': np.mean(all_alphas),
        'mean_beta': np.mean(all_betas),
        'std_alpha': np.std(all_alphas),
    }


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_mse = 0
    total_orth = 0
    n_batches = 0
    nan_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for sequences, expressions, motifs in pbar:
        expressions = expressions.to(device)

        optimizer.zero_grad()

        output = model(sequences, motifs)
        loss, metrics = model.compute_loss(output, expressions)

        # Skip NaN losses
        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            pbar.set_postfix({'loss': 'NaN', 'skipped': nan_batches})
            continue

        loss.backward()

        # Check for NaN gradients and skip if found
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan_grad = True
                break

        if has_nan_grad:
            nan_batches += 1
            optimizer.zero_grad()  # Clear the bad gradients
            pbar.set_postfix({'loss': 'NaN grad', 'skipped': nan_batches})
            continue

        # Aggressive gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += metrics['total_loss']
        total_mse += metrics['mse_loss']
        total_orth += metrics['orthogonality_loss']
        n_batches += 1

        pbar.set_postfix({
            'loss': f"{metrics['total_loss']:.4f}",
            'mse': f"{metrics['mse_loss']:.4f}",
            'α': f"{metrics['mean_alpha']:.3f}",
        })

    if n_batches == 0:
        return {'loss': float('nan'), 'mse': float('nan'), 'orthogonality': float('nan')}

    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'orthogonality': total_orth / n_batches,
        'nan_batches': nan_batches,
    }


def main():
    parser = argparse.ArgumentParser(description='Train SFGN')
    parser.add_argument('--dataset', type=str, default='agarwal', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--orthogonality-weight', type=float, default=0.1, help='Orthogonality loss weight')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples for debugging')
    parser.add_argument('--output-dir', type=str, default='results/sfgn', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df, motif_annotations = load_dataset(args.dataset)

    # Subsample if requested
    if args.max_samples and len(df) > args.max_samples:
        indices = np.random.choice(len(df), args.max_samples, replace=False)
        df = df.iloc[indices].reset_index(drop=True)
        motif_annotations = [motif_annotations[i] for i in indices]
        print(f"Subsampled to {len(df)} sequences")

    # Train/val split
    indices = list(range(len(df)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=args.seed)

    train_dataset = MPRADataset(
        [df.iloc[i]['sequence'] for i in train_idx],
        [df.iloc[i]['expression'] for i in train_idx],
        [motif_annotations[i] for i in train_idx],
    )
    val_dataset = MPRADataset(
        [df.iloc[i]['sequence'] for i in val_idx],
        [df.iloc[i]['expression'] for i in val_idx],
        [motif_annotations[i] for i in val_idx],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    config = SFGNConfig(
        orthogonality_weight=args.orthogonality_weight,
    )
    model = SFGN(config, device=device)

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_r2 = -float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"\nEpoch {epoch}:")
        print(f"  Train: loss={train_metrics['loss']:.4f}, mse={train_metrics['mse']:.4f}")
        print(f"  Val:   R²={val_metrics['r2']:.4f}, r={val_metrics['pearson_r']:.4f}, "
              f"α={val_metrics['mean_alpha']:.3f}±{val_metrics['std_alpha']:.3f}")

        history.append({
            'epoch': epoch,
            'train_loss': float(train_metrics['loss']),
            'train_mse': float(train_metrics['mse']),
            'val_r2': float(val_metrics['r2']),
            'val_pearson': float(val_metrics['pearson_r']),
            'mean_alpha': float(val_metrics['mean_alpha']),
            'mean_beta': float(val_metrics['mean_beta']),
        })

        # Save best model
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'val_metrics': val_metrics,
            }, output_dir / f'{args.dataset}_sfgn_best.pt')
            print(f"  Saved best model (R²={best_val_r2:.4f})")

    # Save final model and history
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, output_dir / f'{args.dataset}_sfgn_final.pt')

    with open(output_dir / f'{args.dataset}_sfgn_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    final_metrics = evaluate(model, val_loader, device)
    print(f"Validation R²: {final_metrics['r2']:.4f}")
    print(f"Validation Pearson r: {final_metrics['pearson_r']:.4f}")
    print(f"Mean α (grammar weight): {final_metrics['mean_alpha']:.3f}")
    print(f"Mean β (composition weight): {final_metrics['mean_beta']:.3f}")

    # Save final metrics (convert numpy types to Python floats)
    final_metrics_clean = {k: float(v) for k, v in final_metrics.items()}
    with open(output_dir / f'{args.dataset}_sfgn_metrics.json', 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'best_val_r2': float(best_val_r2),
            'final_metrics': final_metrics_clean,
            'config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'orthogonality_weight': args.orthogonality_weight,
            },
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
