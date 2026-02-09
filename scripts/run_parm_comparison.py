#!/usr/bin/env python3
"""
GRAMLANG v3: PARM Model Comparison (P2.4)

Compare grammar sensitivity between PARM (MPRA-trained CNN) and foundation models.
PARM has pre-trained models for K562 (Agarwal) and HepG2 (Klein).

Since PARM is trained directly on MPRA expression data, it doesn't need an expression probe,
potentially avoiding the probe quality issues that affect foundation models.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools', 'PARM'))

from PARM.PARM_utils_load_model import load_PARM
from PARM.PARM_predict import get_prediction

from src.perturbation.vocabulary_preserving import generate_vocabulary_preserving_shuffles
from src.utils.io import load_processed, save_json

RESULTS_DIR = Path('results/v3/parm_comparison')
PARM_MODELS_DIR = Path('tools/PARM/pre_trained_models')


class PARMModel:
    """Wrapper to make PARM work like our model interface."""

    def __init__(self, cell_type='K562'):
        self.cell_type = cell_type
        self.models = []
        model_dir = PARM_MODELS_DIR / cell_type

        print(f"Loading PARM models for {cell_type}...")
        for file in os.listdir(model_dir):
            if file.endswith('.parm'):
                model_path = str(model_dir / file)
                model = load_PARM(model_path, train=False)
                if torch.cuda.is_available():
                    model = model.cuda()
                model.eval()
                self.models.append(model)
        print(f"  Loaded {len(self.models)} model folds")

    def predict_expression(self, sequences):
        """Predict expression for a list of sequences."""
        all_preds = []
        for model in self.models:
            preds = get_prediction(sequences, model)
            all_preds.append(preds.flatten())

        # Average across folds
        avg_preds = np.mean(all_preds, axis=0)
        return avg_preds


def compute_gsi_parm(sequence, motif_annotations, model, n_shuffles=100, seed=None):
    """Compute GSI for a sequence using PARM model."""
    # Generate shuffles
    shuffles = generate_vocabulary_preserving_shuffles(
        sequence, motif_annotations, n_shuffles=n_shuffles, seed=seed
    )

    if len(shuffles) < 10:
        return None

    # Get predictions
    original_expr = float(model.predict_expression([sequence])[0])
    shuffle_exprs = model.predict_expression(shuffles)

    shuffle_mean = float(np.mean(shuffle_exprs))
    shuffle_std = float(np.std(shuffle_exprs))

    # GSI = CV of shuffles
    gsi = shuffle_std / max(abs(shuffle_mean), 1e-10)

    # z-score
    z_score = abs(original_expr - shuffle_mean) / max(shuffle_std, 1e-10)

    return {
        'gsi': gsi,
        'z_score': z_score,
        'original_expr': original_expr,
        'shuffle_mean': shuffle_mean,
        'shuffle_std': shuffle_std,
        'n_shuffles': len(shuffles),
    }


def run_parm_comparison(dataset_name, cell_type, n_enhancers=200, n_shuffles=100, seed=42):
    """Run PARM grammar sensitivity analysis."""
    print(f"\n{'='*60}")
    print(f"PARM COMPARISON — {dataset_name} / {cell_type}")
    print(f"{'='*60}")

    # Load data
    dataset = load_processed(f'data/processed/{dataset_name}_processed.parquet')
    motif_hits = pd.read_parquet(f'data/processed/{dataset_name}_processed_motif_hits.parquet')

    # Load PARM model
    parm = PARMModel(cell_type=cell_type)

    # Sample enhancers with multiple motifs
    eligible = dataset[dataset['n_motifs'] >= 2].copy()

    # PARM requires sequences <= 600 bp
    if 'sequence' in eligible.columns:
        eligible = eligible[eligible['sequence'].str.len() <= 600]

    if len(eligible) > n_enhancers:
        eligible = eligible.sample(n=n_enhancers, random_state=seed)

    print(f"  Enhancers: {len(eligible)}, Shuffles: {n_shuffles}")

    results = []

    for idx, (_, row) in enumerate(eligible.iterrows()):
        if idx % 20 == 0:
            print(f"  [{idx}/{len(eligible)}] Processing {row['seq_id']}...")

        seq = row['sequence']
        seq_id = str(row['seq_id'])

        # Skip if sequence too long for PARM
        if len(seq) > 600:
            continue

        # Get motif annotations
        seq_motifs = motif_hits[motif_hits['seq_id'] == seq_id]
        annotation = {
            'sequence': seq,
            'motifs': seq_motifs.to_dict('records'),
        }

        # Compute GSI (use seq_id hash + idx for reproducible but varying seeds)
        seq_seed = seed + idx if seed else idx
        try:
            gsi_result = compute_gsi_parm(
                seq, annotation, parm,
                n_shuffles=n_shuffles, seed=seq_seed
            )

            if gsi_result is not None:
                results.append({
                    'seq_id': seq_id,
                    'n_motifs': len(seq_motifs),
                    'seq_len': len(seq),
                    **gsi_result
                })
        except Exception as e:
            print(f"    Error for {seq_id}: {e}")

    df = pd.DataFrame(results)

    # Summary statistics
    summary = {
        'dataset': dataset_name,
        'cell_type': cell_type,
        'model': 'PARM',
        'n_enhancers': len(df),
        'n_shuffles': n_shuffles,
    }

    if len(df) > 0:
        summary['median_gsi'] = float(df['gsi'].median())
        summary['mean_gsi'] = float(df['gsi'].mean())
        summary['median_z'] = float(df['z_score'].median())
        summary['frac_significant'] = float((df['z_score'] > 1.96).mean())
        summary['frac_gsi_gt_0.1'] = float((df['gsi'] > 0.1).mean())

        print(f"\n  Results:")
        print(f"  Median GSI: {summary['median_gsi']:.4f}")
        print(f"  Mean GSI: {summary['mean_gsi']:.4f}")
        print(f"  Median z-score: {summary['median_z']:.3f}")
        print(f"  Significant (z > 1.96): {summary['frac_significant']*100:.1f}%")
        print(f"  GSI > 0.1: {summary['frac_gsi_gt_0.1']*100:.1f}%")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS_DIR / f'{dataset_name}_{cell_type}_parm_gsi.parquet')
    save_json(summary, RESULTS_DIR / f'{dataset_name}_{cell_type}_parm_summary.json')
    print(f"  Saved to {RESULTS_DIR}/")

    return summary


def compare_with_foundation_models(dataset_name):
    """Compare PARM GSI with foundation model GSI from v2 results."""
    parm_results = RESULTS_DIR / f'{dataset_name}_*_parm_gsi.parquet'

    # Load PARM results
    parm_files = list(RESULTS_DIR.glob(f'{dataset_name}_*_parm_gsi.parquet'))
    if not parm_files:
        print(f"No PARM results found for {dataset_name}")
        return None

    parm_df = pd.read_parquet(parm_files[0])

    # Load v2 foundation model results
    v2_gsi_file = Path(f'results/v2/{dataset_name}_gsi_sensitivity.parquet')
    if not v2_gsi_file.exists():
        print(f"No v2 results found at {v2_gsi_file}")
        return None

    v2_df = pd.read_parquet(v2_gsi_file)

    # Merge on seq_id
    merged = parm_df.merge(
        v2_df[['seq_id', 'gsi', 'z_score']].rename(
            columns={'gsi': 'gsi_v2', 'z_score': 'z_v2'}
        ),
        on='seq_id', how='inner'
    )

    if len(merged) < 10:
        print(f"Too few overlapping sequences ({len(merged)})")
        return None

    # Compare
    gsi_corr = merged['gsi'].corr(merged['gsi_v2'])
    z_corr = merged['z_score'].corr(merged['z_v2'])

    comparison = {
        'dataset': dataset_name,
        'n_overlap': len(merged),
        'gsi_correlation': float(gsi_corr),
        'z_correlation': float(z_corr),
        'parm_median_gsi': float(merged['gsi'].median()),
        'v2_median_gsi': float(merged['gsi_v2'].median()),
        'parm_median_z': float(merged['z_score'].median()),
        'v2_median_z': float(merged['z_v2'].median()),
    }

    print(f"\n  PARM vs Foundation Model Comparison ({dataset_name}):")
    print(f"  GSI correlation: r = {gsi_corr:.3f}")
    print(f"  Z-score correlation: r = {z_corr:.3f}")
    print(f"  PARM median GSI: {comparison['parm_median_gsi']:.4f}")
    print(f"  Foundation median GSI: {comparison['v2_median_gsi']:.4f}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description='PARM Model Comparison')
    parser.add_argument('--n-enhancers', type=int, default=200)
    parser.add_argument('--n-shuffles', type=int, default=100)
    args = parser.parse_args()

    # Dataset to cell type mapping
    # Agarwal (K562), Klein (HepG2)
    mappings = [
        ('agarwal', 'K562'),
        ('klein', 'HepG2'),
    ]

    all_summaries = {}

    for dataset_name, cell_type in mappings:
        try:
            summary = run_parm_comparison(
                dataset_name, cell_type,
                n_enhancers=args.n_enhancers,
                n_shuffles=args.n_shuffles
            )
            all_summaries[dataset_name] = summary

            # Compare with foundation models
            comparison = compare_with_foundation_models(dataset_name)
            if comparison:
                all_summaries[f'{dataset_name}_comparison'] = comparison

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save master summary
    save_json(all_summaries, RESULTS_DIR / 'parm_comparison_summary.json')
    print(f"\n{'='*60}")
    print("PARM COMPARISON COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
