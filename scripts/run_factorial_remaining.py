#!/usr/bin/env python3
"""
Run factorial decomposition for NT v2-500M and HyenaDNA models.
Completes validation 3: Factorial Decomposition All Models.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model
from src.models.expression_probes import load_probe
from src.perturbation.vocabulary_preserving import (
    generate_vocabulary_preserving_shuffles,
    generate_position_only_shuffles,
    generate_orientation_only_shuffles,
    generate_spacer_only_shuffles,
)
from src.perturbation.motif_scanner import MotifScanner
from src.utils.io import load_processed, save_json

RESULTS_DIR = Path('results/v3/factorial_decomposition')
PROBES_DIR = Path('data/probes')
DATASETS = ['agarwal', 'jores', 'de_almeida']
MODELS = ['nt', 'hyenadna']  # Skip dnabert2, already done


def swap_probe(model, model_name, ds_name, device='cuda'):
    """Swap expression probe on already-loaded model."""
    if not hasattr(model, 'set_probe'):
        return
    for probe_name in [f'{model_name}_{ds_name}', f'{model_name}_vaishnav']:
        probe_path = PROBES_DIR / f'{probe_name}_probe.pt'
        if probe_path.exists():
            probe = load_probe(str(PROBES_DIR), probe_name, model.hidden_dim, device=device)
            model.set_probe(probe)
            print(f"  Loaded probe: {probe_name}")
            return
    print(f"  WARNING: No probe found for {model_name}/{ds_name}")


def load_data_and_model(dataset_name, model_name):
    """Load dataset, motif hits, and model with appropriate probe."""
    # Load dataset
    data_path = Path(f'data/processed/{dataset_name}_processed.parquet')
    motif_path = Path(f'data/processed/{dataset_name}_processed_motif_hits.parquet')

    dataset = pd.read_parquet(data_path)
    motif_hits = pd.read_parquet(motif_path) if motif_path.exists() else None

    # Load model
    print(f"  Loading {model_name}...")
    model = load_model(model_name, dataset_name=dataset_name)

    return dataset, motif_hits, model


def get_enhancer_sample(dataset, motif_hits, n=200):
    """Get a sample of enhancers with at least 2 motifs."""
    if motif_hits is None:
        return dataset.sample(n=min(n, len(dataset)), random_state=42)

    # Count motifs per sequence
    motif_counts = motif_hits.groupby('seq_id').size()
    valid_ids = motif_counts[motif_counts >= 2].index

    valid_dataset = dataset[dataset['seq_id'].isin(valid_ids)]
    return valid_dataset.sample(n=min(n, len(valid_dataset)), random_state=42)


def get_annotation(seq, seq_id, motif_hits):
    """Get motif annotation for a sequence."""
    if motif_hits is None:
        return {'motif_count': 0, 'motifs': []}

    seq_motifs = motif_hits[motif_hits['seq_id'] == seq_id]
    motifs = []
    for _, row in seq_motifs.iterrows():
        motifs.append({
            'start': int(row['start']),
            'end': int(row['end']),
            'motif_name': row.get('motif_name', 'unknown'),
            'strand': row.get('strand', '+'),
        })

    return {
        'motif_count': len(motifs),
        'motifs': sorted(motifs, key=lambda x: x['start']),
    }


def run_factorial_decomposition(dataset_name, model_name, n_enhancers=200, n_shuffles=100):
    """
    Run 4 types of shuffles to decompose grammar sensitivity.
    """
    print(f"\n{'='*60}")
    print(f"FACTORIAL DECOMPOSITION: {dataset_name} / {model_name}")
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

        # Original expression - pass as list for batch processing
        try:
            orig_pred = model.predict_expression([seq])
            original_expr = float(orig_pred[0])
        except Exception as e:
            print(f"  Error predicting original for {seq_id}: {e}")
            continue

        enhancer_result = {
            'seq_id': seq_id,
            'original_expression': original_expr,
            'n_motifs': annotation['motif_count'],
        }

        for stype in shuffle_types:
            try:
                shuffles = shuffle_funcs[stype](
                    seq, annotation, n_shuffles=n_shuffles, seed=42 + idx
                )

                # Get predictions - batch process all shuffles at once
                try:
                    exprs = model.predict_expression(shuffles)
                    exprs = np.array(exprs).flatten()
                except Exception as e:
                    print(f"  Error batch predicting {stype} for {seq_id}: {e}")
                    continue

                if len(exprs) < 10:
                    raise ValueError(f"Only {len(exprs)} valid predictions")

                exprs = np.array(exprs)
                shuf_mean = float(np.mean(exprs))
                shuf_std = float(np.std(exprs))
                shuf_median = float(np.median(exprs))

                z = abs(original_expr - shuf_mean) / max(shuf_std, 1e-10)
                variance = float(np.var(exprs))

                enhancer_result[f'{stype}_mean'] = shuf_mean
                enhancer_result[f'{stype}_std'] = shuf_std
                enhancer_result[f'{stype}_median'] = shuf_median
                enhancer_result[f'{stype}_variance'] = variance
                enhancer_result[f'{stype}_z_score'] = float(z)
                enhancer_result[f'{stype}_gsi'] = shuf_std / max(abs(shuf_mean), 1e-10)

            except Exception as e:
                print(f"  Error {stype} for {seq_id}: {e}")
                for key in ['mean', 'std', 'median', 'variance', 'z_score', 'gsi']:
                    enhancer_result[f'{stype}_{key}'] = np.nan

        all_results.append(enhancer_result)

    df = pd.DataFrame(all_results)

    # Compute summary
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
            print(f"  {stype:12s}: median variance={median_var:.6f}")

    # Fraction of full variance explained
    full_var = df['full_variance'].values
    for stype in ['position', 'orientation', 'spacer']:
        var_col = f'{stype}_variance'
        if var_col in df.columns:
            ratios = df[var_col].values / np.maximum(full_var, 1e-20)
            median_frac = float(np.nanmedian(ratios))
            mean_frac = float(np.nanmean(ratios))
            summary['effect_sizes'][stype] = {
                'median_fraction_of_full': median_frac,
                'mean_fraction_of_full': mean_frac,
            }
            print(f"  {stype:12s} / full: {median_frac:.1%}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS_DIR / f'{dataset_name}_{model_name}_factorial.parquet')
    save_json(summary, RESULTS_DIR / f'{dataset_name}_{model_name}_factorial_summary.json')
    print(f"  Saved to {RESULTS_DIR}/")

    # Clean up GPU memory
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main():
    print("="*60)
    print("validation 3 RESOLUTION: Factorial Decomposition - All Models")
    print("="*60)

    all_summaries = {}

    for model_name in MODELS:
        for dataset_name in DATASETS:
            # Check if already done
            summary_path = RESULTS_DIR / f'{dataset_name}_{model_name}_factorial_summary.json'
            if summary_path.exists():
                print(f"\n  Skipping {dataset_name}/{model_name} - already done")
                with open(summary_path) as f:
                    all_summaries[f'{dataset_name}_{model_name}'] = json.load(f)
                continue

            try:
                summary = run_factorial_decomposition(
                    dataset_name, model_name,
                    n_enhancers=200, n_shuffles=100
                )
                all_summaries[f'{dataset_name}_{model_name}'] = summary
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                all_summaries[f'{dataset_name}_{model_name}'] = {'error': str(e)}

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Dataset':<15} {'Model':<12} {'n':<5} {'Spacer %':<10} {'Position %':<12} {'Orientation %'}")
    print("-"*60)

    for key, summary in all_summaries.items():
        if 'error' in summary:
            continue
        dataset = summary['dataset']
        model = summary['model']
        n = summary['n_enhancers']
        spacer_pct = summary['effect_sizes'].get('spacer', {}).get('median_fraction_of_full', 0)
        pos_pct = summary['effect_sizes'].get('position', {}).get('median_fraction_of_full', 0)
        orient_pct = summary['effect_sizes'].get('orientation', {}).get('median_fraction_of_full', 0)
        print(f"{dataset:<15} {model:<12} {n:<5} {spacer_pct*100:>8.1f}% {pos_pct*100:>10.1f}% {orient_pct*100:>12.1f}%")

    # Save combined results
    combined_path = RESULTS_DIR / 'factorial_all_models_summary.json'
    save_json(all_summaries, combined_path)
    print(f"\n  Saved combined summary to {combined_path}")


if __name__ == '__main__':
    main()
