#!/usr/bin/env python3
"""
Run comprehensive confirmatory experiments for the Billboard Model.

Experiments:
1. Per-enhancer grammar classification (billboard/soft/strong/enhanceosome)
2. Motif-pair hotspot analysis (which TF pairs show grammar?)
3. Cross-dataset consistency check
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grammar.sensitivity import compute_gsi
from src.models.model_loader import load_model


def classify_enhancer(gsi_value, p_value, effect_size=None):
    """
    Classify enhancer by grammar contribution.

    Categories:
    - billboard: No significant grammar (p > 0.05 or GSI < 0.1)
    - soft: Weak grammar (p < 0.05, GSI 0.1-0.3)
    - moderate: Moderate grammar (p < 0.01, GSI 0.3-0.5)
    - strong: Strong grammar (p < 0.001, GSI > 0.5)
    """
    if p_value > 0.05 or gsi_value < 0.1:
        return 'billboard'
    elif p_value < 0.001 and gsi_value > 0.5:
        return 'strong'
    elif p_value < 0.01 and gsi_value > 0.3:
        return 'moderate'
    else:
        return 'soft'


def run_per_enhancer_classification(dataset_name, model_name='dnabert2',
                                    n_enhancers=500, n_shuffles=100,
                                    output_dir='results/confirmatory'):
    """Classify each enhancer by grammar contribution."""
    print(f"\n=== Per-Enhancer Classification: {dataset_name} ===")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = Path('data/processed') / f'{dataset_name}_processed.parquet'
    motif_path = Path('data/processed') / f'{dataset_name}_processed_motif_hits.parquet'

    if not data_path.exists():
        print(f"  Dataset not found: {data_path}")
        return None

    df = pd.read_parquet(data_path)
    motif_df = pd.read_parquet(motif_path)

    # Load model with correct probe
    model = load_model(model_name, dataset_name=dataset_name)

    # Sample enhancers
    if len(df) > n_enhancers:
        sample_idx = np.random.choice(len(df), n_enhancers, replace=False)
        df_sample = df.iloc[sample_idx].reset_index(drop=True)
    else:
        df_sample = df

    motif_groups = motif_df.groupby('seq_id')

    results = []
    classifications = defaultdict(int)

    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Classifying enhancers"):
        seq_id = row['seq_id']
        sequence = row['sequence']

        # Get motifs for this sequence
        if seq_id not in motif_groups.groups:
            continue

        motifs = motif_groups.get_group(seq_id)
        if len(motifs) < 2:
            continue

        motif_list = []
        for _, m in motifs.iterrows():
            motif_list.append({
                'start': int(m['start']),
                'end': int(m['end']),
                'motif_name': m.get('motif_name', 'unknown'),
                'strand': m.get('strand', '+')
            })

        try:
            # Compute GSI
            gsi_result = compute_gsi(
                sequence=sequence,
                motif_annotations={'motifs': motif_list},
                model=model,
                n_shuffles=n_shuffles
            )

            gsi_value = gsi_result.get('gsi', 0)
            p_value = gsi_result.get('p_value', 1.0)

            # Handle NaN
            if np.isnan(gsi_value) or np.isnan(p_value):
                classification = 'billboard'
                gsi_value = 0
                p_value = 1.0
            else:
                classification = classify_enhancer(gsi_value, p_value)

            classifications[classification] += 1

            results.append({
                'seq_id': seq_id,
                'gsi': float(gsi_value),
                'p_value': float(p_value),
                'classification': classification,
                'n_motifs': len(motif_list)
            })

        except Exception as e:
            print(f"  Error on {seq_id}: {e}")
            continue

    # Summary
    total = sum(classifications.values())
    summary = {
        'dataset': dataset_name,
        'model': model_name,
        'n_enhancers': total,
        'n_shuffles': n_shuffles,
        'classifications': dict(classifications),
        'percentages': {k: v/total*100 for k, v in classifications.items()},
        'per_enhancer_results': results
    }

    # Save
    out_file = output_path / f'{dataset_name}_enhancer_classification.json'
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Classification Summary:")
    print(f"  {'Class':<12} {'Count':<8} {'Percent':<8}")
    print(f"  {'-'*28}")
    for cls in ['billboard', 'soft', 'moderate', 'strong']:
        count = classifications.get(cls, 0)
        pct = count/total*100 if total > 0 else 0
        print(f"  {cls:<12} {count:<8} {pct:<8.1f}%")

    return summary


def run_motif_pair_analysis(dataset_name, model_name='dnabert2',
                           n_enhancers=300, n_shuffles=50,
                           output_dir='results/confirmatory'):
    """Analyze which motif pairs show grammar effects."""
    print(f"\n=== Motif Pair Analysis: {dataset_name} ===")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = Path('data/processed') / f'{dataset_name}_processed.parquet'
    motif_path = Path('data/processed') / f'{dataset_name}_processed_motif_hits.parquet'

    if not data_path.exists():
        print(f"  Dataset not found: {data_path}")
        return None

    df = pd.read_parquet(data_path)
    motif_df = pd.read_parquet(motif_path)

    # Load model with correct probe
    model = load_model(model_name, dataset_name=dataset_name)

    # Sample enhancers
    if len(df) > n_enhancers:
        sample_idx = np.random.choice(len(df), n_enhancers, replace=False)
        df_sample = df.iloc[sample_idx].reset_index(drop=True)
    else:
        df_sample = df

    motif_groups = motif_df.groupby('seq_id')

    # Track pair-specific effects
    pair_effects = defaultdict(list)

    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Analyzing pairs"):
        seq_id = row['seq_id']
        sequence = row['sequence']

        if seq_id not in motif_groups.groups:
            continue

        motifs = motif_groups.get_group(seq_id)
        if len(motifs) < 2:
            continue

        motif_list = []
        motif_names = []
        for _, m in motifs.iterrows():
            motif_list.append({
                'start': int(m['start']),
                'end': int(m['end']),
                'motif_name': m.get('motif_name', 'unknown'),
                'strand': m.get('strand', '+')
            })
            motif_names.append(m.get('motif_name', 'unknown'))

        try:
            # Compute GSI
            gsi_result = compute_gsi(
                sequence=sequence,
                motif_annotations={'motifs': motif_list},
                model=model,
                n_shuffles=n_shuffles
            )

            gsi_value = gsi_result.get('gsi', 0)
            if np.isnan(gsi_value):
                gsi_value = 0

            # Record for each unique pair
            unique_motifs = sorted(set(motif_names))
            for i, m1 in enumerate(unique_motifs):
                for m2 in unique_motifs[i:]:
                    pair_key = f"{m1}|{m2}" if m1 <= m2 else f"{m2}|{m1}"
                    pair_effects[pair_key].append(gsi_value)

        except Exception as e:
            continue

    # Compute pair statistics
    pair_stats = []
    for pair, gsi_values in pair_effects.items():
        if len(gsi_values) >= 5:  # Require at least 5 observations
            pair_stats.append({
                'pair': pair,
                'mean_gsi': float(np.mean(gsi_values)),
                'std_gsi': float(np.std(gsi_values)),
                'n_observations': len(gsi_values),
                'frac_significant': float(np.mean([g > 0.3 for g in gsi_values]))
            })

    # Sort by mean GSI
    pair_stats = sorted(pair_stats, key=lambda x: x['mean_gsi'], reverse=True)

    # Identify hotspots (top 5%) and inert (bottom 50%)
    n_pairs = len(pair_stats)
    hotspot_threshold = int(n_pairs * 0.05)
    inert_threshold = int(n_pairs * 0.5)

    hotspots = pair_stats[:hotspot_threshold] if hotspot_threshold > 0 else []
    inert = pair_stats[inert_threshold:] if inert_threshold < n_pairs else []

    summary = {
        'dataset': dataset_name,
        'model': model_name,
        'n_enhancers': len(df_sample),
        'n_unique_pairs': n_pairs,
        'hotspot_pairs': hotspots[:20],  # Top 20
        'inert_pairs': inert[:20],  # Bottom 20
        'all_pairs': pair_stats,
        'hotspot_mean_gsi': float(np.mean([p['mean_gsi'] for p in hotspots])) if hotspots else 0,
        'inert_mean_gsi': float(np.mean([p['mean_gsi'] for p in inert])) if inert else 0
    }

    # Save
    out_file = output_path / f'{dataset_name}_motif_pair_analysis.json'
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Pair Analysis Summary:")
    print(f"  Total unique pairs: {n_pairs}")
    print(f"  Hotspot pairs (top 5%): {len(hotspots)}")
    if hotspots:
        print(f"    Mean GSI: {summary['hotspot_mean_gsi']:.3f}")
        print(f"    Top pairs: {[p['pair'] for p in hotspots[:5]]}")
    print(f"  Inert pairs (bottom 50%): {len(inert)}")
    if inert:
        print(f"    Mean GSI: {summary['inert_mean_gsi']:.3f}")

    return summary


def run_cross_dataset_consistency(datasets, model_name='dnabert2',
                                  output_dir='results/confirmatory'):
    """Check if billboard finding is consistent across datasets."""
    print(f"\n=== Cross-Dataset Consistency ===")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load existing classification results
    all_results = []
    for dataset in datasets:
        result_file = output_path / f'{dataset}_enhancer_classification.json'
        if result_file.exists():
            with open(result_file) as f:
                result = json.load(f)
                all_results.append({
                    'dataset': dataset,
                    'billboard_pct': result['percentages'].get('billboard', 0),
                    'soft_pct': result['percentages'].get('soft', 0),
                    'moderate_pct': result['percentages'].get('moderate', 0),
                    'strong_pct': result['percentages'].get('strong', 0),
                    'n_enhancers': result['n_enhancers']
                })

    if not all_results:
        print("  No classification results found. Run per-enhancer classification first.")
        return None

    # Compute cross-dataset statistics
    billboard_pcts = [r['billboard_pct'] for r in all_results]

    summary = {
        'datasets': datasets,
        'per_dataset_results': all_results,
        'cross_dataset_statistics': {
            'mean_billboard_pct': float(np.mean(billboard_pcts)),
            'std_billboard_pct': float(np.std(billboard_pcts)),
            'min_billboard_pct': float(np.min(billboard_pcts)),
            'max_billboard_pct': float(np.max(billboard_pcts)),
            'consistent': float(np.std(billboard_pcts)) < 10  # <10% std = consistent
        }
    }

    # Save
    out_file = output_path / 'cross_dataset_consistency.json'
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Cross-Dataset Summary:")
    print(f"  {'Dataset':<15} {'Billboard%':<12} {'Soft%':<10} {'Strong%':<10}")
    print(f"  {'-'*47}")
    for r in all_results:
        print(f"  {r['dataset']:<15} {r['billboard_pct']:<12.1f} {r['soft_pct']:<10.1f} {r['strong_pct']:<10.1f}")
    print(f"\n  Mean billboard: {summary['cross_dataset_statistics']['mean_billboard_pct']:.1f}%")
    print(f"  Std billboard: {summary['cross_dataset_statistics']['std_billboard_pct']:.1f}%")
    print(f"  Consistent: {summary['cross_dataset_statistics']['consistent']}")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run confirmatory experiments')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['classification', 'pairs', 'consistency', 'all'])
    parser.add_argument('--datasets', nargs='+', default=['agarwal', 'jores', 'klein'])
    parser.add_argument('--model', type=str, default='dnabert2')
    parser.add_argument('--n-enhancers', type=int, default=500)
    parser.add_argument('--n-shuffles', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='results/confirmatory')
    args = parser.parse_args()

    np.random.seed(42)

    results = {}

    if args.experiment in ['classification', 'all']:
        for dataset in args.datasets:
            result = run_per_enhancer_classification(
                dataset, args.model, args.n_enhancers, args.n_shuffles, args.output_dir
            )
            if result:
                results[f'{dataset}_classification'] = result

    if args.experiment in ['pairs', 'all']:
        for dataset in args.datasets:
            result = run_motif_pair_analysis(
                dataset, args.model, min(args.n_enhancers, 300),
                min(args.n_shuffles, 50), args.output_dir
            )
            if result:
                results[f'{dataset}_pairs'] = result

    if args.experiment in ['consistency', 'all']:
        result = run_cross_dataset_consistency(args.datasets, args.model, args.output_dir)
        if result:
            results['consistency'] = result

    print(f"\n=== All experiments complete ===")
    print(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
