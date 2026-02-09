#!/usr/bin/env python3
"""
GRAMLANG v3: Detailed Spacer Ablation Experiment (P3.3)

Given that factorial decomposition showed spacer DNA accounts for 78-86% of GSI variance,
this experiment does a more detailed ablation to understand exactly what aspect of
spacer DNA the models are sensitive to.

Tests:
1. Spacer length distribution effects
2. Spacer GC content effects
3. Spacer dinucleotide composition effects
4. Spacer k-mer content effects
5. Spacer vs motif position interactions

Usage:
    python scripts/run_spacer_ablation.py --datasets agarwal,jores --n-enhancers 100
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model
from src.models.expression_probes import load_probe
from src.perturbation.vocabulary_preserving import merge_overlapping_motifs
from src.utils.sequence import (
    reverse_complement, dinucleotide_shuffle,
    generate_neutral_spacer, gc_content
)
from src.utils.io import load_processed, save_json

RESULTS_DIR = Path('results/v3/spacer_ablation')
PROBES_DIR = 'data/probes'


def swap_probe(model, model_name, ds_name, device='cuda'):
    """Swap expression probe on already-loaded model."""
    if model_name == 'enformer' or not hasattr(model, 'set_probe'):
        return
    for probe_name in [f'{model_name}_{ds_name}', f'{model_name}_vaishnav',
                       f'{model_name}_vaishnav2022']:
        probe_path = os.path.join(PROBES_DIR, f'{probe_name}_probe.pt')
        if os.path.exists(probe_path):
            probe = load_probe(PROBES_DIR, probe_name, model.hidden_dim, device=device)
            model.set_probe(probe)
            print(f"  Loaded probe: {probe_name}")
            return
    print(f"  WARNING: No probe found for {model_name}/{ds_name}")


def extract_motifs_and_spacers(sequence, motif_annotations):
    """Extract motif sequences and spacer regions from an enhancer."""
    motifs = motif_annotations.get('motifs', [])
    if len(motifs) < 2:
        return None, None, None

    motifs_sorted = sorted(motifs, key=lambda m: m['start'])
    merged = merge_overlapping_motifs(motifs_sorted)

    motif_seqs = []
    motif_positions = []
    for m in merged:
        motif_seqs.append({
            'seq': sequence[m['start']:m['end']],
            'start': m['start'],
            'end': m['end'],
            'name': m.get('motif_name', 'unknown'),
        })
        motif_positions.append((m['start'], m['end']))

    # Extract spacer regions
    spacer_regions = []
    prev_end = 0
    for start, end in motif_positions:
        if start > prev_end:
            spacer_regions.append({
                'seq': sequence[prev_end:start],
                'start': prev_end,
                'end': start,
            })
        prev_end = end
    if prev_end < len(sequence):
        spacer_regions.append({
            'seq': sequence[prev_end:],
            'start': prev_end,
            'end': len(sequence),
        })

    return motif_seqs, spacer_regions, motif_positions


def generate_spacer_variants(sequence, motif_seqs, spacer_regions, motif_positions,
                             variant_type='gc_shift', n_variants=20, rng=None):
    """
    Generate sequence variants with modified spacer DNA.

    variant_types:
    - 'gc_shift': Change spacer GC content (+10%, +20%, -10%, -20%)
    - 'length_shift': Change spacer lengths (redistribute)
    - 'dinuc_shuffle': Dinucleotide shuffle each spacer
    - 'random_replace': Replace spacers with random DNA
    - 'kmer_preserve': Preserve k-mer frequencies but shuffle
    """
    if rng is None:
        rng = np.random.default_rng()

    seq_len = len(sequence)
    all_spacer_text = ''.join(s['seq'] for s in spacer_regions)
    original_gc = gc_content(all_spacer_text) if all_spacer_text else 0.5

    variants = []

    if variant_type == 'gc_shift':
        # Generate spacers with shifted GC content
        gc_shifts = [-0.2, -0.1, 0, +0.1, +0.2]
        for gc_shift in gc_shifts:
            target_gc = max(0.1, min(0.9, original_gc + gc_shift))
            for _ in range(n_variants // len(gc_shifts)):
                new_seq = list(sequence)
                for spacer in spacer_regions:
                    new_spacer = generate_neutral_spacer(
                        len(spacer['seq']), gc=target_gc, rng=rng
                    )
                    for i, base in enumerate(new_spacer):
                        if spacer['start'] + i < seq_len:
                            new_seq[spacer['start'] + i] = base
                variants.append({
                    'sequence': ''.join(new_seq),
                    'gc_shift': gc_shift,
                    'target_gc': target_gc,
                })

    elif variant_type == 'dinuc_shuffle':
        # Dinucleotide shuffle each spacer independently
        for _ in range(n_variants):
            new_seq = list(sequence)
            for spacer in spacer_regions:
                if len(spacer['seq']) > 2:
                    shuffled = dinucleotide_shuffle(spacer['seq'], rng=rng)
                    for i, base in enumerate(shuffled):
                        if spacer['start'] + i < seq_len:
                            new_seq[spacer['start'] + i] = base
            variants.append({
                'sequence': ''.join(new_seq),
                'variant_type': 'dinuc_shuffle',
            })

    elif variant_type == 'random_replace':
        # Replace spacers with completely random DNA
        for _ in range(n_variants):
            new_seq = list(sequence)
            for spacer in spacer_regions:
                random_spacer = generate_neutral_spacer(
                    len(spacer['seq']), gc=0.5, rng=rng
                )
                for i, base in enumerate(random_spacer):
                    if spacer['start'] + i < seq_len:
                        new_seq[spacer['start'] + i] = base
            variants.append({
                'sequence': ''.join(new_seq),
                'variant_type': 'random',
            })

    elif variant_type == 'motif_only':
        # Keep spacers fixed, only move motifs to different positions
        total_motif_len = sum(len(m['seq']) for m in motif_seqs)
        total_spacer_len = sum(len(s['seq']) for s in spacer_regions)

        for _ in range(n_variants):
            # Randomly permute motif order
            perm = rng.permutation(len(motif_seqs))
            permuted_motifs = [motif_seqs[i] for i in perm]

            # Reconstruct sequence with same spacer order but permuted motifs
            new_seq = ''
            for i, spacer in enumerate(spacer_regions):
                new_seq += spacer['seq']
                if i < len(permuted_motifs):
                    new_seq += permuted_motifs[i]['seq']

            # Pad/trim to original length
            if len(new_seq) > seq_len:
                new_seq = new_seq[:seq_len]
            elif len(new_seq) < seq_len:
                new_seq += generate_neutral_spacer(seq_len - len(new_seq), gc=original_gc, rng=rng)

            variants.append({
                'sequence': new_seq,
                'variant_type': 'motif_permute',
            })

    return variants


def run_spacer_ablation(dataset_name, model_name='dnabert2', n_enhancers=100,
                        n_variants=50, seed=42):
    """Run detailed spacer ablation experiment."""
    print(f"\n{'='*60}")
    print(f"SPACER ABLATION — {dataset_name} / {model_name}")
    print(f"{'='*60}")

    # Load data
    dataset = load_processed(f'data/processed/{dataset_name}_processed.parquet')
    motif_hits = pd.read_parquet(f'data/processed/{dataset_name}_processed_motif_hits.parquet')

    # Load model
    print(f"  Loading model {model_name}...")
    model = load_model(model_name, dataset_name='__dummy__')
    swap_probe(model, model_name, dataset_name)

    # Sample enhancers
    eligible = dataset[dataset['n_motifs'] >= 3].copy()
    if len(eligible) > n_enhancers:
        eligible = eligible.sample(n=n_enhancers, random_state=seed)

    print(f"  Enhancers: {len(eligible)}, Variants per type: {n_variants}")

    rng = np.random.default_rng(seed)
    variant_types = ['gc_shift', 'dinuc_shuffle', 'random_replace', 'motif_only']

    all_results = []

    for idx, (_, row) in enumerate(eligible.iterrows()):
        if idx % 20 == 0:
            print(f"  [{idx}/{len(eligible)}] Processing {row['seq_id']}...")

        seq = row['sequence']
        seq_id = str(row['seq_id'])

        # Get motif annotations
        seq_motifs = motif_hits[motif_hits['seq_id'] == seq_id]
        annotation = {
            'sequence': seq,
            'motifs': seq_motifs.to_dict('records'),
        }

        # Extract structure
        motif_seqs, spacer_regions, motif_positions = extract_motifs_and_spacers(
            seq, annotation
        )

        if motif_seqs is None or len(spacer_regions) == 0:
            continue

        # Original expression
        original_expr = float(model.predict_expression([seq])[0])

        # Spacer statistics
        all_spacer = ''.join(s['seq'] for s in spacer_regions)
        spacer_gc = gc_content(all_spacer) if all_spacer else 0.5
        spacer_lengths = [len(s['seq']) for s in spacer_regions]

        enhancer_result = {
            'seq_id': seq_id,
            'original_expression': original_expr,
            'n_motifs': len(motif_seqs),
            'n_spacers': len(spacer_regions),
            'total_spacer_len': len(all_spacer),
            'spacer_gc': spacer_gc,
            'mean_spacer_len': np.mean(spacer_lengths),
        }

        # Test each variant type
        for vtype in variant_types:
            try:
                variants = generate_spacer_variants(
                    seq, motif_seqs, spacer_regions, motif_positions,
                    variant_type=vtype, n_variants=n_variants, rng=rng
                )

                if not variants:
                    continue

                var_seqs = [v['sequence'] for v in variants]
                var_exprs = model.predict_expression(var_seqs)

                var_mean = float(np.mean(var_exprs))
                var_std = float(np.std(var_exprs))
                var_min = float(np.min(var_exprs))
                var_max = float(np.max(var_exprs))

                # Effect size: how much does this perturbation change expression?
                delta = abs(original_expr - var_mean)
                z_score = delta / max(var_std, 1e-10)

                enhancer_result[f'{vtype}_mean'] = var_mean
                enhancer_result[f'{vtype}_std'] = var_std
                enhancer_result[f'{vtype}_min'] = var_min
                enhancer_result[f'{vtype}_max'] = var_max
                enhancer_result[f'{vtype}_delta'] = float(delta)
                enhancer_result[f'{vtype}_z'] = float(z_score)
                enhancer_result[f'{vtype}_range'] = float(var_max - var_min)

            except Exception as e:
                print(f"    Error {vtype} for {seq_id}: {e}")

        all_results.append(enhancer_result)

    df = pd.DataFrame(all_results)

    # Summary statistics
    summary = {
        'dataset': dataset_name,
        'model': model_name,
        'n_enhancers': len(df),
        'n_variants': n_variants,
        'variant_effects': {},
    }

    print(f"\n  Results:")
    for vtype in variant_types:
        delta_col = f'{vtype}_delta'
        range_col = f'{vtype}_range'
        z_col = f'{vtype}_z'

        if delta_col in df.columns:
            median_delta = float(df[delta_col].median())
            median_range = float(df[range_col].median())
            median_z = float(df[z_col].median())
            mean_delta = float(df[delta_col].mean())

            summary['variant_effects'][vtype] = {
                'median_delta': median_delta,
                'mean_delta': mean_delta,
                'median_range': median_range,
                'median_z': median_z,
            }

            print(f"  {vtype:15s}: median Δexpr={median_delta:.4f}, "
                  f"range={median_range:.4f}, z={median_z:.2f}")

    # Compare: which perturbation type has the biggest effect?
    if len(summary['variant_effects']) > 0:
        effects = {k: v['median_delta'] for k, v in summary['variant_effects'].items()}
        max_effect = max(effects, key=effects.get)
        summary['dominant_factor'] = max_effect
        summary['effect_ranking'] = sorted(effects.items(), key=lambda x: -x[1])
        print(f"\n  Dominant factor: {max_effect} (Δ={effects[max_effect]:.4f})")

    # Correlation: does spacer GC content predict expression change?
    if 'gc_shift_delta' in df.columns:
        gc_expr_corr = df['spacer_gc'].corr(df['original_expression'])
        summary['gc_expression_correlation'] = float(gc_expr_corr)
        print(f"  Spacer GC vs expression: r={gc_expr_corr:.3f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS_DIR / f'{dataset_name}_{model_name}_spacer_ablation.parquet')
    save_json(summary, RESULTS_DIR / f'{dataset_name}_{model_name}_spacer_summary.json')
    print(f"  Saved to {RESULTS_DIR}/")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Spacer Ablation Experiment')
    parser.add_argument('--datasets', type=str, default='agarwal,jores,de_almeida',
                        help='Comma-separated dataset names')
    parser.add_argument('--model', type=str, default='dnabert2')
    parser.add_argument('--n-enhancers', type=int, default=100)
    parser.add_argument('--n-variants', type=int, default=50)
    args = parser.parse_args()

    datasets = args.datasets.split(',')
    all_summaries = {}

    for ds in datasets:
        summary = run_spacer_ablation(
            ds, model_name=args.model,
            n_enhancers=args.n_enhancers,
            n_variants=args.n_variants
        )
        all_summaries[ds] = summary

    # Cross-dataset summary
    print(f"\n{'='*60}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'='*60}")

    for ds, summary in all_summaries.items():
        if summary and 'dominant_factor' in summary:
            print(f"  {ds}: {summary['dominant_factor']} dominates")

    save_json(all_summaries, RESULTS_DIR / 'spacer_ablation_summary.json')


if __name__ == '__main__':
    main()
