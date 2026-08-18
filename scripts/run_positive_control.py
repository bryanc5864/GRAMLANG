#!/usr/bin/env python3
"""
Positive control on the Georgakopoulos-Soares MPRA library.

209,440 synthetic sequences with spacer DNA held constant by design, so their
measured orientation and order effects are uncontaminated. If the models pick
those up, grammar effects are real and the models can see them once the spacer
confound is gone.

Georgakopoulos-Soares et al., Nat Commun 2023, "Transcription factor binding
site orientation and order are major drivers of gene regulatory activity".
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model
from src.models.expression_probes import load_probe
from src.utils.io import save_json

DATA_DIR = Path('data/raw/georgakopoulos_soares')
RESULTS_DIR = Path('results/v3/positive_control')
PROBES_DIR = Path('data/probes')


def parse_library_file(filepath):
    """Parse the FASTA-like library design file.

    Returns list of (header, sequence) tuples.
    The library has Construct1/Construct2 variants (different backgrounds)
    for each motif configuration.
    """
    sequences = []
    current_header = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_header = line[1:]  # drop '>'
            else:
                sequences.append({
                    'header': current_header,
                    'sequence': line,
                })

    return sequences


def parse_header(header):
    """Parse construct header to extract TFs, orientations, positions, construct variant."""
    info = {
        'construct_type': None,  # single, pair or triplet
        'construct_variant': None,  # 1 or 2, different backgrounds
        'tfs': [],
        'orientations': [],  # template (-) / non-template (+)
        'positions': [],
        'full_orient_string': '',  # full orientation description
    }

    # Construct1 or Construct2
    match = re.match(r'Construct(\d+)', header)
    if match:
        info['construct_variant'] = int(match.group(1))

    if 'Single Motif' in header:
        info['construct_type'] = 'single'
    elif 'Triplet' in header:
        info['construct_type'] = 'triplet'
    else:
        # TF mentions tell pair from triplet
        tf_mentions = len(re.findall(r'[A-Z][A-Z0-9]+,', header))
        info['construct_type'] = 'pair' if tf_mentions <= 2 else 'triplet'

    tf_matches = re.findall(r'\b([A-Z][A-Z0-9]+)\b,', header)
    info['tfs'] = tf_matches

    # orientation string, formatted "Template:XXX" or "Non-template:XXX"
    orient_parts = re.findall(r'(Template|Non-template):[A-Z]+', header)
    info['full_orient_string'] = '|'.join(orient_parts)

    info['orientations'] = []
    for part in orient_parts:
        if part.startswith('Non-template'):
            info['orientations'].append('+')
        else:
            info['orientations'].append('-')

    pos_matches = re.findall(r'At Pos:(\d+)', header)
    info['positions'] = [int(p) for p in pos_matches]

    return info


def find_orientation_pairs_from_library(sequences, max_pairs=1000):
    """
    Pairs from the same construct variant differing only in orientation.

    Construct1 and Construct2 have different backgrounds, so comparisons stay
    inside one variant.
    """
    # group by construct_variant, sorted TFs, positions
    groups = defaultdict(list)

    for idx, item in enumerate(sequences):
        header = item['header']
        seq = item['sequence']
        parsed = parse_header(header)

        if parsed['construct_type'] != 'pair':
            continue
        if len(parsed['tfs']) < 2:
            continue

        # same background, TFs and positions; orientation is what differs
        key = (
            parsed['construct_variant'],
            tuple(sorted(parsed['tfs'])),
            tuple(sorted(parsed['positions'])),
        )

        groups[key].append({
            'idx': idx,
            'header': header,
            'sequence': seq,
            'orientations': tuple(parsed['orientations']),
            'orient_string': parsed['full_orient_string'],
        })

    orientation_pairs = []
    for key, variants in groups.items():
        if len(variants) < 2:
            continue

        for i, v1 in enumerate(variants):
            for v2 in variants[i+1:]:
                if v1['orient_string'] != v2['orient_string']:
                    orientation_pairs.append({
                        'construct_variant': key[0],
                        'tfs': key[1],
                        'positions': key[2],
                        'seq1': v1['sequence'],
                        'seq2': v2['sequence'],
                        'header1': v1['header'],
                        'header2': v2['header'],
                        'orient1': v1['orient_string'],
                        'orient2': v2['orient_string'],
                    })

                    if len(orientation_pairs) >= max_pairs:
                        return orientation_pairs

    return orientation_pairs


def find_order_pairs_from_library(sequences, max_pairs=1000):
    """Pairs from one construct variant, same TFs and orientations, different positions."""
    groups = defaultdict(list)

    for idx, item in enumerate(sequences):
        header = item['header']
        seq = item['sequence']
        parsed = parse_header(header)

        if parsed['construct_type'] != 'pair':
            continue
        if len(parsed['tfs']) < 2:
            continue

        # same background, TFs and orientations; position is what differs
        key = (
            parsed['construct_variant'],
            tuple(sorted(parsed['tfs'])),
            tuple(sorted(parsed['orientations'])),
        )

        groups[key].append({
            'idx': idx,
            'header': header,
            'sequence': seq,
            'positions': tuple(parsed['positions']),
        })

    order_pairs = []
    for key, variants in groups.items():
        if len(variants) < 2:
            continue

        for i, v1 in enumerate(variants):
            for v2 in variants[i+1:]:
                if v1['positions'] != v2['positions']:
                    order_pairs.append({
                        'construct_variant': key[0],
                        'tfs': key[1],
                        'orientations': key[2],
                        'seq1': v1['sequence'],
                        'seq2': v2['sequence'],
                        'header1': v1['header'],
                        'header2': v2['header'],
                        'pos1': v1['positions'],
                        'pos2': v2['positions'],
                    })

                    if len(order_pairs) >= max_pairs:
                        return order_pairs

    return order_pairs


def swap_probe(model, model_name, device='cuda'):
    """Try to load a HepG2-related probe since this is liver MPRA data."""
    for probe_name in [f'{model_name}_klein', f'{model_name}_vaishnav2022']:
        probe_path = PROBES_DIR / f'{probe_name}_probe.pt'
        if probe_path.exists():
            probe = load_probe(str(PROBES_DIR), probe_name, model.hidden_dim, device=device)
            model.set_probe(probe)
            print(f"  Loaded probe: {probe_name}")
            return True
    return False


def test_model_sensitivity(pairs, pair_type, model, batch_size=32):
    """Spacer DNA is constant, so any prediction difference is orientation or order."""
    print(f"\n  Testing {len(pairs)} {pair_type} pairs...")

    seqs1 = [p['seq1'] for p in pairs]
    seqs2 = [p['seq2'] for p in pairs]

    preds1 = model.predict_expression(seqs1)
    preds2 = model.predict_expression(seqs2)

    pred_diffs = preds1 - preds2
    abs_diffs = np.abs(pred_diffs)

    mean_diff = float(np.mean(abs_diffs))
    median_diff = float(np.median(abs_diffs))
    max_diff = float(np.max(abs_diffs))
    std_diff = float(np.std(pred_diffs))

    frac_diff_gt_01 = float(np.mean(abs_diffs > 0.1))
    frac_diff_gt_05 = float(np.mean(abs_diffs > 0.5))

    # are the differences bigger than model noise, and systematic?
    tstat, pval = stats.ttest_1samp(abs_diffs, 0)

    return {
        'n_pairs': len(pairs),
        'mean_abs_diff': mean_diff,
        'median_abs_diff': median_diff,
        'max_abs_diff': max_diff,
        'std_diff': std_diff,
        'frac_diff_gt_0.1': frac_diff_gt_01,
        'frac_diff_gt_0.5': frac_diff_gt_05,
        't_statistic': float(tstat),
        'p_value': float(pval),
    }


def run_positive_control(model_name='dnabert2'):
    """Run the positive control experiment."""
    print(f"\n{'='*60}")
    print(f"POSITIVE CONTROL EXPERIMENT — {model_name}")
    print(f"{'='*60}")
    print("Testing if models detect grammar when spacers are controlled")

    print("\nLoading Georgakopoulos-Soares MPRA library...")
    library_file = DATA_DIR / 'Library_MPRA_TFBSs.txt'

    if not library_file.exists():
        print(f"  library file not found at {library_file}")
        return None

    sequences = parse_library_file(library_file)
    print(f"  Loaded {len(sequences)} sequences from library")

    print("\nFinding controlled pair comparisons...")
    orientation_pairs = find_orientation_pairs_from_library(sequences, max_pairs=500)
    order_pairs = find_order_pairs_from_library(sequences, max_pairs=500)

    print(f"  Found {len(orientation_pairs)} orientation-variant pairs")
    print(f"  Found {len(order_pairs)} order-variant pairs")

    if not orientation_pairs and not order_pairs:
        print("  no controlled pairs found")
        return None

    print(f"\nLoading model {model_name}...")
    model = load_model(model_name, dataset_name='__dummy__')
    swap_probe(model, model_name)

    results = {
        'model': model_name,
        'n_total_sequences': len(sequences),
        'n_orientation_pairs': len(orientation_pairs),
        'n_order_pairs': len(order_pairs),
    }

    if orientation_pairs:
        orient_results = test_model_sensitivity(
            orientation_pairs, 'orientation', model
        )
        results['orientation'] = orient_results
        print(f"\n  Orientation sensitivity:")
        print(f"    Mean |Δpred|: {orient_results['mean_abs_diff']:.4f}")
        print(f"    Frac |Δpred| > 0.1: {orient_results['frac_diff_gt_0.1']*100:.1f}%")
        print(f"    t-test vs 0: t={orient_results['t_statistic']:.2f}, p={orient_results['p_value']:.2e}")

    if order_pairs:
        order_results = test_model_sensitivity(
            order_pairs, 'order', model
        )
        results['order'] = order_results
        print(f"\n  Order/position sensitivity:")
        print(f"    Mean |Δpred|: {order_results['mean_abs_diff']:.4f}")
        print(f"    Frac |Δpred| > 0.1: {order_results['frac_diff_gt_0.1']*100:.1f}%")
        print(f"    t-test vs 0: t={order_results['t_statistic']:.2f}, p={order_results['p_value']:.2e}")

    print(f"\n  Interpretation:")
    orient_mean = results.get('orientation', {}).get('mean_abs_diff', 0)
    order_mean = results.get('order', {}).get('mean_abs_diff', 0)

    if orient_mean > 0.05 or order_mean > 0.05:
        print(f"    Model IS sensitive to grammar (mean |Δ| > 0.05)")
        print(f"    → Models CAN detect orientation/order effects")
        print(f"    → Our negative v3 results are due to spacer confound")
        results['conclusion'] = 'model_detects_grammar'
    else:
        print(f"    Model NOT sensitive to grammar (mean |Δ| ≤ 0.05)")
        print(f"    → Models may not encode grammar effects")
        print(f"    → Need to verify with more pairs or different models")
        results['conclusion'] = 'model_insensitive'

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(results, RESULTS_DIR / f'{model_name}_positive_control.json')
    print(f"\n  Saved to {RESULTS_DIR}/")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Positive Control Experiment')
    parser.add_argument('--model', type=str, default='dnabert2')
    args = parser.parse_args()

    results = run_positive_control(model_name=args.model)

    print(f"\n{'='*60}")
    print("positive control done")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
