"""
Experiment B: Synthetic Grammar Probes.

For each frequent motif pair, construct synthetic sequences with controlled
spacing, orientation, and order to measure clean grammar effects uncontaminated
by vocabulary variation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from itertools import combinations

from src.utils.sequence import reverse_complement, generate_neutral_spacer, gc_content


def run_synthetic_grammar_probes(
    model,
    motif_pairs: List[Tuple[str, str, str, str]],  # (name_a, seq_a, name_b, seq_b)
    target_length: int = 200,
    spacing_ladder: List[int] = None,
    fine_spacing_range: Tuple[int, int] = (5, 25),
    gc_target: float = 0.4,
    cell_type: str = None,
) -> pd.DataFrame:
    """
    Run synthetic grammar probes for a set of motif pairs.

    For each pair, measures:
    1. Spacing ladder: expression across coarse spacing grid
    2. Orientation panel: all 4 orientations at optimal spacing
    3. Order swap: AB vs BA at optimal spacing
    4. Helical scan: fine 1bp spacing from 5-25bp

    Returns DataFrame with one row per (pair, condition) test.
    """
    if spacing_ladder is None:
        spacing_ladder = [2, 5, 8, 10, 12, 15, 20, 30, 50]

    results = []

    for pair_idx, (name_a, seq_a, name_b, seq_b) in enumerate(motif_pairs):
        pair_name = f"{name_a}_{name_b}"

        # 1. Spacing ladder
        spacing_seqs = []
        valid_spacings = []
        for sp in spacing_ladder:
            constructed = _build_synthetic(seq_a, seq_b, sp, target_length, gc_target)
            if constructed:
                spacing_seqs.append(constructed)
                valid_spacings.append(sp)

        if not spacing_seqs:
            continue

        spacing_exprs = model.predict_expression(spacing_seqs, cell_type=cell_type)

        for sp, expr in zip(valid_spacings, spacing_exprs):
            results.append({
                'pair': pair_name,
                'motif_a': name_a,
                'motif_b': name_b,
                'test_type': 'spacing_ladder',
                'spacing': sp,
                'orientation': '+/+',
                'order': 'AB',
                'expression': float(expr),
                'model': model.name,
            })

        # Find optimal spacing from ladder
        opt_sp_idx = np.argmax(spacing_exprs)
        opt_sp = valid_spacings[opt_sp_idx]

        # 2. Orientation panel at optimal spacing
        orientations = ['+/+', '+/-', '-/+', '-/-']
        orient_seqs = []
        for orient in orientations:
            a_strand, b_strand = orient.split('/')
            a_seq = seq_a if a_strand == '+' else reverse_complement(seq_a)
            b_seq = seq_b if b_strand == '+' else reverse_complement(seq_b)
            constructed = _build_synthetic(a_seq, b_seq, opt_sp, target_length, gc_target)
            if constructed:
                orient_seqs.append(constructed)

        if orient_seqs:
            orient_exprs = model.predict_expression(orient_seqs, cell_type=cell_type)
            for orient, expr in zip(orientations[:len(orient_exprs)], orient_exprs):
                results.append({
                    'pair': pair_name,
                    'motif_a': name_a,
                    'motif_b': name_b,
                    'test_type': 'orientation',
                    'spacing': opt_sp,
                    'orientation': orient,
                    'order': 'AB',
                    'expression': float(expr),
                    'model': model.name,
                })

        # 3. Order swap: AB vs BA
        ba_seq = _build_synthetic(seq_b, seq_a, opt_sp, target_length, gc_target)
        if ba_seq:
            ba_expr = model.predict_expression([ba_seq], cell_type=cell_type)
            results.append({
                'pair': pair_name,
                'motif_a': name_a,
                'motif_b': name_b,
                'test_type': 'order_swap',
                'spacing': opt_sp,
                'orientation': '+/+',
                'order': 'BA',
                'expression': float(ba_expr[0]),
                'model': model.name,
            })

        # 4. Fine helical scan (1bp resolution, 5-25bp)
        fine_spacings = list(range(fine_spacing_range[0], fine_spacing_range[1] + 1))
        fine_seqs = []
        valid_fine = []
        for sp in fine_spacings:
            constructed = _build_synthetic(seq_a, seq_b, sp, target_length, gc_target)
            if constructed:
                fine_seqs.append(constructed)
                valid_fine.append(sp)

        if fine_seqs:
            fine_exprs = model.predict_expression(fine_seqs, cell_type=cell_type)
            for sp, expr in zip(valid_fine, fine_exprs):
                results.append({
                    'pair': pair_name,
                    'motif_a': name_a,
                    'motif_b': name_b,
                    'test_type': 'helical_scan',
                    'spacing': sp,
                    'orientation': '+/+',
                    'order': 'AB',
                    'expression': float(expr),
                    'model': model.name,
                })

    return pd.DataFrame(results)


def summarize_synthetic_probes(results_df: pd.DataFrame) -> Dict:
    """Summarize synthetic grammar probe results."""
    summary = {}

    for pair in results_df['pair'].unique():
        pair_data = results_df[results_df['pair'] == pair]

        # Spacing effect
        spacing_data = pair_data[pair_data['test_type'] == 'spacing_ladder']
        if len(spacing_data) > 1:
            spacing_range = float(spacing_data['expression'].max() - spacing_data['expression'].min())
            spacing_fold = float(spacing_data['expression'].max() / max(spacing_data['expression'].min(), 1e-10))
        else:
            spacing_range = 0
            spacing_fold = 1

        # Orientation effect
        orient_data = pair_data[pair_data['test_type'] == 'orientation']
        if len(orient_data) > 1:
            orient_range = float(orient_data['expression'].max() - orient_data['expression'].min())
            best_orient = orient_data.loc[orient_data['expression'].idxmax(), 'orientation']
        else:
            orient_range = 0
            best_orient = '+/+'

        # Order effect
        ab_data = pair_data[(pair_data['test_type'] == 'spacing_ladder') & (pair_data['order'] == 'AB')]
        ba_data = pair_data[pair_data['test_type'] == 'order_swap']
        if len(ab_data) > 0 and len(ba_data) > 0:
            # Compare at same spacing
            order_effect = float(ba_data['expression'].iloc[0]) - float(ab_data['expression'].max())
        else:
            order_effect = 0

        # Helical periodicity
        helical_data = pair_data[pair_data['test_type'] == 'helical_scan'].sort_values('spacing')
        helical_score = 0
        if len(helical_data) >= 10:
            exprs = helical_data['expression'].values
            spacings = helical_data['spacing'].values
            # Detrend and FFT
            coeffs = np.polyfit(spacings, exprs, 1)
            detrended = exprs - np.polyval(coeffs, spacings)
            fft_vals = np.fft.rfft(detrended)
            freqs = np.fft.rfftfreq(len(detrended), d=1.0)
            power = np.abs(fft_vals) ** 2
            helical_freq = 1.0 / 10.5
            freq_idx = np.argmin(np.abs(freqs - helical_freq))
            mean_power = np.mean(power[1:]) if len(power) > 1 else 1e-10
            if mean_power > 0:
                helical_score = float(power[freq_idx] / mean_power)

        summary[pair] = {
            'spacing_range': spacing_range,
            'spacing_fold_change': spacing_fold,
            'orientation_range': orient_range,
            'best_orientation': best_orient,
            'order_effect': order_effect,
            'helical_phase_score': helical_score,
            'grammar_potential': spacing_range + orient_range,
        }

    return summary


def get_top_motif_pairs(rules_df: pd.DataFrame, motif_hits: pd.DataFrame,
                        dataset: pd.DataFrame, n_pairs: int = 20) -> List[Tuple]:
    """Get the top N most frequent motif pairs with their consensus sequences."""
    # Count pair frequencies
    pair_counts = rules_df.groupby(['motif_a', 'motif_b']).size().sort_values(ascending=False)
    top_pairs = pair_counts.head(n_pairs)

    result = []
    for (name_a, name_b), count in top_pairs.items():
        # Get representative sequences for each motif from motif_hits
        hits_a = motif_hits[motif_hits['motif_name'] == name_a]
        hits_b = motif_hits[motif_hits['motif_name'] == name_b]

        if len(hits_a) == 0 or len(hits_b) == 0:
            continue

        # Get the most common sequence for each motif
        seq_a = _get_consensus_sequence(hits_a, dataset)
        seq_b = _get_consensus_sequence(hits_b, dataset)

        if seq_a and seq_b:
            result.append((name_a, seq_a, name_b, seq_b))

    return result


def _get_consensus_sequence(hits_df, dataset):
    """Extract the most representative motif sequence from hits."""
    if len(hits_df) == 0:
        return None

    # Try to extract actual sequences from the dataset
    seqs = []
    for _, hit in hits_df.head(50).iterrows():
        seq_id = hit.get('seq_id')
        start = hit.get('start', 0)
        end = hit.get('end', 0)
        if seq_id is not None and start < end:
            match = dataset[dataset['seq_id'] == seq_id]
            if len(match) > 0:
                full_seq = match.iloc[0]['sequence']
                if end <= len(full_seq):
                    seqs.append(full_seq[int(start):int(end)])

    if not seqs:
        return None

    # Return the most common length sequence (or first one)
    from collections import Counter
    lengths = Counter(len(s) for s in seqs)
    target_len = lengths.most_common(1)[0][0]
    seqs = [s for s in seqs if len(s) == target_len]
    return seqs[0] if seqs else None


def _build_synthetic(seq_a, seq_b, spacing, target_length, gc):
    """Build synthetic sequence with two motifs at given spacing."""
    spacer = generate_neutral_spacer(spacing, gc=gc)
    core = seq_a + spacer + seq_b
    if len(core) > target_length:
        return None

    pad_total = target_length - len(core)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    rng = np.random.default_rng(hash((seq_a, seq_b, spacing)) % (2**32))
    left = generate_neutral_spacer(pad_left, gc=gc, rng=rng)
    right = generate_neutral_spacer(pad_right, gc=gc, rng=rng)

    return left + core + right
