"""
Redesigned Compositionality Test (enhancer-specific ANOVA).

Instead of applying averaged rules across enhancers, this tests
compositionality within each enhancer: do pairwise perturbation effects
predict combined perturbation effects for that specific enhancer?
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from itertools import combinations

from src.utils.sequence import reverse_complement, generate_neutral_spacer, gc_content


def run_enhancer_specific_compositionality(
    dataset: pd.DataFrame,
    motif_hits: pd.DataFrame,
    model,
    cell_type: str = None,
    min_motifs: int = 3,
    max_enhancers: int = 100,
    n_perturbations: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Enhancer-specific compositionality test.

    For each enhancer with k >= 3 motifs:
    1. Measure effect of perturbing each motif pair individually
    2. Measure effect of perturbing pairs simultaneously
    3. Interaction term = combined - sum of individual effects
    4. Compositionality = how well sum predicts combined

    This is a factorial design testing additivity of grammar effects.
    """
    rng = np.random.default_rng(seed)
    results = []

    # Filter to sequences with enough motifs
    seq_counts = motif_hits.groupby('seq_id').size()
    eligible = seq_counts[seq_counts >= min_motifs].index
    eligible_df = dataset[dataset['seq_id'].isin(eligible)]

    if len(eligible_df) > max_enhancers:
        eligible_df = eligible_df.sample(max_enhancers, random_state=seed)

    for _, row in eligible_df.iterrows():
        seq_id = str(row['seq_id'])
        seq = row['sequence']
        seq_len = len(seq)
        seq_gc = gc_content(seq)

        hits = motif_hits[motif_hits['seq_id'] == seq_id].sort_values('start')
        if len(hits) < 3:
            continue

        # Get original prediction
        orig_pred = model.predict_expression([seq], cell_type=cell_type)[0]

        # Select motif pairs to test (up to 5 pairs)
        motif_indices = list(range(min(len(hits), 8)))
        pair_indices = list(combinations(motif_indices, 2))
        if len(pair_indices) > 10:
            selected = rng.choice(len(pair_indices), 10, replace=False)
            pair_indices = [pair_indices[i] for i in selected]

        # For each pair, measure individual perturbation effect
        pair_effects = {}
        for i, j in pair_indices:
            motif_i = hits.iloc[i]
            motif_j = hits.iloc[j]

            # Perturb pair (i,j): swap their relative arrangement
            perturbed_seqs = []
            for _ in range(n_perturbations):
                perturbed = _perturb_pair(seq, motif_i, motif_j, rng, seq_gc)
                if perturbed and len(perturbed) == seq_len:
                    perturbed_seqs.append(perturbed)

            if len(perturbed_seqs) < 5:
                continue

            preds = model.predict_expression(perturbed_seqs, cell_type=cell_type)
            pair_effects[(i, j)] = {
                'mean_pred': float(preds.mean()),
                'std_pred': float(preds.std()),
                'effect': float(preds.mean() - orig_pred),
            }

        # Now test higher-order: perturb triplets simultaneously
        if len(pair_effects) < 3:
            continue

        triplet_indices = list(combinations(motif_indices[:6], 3))
        if len(triplet_indices) > 5:
            selected = rng.choice(len(triplet_indices), 5, replace=False)
            triplet_indices = [triplet_indices[i] for i in selected]

        for i, j, k in triplet_indices:
            # Combined perturbation of all three
            combined_seqs = []
            for _ in range(n_perturbations):
                perturbed = _perturb_triplet(
                    seq, hits.iloc[i], hits.iloc[j], hits.iloc[k], rng, seq_gc
                )
                if perturbed and len(perturbed) == seq_len:
                    combined_seqs.append(perturbed)

            if len(combined_seqs) < 5:
                continue

            combined_preds = model.predict_expression(combined_seqs, cell_type=cell_type)
            combined_effect = float(combined_preds.mean() - orig_pred)

            # Sum of pairwise effects
            pairwise_sum = 0
            n_pairs_found = 0
            for pi, pj in [(i, j), (i, k), (j, k)]:
                key = (min(pi, pj), max(pi, pj))
                if key in pair_effects:
                    pairwise_sum += pair_effects[key]['effect']
                    n_pairs_found += 1

            if n_pairs_found < 2:
                continue

            # Interaction term
            interaction = combined_effect - pairwise_sum

            # Compositionality score: 1 - |interaction| / max(|combined|, epsilon)
            compositionality = 1 - abs(interaction) / max(abs(combined_effect), 1e-6)
            compositionality = max(min(compositionality, 1.0), 0.0)

            results.append({
                'seq_id': seq_id,
                'n_motifs': len(hits),
                'triplet': f'{i}_{j}_{k}',
                'original_pred': float(orig_pred),
                'combined_effect': combined_effect,
                'pairwise_sum': pairwise_sum,
                'interaction': interaction,
                'compositionality': compositionality,
                'n_pairs_found': n_pairs_found,
                'model': model.name,
            })

    return pd.DataFrame(results)


def summarize_compositionality_v2(results_df: pd.DataFrame) -> Dict:
    """Summarize the enhancer-specific compositionality results."""
    if len(results_df) == 0:
        return {'error': 'No results'}

    return {
        'n_tests': len(results_df),
        'n_enhancers': int(results_df['seq_id'].nunique()),
        'mean_compositionality': float(results_df['compositionality'].mean()),
        'median_compositionality': float(results_df['compositionality'].median()),
        'std_compositionality': float(results_df['compositionality'].std()),
        'mean_interaction': float(results_df['interaction'].mean()),
        'mean_abs_interaction': float(results_df['interaction'].abs().mean()),
        'mean_combined_effect': float(results_df['combined_effect'].abs().mean()),
        'mean_pairwise_sum': float(results_df['pairwise_sum'].abs().mean()),
        'frac_additive': float((results_df['compositionality'] > 0.8).mean()),
        'frac_nonadditive': float((results_df['compositionality'] < 0.3).mean()),
        'interaction_vs_combined_corr': float(
            results_df['interaction'].abs().corr(results_df['combined_effect'].abs())
        ),
    }


def _perturb_pair(seq, motif_a, motif_b, rng, gc):
    """Perturb the arrangement of a motif pair within a sequence."""
    try:
        start_a, end_a = int(motif_a['start']), int(motif_a['end'])
        start_b, end_b = int(motif_b['start']), int(motif_b['end'])

        seq_a = seq[start_a:end_a]
        seq_b = seq[start_b:end_b]

        if not seq_a or not seq_b:
            return None

        # Random perturbation: change spacing by ±1-10bp, possibly flip orientation
        new_spacing = max(2, abs(start_b - end_a) + rng.integers(-10, 11))

        # Possibly flip one motif
        if rng.random() < 0.5:
            seq_a = reverse_complement(seq_a)
        if rng.random() < 0.5:
            seq_b = reverse_complement(seq_b)

        # Rebuild the region
        if start_a < start_b:
            spacer = generate_neutral_spacer(new_spacing, gc=gc, rng=rng)
            new_region = seq_a + spacer + seq_b
            # Replace the region in the original sequence
            result = seq[:start_a] + new_region
            remainder_len = len(seq) - len(result)
            if remainder_len > 0:
                result += generate_neutral_spacer(remainder_len, gc=gc, rng=rng)
            elif remainder_len < 0:
                result = result[:len(seq)]
            return result
        else:
            return None  # Skip reversed pairs
    except Exception:
        return None


def _perturb_triplet(seq, motif_a, motif_b, motif_c, rng, gc):
    """Perturb the arrangement of three motifs within a sequence."""
    try:
        motifs = sorted(
            [(motif_a, 'a'), (motif_b, 'b'), (motif_c, 'c')],
            key=lambda x: int(x[0]['start'])
        )

        seqs = []
        for m, _ in motifs:
            s, e = int(m['start']), int(m['end'])
            seqs.append(seq[s:e])

        if any(not s for s in seqs):
            return None

        # Shuffle the order
        order = list(range(3))
        rng.shuffle(order)

        # Random spacings
        spacings = [max(2, rng.integers(2, 30)) for _ in range(2)]

        # Possibly flip orientations
        for i in range(3):
            if rng.random() < 0.5:
                seqs[order[i]] = reverse_complement(seqs[order[i]])

        # Build new arrangement
        start_pos = int(motifs[0][0]['start'])
        spacer1 = generate_neutral_spacer(spacings[0], gc=gc, rng=rng)
        spacer2 = generate_neutral_spacer(spacings[1], gc=gc, rng=rng)
        new_region = seqs[order[0]] + spacer1 + seqs[order[1]] + spacer2 + seqs[order[2]]

        result = seq[:start_pos] + new_region
        remainder = len(seq) - len(result)
        if remainder > 0:
            result += generate_neutral_spacer(remainder, gc=gc, rng=rng)
        elif remainder < 0:
            result = result[:len(seq)]

        return result
    except Exception:
        return None
