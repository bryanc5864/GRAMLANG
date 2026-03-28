"""
Spacer-Factored Grammar Sensitivity Index (SF-GSI)

The original GSI measures expression variance under vocab-preserving shuffles:
    GSI = σ_shuffle / |μ_shuffle|

However, v3 experiments showed that 78-86% of this variance comes from
spacer composition changes, not motif arrangement. SF-GSI decomposes this:

    SF-GSI = σ_grammar / |μ_ref|

Where σ_grammar is the residual variance after controlling for spacer effects.

Methods:
1. Motif-only shuffle: Replace spacers with fixed sequence, shuffle motifs
2. Regression residual: Regress out spacer GC/k-mer effects from predictions
3. Matched shuffles: Match spacer composition while permuting motifs
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings


@dataclass
class SFGSIResult:
    """Results from SF-GSI computation."""
    # Original GSI
    gsi: float
    gsi_se: float

    # Spacer-factored components
    sf_gsi: float  # Grammar-only sensitivity
    sf_gsi_se: float
    spacer_contribution: float  # Fraction of variance from spacers

    # Decomposition
    total_variance: float
    grammar_variance: float
    spacer_variance: float

    # Statistical tests
    grammar_pvalue: float
    n_shuffles: int
    n_sequences: int


def compute_gc_content(sequence: str) -> float:
    """Compute GC content of a sequence."""
    seq = sequence.upper()
    gc = seq.count('G') + seq.count('C')
    return gc / len(seq) if len(seq) > 0 else 0.5


def extract_spacers(sequence: str, motifs: List[Dict]) -> List[str]:
    """Extract spacer sequences between motifs."""
    if not motifs:
        return [sequence]

    sorted_motifs = sorted(motifs, key=lambda m: m['start'])
    spacers = []

    # Before first motif
    if sorted_motifs[0]['start'] > 0:
        spacers.append(sequence[:sorted_motifs[0]['start']])

    # Between motifs
    for i in range(len(sorted_motifs) - 1):
        start = sorted_motifs[i]['end']
        end = sorted_motifs[i + 1]['start']
        if end > start:
            spacers.append(sequence[start:end])

    # After last motif
    if sorted_motifs[-1]['end'] < len(sequence):
        spacers.append(sequence[sorted_motifs[-1]['end']:])

    return spacers


def compute_spacer_features(sequence: str, motifs: List[Dict]) -> np.ndarray:
    """
    Compute spacer composition features for regression.

    Features:
    - Total spacer GC content
    - Mean spacer length
    - GC variance across spacers
    - Dinucleotide frequencies in spacers
    """
    spacers = extract_spacers(sequence, motifs)

    if not spacers or sum(len(s) for s in spacers) == 0:
        return np.zeros(20)  # 20 features

    features = []

    # GC content
    total_spacer = ''.join(spacers)
    features.append(compute_gc_content(total_spacer))

    # Mean spacer length (normalized)
    features.append(np.mean([len(s) for s in spacers]) / 50)  # Normalize by 50bp

    # GC variance
    gc_values = [compute_gc_content(s) for s in spacers if len(s) > 0]
    features.append(np.var(gc_values) if len(gc_values) > 1 else 0)

    # Dinucleotide frequencies (16 features)
    dinucs = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
              'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    total_dinucs = len(total_spacer) - 1
    for dinuc in dinucs:
        count = total_spacer.upper().count(dinuc)
        features.append(count / total_dinucs if total_dinucs > 0 else 0)

    # Pad to 20 features
    while len(features) < 20:
        features.append(0)

    return np.array(features[:20])


def motif_only_shuffle(
    sequence: str,
    motifs: List[Dict],
    n_shuffles: int = 100,
    fixed_spacer: str = 'N'
) -> List[Tuple[str, List[Dict]]]:
    """
    Shuffle motif positions while replacing spacers with fixed sequence.

    This isolates the grammar effect by removing spacer composition variation.
    """
    if not motifs or len(motifs) < 2:
        return [(sequence, motifs)]

    # Extract motif sequences
    motif_seqs = []
    for m in sorted(motifs, key=lambda x: x['start']):
        motif_seqs.append({
            'seq': sequence[m['start']:m['end']],
            'name': m.get('motif_name', 'unknown'),
            'strand': m.get('strand', '+'),
            'length': m['end'] - m['start']
        })

    shuffled = []
    rng = np.random.default_rng(42)

    for _ in range(n_shuffles):
        # Shuffle motif order
        order = rng.permutation(len(motif_seqs))

        # Reconstruct sequence with fixed spacers
        new_seq = []
        new_motifs = []
        pos = 0

        # Fixed spacer length (average of original spacers)
        original_spacers = extract_spacers(sequence, motifs)
        avg_spacer_len = int(np.mean([len(s) for s in original_spacers])) if original_spacers else 10

        for idx in order:
            m = motif_seqs[idx]

            # Add spacer
            if pos > 0:
                spacer = fixed_spacer * avg_spacer_len
                new_seq.append(spacer)
                pos += avg_spacer_len

            # Add motif
            new_motifs.append({
                'start': pos,
                'end': pos + m['length'],
                'motif_name': m['name'],
                'strand': m['strand']
            })
            new_seq.append(m['seq'])
            pos += m['length']

        shuffled.append((''.join(new_seq), new_motifs))

    return shuffled


def matched_spacer_shuffle(
    sequence: str,
    motifs: List[Dict],
    n_shuffles: int = 100,
    gc_tolerance: float = 0.05
) -> List[Tuple[str, List[Dict]]]:
    """
    Shuffle motifs while matching spacer GC content.

    This controls for spacer composition while varying motif arrangement.
    """
    if not motifs or len(motifs) < 2:
        return [(sequence, motifs)]

    original_spacers = extract_spacers(sequence, motifs)
    original_gc = compute_gc_content(''.join(original_spacers))

    # Extract motif info
    sorted_motifs = sorted(motifs, key=lambda x: x['start'])
    motif_seqs = [sequence[m['start']:m['end']] for m in sorted_motifs]

    shuffled = []
    rng = np.random.default_rng(42)
    attempts = 0
    max_attempts = n_shuffles * 10

    while len(shuffled) < n_shuffles and attempts < max_attempts:
        attempts += 1

        # Shuffle motif order
        order = rng.permutation(len(motif_seqs))

        # Reconstruct with original spacers in new positions
        new_seq_parts = []
        new_motifs = []
        pos = 0

        for i, idx in enumerate(order):
            # Add spacer (use original spacers cyclically)
            if i < len(original_spacers):
                spacer = original_spacers[i]
            else:
                spacer = original_spacers[i % len(original_spacers)] if original_spacers else ''

            if pos > 0 and spacer:
                new_seq_parts.append(spacer)
                pos += len(spacer)

            # Add motif
            m = sorted_motifs[idx]
            new_motifs.append({
                'start': pos,
                'end': pos + len(motif_seqs[idx]),
                'motif_name': m.get('motif_name', 'unknown'),
                'strand': m.get('strand', '+')
            })
            new_seq_parts.append(motif_seqs[idx])
            pos += len(motif_seqs[idx])

        new_seq = ''.join(new_seq_parts)
        new_spacers = extract_spacers(new_seq, new_motifs)
        new_gc = compute_gc_content(''.join(new_spacers))

        # Check GC tolerance
        if abs(new_gc - original_gc) <= gc_tolerance:
            shuffled.append((new_seq, new_motifs))

    if len(shuffled) < n_shuffles:
        warnings.warn(f"Only generated {len(shuffled)}/{n_shuffles} matched shuffles")

    return shuffled if shuffled else [(sequence, motifs)]


def compute_sf_gsi(
    sequences: List[str],
    motif_annotations: List[List[Dict]],
    predict_fn,  # Function: (sequences, motifs) -> predictions
    n_shuffles: int = 100,
    method: str = 'matched',  # 'motif_only', 'matched', 'regression'
    device: str = 'cuda'
) -> SFGSIResult:
    """
    Compute Spacer-Factored Grammar Sensitivity Index.

    Args:
        sequences: List of DNA sequences
        motif_annotations: List of motif annotation lists
        predict_fn: Function that takes (sequences, motifs) and returns predictions
        n_shuffles: Number of shuffles per sequence
        method: 'motif_only', 'matched', or 'regression'
        device: Device for computation

    Returns:
        SFGSIResult with decomposed sensitivity metrics
    """
    n_seqs = len(sequences)

    # Get original predictions
    with torch.no_grad():
        original_preds = predict_fn(sequences, motif_annotations)
        if isinstance(original_preds, torch.Tensor):
            original_preds = original_preds.cpu().numpy()

    # Compute standard GSI (full shuffles)
    all_shuffle_preds = []
    for seq, motifs in zip(sequences, motif_annotations):
        if len(motifs) < 2:
            continue

        # Standard vocab-preserving shuffle
        shuffled = matched_spacer_shuffle(seq, motifs, n_shuffles, gc_tolerance=1.0)

        shuffle_seqs = [s[0] for s in shuffled]
        shuffle_motifs = [s[1] for s in shuffled]

        with torch.no_grad():
            preds = predict_fn(shuffle_seqs, shuffle_motifs)
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().numpy()
        all_shuffle_preds.append(preds)

    if not all_shuffle_preds:
        return SFGSIResult(
            gsi=0.0, gsi_se=0.0, sf_gsi=0.0, sf_gsi_se=0.0,
            spacer_contribution=0.0, total_variance=0.0,
            grammar_variance=0.0, spacer_variance=0.0,
            grammar_pvalue=1.0, n_shuffles=0, n_sequences=0
        )

    # Standard GSI
    all_preds_flat = np.concatenate(all_shuffle_preds)
    total_variance = np.var(all_preds_flat)
    mean_pred = np.mean(all_preds_flat)
    gsi = np.std(all_preds_flat) / abs(mean_pred) if abs(mean_pred) > 1e-8 else 0.0
    gsi_se = gsi / np.sqrt(2 * len(all_preds_flat))

    # Compute SF-GSI based on method
    if method == 'motif_only':
        # Use fixed spacers to isolate grammar effect
        grammar_preds = []
        for seq, motifs in zip(sequences, motif_annotations):
            if len(motifs) < 2:
                continue
            shuffled = motif_only_shuffle(seq, motifs, n_shuffles)
            shuffle_seqs = [s[0] for s in shuffled]
            shuffle_motifs = [s[1] for s in shuffled]

            with torch.no_grad():
                preds = predict_fn(shuffle_seqs, shuffle_motifs)
                if isinstance(preds, torch.Tensor):
                    preds = preds.cpu().numpy()
            grammar_preds.append(preds)

        if grammar_preds:
            grammar_preds_flat = np.concatenate(grammar_preds)
            grammar_variance = np.var(grammar_preds_flat)
            sf_gsi = np.std(grammar_preds_flat) / abs(mean_pred) if abs(mean_pred) > 1e-8 else 0.0
        else:
            grammar_variance = 0.0
            sf_gsi = 0.0

    elif method == 'matched':
        # Use GC-matched shuffles
        grammar_preds = []
        for seq, motifs in zip(sequences, motif_annotations):
            if len(motifs) < 2:
                continue
            shuffled = matched_spacer_shuffle(seq, motifs, n_shuffles, gc_tolerance=0.02)
            shuffle_seqs = [s[0] for s in shuffled]
            shuffle_motifs = [s[1] for s in shuffled]

            with torch.no_grad():
                preds = predict_fn(shuffle_seqs, shuffle_motifs)
                if isinstance(preds, torch.Tensor):
                    preds = preds.cpu().numpy()
            grammar_preds.append(preds)

        if grammar_preds:
            grammar_preds_flat = np.concatenate(grammar_preds)
            grammar_variance = np.var(grammar_preds_flat)
            sf_gsi = np.std(grammar_preds_flat) / abs(mean_pred) if abs(mean_pred) > 1e-8 else 0.0
        else:
            grammar_variance = 0.0
            sf_gsi = 0.0

    elif method == 'regression':
        # Regress out spacer effects
        spacer_features = []
        for seq, motifs in zip(sequences, motif_annotations):
            spacer_features.append(compute_spacer_features(seq, motifs))
        spacer_features = np.array(spacer_features)

        # Fit regression on shuffled predictions
        from sklearn.linear_model import Ridge

        # Collect all shuffle predictions with spacer features
        X_all = []
        y_all = []
        for i, (seq, motifs) in enumerate(zip(sequences, motif_annotations)):
            if len(motifs) < 2:
                continue
            shuffled = matched_spacer_shuffle(seq, motifs, n_shuffles, gc_tolerance=1.0)

            for shuf_seq, shuf_motifs in shuffled:
                X_all.append(compute_spacer_features(shuf_seq, shuf_motifs))

            shuffle_seqs = [s[0] for s in shuffled]
            shuffle_motifs = [s[1] for s in shuffled]
            with torch.no_grad():
                preds = predict_fn(shuffle_seqs, shuffle_motifs)
                if isinstance(preds, torch.Tensor):
                    preds = preds.cpu().numpy()
            y_all.extend(preds)

        if len(X_all) > 20:
            X_all = np.array(X_all)
            y_all = np.array(y_all)

            # Fit spacer regression
            reg = Ridge(alpha=1.0)
            reg.fit(X_all, y_all)

            # Residual variance = grammar variance
            residuals = y_all - reg.predict(X_all)
            grammar_variance = np.var(residuals)
            sf_gsi = np.std(residuals) / abs(mean_pred) if abs(mean_pred) > 1e-8 else 0.0
        else:
            grammar_variance = total_variance
            sf_gsi = gsi
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute spacer contribution
    spacer_variance = total_variance - grammar_variance
    spacer_contribution = spacer_variance / total_variance if total_variance > 1e-8 else 0.0
    spacer_contribution = max(0, min(1, spacer_contribution))  # Clamp to [0, 1]

    # Statistical test: is grammar variance significant?
    # Use permutation test
    if grammar_variance > 0:
        # F-test approximation
        f_stat = grammar_variance / (total_variance / n_shuffles) if total_variance > 0 else 0
        grammar_pvalue = 1 - stats.f.cdf(f_stat, n_shuffles - 1, n_shuffles * n_seqs - 1)
    else:
        grammar_pvalue = 1.0

    sf_gsi_se = sf_gsi / np.sqrt(2 * len(all_preds_flat)) if len(all_preds_flat) > 0 else 0.0

    return SFGSIResult(
        gsi=gsi,
        gsi_se=gsi_se,
        sf_gsi=sf_gsi,
        sf_gsi_se=sf_gsi_se,
        spacer_contribution=spacer_contribution,
        total_variance=total_variance,
        grammar_variance=grammar_variance,
        spacer_variance=spacer_variance,
        grammar_pvalue=grammar_pvalue,
        n_shuffles=n_shuffles,
        n_sequences=n_seqs
    )


def compute_sf_gsi_with_sfgn(
    model,  # SFGN model
    sequences: List[str],
    motif_annotations: List[List[Dict]],
    n_shuffles: int = 100,
) -> Dict:
    """
    Compute SF-GSI using SFGN's built-in decomposition.

    SFGN provides α (grammar weight) and β (composition weight) directly,
    allowing us to compute grammar-specific sensitivity.
    """
    model.eval()

    # Get original decomposition
    with torch.no_grad():
        output = model(sequences, motif_annotations)
        original_grammar = output.grammar_vector.cpu().numpy()
        original_comp = output.composition_vector.cpu().numpy()
        original_alpha = output.alpha.cpu().numpy()
        original_pred = output.prediction.cpu().numpy()

    # Shuffle and track grammar vs composition contributions
    grammar_preds = []
    comp_preds = []
    total_preds = []

    for seq, motifs in zip(sequences, motif_annotations):
        if len(motifs) < 2:
            continue

        shuffled = matched_spacer_shuffle(seq, motifs, n_shuffles, gc_tolerance=0.02)

        for shuf_seq, shuf_motifs in shuffled:
            with torch.no_grad():
                output = model([shuf_seq], [shuf_motifs])

                # Grammar-only prediction (α * grammar_transformed)
                g_vec = output.grammar_vector
                alpha = output.alpha

                grammar_preds.append(alpha.item())
                total_preds.append(output.prediction.item())

    if not total_preds:
        return {
            'sf_gsi': 0.0,
            'gsi': 0.0,
            'mean_alpha': 0.0,
            'alpha_variance': 0.0,
            'n_samples': 0
        }

    # Compute metrics
    total_preds = np.array(total_preds)
    grammar_preds = np.array(grammar_preds)

    gsi = np.std(total_preds) / abs(np.mean(total_preds)) if abs(np.mean(total_preds)) > 1e-8 else 0.0

    # SF-GSI from SFGN: variance explained by grammar pathway
    # Approximate by α * total_variance
    mean_alpha = np.mean(grammar_preds)
    sf_gsi = gsi * mean_alpha  # Grammar's contribution to sensitivity

    return {
        'sf_gsi': sf_gsi,
        'gsi': gsi,
        'mean_alpha': mean_alpha,
        'alpha_variance': np.var(grammar_preds),
        'n_samples': len(total_preds),
        'spacer_contribution': 1 - mean_alpha
    }
