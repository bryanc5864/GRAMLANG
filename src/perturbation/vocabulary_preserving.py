"""
Vocabulary-preserving perturbations for grammar analysis.

Generates shuffled versions of enhancer sequences that preserve
motif content (vocabulary) while altering arrangement (syntax).
"""

import numpy as np
from typing import List, Optional
from src.utils.sequence import (
    reverse_complement, dinucleotide_shuffle,
    generate_neutral_spacer, random_partition
)


def merge_overlapping_motifs(motifs_sorted: list) -> list:
    """Merge overlapping motif annotations."""
    if not motifs_sorted:
        return []

    merged = [motifs_sorted[0].copy()]
    for m in motifs_sorted[1:]:
        if m['start'] < merged[-1]['end']:
            merged[-1]['end'] = max(merged[-1]['end'], m['end'])
            prev_name = merged[-1].get('motif_name', '')
            curr_name = m.get('motif_name', '')
            merged[-1]['motif_name'] = f"{prev_name}+{curr_name}"
        else:
            merged.append(m.copy())

    return merged


def generate_vocabulary_preserving_shuffles(
    sequence: str,
    motif_annotations: dict,
    n_shuffles: int = 100,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Generate shuffled versions preserving motif vocabulary but
    altering arrangement (spacing, ordering, orientation).

    Strategy:
    1. Identify motif instances and their positions
    2. Extract motif sequences and inter-motif spacers
    3. Randomly reassign motif positions/order
    4. Fill gaps with dinucleotide-shuffled spacer DNA

    Args:
        sequence: Original DNA sequence
        motif_annotations: Dict with 'motifs' list
        n_shuffles: Number of shuffled variants
        seed: Random seed

    Returns:
        List of shuffled sequences (same length, same motifs, different arrangement)
    """
    rng = np.random.default_rng(seed)
    motifs = motif_annotations.get('motifs', [])
    seq_len = len(sequence)

    if len(motifs) < 2:
        # Can't shuffle arrangement — do dinucleotide shuffle instead
        return [dinucleotide_shuffle(sequence, rng=np.random.default_rng(rng.integers(1e9)))
                for _ in range(n_shuffles)]

    # Sort and merge overlapping motifs
    motifs_sorted = sorted(motifs, key=lambda m: m['start'])
    merged = merge_overlapping_motifs(motifs_sorted)

    # Extract motif sequences
    motif_seqs = []
    for m in merged:
        motif_seqs.append({
            'seq': sequence[m['start']:m['end']],
            'name': m.get('motif_name', 'unknown'),
            'strand': m.get('strand', '+'),
            'length': m['end'] - m['start']
        })

    # Extract spacer content (between motifs + flanks)
    spacers = []
    prev_end = 0
    for m in merged:
        if m['start'] > prev_end:
            spacers.append(sequence[prev_end:m['start']])
        prev_end = m['end']
    if prev_end < seq_len:
        spacers.append(sequence[prev_end:seq_len])

    all_spacer_text = ''.join(spacers)
    total_spacer_len = len(all_spacer_text)
    total_motif_len = sum(m['length'] for m in motif_seqs)

    # Compute GC content of spacers for neutral generation
    gc = sum(1 for c in all_spacer_text.upper() if c in 'GC') / max(len(all_spacer_text), 1)

    shuffled = []
    for _ in range(n_shuffles):
        # 1. Random permutation of motif order
        perm = rng.permutation(len(motif_seqs))
        permuted = [motif_seqs[i].copy() for i in perm]

        # 2. Randomly flip orientation (50% chance per motif)
        for m in permuted:
            if rng.random() < 0.5:
                m['seq'] = reverse_complement(m['seq'])

        # 3. Random spacer distribution
        n_spacers = len(permuted) + 1
        spacer_lens = random_partition(
            max(total_spacer_len, n_spacers * 2),
            n_spacers, min_len=2, rng=rng
        )

        # Adjust to ensure total length is correct
        current_total = sum(spacer_lens) + total_motif_len
        if current_total != seq_len:
            diff = seq_len - current_total
            spacer_lens[-1] = max(spacer_lens[-1] + diff, 1)

        # 4. Generate spacer sequences
        spacer_dna = dinucleotide_shuffle(all_spacer_text,
                                          rng=np.random.default_rng(rng.integers(1e9)))

        # 5. Assemble
        new_seq = ''
        sp_pos = 0
        for i, motif in enumerate(permuted):
            sp_len = spacer_lens[i]
            new_seq += spacer_dna[sp_pos:sp_pos + sp_len]
            sp_pos += sp_len
            new_seq += motif['seq']
        # Final spacer
        final_len = spacer_lens[-1]
        new_seq += spacer_dna[sp_pos:sp_pos + final_len]

        # Ensure exact length
        if len(new_seq) > seq_len:
            new_seq = new_seq[:seq_len]
        elif len(new_seq) < seq_len:
            pad = generate_neutral_spacer(seq_len - len(new_seq), gc=gc, rng=rng)
            new_seq += pad

        shuffled.append(new_seq)

    return shuffled
