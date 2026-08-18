"""
Vocabulary-preserving perturbations.

Shuffled enhancers that keep the motif multiset but change the arrangement.
v3 added factorial variants so sensitivity can be split into position,
orientation and spacer components.
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
    Shuffles that keep the motif vocabulary and change spacing, order and
    orientation. Motif positions are reassigned at random and the gaps are
    filled with dinucleotide-shuffled spacer DNA, so the result is the same
    length with the same motifs in a different arrangement.
    """
    rng = np.random.default_rng(seed)
    motifs = motif_annotations.get('motifs', [])
    seq_len = len(sequence)

    if len(motifs) < 2:
        # nothing to rearrange, fall back to a dinucleotide shuffle
        return [dinucleotide_shuffle(sequence, rng=np.random.default_rng(rng.integers(1e9)))
                for _ in range(n_shuffles)]

        # sort and merge overlaps
    motifs_sorted = sorted(motifs, key=lambda m: m['start'])
    merged = merge_overlapping_motifs(motifs_sorted)

    motif_seqs = []
    for m in merged:
        motif_seqs.append({
            'seq': sequence[m['start']:m['end']],
            'name': m.get('motif_name', 'unknown'),
            'strand': m.get('strand', '+'),
            'length': m['end'] - m['start']
        })

        # spacer content: between motifs plus the flanks
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

    # spacer GC, for generating neutral filler
    gc = sum(1 for c in all_spacer_text.upper() if c in 'GC') / max(len(all_spacer_text), 1)

    shuffled = []
    for _ in range(n_shuffles):
        # permute motif order
        perm = rng.permutation(len(motif_seqs))
        permuted = [motif_seqs[i].copy() for i in perm]

        # flip orientation, 50% per motif
        for m in permuted:
            if rng.random() < 0.5:
                m['seq'] = reverse_complement(m['seq'])

        # random spacer distribution
        n_spacers = len(permuted) + 1
        spacer_lens = random_partition(
            max(total_spacer_len, n_spacers * 2),
            n_spacers, min_len=2, rng=rng
        )

        # adjust so the total length still matches
        current_total = sum(spacer_lens) + total_motif_len
        if current_total != seq_len:
            diff = seq_len - current_total
            spacer_lens[-1] = max(spacer_lens[-1] + diff, 1)

        spacer_dna = dinucleotide_shuffle(all_spacer_text,
                                          rng=np.random.default_rng(rng.integers(1e9)))

        # assemble
        new_seq = ''
        sp_pos = 0
        for i, motif in enumerate(permuted):
            sp_len = spacer_lens[i]
            new_seq += spacer_dna[sp_pos:sp_pos + sp_len]
            sp_pos += sp_len
            new_seq += motif['seq']
        # final spacer
        final_len = spacer_lens[-1]
        new_seq += spacer_dna[sp_pos:sp_pos + final_len]

        # force exact length
        if len(new_seq) > seq_len:
            new_seq = new_seq[:seq_len]
        elif len(new_seq) < seq_len:
            pad = generate_neutral_spacer(seq_len - len(new_seq), gc=gc, rng=rng)
            new_seq += pad

        shuffled.append(new_seq)

    return shuffled


def generate_position_only_shuffles(
    sequence: str,
    motif_annotations: dict,
    n_shuffles: int = 100,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Shuffle motif positions only. Orientations are kept and the spacer DNA
    comes from the original sequence, shifted rather than reshuffled.
    """
    rng = np.random.default_rng(seed)
    motifs = motif_annotations.get('motifs', [])
    seq_len = len(sequence)

    if len(motifs) < 2:
        return [sequence] * n_shuffles

    motifs_sorted = sorted(motifs, key=lambda m: m['start'])
    merged = merge_overlapping_motifs(motifs_sorted)

    motif_seqs = []
    for m in merged:
        motif_seqs.append({
            'seq': sequence[m['start']:m['end']],
            'name': m.get('motif_name', 'unknown'),
            'strand': m.get('strand', '+'),
            'length': m['end'] - m['start']
        })

    # keep the original spacer DNA
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
    gc = sum(1 for c in all_spacer_text.upper() if c in 'GC') / max(len(all_spacer_text), 1)

    shuffled = []
    for _ in range(n_shuffles):
        # permute order but keep each motif's strand
        perm = rng.permutation(len(motif_seqs))
        permuted = [motif_seqs[i].copy() for i in perm]
        # no orientation flip: this is the difference from the full shuffle

        n_spacers = len(permuted) + 1
        spacer_lens = random_partition(max(total_spacer_len, n_spacers * 2),
                                        n_spacers, min_len=2, rng=rng)
        current_total = sum(spacer_lens) + total_motif_len
        if current_total != seq_len:
            diff = seq_len - current_total
            spacer_lens[-1] = max(spacer_lens[-1] + diff, 1)

        # original spacer DNA, not reshuffled
        new_seq = ''
        sp_pos = 0
        for k, motif in enumerate(permuted):
            sp_len = spacer_lens[k]
            new_seq += all_spacer_text[sp_pos:sp_pos + sp_len]
            sp_pos += sp_len
            new_seq += motif['seq']
        final_len = spacer_lens[-1]
        new_seq += all_spacer_text[sp_pos:sp_pos + final_len]

        if len(new_seq) > seq_len:
            new_seq = new_seq[:seq_len]
        elif len(new_seq) < seq_len:
            pad = generate_neutral_spacer(seq_len - len(new_seq), gc=gc, rng=rng)
            new_seq += pad

        shuffled.append(new_seq)

    return shuffled


def generate_orientation_only_shuffles(
    sequence: str,
    motif_annotations: dict,
    n_shuffles: int = 100,
    seed: Optional[int] = None,
) -> List[str]:
    """Flip motif orientations only; positions and spacer DNA stay put."""
    rng = np.random.default_rng(seed)
    motifs = motif_annotations.get('motifs', [])
    seq_len = len(sequence)

    if len(motifs) < 2:
        return [sequence] * n_shuffles

    motifs_sorted = sorted(motifs, key=lambda m: m['start'])
    merged = merge_overlapping_motifs(motifs_sorted)

    shuffled = []
    for _ in range(n_shuffles):
        new_seq = list(sequence)
        for m in merged:
            if rng.random() < 0.5:
                motif_seq = sequence[m['start']:m['end']]
                rc = reverse_complement(motif_seq)
                for idx, base in enumerate(rc):
                    new_seq[m['start'] + idx] = base
        shuffled.append(''.join(new_seq))

    return shuffled


def generate_spacer_only_shuffles(
    sequence: str,
    motif_annotations: dict,
    n_shuffles: int = 100,
    seed: Optional[int] = None,
) -> List[str]:
    """Dinucleotide-shuffle the spacers only; motifs stay exactly where they are."""
    rng = np.random.default_rng(seed)
    motifs = motif_annotations.get('motifs', [])
    seq_len = len(sequence)

    if len(motifs) < 2:
        return [dinucleotide_shuffle(sequence, rng=np.random.default_rng(rng.integers(1e9)))
                for _ in range(n_shuffles)]

    motifs_sorted = sorted(motifs, key=lambda m: m['start'])
    merged = merge_overlapping_motifs(motifs_sorted)

    spacer_regions = []
    prev_end = 0
    for m in merged:
        if m['start'] > prev_end:
            spacer_regions.append((prev_end, m['start']))
        prev_end = m['end']
    if prev_end < seq_len:
        spacer_regions.append((prev_end, seq_len))

    shuffled = []
    for _ in range(n_shuffles):
        new_seq = list(sequence)
        for start, end in spacer_regions:
            spacer = sequence[start:end]
            shuffled_spacer = dinucleotide_shuffle(
                spacer, rng=np.random.default_rng(rng.integers(1e9))
            )
            for idx, base in enumerate(shuffled_spacer):
                if start + idx < seq_len:
                    new_seq[start + idx] = base
        shuffled.append(''.join(new_seq))

    return shuffled
