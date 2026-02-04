"""DNA sequence utilities for GRAMLANG."""

import numpy as np
from typing import List, Optional


# Complement mapping
_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
               'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'}


def reverse_complement(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    return ''.join(_COMPLEMENT.get(c, 'N') for c in reversed(seq))


def one_hot_encode(seq: str) -> np.ndarray:
    """
    One-hot encode DNA sequence.

    Args:
        seq: DNA string (ACGT/N)

    Returns:
        np.ndarray of shape (len(seq), 4) with ACGT encoding
    """
    mapping = {
        'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1],
        'N': [0.25, 0.25, 0.25, 0.25]
    }
    return np.array([mapping.get(c.upper(), mapping['N']) for c in seq], dtype=np.float32)


def decode_one_hot(arr: np.ndarray) -> str:
    """Decode one-hot encoded array back to DNA string."""
    bases = ['A', 'C', 'G', 'T']
    return ''.join(bases[i] for i in arr.argmax(axis=1))


def gc_content(seq: str) -> float:
    """Compute GC content of sequence."""
    seq = seq.upper()
    gc = seq.count('G') + seq.count('C')
    total = len(seq) - seq.count('N')
    return gc / max(total, 1)


def dinucleotide_frequencies(seq: str) -> dict:
    """Compute dinucleotide frequencies."""
    seq = seq.upper()
    dinucs = {}
    total = max(len(seq) - 1, 1)
    for i in range(len(seq) - 1):
        dn = seq[i:i+2]
        dinucs[dn] = dinucs.get(dn, 0) + 1
    return {k: v / total for k, v in dinucs.items()}


def dinucleotide_shuffle(seq: str, rng: Optional[np.random.Generator] = None) -> str:
    """
    Shuffle sequence preserving dinucleotide frequencies.
    Uses the Altschul-Erickson algorithm via an Euler path approach.

    Args:
        seq: DNA sequence to shuffle
        rng: Optional random number generator

    Returns:
        Shuffled sequence with same dinucleotide composition
    """
    if rng is None:
        rng = np.random.default_rng()

    seq = seq.upper()
    if len(seq) <= 2:
        return seq

    # Build the dinucleotide graph
    from collections import defaultdict
    edges = defaultdict(list)
    for i in range(len(seq) - 1):
        edges[seq[i]].append(seq[i + 1])

    # Shuffle edge lists
    for key in edges:
        rng.shuffle(edges[key])

    # Construct Euler path
    result = [seq[0]]
    idx = defaultdict(int)
    current = seq[0]

    for _ in range(len(seq) - 1):
        if idx[current] < len(edges[current]):
            nxt = edges[current][idx[current]]
            idx[current] += 1
            result.append(nxt)
            current = nxt
        else:
            # Fallback
            result.append(rng.choice(['A', 'C', 'G', 'T']))
            current = result[-1]

    return ''.join(result)


def generate_neutral_spacer(length: int, gc: float = 0.5,
                            rng: Optional[np.random.Generator] = None) -> str:
    """Generate neutral spacer DNA with given GC content."""
    if length <= 0:
        return ''
    if rng is None:
        rng = np.random.default_rng()
    bases = []
    for _ in range(length):
        if rng.random() < gc:
            bases.append(rng.choice(['G', 'C']))
        else:
            bases.append(rng.choice(['A', 'T']))
    return ''.join(bases)


def pad_sequence(seq: str, target_len: int, seed: Optional[int] = None) -> str:
    """
    Pad a short sequence to target length with deterministic flanking DNA.

    Centers the sequence and adds random DNA on both sides.
    Uses seed based on sequence hash for reproducibility.
    """
    if len(seq) >= target_len:
        # Center-crop if too long
        start = (len(seq) - target_len) // 2
        return seq[start:start + target_len]

    if seed is None:
        seed = hash(seq) % (2**32)
    rng = np.random.default_rng(seed)

    pad_total = target_len - len(seq)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    gc = gc_content(seq)
    left = generate_neutral_spacer(pad_left, gc=gc, rng=rng)
    right = generate_neutral_spacer(pad_right, gc=gc, rng=rng)

    return left + seq + right


def random_partition(total: int, n_parts: int, min_len: int = 1,
                     rng: Optional[np.random.Generator] = None) -> List[int]:
    """
    Randomly partition total into n_parts integers, each >= min_len.
    """
    if rng is None:
        rng = np.random.default_rng()

    if total < n_parts * min_len:
        # Not enough — distribute evenly
        base = total // n_parts
        parts = [base] * n_parts
        remainder = total - base * n_parts
        for i in range(remainder):
            parts[i] += 1
        return parts

    # Stars and bars: place n_parts-1 dividers in total-n_parts*min_len slots
    remainder = total - n_parts * min_len
    if remainder == 0:
        return [min_len] * n_parts

    # Generate random breaks
    breaks = sorted(rng.choice(range(1, remainder + n_parts), size=n_parts - 1, replace=False))
    breaks = [0] + list(breaks) + [remainder + n_parts - 1]
    parts = [breaks[i+1] - breaks[i] for i in range(n_parts)]

    # Add minimum
    parts = [p + min_len - 1 for p in parts]

    # Fix sum
    diff = total - sum(parts)
    parts[-1] += diff

    # Ensure no negatives
    for i in range(len(parts)):
        if parts[i] < min_len:
            deficit = min_len - parts[i]
            parts[i] = min_len
            parts[-1] -= deficit

    return parts
