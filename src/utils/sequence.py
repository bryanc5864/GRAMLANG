"""DNA sequence utilities for GRAMLANG."""

import numpy as np
from typing import List, Optional


# complement map
_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N',
               'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'n': 'n'}


def reverse_complement(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    return ''.join(_COMPLEMENT.get(c, 'N') for c in reversed(seq))


def one_hot_encode(seq: str) -> np.ndarray:
    """One-hot encode DNA (ACGT/N) to (len(seq), 4)."""
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
    Shuffle preserving dinucleotide frequencies, Altschul-Erickson via an
    Euler path.
    """
    if rng is None:
        rng = np.random.default_rng()

    seq = seq.upper()
    if len(seq) <= 2:
        return seq

    # dinucleotide graph
    from collections import defaultdict
    edges = defaultdict(list)
    for i in range(len(seq) - 1):
        edges[seq[i]].append(seq[i + 1])

    # shuffle the edge lists
    for key in edges:
        rng.shuffle(edges[key])

    # euler path
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
            # fallback
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
    Pad a short sequence to target length with flanking DNA, centred. The
    flanks are seeded off the sequence hash so padding is reproducible.
    """
    if len(seq) >= target_len:
        # center-crop if too long
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
        # not enough room, distribute evenly
        base = total // n_parts
        parts = [base] * n_parts
        remainder = total - base * n_parts
        for i in range(remainder):
            parts[i] += 1
        return parts

    # stars and bars: n_parts-1 dividers over total - n_parts*min_len slots
    remainder = total - n_parts * min_len
    if remainder == 0:
        return [min_len] * n_parts

    breaks = sorted(rng.choice(range(1, remainder + n_parts), size=n_parts - 1, replace=False))
    breaks = [0] + list(breaks) + [remainder + n_parts - 1]
    parts = [breaks[i+1] - breaks[i] for i in range(n_parts)]

    # add the minimum back
    parts = [p + min_len - 1 for p in parts]

    # fix the sum
    diff = total - sum(parts)
    parts[-1] += diff

    # no negatives
    for i in range(len(parts)):
        if parts[i] < min_len:
            deficit = min_len - parts[i]
            parts[i] = min_len
            parts[-1] -= deficit

    return parts
