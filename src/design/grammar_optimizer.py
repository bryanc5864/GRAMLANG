"""Grammar-aware sequence design."""

import numpy as np
from typing import List, Dict, Optional
from src.utils.sequence import generate_neutral_spacer, reverse_complement, random_partition


def design_grammar_optimized_enhancer(
    motif_set: List[str],
    grammar_rules: dict,
    model,
    sequence_length: int = 200,
    n_candidates: int = 1000,
    cell_type: str = None,
    optimization_target: str = 'grammar',
    seed: int = 42,
) -> dict:
    """
    Design a synthetic enhancer optimized for grammar.

    Three strategies:
    - 'expression': Optimize predicted expression via random search
    - 'grammar': Fix motifs, optimize arrangement using grammar rules
    - 'joint': Optimize both

    Returns:
        Dict with best sequence, predicted expression, grammar score
    """
    rng = np.random.default_rng(seed)

    if optimization_target == 'grammar':
        candidates = _generate_grammar_optimized(
            motif_set, grammar_rules, sequence_length, n_candidates, rng
        )
    elif optimization_target == 'expression':
        candidates = _generate_random(
            motif_set, sequence_length, n_candidates, rng
        )
    else:  # joint
        half = n_candidates // 2
        c1 = _generate_grammar_optimized(
            motif_set, grammar_rules, sequence_length, half, rng
        )
        c2 = _generate_random(
            motif_set, sequence_length, half, rng
        )
        candidates = c1 + c2

    if not candidates:
        return {'error': 'Failed to generate candidates'}

    # Score all candidates
    expressions = model.predict_expression(candidates, cell_type=cell_type)
    best_idx = np.argmax(expressions)

    return {
        'best_sequence': candidates[best_idx],
        'predicted_expression': float(expressions[best_idx]),
        'mean_expression': float(np.mean(expressions)),
        'std_expression': float(np.std(expressions)),
        'optimization_target': optimization_target,
        'n_candidates': len(candidates),
        'n_motifs': len(motif_set),
    }


def _generate_grammar_optimized(motif_set, grammar_rules, seq_len, n, rng):
    """Generate candidates using grammar-informed spacing."""
    candidates = []
    for _ in range(n):
        perm = rng.permutation(len(motif_set))
        motifs = [motif_set[i] for i in perm]

        parts = []
        total_motif_len = sum(len(m) for m in motifs)
        remaining = max(seq_len - total_motif_len, len(motifs) + 1)

        for i, motif in enumerate(motifs):
            if i > 0:
                # Look up optimal spacing
                pair = f"{motifs[i-1][:6]}_{motif[:6]}"
                if pair in grammar_rules:
                    sp = grammar_rules[pair].get('optimal_spacing', 10)
                else:
                    sp = rng.integers(5, 25)
                sp = min(sp, remaining - (len(motifs) - i))
                sp = max(sp, 2)
                remaining -= sp
                parts.append(generate_neutral_spacer(sp, gc=0.5, rng=rng))

            # Random orientation
            if rng.random() < 0.5:
                parts.append(reverse_complement(motif))
            else:
                parts.append(motif)

        # Pad to target length
        assembled = ''.join(parts)
        if len(assembled) < seq_len:
            pad = generate_neutral_spacer(seq_len - len(assembled), gc=0.5, rng=rng)
            assembled = pad[:len(pad)//2] + assembled + pad[len(pad)//2:]
        candidates.append(assembled[:seq_len])

    return candidates


def _generate_random(motif_set, seq_len, n, rng):
    """Generate candidates with random spacing."""
    candidates = []
    for _ in range(n):
        perm = rng.permutation(len(motif_set))
        motifs = [motif_set[i] for i in perm]
        total_motif_len = sum(len(m) for m in motifs)
        total_spacer = max(seq_len - total_motif_len, len(motifs) + 1)
        spacer_lens = random_partition(total_spacer, len(motifs) + 1, min_len=1, rng=rng)

        parts = []
        for i, motif in enumerate(motifs):
            parts.append(generate_neutral_spacer(spacer_lens[i], gc=0.5, rng=rng))
            if rng.random() < 0.5:
                parts.append(reverse_complement(motif))
            else:
                parts.append(motif)
        parts.append(generate_neutral_spacer(spacer_lens[-1], gc=0.5, rng=rng))

        assembled = ''.join(parts)[:seq_len]
        if len(assembled) < seq_len:
            assembled += generate_neutral_spacer(seq_len - len(assembled), gc=0.5, rng=rng)
        candidates.append(assembled)

    return candidates
