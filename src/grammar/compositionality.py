"""
Compositionality testing: do pairwise grammar rules predict higher-order effects?

This is the core test for formal grammar complexity.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from src.utils.sequence import (
    generate_neutral_spacer, gc_content, random_partition, reverse_complement
)


class ComposionalityTester:
    """Test whether pairwise rules compose to predict higher-order effects."""

    def __init__(self, model, cell_type: str = None):
        self.model = model
        self.cell_type = cell_type

    def test_compositionality(
        self,
        sequence: str,
        motif_annotations: dict,
        pairwise_rules: Dict[str, dict] = None,
        n_arrangements: int = 200,
        seed: int = 42,
    ) -> Optional[dict]:
        """
        For an enhancer with k >= 3 motifs, test whether pairwise rules
        predict arrangement effects.

        Args:
            sequence: Enhancer sequence
            motif_annotations: Motif annotation dict
            pairwise_rules: Previously extracted pairwise rules (optional)
            n_arrangements: Number of random arrangements
            seed: Random seed

        Returns:
            Dict with compositionality metrics, or None if < 3 motifs
        """
        motifs = motif_annotations.get('motifs', [])
        n_motifs = len(motifs)

        if n_motifs < 3:
            return None

        rng = np.random.default_rng(seed)
        seq_gc = gc_content(sequence)
        seq_len = len(sequence)

        # Extract motif sequences
        motif_seqs = []
        for m in motifs[:10]:  # Cap at 10 motifs
            mseq = sequence[m['start']:m['end']]
            if mseq:
                motif_seqs.append({
                    'seq': mseq,
                    'name': m.get('motif_name', 'unknown'),
                    'length': len(mseq)
                })

        if len(motif_seqs) < 3:
            return None

        total_motif_len = sum(m['length'] for m in motif_seqs)
        total_spacer_len = max(seq_len - total_motif_len, len(motif_seqs) + 1)

        # Generate random arrangements
        arrangements = []
        arrangement_metadata = []  # Track motif positions for pairwise scoring

        for _ in range(n_arrangements):
            perm = rng.permutation(len(motif_seqs))
            permuted = [motif_seqs[i] for i in perm]

            # Random orientations
            orientations = []
            for i in range(len(permuted)):
                if rng.random() < 0.5:
                    orientations.append('-')
                    permuted[i] = {**permuted[i], 'seq': reverse_complement(permuted[i]['seq'])}
                else:
                    orientations.append('+')

            # Random spacer distribution
            n_spacers = len(permuted) + 1
            spacer_lens = random_partition(
                total_spacer_len, n_spacers, min_len=1, rng=rng
            )

            # Assemble
            parts = []
            positions = []
            pos = 0
            for i, motif in enumerate(permuted):
                spacer = generate_neutral_spacer(spacer_lens[i], gc=seq_gc, rng=rng)
                parts.append(spacer)
                pos += spacer_lens[i]
                positions.append({'start': pos, 'end': pos + motif['length'],
                                  'name': motif['name'], 'orient': orientations[i]})
                parts.append(motif['seq'])
                pos += motif['length']

            # Final spacer
            parts.append(generate_neutral_spacer(spacer_lens[-1], gc=seq_gc, rng=rng))

            new_seq = ''.join(parts)
            if len(new_seq) > seq_len:
                new_seq = new_seq[:seq_len]
            elif len(new_seq) < seq_len:
                new_seq += generate_neutral_spacer(seq_len - len(new_seq), gc=seq_gc, rng=rng)

            arrangements.append(new_seq)
            arrangement_metadata.append({
                'perm': perm.tolist(),
                'positions': positions,
                'spacer_lens': spacer_lens
            })

        # Get model predictions (ground truth)
        model_preds = self.model.predict_expression(
            arrangements, cell_type=self.cell_type
        )

        # Compute pairwise scores for each arrangement
        pairwise_preds = []
        for meta in arrangement_metadata:
            score = self._compute_pairwise_score(meta['positions'])
            pairwise_preds.append(score)
        pairwise_preds = np.array(pairwise_preds)

        # Compute correlation
        if (np.std(model_preds) > 1e-10 and np.std(pairwise_preds) > 1e-10):
            corr, p_val = pearsonr(model_preds, pairwise_preds)
            r2 = corr ** 2
        else:
            corr, p_val, r2 = 0.0, 1.0, 0.0

        compositionality_gap = 1.0 - r2

        return {
            'n_motifs': len(motif_seqs),
            'n_arrangements': len(arrangements),
            'pairwise_corr': float(corr),
            'pairwise_r2': float(r2),
            'pairwise_p_value': float(p_val),
            'compositionality_gap': float(compositionality_gap),
            'model_pred_mean': float(np.mean(model_preds)),
            'model_pred_std': float(np.std(model_preds)),
            'pairwise_pred_std': float(np.std(pairwise_preds)),
        }

    def _compute_pairwise_score(self, positions: list) -> float:
        """
        Compute additive pairwise interaction score from motif positions.

        Uses simple distance-based scoring:
        - Closer motifs → stronger interaction (log decay)
        - Same-strand motifs → bonus
        - ~10.5bp spacing multiples → bonus (helical phasing)
        """
        from itertools import combinations
        score = 0.0

        for (p1, p2) in combinations(positions, 2):
            dist = abs(p1['start'] - p2['start'])
            if dist == 0:
                dist = 1

            # Distance effect (log decay)
            distance_score = 1.0 / np.log2(dist + 1)

            # Orientation effect
            if p1['orient'] == p2['orient']:
                orient_bonus = 0.1
            else:
                orient_bonus = -0.05

            # Helical phasing bonus
            phase = dist % 10.5
            if phase < 2 or phase > 8.5:
                phase_bonus = 0.1
            else:
                phase_bonus = 0.0

            score += distance_score + orient_bonus + phase_bonus

        return score


def run_compositionality_sweep(
    dataset: pd.DataFrame,
    motif_hits: pd.DataFrame,
    model,
    cell_type: str = None,
    target_ks: List[int] = None,
    max_per_k: int = 50,
    n_arrangements: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run compositionality test across enhancers grouped by motif count.

    Args:
        dataset: Preprocessed MPRA data
        motif_hits: Motif annotations
        model: GrammarModel instance
        cell_type: Cell type
        target_ks: Motif counts to test
        max_per_k: Max enhancers per motif count
        n_arrangements: Arrangements per enhancer
        seed: Random seed

    Returns:
        DataFrame with compositionality results
    """
    if target_ks is None:
        target_ks = [3, 4, 5, 6, 7, 8]

    tester = ComposionalityTester(model, cell_type)
    results = []

    for k in target_ks:
        # Find enhancers with exactly k unique motifs
        eligible = dataset[
            (dataset['n_motifs'] >= k) &
            (dataset['n_motifs'] <= k + 1)
        ]

        if len(eligible) < 3:
            print(f"  k={k}: too few enhancers ({len(eligible)}), skipping")
            continue

        sample = eligible.sample(n=min(max_per_k, len(eligible)), random_state=seed)
        print(f"  k={k}: testing {len(sample)} enhancers")

        for idx, row in tqdm(sample.iterrows(), total=len(sample),
                             desc=f"k={k}"):
            seq_id = str(row['seq_id'])
            seq = row['sequence']
            seq_motifs = motif_hits[motif_hits['seq_id'] == seq_id]

            annotation = {
                'sequence': seq,
                'motifs': seq_motifs.to_dict('records'),
                'motif_count': len(seq_motifs),
            }

            try:
                result = tester.test_compositionality(
                    seq, annotation,
                    n_arrangements=n_arrangements,
                    seed=seed + hash(seq_id) % 10000
                )
                if result:
                    result['seq_id'] = seq_id
                    result['model'] = model.name
                    results.append(result)
            except Exception as e:
                print(f"    Error for {seq_id}: {e}")

    return pd.DataFrame(results)
