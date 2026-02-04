"""
Cross-species grammar transfer experiments.

Tests whether grammar rules from one species predict arrangement
effects in another species.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from src.utils.sequence import generate_neutral_spacer, gc_content, random_partition, reverse_complement


def build_grammar_predictor(rules_df: pd.DataFrame) -> 'GrammarPredictor':
    """Build a predictive grammar model from extracted rules."""
    return GrammarPredictor(rules_df)


class GrammarPredictor:
    """Predict expression from grammar rules alone."""

    def __init__(self, rules_df: pd.DataFrame):
        """
        Build lookup tables from extracted pairwise rules.

        For each motif pair, store:
        - Optimal spacing
        - Spacing sensitivity profile
        - Optimal orientation
        """
        self.pair_rules = {}

        for _, row in rules_df.iterrows():
            pair = row.get('pair', f"{row.get('motif_a', '')}_{row.get('motif_b', '')}")
            if pair not in self.pair_rules:
                self.pair_rules[pair] = {
                    'optimal_spacing': row.get('optimal_spacing', 15),
                    'spacing_sensitivity': row.get('spacing_sensitivity', 0),
                    'optimal_orientation': row.get('optimal_orientation', '+/+'),
                    'fold_change': row.get('fold_change', 1.0),
                }

    def predict_arrangement_score(self, motif_positions: list) -> float:
        """
        Predict a grammar score for a given motif arrangement.

        Args:
            motif_positions: List of dicts with 'name', 'start', 'end', 'orient'

        Returns:
            Predicted grammar quality score
        """
        from itertools import combinations
        score = 0.0

        for p1, p2 in combinations(motif_positions, 2):
            pair_key = f"{p1['name']}_{p2['name']}"
            reverse_key = f"{p2['name']}_{p1['name']}"

            rule = self.pair_rules.get(pair_key, self.pair_rules.get(reverse_key, None))
            if rule is None:
                continue

            # Spacing score
            actual_spacing = abs(p1['start'] - p2['start'])
            optimal = rule['optimal_spacing']
            spacing_penalty = -abs(actual_spacing - optimal) * rule['spacing_sensitivity']

            # Orientation score
            actual_orient = f"{p1.get('orient', '+')}/{p2.get('orient', '+')}"
            if actual_orient == rule['optimal_orientation']:
                orient_bonus = rule['fold_change'] * 0.1
            else:
                orient_bonus = 0

            score += spacing_penalty + orient_bonus

        return score


def compute_grammar_transfer(
    source_rules: pd.DataFrame,
    target_dataset: pd.DataFrame,
    target_motif_hits: pd.DataFrame,
    model,
    cell_type: str = None,
    n_test_enhancers: int = 100,
    n_arrangements: int = 200,
    seed: int = 42,
) -> dict:
    """
    Test grammar rule transfer from source to target species.

    Args:
        source_rules: Grammar rules from source species
        target_dataset: MPRA data from target species
        target_motif_hits: Motif annotations for target
        model: Model for ground-truth predictions
        cell_type: Cell type
        n_test_enhancers: Max enhancers to test
        n_arrangements: Arrangements per enhancer
        seed: Random seed

    Returns:
        Dict with transfer R^2 and correlation
    """
    rng = np.random.default_rng(seed)
    predictor = build_grammar_predictor(source_rules)

    # Select target enhancers with enough motifs
    eligible = target_dataset[target_dataset['n_motifs'] >= 3]
    if len(eligible) > n_test_enhancers:
        eligible = eligible.sample(n=n_test_enhancers, random_state=seed)

    all_predicted = []
    all_actual = []

    for _, row in tqdm(eligible.iterrows(), total=len(eligible), desc="Transfer"):
        seq = row['sequence']
        seq_id = str(row['seq_id'])
        seq_gc = gc_content(seq)
        seq_len = len(seq)

        seq_motifs = target_motif_hits[target_motif_hits['seq_id'] == seq_id]
        motifs = seq_motifs.to_dict('records')

        if len(motifs) < 3:
            continue

        # Extract motif sequences
        motif_seqs = []
        for m in motifs[:8]:
            mseq = seq[m['start']:m['end']]
            if mseq:
                motif_seqs.append({
                    'seq': mseq,
                    'name': m.get('motif_name', 'unknown'),
                    'length': len(mseq)
                })

        if len(motif_seqs) < 3:
            continue

        total_motif_len = sum(m['length'] for m in motif_seqs)
        total_spacer = max(seq_len - total_motif_len, len(motif_seqs) + 1)

        # Generate random arrangements
        arr_seqs = []
        arr_positions = []

        for _ in range(n_arrangements):
            perm = rng.permutation(len(motif_seqs))
            permuted = [motif_seqs[i] for i in perm]

            orientations = ['+' if rng.random() > 0.5 else '-' for _ in permuted]
            spacer_lens = random_partition(total_spacer, len(permuted) + 1, min_len=1, rng=rng)

            parts = []
            positions = []
            pos = 0
            for i, motif in enumerate(permuted):
                spacer = generate_neutral_spacer(spacer_lens[i], gc=seq_gc, rng=rng)
                parts.append(spacer)
                pos += spacer_lens[i]
                mseq = motif['seq'] if orientations[i] == '+' else reverse_complement(motif['seq'])
                positions.append({
                    'name': motif['name'], 'start': pos,
                    'end': pos + motif['length'], 'orient': orientations[i]
                })
                parts.append(mseq)
                pos += motif['length']
            parts.append(generate_neutral_spacer(spacer_lens[-1], gc=seq_gc, rng=rng))

            new_seq = ''.join(parts)[:seq_len]
            if len(new_seq) < seq_len:
                new_seq += generate_neutral_spacer(seq_len - len(new_seq), gc=seq_gc, rng=rng)

            arr_seqs.append(new_seq)
            arr_positions.append(positions)

        # Model predictions (ground truth)
        actual = model.predict_expression(arr_seqs, cell_type=cell_type)

        # Source grammar predictions
        predicted = np.array([predictor.predict_arrangement_score(pos) for pos in arr_positions])

        all_actual.extend(actual)
        all_predicted.extend(predicted)

    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)

    if (len(all_predicted) > 2 and np.std(all_predicted) > 1e-10
            and np.std(all_actual) > 1e-10):
        transfer_corr, transfer_p = spearmanr(all_predicted, all_actual)
        pearson_r, _ = pearsonr(all_predicted, all_actual)
        transfer_r2 = pearson_r ** 2
    else:
        transfer_corr = 0.0
        transfer_p = 1.0
        transfer_r2 = 0.0

    return {
        'transfer_r2': float(transfer_r2),
        'transfer_corr': float(transfer_corr),
        'transfer_p': float(transfer_p),
        'n_enhancers_tested': len(eligible),
        'n_total_predictions': len(all_predicted),
    }


def compute_transfer_matrix(
    species_rules: Dict[str, pd.DataFrame],
    species_datasets: Dict[str, pd.DataFrame],
    species_motif_hits: Dict[str, pd.DataFrame],
    model,
    species_list: List[str] = None,
    cell_type: str = None,
) -> pd.DataFrame:
    """Compute full grammar transfer matrix."""
    if species_list is None:
        species_list = list(species_rules.keys())

    results = []
    for source in species_list:
        for target in species_list:
            print(f"  Transfer: {source} -> {target}")
            result = compute_grammar_transfer(
                source_rules=species_rules[source],
                target_dataset=species_datasets[target],
                target_motif_hits=species_motif_hits[target],
                model=model,
                cell_type=cell_type,
                n_test_enhancers=50,
                n_arrangements=100,
            )
            results.append({
                'source': source,
                'target': target,
                **result
            })

    return pd.DataFrame(results)
