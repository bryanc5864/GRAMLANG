"""
Causal grammar rule extraction via controlled perturbation.

For each motif pair, systematically varies spacing and orientation
to extract explicit, interpretable grammar rules.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from scipy.interpolate import interp1d

from src.utils.sequence import reverse_complement, generate_neutral_spacer, gc_content


class GrammarRuleExtractor:
    """Extract grammar rules from model predictions via perturbation."""

    def __init__(self, model, cell_type: str = None):
        self.model = model
        self.cell_type = cell_type

    def extract_pairwise_rules(
        self,
        sequence: str,
        motif_annotations: dict,
        spacing_range: Tuple[int, int] = (2, 50),
        spacing_step: int = 1,
    ) -> Dict[str, dict]:
        """
        For each motif pair, extract spacing and orientation preferences.

        Returns:
            Dict keyed by "motifA_motifB", values contain spacing/orientation profiles
        """
        motifs = motif_annotations.get('motifs', [])
        if len(motifs) < 2:
            return {}

        seq_gc = gc_content(sequence)
        seq_len = len(sequence)
        rules = {}

        for i, j in combinations(range(min(len(motifs), 10)), 2):  # Cap at 10 motifs
            motif_a = motifs[i]
            motif_b = motifs[j]
            pair_key = f"{motif_a.get('motif_name', 'A')}_{motif_b.get('motif_name', 'B')}"

            seq_a = sequence[motif_a['start']:motif_a['end']]
            seq_b = sequence[motif_b['start']:motif_b['end']]

            if not seq_a or not seq_b:
                continue

            # Spacing scan
            spacings = list(range(spacing_range[0], spacing_range[1] + 1, spacing_step))
            spacing_seqs = []

            for sp in spacings:
                constructed = self._build_pair_sequence(
                    seq_a, seq_b, sp, seq_len, seq_gc
                )
                if constructed:
                    spacing_seqs.append(constructed)

            if not spacing_seqs:
                continue

            # Batch predict
            spacing_exprs = self.model.predict_expression(
                spacing_seqs, cell_type=self.cell_type
            )

            # Orientation scan at optimal spacing
            optimal_sp_idx = np.argmax(spacing_exprs)
            optimal_sp = spacings[min(optimal_sp_idx, len(spacings) - 1)]

            orientations = ['+/+', '+/-', '-/+', '-/-']
            orient_seqs = []
            for orient in orientations:
                a_strand, b_strand = orient.split('/')
                a_seq = seq_a if a_strand == '+' else reverse_complement(seq_a)
                b_seq = seq_b if b_strand == '+' else reverse_complement(seq_b)
                constructed = self._build_pair_sequence(
                    a_seq, b_seq, optimal_sp, seq_len, seq_gc
                )
                if constructed:
                    orient_seqs.append(constructed)

            if orient_seqs:
                orient_exprs = self.model.predict_expression(
                    orient_seqs, cell_type=self.cell_type
                )
                orientation_effects = dict(zip(orientations[:len(orient_exprs)],
                                               orient_exprs.tolist()))
            else:
                orientation_effects = {}

            # Helical phase score
            helical_score = self._compute_helical_phase(
                np.array(spacings[:len(spacing_exprs)]), spacing_exprs
            )

            # Compile rule
            rules[pair_key] = {
                'motif_a_name': motif_a.get('motif_name', 'A'),
                'motif_b_name': motif_b.get('motif_name', 'B'),
                'spacing_profile': spacing_exprs.tolist(),
                'spacings': spacings[:len(spacing_exprs)],
                'optimal_spacing': int(spacings[np.argmax(spacing_exprs)]),
                'spacing_sensitivity': float(np.std(spacing_exprs)),
                'orientation_effects': orientation_effects,
                'optimal_orientation': (
                    orientations[np.argmax(orient_exprs)]
                    if orient_seqs else '+/+'
                ),
                'orientation_sensitivity': (
                    float(np.std(orient_exprs)) if orient_seqs else 0.0
                ),
                'helical_phase_score': float(helical_score),
                'expression_at_optimal': float(np.max(spacing_exprs)),
                'expression_at_worst': float(np.min(spacing_exprs)),
                'fold_change': float(
                    np.max(spacing_exprs) / max(np.min(spacing_exprs), 1e-10)
                ),
            }

        return rules

    def _build_pair_sequence(self, seq_a, seq_b, spacing, target_len, gc):
        """Build a synthetic sequence with two motifs at given spacing."""
        spacer = generate_neutral_spacer(spacing, gc=gc)
        core = seq_a + spacer + seq_b
        if len(core) > target_len:
            return None

        pad_total = target_len - len(core)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        rng = np.random.default_rng(hash((seq_a, seq_b, spacing)) % (2**32))
        left = generate_neutral_spacer(pad_left, gc=gc, rng=rng)
        right = generate_neutral_spacer(pad_right, gc=gc, rng=rng)

        return left + core + right

    def _compute_helical_phase(self, spacings, expressions):
        """Compute strength of ~10.5bp periodicity in spacing profile."""
        if len(spacings) < 10:
            return 0.0

        spacings = np.array(spacings, dtype=float)
        expressions = np.array(expressions, dtype=float)

        # Detrend
        coeffs = np.polyfit(spacings, expressions, 1)
        detrended = expressions - np.polyval(coeffs, spacings)

        # Interpolate to uniform spacing
        try:
            fn = interp1d(spacings, detrended, kind='linear', fill_value='extrapolate')
            uniform = np.arange(spacings[0], spacings[-1] + 1)
            uniform_expr = fn(uniform)
        except Exception:
            return 0.0

        # FFT
        if len(uniform_expr) < 5:
            return 0.0

        fft_vals = np.fft.rfft(uniform_expr)
        freqs = np.fft.rfftfreq(len(uniform_expr), d=1.0)
        power = np.abs(fft_vals) ** 2

        # Find power at helical repeat frequency (1/10.5 ~ 0.095 cycles/bp)
        helical_freq = 1.0 / 10.5
        freq_idx = np.argmin(np.abs(freqs - helical_freq))

        mean_power = np.mean(power[1:]) if len(power) > 1 else 1e-10
        if mean_power > 0:
            return float(power[freq_idx] / mean_power)
        return 0.0
