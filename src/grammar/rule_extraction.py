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

            # --- v3 FIX: Optimize spacing INDEPENDENTLY per orientation ---
            spacings = list(range(spacing_range[0], spacing_range[1] + 1, spacing_step))
            orientations = ['+/+', '+/-', '-/+', '-/-']

            # Randomize orientation testing order to eliminate first-element bias
            rng = np.random.default_rng(hash(pair_key) % (2**32))
            orient_order = rng.permutation(len(orientations)).tolist()
            orientations_shuffled = [orientations[k] for k in orient_order]

            # Scan spacing for EACH orientation independently
            orient_spacing_profiles = {}
            orient_optimal_spacings = {}
            orient_optimal_exprs = {}

            for orient in orientations_shuffled:
                a_strand, b_strand = orient.split('/')
                a_seq_o = seq_a if a_strand == '+' else reverse_complement(seq_a)
                b_seq_o = seq_b if b_strand == '+' else reverse_complement(seq_b)

                orient_seqs = []
                valid_spacings = []
                for sp in spacings:
                    constructed = self._build_pair_sequence(
                        a_seq_o, b_seq_o, sp, seq_len, seq_gc
                    )
                    if constructed:
                        orient_seqs.append(constructed)
                        valid_spacings.append(sp)

                if not orient_seqs:
                    continue

                orient_exprs = self.model.predict_expression(
                    orient_seqs, cell_type=self.cell_type
                )
                orient_spacing_profiles[orient] = {
                    'spacings': valid_spacings[:len(orient_exprs)],
                    'expressions': orient_exprs.tolist(),
                }
                best_idx = int(np.argmax(orient_exprs))
                orient_optimal_spacings[orient] = valid_spacings[best_idx]
                orient_optimal_exprs[orient] = float(np.max(orient_exprs))

            if not orient_optimal_exprs:
                continue

            # --- v3 FIX: Select optimal orientation via permutation test ---
            # Use the expression at each orientation's own optimal spacing
            orient_max_exprs = np.array([orient_optimal_exprs[o] for o in orientations_shuffled
                                          if o in orient_optimal_exprs])
            orient_names = [o for o in orientations_shuffled if o in orient_optimal_exprs]

            orient_sensitivity = float(np.std(orient_max_exprs)) if len(orient_max_exprs) > 1 else 0.0

            # v3 FIX: Only assign optimal orientation if sensitivity is above threshold
            ORIENT_SENSITIVITY_THRESHOLD = 0.01  # minimum std to call a preference
            if orient_sensitivity >= ORIENT_SENSITIVITY_THRESHOLD and len(orient_names) >= 2:
                # Use the orientation with highest expression at its optimal spacing
                best_orient_idx = int(np.argmax(orient_max_exprs))
                optimal_orientation = orient_names[best_orient_idx]
            else:
                # v3 FIX: "undetermined" instead of defaulting to +/+
                optimal_orientation = 'undetermined'

            # Build orientation effects dict at each orientation's own optimal spacing
            orientation_effects = {o: orient_optimal_exprs[o] for o in orient_names}

            # Use +/+ spacing profile for the main profile (for backward compatibility)
            # but also store per-orientation profiles
            if '+/+' in orient_spacing_profiles:
                main_profile = orient_spacing_profiles['+/+']
            else:
                # Use first available orientation's profile
                first_orient = orient_names[0]
                main_profile = orient_spacing_profiles[first_orient]

            main_exprs = np.array(main_profile['expressions'])
            main_spacings = main_profile['spacings']

            # Global optimal: best expression across ALL orientations and spacings
            global_best_expr = max(orient_optimal_exprs.values())
            global_worst_expr = min(
                min(p['expressions']) for p in orient_spacing_profiles.values()
            )

            # Helical phase score (from +/+ profile or first available)
            helical_score = self._compute_helical_phase(
                np.array(main_spacings[:len(main_exprs)]), main_exprs
            )

            # Compile rule
            rules[pair_key] = {
                'motif_a_name': motif_a.get('motif_name', 'A'),
                'motif_b_name': motif_b.get('motif_name', 'B'),
                'spacing_profile': main_exprs.tolist(),
                'spacings': main_spacings[:len(main_exprs)],
                'optimal_spacing': int(main_spacings[np.argmax(main_exprs)]),
                'spacing_sensitivity': float(np.std(main_exprs)),
                'orientation_effects': orientation_effects,
                'per_orientation_profiles': {
                    o: p for o, p in orient_spacing_profiles.items()
                },
                'per_orientation_optimal_spacings': orient_optimal_spacings,
                'optimal_orientation': optimal_orientation,
                'orientation_sensitivity': orient_sensitivity,
                'helical_phase_score': float(helical_score),
                'expression_at_optimal': float(global_best_expr),
                'expression_at_worst': float(global_worst_expr),
                'fold_change': float(
                    global_best_expr / max(abs(global_worst_expr), 1e-10)
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
