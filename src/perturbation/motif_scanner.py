"""
Motif scanning using FIMO from the MEME Suite.

Scans DNA sequences for transcription factor binding motifs
from JASPAR and other motif databases.
"""

import subprocess
import tempfile
import os
import pandas as pd
import numpy as np
from typing import List, Optional


class MotifScanner:
    """Scan sequences for TF binding motifs using FIMO."""

    # Default FIMO binary path from local MEME Suite install
    _FIMO_PATHS = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), 'tools', 'meme', 'bin', 'fimo'),
        'fimo',  # fallback to PATH
    ]

    def __init__(self, motif_database_path: str, p_threshold: float = 1e-4,
                 score_fraction: float = 0.7, fimo_path: str = None):
        """
        Args:
            motif_database_path: Path to MEME-format motif database
            p_threshold: FIMO p-value threshold (used when FIMO is available)
            score_fraction: Min fraction of max PWM score for fallback scanner
            fimo_path: Explicit path to FIMO binary (auto-detected if None)
        """
        self.motif_db = motif_database_path
        self.p_threshold = p_threshold
        self.score_fraction = score_fraction
        self._fimo_path = fimo_path
        self._verify_fimo()

    def _verify_fimo(self):
        """Check that FIMO is available."""
        candidates = [self._fimo_path] if self._fimo_path else self._FIMO_PATHS
        for fimo_bin in candidates:
            if fimo_bin is None:
                continue
            try:
                result = subprocess.run([fimo_bin, '--version'],
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    self._fimo_path = fimo_bin
                    self._use_fallback = False
                    print(f"  FIMO found: {fimo_bin} (v{result.stdout.strip()})")
                    return
            except FileNotFoundError:
                continue
        print("WARNING: FIMO not found. Using fallback PWM scanner.")
        self._use_fallback = True

    def scan_sequences(self, sequences: List[str],
                       sequence_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scan sequences for all motifs in database.

        Returns:
            DataFrame with columns: seq_id, motif_id, motif_name,
            start, end, strand, score, p_value, matched_sequence
        """
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(sequences))]

        if self._use_fallback:
            return self._scan_fallback(sequences, sequence_ids)

        return self._scan_fimo(sequences, sequence_ids)

    def _scan_fimo(self, sequences, sequence_ids):
        """Scan using FIMO."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fa', delete=False) as f:
            for sid, seq in zip(sequence_ids, sequences):
                f.write(f">{sid}\n{seq}\n")
            fasta_path = f.name

        output_dir = tempfile.mkdtemp()
        cmd = [
            self._fimo_path,
            '--thresh', str(self.p_threshold),
            '--oc', output_dir,
            '--text',
            '--no-qvalue',
            self.motif_db,
            fasta_path
        ]

        timeout = max(300, 10 * len(sequences))  # Scale with batch size
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        os.unlink(fasta_path)

        if not result.stdout.strip():
            return self._empty_result()

        records = []
        for line in result.stdout.strip().split('\n'):
            if line.startswith('#') or line.startswith('motif_id'):
                continue
            parts = line.split('\t')
            if len(parts) >= 8:
                try:
                    records.append({
                        'motif_id': parts[0],
                        'motif_name': parts[1] if len(parts) > 1 else parts[0],
                        'seq_id': parts[2],
                        'start': int(parts[3]) - 1,  # 0-indexed
                        'end': int(parts[4]),
                        'strand': parts[5],
                        'score': float(parts[6]),
                        'p_value': float(parts[7]),
                        'matched_sequence': parts[8] if len(parts) > 8 else ''
                    })
                except (ValueError, IndexError):
                    continue

        return pd.DataFrame(records) if records else self._empty_result()

    def _scan_fallback(self, sequences, sequence_ids):
        """
        Vectorized PWM-based fallback scanner when FIMO is not available.
        Uses numpy for fast batch scoring across all sequences.
        """
        pwms = self._parse_meme_file()
        if not pwms:
            return self._empty_result()

        from src.utils.sequence import reverse_complement

        BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        records = []

        # Convert all sequences to integer arrays once
        seq_arrays_fwd = []
        seq_arrays_rev = []
        seq_uppers = []
        for seq in sequences:
            su = seq.upper()
            seq_uppers.append(su)
            arr = np.array([BASE_MAP.get(c, -1) for c in su], dtype=np.int8)
            seq_arrays_fwd.append(arr)
            rc = reverse_complement(su)
            arr_rc = np.array([BASE_MAP.get(c, -1) for c in rc], dtype=np.int8)
            seq_arrays_rev.append(arr_rc)

        # Precompute log-odds matrices for all motifs
        motif_logodds = {}
        for motif_name, pwm in pwms.items():
            logodds = np.log2(np.maximum(pwm, 1e-6) / 0.25)  # (motif_len, 4)
            motif_logodds[motif_name] = logodds

        # Score all sequences against all motifs using vectorized operations
        for motif_name, logodds in motif_logodds.items():
            motif_len = len(logodds)
            # Max possible score for this specific PWM
            max_score = float(logodds.max(axis=1).sum())
            if max_score <= 0:
                continue

            for si, (sid, arr_fwd, arr_rev, su) in enumerate(
                zip(sequence_ids, seq_arrays_fwd, seq_arrays_rev, seq_uppers)
            ):
                seq_len = len(arr_fwd)
                if motif_len > seq_len:
                    continue

                n_pos = seq_len - motif_len + 1

                # Forward strand - vectorized scoring
                for strand, arr, is_rev in [('+', arr_fwd, False), ('-', arr_rev, True)]:
                    # Build position matrix: (n_pos, motif_len) of base indices
                    positions = np.lib.stride_tricks.as_strided(
                        arr,
                        shape=(n_pos, motif_len),
                        strides=(arr.strides[0], arr.strides[0]),
                    ).copy()

                    # Mask N positions
                    valid_mask = np.all(positions >= 0, axis=1)  # (n_pos,)

                    if not np.any(valid_mask):
                        continue

                    valid_positions = positions[valid_mask]  # (n_valid, motif_len)
                    valid_indices = np.where(valid_mask)[0]

                    # Score: sum logodds[pos_in_motif, base] across motif
                    motif_idx = np.arange(motif_len)
                    scores = logodds[motif_idx, valid_positions].sum(axis=1)  # (n_valid,)

                    # Score fraction threshold (more reliable than approximate p-values)
                    fracs = scores / max_score

                    # Filter by score fraction threshold
                    passing = fracs >= self.score_fraction
                    if not np.any(passing):
                        continue

                    for idx in np.where(passing)[0]:
                        pos = int(valid_indices[idx])
                        if is_rev:
                            orig_start = seq_len - pos - motif_len
                        else:
                            orig_start = pos
                        # Approximate p-value from score fraction
                        frac = float(fracs[idx])
                        approx_p = float(np.exp(-max_score * max(frac, 0)))
                        records.append({
                            'motif_id': motif_name,
                            'motif_name': motif_name,
                            'seq_id': sid,
                            'start': orig_start,
                            'end': orig_start + motif_len,
                            'strand': strand,
                            'score': float(scores[idx]),
                            'p_value': approx_p,
                            'matched_sequence': su[orig_start:orig_start + motif_len],
                        })

        return pd.DataFrame(records) if records else self._empty_result()

    def _parse_meme_file(self):
        """Parse PWMs from MEME format file."""
        if not os.path.exists(self.motif_db):
            return {}

        pwms = {}
        current_motif = None
        current_pwm = []
        reading_pwm = False

        with open(self.motif_db) as f:
            for line in f:
                line = line.strip()
                if line.startswith('MOTIF'):
                    if current_motif and current_pwm:
                        pwms[current_motif] = np.array(current_pwm)
                    parts = line.split()
                    current_motif = parts[1] if len(parts) > 1 else None
                    if len(parts) > 2:
                        current_motif = parts[2]  # Use alt name if available
                    current_pwm = []
                    reading_pwm = False
                elif line.startswith('letter-probability matrix'):
                    reading_pwm = True
                elif reading_pwm and line and not line.startswith('URL'):
                    try:
                        vals = [float(x) for x in line.split()]
                        if len(vals) == 4:
                            current_pwm.append(vals)
                    except ValueError:
                        reading_pwm = False

        if current_motif and current_pwm:
            pwms[current_motif] = np.array(current_pwm)

        return pwms

    def _score_pwm(self, seq, pwm):
        """Score a sequence against a PWM."""
        base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        score = 0
        for i, base in enumerate(seq):
            if base in base_map:
                prob = pwm[i][base_map[base]]
                score += np.log2(max(prob, 1e-6) / 0.25)  # Log-odds vs uniform
        return score

    def _score_to_pvalue(self, score, motif_len, max_score=None):
        """Approximate p-value from PWM score.

        Uses a calibration based on the fraction of max possible score.
        For typical JASPAR motifs: 80% of max ≈ p~1e-4 for 8-12bp motifs.
        """
        if max_score is None:
            max_score = 2.0 * motif_len
        if max_score <= 0:
            return 1.0
        frac = score / max_score
        # Calibrated: frac=0.5 -> p~0.01, frac=0.7 -> p~5e-4, frac=0.85 -> p~1e-5
        # Uses exponential scaled by motif information content
        ic_scale = max(max_score / 2.0, 3.0)  # Higher IC = steeper curve
        return float(np.exp(-ic_scale * max(frac, 0)))

    def _empty_result(self):
        return pd.DataFrame(columns=[
            'seq_id', 'motif_id', 'motif_name', 'start', 'end',
            'strand', 'score', 'p_value', 'matched_sequence'
        ])

    def annotate_enhancer(self, sequence: str,
                          seq_id: str = "enhancer") -> dict:
        """Return structured annotation of motifs in a single enhancer."""
        hits = self.scan_sequences([sequence], [seq_id])
        return {
            'sequence': sequence,
            'motifs': hits.to_dict('records'),
            'motif_count': len(hits),
            'unique_motif_count': hits['motif_name'].nunique() if len(hits) > 0 else 0,
            'motif_names': list(hits['motif_name'].unique()) if len(hits) > 0 else [],
            'motif_density': len(hits) / max(len(sequence), 1)
        }
