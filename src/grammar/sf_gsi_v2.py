"""
Corrected spacer-factored grammar sensitivity index.

The submitted SF-GSI subtracted a null whose expected variance equalled the
alternative's, so the contrast was zero by construction. Worse, the census that
produced it used the naive vocab-preserving shuffle, which reshuffles spacer DNA
too, so spacer composition was never factored out at all.

The fix is to make the arrangement operators spacer-exact instead of
spacer-matched, which removes the subtraction and the unstable ratio with it:

    pi_orient   reverse-complement a random subset of motifs in place
    pi_order    permute motifs among equal-length slots
    pi_arrange  both of the above; the arrangement-only alternative
    pi_null     arrangement fixed, spacers dinucleotide-shuffled in place
    pi_full     the naive shuffle, kept for comparison with the submission

pi_orient/pi_order/pi_arrange leave every spacer byte, every motif boundary and
the motif multiset untouched, so

    SF-GSI = sd[f(pi_arrange(x))] / |f(x)|

needs no variance subtraction. Arrangement share of the naive GSI variance is
var_arrange / var_full and the spacer share var_null / var_full. Per-enhancer
significance is a permutation test on log(var_arrange / var_null), exact up to
the permutation count and free of any normality assumption.

Enhancers with no two equal-length motifs admit no order permutation and are
recorded as ineligible rather than silently dropped.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.perturbation.vocabulary_preserving import (
    generate_vocabulary_preserving_shuffles,
    merge_overlapping_motifs,
)
from src.utils.sequence import dinucleotide_shuffle, reverse_complement


# perturbation operators
def _merged(sequence: str, motifs: Sequence[dict]) -> List[dict]:
    if not motifs:
        return []
    return merge_overlapping_motifs(sorted(motifs, key=lambda m: m["start"]))


def pi_orient(sequence: str, motifs, n: int = 30, seed: int = 0) -> List[str]:
    """Flip motif orientations in place.  Spacer bytes untouched."""
    rng = np.random.default_rng(seed)
    merged = _merged(sequence, motifs)
    if len(merged) < 1:
        return []
    out = []
    for _ in range(n):
        s = list(sequence)
        flipped = False
        for m in merged:
            if rng.random() < 0.5:
                rc = reverse_complement(sequence[m["start"] : m["end"]])
                s[m["start"] : m["end"]] = list(rc)
                flipped = True
        if not flipped:  # force at least one flip so the draw is a real perturbation
            m = merged[rng.integers(len(merged))]
            rc = reverse_complement(sequence[m["start"] : m["end"]])
            s[m["start"] : m["end"]] = list(rc)
        out.append("".join(s))
    return out


def _length_classes(merged: List[dict]) -> Dict[int, List[int]]:
    groups = defaultdict(list)
    for i, m in enumerate(merged):
        groups[m["end"] - m["start"]].append(i)
    return {k: v for k, v in groups.items() if len(v) > 1}


def order_eligible(sequence: str, motifs) -> bool:
    return len(_length_classes(_merged(sequence, motifs))) > 0


def pi_order(sequence: str, motifs, n: int = 30, seed: int = 0) -> List[str]:
    """Permute motifs within equal-length classes; spacer bytes untouched."""
    rng = np.random.default_rng(seed)
    merged = _merged(sequence, motifs)
    groups = _length_classes(merged)
    if not groups:
        return []
    out = []
    for _ in range(n):
        s = list(sequence)
        moved = False
        for _, idxs in groups.items():
            perm = rng.permutation(len(idxs))
            if np.all(perm == np.arange(len(idxs))) and len(idxs) > 1:
                perm = np.roll(perm, 1)
            for dst, src in zip(idxs, [idxs[p] for p in perm]):
                if dst != src:
                    moved = True
                md, ms = merged[dst], merged[src]
                s[md["start"] : md["end"]] = list(
                    sequence[ms["start"] : ms["end"]]
                )
        if moved:
            out.append("".join(s))
    return out


def pi_arrange(sequence: str, motifs, n: int = 30, seed: int = 0) -> List[str]:
    """Orientation flips + equal-length order permutation. Spacer-exact."""
    rng = np.random.default_rng(seed)
    merged = _merged(sequence, motifs)
    if not merged:
        return []
    groups = _length_classes(merged)
    out = []
    for _ in range(n):
        s = list(sequence)
        # order permutation within length classes
        for _, idxs in groups.items():
            perm = rng.permutation(len(idxs))
            src_seqs = [sequence[merged[idxs[p]]["start"] : merged[idxs[p]]["end"]] for p in perm]
            for dst, sub in zip(idxs, src_seqs):
                md = merged[dst]
                s[md["start"] : md["end"]] = list(sub)
        # orientation flips
        changed = False
        for m in merged:
            if rng.random() < 0.5:
                cur = "".join(s[m["start"] : m["end"]])
                s[m["start"] : m["end"]] = list(reverse_complement(cur))
                changed = True
        cand = "".join(s)
        if cand == sequence and not changed:
            m = merged[rng.integers(len(merged))]
            cur = "".join(s[m["start"] : m["end"]])
            s[m["start"] : m["end"]] = list(reverse_complement(cur))
            cand = "".join(s)
        out.append(cand)
    return out


def pi_null(sequence: str, motifs, n: int = 30, seed: int = 0) -> List[str]:
    """Arrangement-fixed null: dinucleotide-shuffle spacers in place."""
    rng = np.random.default_rng(seed)
    merged = _merged(sequence, motifs)
    regions, prev = [], 0
    for m in merged:
        if m["start"] > prev:
            regions.append((prev, m["start"]))
        prev = m["end"]
    if prev < len(sequence):
        regions.append((prev, len(sequence)))
    if not regions:
        return []
    out = []
    for _ in range(n):
        s = list(sequence)
        for a, b in regions:
            sh = dinucleotide_shuffle(
                sequence[a:b], rng=np.random.default_rng(rng.integers(1_000_000_000))
            )
            s[a : a + len(sh)] = list(sh)
        out.append("".join(s))
    return out


def pi_full(sequence: str, motifs, n: int = 30, seed: int = 0) -> List[str]:
    """The naive vocabulary-preserving shuffle used by the submitted census."""
    return generate_vocabulary_preserving_shuffles(
        sequence, {"motifs": list(motifs)}, n_shuffles=n, seed=seed
    )


OPERATORS = {
    "orient": pi_orient,
    "order": pi_order,
    "arrange": pi_arrange,
    "null": pi_null,
    "full": pi_full,
}


# statistics
def log_var_ratio_permutation_p(
    alt: np.ndarray, null: np.ndarray, n_perm: int = 10_000, seed: int = 0
) -> Dict[str, float]:
    """One-sided permutation test for Var(alt) > Var(null).

    Both samples are centred before pooling so the test targets scale rather
    than location. Returns the statistic, the (r+1)/(B+1) p-value, and a GPD
    tail estimate for when the observed statistic beats every permutation.
    """
    alt = np.asarray(alt, dtype=float)
    null = np.asarray(null, dtype=float)
    na, nn = len(alt), len(null)
    if na < 3 or nn < 3:
        return {"stat": np.nan, "p_perm": np.nan, "p_gpd": np.nan, "n_perm": 0}

    va, vn = alt.var(ddof=1), null.var(ddof=1)
    eps = 1e-300
    obs = np.log((va + eps) / (vn + eps))

    pooled = np.concatenate([alt - alt.mean(), null - null.mean()])
    rng = np.random.default_rng(seed)
    # vectorised: one (n_perm, na+nn) matrix of independent permutations
    order = np.argsort(rng.random((n_perm, na + nn)), axis=1)
    P = pooled[order]
    va_p = P[:, :na].var(axis=1, ddof=1)
    vn_p = P[:, na:].var(axis=1, ddof=1)
    stats_ = np.log((va_p + eps) / (vn_p + eps))

    r = int((stats_ >= obs).sum())
    p_perm = (r + 1) / (n_perm + 1)

    p_gpd = np.nan
    if r < 10:  # extreme tail: fit a GPD to the top 250 permutation statistics
        p_gpd = _gpd_tail_p(stats_, obs)

    return {
        "stat": float(obs),
        "var_alt": float(va),
        "var_null": float(vn),
        "p_perm": float(p_perm),
        "p_gpd": float(p_gpd),
        "n_perm": n_perm,
        "n_exceed": r,
    }


def _gpd_tail_p(null_stats: np.ndarray, obs: float, n_exc: int = 250) -> float:
    """Knijnenburg-style GPD approximation to a permutation tail p-value."""
    from scipy import stats as sst

    n_exc = min(n_exc, max(50, len(null_stats) // 20))
    thr = np.sort(null_stats)[-n_exc]
    exc = null_stats[null_stats > thr] - thr
    if len(exc) < 20 or obs <= thr:
        return np.nan
    try:
        c, loc, scale = sst.genpareto.fit(exc, floc=0)
        tail = sst.genpareto.sf(obs - thr, c, loc=loc, scale=scale)
        return float((n_exc / len(null_stats)) * tail)
    except Exception:
        return np.nan


def summarize_enhancer(preds: Dict[str, np.ndarray], reference: float) -> Dict:
    """Turn per-operator prediction vectors into the SF-GSI v2 record."""
    rec: Dict[str, float] = {"reference": float(reference)}
    for k, v in preds.items():
        v = np.asarray(v, dtype=float)
        rec[f"n_{k}"] = int(len(v))
        rec[f"mean_{k}"] = float(v.mean()) if len(v) else np.nan
        rec[f"var_{k}"] = float(v.var(ddof=1)) if len(v) > 1 else np.nan
        rec[f"sd_{k}"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan

    denom = max(abs(reference), 1e-8)
    rec["gsi_naive"] = (
        rec.get("sd_full", np.nan) / max(abs(rec.get("mean_full", np.nan)), 1e-8)
        if not np.isnan(rec.get("sd_full", np.nan))
        else np.nan
    )
    rec["sf_gsi"] = rec.get("sd_arrange", np.nan) / denom
    vf = rec.get("var_full", np.nan)
    if vf and vf > 0:
        rec["arrangement_share"] = rec.get("var_arrange", np.nan) / vf
        rec["spacer_share"] = rec.get("var_null", np.nan) / vf
    else:
        rec["arrangement_share"] = np.nan
        rec["spacer_share"] = np.nan
    return rec
