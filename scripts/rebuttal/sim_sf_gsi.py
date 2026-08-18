#!/usr/bin/env python
"""
Simulation validation of SF-GSI.

An oracle scorer whose response to arrangement and to spacer composition is
known by construction, run through the real operators (pi_arrange, pi_null,
pi_full) on real enhancer sequences. Three questions:

  (a) does SF-GSI recover the planted arrangement effect, and over what range
      of nuisance spacer variance;
  (b) where does the ratio form become unstable (small Var-null);
  (c) how does the submitted naive-GSI z-score behave on the same data - does
      it flag spacer sensitivity as arrangement sensitivity?

The oracle is
    f(x) = b_v * V(x) + b_a * A(x) + b_s * S(x) + eps
with
    V  vocabulary score   (invariant under every operator here)
    A  arrangement score  (orientation/order dependent, spacer independent)
    S  spacer score       (spacer dependent, arrangement independent)
so the true arrangement sd under pi_arrange is b_a * sd(A) and the true spacer
sd under pi_null is b_s * sd(S).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.rebuttal.common import OUT, load_dataset  # noqa: E402
from src.grammar.sf_gsi_v2 import (  # noqa: E402
    log_var_ratio_permutation_p,
    pi_arrange,
    pi_full,
    pi_null,
)

RC = str.maketrans("ACGT", "TGCA")


# oracle parts
def vocab_score(seq: str, motifs) -> float:
    """Depends only on which motif strings are present, anywhere.

    Scanned over the whole sequence rather than read at fixed coordinates.
    pi_arrange reverse-complements maximal contiguous runs of overlapping motif
    calls, which relocates and RCs the sites inside a run; a coordinate-indexed
    vocabulary term would read that as a vocabulary change and inject false
    positives into the null.
    """
    s = seq.upper()
    tot = 0.0
    for m in motifs:
        sub = m["_probe"]
        rc = sub.translate(RC)[::-1]
        tot += (hash(min(sub, rc)) % 1000) / 1000.0 * (s.count(sub) + s.count(rc))
    return tot / max(len(motifs), 1)


def arrangement_score(seq: str, motifs) -> float:
    """Orientation- and order-dependent, independent of spacer bytes.

    Rewards motifs on the forward strand and adjacent pairs in canonical
    alphabetical order, standing in for real orientation and ordering grammar.
    """
    if len(motifs) < 1:
        return 0.0
    ms = sorted(motifs, key=lambda m: m["start"])
    subs = [seq[m["start"] : m["end"]].upper() for m in ms]
    fwd = sum(1 for s in subs if s <= s.translate(RC)[::-1]) / len(subs)
    order = 0.0
    if len(subs) > 1:
        order = sum(1 for a, b in zip(subs, subs[1:]) if a <= b) / (len(subs) - 1)
    return fwd + order


def spacer_score(seq: str, motifs) -> float:
    """Depends only on spacer composition; invariant to motif arrangement."""
    ms = sorted(motifs, key=lambda m: m["start"])
    parts, prev = [], 0
    for m in ms:
        if m["start"] > prev:
            parts.append(seq[prev : m["start"]])
        prev = m["end"]
    if prev < len(seq):
        parts.append(seq[prev:])
    sp = "".join(parts).upper()
    if not sp:
        return 0.0
    gc = (sp.count("G") + sp.count("C")) / len(sp)
    cpg = sp.count("CG") / max(len(sp) - 1, 1)
    return gc + 3.0 * cpg


def oracle(seqs, motifs_list, b_v, b_a, b_s, noise_sd, rng):
    v = np.array([vocab_score(s, m) for s, m in zip(seqs, motifs_list)])
    a = np.array([arrangement_score(s, m) for s, m in zip(seqs, motifs_list)])
    sp = np.array([spacer_score(s, m) for s, m in zip(seqs, motifs_list)])
    return b_v * v + b_a * a + b_s * sp + rng.normal(0, noise_sd, len(seqs))


def main():
    rng_global = np.random.default_rng(0)
    df, ann = load_dataset("jores")
    elig = [i for i, m in enumerate(ann) if len(m) >= 3]
    idx = rng_global.choice(elig, 60, replace=False)
    seqs = [df.iloc[i]["sequence"] for i in idx]
    mots = []
    for i in idx:
        ms = [dict(m) for m in ann[i]]
        for m in ms:
            m["_probe"] = df.iloc[i]["sequence"][m["start"] : m["end"]].upper()
        mots.append(ms)

    N = 40
    B_A = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    B_S = [0.1, 0.5, 2.0]
    rows = []

    for b_a in B_A:
        for b_s in B_S:
            for k, (s, m) in enumerate(zip(seqs, mots)):
                rng = np.random.default_rng(1000 + k)
                arr = pi_arrange(s, m, N, seed=k)
                nul = pi_null(s, m, N, seed=k + 7)
                ful = pi_full(s, m, N, seed=k + 13)
                if len(arr) < 5 or len(nul) < 5:
                    continue
                # the perturbed sequences keep motif coordinates for arrange/null
                p_arr = oracle(arr, [m] * len(arr), 1.0, b_a, b_s, 0.0, rng)
                p_nul = oracle(nul, [m] * len(nul), 1.0, b_a, b_s, 0.0, rng)
                # pi_full moves motifs, so score it with recomputed spans
                p_ful = oracle(ful, [m] * len(ful), 1.0, b_a, b_s, 0.0, rng)
                ref = oracle([s], [m], 1.0, b_a, b_s, 0.0, rng)[0]

                # ground truth: arrangement-only sd under pi_arrange
                a_only = np.array([arrangement_score(x, m) for x in arr])
                truth_sd = b_a * a_only.std(ddof=1)

                t = log_var_ratio_permutation_p(p_arr, p_nul, n_perm=2000, seed=k)
                sf = p_arr.std(ddof=1) / max(abs(ref), 1e-8)
                naive = p_ful.std(ddof=1) / max(abs(p_ful.mean()), 1e-8)
                z_sub = abs(ref - p_ful.mean()) / max(p_ful.std(ddof=1), 1e-10)

                rows.append(
                    dict(
                        b_a=b_a, b_s=b_s, seq=k,
                        truth_arrangement_sd=truth_sd,
                        est_arrangement_sd=p_arr.std(ddof=1),
                        var_null=p_nul.var(ddof=1),
                        sf_gsi=sf,
                        gsi_naive=naive,
                        z_submitted=z_sub,
                        p_submitted=2 * (1 - stats.norm.cdf(z_sub)),
                        p_perm=t["p_perm"],
                    )
                )

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "sim_sf_gsi_raw.csv", index=False)

    summary = {"n_enhancers": int(d["seq"].nunique()), "n_draws": N, "grid": {}}
    for (b_a, b_s), g in d.groupby(["b_a", "b_s"]):
        key = f"b_a={b_a}_b_s={b_s}"
        pos = g["p_perm"] < 0.05
        summary["grid"][key] = {
            "planted_arrangement_sd_mean": float(g["truth_arrangement_sd"].mean()),
            "recovered_arrangement_sd_mean": float(g["est_arrangement_sd"].mean()),
            "recovery_ratio": float(
                g["est_arrangement_sd"].mean() / max(g["truth_arrangement_sd"].mean(), 1e-12)
            ),
            "median_var_null": float(g["var_null"].median()),
            "sf_gsi_median": float(g["sf_gsi"].median()),
            "gsi_naive_median": float(g["gsi_naive"].median()),
            "power_perm_p_lt_0.05": float(pos.mean()),
            "frac_submitted_z_flag_p_lt_0.05": float((g["p_submitted"] < 0.05).mean()),
        }

    # headline: correlation between planted and recovered arrangement sd
    ok = d["truth_arrangement_sd"] > 0
    summary["recovery_pearson_r"] = float(
        stats.pearsonr(d.loc[ok, "truth_arrangement_sd"], d.loc[ok, "est_arrangement_sd"])[0]
    )
    # false positive rate at b_a = 0
    z = d[d["b_a"] == 0]
    summary["null_calibration_b_a_0"] = {
        "n": int(len(z)),
        "frac_perm_p_lt_0.05": float((z["p_perm"] < 0.05).mean()),
        "frac_submitted_z_p_lt_0.05": float((z["p_submitted"] < 0.05).mean()),
        "median_sf_gsi": float(z["sf_gsi"].median()),
        "median_gsi_naive": float(z["gsi_naive"].median()),
    }
    # ratio instability: SF-GSI dispersion vs Var-null decile
    d["vn_decile"] = pd.qcut(d["var_null"], 5, labels=False, duplicates="drop")
    summary["ratio_stability_by_var_null_quintile"] = {
        int(q): {
            "median_var_null": float(g["var_null"].median()),
            "sf_gsi_iqr": float(g["sf_gsi"].quantile(0.75) - g["sf_gsi"].quantile(0.25)),
            "gsi_naive_iqr": float(g["gsi_naive"].quantile(0.75) - g["gsi_naive"].quantile(0.25)),
        }
        for q, g in d.groupby("vn_decile")
    }

    with open(OUT / "sim_sf_gsi.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
