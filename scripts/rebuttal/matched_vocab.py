#!/usr/bin/env python
"""
Is the billboard result about models or about biology?

The submitted paper argued arrangement matters biologically because 363 MPRA
pairs with identical called motif vocabulary differ substantially in measured
expression. That could just as easily be ordinary biological and technical
variation. Three things are needed to tell the difference and none were in the
submission:

  1. a per-dataset replicate-noise floor, so |delta| can be judged against
     within-library reproducibility;
  2. a composition-matched control set of pairs equally similar in sequence but
     differing in vocabulary, to check whether matched-vocabulary pairs are
     just the tail of a generic pair distribution;
  3. a test of whether arrangement distance predicts |delta| among
     matched-vocabulary pairs after controlling for spacer composition.

Every pair here is "vocabulary-matched under FIMO against JASPAR2024 at the
stated p-value threshold", not "identical regulatory content".
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rebuttal.common import OUT, PROC, bootstrap_ci  # noqa: E402

DATASETS = ["agarwal", "jores", "klein", "vaishnav", "inoue"]
THRESHOLDS = [1e-4, 5e-5, 1e-5]


def kmer_vec(seq: str, k: int = 4) -> np.ndarray:
    idx = {c: i for i, c in enumerate("ACGT")}
    v = np.zeros(4**k)
    s = seq.upper()
    for i in range(len(s) - k + 1):
        sub = s[i : i + k]
        if any(c not in idx for c in sub):
            continue
        j = 0
        for c in sub:
            j = j * 4 + idx[c]
        v[j] += 1
    return v / max(v.sum(), 1)


def gc(seq: str) -> float:
    s = seq.upper()
    return (s.count("G") + s.count("C")) / max(len(s), 1)


def arrangement_signature(hits: pd.DataFrame, seq_len: int):
    """Order + orientation + relative-spacing signature of one sequence."""
    h = hits.sort_values("start")
    order = tuple(h["motif_name"])
    strands = tuple(h["strand"])
    pos = (h["start"].to_numpy() / max(seq_len, 1)).round(2)
    return order, strands, tuple(pos)


def arrangement_distance(a, b) -> float:
    """0 = same order+orientation+spacing; 1 = maximally different."""
    (oa, sa, pa), (ob, sb, pb) = a, b
    d_order = 0.0 if oa == ob else 1.0
    d_strand = np.mean([x != y for x, y in zip(sa, sb)]) if len(sa) == len(sb) else 1.0
    d_pos = float(np.mean(np.abs(np.array(pa) - np.array(pb)))) if len(pa) == len(pb) else 1.0
    return float((d_order + d_strand + min(d_pos * 2, 1.0)) / 3.0)


def build_pairs(dataset: str, pthr: float, max_seqs: int = 5000):
    df = pd.read_parquet(PROC / f"{dataset}_processed.parquet")
    df["seq_id"] = df["seq_id"].astype(str)
    df = df.head(max_seqs).reset_index(drop=True)
    hits = pd.read_parquet(PROC / f"{dataset}_processed_motif_hits.parquet")
    hits["seq_id"] = hits["seq_id"].astype(str)
    hits = hits[hits["p_value"] < pthr]
    hits = hits[hits["seq_id"].isin(set(df["seq_id"]))]

    by_id = {sid: g for sid, g in hits.groupby("seq_id")}
    vocab = {}
    for sid, g in by_id.items():
        if len(g) >= 2:
            vocab[sid] = tuple(sorted(g["motif_name"]))

    groups = defaultdict(list)
    for sid, v in vocab.items():
        groups[v].append(sid)

    expr = dict(zip(df["seq_id"], df["expression"]))
    seqs = dict(zip(df["seq_id"], df["sequence"]))

    pairs = []
    for v, ids in groups.items():
        if len(ids) < 2:
            continue
        for a, b in combinations(sorted(ids), 2):
            sa = arrangement_signature(by_id[a], len(seqs[a]))
            sb = arrangement_signature(by_id[b], len(seqs[b]))
            if sa == sb:
                continue  # same arrangement too -> not informative
            pairs.append(
                dict(
                    dataset=dataset, a=a, b=b, vocab_size=len(v),
                    delta=float(expr[a] - expr[b]),
                    abs_delta=abs(float(expr[a] - expr[b])),
                    arr_dist=arrangement_distance(sa, sb),
                    gc_dist=abs(gc(seqs[a]) - gc(seqs[b])),
                    kmer_dist=float(np.abs(kmer_vec(seqs[a]) - kmer_vec(seqs[b])).sum()),
                )
            )
    return df, seqs, expr, vocab, pd.DataFrame(pairs)


def composition_matched_controls(df, seqs, expr, vocab, mv: pd.DataFrame, seed=0):
    """For each matched-vocabulary pair, draw a pair with a different
    vocabulary but a comparable 4-mer distance.
    """
    rng = np.random.default_rng(seed)
    ids = list(vocab)
    if len(ids) < 20 or len(mv) == 0:
        return pd.DataFrame()
    sample = rng.choice(ids, min(len(ids), 700), replace=False)
    kv = {s: kmer_vec(seqs[s]) for s in sample}
    rows = []
    for target in mv["kmer_dist"].to_numpy():
        best, bestd = None, np.inf
        for _ in range(60):
            a, b = rng.choice(sample, 2, replace=False)
            if vocab[a] == vocab[b]:
                continue
            d = float(np.abs(kv[a] - kv[b]).sum())
            if abs(d - target) < bestd:
                best, bestd = (a, b, d), abs(d - target)
        if best is None:
            continue
        a, b, d = best
        rows.append(
            dict(a=a, b=b, abs_delta=abs(float(expr[a] - expr[b])), kmer_dist=d,
                 target_kmer_dist=float(target))
        )
    return pd.DataFrame(rows)


def noise_floor(dataset: str) -> dict:
    """Replicate-noise floor for |delta| between two independent measurements of
    the same sequence, where replicate information exists.
    """
    df = pd.read_parquet(PROC / f"{dataset}_processed.parquet")
    if "expression_std" not in df.columns:
        return {"available": False}
    sd = df["expression_std"].to_numpy(dtype=float)
    n = df.get("n_replicates", pd.Series(np.ones(len(df)))).to_numpy(dtype=float)
    sem = sd / np.sqrt(np.maximum(n, 1))
    sem = sem[np.isfinite(sem)]
    # |X1 - X2| for two independent draws with SEM s  =>  E = 2s/sqrt(pi)
    exp_abs = 2 * sem / np.sqrt(np.pi)
    return {
        "available": True,
        "median_sem": float(np.median(sem)),
        "median_expected_abs_delta": float(np.median(exp_abs)),
        "p95_expected_abs_delta": float(np.percentile(exp_abs, 95)),
    }


def main():
    out = {"threshold_sensitivity": {}, "per_dataset": {}}
    all_pairs = []

    for pthr in THRESHOLDS:
        n_tot = 0
        for ds in DATASETS:
            _, _, _, _, mv = build_pairs(ds, pthr)
            n_tot += len(mv)
        out["threshold_sensitivity"][f"p<{pthr:g}"] = int(n_tot)

    PTHR = 1e-4
    for ds in DATASETS:
        df, seqs, expr, vocab, mv = build_pairs(ds, PTHR)
        if len(mv) == 0:
            out["per_dataset"][ds] = {"n_pairs": 0}
            continue
        mv["dataset"] = ds
        all_pairs.append(mv)
        ctrl = composition_matched_controls(df, seqs, expr, vocab, mv)
        nf = noise_floor(ds)

        # random-pair ceiling
        rng = np.random.default_rng(1)
        ids = list(expr)
        rp = np.array(
            [abs(expr[a] - expr[b]) for a, b in
             zip(rng.choice(ids, 2000), rng.choice(ids, 2000))]
        )

        rec = {
            "n_pairs": int(len(mv)),
            "median_abs_delta": float(mv["abs_delta"].median()),
            "frac_abs_delta_gt_1": float((mv["abs_delta"] > 1).mean()),
            "max_abs_delta": float(mv["abs_delta"].max()),
            "random_pair_median_abs_delta": float(np.median(rp)),
            "replicate_noise_floor": nf,
        }
        if len(ctrl):
            u = stats.mannwhitneyu(mv["abs_delta"], ctrl["abs_delta"], alternative="two-sided")
            rec["composition_matched_control"] = {
                "n": int(len(ctrl)),
                "median_abs_delta": float(ctrl["abs_delta"].median()),
                "median_kmer_dist_matched": float(ctrl["kmer_dist"].median()),
                "median_kmer_dist_target": float(mv["kmer_dist"].median()),
                "mannwhitney_u_p": float(u.pvalue),
                "ratio_matchedvocab_over_control": float(
                    mv["abs_delta"].median() / max(ctrl["abs_delta"].median(), 1e-9)
                ),
            }
        if nf.get("available"):
            rec["frac_pairs_above_noise_p95"] = float(
                (mv["abs_delta"] > nf["p95_expected_abs_delta"]).mean()
            )
        # does arrangement distance predict |delta| controlling for composition?
        if len(mv) >= 25:
            sp_arr = stats.spearmanr(mv["arr_dist"], mv["abs_delta"])
            sp_kmer = stats.spearmanr(mv["kmer_dist"], mv["abs_delta"])
            X = np.column_stack(
                [np.ones(len(mv)), mv["arr_dist"], mv["kmer_dist"], mv["gc_dist"]]
            )
            y = mv["abs_delta"].to_numpy()
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            se = np.sqrt(
                np.sum(resid**2) / (len(y) - X.shape[1]) * np.diag(np.linalg.pinv(X.T @ X))
            )
            rec["arrangement_vs_composition"] = {
                "spearman_arr_dist": [float(sp_arr.statistic), float(sp_arr.pvalue)],
                "spearman_kmer_dist": [float(sp_kmer.statistic), float(sp_kmer.pvalue)],
                "ols_beta_arr_dist": float(beta[1]),
                "ols_t_arr_dist": float(beta[1] / se[1]),
                "ols_beta_kmer_dist": float(beta[2]),
                "ols_t_kmer_dist": float(beta[2] / se[2]),
            }
        lo, hi = bootstrap_ci(mv["abs_delta"].to_numpy(), np.median)
        rec["median_abs_delta_ci95"] = [lo, hi]
        out["per_dataset"][ds] = rec

    if all_pairs:
        P = pd.concat(all_pairs, ignore_index=True)
        P.to_csv(OUT / "matched_vocab_pairs.csv", index=False)
        out["overall"] = {
            "n_pairs": int(len(P)),
            "median_abs_delta": float(P["abs_delta"].median()),
            "frac_gt_1": float((P["abs_delta"] > 1).mean()),
            "max_abs_delta": float(P["abs_delta"].max()),
        }

    with open(OUT / "matched_vocab.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
