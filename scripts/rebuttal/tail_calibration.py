#!/usr/bin/env python
"""
Extreme-tail calibration of the census p-values.

The submitted census reports 9/7,650 survivors after Benjamini-Hochberg at
q<0.05. The BH threshold at rank 9 is p = 5.9e-5, i.e. |z| > 4.02. The census
used 100 shuffles per enhancer, so the smallest attainable empirical p-value is
1e-2: every survivor comes from extrapolating a normal tail roughly four
standard deviations past any observed draw.

This measures how wrong that extrapolation is. For a stratified subset of
enhancers, draw N_BIG shuffles from the operator the census actually used and
compare the parametric normal-tail p-value, the empirical p-value from N_BIG
draws, and a generalized-Pareto fit to the upper tail. Then check whether the
same enhancers still survive BH under each rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rebuttal.common import OUT, ROOT, census_ids, load_dataset  # noqa: E402
from src.grammar.sf_gsi_v2 import pi_full  # noqa: E402
from src.models.backbones import Backbone  # noqa: E402
from src.models.readouts import build_head  # noqa: E402


def gpd_upper_p(draws: np.ndarray, obs: float, frac: float = 0.1) -> float:
    """P(|X - mu|/sigma >= obs) from a GPD fit to the upper tail of |z|."""
    z = np.abs(draws - draws.mean()) / max(draws.std(ddof=1), 1e-12)
    k = max(int(frac * len(z)), 50)
    thr = np.sort(z)[-k]
    exc = z[z > thr] - thr
    if len(exc) < 20:
        return np.nan
    if obs <= thr:
        return float((z >= obs).mean())
    try:
        c, loc, scale = stats.genpareto.fit(exc, floc=0)
        return float((k / len(z)) * stats.genpareto.sf(obs - thr, c, loc=loc, scale=scale))
    except Exception:
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dnabert2")
    ap.add_argument("--dataset", default="agarwal")
    ap.add_argument("--head", default="mean_linear")
    ap.add_argument("--n-big", type=int, default=1000)
    ap.add_argument("--n-enhancers", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # stratified subset: the census survivors + high-z + median-z + low-z
    cen = pd.read_parquet(
        ROOT / "results/v2/module1"
        / f"{'inoue' if args.dataset=='inoue' else args.dataset}_{args.model}_gsi.parquet"
    )
    cen["seq_id"] = cen["seq_id"].astype(str)
    cen = cen.sort_values("z_score", ascending=False).reset_index(drop=True)
    n = args.n_enhancers
    take = pd.concat([
        cen.head(n // 3),
        cen.iloc[len(cen) // 2 - n // 6 : len(cen) // 2 + n // 6],
        cen.tail(n // 3),
    ]).drop_duplicates("seq_id")

    df, ann = load_dataset(args.dataset)
    df["seq_id"] = df["seq_id"].astype(str)
    pos = {s: i for i, s in enumerate(df["seq_id"])}

    ck = torch.load(OUT / "heads" / f"{args.model}_{args.dataset}_{args.head}.pt",
                    map_location=args.device, weights_only=False)
    head = build_head(ck["kind"], ck["d_in"]).to(args.device)
    head.load_state_dict(ck["state_dict"])
    head.eval()
    backbone = Backbone(args.model, device=args.device)

    rows = []
    for _, r in take.iterrows():
        i = pos.get(r["seq_id"])
        if i is None or len(ann[i]) < 2:
            continue
        seq = df.iloc[i]["sequence"]
        shuf = pi_full(seq, ann[i], args.n_big, seed=i)
        preds = backbone.predict_with_head([seq] + shuf, head, batch_size=256)
        ref, sh = float(preds[0]), preds[1:].astype(float)
        mu, sd = sh.mean(), sh.std(ddof=1)
        z = abs(ref - mu) / max(sd, 1e-12)

        # empirical two-sided p from N_BIG draws
        n_ge = int((np.abs(sh - mu) >= abs(ref - mu)).sum())
        p_emp = (n_ge + 1) / (args.n_big + 1)
        p_par = 2 * (1 - stats.norm.cdf(z))
        p_gpd = gpd_upper_p(sh, z)

        # how much of the census z is just noise in a 100-draw SD estimate?
        # recompute z from disjoint 100-draw subsamples of these draws, so probe,
        # operator and enhancer are all fixed and only the draw count changes.
        n_sub = args.n_big // 100
        z100 = []
        for k in range(n_sub):
            s100 = sh[k * 100 : (k + 1) * 100]
            z100.append(abs(ref - s100.mean()) / max(s100.std(ddof=1), 1e-12))
        z100 = np.array(z100)

        # normality of the shuffle distribution itself
        sw = stats.shapiro(sh[:5000])
        rows.append(dict(
            seq_id=r["seq_id"], z_census=float(r["z_score"]),
            z_at_100_mean=float(z100.mean()), z_at_100_sd=float(z100.std(ddof=1)),
            z_at_100_max=float(z100.max()), z_at_100_min=float(z100.min()),
            z_big=float(z), p_parametric=float(p_par), p_empirical=float(p_emp),
            p_gpd=float(p_gpd) if p_gpd == p_gpd else np.nan,
            n_exceed=n_ge, shuffle_skew=float(stats.skew(sh)),
            shuffle_kurtosis=float(stats.kurtosis(sh)),
            shapiro_W=float(sw.statistic), shapiro_p=float(sw.pvalue),
        ))
        print(f"  {r['seq_id']:>24s} z_census={r['z_score']:6.2f} z_big={z:6.2f} "
              f"p_par={p_par:.3g} p_emp={p_emp:.3g} p_gpd={p_gpd:.3g}", flush=True)

    backbone.unload()
    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"tail_calibration_{args.model}_{args.dataset}.csv", index=False)

    hi = d[d["z_census"] > 3]
    out = {
        "model": args.model, "dataset": args.dataset, "head": args.head,
        "n_big_draws": args.n_big, "n_enhancers": int(len(d)),
        "bh_threshold_p_at_rank9_of_7650": 5.882e-05,
        "z_stability": {
            "pearson_z_census_vs_z_big": float(stats.pearsonr(d["z_census"], d["z_big"])[0]),
            "median_abs_change": float(np.median(np.abs(d["z_big"] - d["z_census"]))),
            "median_z_census": float(d["z_census"].median()),
            "median_z_big": float(d["z_big"].median()),
            "note": (
                "z_census comes from the submitted probe; z_big from a freshly "
                "trained mean-pooled head, so that comparison mixes two changes. "
                "The z_at_100 columns hold everything fixed except the number of "
                "draws and are the clean measurement."
            ),
        },
        "draw_count_instability": {
            "median_sd_of_z_across_disjoint_100_draw_subsamples": float(
                d["z_at_100_sd"].median()
            ),
            "median_z_at_100_minus_z_at_1000": float(
                (d["z_at_100_mean"] - d["z_big"]).median()
            ),
            "median_spread_max_minus_min_at_100": float(
                (d["z_at_100_max"] - d["z_at_100_min"]).median()
            ),
            "n_enhancers_where_some_100_draw_subsample_exceeds_bh_z_4.02": int(
                (d["z_at_100_max"] > 4.017).sum()
            ),
            "n_enhancers_where_1000_draw_z_exceeds_bh_z_4.02": int((d["z_big"] > 4.017).sum()),
            "interpretation": (
                "A z threshold of 4.02 is what BH survival required. If a single "
                "100-draw subsample can push an enhancer over that line while the "
                "1000-draw estimate does not, the census survivors are largely "
                "sampling noise in the shuffle SD denominator."
            ),
        },
        "shuffle_distribution_normality": {
            "median_shapiro_W": float(d["shapiro_W"].median()),
            "frac_shapiro_p_lt_0.05": float((d["shapiro_p"] < 0.05).mean()),
            "median_skew": float(d["shuffle_skew"].median()),
            "median_excess_kurtosis": float(d["shuffle_kurtosis"].median()),
        },
        "tail_agreement": {
            "median_log10_ratio_emp_over_par": float(
                np.median(np.log10((d["p_empirical"] + 1e-12) / (d["p_parametric"] + 1e-12)))
            ),
            "n_parametric_lt_bh_thr": int((d["p_parametric"] < 5.882e-05).sum()),
            "n_empirical_lt_bh_thr": int((d["p_empirical"] < 5.882e-05).sum()),
            "n_gpd_lt_bh_thr": int((d["p_gpd"] < 5.882e-05).sum()),
            "note": (
                "With 1000 draws the smallest attainable empirical p is 1/1001 = "
                "1.0e-3, still above the BH threshold of 5.9e-5, so the empirical "
                "column can only bound the parametric one, not replace it; the GPD "
                "column is the only one that can reach the threshold from finite "
                "draws."
            ),
        },
    }
    if len(hi):
        out["high_z_subset"] = {
            "n": int(len(hi)),
            "median_p_parametric": float(hi["p_parametric"].median()),
            "median_p_gpd": float(hi["p_gpd"].median()),
            "median_log10_ratio_gpd_over_par": float(
                np.median(np.log10((hi["p_gpd"] + 1e-30) / (hi["p_parametric"] + 1e-30)))
            ),
        }
    with open(OUT / f"tail_calibration_{args.model}_{args.dataset}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
