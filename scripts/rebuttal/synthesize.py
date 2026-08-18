#!/usr/bin/env python
"""
Assemble every rebuttal experiment into one report keyed by reviewer point.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rebuttal.common import OUT, bh_fdr, bootstrap_ci, storey_pi0  # noqa: E402


def jload(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def readout_table() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(OUT / "readout_metrics_*.json"))):
        rows.extend(json.loads(Path(f).read_text()))
    return pd.DataFrame(rows)


def census_table() -> pd.DataFrame:
    fs = sorted(glob.glob(str(OUT / "census_v2" / "*_census_v2.parquet")))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def main():
    rep = {}

    # does a sequence-aware readout change the answer?
    rt = readout_table()
    if len(rt):
        piv = rt.pivot_table(index=["model", "dataset"], columns="head", values="test_r")
        best_oa = piv[["attn_pos", "cnn1d", "transformer"]].max(axis=1)
        rep["readout_ladder"] = {
            "n_pairs": int(len(piv)),
            "mean_test_r_by_head": rt.groupby("head")["test_r"].mean().round(4).to_dict(),
            "mean_gain_order_aware_over_mean_linear": float(
                (best_oa - piv["mean_linear"]).mean()
            ),
            "n_pairs_order_aware_better": int((best_oa > piv["mean_linear"]).sum()),
            "per_pair_gain": {f"{a}__{b}": float(v) for (a, b), v in (best_oa - piv["mean_linear"]).round(4).items()},
        }
        rt.to_csv(OUT / "readout_table.csv", index=False)

    ct = census_table()
    if len(ct):
        ct.to_parquet(OUT / "census_v2_all.parquet", index=False)
        g = ct.groupby("head")
        rows = {}
        for head, d in g:
            p = d["p_perm"].to_numpy()
            rej, _ = bh_fdr(p)
            pi0 = storey_pi0(p)
            lo, hi = bootstrap_ci(p, storey_pi0, n_boot=1000)
            rows[head] = {
                "n": int(len(d)),
                "perm_invariant": bool(d["perm_invariant"].iloc[0]),
                "mean_probe_r": float(d["probe_test_r"].mean()),
                "median_sf_gsi": float(d["sf_gsi"].median()),
                "median_gsi_naive": float(d["gsi_naive"].median()),
                "median_arrangement_share": float(d["arrangement_share"].median()),
                "median_spacer_share": float(d["spacer_share"].median()),
                "frac_p_perm_lt_0.05": float((p < 0.05).mean()),
                "frac_survive_bh_q0.05": float(rej.mean()),
                "billboard_fraction_pi0_pct": 100 * pi0,
                "billboard_fraction_ci95_pct": [100 * lo, 100 * hi],
            }
        rep["census_v2_by_head"] = rows

        # per (model, dataset, head) so the claim can be checked pair by pair
        per = []
        for (m, ds, h), d in ct.groupby(["model", "dataset", "head"]):
            p = d["p_perm"].to_numpy()
            rej, _ = bh_fdr(p)
            per.append(dict(model=m, dataset=ds, head=h, n=len(d),
                            probe_r=float(d["probe_test_r"].iloc[0]),
                            median_sf_gsi=float(d["sf_gsi"].median()),
                            median_arrangement_share=float(d["arrangement_share"].median()),
                            median_spacer_share=float(d["spacer_share"].median()),
                            frac_sig=float((p < 0.05).mean()),
                            n_survive_bh=int(rej.sum()),
                            pi0=storey_pi0(p)))
        pdf = pd.DataFrame(per)
        pdf.to_csv(OUT / "census_v2_per_pair.csv", index=False)

        # is measured arrangement sensitivity associated with probe quality?
        from scipy import stats as sst

        ok = pdf["probe_r"].notna()
        rep["sensitivity_vs_probe_quality_v2"] = {
            "spearman_probe_r_vs_median_sf_gsi": list(
                map(float, sst.spearmanr(pdf.loc[ok, "probe_r"], pdf.loc[ok, "median_sf_gsi"]))
            ),
            "spearman_probe_r_vs_frac_sig": list(
                map(float, sst.spearmanr(pdf.loc[ok, "probe_r"], pdf.loc[ok, "frac_sig"]))
            ),
        }
        # viability-restricted headline
        strat = {}
        for thr in (0.0, 0.3, 0.4):
            sub = ct[ct["probe_test_r"] > thr] if thr else ct
            if not len(sub):
                continue
            p = sub["p_perm"].to_numpy()
            rej, _ = bh_fdr(p)
            strat[f"probe_r_gt_{thr}"] = {
                "n": int(len(sub)),
                "n_survive_bh": int(rej.sum()),
                "billboard_fraction_pi0_pct": 100 * storey_pi0(p),
                "median_arrangement_share": float(sub["arrangement_share"].median()),
            }
        rep["census_v2_probe_viability"] = strat

    for key, path in [
        ("audit_of_submitted_census", OUT / "audit_census.json"),
        ("operator_validity", OUT / "operator_checks.json"),
        ("sf_gsi_simulation", OUT / "sim_sf_gsi.json"),
        ("planted_grammar_positive_control", OUT / "planted_grammar.json"),
        ("matched_vocabulary_biology", OUT / "matched_vocab.json"),
        ("completeness_reconciliation", OUT / "completeness_v2.json"),
        ("parm_end_to_end_agarwal", OUT / "parm_census_agarwal_K562.json"),
        ("orientation_control_agarwal", OUT / "orientation_control_agarwal.json"),
        ("tail_calibration", OUT / "tail_calibration_dnabert2_agarwal.json"),
    ]:
        v = jload(path)
        if v is not None:
            rep[key] = v

    with open(OUT / "REBUTTAL_RESULTS.json", "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(json.dumps({k: (list(v.keys())[:6] if isinstance(v, dict) else v)
                      for k, v in rep.items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
