#!/usr/bin/env python
"""
Audit the submitted 7,650-triple census.

Do the headline numbers reproduce from the stored per-enhancer records, which
perturbation operator actually generated them, do they hold up when stratified
by probe quality, is the parametric normal tail carrying the BH threshold, and
are the dataset labels right?
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rebuttal.common import OUT, ROOT, bh_fdr, bootstrap_ci, probe_r_table, storey_pi0  # noqa: E402


def load_census() -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "results/v2/module1/*_gsi.parquet")))
    files = [f for f in files if "all_gsi_results" not in f]
    frames = []
    for f in files:
        d = pd.read_parquet(f)
        d["source_file"] = Path(f).name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main():
    out = {}
    d = load_census()
    out["n_triples"] = int(len(d))

    # do the headline numbers reproduce?
    z = d["z_score"].to_numpy()
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    d["p_parametric"] = p
    agree = np.allclose(p, d["p_value_corrected"].to_numpy(), atol=1e-12)
    rej, q = bh_fdr(p)
    out["reproduction"] = {
        "stored_p_matches_normal_two_sided": bool(agree),
        "n_nominal_p_lt_0.05": int((p < 0.05).sum()),
        "frac_nominal": float((p < 0.05).mean()),
        "n_survive_bh_q0.05": int(rej.sum()),
        "frac_survive_pct": float(100 * rej.mean()),
        "storey_pi0": {str(l): storey_pi0(p, l) for l in (0.3, 0.4, 0.5, 0.6, 0.7)},
    }
    lo, hi = bootstrap_ci(p, lambda x: storey_pi0(x, 0.5))
    out["reproduction"]["pi0_lambda0.5_ci95"] = [lo, hi]

    # what fraction of the 9 survivors have a numerically-zero parametric p?
    surv = d[rej]
    out["survivors"] = {
        "n": int(len(surv)),
        "n_with_p_exactly_zero": int((surv["p_parametric"] == 0).sum()),
        "min_abs_z": float(np.abs(surv["z_score"]).min()),
        "max_abs_z": float(np.abs(surv["z_score"]).max()),
        "by_pair": surv.groupby(["dataset", "model"]).size().to_dict().__str__(),
    }
    # BH threshold in z units: what |z| would a test need to survive?
    k = int(rej.sum())
    thresh_p = 0.05 * max(k, 1) / len(d)
    out["survivors"]["bh_p_threshold"] = float(thresh_p)
    out["survivors"]["bh_z_threshold"] = float(stats.norm.isf(thresh_p / 2))
    # with 100 shuffles the smallest attainable empirical p is 1/100
    out["survivors"]["min_empirical_p_at_n100"] = 0.01
    out["survivors"]["parametric_tail_required"] = bool(thresh_p < 0.01)

    # stratify by probe quality
    pr = probe_r_table()
    d2 = d.merge(pr, on=["model", "dataset"], how="left")
    strat = {}
    for thr in (0.0, 0.3, 0.4):
        sub = d2[d2["probe_r"].fillna(-1) > thr] if thr > 0 else d2
        pp = sub["p_parametric"].to_numpy()
        r, _ = bh_fdr(pp)
        strat[f"probe_r_gt_{thr}"] = {
            "n": int(len(sub)),
            "n_pairs": int(sub.groupby(["model", "dataset"]).ngroups),
            "n_survive": int(r.sum()),
            "frac_survive_pct": float(100 * r.mean()) if len(sub) else np.nan,
            "pi0": storey_pi0(pp, 0.5),
            "median_gsi": float(sub["gsi"].median()),
            "median_z": float(sub["z_score"].median()),
        }
    out["probe_viability_stratification"] = strat

    # association between per-enhancer sensitivity and probe quality
    ok = d2["probe_r"].notna()
    pair = (
        d2[ok]
        .groupby(["model", "dataset"])
        .agg(probe_r=("probe_r", "first"), med_z=("z_score", "median"),
             med_gsi=("gsi", "median"), frac_nom=("p_parametric", lambda x: (x < 0.05).mean()))
        .reset_index()
    )
    out["sensitivity_vs_probe_quality"] = {
        "spearman_probe_r_vs_median_z": list(
            map(float, stats.spearmanr(pair["probe_r"], pair["med_z"]))
        ),
        "spearman_probe_r_vs_frac_nominal": list(
            map(float, stats.spearmanr(pair["probe_r"], pair["frac_nom"]))
        ),
        "n_pairs": int(len(pair)),
    }
    pair.to_csv(OUT / "audit_pair_table.csv", index=False)

    # dataset label integrity
    out["dataset_labels"] = {
        "labels_in_census": sorted(d["dataset"].unique().tolist()),
        "file_stems": sorted({f.split("_")[0] for f in d["source_file"]}),
    }
    inoue_ids = set(
        pd.read_parquet(ROOT / "data/processed/inoue_processed.parquet")["seq_id"].astype(str)
    )
    lab = d[d["dataset"] == "de_almeida"]["seq_id"].astype(str)
    out["dataset_labels"]["de_almeida_rows_matching_inoue_ids"] = int(
        len(set(lab) & inoue_ids)
    )
    out["dataset_labels"]["de_almeida_n_rows"] = int(len(lab))

    # per (model,dataset) counts -- checks the "510 enhancers per pair" claim
    counts = d.groupby(["dataset", "model"]).size()
    out["per_pair_counts"] = {f"{a}__{b}": int(c) for (a, b), c in counts.items()}
    out["per_pair_counts_note"] = (
        "7650 = 15 language-model pairs x 500 + 3 Enformer pairs x 50. "
        "The manuscript's '510 enhancers per (model,dataset) pair' is 7650/15 "
        "and does not correspond to any pair's actual size."
    )

    with open(OUT / "audit_census.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
