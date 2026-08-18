#!/usr/bin/env python
"""
SF-GSI v2 on PARM, end to end.

PARM is trained directly on MPRA libraries and has its own expression head, so
it needs no ridge probe and no mean pooling. Running the corrected census on it
removes both readout confounds at once: whatever comes out is a property of a
supervised regulatory-sequence model, not of our probing choices.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "PARM"))

from scripts.rebuttal.common import OUT, bh_fdr, census_ids, load_dataset, storey_pi0  # noqa: E402
from src.grammar.sf_gsi_v2 import (  # noqa: E402
    log_var_ratio_permutation_p, pi_arrange, pi_full, pi_null,
)

PARM_DIR = ROOT / "tools" / "PARM" / "pre_trained_models"


class PARMEnsemble:
    def __init__(self, cell_type: str, device="cuda"):
        from PARM.PARM_utils_load_model import load_PARM

        self.device = device
        self.models = []
        for p in sorted((PARM_DIR / cell_type).glob("*.parm")) or sorted(
            (PARM_DIR / cell_type).glob("*.pt")
        ):
            m = load_PARM(str(p), train=False)
            m.to(device).eval()
            self.models.append(m)
        if not self.models:
            raise SystemExit(f"no PARM checkpoints under {PARM_DIR/cell_type}")
        self.name = f"PARM-{cell_type}"

    @torch.no_grad()
    def predict(self, seqs, batch_size=256):
        from PARM.PARM_predict import get_prediction

        preds = []
        for m in self.models:
            out = []
            for i in range(0, len(seqs), batch_size):
                out.append(np.asarray(get_prediction(seqs[i : i + batch_size], m)).ravel())
            preds.append(np.concatenate(out))
        return np.mean(preds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="agarwal")
    ap.add_argument("--cell-type", default="K562")
    ap.add_argument("--n-enhancers", type=int, default=200)
    ap.add_argument("--n-draws", type=int, default=40)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    df, ann = load_dataset(args.dataset)
    df["seq_id"] = df["seq_id"].astype(str)
    pos = {s: i for i, s in enumerate(df["seq_id"])}
    want = census_ids(args.dataset, "dnabert2")
    idx = [pos[s] for s in want if s in pos] or list(range(len(df)))
    idx = [i for i in idx if len(ann[i]) >= 2][: args.n_enhancers]

    model = PARMEnsemble(args.cell_type, args.device)
    print(f"{model.name}: {len(model.models)} checkpoints, {len(idx)} enhancers", flush=True)

    rows = []
    for c, i in enumerate(idx):
        seq, mot = df.iloc[i]["sequence"], ann[i]
        arr = pi_arrange(seq, mot, args.n_draws, seed=i)
        nul = pi_null(seq, mot, args.n_draws, seed=i + 1)
        ful = pi_full(seq, mot, args.n_draws, seed=i + 2)
        if len(arr) < 5 or len(nul) < 5:
            continue
        p = model.predict([seq] + arr + nul + ful)
        ref = float(p[0])
        a = p[1 : 1 + len(arr)].astype(float)
        n_ = p[1 + len(arr) : 1 + len(arr) + len(nul)].astype(float)
        f = p[1 + len(arr) + len(nul) :].astype(float)
        t = log_var_ratio_permutation_p(a, n_, n_perm=args.n_perm, seed=i)
        rows.append(dict(
            seq_id=df.iloc[i]["seq_id"], model=model.name, dataset=args.dataset,
            reference=ref, n_motifs=len(mot),
            sd_arrange=a.std(ddof=1), sd_null=n_.std(ddof=1), sd_full=f.std(ddof=1),
            var_arrange=a.var(ddof=1), var_null=n_.var(ddof=1), var_full=f.var(ddof=1),
            sf_gsi=a.std(ddof=1) / max(abs(ref), 1e-8),
            gsi_naive=f.std(ddof=1) / max(abs(f.mean()), 1e-8),
            arrangement_share=a.var(ddof=1) / max(f.var(ddof=1), 1e-12),
            spacer_share=n_.var(ddof=1) / max(f.var(ddof=1), 1e-12),
            p_perm=t["p_perm"], p_gpd=t["p_gpd"],
            z_submitted=abs(ref - f.mean()) / max(f.std(ddof=1), 1e-10),
        ))
        if (c + 1) % 25 == 0:
            print(f"  {c+1}/{len(idx)}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"parm_census_{args.dataset}_{args.cell_type}.csv", index=False)
    rej, q = bh_fdr(d["p_perm"].to_numpy())
    summ = {
        "model": model.name, "dataset": args.dataset, "n": int(len(d)),
        "median_sf_gsi": float(d["sf_gsi"].median()),
        "median_gsi_naive": float(d["gsi_naive"].median()),
        "median_arrangement_share": float(d["arrangement_share"].median()),
        "median_spacer_share": float(d["spacer_share"].median()),
        "frac_p_perm_lt_0.05": float((d["p_perm"] < 0.05).mean()),
        "frac_survive_bh": float(rej.mean()),
        "storey_pi0_billboard_fraction": storey_pi0(d["p_perm"].to_numpy()),
        "note": "no probe and no pooling: PARM's own expression head is used",
    }
    with open(OUT / f"parm_census_{args.dataset}_{args.cell_type}.json", "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
