#!/usr/bin/env python
"""
Re-run the per-enhancer census with the corrected spacer-exact arrangement
operators and the full readout ladder, on the same enhancers the submitted
census used.

Perturbed sequences are generated once per enhancer and pushed through the
frozen backbone once, then scored by all five heads, so the head comparison is
exactly controlled: same sequences, same frozen representation, only the
readout differs.

Writes one parquet per (model, dataset) holding, per enhancer and head, the
var/sd/mean of predictions under pi_arrange (spacer-exact), pi_null
(arrangement-fixed) and pi_full (the submitted naive operator), plus the
reference prediction and the permutation test of Var(arrange) > Var(null).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.rebuttal.common import OUT, census_ids, load_dataset  # noqa: E402
from src.grammar.sf_gsi_v2 import (  # noqa: E402
    log_var_ratio_permutation_p,
    order_eligible,
    pi_arrange,
    pi_full,
    pi_null,
    pi_orient,
)
from src.models.backbones import Backbone  # noqa: E402
from src.models.readouts import build_head  # noqa: E402

CENSUS_DIR = OUT / "census_v2"
CENSUS_DIR.mkdir(parents=True, exist_ok=True)


def load_heads(model: str, dataset: str, device: str):
    heads = {}
    for p in sorted((OUT / "heads").glob(f"{model}_{dataset}_*.pt")):
        ck = torch.load(p, map_location=device, weights_only=False)
        h = build_head(ck["kind"], ck["d_in"]).to(device)
        h.load_state_dict(ck["state_dict"])
        h.eval()
        heads[ck["kind"]] = (h, ck["metrics"])
    return heads


@torch.no_grad()
def score_all(backbone, heads, seqs, batch_size=128):
    """One backbone pass; all heads scored off the same hidden states."""
    outs = {k: [] for k in heads}
    for i in range(0, len(seqs), batch_size):
        h, m = backbone._forward_batch(seqs[i : i + batch_size])
        for k, (head, _) in heads.items():
            outs[k].append(head(h, m).squeeze(-1).float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-draws", type=int, default=40)
    ap.add_argument("--n-enhancers", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    df, ann = load_dataset(args.dataset)
    df["seq_id"] = df["seq_id"].astype(str)
    want = census_ids(args.dataset, args.model)
    pos = {s: i for i, s in enumerate(df["seq_id"])}
    if want:
        idx = [pos[s] for s in want if s in pos][: args.n_enhancers]
    else:
        idx = [i for i, m in enumerate(ann) if len(m) >= 2][: args.n_enhancers]
    print(f"[{args.model}/{args.dataset}] {len(idx)} enhancers "
          f"({'reusing submitted census IDs' if want else 'fresh sample'})")

    heads = load_heads(args.model, args.dataset, args.device)
    if not heads:
        raise SystemExit(f"no trained heads for {args.model}/{args.dataset}")
    print("  heads:", {k: round(v[1]["test_r"], 3) for k, v in heads.items()})

    backbone = Backbone(args.model, device=args.device)
    N = args.n_draws
    rows = []
    t0 = time.time()

    for n_done, i in enumerate(idx):
        seq = df.iloc[i]["sequence"]
        mot = ann[i]
        if len(mot) < 2:
            continue
        pert = {
            "arrange": pi_arrange(seq, mot, N, seed=i),
            "orient": pi_orient(seq, mot, N, seed=i + 1),
            "null": pi_null(seq, mot, N, seed=i + 2),
            "full": pi_full(seq, mot, N, seed=i + 3),
        }
        pert = {k: v for k, v in pert.items() if len(v) >= 5}
        if "arrange" not in pert or "null" not in pert:
            continue

        flat, spans = [seq], {}
        for k, v in pert.items():
            spans[k] = (len(flat), len(flat) + len(v))
            flat.extend(v)
        preds = score_all(backbone, heads, flat, args.batch_size)

        for head_name, p in preds.items():
            ref = float(p[0])
            rec = {
                "seq_id": df.iloc[i]["seq_id"],
                "dataset": args.dataset,
                "model": args.model,
                "head": head_name,
                "probe_test_r": heads[head_name][1]["test_r"],
                "perm_invariant": heads[head_name][1]["perm_invariant"],
                "n_motifs": len(mot),
                "reference": ref,
                "order_eligible": order_eligible(seq, mot),
            }
            vecs = {}
            for k, (a, b) in spans.items():
                v = p[a:b].astype(float)
                vecs[k] = v
                rec[f"n_{k}"] = len(v)
                rec[f"mean_{k}"] = float(v.mean())
                rec[f"sd_{k}"] = float(v.std(ddof=1))
                rec[f"var_{k}"] = float(v.var(ddof=1))
            t = log_var_ratio_permutation_p(
                vecs["arrange"], vecs["null"], n_perm=args.n_perm, seed=i
            )
            rec["logvr_stat"] = t["stat"]
            rec["p_perm"] = t["p_perm"]
            rec["p_gpd"] = t["p_gpd"]
            rec["n_exceed"] = t["n_exceed"]
            # submitted-style statistic on the same predictions, for comparison
            sdf = rec["sd_full"]
            rec["z_submitted"] = abs(ref - rec["mean_full"]) / max(sdf, 1e-10)
            rec["gsi_naive"] = sdf / max(abs(rec["mean_full"]), 1e-8)
            rec["sf_gsi"] = rec["sd_arrange"] / max(abs(ref), 1e-8)
            vf = rec["var_full"]
            rec["arrangement_share"] = rec["var_arrange"] / vf if vf > 0 else np.nan
            rec["spacer_share"] = rec["var_null"] / vf if vf > 0 else np.nan
            rows.append(rec)

        if (n_done + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {n_done+1}/{len(idx)}  {el:.0f}s  "
                  f"({el/(n_done+1):.2f}s/enh)", flush=True)

    backbone.unload()
    d = pd.DataFrame(rows)
    fp = CENSUS_DIR / f"{args.model}_{args.dataset}_census_v2.parquet"
    d.to_parquet(fp, index=False)
    print(f"  wrote {fp} ({len(d)} rows)")

    summ = (
        d.groupby("head")
        .agg(
            n=("seq_id", "size"),
            median_sf_gsi=("sf_gsi", "median"),
            median_gsi_naive=("gsi_naive", "median"),
            median_arrangement_share=("arrangement_share", "median"),
            median_spacer_share=("spacer_share", "median"),
            frac_p_perm_lt_05=("p_perm", lambda x: float((x < 0.05).mean())),
            probe_r=("probe_test_r", "first"),
        )
        .reset_index()
    )
    print(summ.to_string(index=False))
    summ.to_json(OUT / f"census_v2_summary_{args.model}_{args.dataset}.json",
                 orient="records", indent=2)


if __name__ == "__main__":
    main()
