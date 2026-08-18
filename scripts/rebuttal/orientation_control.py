#!/usr/bin/env python
"""
Orientation positive control, redone with effect measures tied to predictive
utility.

The submitted control reported Cohen's d for |delta| under a strand flip and
got d ~ 11 for NT and HyenaDNA even though 0% of pairs exceeded a 0.05
absolute-difference threshold. A huge standardized effect off a nearly
degenerate output distribution is not orientation discrimination, it just means
the model emits almost the same offset every time. (Identical d and t to 12
significant figures for two different models is itself a sign the same array
got scored twice.)

Four measures that cannot be inflated that way:

  signed delta      change in prediction on the model's native scale, mean and SD
  dynamic range     SD of predictions across real enhancers, so the flip
                    response can be read as a fraction of the range
  arrangement vs    AUROC of |delta_flip| > |delta_spacer|, where delta_spacer
  composition       comes from an arrangement-fixed spacer shuffle. 0.5 means
                    the model reacts no more to an orientation change than to a
                    composition change
  pair specificity  how much of the variance in delta_flip is explained by which
                    and how many motifs were flipped. a constant offset scores
                    ~0 here however large its d
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
from scripts.rebuttal.common import OUT, census_ids, load_dataset  # noqa: E402
from src.grammar.sf_gsi_v2 import _merged, pi_null  # noqa: E402
from src.models.backbones import Backbone  # noqa: E402
from src.models.readouts import build_head  # noqa: E402
from src.utils.sequence import reverse_complement  # noqa: E402


def flip_all(seq: str, motifs):
    """Reverse-complement every motif in place; count how many were flipped."""
    merged = _merged(seq, motifs)
    s = list(seq)
    for m in merged:
        s[m["start"] : m["end"]] = list(reverse_complement(seq[m["start"] : m["end"]]))
    return "".join(s), len(merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="agarwal")
    ap.add_argument("--models", nargs="+", default=["dnabert2", "nt", "hyenadna"])
    ap.add_argument("--heads", nargs="+", default=["mean_linear", "cnn1d"])
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    df, ann = load_dataset(args.dataset)
    df["seq_id"] = df["seq_id"].astype(str)
    pos = {s: i for i, s in enumerate(df["seq_id"])}

    out = {"dataset": args.dataset, "models": {}}
    rows = []

    for model in args.models:
        want = census_ids(args.dataset, model)
        idx = [pos[s] for s in want if s in pos][: args.n_pairs] or [
            i for i, m in enumerate(ann) if len(m) >= 2
        ][: args.n_pairs]
        idx = [i for i in idx if len(ann[i]) >= 2]

        originals, flipped, spacer_ctl, nflip = [], [], [], []
        for i in idx:
            seq, mot = df.iloc[i]["sequence"], ann[i]
            f, k = flip_all(seq, mot)
            sp = pi_null(seq, mot, 1, seed=i)
            if not sp:
                continue
            originals.append(seq)
            flipped.append(f)
            spacer_ctl.append(sp[0])
            nflip.append(k)
        nflip = np.array(nflip)

        backbone = Backbone(model, device=args.device)
        for head_name in args.heads:
            hp = OUT / "heads" / f"{model}_{args.dataset}_{head_name}.pt"
            if not hp.exists():
                continue
            ck = torch.load(hp, map_location=args.device, weights_only=False)
            head = build_head(ck["kind"], ck["d_in"]).to(args.device)
            head.load_state_dict(ck["state_dict"])
            head.eval()

            p0 = backbone.predict_with_head(originals, head, 128).astype(float)
            p1 = backbone.predict_with_head(flipped, head, 128).astype(float)
            p2 = backbone.predict_with_head(spacer_ctl, head, 128).astype(float)
            # dynamic range: predictions on unperturbed enhancers
            pop_sd = float(p0.std(ddof=1))

            d_flip = p1 - p0
            d_spac = p2 - p0
            u = stats.mannwhitneyu(np.abs(d_flip), np.abs(d_spac), alternative="greater")
            auroc = float(u.statistic / (len(d_flip) * len(d_spac)))

            # pair specificity: does delta_flip depend on what was flipped?
            X = np.column_stack([np.ones(len(nflip)), nflip, [len(s) for s in originals]])
            beta, *_ = np.linalg.lstsq(X, d_flip, rcond=None)
            resid = d_flip - X @ beta
            r2_pairspec = float(1 - resid.var() / max(d_flip.var(), 1e-30))
            # a constant offset has near-zero relative dispersion
            cv = float(np.std(d_flip) / max(abs(np.mean(d_flip)), 1e-12))

            rec = {
                "model": model, "head": head_name, "n_pairs": int(len(d_flip)),
                "signed_delta_mean": float(d_flip.mean()),
                "signed_delta_sd": float(d_flip.std(ddof=1)),
                "signed_delta_q05": float(np.percentile(d_flip, 5)),
                "signed_delta_q95": float(np.percentile(d_flip, 95)),
                "abs_delta_mean": float(np.abs(d_flip).mean()),
                "population_prediction_sd": pop_sd,
                "delta_sd_over_population_sd": float(d_flip.std(ddof=1) / max(pop_sd, 1e-12)),
                "abs_delta_mean_over_population_sd": float(
                    np.abs(d_flip).mean() / max(pop_sd, 1e-12)
                ),
                "spacer_control_abs_delta_mean": float(np.abs(d_spac).mean()),
                "auroc_flip_gt_spacer": auroc,
                "auroc_p": float(u.pvalue),
                "pair_specificity_r2": r2_pairspec,
                "coefficient_of_variation_of_delta": cv,
                # the submitted statistic, for comparison only
                "submitted_cohens_d_on_abs_delta": float(
                    np.abs(d_flip).mean() / max(np.abs(d_flip).std(ddof=1), 1e-12)
                ),
                "frac_abs_delta_gt_0.05_of_pop_sd": float(
                    (np.abs(d_flip) > 0.05 * pop_sd).mean()
                ),
            }
            rows.append(rec)
            out["models"].setdefault(model, {})[head_name] = rec
            print(f"{model:9s} {head_name:12s} "
                  f"signed d mean={rec['signed_delta_mean']:+.4f} sd={rec['signed_delta_sd']:.4f} "
                  f"| pop sd={pop_sd:.4f} | delta/pop={rec['delta_sd_over_population_sd']:.3f} "
                  f"| AUROC(flip>spacer)={auroc:.3f} | pair-spec R2={r2_pairspec:.3f} "
                  f"| submitted d={rec['submitted_cohens_d_on_abs_delta']:.2f}", flush=True)
        backbone.unload()

    pd.DataFrame(rows).to_csv(OUT / f"orientation_control_{args.dataset}.csv", index=False)
    out["interpretation"] = (
        "AUROC(|delta_flip| > |delta_spacer|) near 0.5 means the model does not "
        "treat an orientation change as more consequential than a "
        "composition-preserving spacer change. A large submitted-style Cohen's d "
        "alongside a near-zero pair-specificity R2 identifies a constant offset "
        "rather than orientation discrimination."
    )
    with open(OUT / f"orientation_control_{args.dataset}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
