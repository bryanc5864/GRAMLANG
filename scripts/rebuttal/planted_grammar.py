#!/usr/bin/env python
"""
Positive control: does GRAMLANG call an arrangement-aware model
arrangement-aware?

The submitted paper only ever pushed synthetic stimuli through the evaluated
models. That shows the stimuli carry arrangement signal, not that the SF-GSI
pipeline can separate an arrangement-aware predictor from an
arrangement-invariant one. Without that, "the models do not encode grammar" and
"the evaluation cannot recover grammar" look the same.

So build both on one synthetic library:

  billboard generator   y = sum of motif weights                 (+ noise)
  grammar   generator   y = same vocabulary term
                          + order bonus (A before B)
                          + orientation bonus (same strand)
                          + helical bonus cos(2*pi*d/10.5) on A-B spacing

An identical CNN is trained on each and both go through the same census. A
benchmark that works has to call the grammar-trained CNN arrangement-sensitive
and the billboard-trained one arrangement-invariant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rebuttal.common import OUT, bh_fdr, storey_pi0  # noqa: E402
from src.grammar.sf_gsi_v2 import (  # noqa: E402
    log_var_ratio_permutation_p, pi_arrange, pi_full, pi_null,
)
from src.utils.sequence import one_hot_encode, reverse_complement  # noqa: E402

MOTIFS = {
    "A": "TGASTCA",      # AP-1 like
    "B": "CACGTG",       # E-box
    "C": "GATAAG",       # GATA
    "D": "TGCCAAG",      # RUNX-like
    "E": "CCGGAAGT",     # ETS-like
    "F": "TTGCGCAA",     # CEBP-like
}
W = {"A": 0.9, "B": 0.6, "C": -0.5, "D": 0.35, "E": 0.8, "F": -0.7}
SEQ_LEN = 200
IUPAC = {"S": "CG"}


def realize(m: str, rng) -> str:
    return "".join(rng.choice(list(IUPAC[c])) if c in IUPAC else c for c in m)


def make_library(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        bg = "".join(rng.choice(list("ACGT"), SEQ_LEN, p=[0.29, 0.21, 0.21, 0.29]))
        s = list(bg)
        k = rng.integers(2, 5)
        names = rng.choice(list(MOTIFS), k, replace=True)
        placed, used = [], []
        for nm in names:
            core = realize(MOTIFS[nm], rng)
            strand = "+" if rng.random() < 0.5 else "-"
            sub = core if strand == "+" else reverse_complement(core)
            for _ in range(40):
                p = int(rng.integers(5, SEQ_LEN - len(sub) - 5))
                if all(p + len(sub) + 4 < a or p > b + 4 for a, b in used):
                    break
            else:
                continue
            used.append((p, p + len(sub)))
            s[p : p + len(sub)] = list(sub)
            placed.append({"start": p, "end": p + len(sub), "motif_name": nm,
                           "strand": strand})
        recs.append({"sequence": "".join(s), "motifs": sorted(placed, key=lambda m: m["start"])})
    return recs


def y_billboard(rec) -> float:
    return float(sum(W[m["motif_name"]] for m in rec["motifs"]))


def _arrangement_terms(ms) -> float:
    """Order, orientation and helical-phase terms only."""
    if len(ms) < 2:
        return 0.0
    y = 0.0
    for i in range(len(ms) - 1):
        a, b = ms[i], ms[i + 1]
        if a["motif_name"] <= b["motif_name"]:
            y += 1.5                                    # order rule
        if a["strand"] == b["strand"]:
            y += 1.5                                    # orientation rule
        d = b["start"] - a["end"]
        y += 1.0 * np.cos(2 * np.pi * d / 10.5)         # helical phasing
    return float(y / (len(ms) - 1))


def y_grammar(rec) -> float:
    """Mixed: vocabulary plus arrangement."""
    return float(sum(W[m["motif_name"]] for m in rec["motifs"])
                 + 2.0 * _arrangement_terms(rec["motifs"]))


def y_pure_arrangement(rec) -> float:
    """Arrangement only. Motif identity carries no information, so anything that
    predicts this above chance has learned arrangement.
    """
    return _arrangement_terms(rec["motifs"])


def y_orientation_only(rec) -> float:
    """Floor-level positive control: fraction of motifs on the + strand.

    The easiest arrangement rule there is. Invisible to motif identity (a flip
    preserves the multiset) and a CNN can express it with strand-specific
    filters. If a model that predicts this well is still scored as a billboard,
    the benchmark is broken; if it scores as arrangement-sensitive, a null on a
    harder target means the model failed, not the benchmark.
    """
    ms = rec["motifs"]
    if not ms:
        return 0.0
    return float(sum(m["strand"] == "+" for m in ms) / len(ms))


class SeqCNN(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, ch, 11, padding=5), nn.BatchNorm1d(ch), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(ch, ch, 7, padding=6, dilation=2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, 7, padding=12, dilation=4), nn.BatchNorm1d(ch), nn.GELU(),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.LazyLinear(128), nn.GELU(),
                                nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, x):
        return self.fc(self.net(x))


def encode(seqs, device):
    X = np.stack([one_hot_encode(s[:SEQ_LEN].ljust(SEQ_LEN, "N")) for s in seqs])
    return torch.from_numpy(X).float().permute(0, 2, 1).to(device)


def train_cnn(seqs, y, device, epochs=80, seed=0):
    torch.manual_seed(seed)
    model = SeqCNN().to(device)
    X = encode(seqs, "cpu")
    yt = torch.from_numpy(((y - y.mean()) / y.std()).astype(np.float32))
    n = len(seqs)
    idx = np.random.default_rng(0).permutation(n)
    tr, va = idx[: int(0.85 * n)], idx[int(0.85 * n) :]
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.MSELoss()
    best = -np.inf
    # materialise the LazyLinear, then seed with the initial weights: if
    # validation r is NaN on every epoch (constant predictions) we must still
    # return a usable model rather than crash the whole run
    with torch.no_grad():
        model(X[:2].to(device))
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), 128):
            b = perm[i : i + 128]
            opt.zero_grad()
            loss = lossf(model(X[b].to(device)).squeeze(-1), yt[b].to(device))
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = np.concatenate([
                model(X[va[i:i+512]].to(device)).squeeze(-1).cpu().numpy()
                for i in range(0, len(va), 512)])
        r = stats.pearsonr(pv, yt[va].numpy())[0] if np.std(pv) > 1e-8 else -np.inf
        if np.isfinite(r) and r > best:
            best = r
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, float(best)


@torch.no_grad()
def predict(model, seqs, device, bs=512):
    out = []
    for i in range(0, len(seqs), bs):
        out.append(model(encode(seqs[i : i + bs], device)).squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def census(model, recs, device, n_draws=40, n_perm=5000, label=""):
    rows = []
    for i, rec in enumerate(recs):
        seq, mot = rec["sequence"], rec["motifs"]
        if len(mot) < 2:
            continue
        arr = pi_arrange(seq, mot, n_draws, seed=i)
        nul = pi_null(seq, mot, n_draws, seed=i + 1)
        ful = pi_full(seq, mot, n_draws, seed=i + 2)
        if len(arr) < 5 or len(nul) < 5:
            continue
        flat = [seq] + arr + nul + ful
        p = predict(model, flat, device)
        ref = float(p[0])
        a = p[1 : 1 + len(arr)].astype(float)
        nn_ = p[1 + len(arr) : 1 + len(arr) + len(nul)].astype(float)
        f = p[1 + len(arr) + len(nul) :].astype(float)
        t = log_var_ratio_permutation_p(a, nn_, n_perm=n_perm, seed=i)
        rows.append(dict(
            model=label, reference=ref,
            sd_arrange=a.std(ddof=1), sd_null=nn_.std(ddof=1), sd_full=f.std(ddof=1),
            var_arrange=a.var(ddof=1), var_null=nn_.var(ddof=1), var_full=f.var(ddof=1),
            sf_gsi=a.std(ddof=1) / max(abs(ref), 1e-8),
            gsi_naive=f.std(ddof=1) / max(abs(f.mean()), 1e-8),
            arrangement_share=a.var(ddof=1) / max(f.var(ddof=1), 1e-12),
            spacer_share=nn_.var(ddof=1) / max(f.var(ddof=1), 1e-12),
            p_perm=t["p_perm"], p_gpd=t["p_gpd"],
            z_submitted=abs(ref - f.mean()) / max(f.std(ddof=1), 1e-10),
        ))
    return pd.DataFrame(rows)


def helical_scan(model, device, n_per=40, seed=0):
    """Can the model see 10.5-bp phasing between two fixed motifs?"""
    rng = np.random.default_rng(seed)
    a, b = MOTIFS["A"].replace("S", "C"), MOTIFS["B"]
    spacings = list(range(4, 41))
    means, allv = {}, []
    for d in spacings:
        seqs = []
        for _ in range(n_per):
            bg = list("".join(rng.choice(list("ACGT"), SEQ_LEN)))
            p0 = 60
            bg[p0 : p0 + len(a)] = list(a)
            p1 = p0 + len(a) + d
            bg[p1 : p1 + len(b)] = list(b)
            seqs.append("".join(bg))
        pr = predict(model, seqs, device)
        means[d] = float(pr.mean())
        allv.append(pr)
    y = np.array([means[d] for d in spacings])
    x = np.array(spacings)
    # is a 10.5-bp sinusoid a better fit than a constant?
    Xd = np.column_stack([np.ones_like(x, float), x.astype(float),
                          np.cos(2 * np.pi * x / 10.5), np.sin(2 * np.pi * x / 10.5)])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    rss_full = float(np.sum((y - Xd @ beta) ** 2))
    X0 = Xd[:, :2]
    b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    rss_red = float(np.sum((y - X0 @ b0) ** 2))
    dfn, dfd = 2, len(x) - 4
    F = ((rss_red - rss_full) / dfn) / (rss_full / dfd) if rss_full > 0 else np.inf
    return {
        "spacings": spacings,
        "mean_prediction": [means[d] for d in spacings],
        "amplitude_10.5bp": float(np.hypot(beta[2], beta[3])),
        "residual_sd": float(np.std(y - Xd @ beta)),
        "F": float(F),
        "p_value": float(stats.f.sf(F, dfn, dfd)),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("building synthetic library ...")
    train_recs = make_library(30000, seed=0)
    test_recs = make_library(400, seed=99)

    seqs = [r["sequence"] for r in train_recs]
    targets = {
        "billboard": np.array([y_billboard(r) for r in train_recs]),
        "grammar": np.array([y_grammar(r) for r in train_recs]),
        "pure_arrangement": np.array([y_pure_arrangement(r) for r in train_recs]),
        "orientation_only": np.array([y_orientation_only(r) for r in train_recs]),
    }
    # how much of each target is reachable from vocabulary alone?
    yb = targets["billboard"]
    ground = {
        f"vocab_r2_of_{k}_target": float(stats.pearsonr(yb, v)[0] ** 2)
        for k, v in targets.items()
    }
    ground["note"] = (
        "vocab_r2_of_pure_arrangement_target should be ~0: a model that predicts "
        "that target above chance cannot be doing so from motif identity."
    )

    out = {"ground_truth": ground, "models": {}}
    frames = []
    for name, y in targets.items():
        print(f"training CNN on {name} target ...")
        model, r = train_cnn(seqs, y, device)
        print(f"  held-out r = {r:.3f}")
        d = census(model, test_recs, device, label=name)
        frames.append(d)
        rej, q = bh_fdr(d["p_perm"].to_numpy())
        hel = helical_scan(model, device)
        out["models"][name] = {
            "heldout_r": r,
            "n_enhancers": int(len(d)),
            "median_sf_gsi": float(d["sf_gsi"].median()),
            "median_gsi_naive": float(d["gsi_naive"].median()),
            "median_arrangement_share": float(d["arrangement_share"].median()),
            "median_spacer_share": float(d["spacer_share"].median()),
            "median_sd_arrange": float(d["sd_arrange"].median()),
            "median_sd_null": float(d["sd_null"].median()),
            "frac_p_perm_lt_0.05": float((d["p_perm"] < 0.05).mean()),
            "frac_survive_bh": float(rej.mean()),
            "storey_pi0": storey_pi0(d["p_perm"].to_numpy()),
            "billboard_fraction_pct": 100 * storey_pi0(d["p_perm"].to_numpy()),
            "helical": hel,
        }
        print(f"  SF-GSI median={out['models'][name]['median_sf_gsi']:.4f}  "
              f"arrangement share={out['models'][name]['median_arrangement_share']:.3f}  "
              f"frac sig={out['models'][name]['frac_p_perm_lt_0.05']:.3f}  "
              f"helical p={hel['p_value']:.3g}")

    D = pd.concat(frames, ignore_index=True)
    D.to_csv(OUT / "planted_grammar_census.csv", index=False)
    b = D[D["model"] == "billboard"]["sd_arrange"]
    out["discrimination"] = {}
    for name in ("grammar", "pure_arrangement", "orientation_only"):
        g = D[D["model"] == name]["sd_arrange"]
        if not len(g):
            continue
        u = stats.mannwhitneyu(g, b, alternative="greater")
        out["discrimination"][f"{name}_vs_billboard"] = {
            "mannwhitney_p": float(u.pvalue),
            "median_ratio_sd_arrange": float(g.median() / max(b.median(), 1e-12)),
            "auroc_sd_arrange": float(u.statistic / (len(g) * len(b))),
        }
    out["benchmark_verdict"] = (
        "GRAMLANG discriminates an arrangement-aware model from a billboard "
        "model if and only if the pure_arrangement row shows AUROC well above "
        "0.5, a materially higher significant fraction, and a detectable "
        "10.5-bp helical amplitude."
    )
    with open(OUT / "planted_grammar.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["discrimination"], indent=2))


if __name__ == "__main__":
    main()
