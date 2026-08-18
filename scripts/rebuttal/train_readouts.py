#!/usr/bin/env python
"""
Train the readout ladder (mean_linear through transformer) on frozen
token-level embeddings for one (model, dataset) pair.

The backbone and its frozen embeddings are identical across heads; the only
thing that varies is how much arrangement information the readout can express.

Usage:
    python scripts/rebuttal/train_readouts.py --model dnabert2 --dataset jores
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.rebuttal.common import OUT, load_dataset  # noqa: E402
from src.models.backbones import Backbone  # noqa: E402
from src.models.readouts import HEADS, build_head  # noqa: E402

HEAD_DIR = OUT / "heads"
HEAD_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR = OUT / "token_embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)


def get_token_embeddings(backbone: Backbone, seqs, cache: Path):
    if cache.exists():
        z = np.load(cache)
        return z["H"], z["M"]
    H, M = backbone.token_embeddings(seqs, batch_size=32)
    np.savez_compressed(cache, H=H, M=M)
    return H, M


def train_one(kind, Ht, Mt, yt, y, idx_tr, idx_va, idx_te, device, epochs=120, seed=0):
    torch.manual_seed(seed)
    d_in = Ht.shape[2]
    head = build_head(kind, d_in).to(device)
    # weight decay is the ridge penalty for the mean_linear baseline
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.MSELoss()

    best, best_state, patience = -np.inf, None, 0
    bs = 128
    idx_tr_t = torch.as_tensor(idx_tr, device=Ht.device)
    for ep in range(epochs):
        head.train()
        perm = idx_tr_t[torch.randperm(len(idx_tr_t), device=Ht.device)]
        for i in range(0, len(perm), bs):
            b = perm[i : i + bs]
            xb = Ht[b].float()
            mb = Mt[b]
            yb = yt[b]
            opt.zero_grad()
            loss = lossf(head(xb, mb).squeeze(-1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            pv = []
            for i in range(0, len(idx_va), 512):
                b = idx_va[i : i + 512]
                pv.append(head(Ht[b].float(), Mt[b]).squeeze(-1).cpu().numpy())
            pv = np.concatenate(pv)
        r = stats.pearsonr(pv, y[idx_va])[0] if np.std(pv) > 1e-9 else -1.0
        if r > best:
            best, best_state, patience = r, {k: v.detach().clone() for k, v in head.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 25:
                break

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pt = []
        for i in range(0, len(idx_te), 512):
            b = idx_te[i : i + 512]
            pt.append(head(Ht[b].float(), Mt[b]).squeeze(-1).cpu().numpy())
        pt = np.concatenate(pt)
    r_te = stats.pearsonr(pt, y[idx_te])[0] if np.std(pt) > 1e-9 else 0.0
    rho_te = stats.spearmanr(pt, y[idx_te])[0] if np.std(pt) > 1e-9 else 0.0
    return head, {
        "head": kind,
        "val_r": float(best),
        "test_r": float(r_te),
        "test_rho": float(rho_te),
        "test_r2": float(1 - np.mean((pt - y[idx_te]) ** 2) / np.var(y[idx_te])),
        "n_params": int(sum(p.numel() for p in head.parameters())),
        "perm_invariant": bool(HEADS[kind].perm_invariant),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    df, _ = load_dataset(args.dataset)
    df = df.head(args.n_train).reset_index(drop=True)
    seqs = df["sequence"].tolist()
    y = df["expression"].to_numpy(dtype=np.float32)
    y = (y - y.mean()) / (y.std() + 1e-8)

    cache = EMB_DIR / f"{args.model}_{args.dataset}_tokens.npz"
    print(f"[{args.model}/{args.dataset}] {len(seqs)} sequences; embeddings -> {cache.name}")
    backbone = Backbone(args.model, device=args.device)
    H, M = get_token_embeddings(backbone, seqs, cache)
    backbone.unload()
    print(f"  token embeddings {H.shape}", flush=True)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(seqs))
    n_tr, n_va = int(0.7 * len(idx)), int(0.15 * len(idx))
    idx_tr, idx_va, idx_te = idx[:n_tr], idx[n_tr : n_tr + n_va], idx[n_tr + n_va :]

    # keep the frozen embeddings resident on the GPU: the heads are tiny and
    # the host->device copy otherwise dominates wall-clock
    Ht = torch.from_numpy(H).to(args.device)          # float16
    Mt = torch.from_numpy(M).float().to(args.device)
    yt = torch.from_numpy(y).float().to(args.device)

    metrics = []
    for kind in HEADS:
        head, m = train_one(kind, Ht, Mt, yt, y, idx_tr, idx_va, idx_te,
                            args.device, args.epochs)
        m.update(model=args.model, dataset=args.dataset)
        metrics.append(m)
        torch.save(
            {"state_dict": head.state_dict(), "kind": kind, "d_in": H.shape[2], "metrics": m},
            HEAD_DIR / f"{args.model}_{args.dataset}_{kind}.pt",
        )
        print(f"  {kind:12s} test_r={m['test_r']:.3f}  test_r2={m['test_r2']:.3f}  "
              f"params={m['n_params']:,}  perm_inv={m['perm_invariant']}", flush=True)

    with open(OUT / f"readout_metrics_{args.model}_{args.dataset}.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
