"""Shared data loading for the rebuttal experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "rebuttal"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["agarwal", "jores", "klein", "vaishnav", "inoue"]
MODELS = ["dnabert2", "nt", "hyenadna"]


def load_dataset(name: str) -> Tuple[pd.DataFrame, List[List[dict]]]:
    df = pd.read_parquet(PROC / f"{name}_processed.parquet")
    hits = pd.read_parquet(PROC / f"{name}_processed_motif_hits.parquet")
    hits["seq_id"] = hits["seq_id"].astype(str)
    df["seq_id"] = df["seq_id"].astype(str)

    by_id: Dict[str, List[dict]] = {}
    cols = ["start", "end", "motif_name", "strand"]
    have = [c for c in cols if c in hits.columns]
    for sid, g in hits.groupby("seq_id"):
        recs = []
        for r in g[have].itertuples(index=False):
            d = dict(zip(have, r))
            d["start"] = int(d["start"])
            d["end"] = int(d["end"])
            d.setdefault("motif_name", "unknown")
            d.setdefault("strand", "+")
            recs.append(d)
        by_id[sid] = recs

    annotations = [by_id.get(sid, []) for sid in df["seq_id"]]
    return df, annotations


def census_ids(dataset: str, model: str) -> List[str]:
    """The exact enhancers used by the submitted census, so new results are
    computed on the same units rather than a fresh sample."""
    fname = "inoue" if dataset == "inoue" else dataset
    p = ROOT / "results" / "v2" / "module1" / f"{fname}_{model}_gsi.parquet"
    if not p.exists():
        return []
    return pd.read_parquet(p)["seq_id"].astype(str).tolist()


def probe_r_table() -> pd.DataFrame:
    """Held-out Pearson r of the submitted mean-pooled ridge probes."""
    rows = []
    with open(ROOT / "results" / "probe_training_all_results.json") as f:
        for rec in json.load(f):
            rows.append(
                {
                    "model": rec["model"],
                    "dataset": rec["dataset"],
                    "probe_r": rec["pearson_r"],
                    "probe_r2": rec["r_squared"],
                    "viable": rec["viable"],
                }
            )
    return pd.DataFrame(rows)


def bh_fdr(p: np.ndarray, alpha: float = 0.05):
    from statsmodels.stats.multitest import multipletests

    ok = ~np.isnan(p)
    rej = np.zeros(len(p), dtype=bool)
    q = np.full(len(p), np.nan)
    if ok.sum():
        r, qq, _, _ = multipletests(p[ok], alpha=alpha, method="fdr_bh")
        rej[ok] = r
        q[ok] = qq
    return rej, q


def storey_pi0(p: np.ndarray, lam: float = 0.5) -> float:
    p = p[~np.isnan(p)]
    return float(min(1.0, (p > lam).mean() / (1 - lam)))


def bootstrap_ci(x: np.ndarray, fn, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    stats_ = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    return float(np.percentile(stats_, 2.5)), float(np.percentile(stats_, 97.5))
