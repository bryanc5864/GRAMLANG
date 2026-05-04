"""
Validation Experiment I: Cross-Dataset Replication of Grammar Rules
===================================================================
Tests whether GSI patterns from the GRAMLANG project replicate across MPRA
datasets measuring the same biology.

Hypothesis:
- Human datasets (agarwal, klein, inoue) should show similar GSI patterns
- Cross-species comparisons (human vs yeast vs plant) should NOT correlate

Approach:
Since seq_ids do not overlap between datasets, we use:
1. Motif-feature-conditioned GSI comparison (binned by n_motifs and motif_density)
2. GSI distribution comparison (KS tests, Mann-Whitney U)
3. GSI~motif_feature regression slope comparison across datasets
4. Cohen's d effect sizes for within-species vs cross-species pairs
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
DATA_DIR = "./results/v2/module1/"
OUTPUT_PATH = "./results/v2/module1/cross_dataset_validation.json"

DATASETS = ["agarwal", "inoue", "klein", "vaishnav", "jores"]
MODELS = ["dnabert2", "nt", "hyenadna"]
HUMAN_DATASETS = ["agarwal", "inoue", "klein"]
SPECIES_MAP = {
    "agarwal": "human", "inoue": "human", "klein": "human",
    "vaishnav": "yeast", "jores": "plant"
}

# ──────────────────────────────────────────────────────────────────────
# Load all data
# ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("VALIDATION EXPERIMENT I: Cross-Dataset GSI Replication")
print("=" * 70)

data = {}
for ds in DATASETS:
    for model in MODELS:
        fname = f"{ds}_{model}_gsi.parquet"
        fpath = os.path.join(DATA_DIR, fname)
        df = pd.read_parquet(fpath)
        data[(ds, model)] = df
        print(f"  Loaded {fname}: {len(df)} sequences")

print(f"\nTotal files loaded: {len(data)}")

results = {
    "experiment": "Validation Experiment I - Cross-Dataset GSI Replication",
    "description": "Tests whether grammar rules replicate across MPRA datasets measuring the same biology",
    "datasets": {ds: SPECIES_MAP[ds] for ds in DATASETS},
    "models": MODELS,
    "n_sequences_per_file": 500,
}

# ──────────────────────────────────────────────────────────────────────
# Step 1: Check seq_id overlap (confirm no direct overlap)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 1: Sequence ID Overlap Check")
print("=" * 70)

overlap_results = {}
for model in MODELS:
    for ds1, ds2 in combinations(DATASETS, 2):
        ids1 = set(data[(ds1, model)]["seq_id"])
        ids2 = set(data[(ds2, model)]["seq_id"])
        overlap = len(ids1 & ids2)
        key = f"{ds1}_vs_{ds2}"
        overlap_results.setdefault(model, {})[key] = overlap
        if overlap > 0:
            print(f"  {model} | {ds1} vs {ds2}: {overlap} shared seq_ids")

if all(v == 0 for m in overlap_results.values() for v in m.values()):
    print("  No seq_id overlap found between any dataset pair.")
    print("  -> Using rank-based / distributional approach instead.")

results["seq_id_overlap"] = overlap_results

# ──────────────────────────────────────────────────────────────────────
# Step 2: GSI Distribution Comparison
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: GSI Distribution Comparison (KS test & Mann-Whitney U)")
print("=" * 70)

distribution_results = {}
for model in MODELS:
    print(f"\n  Model: {model}")
    model_results = {}
    for ds1, ds2 in combinations(DATASETS, 2):
        gsi1 = data[(ds1, model)]["gsi"].values
        gsi2 = data[(ds2, model)]["gsi"].values

        # KS test: are the distributions different?
        ks_stat, ks_p = stats.ks_2samp(gsi1, gsi2)
        # Mann-Whitney U: are the central tendencies different?
        mw_stat, mw_p = stats.mannwhitneyu(gsi1, gsi2, alternative="two-sided")

        pair_type = "within_species" if SPECIES_MAP[ds1] == SPECIES_MAP[ds2] else "cross_species"
        species_pair = f"{SPECIES_MAP[ds1]}_vs_{SPECIES_MAP[ds2]}"

        model_results[f"{ds1}_vs_{ds2}"] = {
            "pair_type": pair_type,
            "species_pair": species_pair,
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": float(f"{ks_p:.2e}"),
            "mannwhitney_statistic": float(mw_stat),
            "mannwhitney_p_value": float(f"{mw_p:.2e}"),
            "mean_gsi_ds1": round(float(np.mean(gsi1)), 4),
            "mean_gsi_ds2": round(float(np.mean(gsi2)), 4),
            "median_gsi_ds1": round(float(np.median(gsi1)), 4),
            "median_gsi_ds2": round(float(np.median(gsi2)), 4),
        }

        flag = "***" if ks_p < 0.001 else ("**" if ks_p < 0.01 else ("*" if ks_p < 0.05 else ""))
        print(f"    {ds1:12s} vs {ds2:12s} [{pair_type:14s}] KS={ks_stat:.3f} p={ks_p:.2e} {flag}")

    distribution_results[model] = model_results

results["gsi_distribution_comparison"] = distribution_results

# ──────────────────────────────────────────────────────────────────────
# Step 3: Motif-Feature-Conditioned GSI Correlation (Rank-Based)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Motif-Conditioned GSI Patterns (Rank-Based Approach)")
print("=" * 70)
print("  For each model, bin sequences by n_motifs, compute mean GSI per bin,")
print("  then correlate these bin-level profiles across datasets.")

motif_conditioned_results = {}
for model in MODELS:
    print(f"\n  Model: {model}")
    model_results = {}

    # Create common bins for n_motifs across all datasets
    all_n_motifs = pd.concat([data[(ds, model)]["n_motifs"] for ds in DATASETS])
    # Use quantile-based bins to ensure comparable bin sizes
    bin_edges = np.percentile(all_n_motifs, [0, 20, 40, 60, 80, 100])
    bin_edges = np.unique(bin_edges)  # Remove duplicates
    bin_labels = [f"bin_{i}" for i in range(len(bin_edges) - 1)]

    # Compute mean GSI per n_motifs bin for each dataset
    bin_profiles = {}
    for ds in DATASETS:
        df = data[(ds, model)].copy()
        df["motif_bin"] = pd.cut(df["n_motifs"], bins=bin_edges, labels=bin_labels, include_lowest=True)
        profile = df.groupby("motif_bin", observed=False)["gsi"].mean()
        bin_profiles[ds] = profile

    # Now correlate profiles between dataset pairs
    for ds1, ds2 in combinations(DATASETS, 2):
        p1 = bin_profiles[ds1].values
        p2 = bin_profiles[ds2].values

        # Remove NaN bins
        mask = ~(np.isnan(p1) | np.isnan(p2))
        if mask.sum() < 3:
            print(f"    {ds1:12s} vs {ds2:12s}: insufficient overlapping bins")
            continue

        p1_clean, p2_clean = p1[mask], p2[mask]
        spearman_r, spearman_p = stats.spearmanr(p1_clean, p2_clean)
        pearson_r, pearson_p = stats.pearsonr(p1_clean, p2_clean)

        pair_type = "within_species" if SPECIES_MAP[ds1] == SPECIES_MAP[ds2] else "cross_species"

        model_results[f"{ds1}_vs_{ds2}"] = {
            "pair_type": pair_type,
            "n_bins_used": int(mask.sum()),
            "spearman_r": round(float(spearman_r), 4),
            "spearman_p": float(f"{spearman_p:.4f}"),
            "pearson_r": round(float(pearson_r), 4),
            "pearson_p": float(f"{pearson_p:.4f}"),
        }

        flag = " <-- SAME SPECIES" if pair_type == "within_species" else ""
        print(f"    {ds1:12s} vs {ds2:12s} [{pair_type:14s}] "
              f"Spearman r={spearman_r:+.3f} (p={spearman_p:.3f}), "
              f"Pearson r={pearson_r:+.3f} (p={pearson_p:.3f}){flag}")

    motif_conditioned_results[model] = model_results

results["motif_conditioned_gsi_correlation"] = motif_conditioned_results

# ──────────────────────────────────────────────────────────────────────
# Step 3b: GSI ~ motif_density regression slope comparison
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3b: GSI ~ Motif Density Regression Slope Comparison")
print("=" * 70)
print("  If GSI captures real biology, the relationship between GSI and motif")
print("  density should be consistent within species but differ across species.")

regression_results = {}
for model in MODELS:
    print(f"\n  Model: {model}")
    model_reg = {}
    for ds in DATASETS:
        df = data[(ds, model)]
        # Use log-transformed GSI to handle skewness
        log_gsi = np.log1p(df["gsi"].values)
        md = df["motif_density"].values
        nm = df["n_motifs"].values

        slope_md, intercept_md, r_md, p_md, se_md = stats.linregress(md, log_gsi)
        slope_nm, intercept_nm, r_nm, p_nm, se_nm = stats.linregress(nm, log_gsi)

        model_reg[ds] = {
            "species": SPECIES_MAP[ds],
            "log_gsi_vs_motif_density": {
                "slope": round(float(slope_md), 4),
                "r_squared": round(float(r_md**2), 4),
                "p_value": float(f"{p_md:.2e}"),
                "stderr": round(float(se_md), 4),
            },
            "log_gsi_vs_n_motifs": {
                "slope": round(float(slope_nm), 4),
                "r_squared": round(float(r_nm**2), 4),
                "p_value": float(f"{p_nm:.2e}"),
                "stderr": round(float(se_nm), 4),
            },
        }
        print(f"    {ds:12s} [{SPECIES_MAP[ds]:6s}] "
              f"slope(density)={slope_md:+.3f} (R²={r_md**2:.3f}), "
              f"slope(n_motifs)={slope_nm:+.4f} (R²={r_nm**2:.3f})")

    regression_results[model] = model_reg

results["regression_slope_comparison"] = regression_results

# ──────────────────────────────────────────────────────────────────────
# Step 3c: Motif-density-binned GSI profile correlation (finer bins)
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3c: Motif-Density-Binned GSI Profile Correlation")
print("=" * 70)

density_profile_results = {}
for model in MODELS:
    print(f"\n  Model: {model}")
    model_results = {}

    # Create 10 density bins across all datasets
    all_density = pd.concat([data[(ds, model)]["motif_density"] for ds in DATASETS])
    density_bin_edges = np.percentile(all_density, np.linspace(0, 100, 11))
    density_bin_edges = np.unique(density_bin_edges)
    d_labels = [f"d{i}" for i in range(len(density_bin_edges) - 1)]

    density_profiles = {}
    for ds in DATASETS:
        df = data[(ds, model)].copy()
        df["density_bin"] = pd.cut(df["motif_density"], bins=density_bin_edges,
                                    labels=d_labels, include_lowest=True)
        profile = df.groupby("density_bin", observed=False)["gsi"].median()
        density_profiles[ds] = profile

    for ds1, ds2 in combinations(DATASETS, 2):
        p1 = density_profiles[ds1].values.astype(float)
        p2 = density_profiles[ds2].values.astype(float)
        mask = ~(np.isnan(p1) | np.isnan(p2))
        if mask.sum() < 3:
            continue
        p1c, p2c = p1[mask], p2[mask]
        sr, sp = stats.spearmanr(p1c, p2c)
        pair_type = "within_species" if SPECIES_MAP[ds1] == SPECIES_MAP[ds2] else "cross_species"

        model_results[f"{ds1}_vs_{ds2}"] = {
            "pair_type": pair_type,
            "spearman_r": round(float(sr), 4),
            "spearman_p": round(float(sp), 4),
            "n_bins": int(mask.sum()),
        }

        flag = " <-- SAME SPECIES" if pair_type == "within_species" else ""
        print(f"    {ds1:12s} vs {ds2:12s} [{pair_type:14s}] "
              f"Spearman r={sr:+.3f} (p={sp:.3f}){flag}")

    density_profile_results[model] = model_results

results["density_binned_gsi_profiles"] = density_profile_results

# ──────────────────────────────────────────────────────────────────────
# Step 4: Cross-Species Control - GSI Distribution by Species Group
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Cross-Species Control - GSI Distribution by Species")
print("=" * 70)

species_control = {}
for model in MODELS:
    print(f"\n  Model: {model}")
    species_gsi = {"human": [], "yeast": [], "plant": []}

    for ds in DATASETS:
        gsi_vals = data[(ds, model)]["gsi"].values
        species_gsi[SPECIES_MAP[ds]].extend(gsi_vals.tolist())

    # Convert to arrays
    for sp in species_gsi:
        species_gsi[sp] = np.array(species_gsi[sp])

    model_ctrl = {}
    for sp in ["human", "yeast", "plant"]:
        vals = species_gsi[sp]
        model_ctrl[f"{sp}_summary"] = {
            "n": len(vals),
            "mean": round(float(np.mean(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "iqr": round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 4),
            "pct_significant": round(float(np.mean(vals > 1.0) * 100), 2),
        }
        print(f"    {sp:6s}: n={len(vals):5d}, mean={np.mean(vals):.3f}, "
              f"median={np.median(vals):.3f}, std={np.std(vals):.3f}, "
              f"pct>1.0={np.mean(vals > 1.0) * 100:.1f}%")

    # Kruskal-Wallis test across all 3 species
    kw_stat, kw_p = stats.kruskal(species_gsi["human"], species_gsi["yeast"], species_gsi["plant"])
    model_ctrl["kruskal_wallis"] = {
        "statistic": round(float(kw_stat), 4),
        "p_value": float(f"{kw_p:.2e}"),
        "interpretation": "significant" if kw_p < 0.05 else "not significant"
    }
    print(f"    Kruskal-Wallis H={kw_stat:.2f}, p={kw_p:.2e}")

    # Pairwise comparisons
    for sp1, sp2 in combinations(["human", "yeast", "plant"], 2):
        mw_stat, mw_p = stats.mannwhitneyu(species_gsi[sp1], species_gsi[sp2], alternative="two-sided")
        model_ctrl[f"{sp1}_vs_{sp2}"] = {
            "mannwhitney_U": float(mw_stat),
            "p_value": float(f"{mw_p:.2e}"),
        }
        print(f"    {sp1} vs {sp2}: Mann-Whitney U={mw_stat:.0f}, p={mw_p:.2e}")

    species_control[model] = model_ctrl

results["cross_species_control"] = species_control

# ──────────────────────────────────────────────────────────────────────
# Step 5: Cohen's d Effect Sizes
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Cohen's d Effect Sizes (Within-Species vs Cross-Species)")
print("=" * 70)

def cohens_d(x, y):
    """Compute Cohen's d between two groups."""
    nx, ny = len(x), len(y)
    # Pooled standard deviation
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std

effect_size_results = {}
for model in MODELS:
    print(f"\n  Model: {model}")
    within_ds = []
    cross_ds = []
    model_effects = {}

    for ds1, ds2 in combinations(DATASETS, 2):
        gsi1 = data[(ds1, model)]["gsi"].values
        gsi2 = data[(ds2, model)]["gsi"].values

        # Use log-transformed GSI for more meaningful effect sizes
        log_gsi1 = np.log1p(gsi1)
        log_gsi2 = np.log1p(gsi2)

        d = cohens_d(log_gsi1, log_gsi2)
        pair_type = "within_species" if SPECIES_MAP[ds1] == SPECIES_MAP[ds2] else "cross_species"

        model_effects[f"{ds1}_vs_{ds2}"] = {
            "pair_type": pair_type,
            "species_pair": f"{SPECIES_MAP[ds1]}_vs_{SPECIES_MAP[ds2]}",
            "cohens_d": round(float(d), 4),
            "abs_cohens_d": round(float(abs(d)), 4),
            "interpretation": (
                "negligible" if abs(d) < 0.2 else
                "small" if abs(d) < 0.5 else
                "medium" if abs(d) < 0.8 else
                "large"
            ),
        }

        if pair_type == "within_species":
            within_ds.append(abs(d))
        else:
            cross_ds.append(abs(d))

        flag = " <-- SAME SPECIES" if pair_type == "within_species" else ""
        print(f"    {ds1:12s} vs {ds2:12s} [{pair_type:14s}] "
              f"|Cohen's d| = {abs(d):.4f} ({model_effects[f'{ds1}_vs_{ds2}']['interpretation']}){flag}")

    # Summary: are within-species effect sizes smaller than cross-species?
    mean_within = np.mean(within_ds) if within_ds else float("nan")
    mean_cross = np.mean(cross_ds) if cross_ds else float("nan")

    model_effects["summary"] = {
        "mean_abs_d_within_species": round(float(mean_within), 4),
        "mean_abs_d_cross_species": round(float(mean_cross), 4),
        "ratio_cross_to_within": round(float(mean_cross / mean_within), 4) if mean_within > 0 else None,
        "within_species_more_similar": bool(mean_within < mean_cross),
    }

    print(f"\n    SUMMARY: Mean |d| within-species = {mean_within:.4f}, "
          f"cross-species = {mean_cross:.4f}")
    print(f"    Ratio (cross/within) = {mean_cross / mean_within:.2f}x" if mean_within > 0 else "")
    print(f"    Within-species more similar? {mean_within < mean_cross}")

    effect_size_results[model] = model_effects

results["effect_size_analysis"] = effect_size_results

# ──────────────────────────────────────────────────────────────────────
# Step 6: Model Consistency - Do models agree on which datasets are similar?
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Cross-Model Consistency Check")
print("=" * 70)

consistency_results = {}
# For each pair of datasets, gather Cohen's d across models
for ds1, ds2 in combinations(DATASETS, 2):
    pair_key = f"{ds1}_vs_{ds2}"
    ds_across_models = []
    for model in MODELS:
        d_val = effect_size_results[model][pair_key]["cohens_d"]
        ds_across_models.append(d_val)

    consistency_results[pair_key] = {
        "pair_type": "within_species" if SPECIES_MAP[ds1] == SPECIES_MAP[ds2] else "cross_species",
        "cohens_d_by_model": {m: round(float(d), 4) for m, d in zip(MODELS, ds_across_models)},
        "mean_cohens_d": round(float(np.mean(ds_across_models)), 4),
        "std_cohens_d": round(float(np.std(ds_across_models)), 4),
        "models_agree_on_sign": bool(all(d > 0 for d in ds_across_models) or all(d < 0 for d in ds_across_models)),
    }

    pair_type = consistency_results[pair_key]["pair_type"]
    ds_strs = [f"{m}={d:.3f}" for m, d in zip(MODELS, ds_across_models)]
    print(f"  {ds1:12s} vs {ds2:12s} [{pair_type:14s}]: {', '.join(ds_strs)}")

results["cross_model_consistency"] = consistency_results

# ──────────────────────────────────────────────────────────────────────
# Step 7: GSI Percentile Profile by Motif Count
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7: GSI Percentile Rank Profile by n_motifs (Rank-Based)")
print("=" * 70)
print("  Rank GSI within each dataset (percentile), then compare how")
print("  percentile rank relates to n_motifs across datasets.")

rank_profile_results = {}
for model in MODELS:
    print(f"\n  Model: {model}")
    model_rp = {}

    # For each dataset, compute percentile rank of GSI, then
    # compute Spearman correlation between GSI percentile rank and n_motifs
    rank_slopes = {}
    for ds in DATASETS:
        df = data[(ds, model)].copy()
        df["gsi_pctrank"] = df["gsi"].rank(pct=True)
        sr, sp = stats.spearmanr(df["n_motifs"], df["gsi_pctrank"])
        rank_slopes[ds] = {"spearman_r": sr, "spearman_p": sp}
        print(f"    {ds:12s} [{SPECIES_MAP[ds]:6s}]: "
              f"Spearman(n_motifs, GSI_rank) = {sr:+.3f} (p={sp:.3f})")

    # Compare rank_slope patterns between dataset pairs
    for ds1, ds2 in combinations(DATASETS, 2):
        pair_type = "within_species" if SPECIES_MAP[ds1] == SPECIES_MAP[ds2] else "cross_species"
        r1 = rank_slopes[ds1]["spearman_r"]
        r2 = rank_slopes[ds2]["spearman_r"]
        # Similarity: how close are the correlations?
        diff = abs(r1 - r2)
        same_direction = (r1 > 0 and r2 > 0) or (r1 < 0 and r2 < 0) or (r1 == 0 or r2 == 0)

        model_rp[f"{ds1}_vs_{ds2}"] = {
            "pair_type": pair_type,
            "r1": round(float(r1), 4),
            "r2": round(float(r2), 4),
            "abs_difference": round(float(diff), 4),
            "same_direction": bool(same_direction),
        }

    rank_profile_results[model] = model_rp

results["rank_based_profiles"] = rank_profile_results

# ──────────────────────────────────────────────────────────────────────
# Final Summary
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

# Aggregate evidence across models
within_ds_all = []
cross_ds_all = []
for model in MODELS:
    for key, val in effect_size_results[model].items():
        if key == "summary":
            continue
        if val["pair_type"] == "within_species":
            within_ds_all.append(val["abs_cohens_d"])
        else:
            cross_ds_all.append(val["abs_cohens_d"])

mean_w = np.mean(within_ds_all)
mean_c = np.mean(cross_ds_all)

# Permutation test: is the within-species mean |d| significantly smaller
# than cross-species mean |d|?
all_ds = within_ds_all + cross_ds_all
n_within = len(within_ds_all)
observed_diff = mean_c - mean_w

n_perm = 10000
np.random.seed(42)
perm_diffs = []
for _ in range(n_perm):
    perm = np.random.permutation(all_ds)
    perm_within = np.mean(perm[:n_within])
    perm_cross = np.mean(perm[n_within:])
    perm_diffs.append(perm_cross - perm_within)

perm_p = np.mean(np.array(perm_diffs) >= observed_diff)

summary = {
    "overall_mean_abs_cohens_d_within_species": round(float(mean_w), 4),
    "overall_mean_abs_cohens_d_cross_species": round(float(mean_c), 4),
    "ratio_cross_to_within": round(float(mean_c / mean_w), 4) if mean_w > 0 else None,
    "permutation_test_p_value": round(float(perm_p), 4),
    "n_permutations": n_perm,
    "within_species_pairs": n_within,
    "cross_species_pairs": len(cross_ds_all),
    "conclusion": "",
}

if perm_p < 0.05 and mean_c > mean_w:
    summary["conclusion"] = (
        f"SUPPORTED: Within-species GSI distributions are significantly more similar "
        f"(mean |d|={mean_w:.3f}) than cross-species (mean |d|={mean_c:.3f}), "
        f"permutation p={perm_p:.4f}. This supports the hypothesis that GSI "
        f"captures species-specific grammar rules."
    )
elif mean_c > mean_w:
    summary["conclusion"] = (
        f"TREND: Within-species GSI distributions tend to be more similar "
        f"(mean |d|={mean_w:.3f}) than cross-species (mean |d|={mean_c:.3f}), "
        f"but the permutation test is not significant (p={perm_p:.4f})."
    )
else:
    summary["conclusion"] = (
        f"NOT SUPPORTED: Within-species pairs are not more similar than cross-species "
        f"(within |d|={mean_w:.3f}, cross |d|={mean_c:.3f}, p={perm_p:.4f})."
    )

results["summary"] = summary

print(f"\n  Mean |Cohen's d| within-species:  {mean_w:.4f} (n={n_within} pairs)")
print(f"  Mean |Cohen's d| cross-species:   {mean_c:.4f} (n={len(cross_ds_all)} pairs)")
print(f"  Ratio (cross/within):             {mean_c / mean_w:.2f}x" if mean_w > 0 else "")
print(f"  Permutation test p-value:         {perm_p:.4f}")
print(f"\n  CONCLUSION: {summary['conclusion']}")

# ──────────────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved to: {OUTPUT_PATH}")
print("=" * 70)
