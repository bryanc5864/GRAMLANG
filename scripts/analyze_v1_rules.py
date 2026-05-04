#!/usr/bin/env python3
"""
Analyze the v1 grammar rules database to understand existing rules
before v2 re-extraction.

Outputs:
  - Detailed printed report
  - Summary JSON at ./results/v2/v1_rules_analysis.json
"""

import json
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

# -- 0. Load --
PARQUET = Path("./results/module2/grammar_rules_database.parquet")
OUT_JSON = Path("./results/v2/v1_rules_analysis.json")

df = pd.read_parquet(PARQUET)
summary = {}

# -- 1. Shape and columns --
print("=" * 80)
print("1. DATABASE SHAPE AND COLUMNS")
print("=" * 80)
print(f"  Rows:    {df.shape[0]}")
print(f"  Columns: {df.shape[1]}")
print()
for col in df.columns:
    non_null = df[col].notna().sum()
    print(f"  {col:30s}  dtype={str(df[col].dtype):10s}  non-null={non_null}/{len(df)}")
print()

summary["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
summary["column_names"] = list(df.columns)

# -- 2. Unique counts --
print("=" * 80)
print("2. UNIQUE COUNTS")
print("=" * 80)
n_rules = len(df)
n_pairs = df["pair"].nunique()
n_motif_a = df["motif_a"].nunique()
n_motif_b = df["motif_b"].nunique()
n_unique_motifs = len(set(df["motif_a"].unique()) | set(df["motif_b"].unique()))
n_datasets = df["dataset"].nunique()
n_models = df["model"].nunique()
n_seqs = df["seq_id"].nunique()

print(f"  Total rule rows:       {n_rules}")
print(f"  Unique motif pairs:    {n_pairs}")
print(f"  Unique motif_a:        {n_motif_a}")
print(f"  Unique motif_b:        {n_motif_b}")
print(f"  Unique motifs (union): {n_unique_motifs}")
print(f"  Datasets:              {n_datasets}  {sorted(df['dataset'].unique())}")
print(f"  Models:                {n_models}  {sorted(df['model'].unique())}")
print(f"  Unique sequences:      {n_seqs}")
print()

summary["unique_counts"] = {
    "total_rule_rows": n_rules,
    "unique_motif_pairs": n_pairs,
    "unique_motif_a": n_motif_a,
    "unique_motif_b": n_motif_b,
    "unique_motifs_union": n_unique_motifs,
    "datasets": sorted(df["dataset"].unique().tolist()),
    "models": sorted(df["model"].unique().tolist()),
    "unique_sequences": n_seqs,
}

# -- 3. Rules per (dataset, model) --
print("=" * 80)
print("3. RULES PER (DATASET, MODEL)")
print("=" * 80)
dm_counts = df.groupby(["dataset", "model"]).size().reset_index(name="n_rules")
dm_counts = dm_counts.sort_values("n_rules", ascending=False)
dm_pairs_unique = df.groupby(["dataset", "model"])["pair"].nunique().reset_index(name="n_unique_pairs")
dm_seqs = df.groupby(["dataset", "model"])["seq_id"].nunique().reset_index(name="n_unique_seqs")

dm_merged = dm_counts.merge(dm_pairs_unique).merge(dm_seqs)
print(f"  {'Dataset':<20s} {'Model':<12s} {'Rules':>8s} {'Pairs':>8s} {'Seqs':>8s}")
print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
dm_dict = {}
for _, row in dm_merged.iterrows():
    print(f"  {row['dataset']:<20s} {row['model']:<12s} {row['n_rules']:>8d} {row['n_unique_pairs']:>8d} {row['n_unique_seqs']:>8d}")
    dm_dict[f"{row['dataset']}__{row['model']}"] = {
        "n_rules": int(row["n_rules"]),
        "n_unique_pairs": int(row["n_unique_pairs"]),
        "n_unique_seqs": int(row["n_unique_seqs"]),
    }
print()
summary["rules_per_dataset_model"] = dm_dict

# -- 4. Rule type analysis --
print("=" * 80)
print("4. RULE TYPE ANALYSIS")
print("=" * 80)

# 4a. Spacing preferences
print("\n  4a. SPACING PREFERENCES")
print(f"      optimal_spacing range: [{df['optimal_spacing'].min()}, {df['optimal_spacing'].max()}]")
print(f"      mean optimal_spacing:  {df['optimal_spacing'].mean():.2f}")
print(f"      median optimal_spacing: {df['optimal_spacing'].median():.1f}")

spacing_median = df["spacing_sensitivity"].median()
strong_spacing = (df["spacing_sensitivity"] > spacing_median).sum()
print(f"      spacing_sensitivity median: {spacing_median:.4f}")
print(f"      Rules with above-median spacing sensitivity: {strong_spacing} ({100*strong_spacing/len(df):.1f}%)")

def is_helical(spacing, tol=1.5):
    remainder = spacing % 10.5
    return min(remainder, 10.5 - remainder) <= tol

df["helical_spacing"] = df["optimal_spacing"].apply(is_helical)
n_helical = df["helical_spacing"].sum()
print(f"      Rules with helical-periodic spacing (within 1.5bp of 10.5n): {n_helical} ({100*n_helical/len(df):.1f}%)")

spacing_bins = pd.cut(df["optimal_spacing"], bins=[0, 10, 20, 30, 40, 50, 100], right=True)
print(f"\n      Spacing distribution:")
for interval, count in spacing_bins.value_counts().sort_index().items():
    print(f"        {str(interval):>12s}: {count:>6d} ({100*count/len(df):5.1f}%)")

# 4b. Orientation preferences
print("\n  4b. ORIENTATION PREFERENCES")
orient_counts = df["optimal_orientation"].value_counts()
for orient, count in orient_counts.items():
    print(f"      {orient}: {count:>6d} ({100*count/len(df):5.1f}%)")

orient_median = df["orientation_sensitivity"].median()
strong_orient = (df["orientation_sensitivity"] > orient_median).sum()
print(f"      orientation_sensitivity median: {orient_median:.4f}")
print(f"      Rules with above-median orient sensitivity: {strong_orient} ({100*strong_orient/len(df):.1f}%)")

# 4c. Helical phase
print("\n  4c. HELICAL PHASE SCORES")
print(f"      helical_phase_score range: [{df['helical_phase_score'].min():.4f}, {df['helical_phase_score'].max():.4f}]")
print(f"      mean:   {df['helical_phase_score'].mean():.4f}")
print(f"      median: {df['helical_phase_score'].median():.4f}")
print(f"      std:    {df['helical_phase_score'].std():.4f}")
strong_helical = (df["helical_phase_score"] > 2.0).sum()
print(f"      Rules with helical_phase_score > 2.0 (strong periodicity): {strong_helical} ({100*strong_helical/len(df):.1f}%)")

# 4d. Classify rules by dominant grammar feature
print("\n  4d. RULE CLASSIFICATION BY DOMINANT FEATURE")

spacing_z = (df["spacing_sensitivity"] - df["spacing_sensitivity"].mean()) / df["spacing_sensitivity"].std()
orient_z = (df["orientation_sensitivity"] - df["orientation_sensitivity"].mean()) / df["orientation_sensitivity"].std()
helical_z = (df["helical_phase_score"] - df["helical_phase_score"].mean()) / df["helical_phase_score"].std()

def classify_rule(row_idx):
    categories = []
    if spacing_z.iloc[row_idx] > 0.5:
        categories.append("spacing")
    if orient_z.iloc[row_idx] > 0.5:
        categories.append("orientation")
    if helical_z.iloc[row_idx] > 0.5:
        categories.append("helical")
    if df["fold_change"].iloc[row_idx] > 2.0:
        categories.append("strong_effect")
    if not categories:
        categories.append("weak")
    return "+".join(sorted(categories))

df["rule_type"] = [classify_rule(i) for i in range(len(df))]
type_counts = df["rule_type"].value_counts()
print(f"      {'Rule Type':<45s} {'Count':>7s} {'Pct':>7s}")
print(f"      {'-'*45} {'-'*7} {'-'*7}")
rule_type_dict = {}
for rtype, count in type_counts.head(20).items():
    print(f"      {rtype:<45s} {count:>7d} {100*count/len(df):>6.1f}%")
    rule_type_dict[rtype] = int(count)
print()

summary["rule_types"] = {
    "spacing": {
        "optimal_spacing_range": [int(df["optimal_spacing"].min()), int(df["optimal_spacing"].max())],
        "mean_optimal_spacing": round(float(df["optimal_spacing"].mean()), 2),
        "median_optimal_spacing": round(float(df["optimal_spacing"].median()), 1),
        "spacing_sensitivity_median": round(float(spacing_median), 4),
        "helical_periodic_count": int(n_helical),
        "helical_periodic_pct": round(100 * n_helical / len(df), 1),
    },
    "orientation": {
        "distribution": {k: int(v) for k, v in orient_counts.items()},
        "orientation_sensitivity_median": round(float(orient_median), 4),
    },
    "helical_phase": {
        "mean": round(float(df["helical_phase_score"].mean()), 4),
        "median": round(float(df["helical_phase_score"].median()), 4),
        "strong_periodicity_count": int(strong_helical),
    },
    "rule_classification": rule_type_dict,
}

# -- 5. Top 20 most common motif pairs --
print("=" * 80)
print("5. TOP 20 MOST COMMON MOTIF PAIRS")
print("=" * 80)
pair_counts = df["pair"].value_counts().head(20)
print(f"  {'Rank':>4s}  {'Motif Pair':<45s} {'Count':>7s} {'Pct':>7s}")
print(f"  {'-'*4}  {'-'*45} {'-'*7} {'-'*7}")
top_pairs_dict = {}
for rank, (pair, count) in enumerate(pair_counts.items(), 1):
    print(f"  {rank:>4d}  {pair:<45s} {count:>7d} {100*count/len(df):>6.1f}%")
    top_pairs_dict[pair] = int(count)
print()

summary["top_20_motif_pairs"] = top_pairs_dict

# -- 6. Effect size statistics --
print("=" * 80)
print("6. EFFECT SIZE STATISTICS")
print("=" * 80)

print("\n  6a. FOLD CHANGE")
fc = df["fold_change"]
print(f"      count:    {fc.notna().sum()}")
print(f"      min:      {fc.min():.4f}")
print(f"      25%:      {fc.quantile(0.25):.4f}")
print(f"      median:   {fc.median():.4f}")
print(f"      75%:      {fc.quantile(0.75):.4f}")
print(f"      max:      {fc.max():.4f}")
print(f"      mean:     {fc.mean():.4f}")
print(f"      std:      {fc.std():.4f}")

for thresh in [1.5, 2.0, 3.0, 5.0]:
    n_above = (fc >= thresh).sum()
    print(f"      fold_change >= {thresh}: {n_above} ({100*n_above/len(df):.1f}%)")

print("\n  6b. SPACING SENSITIVITY")
ss = df["spacing_sensitivity"]
print(f"      count:    {ss.notna().sum()}")
print(f"      min:      {ss.min():.4f}")
print(f"      25%:      {ss.quantile(0.25):.4f}")
print(f"      median:   {ss.median():.4f}")
print(f"      75%:      {ss.quantile(0.75):.4f}")
print(f"      max:      {ss.max():.4f}")
print(f"      mean:     {ss.mean():.4f}")
print(f"      std:      {ss.std():.4f}")

print("\n  6c. ORIENTATION SENSITIVITY")
os_ = df["orientation_sensitivity"]
print(f"      count:    {os_.notna().sum()}")
print(f"      min:      {os_.min():.4f}")
print(f"      25%:      {os_.quantile(0.25):.4f}")
print(f"      median:   {os_.median():.4f}")
print(f"      75%:      {os_.quantile(0.75):.4f}")
print(f"      max:      {os_.max():.4f}")
print(f"      mean:     {os_.mean():.4f}")
print(f"      std:      {os_.std():.4f}")

print("\n  6d. FOLD CHANGE DISTRIBUTION BY DATASET")
fc_by_ds = df.groupby("dataset")["fold_change"].describe()
print(fc_by_ds.to_string())
print()

summary["effect_sizes"] = {
    "fold_change": {
        "min": round(float(fc.min()), 4),
        "q25": round(float(fc.quantile(0.25)), 4),
        "median": round(float(fc.median()), 4),
        "q75": round(float(fc.quantile(0.75)), 4),
        "max": round(float(fc.max()), 4),
        "mean": round(float(fc.mean()), 4),
        "std": round(float(fc.std()), 4),
        "above_1.5": int((fc >= 1.5).sum()),
        "above_2.0": int((fc >= 2.0).sum()),
        "above_3.0": int((fc >= 3.0).sum()),
        "above_5.0": int((fc >= 5.0).sum()),
    },
    "spacing_sensitivity": {
        "min": round(float(ss.min()), 4),
        "median": round(float(ss.median()), 4),
        "max": round(float(ss.max()), 4),
        "mean": round(float(ss.mean()), 4),
        "std": round(float(ss.std()), 4),
    },
    "orientation_sensitivity": {
        "min": round(float(os_.min()), 4),
        "median": round(float(os_.median()), 4),
        "max": round(float(os_.max()), 4),
        "mean": round(float(os_.mean()), 4),
        "std": round(float(os_.std()), 4),
    },
}

# -- 7. Consensus rules --
print("=" * 80)
print("7. CONSENSUS RULES (AGREED BY MULTIPLE MODELS WITHIN A DATASET)")
print("=" * 80)

consensus_groups = df.groupby(["dataset", "pair"]).agg(
    n_models=("model", "nunique"),
    models=("model", lambda x: sorted(x.unique().tolist())),
    spacings=("optimal_spacing", list),
    orientations=("optimal_orientation", list),
    mean_fc=("fold_change", "mean"),
    max_fc=("fold_change", "max"),
    spacing_std=("optimal_spacing", "std"),
).reset_index()

multi_model = consensus_groups[consensus_groups["n_models"] >= 2].copy()
print(f"\n  Pairs appearing in 2+ models within same dataset: {len(multi_model)}")

if len(multi_model) > 0:
    multi_model["orient_agree"] = multi_model["orientations"].apply(
        lambda x: len(set(x)) == 1
    )
    multi_model["spacing_agree"] = multi_model["spacing_std"].apply(
        lambda x: x <= 5.0 if pd.notna(x) else False
    )
    multi_model["full_consensus"] = multi_model["orient_agree"] & multi_model["spacing_agree"]

    n_orient_agree = multi_model["orient_agree"].sum()
    n_spacing_agree = multi_model["spacing_agree"].sum()
    n_full = multi_model["full_consensus"].sum()

    print(f"  Orientation agreement:   {n_orient_agree} ({100*n_orient_agree/len(multi_model):.1f}%)")
    print(f"  Spacing agreement (std<=5): {n_spacing_agree} ({100*n_spacing_agree/len(multi_model):.1f}%)")
    print(f"  Full consensus (both):   {n_full} ({100*n_full/len(multi_model):.1f}%)")

    three_model = multi_model[multi_model["n_models"] == 3]
    print(f"\n  Pairs in ALL 3 models: {len(three_model)}")
    three_full = 0
    if len(three_model) > 0:
        three_full = three_model["full_consensus"].sum()
        print(f"  3-model full consensus: {three_full}")

    top_consensus = multi_model.sort_values("mean_fc", ascending=False).head(15)
    print(f"\n  Top 15 multi-model rules by mean fold change:")
    print(f"  {'Dataset':<16s} {'Pair':<40s} {'#Mod':>4s} {'Orient':>7s} {'SpStd':>6s} {'MeanFC':>7s} {'Consensus':>9s}")
    print(f"  {'-'*16} {'-'*40} {'-'*4} {'-'*7} {'-'*6} {'-'*7} {'-'*9}")
    for _, row in top_consensus.iterrows():
        orient_str = "yes" if row["orient_agree"] else "no"
        sp_std_str = f"{row['spacing_std']:.1f}" if pd.notna(row["spacing_std"]) else "N/A"
        cons_str = "FULL" if row["full_consensus"] else "partial"
        print(f"  {row['dataset']:<16s} {row['pair']:<40s} {row['n_models']:>4d} {orient_str:>7s} {sp_std_str:>6s} {row['mean_fc']:>7.3f} {cons_str:>9s}")

    summary["consensus"] = {
        "pairs_in_2plus_models": int(len(multi_model)),
        "orientation_agreement": int(n_orient_agree),
        "spacing_agreement": int(n_spacing_agree),
        "full_consensus": int(n_full),
        "pairs_in_all_3_models": int(len(three_model)),
        "three_model_full_consensus": int(three_full),
    }
else:
    print("  No multi-model rules found.")
    summary["consensus"] = {"pairs_in_2plus_models": 0}

print()

# -- 8. Cross-species analysis --
print("=" * 80)
print("8. CROSS-SPECIES ANALYSIS")
print("=" * 80)

species_map = {
    "agarwal": "human",
    "inoue": "human",
    "klein": "human",
    "vaishnav": "yeast",
    "jores": "plant",
}

df["species"] = df["dataset"].map(species_map)
unmapped = df[df["species"].isna()]["dataset"].unique()
if len(unmapped) > 0:
    print(f"  WARNING: Unmapped datasets: {unmapped}")
    for ds in unmapped:
        species_map[ds] = "unknown"
    df["species"] = df["dataset"].map(species_map)

print(f"\n  Species classification:")
for species in sorted(df["species"].unique()):
    datasets = sorted(df[df["species"] == species]["dataset"].unique())
    n_rules_sp = (df["species"] == species).sum()
    n_pairs_sp = df[df["species"] == species]["pair"].nunique()
    print(f"    {species:<10s}: datasets={datasets}, rules={n_rules_sp}, unique_pairs={n_pairs_sp}")

print(f"\n  Motif pairs by species:")
species_pairs = {}
for species in sorted(df["species"].unique()):
    pairs_set = set(df[df["species"] == species]["pair"].unique())
    species_pairs[species] = pairs_set
    print(f"    {species:<10s}: {len(pairs_set)} unique pairs")

species_list = sorted(species_pairs.keys())
print(f"\n  Pairwise pair overlaps:")
cross_species_shared = {}
for i, sp1 in enumerate(species_list):
    for sp2 in species_list[i+1:]:
        shared = species_pairs[sp1] & species_pairs[sp2]
        print(f"    {sp1} & {sp2}: {len(shared)} shared pairs")
        if len(shared) > 0:
            print(f"      Examples: {sorted(list(shared))[:5]}")
            cross_species_shared[f"{sp1}__{sp2}"] = sorted(list(shared))

print(f"\n  Individual motifs by species:")
species_motifs = {}
for species in sorted(df["species"].unique()):
    sub = df[df["species"] == species]
    motifs = set(sub["motif_a"].unique()) | set(sub["motif_b"].unique())
    species_motifs[species] = motifs
    print(f"    {species:<10s}: {len(motifs)} unique motifs")

print(f"\n  Pairwise motif overlaps:")
shared_motifs_count = {}
for i, sp1 in enumerate(species_list):
    for sp2 in species_list[i+1:]:
        shared = species_motifs[sp1] & species_motifs[sp2]
        shared_motifs_count[f"{sp1}__{sp2}"] = len(shared)
        print(f"    {sp1} & {sp2}: {len(shared)} shared motifs")
        if len(shared) > 0:
            print(f"      Examples: {sorted(list(shared))[:10]}")

summary["cross_species"] = {
    "species_map": species_map,
    "rules_per_species": {sp: int((df["species"] == sp).sum()) for sp in species_list},
    "pairs_per_species": {sp: len(species_pairs[sp]) for sp in species_list},
    "motifs_per_species": {sp: len(species_motifs[sp]) for sp in species_list},
    "shared_pairs": {k: v for k, v in cross_species_shared.items()},
    "shared_motifs_count": shared_motifs_count,
}

# -- 9. Summary observations for v2 --
print()
print("=" * 80)
print("9. KEY OBSERVATIONS FOR v2 MODULE 2 IMPROVEMENTS")
print("=" * 80)

obs = []

obs.append(f"v1 database has {n_rules} rule rows across {n_datasets} datasets and {n_models} models.")

weak_rules = (fc < 1.5).sum()
obs.append(f"{weak_rules} rules ({100*weak_rules/len(df):.1f}%) have fold_change < 1.5, suggesting many weak/noisy rules. v2 should apply stricter filtering.")

if len(multi_model) > 0:
    obs.append(f"Only {n_full} of {len(multi_model)} multi-model pairs ({100*n_full/len(multi_model):.1f}%) have full consensus. v2 should prioritize cross-model agreement.")

obs.append("v1 has no p_value column - statistical significance was not tracked. v2 should add significance testing.")

total_cross = sum(len(v) for v in cross_species_shared.values())
if total_cross == 0:
    obs.append("No motif pairs are shared across species, likely because different motif databases were used per species. v2 could explore orthologous TF comparisons.")
else:
    obs.append(f"{total_cross} motif pairs are shared across species, enabling cross-species grammar comparison in v2.")

most_common_orient = orient_counts.index[0]
most_common_orient_pct = 100 * orient_counts.iloc[0] / len(df)
obs.append(f"Most common orientation is {most_common_orient} ({most_common_orient_pct:.1f}%). v2 should check if this reflects biology or extraction bias.")

obs.append("v1 stores raw spacing_profile arrays but no summary statistics (periodicity p-value, peak sharpness). v2 should compute and store derived features.")

for i, o in enumerate(obs, 1):
    print(f"  {i}. {o}")
print()

summary["v2_observations"] = obs

# -- Save JSON --
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"Summary JSON saved to: {OUT_JSON}")
print("Done.")
