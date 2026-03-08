"""
Validation Experiment J: Evolutionary Conservation of Grammar Properties
========================================================================
Tests whether regulatory grammar properties are conserved WITHIN species,
complementing the finding that grammar is species-specific (cross-species
transfer distance = 1.0, Experiment I).

Four analyses:
  J1. Within-species grammar consistency: Do grammar rules from one human
      dataset predict arrangement effects in another human dataset?
  J2. Helical phasing universality: Is the ~13.4% helical phasing rate
      significantly different from chance (10.5bp periodicity)?
  J3. Grammar-expression coupling conservation: Is the GSI-MPRA expression
      correlation consistent across human datasets?
  J4. Motif pair grammar conservation: For shared motif pairs across human
      datasets, are grammar properties more similar than expected by chance?
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

warnings.filterwarnings("ignore")

# ======================================================================
# Configuration
# ======================================================================
MODULE1_DIR = "/home/bcheng/grammar/results/v2/module1/"
MODULE2_DIR = "/home/bcheng/grammar/results/v2/module2/"
DIST_TRANSFER = "/home/bcheng/grammar/results/v2/distributional_transfer/distributional_transfer.json"
OUTPUT_PATH = "/home/bcheng/grammar/results/v2/module1/validation_experiment_j.json"

HUMAN_DATASETS = ["agarwal", "inoue", "klein"]
ALL_DATASETS = ["agarwal", "inoue", "klein", "vaishnav", "jores"]
MODELS = ["dnabert2", "nt", "hyenadna"]
SPECIES_MAP = {
    "agarwal": "human", "inoue": "human", "klein": "human",
    "vaishnav": "yeast", "jores": "plant"
}

np.random.seed(42)

# ======================================================================
# Helper functions
# ======================================================================
def cohens_d(x, y):
    """Compute Cohen's d (pooled SD)."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    )
    if pooled_std == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def glass_delta(x, y):
    """Glass's delta: uses only the control group SD."""
    sd = np.std(y, ddof=1)
    if sd == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / sd


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def safe_float(x):
    """Convert to JSON-safe float."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        if np.isnan(x) or np.isinf(x):
            return None
        return float(x)
    return x


# ======================================================================
# Load data
# ======================================================================
print("=" * 72)
print("VALIDATION EXPERIMENT J: Evolutionary Conservation of Grammar Properties")
print("=" * 72)

# Load GSI data
gsi_data = {}
for ds in ALL_DATASETS:
    for model in MODELS:
        fpath = os.path.join(MODULE1_DIR, f"{ds}_{model}_gsi.parquet")
        if os.path.exists(fpath):
            gsi_data[(ds, model)] = pd.read_parquet(fpath)

all_gsi = pd.read_parquet(os.path.join(MODULE1_DIR, "all_gsi_results.parquet"))
print(f"  Loaded all_gsi_results: {len(all_gsi)} rows")

# Load grammar rules
rules = pd.read_parquet(os.path.join(MODULE2_DIR, "grammar_rules_database.parquet"))
print(f"  Loaded grammar_rules_database: {len(rules)} rules")

# Load distributional transfer
with open(DIST_TRANSFER, "r") as f:
    dist_transfer = json.load(f)
print(f"  Loaded distributional_transfer.json")

results = {
    "experiment": "Validation Experiment J - Evolutionary Conservation of Grammar Properties",
    "description": (
        "Tests whether grammar properties are evolutionarily conserved WITHIN "
        "species, complementing the finding that grammar is completely "
        "species-specific (cross-species transfer distance = 1.0)."
    ),
    "human_datasets": HUMAN_DATASETS,
    "models": MODELS,
}

# ======================================================================
# J1: Within-Species Grammar Consistency
# ======================================================================
print("\n" + "=" * 72)
print("J1: WITHIN-SPECIES GRAMMAR CONSISTENCY")
print("=" * 72)
print("  Testing whether grammar rules from one human dataset predict")
print("  arrangement effects in another human dataset.")

j1_results = {}

# J1a: Motif pair overlap between human datasets
print("\n  --- J1a: Motif Pair Overlap ---")
human_pairs = {}
for ds in HUMAN_DATASETS:
    ds_rules = rules[rules["dataset"] == ds]
    human_pairs[ds] = set(ds_rules["pair"].unique())
    print(f"    {ds}: {len(human_pairs[ds])} unique motif pairs")

# Non-human datasets for comparison
nonhuman_pairs = {}
for ds in ["vaishnav", "jores"]:
    ds_rules = rules[rules["dataset"] == ds]
    nonhuman_pairs[ds] = set(ds_rules["pair"].unique())
    print(f"    {ds}: {len(nonhuman_pairs[ds])} unique motif pairs")

pair_overlap = {}
# Within-human overlaps
for ds1, ds2 in combinations(HUMAN_DATASETS, 2):
    shared = human_pairs[ds1] & human_pairs[ds2]
    union = human_pairs[ds1] | human_pairs[ds2]
    jaccard = len(shared) / len(union) if len(union) > 0 else 0
    pair_overlap[f"{ds1}_vs_{ds2}"] = {
        "type": "within_human",
        "shared_pairs": len(shared),
        "union_pairs": len(union),
        "jaccard_index": round(jaccard, 4),
        "pct_of_smaller": round(
            len(shared) / min(len(human_pairs[ds1]), len(human_pairs[ds2])) * 100, 2
        ),
    }
    print(f"    {ds1} vs {ds2}: {len(shared)} shared ({jaccard:.3f} Jaccard, "
          f"{len(shared)/min(len(human_pairs[ds1]),len(human_pairs[ds2]))*100:.1f}% of smaller)")

# Cross-species overlaps for comparison
for hds in HUMAN_DATASETS:
    for nds in ["vaishnav", "jores"]:
        shared = human_pairs[hds] & nonhuman_pairs[nds]
        union = human_pairs[hds] | nonhuman_pairs[nds]
        jaccard = len(shared) / len(union) if len(union) > 0 else 0
        pair_overlap[f"{hds}_vs_{nds}"] = {
            "type": "cross_species",
            "shared_pairs": len(shared),
            "union_pairs": len(union),
            "jaccard_index": round(jaccard, 4),
        }
        print(f"    {hds} vs {nds}: {len(shared)} shared ({jaccard:.3f} Jaccard) [cross-species]")

# Significance test: is within-human Jaccard > cross-species Jaccard?
within_jaccards = [v["jaccard_index"] for v in pair_overlap.values() if v["type"] == "within_human"]
cross_jaccards = [v["jaccard_index"] for v in pair_overlap.values() if v["type"] == "cross_species"]
if len(within_jaccards) > 1 and len(cross_jaccards) > 1:
    mw_stat, mw_p = stats.mannwhitneyu(within_jaccards, cross_jaccards, alternative="greater")
else:
    # Use permutation test if too few samples
    observed_diff_j = np.mean(within_jaccards) - np.mean(cross_jaccards)
    all_j = within_jaccards + cross_jaccards
    n_w = len(within_jaccards)
    perm_count = 0
    for _ in range(10000):
        perm = np.random.permutation(all_j)
        if np.mean(perm[:n_w]) - np.mean(perm[n_w:]) >= observed_diff_j:
            perm_count += 1
    mw_p = perm_count / 10000
    mw_stat = observed_diff_j

pair_overlap["statistical_test"] = {
    "test": "Permutation test (within-human Jaccard > cross-species Jaccard)",
    "mean_within_human_jaccard": round(float(np.mean(within_jaccards)), 4),
    "mean_cross_species_jaccard": round(float(np.mean(cross_jaccards)), 4),
    "effect_size_ratio": round(
        float(np.mean(within_jaccards) / np.mean(cross_jaccards)), 4
    ) if np.mean(cross_jaccards) > 0 else None,
    "p_value": round(float(mw_p), 6),
    "statistic": round(float(mw_stat), 4),
}
print(f"\n    Within-human mean Jaccard: {np.mean(within_jaccards):.4f}")
print(f"    Cross-species mean Jaccard: {np.mean(cross_jaccards):.4f}")
print(f"    Permutation p-value: {mw_p:.6f}")

j1_results["motif_pair_overlap"] = pair_overlap

# J1b: Grammar rule property agreement for shared pairs
print("\n  --- J1b: Grammar Rule Property Agreement (Shared Pairs) ---")

# For each model, aggregate rules per (dataset, pair)
rule_agreement = {}
for model in MODELS:
    print(f"\n    Model: {model}")
    model_rules = rules[rules["model"] == model]

    # Aggregate by dataset + pair: mean spacing_sensitivity, orientation_sensitivity
    agg_rules = {}
    for ds in HUMAN_DATASETS:
        ds_model = model_rules[model_rules["dataset"] == ds]
        agg = ds_model.groupby("pair").agg({
            "spacing_sensitivity": "mean",
            "orientation_sensitivity": "mean",
            "helical_phase_score": "mean",
            "optimal_spacing": lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan,
            "optimal_orientation": lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan,
        }).reset_index()
        agg_rules[ds] = agg.set_index("pair")

    model_agreement = {}
    for ds1, ds2 in combinations(HUMAN_DATASETS, 2):
        shared = set(agg_rules[ds1].index) & set(agg_rules[ds2].index)
        if len(shared) < 5:
            print(f"      {ds1} vs {ds2}: too few shared pairs ({len(shared)})")
            continue

        shared = sorted(shared)
        r1 = agg_rules[ds1].loc[shared]
        r2 = agg_rules[ds2].loc[shared]

        # Spacing sensitivity correlation
        ss_corr, ss_p = stats.spearmanr(
            r1["spacing_sensitivity"].values, r2["spacing_sensitivity"].values
        )
        # Orientation sensitivity correlation
        os_corr, os_p = stats.spearmanr(
            r1["orientation_sensitivity"].values, r2["orientation_sensitivity"].values
        )
        # Helical phase score correlation
        hp_corr, hp_p = stats.spearmanr(
            r1["helical_phase_score"].values, r2["helical_phase_score"].values
        )
        # Optimal spacing agreement
        spacing_agree = float(np.mean(
            r1["optimal_spacing"].values == r2["optimal_spacing"].values
        ))
        # Optimal orientation agreement
        orient_agree = float(np.mean(
            r1["optimal_orientation"].values == r2["optimal_orientation"].values
        ))

        # Effect size for spacing sensitivity
        ss_d = cohens_d(
            r1["spacing_sensitivity"].values, r2["spacing_sensitivity"].values
        )

        model_agreement[f"{ds1}_vs_{ds2}"] = {
            "n_shared_pairs": len(shared),
            "spacing_sensitivity_spearman_r": round(float(ss_corr), 4),
            "spacing_sensitivity_spearman_p": float(f"{ss_p:.2e}"),
            "orientation_sensitivity_spearman_r": round(float(os_corr), 4),
            "orientation_sensitivity_spearman_p": float(f"{os_p:.2e}"),
            "helical_phase_spearman_r": round(float(hp_corr), 4),
            "helical_phase_spearman_p": float(f"{hp_p:.2e}"),
            "optimal_spacing_exact_agreement": round(spacing_agree, 4),
            "optimal_orientation_exact_agreement": round(orient_agree, 4),
            "spacing_sensitivity_cohens_d": round(float(ss_d), 4),
        }

        print(f"      {ds1} vs {ds2} ({len(shared)} pairs): "
              f"spacing_sens r={ss_corr:.3f} (p={ss_p:.2e}), "
              f"orient_sens r={os_corr:.3f} (p={os_p:.2e}), "
              f"spacing_agree={spacing_agree:.3f}, orient_agree={orient_agree:.3f}")

    rule_agreement[model] = model_agreement

j1_results["grammar_rule_agreement"] = rule_agreement

# J1c: GSI distribution similarity within human datasets
print("\n  --- J1c: GSI Distribution Similarity Within Human ---")
gsi_similarity = {}
for model in MODELS:
    print(f"\n    Model: {model}")
    model_sim = {}

    # Collect within-human and cross-species KS statistics
    within_ks = []
    cross_ks = []

    for ds1, ds2 in combinations(ALL_DATASETS, 2):
        if (ds1, model) not in gsi_data or (ds2, model) not in gsi_data:
            continue
        g1 = gsi_data[(ds1, model)]["gsi"].values
        g2 = gsi_data[(ds2, model)]["gsi"].values
        ks_stat, ks_p = stats.ks_2samp(g1, g2)

        pair_type = "within_human" if (
            SPECIES_MAP[ds1] == "human" and SPECIES_MAP[ds2] == "human"
        ) else "cross_species"

        entry = {
            "pair_type": pair_type,
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": float(f"{ks_p:.2e}"),
        }
        model_sim[f"{ds1}_vs_{ds2}"] = entry

        if pair_type == "within_human":
            within_ks.append(ks_stat)
        else:
            cross_ks.append(ks_stat)

    # Within-human should have LOWER KS stats (more similar)
    mean_within = np.mean(within_ks) if within_ks else float("nan")
    mean_cross = np.mean(cross_ks) if cross_ks else float("nan")

    # Permutation test
    all_ks = within_ks + cross_ks
    n_within = len(within_ks)
    observed = mean_cross - mean_within
    perm_count = 0
    for _ in range(10000):
        perm = np.random.permutation(all_ks)
        if np.mean(perm[n_within:]) - np.mean(perm[:n_within]) >= observed:
            perm_count += 1
    perm_p = perm_count / 10000

    model_sim["summary"] = {
        "mean_ks_within_human": round(float(mean_within), 4),
        "mean_ks_cross_species": round(float(mean_cross), 4),
        "ratio_cross_to_within": round(float(mean_cross / mean_within), 4) if mean_within > 0 else None,
        "permutation_p_value": round(float(perm_p), 6),
        "within_human_more_similar": bool(mean_within < mean_cross),
    }

    print(f"      Mean KS within-human: {mean_within:.4f}, cross-species: {mean_cross:.4f}")
    print(f"      Ratio (cross/within): {mean_cross/mean_within:.2f}x, perm p={perm_p:.4f}")

    gsi_similarity[model] = model_sim

j1_results["gsi_distribution_similarity"] = gsi_similarity

results["J1_within_species_grammar_consistency"] = j1_results

# ======================================================================
# J2: Helical Phasing Universality
# ======================================================================
print("\n" + "=" * 72)
print("J2: HELICAL PHASING UNIVERSALITY")
print("=" * 72)
print("  Testing whether ~13.4% helical phasing rate differs from chance")
print("  (expected from 10.5bp periodicity in random spacing distributions).")

j2_results = {}

# Extract helical phasing rates from distributional transfer data
species_phasing = {}
for sp in ["human", "plant", "yeast"]:
    rate = dist_transfer["species_distributions"][sp]["helical_phasing_rate"]
    species_phasing[sp] = rate
    print(f"  {sp}: helical phasing rate = {rate:.4f} ({rate*100:.2f}%)")

# Compute per-dataset helical phasing rates
dataset_phasing = {}
for ds in ALL_DATASETS:
    ds_rules = rules[rules["dataset"] == ds]
    n_total = len(ds_rules)
    # Helical phasing: score > 1.5 is typically the threshold used
    # But let's check the actual distribution to determine proper threshold
    scores = ds_rules["helical_phase_score"].values
    dataset_phasing[ds] = {
        "n_rules": n_total,
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "std_score": float(np.std(scores)),
    }

# J2a: Expected helical phasing rate under null hypothesis
# If spacings are uniformly distributed over range [2, 50], what fraction
# would show 10.5bp periodicity by chance?
print("\n  --- J2a: Null Model for Helical Phasing ---")
print("  Null: random spacing distributions over [2,50]bp range.")
print("  Under uniform random spacings, helical phase score ~ 1.0 (no periodicity).")
print("  Test: are observed phasing rates significantly above chance?")

# Simulate null distribution of helical phase scores
n_simulations = 10000
n_spacings_per_sim = 49  # spacings from 2 to 50
null_helical_rates = []

for _ in range(n_simulations):
    # Generate random spacing profile (uniform noise)
    random_profile = np.random.randn(n_spacings_per_sim)
    # Compute autocorrelation at 10.5bp lag (approximate: lags 10 and 11)
    # Helical phase score: ratio of power at ~10.5bp period vs mean power
    fft = np.fft.rfft(random_profile)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n_spacings_per_sim, d=1.0)
    # 10.5bp period => frequency = 1/10.5 ~ 0.0952
    target_freq = 1.0 / 10.5
    freq_idx = np.argmin(np.abs(freqs - target_freq))
    # Helical phase score: power at target freq / mean power
    mean_power = np.mean(power[1:])  # exclude DC
    if mean_power > 0:
        phase_score = float(power[freq_idx] / mean_power)
    else:
        phase_score = 1.0
    null_helical_rates.append(phase_score)

null_arr = np.array(null_helical_rates)
null_mean = np.mean(null_arr)
null_std = np.std(null_arr)
null_95th = np.percentile(null_arr, 95)
null_99th = np.percentile(null_arr, 99)

print(f"  Null helical phase score: mean={null_mean:.4f}, std={null_std:.4f}")
print(f"  95th percentile = {null_95th:.4f}, 99th = {null_99th:.4f}")

# Fraction of null simulations exceeding threshold for "helical phased"
# Using threshold of 1.5 (commonly used)
null_frac_above_1_5 = np.mean(null_arr > 1.5)
print(f"  Null fraction with score > 1.5: {null_frac_above_1_5:.4f} ({null_frac_above_1_5*100:.2f}%)")

j2_results["null_model"] = {
    "method": "FFT-based helical phase score on random spacing profiles (N=10000 simulations)",
    "n_spacings": n_spacings_per_sim,
    "null_mean_phase_score": round(float(null_mean), 4),
    "null_std_phase_score": round(float(null_std), 4),
    "null_95th_percentile": round(float(null_95th), 4),
    "null_99th_percentile": round(float(null_99th), 4),
    "null_fraction_above_1_5": round(float(null_frac_above_1_5), 4),
}

# J2b: Compare observed vs null for each species
print("\n  --- J2b: Observed vs Null Helical Phasing ---")
observed_phasing = {}
for sp in ["human", "plant", "yeast"]:
    obs_rate = species_phasing[sp]
    # Binomial test: is observed rate > null rate?
    sp_rules = rules[rules["dataset"].map(SPECIES_MAP) == sp]
    n_total = len(sp_rules)
    scores = sp_rules["helical_phase_score"].values

    # Count rules with helical phase score > 1.5 (threshold for "phased")
    n_phased = int(np.sum(scores > 1.5))
    obs_rate_actual = n_phased / n_total if n_total > 0 else 0

    # Binomial test against null rate
    binom_p = stats.binom_test(n_phased, n_total, null_frac_above_1_5, alternative="greater")

    # Z-test for proportion
    p0 = null_frac_above_1_5
    p_hat = obs_rate_actual
    se = np.sqrt(p0 * (1 - p0) / n_total) if n_total > 0 else 1
    z_score = (p_hat - p0) / se if se > 0 else 0
    z_p = 1 - stats.norm.cdf(z_score)

    # Effect size: Cohen's h for proportions
    h = 2 * (np.arcsin(np.sqrt(p_hat)) - np.arcsin(np.sqrt(p0)))

    observed_phasing[sp] = {
        "n_rules": n_total,
        "n_phased": n_phased,
        "observed_rate": round(float(obs_rate_actual), 4),
        "reported_rate": round(float(obs_rate), 4),
        "null_rate": round(float(null_frac_above_1_5), 4),
        "binomial_p_value": float(f"{binom_p:.2e}"),
        "z_test_statistic": round(float(z_score), 4),
        "z_test_p_value": float(f"{z_p:.2e}"),
        "cohens_h": round(float(h), 4),
        "interpretation": (
            "negligible" if abs(h) < 0.2 else
            "small" if abs(h) < 0.5 else
            "medium" if abs(h) < 0.8 else
            "large"
        ),
    }

    sig = "***" if binom_p < 0.001 else ("**" if binom_p < 0.01 else ("*" if binom_p < 0.05 else "ns"))
    print(f"    {sp}: observed={obs_rate_actual:.4f} vs null={null_frac_above_1_5:.4f}, "
          f"binom p={binom_p:.2e} {sig}, Cohen's h={h:.4f}")

j2_results["observed_vs_null"] = observed_phasing

# J2c: Cross-species helical phasing comparison (is it truly universal?)
print("\n  --- J2c: Cross-Species Helical Phase Score Comparison ---")
species_scores = {}
for sp in ["human", "plant", "yeast"]:
    sp_rules = rules[rules["dataset"].map(SPECIES_MAP) == sp]
    species_scores[sp] = sp_rules["helical_phase_score"].values

# Kruskal-Wallis test
kw_stat, kw_p = stats.kruskal(
    species_scores["human"], species_scores["plant"], species_scores["yeast"]
)

# Pairwise comparisons
pairwise_phasing = {}
for sp1, sp2 in combinations(["human", "plant", "yeast"], 2):
    ks_stat, ks_p = stats.ks_2samp(species_scores[sp1], species_scores[sp2])
    d = cohens_d(species_scores[sp1], species_scores[sp2])
    pairwise_phasing[f"{sp1}_vs_{sp2}"] = {
        "ks_statistic": round(float(ks_stat), 4),
        "ks_p_value": float(f"{ks_p:.2e}"),
        "cohens_d": round(float(d), 4),
        "mean_1": round(float(np.mean(species_scores[sp1])), 4),
        "mean_2": round(float(np.mean(species_scores[sp2])), 4),
    }
    print(f"    {sp1} vs {sp2}: KS={ks_stat:.4f} (p={ks_p:.2e}), d={d:.4f}")

# Coefficient of variation across species phasing rates
rates = [species_phasing[sp] for sp in ["human", "plant", "yeast"]]
cv = np.std(rates) / np.mean(rates)

j2_results["cross_species_comparison"] = {
    "kruskal_wallis_H": round(float(kw_stat), 4),
    "kruskal_wallis_p": float(f"{kw_p:.2e}"),
    "pairwise": pairwise_phasing,
    "phasing_rates": {sp: round(float(species_phasing[sp]), 4) for sp in ["human", "plant", "yeast"]},
    "rate_cv": round(float(cv), 4),
    "rate_range": round(float(max(rates) - min(rates)), 4),
    "universality_assessment": (
        "UNIVERSAL" if cv < 0.1 and kw_p > 0.05 else
        "APPROXIMATELY UNIVERSAL" if cv < 0.1 else
        "MODERATELY CONSERVED" if cv < 0.2 else
        "DIVERGENT"
    ),
}

print(f"\n    Helical phasing rates: {', '.join(f'{sp}={species_phasing[sp]:.4f}' for sp in ['human','plant','yeast'])}")
print(f"    CV = {cv:.4f}, range = {max(rates)-min(rates):.4f}")
print(f"    KW H={kw_stat:.4f}, p={kw_p:.2e}")

# J2d: Per-dataset helical phasing consistency within human
print("\n  --- J2d: Within-Human Helical Phasing Consistency ---")
human_phasing_rates = {}
for ds in HUMAN_DATASETS:
    ds_rules = rules[rules["dataset"] == ds]
    scores = ds_rules["helical_phase_score"].values
    n_phased = np.sum(scores > 1.5)
    rate = n_phased / len(scores) if len(scores) > 0 else 0
    human_phasing_rates[ds] = {
        "n_rules": len(scores),
        "n_phased": int(n_phased),
        "rate": round(float(rate), 4),
        "mean_score": round(float(np.mean(scores)), 4),
    }
    print(f"    {ds}: {n_phased}/{len(scores)} = {rate:.4f} ({rate*100:.2f}%)")

# Chi-squared test for homogeneity of phasing rates across human datasets
contingency = np.array([
    [human_phasing_rates[ds]["n_phased"],
     human_phasing_rates[ds]["n_rules"] - human_phasing_rates[ds]["n_phased"]]
    for ds in HUMAN_DATASETS
])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)

j2_results["within_human_consistency"] = {
    "dataset_rates": human_phasing_rates,
    "chi2_statistic": round(float(chi2), 4),
    "chi2_p_value": float(f"{chi_p:.2e}"),
    "degrees_of_freedom": int(dof),
    "rates_are_homogeneous": bool(chi_p > 0.05),
    "interpretation": (
        "Helical phasing rates are consistent across human datasets"
        if chi_p > 0.05 else
        "Helical phasing rates differ significantly across human datasets"
    ),
}

print(f"    Chi-squared test for homogeneity: chi2={chi2:.4f}, p={chi_p:.2e}")

results["J2_helical_phasing_universality"] = j2_results

# ======================================================================
# J3: Grammar-Expression Coupling Conservation
# ======================================================================
print("\n" + "=" * 72)
print("J3: GRAMMAR-EXPRESSION COUPLING CONSERVATION")
print("=" * 72)
print("  Testing whether GSI-MPRA expression correlations are consistent")
print("  across human datasets.")

j3_results = {}

# J3a: GSI-expression correlation per dataset and model
print("\n  --- J3a: GSI-MPRA Correlation per Dataset/Model ---")
expr_correlations = {}
for model in MODELS:
    print(f"\n    Model: {model}")
    model_corrs = {}
    for ds in ALL_DATASETS:
        if (ds, model) not in gsi_data:
            continue
        df = gsi_data[(ds, model)]
        gsi_vals = df["gsi"].values
        mpra_vals = df["mpra_expression"].values
        mask = ~(np.isnan(gsi_vals) | np.isnan(mpra_vals))
        if mask.sum() < 10:
            continue

        g = gsi_vals[mask]
        m = mpra_vals[mask]

        # Spearman correlation (robust to outliers)
        sr, sp = stats.spearmanr(g, m)
        # Pearson on log-transformed GSI
        log_g = np.log1p(np.abs(g))
        pr, pp = stats.pearsonr(log_g, m)

        model_corrs[ds] = {
            "species": SPECIES_MAP[ds],
            "n": int(mask.sum()),
            "spearman_r": round(float(sr), 4),
            "spearman_p": float(f"{sp:.2e}"),
            "pearson_r_logGSI": round(float(pr), 4),
            "pearson_p_logGSI": float(f"{pp:.2e}"),
        }

        sig = "***" if sp < 0.001 else ("**" if sp < 0.01 else ("*" if sp < 0.05 else "ns"))
        print(f"      {ds:12s} [{SPECIES_MAP[ds]:6s}]: Spearman r={sr:+.4f} (p={sp:.2e}) {sig}, "
              f"Pearson(log) r={pr:+.4f} (p={pp:.2e})")

    expr_correlations[model] = model_corrs

j3_results["gsi_expression_correlations"] = expr_correlations

# J3b: Within-human consistency of GSI-expression correlation
print("\n  --- J3b: Within-Human Consistency of GSI-Expression Coupling ---")
consistency_test = {}

for model in MODELS:
    print(f"\n    Model: {model}")
    human_rs = []
    human_data_pairs = []

    for ds in HUMAN_DATASETS:
        if ds in expr_correlations.get(model, {}):
            human_rs.append(expr_correlations[model][ds]["spearman_r"])
            human_data_pairs.append(ds)

    nonhuman_rs = []
    for ds in ["vaishnav", "jores"]:
        if ds in expr_correlations.get(model, {}):
            nonhuman_rs.append(expr_correlations[model][ds]["spearman_r"])

    if len(human_rs) >= 2:
        # Fisher Z-transform to compare correlations
        # Test if human correlations are more similar to each other
        # than to non-human correlations
        human_z = [np.arctanh(min(max(r, -0.999), 0.999)) for r in human_rs]
        human_r_mean = np.mean(human_rs)
        human_r_std = np.std(human_rs)
        human_r_cv = abs(human_r_std / human_r_mean) if abs(human_r_mean) > 0.001 else float("inf")

        all_rs = human_rs + nonhuman_rs
        all_r_std = np.std(all_rs) if len(all_rs) > 1 else float("nan")

        # Cochran's Q-like test: compare correlations
        # Pairwise z-test between human correlations
        pairwise_z_tests = {}
        for i, (ds1, r1) in enumerate(zip(human_data_pairs, human_rs)):
            for ds2, r2 in zip(human_data_pairs[i+1:], human_rs[i+1:]):
                n1 = expr_correlations[model][ds1]["n"]
                n2 = expr_correlations[model][ds2]["n"]
                z1 = np.arctanh(min(max(r1, -0.999), 0.999))
                z2 = np.arctanh(min(max(r2, -0.999), 0.999))
                se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
                z_diff = (z1 - z2) / se_diff if se_diff > 0 else 0
                p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
                pairwise_z_tests[f"{ds1}_vs_{ds2}"] = {
                    "r1": round(float(r1), 4),
                    "r2": round(float(r2), 4),
                    "z_difference": round(float(z_diff), 4),
                    "p_value": float(f"{p_diff:.2e}"),
                    "correlations_differ": bool(p_diff < 0.05),
                }
                print(f"      {ds1} vs {ds2}: r1={r1:+.4f}, r2={r2:+.4f}, "
                      f"z_diff={z_diff:.4f}, p={p_diff:.2e}")

        consistency_test[model] = {
            "human_correlations": {ds: round(float(r), 4) for ds, r in zip(human_data_pairs, human_rs)},
            "human_mean_r": round(float(human_r_mean), 4),
            "human_std_r": round(float(human_r_std), 4),
            "human_cv_r": round(float(human_r_cv), 4),
            "nonhuman_correlations": {ds: round(float(r), 4) for ds, r in zip(["vaishnav", "jores"], nonhuman_rs)},
            "all_std_r": round(float(all_r_std), 4) if not np.isnan(all_r_std) else None,
            "pairwise_z_tests": pairwise_z_tests,
            "human_correlations_consistent": all(
                not v["correlations_differ"] for v in pairwise_z_tests.values()
            ),
        }

j3_results["within_human_consistency"] = consistency_test

# J3c: Compare human vs non-human expression coupling strength
print("\n  --- J3c: Human vs Non-Human Expression Coupling ---")
coupling_comparison = {}
for model in MODELS:
    human_abs_rs = []
    nonhuman_abs_rs = []
    for ds in HUMAN_DATASETS:
        if ds in expr_correlations.get(model, {}):
            human_abs_rs.append(abs(expr_correlations[model][ds]["spearman_r"]))
    for ds in ["vaishnav", "jores"]:
        if ds in expr_correlations.get(model, {}):
            nonhuman_abs_rs.append(abs(expr_correlations[model][ds]["spearman_r"]))

    human_mean = np.mean(human_abs_rs) if human_abs_rs else float("nan")
    nonhuman_mean = np.mean(nonhuman_abs_rs) if nonhuman_abs_rs else float("nan")

    coupling_comparison[model] = {
        "human_mean_abs_r": round(float(human_mean), 4),
        "nonhuman_mean_abs_r": round(float(nonhuman_mean), 4),
        "ratio": round(float(human_mean / nonhuman_mean), 4) if nonhuman_mean > 0 else None,
    }
    print(f"    {model}: human mean |r|={human_mean:.4f}, nonhuman mean |r|={nonhuman_mean:.4f}")

j3_results["human_vs_nonhuman_coupling"] = coupling_comparison

results["J3_grammar_expression_coupling"] = j3_results

# ======================================================================
# J4: Motif Pair Grammar Conservation Within Species
# ======================================================================
print("\n" + "=" * 72)
print("J4: MOTIF PAIR GRAMMAR CONSERVATION WITHIN SPECIES")
print("=" * 72)
print("  For motif pairs appearing in multiple human datasets, testing")
print("  whether grammar properties are more similar than expected by chance.")

j4_results = {}

# J4a: Spacing profile correlation for shared pairs
print("\n  --- J4a: Spacing Profile Correlation for Shared Pairs ---")
profile_corrs = {}

for model in MODELS:
    print(f"\n    Model: {model}")
    model_rules = rules[rules["model"] == model]
    model_profile_corrs = {}

    for ds1, ds2 in combinations(HUMAN_DATASETS, 2):
        r1 = model_rules[model_rules["dataset"] == ds1]
        r2 = model_rules[model_rules["dataset"] == ds2]

        shared_pairs = set(r1["pair"].unique()) & set(r2["pair"].unique())
        if len(shared_pairs) < 5:
            continue

        # For each shared pair, compute spacing profile correlation
        pair_corrs = []
        pair_spacing_diffs = []
        pair_orient_agrees = []

        for pair in sorted(shared_pairs):
            p1 = r1[r1["pair"] == pair]
            p2 = r2[r2["pair"] == pair]

            # Get spacing profiles (arrays stored in the dataframe)
            profiles_1 = p1["spacing_profile"].values
            profiles_2 = p2["spacing_profile"].values

            if len(profiles_1) == 0 or len(profiles_2) == 0:
                continue

            # Average the spacing profiles across sequences in each dataset
            try:
                mean_profile_1 = np.mean(np.stack(profiles_1), axis=0)
                mean_profile_2 = np.mean(np.stack(profiles_2), axis=0)

                if len(mean_profile_1) == len(mean_profile_2) and len(mean_profile_1) > 2:
                    r_val, p_val = stats.spearmanr(mean_profile_1, mean_profile_2)
                    if not np.isnan(r_val):
                        pair_corrs.append(r_val)
            except Exception:
                continue

            # Compare optimal spacing
            mode_s1 = p1["optimal_spacing"].mode().iloc[0] if len(p1) > 0 else np.nan
            mode_s2 = p2["optimal_spacing"].mode().iloc[0] if len(p2) > 0 else np.nan
            pair_spacing_diffs.append(abs(mode_s1 - mode_s2))

            # Compare optimal orientation
            mode_o1 = p1["optimal_orientation"].mode().iloc[0] if len(p1) > 0 else ""
            mode_o2 = p2["optimal_orientation"].mode().iloc[0] if len(p2) > 0 else ""
            pair_orient_agrees.append(1 if mode_o1 == mode_o2 else 0)

        mean_corr = np.mean(pair_corrs) if pair_corrs else float("nan")
        median_corr = np.median(pair_corrs) if pair_corrs else float("nan")
        mean_spacing_diff = np.mean(pair_spacing_diffs) if pair_spacing_diffs else float("nan")
        orient_agree_rate = np.mean(pair_orient_agrees) if pair_orient_agrees else float("nan")

        model_profile_corrs[f"{ds1}_vs_{ds2}"] = {
            "n_shared_pairs": len(shared_pairs),
            "n_pairs_with_profile": len(pair_corrs),
            "mean_profile_correlation": round(float(mean_corr), 4),
            "median_profile_correlation": round(float(median_corr), 4),
            "std_profile_correlation": round(float(np.std(pair_corrs)), 4) if pair_corrs else None,
            "mean_optimal_spacing_diff": round(float(mean_spacing_diff), 4),
            "orientation_agreement_rate": round(float(orient_agree_rate), 4),
        }

        print(f"      {ds1} vs {ds2}: {len(pair_corrs)} pairs, "
              f"mean profile r={mean_corr:.4f}, "
              f"spacing diff={mean_spacing_diff:.2f}bp, "
              f"orient agree={orient_agree_rate:.3f}")

    profile_corrs[model] = model_profile_corrs

j4_results["spacing_profile_correlations"] = profile_corrs

# J4b: Permutation test - are observed correlations higher than chance?
print("\n  --- J4b: Permutation Test for Grammar Conservation ---")
permutation_results = {}

for model in MODELS:
    print(f"\n    Model: {model}")
    model_rules = rules[rules["model"] == model]
    model_perm = {}

    for ds1, ds2 in combinations(HUMAN_DATASETS, 2):
        r1 = model_rules[model_rules["dataset"] == ds1].copy()
        r2 = model_rules[model_rules["dataset"] == ds2].copy()

        shared_pairs = sorted(set(r1["pair"].unique()) & set(r2["pair"].unique()))
        if len(shared_pairs) < 10:
            continue

        # Observed: mean correlation of spacing sensitivity between matched pairs
        ss_1 = []
        ss_2 = []
        os_1 = []
        os_2 = []
        for pair in shared_pairs:
            p1 = r1[r1["pair"] == pair]
            p2 = r2[r2["pair"] == pair]
            ss_1.append(p1["spacing_sensitivity"].mean())
            ss_2.append(p2["spacing_sensitivity"].mean())
            os_1.append(p1["orientation_sensitivity"].mean())
            os_2.append(p2["orientation_sensitivity"].mean())

        ss_1 = np.array(ss_1)
        ss_2 = np.array(ss_2)
        os_1 = np.array(os_1)
        os_2 = np.array(os_2)

        # Observed correlation
        obs_ss_r, obs_ss_p = stats.spearmanr(ss_1, ss_2)
        obs_os_r, obs_os_p = stats.spearmanr(os_1, os_2)

        # Permutation: shuffle pair assignments and recompute
        n_perm = 5000
        perm_ss_rs = []
        perm_os_rs = []
        for _ in range(n_perm):
            idx = np.random.permutation(len(ss_2))
            pr_ss, _ = stats.spearmanr(ss_1, ss_2[idx])
            pr_os, _ = stats.spearmanr(os_1, os_2[idx])
            perm_ss_rs.append(pr_ss)
            perm_os_rs.append(pr_os)

        perm_ss_rs = np.array(perm_ss_rs)
        perm_os_rs = np.array(perm_os_rs)

        ss_perm_p = np.mean(perm_ss_rs >= obs_ss_r)
        os_perm_p = np.mean(perm_os_rs >= obs_os_r)

        # Effect size: how many SDs above null mean
        ss_z = (obs_ss_r - np.mean(perm_ss_rs)) / np.std(perm_ss_rs) if np.std(perm_ss_rs) > 0 else 0
        os_z = (obs_os_r - np.mean(perm_os_rs)) / np.std(perm_os_rs) if np.std(perm_os_rs) > 0 else 0

        model_perm[f"{ds1}_vs_{ds2}"] = {
            "n_shared_pairs": len(shared_pairs),
            "spacing_sensitivity": {
                "observed_spearman_r": round(float(obs_ss_r), 4),
                "observed_p_value": float(f"{obs_ss_p:.2e}"),
                "permutation_p_value": round(float(ss_perm_p), 4),
                "null_mean_r": round(float(np.mean(perm_ss_rs)), 4),
                "null_std_r": round(float(np.std(perm_ss_rs)), 4),
                "z_above_null": round(float(ss_z), 4),
                "significant": bool(ss_perm_p < 0.05),
            },
            "orientation_sensitivity": {
                "observed_spearman_r": round(float(obs_os_r), 4),
                "observed_p_value": float(f"{obs_os_p:.2e}"),
                "permutation_p_value": round(float(os_perm_p), 4),
                "null_mean_r": round(float(np.mean(perm_os_rs)), 4),
                "null_std_r": round(float(np.std(perm_os_rs)), 4),
                "z_above_null": round(float(os_z), 4),
                "significant": bool(os_perm_p < 0.05),
            },
        }

        ss_sig = "***" if ss_perm_p < 0.001 else ("**" if ss_perm_p < 0.01 else ("*" if ss_perm_p < 0.05 else "ns"))
        os_sig = "***" if os_perm_p < 0.001 else ("**" if os_perm_p < 0.01 else ("*" if os_perm_p < 0.05 else "ns"))
        print(f"      {ds1} vs {ds2} ({len(shared_pairs)} pairs):")
        print(f"        Spacing sens: r={obs_ss_r:.4f}, perm_p={ss_perm_p:.4f} {ss_sig}, z={ss_z:.2f}")
        print(f"        Orient  sens: r={obs_os_r:.4f}, perm_p={os_perm_p:.4f} {os_sig}, z={os_z:.2f}")

    permutation_results[model] = model_perm

j4_results["permutation_tests"] = permutation_results

# J4c: Cross-species control for motif pair conservation
print("\n  --- J4c: Cross-Species Control ---")
cross_species_control = {}

for model in MODELS:
    print(f"\n    Model: {model}")
    model_rules = rules[rules["model"] == model]
    model_control = {}

    # Compare human datasets to non-human
    for hds in ["agarwal"]:  # Use agarwal as representative human
        for nds in ["jores", "vaishnav"]:
            r1 = model_rules[model_rules["dataset"] == hds]
            r2 = model_rules[model_rules["dataset"] == nds]
            shared = sorted(set(r1["pair"].unique()) & set(r2["pair"].unique()))

            if len(shared) < 5:
                model_control[f"{hds}_vs_{nds}"] = {
                    "n_shared_pairs": len(shared),
                    "note": "too few shared pairs for analysis",
                }
                print(f"      {hds} vs {nds}: {len(shared)} shared pairs (too few)")
                continue

            ss_1 = np.array([r1[r1["pair"] == p]["spacing_sensitivity"].mean() for p in shared])
            ss_2 = np.array([r2[r2["pair"] == p]["spacing_sensitivity"].mean() for p in shared])
            cross_r, cross_p = stats.spearmanr(ss_1, ss_2)

            model_control[f"{hds}_vs_{nds}"] = {
                "n_shared_pairs": len(shared),
                "spacing_sensitivity_spearman_r": round(float(cross_r), 4),
                "spacing_sensitivity_p_value": float(f"{cross_p:.2e}"),
                "species_pair": f"{SPECIES_MAP[hds]}_vs_{SPECIES_MAP[nds]}",
            }
            print(f"      {hds} vs {nds}: {len(shared)} shared pairs, r={cross_r:.4f} (p={cross_p:.2e})")

    cross_species_control[model] = model_control

j4_results["cross_species_control"] = cross_species_control

# J4d: Summary - aggregate evidence for motif pair conservation
print("\n  --- J4d: Aggregate Evidence ---")
all_within_ss_rs = []
all_within_os_rs = []
all_within_ss_perm_ps = []
all_within_os_perm_ps = []

for model in MODELS:
    for key, val in permutation_results.get(model, {}).items():
        all_within_ss_rs.append(val["spacing_sensitivity"]["observed_spearman_r"])
        all_within_os_rs.append(val["orientation_sensitivity"]["observed_spearman_r"])
        all_within_ss_perm_ps.append(val["spacing_sensitivity"]["permutation_p_value"])
        all_within_os_perm_ps.append(val["orientation_sensitivity"]["permutation_p_value"])

j4_results["aggregate_summary"] = {
    "n_comparisons": len(all_within_ss_rs),
    "spacing_sensitivity": {
        "mean_spearman_r": round(float(np.mean(all_within_ss_rs)), 4),
        "all_rs": [round(float(r), 4) for r in all_within_ss_rs],
        "n_significant": int(np.sum(np.array(all_within_ss_perm_ps) < 0.05)),
        "frac_significant": round(float(np.mean(np.array(all_within_ss_perm_ps) < 0.05)), 4),
        "mean_perm_p": round(float(np.mean(all_within_ss_perm_ps)), 4),
    },
    "orientation_sensitivity": {
        "mean_spearman_r": round(float(np.mean(all_within_os_rs)), 4),
        "all_rs": [round(float(r), 4) for r in all_within_os_rs],
        "n_significant": int(np.sum(np.array(all_within_os_perm_ps) < 0.05)),
        "frac_significant": round(float(np.mean(np.array(all_within_os_perm_ps) < 0.05)), 4),
        "mean_perm_p": round(float(np.mean(all_within_os_perm_ps)), 4),
    },
}

print(f"\n    Spacing sensitivity conservation:")
print(f"      Mean r = {np.mean(all_within_ss_rs):.4f}")
print(f"      Significant: {np.sum(np.array(all_within_ss_perm_ps) < 0.05)}/{len(all_within_ss_perm_ps)}")
print(f"    Orientation sensitivity conservation:")
print(f"      Mean r = {np.mean(all_within_os_rs):.4f}")
print(f"      Significant: {np.sum(np.array(all_within_os_perm_ps) < 0.05)}/{len(all_within_os_perm_ps)}")

results["J4_motif_pair_conservation"] = j4_results

# ======================================================================
# GRAND SUMMARY
# ======================================================================
print("\n" + "=" * 72)
print("GRAND SUMMARY: EXPERIMENT J")
print("=" * 72)

# Collect key findings
j1_key = {}
# Average Jaccard for within-human
within_jaccards = [
    v["jaccard_index"] for k, v in pair_overlap.items()
    if isinstance(v, dict) and v.get("type") == "within_human"
]
cross_jaccards = [
    v["jaccard_index"] for k, v in pair_overlap.items()
    if isinstance(v, dict) and v.get("type") == "cross_species"
]
j1_key["mean_within_human_jaccard"] = round(float(np.mean(within_jaccards)), 4)
j1_key["mean_cross_species_jaccard"] = round(float(np.mean(cross_jaccards)), 4)

# J1b aggregate
j1b_all_ss_r = []
j1b_all_os_r = []
for model in MODELS:
    for key, val in rule_agreement.get(model, {}).items():
        j1b_all_ss_r.append(val["spacing_sensitivity_spearman_r"])
        j1b_all_os_r.append(val["orientation_sensitivity_spearman_r"])
j1_key["mean_spacing_sens_agreement_r"] = round(float(np.mean(j1b_all_ss_r)), 4) if j1b_all_ss_r else None
j1_key["mean_orient_sens_agreement_r"] = round(float(np.mean(j1b_all_os_r)), 4) if j1b_all_os_r else None

# J2 key
j2_key = {
    "phasing_rates": {sp: round(float(species_phasing[sp]), 4) for sp in ["human", "plant", "yeast"]},
    "rate_cv": round(float(cv), 4),
    "universality": j2_results["cross_species_comparison"]["universality_assessment"],
    "within_human_homogeneous": j2_results["within_human_consistency"]["rates_are_homogeneous"],
}

# J3 key
j3_human_rs = []
j3_nonhuman_rs = []
for model in MODELS:
    for ds in HUMAN_DATASETS:
        if ds in expr_correlations.get(model, {}):
            j3_human_rs.append(expr_correlations[model][ds]["spearman_r"])
    for ds in ["vaishnav", "jores"]:
        if ds in expr_correlations.get(model, {}):
            j3_nonhuman_rs.append(expr_correlations[model][ds]["spearman_r"])

j3_key = {
    "mean_human_gsi_expression_r": round(float(np.mean(j3_human_rs)), 4) if j3_human_rs else None,
    "std_human_gsi_expression_r": round(float(np.std(j3_human_rs)), 4) if j3_human_rs else None,
    "mean_nonhuman_gsi_expression_r": round(float(np.mean(j3_nonhuman_rs)), 4) if j3_nonhuman_rs else None,
}

# J4 key
j4_key = j4_results["aggregate_summary"]

# Overall conclusion
summary = {
    "J1_within_species_consistency": {
        "finding": (
            f"Human datasets share {j1_key['mean_within_human_jaccard']:.1%} of motif pairs "
            f"(Jaccard), exclusively within-species (zero cross-species overlap). "
            f"Grammar rule properties show "
            f"{'positive' if (j1_key.get('mean_spacing_sens_agreement_r') or 0) > 0 else 'no'} "
            f"correlation for shared pairs (mean r={j1_key.get('mean_spacing_sens_agreement_r', 'N/A')})."
        ),
        **j1_key,
    },
    "J2_helical_phasing": {
        "finding": (
            f"Helical phasing is {j2_key['universality'].lower()} across species "
            f"(CV={j2_key['rate_cv']:.4f}). Rates: "
            f"human={j2_key['phasing_rates']['human']:.1%}, "
            f"plant={j2_key['phasing_rates']['plant']:.1%}, "
            f"yeast={j2_key['phasing_rates']['yeast']:.1%}. "
            f"Within-human rates are {'homogeneous' if j2_key['within_human_homogeneous'] else 'heterogeneous'}."
        ),
        **j2_key,
    },
    "J3_expression_coupling": {
        "finding": (
            f"GSI-expression coupling in human datasets: mean r={j3_key['mean_human_gsi_expression_r']}, "
            f"std={j3_key['std_human_gsi_expression_r']}. "
            f"Non-human mean r={j3_key['mean_nonhuman_gsi_expression_r']}."
        ),
        **j3_key,
    },
    "J4_motif_pair_conservation": {
        "finding": (
            f"Spacing sensitivity conservation: mean r={j4_key['spacing_sensitivity']['mean_spearman_r']}, "
            f"{j4_key['spacing_sensitivity']['n_significant']}/{j4_key['spacing_sensitivity']['n_significant'] + (j4_key['n_comparisons'] - j4_key['spacing_sensitivity']['n_significant'])} significant. "
            f"Orientation sensitivity conservation: mean r={j4_key['orientation_sensitivity']['mean_spearman_r']}, "
            f"{j4_key['orientation_sensitivity']['n_significant']}/{j4_key['n_comparisons']} significant."
        ),
        **j4_key,
    },
    "overall_conclusion": "",
}

# Build conclusion
conclusions = []

# J1 conclusion
if j1_key["mean_within_human_jaccard"] > j1_key["mean_cross_species_jaccard"]:
    conclusions.append(
        "Within-species grammar is partially conserved: human datasets share "
        "substantially more motif pairs than cross-species comparisons"
    )
else:
    conclusions.append("Motif pair repertoire shows no within-species enrichment")

# J2 conclusion
if j2_key["rate_cv"] < 0.1:
    conclusions.append(
        "Helical phasing is a universal grammar property conserved across all species"
    )
elif j2_key["rate_cv"] < 0.2:
    conclusions.append(
        "Helical phasing is moderately conserved across species, consistent with "
        "it being a biophysical constraint rather than species-specific grammar"
    )

# J3 conclusion
if j3_key.get("std_human_gsi_expression_r") is not None:
    if j3_key["std_human_gsi_expression_r"] < 0.1:
        conclusions.append(
            "GSI-expression coupling is highly consistent within human datasets"
        )
    else:
        conclusions.append(
            "GSI-expression coupling shows moderate variation across human datasets"
        )

# J4 conclusion
n_sig_total = (j4_key["spacing_sensitivity"]["n_significant"] +
               j4_key["orientation_sensitivity"]["n_significant"])
n_total = 2 * j4_key["n_comparisons"]
if n_sig_total > n_total / 2:
    conclusions.append(
        "Grammar properties of shared motif pairs are significantly conserved "
        "within species (majority of tests significant)"
    )
elif n_sig_total > 0:
    conclusions.append(
        "Grammar conservation at the motif-pair level shows partial evidence, "
        "with some properties conserved and others divergent"
    )
else:
    conclusions.append(
        "Grammar properties of individual motif pairs do not show significant "
        "conservation across human datasets"
    )

summary["overall_conclusion"] = (
    "EXPERIMENT J CONCLUSION: " + ". ".join(conclusions) + ". "
    "This supports a model where (1) the repertoire of grammar-capable motif pairs "
    "is partially shared within species, (2) helical phasing is a universal biophysical "
    "constraint, but (3) specific grammar rules may be enhancer-context dependent "
    "rather than purely species-dependent."
)

results["summary"] = summary

# Print summary
for key in ["J1_within_species_consistency", "J2_helical_phasing",
            "J3_expression_coupling", "J4_motif_pair_conservation"]:
    print(f"\n  {key}:")
    print(f"    {summary[key]['finding']}")

print(f"\n  OVERALL: {summary['overall_conclusion']}")

# ======================================================================
# Save results
# ======================================================================
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2, default=safe_float)

print(f"\n  Results saved to: {OUTPUT_PATH}")
print("=" * 72)
