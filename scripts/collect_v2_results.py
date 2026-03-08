#!/usr/bin/env python3
"""
collect_v2_results.py  --  Collect and summarize all v2 pipeline results.

Run AFTER the v2 pipeline completes (or partially completes).
Produces:
  - results/v2/v2_results_summary.json  (machine-readable)
  - Formatted report to stdout

Usage:
    conda run -n gramlang python scripts/collect_v2_results.py
"""

import json
import os
import sys
import glob
from datetime import datetime
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
V1_DIR = RESULTS_DIR  # v1 results are at results/module1-6/
V2_DIR = RESULTS_DIR / "v2"

DATASETS = ["agarwal", "inoue", "vaishnav", "jores", "klein"]
FOUNDATION_MODELS = ["dnabert2", "nt", "hyenadna"]
ALL_MODELS = FOUNDATION_MODELS + ["enformer"]

OUTPUT_JSON = V2_DIR / "v2_results_summary.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_load_json(path):
    """Load a JSON file, returning None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def safe_read_parquet(path):
    """Read a parquet file, returning None on failure."""
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def safe_float(x, decimals=6):
    """Convert to float, handling numpy types and NaN."""
    if x is None:
        return None
    try:
        val = float(x)
        if np.isnan(val) or np.isinf(val):
            return None
        return round(val, decimals)
    except (TypeError, ValueError):
        return None


def fmt_pct(x, decimals=1):
    """Format a fraction as a percentage string."""
    if x is None:
        return "N/A"
    return f"{x * 100:.{decimals}f}%"


def fmt_float(x, decimals=4):
    """Format a float with fixed decimals."""
    if x is None:
        return "N/A"
    return f"{x:.{decimals}f}"


def section_header(title, char="=", width=80):
    """Print a section header."""
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def sub_header(title, char="-", width=60):
    """Print a sub-section header."""
    print()
    print(f"  {char * 4} {title} {char * 4}")


# ---------------------------------------------------------------------------
# Module 1: GSI Census
# ---------------------------------------------------------------------------

def collect_module1():
    """Collect v1 and v2 Module 1 (GSI Census) results."""
    result = {
        "status": "not_found",
        "v1": None,
        "v2": None,
        "comparison": None,
    }

    # --- v1 ---
    v1_summary = safe_load_json(V1_DIR / "module1" / "gsi_summary.json")
    v1_gsi = safe_read_parquet(V1_DIR / "module1" / "all_gsi_results.parquet")
    if v1_summary is not None:
        result["v1"] = {
            "summary": v1_summary,
            "n_total": int(len(v1_gsi)) if v1_gsi is not None else None,
            "overall_mean_gsi": safe_float(v1_gsi["gsi"].mean()) if v1_gsi is not None else None,
            "overall_median_gsi": safe_float(v1_gsi["gsi"].median()) if v1_gsi is not None else None,
        }

    # --- v2 ---
    v2_dir = V2_DIR / "module1"
    if not v2_dir.exists():
        return result

    # Discover available parquet files
    v2_parquets = sorted(v2_dir.glob("*_gsi.parquet"))
    v2_parquets = [p for p in v2_parquets if p.name != "all_gsi_results.parquet"]

    if not v2_parquets:
        return result

    result["status"] = "complete"

    # Load individual files and combine
    frames = []
    file_inventory = {}
    for pq in v2_parquets:
        name = pq.stem  # e.g. "agarwal_dnabert2_gsi"
        df = safe_read_parquet(pq)
        if df is not None:
            frames.append(df)
            file_inventory[name] = len(df)

    if not frames:
        result["status"] = "empty"
        return result

    v2_gsi = pd.concat(frames, ignore_index=True)
    result["v2"] = {
        "n_total": int(len(v2_gsi)),
        "n_files": len(file_inventory),
        "file_inventory": file_inventory,
        "overall_mean_gsi": safe_float(v2_gsi["gsi"].mean()),
        "overall_median_gsi": safe_float(v2_gsi["gsi"].median()),
    }

    # Per-dataset summary
    per_dataset = {}
    for ds in v2_gsi["dataset"].unique():
        ds_data = v2_gsi[v2_gsi["dataset"] == ds]
        per_dataset[ds] = {
            "n": int(len(ds_data)),
            "n_enhancers": int(ds_data["seq_id"].nunique()),
            "models": sorted(ds_data["model"].unique().tolist()) if "model" in ds_data.columns else [],
            "mean_gsi": safe_float(ds_data["gsi"].mean()),
            "median_gsi": safe_float(ds_data["gsi"].median()),
            "std_gsi": safe_float(ds_data["gsi"].std()),
        }
    result["v2"]["per_dataset"] = per_dataset

    # Per-model summary
    if "model" in v2_gsi.columns:
        per_model = {}
        for model in v2_gsi["model"].unique():
            m_data = v2_gsi[v2_gsi["model"] == model]
            per_model[model] = {
                "n": int(len(m_data)),
                "mean_gsi": safe_float(m_data["gsi"].mean()),
                "median_gsi": safe_float(m_data["gsi"].median()),
            }
        result["v2"]["per_model"] = per_model

    # Load p-value correction summary
    pval_correction = safe_load_json(v2_dir / "p_value_correction_summary.json")
    if pval_correction is not None:
        result["v2"]["p_value_correction"] = pval_correction
        # Compute overall corrected significance rate
        total_sig = sum(v.get("n_sig", 0) for v in pval_correction.values())
        total_n = sum(v.get("n_sig", 0) / v["frac_sig"]
                      if v.get("frac_sig", 0) > 0 else 0
                      for v in pval_correction.values())
        if total_n > 0:
            result["v2"]["corrected_frac_significant"] = safe_float(total_sig / total_n)
        median_frac_sig = np.median([v.get("frac_sig", 0) for v in pval_correction.values()])
        result["v2"]["median_frac_significant_per_combo"] = safe_float(median_frac_sig)

    # Load extended analysis
    v2_analysis = safe_load_json(v2_dir / "v2_gsi_analysis.json")
    if v2_analysis is not None:
        result["v2"]["extended_analysis"] = {
            "best_model_mean_gsi": v2_analysis.get("best_model_mean_gsi"),
            "best_model_median_gsi": v2_analysis.get("best_model_median_gsi"),
            "strongest_dataset_mean_gsi": v2_analysis.get("strongest_dataset_mean_gsi"),
            "strongest_dataset_median_gsi": v2_analysis.get("strongest_dataset_median_gsi"),
            "anova": v2_analysis.get("anova"),
            "kruskal_wallis": v2_analysis.get("kruskal_wallis"),
            "motif_density_summary": v2_analysis.get("motif_density_summary"),
            "expression_vs_gsi_summary": v2_analysis.get("expression_vs_gsi_summary"),
        }

    # Cross-dataset validation
    cross_val = safe_load_json(v2_dir / "cross_dataset_validation.json")
    if cross_val is not None:
        result["v2"]["cross_dataset_validation_available"] = True

    # --- v1 vs v2 comparison ---
    if result["v1"] is not None and v1_gsi is not None:
        comp = {}
        # Overall GSI shift
        v1_mean = v1_gsi["gsi"].mean()
        v2_mean = v2_gsi["gsi"].mean()
        comp["gsi_mean_v1"] = safe_float(v1_mean)
        comp["gsi_mean_v2"] = safe_float(v2_mean)
        comp["gsi_mean_ratio_v2_over_v1"] = safe_float(v2_mean / v1_mean) if v1_mean != 0 else None
        comp["gsi_median_v1"] = safe_float(v1_gsi["gsi"].median())
        comp["gsi_median_v2"] = safe_float(v2_gsi["gsi"].median())

        # v1 had 200 enhancers per dataset, v2 has 500
        comp["sample_size_v1"] = int(len(v1_gsi))
        comp["sample_size_v2"] = int(len(v2_gsi))

        # v1 significance was 100% (artifact); v2 has corrected p-values
        comp["v1_frac_significant_uncorrected"] = 1.0
        if "corrected_frac_significant" in result["v2"]:
            comp["v2_frac_significant_corrected"] = result["v2"]["corrected_frac_significant"]

        # Per-dataset GSI comparison
        per_ds_comp = {}
        for ds in DATASETS:
            v1_ds = v1_summary.get(ds, {}) if v1_summary else {}
            v2_ds = per_dataset.get(ds, {})
            if v1_ds and v2_ds:
                per_ds_comp[ds] = {
                    "v1_mean_gsi": safe_float(v1_ds.get("mean_gsi")),
                    "v2_mean_gsi": v2_ds.get("mean_gsi"),
                    "v1_median_gsi": safe_float(v1_ds.get("median_gsi")),
                    "v2_median_gsi": v2_ds.get("median_gsi"),
                    "v1_n": int(v1_ds.get("n_enhancers", 0)),
                    "v2_n": v2_ds.get("n_enhancers", 0),
                }
        comp["per_dataset"] = per_ds_comp
        result["comparison"] = comp

    return result


# ---------------------------------------------------------------------------
# Module 2: Grammar Rules
# ---------------------------------------------------------------------------

def collect_module2():
    """Collect v1 and v2 Module 2 (Grammar Rules) results."""
    result = {
        "status": "not_found",
        "v1": None,
        "v2": None,
        "comparison": None,
    }

    # --- v1 ---
    v1_consensus = safe_load_json(V1_DIR / "module2" / "global_consensus.json")
    v1_rules = safe_read_parquet(V1_DIR / "module2" / "grammar_rules_database.parquet")
    if v1_consensus is not None:
        v1_info = dict(v1_consensus)
        if v1_rules is not None:
            v1_info["n_rules_total"] = int(len(v1_rules))
            v1_info["n_unique_pairs"] = int(v1_rules["pair"].nunique())
            v1_info["n_datasets"] = int(v1_rules["dataset"].nunique())
            v1_info["n_models"] = int(v1_rules["model"].nunique())
            v1_info["mean_fold_change"] = safe_float(v1_rules["fold_change"].mean())
            v1_info["mean_spacing_sensitivity"] = safe_float(v1_rules["spacing_sensitivity"].mean())
            v1_info["mean_orientation_sensitivity"] = safe_float(v1_rules["orientation_sensitivity"].mean())
            # Orientation bias
            if "optimal_orientation" in v1_rules.columns:
                orient_counts = v1_rules["optimal_orientation"].value_counts()
                v1_info["orientation_distribution"] = {
                    str(k): int(v) for k, v in orient_counts.items()
                }
                plus_plus = orient_counts.get("++", orient_counts.get(1, 0))
                v1_info["plus_plus_fraction"] = safe_float(
                    plus_plus / len(v1_rules) if len(v1_rules) > 0 else 0
                )
        result["v1"] = v1_info

    # v1 rules analysis (from v2 directory)
    v1_rules_analysis = safe_load_json(V2_DIR / "v1_rules_analysis.json")
    if v1_rules_analysis is not None and result["v1"] is not None:
        result["v1"]["v1_rules_analysis_available"] = True
        result["v1"]["unique_motifs_union"] = v1_rules_analysis.get("unique_counts", {}).get(
            "unique_motifs_union"
        )

    # --- v2 ---
    v2_dir = V2_DIR / "module2"
    if not v2_dir.exists():
        return result

    v2_consensus = safe_load_json(v2_dir / "global_consensus.json")
    v2_rules = safe_read_parquet(v2_dir / "grammar_rules_database.parquet")
    v2_consensus_df = safe_read_parquet(v2_dir / "consensus_scores.parquet")

    if v2_consensus is None and v2_rules is None:
        return result

    result["status"] = "complete"
    v2_info = {}

    if v2_consensus is not None:
        v2_info.update(v2_consensus)

    if v2_rules is not None:
        v2_info["n_rules_total"] = int(len(v2_rules))
        v2_info["n_unique_pairs"] = int(v2_rules["pair"].nunique()) if "pair" in v2_rules.columns else None
        v2_info["n_datasets"] = int(v2_rules["dataset"].nunique()) if "dataset" in v2_rules.columns else None
        v2_info["n_models"] = int(v2_rules["model"].nunique()) if "model" in v2_rules.columns else None
        v2_info["mean_fold_change"] = safe_float(v2_rules["fold_change"].mean()) if "fold_change" in v2_rules.columns else None
        v2_info["mean_spacing_sensitivity"] = safe_float(
            v2_rules["spacing_sensitivity"].mean()
        ) if "spacing_sensitivity" in v2_rules.columns else None
        v2_info["mean_orientation_sensitivity"] = safe_float(
            v2_rules["orientation_sensitivity"].mean()
        ) if "orientation_sensitivity" in v2_rules.columns else None
        if "optimal_orientation" in v2_rules.columns:
            orient_counts = v2_rules["optimal_orientation"].value_counts()
            v2_info["orientation_distribution"] = {
                str(k): int(v) for k, v in orient_counts.items()
            }
            plus_plus = orient_counts.get("++", orient_counts.get(1, 0))
            v2_info["plus_plus_fraction"] = safe_float(
                plus_plus / len(v2_rules) if len(v2_rules) > 0 else 0
            )
        # Per-dataset rule counts
        if "dataset" in v2_rules.columns:
            per_ds = {}
            for ds in v2_rules["dataset"].unique():
                ds_rules = v2_rules[v2_rules["dataset"] == ds]
                per_ds[ds] = {
                    "n_rules": int(len(ds_rules)),
                    "n_unique_pairs": int(ds_rules["pair"].nunique()) if "pair" in ds_rules.columns else None,
                }
            v2_info["per_dataset"] = per_ds

    if v2_consensus_df is not None:
        v2_info["n_consensus_entries"] = int(len(v2_consensus_df))
        if "consensus_score" in v2_consensus_df.columns:
            v2_info["consensus_score_mean"] = safe_float(v2_consensus_df["consensus_score"].mean())
            v2_info["consensus_score_median"] = safe_float(v2_consensus_df["consensus_score"].median())

    result["v2"] = v2_info

    # --- Comparison ---
    if result["v1"] is not None and result["v2"] is not None:
        comp = {}
        for key in ["n_rules_total", "mean_consensus", "mean_orientation_agreement",
                     "mean_spacing_correlation", "frac_high_consensus",
                     "mean_fold_change", "mean_spacing_sensitivity",
                     "mean_orientation_sensitivity", "plus_plus_fraction"]:
            v1_val = result["v1"].get(key)
            v2_val = result["v2"].get(key)
            if v1_val is not None and v2_val is not None:
                comp[key] = {
                    "v1": safe_float(v1_val),
                    "v2": safe_float(v2_val),
                }
                try:
                    if float(v1_val) != 0:
                        comp[key]["ratio_v2_over_v1"] = safe_float(float(v2_val) / float(v1_val))
                except (TypeError, ValueError):
                    pass
        result["comparison"] = comp

    return result


# ---------------------------------------------------------------------------
# Module 3: Compositionality
# ---------------------------------------------------------------------------

def collect_module3():
    """Collect v1 and v2 Module 3 (Compositionality) results."""
    result = {
        "status": "not_found",
        "v1": None,
        "v2": None,
        "comparison": None,
    }

    # --- v1 ---
    v1_class = safe_load_json(V1_DIR / "module3" / "complexity_classification.json")
    v1_comp = safe_read_parquet(V1_DIR / "module3" / "compositionality_results.parquet")
    if v1_class is not None:
        v1_info = {
            "classification": v1_class.get("classification"),
            "mean_gap": safe_float(v1_class.get("mean_gap")),
            "confidence": safe_float(v1_class.get("confidence")),
            "best_bic_model": v1_class.get("best_bic_model"),
        }
        if v1_comp is not None:
            v1_info["n_tests"] = int(len(v1_comp))
            v1_info["overall_mean_gap"] = safe_float(v1_comp["compositionality_gap"].mean())
            v1_info["overall_mean_r2"] = safe_float(v1_comp["pairwise_r2"].mean())
            # Per-k breakdown
            if "n_motifs" in v1_comp.columns:
                per_k = {}
                for k in sorted(v1_comp["n_motifs"].unique()):
                    k_data = v1_comp[v1_comp["n_motifs"] == k]
                    per_k[int(k)] = {
                        "n_tests": int(len(k_data)),
                        "mean_gap": safe_float(k_data["compositionality_gap"].mean()),
                        "mean_r2": safe_float(k_data["pairwise_r2"].mean()),
                    }
                v1_info["per_k"] = per_k
        result["v1"] = v1_info

    # --- v2 ---
    v2_dir = V2_DIR / "module3"
    if not v2_dir.exists():
        return result

    v2_class = safe_load_json(v2_dir / "complexity_classification.json")
    v2_comp = safe_read_parquet(v2_dir / "compositionality_results.parquet")

    if v2_class is None and v2_comp is None:
        return result

    result["status"] = "complete"
    v2_info = {}

    if v2_class is not None:
        v2_info["classification"] = v2_class.get("classification")
        v2_info["mean_gap"] = safe_float(v2_class.get("mean_gap"))
        v2_info["confidence"] = safe_float(v2_class.get("confidence"))
        v2_info["best_bic_model"] = v2_class.get("best_bic_model")

    if v2_comp is not None:
        v2_info["n_tests"] = int(len(v2_comp))
        v2_info["overall_mean_gap"] = safe_float(v2_comp["compositionality_gap"].mean())
        v2_info["overall_mean_r2"] = safe_float(
            v2_comp["pairwise_r2"].mean()
        ) if "pairwise_r2" in v2_comp.columns else None
        if "n_motifs" in v2_comp.columns:
            per_k = {}
            for k in sorted(v2_comp["n_motifs"].unique()):
                k_data = v2_comp[v2_comp["n_motifs"] == k]
                per_k[int(k)] = {
                    "n_tests": int(len(k_data)),
                    "mean_gap": safe_float(k_data["compositionality_gap"].mean()),
                }
            v2_info["per_k"] = per_k

    result["v2"] = v2_info

    # --- Comparison ---
    if result["v1"] is not None and result["v2"] is not None:
        comp = {}
        comp["classification_v1"] = result["v1"].get("classification")
        comp["classification_v2"] = result["v2"].get("classification")
        comp["classification_changed"] = comp["classification_v1"] != comp["classification_v2"]
        comp["mean_gap_v1"] = result["v1"].get("mean_gap") or result["v1"].get("overall_mean_gap")
        comp["mean_gap_v2"] = result["v2"].get("mean_gap") or result["v2"].get("overall_mean_gap")
        comp["n_tests_v1"] = result["v1"].get("n_tests")
        comp["n_tests_v2"] = result["v2"].get("n_tests")
        result["comparison"] = comp

    return result


# ---------------------------------------------------------------------------
# Module 4: Cross-Species Transfer
# ---------------------------------------------------------------------------

def collect_module4():
    """Collect v1 and v2 Module 4 (Transfer) results."""
    result = {
        "status": "not_found",
        "v1": None,
        "v2": None,
        "comparison": None,
    }

    # --- v1 ---
    v1_phylo = safe_load_json(V1_DIR / "module4" / "grammar_phylogeny.json")
    v1_transfer = safe_read_parquet(V1_DIR / "module4" / "transfer_matrix.parquet")
    if v1_transfer is not None:
        v1_info = {
            "n_entries": int(len(v1_transfer)),
            "species_list": v1_phylo.get("species_list", []) if v1_phylo else [],
        }
        # Transfer R2 matrix
        transfer_dict = {}
        for _, row in v1_transfer.iterrows():
            key = f"{row['source']}_to_{row['target']}"
            transfer_dict[key] = safe_float(row.get("transfer_r2", row.get("transfer_corr", 0)))
        v1_info["transfer_r2"] = transfer_dict

        # Within vs cross species
        within = v1_transfer[v1_transfer["source"] == v1_transfer["target"]]["transfer_r2"]
        cross = v1_transfer[v1_transfer["source"] != v1_transfer["target"]]["transfer_r2"]
        v1_info["mean_within_r2"] = safe_float(within.mean())
        v1_info["mean_cross_r2"] = safe_float(cross.mean())

        result["v1"] = v1_info

    # --- v2 ---
    v2_dir = V2_DIR / "module4"
    if not v2_dir.exists():
        return result

    v2_phylo = safe_load_json(v2_dir / "phylogeny.json") or safe_load_json(v2_dir / "grammar_phylogeny.json")
    v2_transfer = safe_read_parquet(v2_dir / "transfer_matrix.parquet")

    if v2_transfer is None and v2_phylo is None:
        return result

    result["status"] = "complete"
    v2_info = {}

    if v2_phylo is not None:
        v2_info["species_list"] = v2_phylo.get("species_list", [])

    if v2_transfer is not None:
        v2_info["n_entries"] = int(len(v2_transfer))
        transfer_dict = {}
        for _, row in v2_transfer.iterrows():
            key = f"{row['source']}_to_{row['target']}"
            r2_col = "transfer_r2" if "transfer_r2" in v2_transfer.columns else "transfer_corr"
            transfer_dict[key] = safe_float(row.get(r2_col, 0))
        v2_info["transfer_r2"] = transfer_dict

        within = v2_transfer[v2_transfer["source"] == v2_transfer["target"]]
        cross = v2_transfer[v2_transfer["source"] != v2_transfer["target"]]
        r2_col = "transfer_r2" if "transfer_r2" in v2_transfer.columns else "transfer_corr"
        v2_info["mean_within_r2"] = safe_float(within[r2_col].mean()) if len(within) > 0 else None
        v2_info["mean_cross_r2"] = safe_float(cross[r2_col].mean()) if len(cross) > 0 else None

    result["v2"] = v2_info

    # --- Comparison ---
    if result["v1"] is not None and result["v2"] is not None:
        result["comparison"] = {
            "v1_mean_within_r2": result["v1"].get("mean_within_r2"),
            "v2_mean_within_r2": result["v2"].get("mean_within_r2"),
            "v1_mean_cross_r2": result["v1"].get("mean_cross_r2"),
            "v2_mean_cross_r2": result["v2"].get("mean_cross_r2"),
            "v1_approach": "R2_based",
            "v2_approach": "R2_based (distributional_transfer in phase3)",
        }

    return result


# ---------------------------------------------------------------------------
# Module 5: Biophysics Decomposition
# ---------------------------------------------------------------------------

def collect_module5():
    """Collect v1 and v2 Module 5 (Biophysics) results."""
    result = {
        "status": "not_found",
        "v1": None,
        "v2": None,
        "comparison": None,
    }

    # --- v1 ---
    v1_biophysics = {}
    for ds in DATASETS:
        bio = safe_load_json(V1_DIR / "module5" / f"{ds}_biophysics.json")
        if bio is not None:
            v1_biophysics[ds] = {
                "biophysics_r2": safe_float(bio.get("biophysics_r2")),
                "biophysics_r2_std": safe_float(bio.get("biophysics_r2_std")),
                "n_samples": bio.get("n_samples"),
                "n_features": bio.get("n_features"),
                "top_feature": max(bio.get("feature_importances", {}),
                                   key=lambda k: bio["feature_importances"][k],
                                   default=None) if bio.get("feature_importances") else None,
            }
    v1_structure = safe_load_json(V1_DIR / "module5" / "structure_predicts_grammar.json")
    if v1_biophysics:
        result["v1"] = {
            "per_dataset": v1_biophysics,
            "mean_biophysics_r2": safe_float(
                np.mean([v["biophysics_r2"] for v in v1_biophysics.values()
                         if v["biophysics_r2"] is not None])
            ),
        }
        if v1_structure:
            result["v1"]["structure_predicts_grammar"] = {
                "accuracy": safe_float(v1_structure.get("accuracy")),
                "baseline_accuracy": safe_float(v1_structure.get("baseline_accuracy")),
                "improvement": safe_float(v1_structure.get("improvement")),
            }

    # --- v2 ---
    v2_dir = V2_DIR / "module5"
    if not v2_dir.exists():
        return result

    v2_biophysics = {}
    for ds in DATASETS:
        bio = safe_load_json(v2_dir / f"{ds}_biophysics.json")
        if bio is None:
            bio = safe_load_json(v2_dir / f"biophysics_{ds}.json")
        if bio is not None:
            v2_biophysics[ds] = {
                "biophysics_r2": safe_float(bio.get("biophysics_r2")),
                "biophysics_r2_std": safe_float(bio.get("biophysics_r2_std")),
                "n_samples": bio.get("n_samples"),
                "top_feature": max(bio.get("feature_importances", {}),
                                   key=lambda k: bio["feature_importances"][k],
                                   default=None) if bio.get("feature_importances") else None,
            }

    # Also check for a combined biophysics.json
    combined_bio = safe_load_json(v2_dir / "biophysics.json")
    if combined_bio is not None and not v2_biophysics:
        # Combined file might contain per-dataset results
        if isinstance(combined_bio, dict):
            for ds in DATASETS:
                if ds in combined_bio:
                    v2_biophysics[ds] = {
                        "biophysics_r2": safe_float(combined_bio[ds].get("biophysics_r2")),
                    }

    if not v2_biophysics and combined_bio is None:
        return result

    result["status"] = "complete"
    result["v2"] = {
        "per_dataset": v2_biophysics,
    }
    if v2_biophysics:
        r2_vals = [v["biophysics_r2"] for v in v2_biophysics.values()
                   if v.get("biophysics_r2") is not None]
        result["v2"]["mean_biophysics_r2"] = safe_float(np.mean(r2_vals)) if r2_vals else None

    # Check for phase diagrams
    phase_diagrams = list(v2_dir.glob("*_phase_diagram.json"))
    result["v2"]["n_phase_diagrams"] = len(phase_diagrams)

    # --- Comparison ---
    if result["v1"] is not None and v2_biophysics:
        comp = {}
        for ds in DATASETS:
            v1_ds = result["v1"]["per_dataset"].get(ds, {})
            v2_ds = v2_biophysics.get(ds, {})
            if v1_ds.get("biophysics_r2") is not None and v2_ds.get("biophysics_r2") is not None:
                comp[ds] = {
                    "v1_r2": v1_ds["biophysics_r2"],
                    "v2_r2": v2_ds["biophysics_r2"],
                    "delta": safe_float(v2_ds["biophysics_r2"] - v1_ds["biophysics_r2"]),
                }
        if comp:
            result["comparison"] = comp

    return result


# ---------------------------------------------------------------------------
# Module 6: Grammar Completeness
# ---------------------------------------------------------------------------

def collect_module6():
    """Collect v1 and v2 Module 6 (Completeness) results."""
    result = {
        "status": "not_found",
        "v1": None,
        "v2": None,
        "comparison": None,
    }

    # --- v1 ---
    v1_completeness = {}
    for ds in DATASETS:
        comp = safe_load_json(V1_DIR / "module6" / f"{ds}_completeness.json")
        if comp is not None:
            v1_completeness[ds] = {
                "vocabulary_r2": safe_float(comp.get("vocabulary_r2")),
                "vocab_plus_grammar_r2": safe_float(
                    comp.get("vocab_plus_full_grammar_r2",
                             comp.get("vocab_plus_simple_grammar_r2"))
                ),
                "full_model_r2": safe_float(comp.get("full_model_r2")),
                "replicate_r2": safe_float(comp.get("replicate_r2")),
                "grammar_contribution": safe_float(comp.get("grammar_contribution")),
                "grammar_gap": safe_float(comp.get("grammar_gap")),
                "grammar_completeness": safe_float(comp.get("grammar_completeness")),
                "n_samples": comp.get("n_samples"),
            }

    if v1_completeness:
        result["v1"] = {
            "per_dataset": v1_completeness,
            "mean_grammar_completeness": safe_float(
                np.mean([v["grammar_completeness"] for v in v1_completeness.values()
                         if v["grammar_completeness"] is not None])
            ),
            "mean_grammar_contribution": safe_float(
                np.mean([v["grammar_contribution"] for v in v1_completeness.values()
                         if v["grammar_contribution"] is not None])
            ),
        }

    # --- v2 ---
    v2_dir = V2_DIR / "module6"
    if not v2_dir.exists():
        return result

    v2_completeness = {}
    for ds in DATASETS:
        comp = safe_load_json(v2_dir / f"{ds}_completeness.json")
        if comp is None:
            comp = safe_load_json(v2_dir / f"completeness_{ds}.json")
        if comp is not None:
            v2_completeness[ds] = {
                "vocabulary_r2": safe_float(comp.get("vocabulary_r2")),
                "vocab_plus_grammar_r2": safe_float(
                    comp.get("vocab_plus_full_grammar_r2",
                             comp.get("vocab_plus_simple_grammar_r2"))
                ),
                "full_model_r2": safe_float(comp.get("full_model_r2")),
                "replicate_r2": safe_float(comp.get("replicate_r2")),
                "grammar_contribution": safe_float(comp.get("grammar_contribution")),
                "grammar_gap": safe_float(comp.get("grammar_gap")),
                "grammar_completeness": safe_float(comp.get("grammar_completeness")),
                "n_samples": comp.get("n_samples"),
            }

    # Also check combined completeness.json
    combined = safe_load_json(v2_dir / "completeness.json")
    if combined is not None and not v2_completeness:
        if isinstance(combined, dict):
            for ds in DATASETS:
                if ds in combined:
                    v2_completeness[ds] = {
                        "grammar_completeness": safe_float(combined[ds].get("grammar_completeness")),
                        "grammar_contribution": safe_float(combined[ds].get("grammar_contribution")),
                    }

    if not v2_completeness:
        return result

    result["status"] = "complete"
    result["v2"] = {
        "per_dataset": v2_completeness,
        "mean_grammar_completeness": safe_float(
            np.mean([v["grammar_completeness"] for v in v2_completeness.values()
                     if v.get("grammar_completeness") is not None])
        ),
        "mean_grammar_contribution": safe_float(
            np.mean([v["grammar_contribution"] for v in v2_completeness.values()
                     if v.get("grammar_contribution") is not None])
        ),
    }

    # --- Comparison ---
    if result["v1"] is not None and result["v2"] is not None:
        comp = {}
        for ds in DATASETS:
            v1_ds = result["v1"]["per_dataset"].get(ds, {})
            v2_ds = v2_completeness.get(ds, {})
            if v1_ds and v2_ds:
                v1_gc = v1_ds.get("grammar_completeness")
                v2_gc = v2_ds.get("grammar_completeness")
                if v1_gc is not None and v2_gc is not None:
                    comp[ds] = {
                        "v1_completeness": v1_gc,
                        "v2_completeness": v2_gc,
                        "delta": safe_float(v2_gc - v1_gc),
                        "v1_grammar_contribution": v1_ds.get("grammar_contribution"),
                        "v2_grammar_contribution": v2_ds.get("grammar_contribution"),
                    }
        comp["v1_mean_completeness"] = result["v1"].get("mean_grammar_completeness")
        comp["v2_mean_completeness"] = result["v2"].get("mean_grammar_completeness")
        result["comparison"] = comp

    return result


# ---------------------------------------------------------------------------
# Phase 2: New Experiments
# ---------------------------------------------------------------------------

def collect_experiment_b():
    """Experiment B: Synthetic Grammar Probes."""
    exp_dir = V2_DIR / "experiments" / "experiment_b"
    if not exp_dir.exists():
        # Also check the alternate path used by the pipeline
        exp_dir = V2_DIR / "experiment_b"
    if not exp_dir.exists():
        return {"status": "not_found"}

    summaries = {}
    parquets = {}
    for f in sorted(exp_dir.glob("*_summary.json")):
        key = f.stem.replace("_summary", "")
        summaries[key] = safe_load_json(f)
    for f in sorted(exp_dir.glob("*_synthetic.parquet")):
        key = f.stem.replace("_synthetic", "")
        df = safe_read_parquet(f)
        if df is not None:
            parquets[key] = int(len(df))

    if not summaries and not parquets:
        return {"status": "not_found"}

    result = {
        "status": "complete",
        "n_summary_files": len(summaries),
        "n_parquet_files": len(parquets),
        "summaries": summaries,
        "parquet_sizes": parquets,
    }

    # Aggregate key metrics across all summaries
    all_confirm_rates = []
    for key, s in summaries.items():
        if s and "confirmation_rate" in s:
            all_confirm_rates.append(s["confirmation_rate"])
        elif s and "frac_confirmed" in s:
            all_confirm_rates.append(s["frac_confirmed"])
    if all_confirm_rates:
        result["mean_confirmation_rate"] = safe_float(np.mean(all_confirm_rates))

    return result


def collect_experiment_e():
    """Experiment E: Information Theory Decomposition."""
    exp_dir = V2_DIR / "experiments" / "experiment_e"
    if not exp_dir.exists():
        exp_dir = V2_DIR / "experiment_e"
    if not exp_dir.exists():
        return {"status": "not_found"}

    info_files = {}
    for ds in DATASETS:
        info = safe_load_json(exp_dir / f"{ds}_information.json")
        if info is not None:
            info_files[ds] = {
                "r2_vocabulary_gb": safe_float(info.get("r2_vocabulary_gb")),
                "grammar_information": safe_float(info.get("grammar_information")),
                "r2_total": safe_float(info.get("r2_total")),
                "redundancy": safe_float(info.get("redundancy")),
                "synergy": safe_float(info.get("synergy")),
            }

    if not info_files:
        return {"status": "not_found"}

    result = {
        "status": "complete",
        "per_dataset": info_files,
        "n_datasets": len(info_files),
    }

    # Aggregate
    grammar_infos = [v["grammar_information"] for v in info_files.values()
                     if v.get("grammar_information") is not None]
    if grammar_infos:
        result["mean_grammar_information"] = safe_float(np.mean(grammar_infos))

    vocab_r2s = [v["r2_vocabulary_gb"] for v in info_files.values()
                 if v.get("r2_vocabulary_gb") is not None]
    if vocab_r2s:
        result["mean_vocab_r2"] = safe_float(np.mean(vocab_r2s))

    return result


def collect_experiment_f():
    """Experiment F: Grammar Heterogeneity."""
    exp_dir = V2_DIR / "experiments" / "experiment_f"
    if not exp_dir.exists():
        exp_dir = V2_DIR / "experiment_f"
    if not exp_dir.exists():
        return {"status": "not_found"}

    het_files = {}
    for ds in DATASETS:
        het = safe_load_json(exp_dir / f"{ds}_heterogeneity.json")
        if het is not None and "error" not in het:
            het_files[ds] = {
                "n_grammar_rich": het.get("n_grammar_rich"),
                "n_grammar_poor": het.get("n_grammar_poor"),
                "predictor_r2_cv": safe_float(het.get("predictor_r2_cv")),
                "best_predictor": het.get("best_predictor"),
                "grammar_rich_fraction": safe_float(het.get("grammar_rich_fraction")),
            }

    if not het_files:
        return {"status": "not_found"}

    result = {
        "status": "complete",
        "per_dataset": het_files,
        "n_datasets": len(het_files),
    }

    r2_vals = [v["predictor_r2_cv"] for v in het_files.values()
               if v.get("predictor_r2_cv") is not None]
    if r2_vals:
        result["mean_predictor_r2"] = safe_float(np.mean(r2_vals))

    return result


def collect_experiment_g():
    """Experiment G: Grammar Potential (Counterfactual)."""
    exp_dir = V2_DIR / "experiments" / "experiment_g"
    if not exp_dir.exists():
        exp_dir = V2_DIR / "experiment_g"
    if not exp_dir.exists():
        return {"status": "not_found"}

    potential_files = {}
    for ds in DATASETS:
        summary = safe_load_json(exp_dir / f"{ds}_potential_summary.json")
        if summary is not None and "error" not in summary:
            potential_files[ds] = {
                "mean_potential": safe_float(summary.get("mean_potential")),
                "mean_utilization": safe_float(summary.get("mean_utilization")),
                "n_enhancers": summary.get("n_enhancers"),
                "frac_suboptimal": safe_float(summary.get("frac_suboptimal")),
            }

    if not potential_files:
        return {"status": "not_found"}

    result = {
        "status": "complete",
        "per_dataset": potential_files,
        "n_datasets": len(potential_files),
    }

    potentials = [v["mean_potential"] for v in potential_files.values()
                  if v.get("mean_potential") is not None]
    if potentials:
        result["mean_potential"] = safe_float(np.mean(potentials))

    utilizations = [v["mean_utilization"] for v in potential_files.values()
                    if v.get("mean_utilization") is not None]
    if utilizations:
        result["mean_utilization"] = safe_float(np.mean(utilizations))

    return result


# ---------------------------------------------------------------------------
# Phase 3: Redesigned Tests
# ---------------------------------------------------------------------------

def collect_phase3_compositionality_v2():
    """Phase 3: Redesigned Compositionality (enhancer-specific)."""
    p3_dir = V2_DIR / "phase3" / "compositionality_v2"
    if not p3_dir.exists():
        p3_dir = V2_DIR / "compositionality_v2"
    if not p3_dir.exists():
        return {"status": "not_found"}

    summary = safe_load_json(p3_dir / "compositionality_v2_summary.json")
    results_df = safe_read_parquet(p3_dir / "compositionality_v2_results.parquet")

    if summary is None and results_df is None:
        return {"status": "not_found"}

    result = {"status": "complete"}
    if summary is not None:
        result["summary"] = summary
    if results_df is not None:
        result["n_tests"] = int(len(results_df))
        if "compositionality" in results_df.columns:
            result["mean_compositionality"] = safe_float(results_df["compositionality"].mean())
        if "dataset" in results_df.columns:
            per_ds = {}
            for ds in results_df["dataset"].unique():
                ds_data = results_df[results_df["dataset"] == ds]
                per_ds[ds] = {"n_tests": int(len(ds_data))}
                if "compositionality" in ds_data.columns:
                    per_ds[ds]["mean_compositionality"] = safe_float(ds_data["compositionality"].mean())
            result["per_dataset"] = per_ds

    return result


def collect_phase3_distributional_transfer():
    """Phase 3: Distributional Transfer."""
    p3_dir = V2_DIR / "phase3" / "distributional_transfer"
    if not p3_dir.exists():
        p3_dir = V2_DIR / "distributional_transfer"
    if not p3_dir.exists():
        return {"status": "not_found"}

    dt = safe_load_json(p3_dir / "distributional_transfer.json")
    if dt is None:
        return {"status": "not_found"}

    result = {
        "status": "complete",
        "data": dt,
    }
    if "grammar_properties_conserved" in dt:
        result["properties_conserved"] = safe_float(dt["grammar_properties_conserved"])

    return result


def collect_phase3_attention():
    """Phase 3: Attention Analysis."""
    p3_dir = V2_DIR / "phase3" / "attention"
    if not p3_dir.exists():
        p3_dir = V2_DIR / "attention"
    if not p3_dir.exists():
        return {"status": "not_found"}

    grammar_heads_files = sorted(p3_dir.glob("*_grammar_heads.json"))
    attention_parquets = sorted(p3_dir.glob("*_attention.parquet"))

    if not grammar_heads_files and not attention_parquets:
        return {"status": "not_found"}

    result = {
        "status": "complete",
        "n_grammar_head_files": len(grammar_heads_files),
        "n_attention_parquets": len(attention_parquets),
        "per_analysis": {},
    }

    for f in grammar_heads_files:
        key = f.stem.replace("_grammar_heads", "")
        data = safe_load_json(f)
        if data is not None:
            result["per_analysis"][key] = {
                "n_grammar_heads": data.get("n_grammar_heads"),
                "total_heads": data.get("total_heads"),
            }

    # Aggregate
    total_grammar_heads = sum(
        v.get("n_grammar_heads", 0) for v in result["per_analysis"].values()
        if v.get("n_grammar_heads") is not None
    )
    result["total_grammar_heads"] = total_grammar_heads

    return result


def collect_phase3_anova():
    """Phase 3: ANOVA Decomposition."""
    p3_dir = V2_DIR / "phase3" / "anova"
    if not p3_dir.exists():
        p3_dir = V2_DIR / "anova"
    if not p3_dir.exists():
        return {"status": "not_found"}

    anova_files = {}
    power_files = {}
    for ds in DATASETS:
        anova = safe_load_json(p3_dir / f"{ds}_anova.json")
        if anova is not None and "error" not in anova:
            anova_files[ds] = {
                "eta2_vocabulary": safe_float(anova.get("eta2_vocabulary")),
                "grammar_total_eta2": safe_float(anova.get("grammar_total_eta2")),
                "n_sequences": anova.get("n_sequences"),
            }
        power = safe_load_json(p3_dir / f"{ds}_grammar_eta2_power.json")
        if power is not None:
            power_files[ds] = power

    if not anova_files:
        return {"status": "not_found"}

    result = {
        "status": "complete",
        "per_dataset": anova_files,
        "n_datasets": len(anova_files),
    }

    # Aggregates
    vocab_eta2 = [v["eta2_vocabulary"] for v in anova_files.values()
                  if v.get("eta2_vocabulary") is not None]
    grammar_eta2 = [v["grammar_total_eta2"] for v in anova_files.values()
                    if v.get("grammar_total_eta2") is not None]
    if vocab_eta2:
        result["mean_vocab_eta2"] = safe_float(np.mean(vocab_eta2))
    if grammar_eta2:
        result["mean_grammar_eta2"] = safe_float(np.mean(grammar_eta2))

    if power_files:
        result["power_analyses"] = power_files

    return result


# ---------------------------------------------------------------------------
# Pipeline Completion Status
# ---------------------------------------------------------------------------

def check_completion_status():
    """Check which result directories exist and have content."""
    status = OrderedDict()

    checks = [
        ("module1", V2_DIR / "module1", ["*_gsi.parquet"]),
        ("module2", V2_DIR / "module2", ["grammar_rules_database.parquet", "global_consensus.json"]),
        ("module3", V2_DIR / "module3", ["compositionality_results.parquet", "complexity_classification.json"]),
        ("module4", V2_DIR / "module4", ["transfer_matrix.parquet"]),
        ("module5", V2_DIR / "module5", ["*_biophysics.json"]),
        ("module6", V2_DIR / "module6", ["*_completeness.json"]),
        ("experiment_b", [V2_DIR / "experiments" / "experiment_b", V2_DIR / "experiment_b"],
         ["*_summary.json", "*_synthetic.parquet"]),
        ("experiment_e", [V2_DIR / "experiments" / "experiment_e", V2_DIR / "experiment_e"],
         ["*_information.json"]),
        ("experiment_f", [V2_DIR / "experiments" / "experiment_f", V2_DIR / "experiment_f"],
         ["*_heterogeneity.json"]),
        ("experiment_g", [V2_DIR / "experiments" / "experiment_g", V2_DIR / "experiment_g"],
         ["*_potential_summary.json"]),
        ("compositionality_v2", [V2_DIR / "phase3" / "compositionality_v2", V2_DIR / "compositionality_v2"],
         ["compositionality_v2_results.parquet"]),
        ("distributional_transfer", [V2_DIR / "phase3" / "distributional_transfer", V2_DIR / "distributional_transfer"],
         ["distributional_transfer.json"]),
        ("attention", [V2_DIR / "phase3" / "attention", V2_DIR / "attention"],
         ["*_grammar_heads.json"]),
        ("anova", [V2_DIR / "phase3" / "anova", V2_DIR / "anova"],
         ["*_anova.json"]),
    ]

    for item in checks:
        name = item[0]
        dirs = item[1] if isinstance(item[1], list) else [item[1]]
        patterns = item[2]

        found_dir = None
        for d in dirs:
            if d.exists():
                found_dir = d
                break

        if found_dir is None:
            status[name] = {"exists": False, "n_files": 0, "status": "MISSING"}
            continue

        n_files = 0
        for pattern in patterns:
            n_files += len(list(found_dir.glob(pattern)))

        if n_files > 0:
            status[name] = {"exists": True, "n_files": n_files, "status": "COMPLETE",
                            "path": str(found_dir)}
        else:
            status[name] = {"exists": True, "n_files": 0, "status": "EMPTY",
                            "path": str(found_dir)}

    return status


# ---------------------------------------------------------------------------
# Report Printer
# ---------------------------------------------------------------------------

def print_report(summary):
    """Print a formatted report to stdout."""
    section_header("GRAMLANG v2 PIPELINE RESULTS REPORT")
    print(f"  Generated: {summary['metadata']['timestamp']}")
    print(f"  Project:   {summary['metadata']['project_dir']}")

    # --- Completion Status ---
    section_header("1. PIPELINE COMPLETION STATUS")
    completion = summary["completion_status"]
    n_complete = sum(1 for v in completion.values() if v["status"] == "COMPLETE")
    n_total = len(completion)
    print(f"  Overall: {n_complete}/{n_total} components have results")
    print()
    print(f"  {'Component':<28s} {'Status':<10s} {'Files':<8s} {'Path'}")
    print(f"  {'-'*28} {'-'*10} {'-'*8} {'-'*40}")
    for name, info in completion.items():
        status_str = info["status"]
        if status_str == "COMPLETE":
            marker = "[OK]"
        elif status_str == "EMPTY":
            marker = "[--]"
        else:
            marker = "[  ]"
        path_str = info.get("path", "")
        print(f"  {name:<28s} {marker:<10s} {info['n_files']:<8d} {path_str}")

    # --- Module 1 ---
    m1 = summary.get("module1", {})
    section_header("2. MODULE 1: GSI CENSUS")
    if m1.get("status") == "not_found":
        print("  [NOT AVAILABLE]")
    else:
        v2 = m1.get("v2", {})
        print(f"  v2 Total measurements:  {v2.get('n_total', 'N/A')}")
        print(f"  v2 Parquet files:       {v2.get('n_files', 'N/A')}")
        print(f"  v2 Overall mean GSI:    {fmt_float(v2.get('overall_mean_gsi'))}")
        print(f"  v2 Overall median GSI:  {fmt_float(v2.get('overall_median_gsi'))}")

        if v2.get("corrected_frac_significant") is not None:
            print(f"  v2 Corrected frac sig:  {fmt_pct(v2['corrected_frac_significant'])}")
        if v2.get("median_frac_significant_per_combo") is not None:
            print(f"  v2 Median frac sig:     {fmt_pct(v2['median_frac_significant_per_combo'])}")

        ext = v2.get("extended_analysis", {})
        if ext:
            print(f"  Best model (mean GSI):  {ext.get('best_model_mean_gsi', 'N/A')}")
            print(f"  Strongest dataset:      {ext.get('strongest_dataset_mean_gsi', 'N/A')}")
            anova = ext.get("anova", {})
            if anova:
                ds_eta2 = anova.get("C(dataset)", {}).get("eta_squared")
                model_eta2 = anova.get("C(model)", {}).get("eta_squared")
                print(f"  ANOVA dataset eta2:     {fmt_float(ds_eta2)}")
                print(f"  ANOVA model eta2:       {fmt_float(model_eta2)}")
                print(f"  Dominant factor:        {anova.get('dominant_factor', 'N/A')}")

        # Per-dataset table
        per_ds = v2.get("per_dataset", {})
        if per_ds:
            sub_header("Per-Dataset GSI (v2)")
            print(f"  {'Dataset':<16s} {'N':>6s} {'Enh':>6s} {'Mean GSI':>10s} {'Med GSI':>10s} {'Models'}")
            print(f"  {'-'*16} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*20}")
            for ds in DATASETS:
                d = per_ds.get(ds, {})
                if d:
                    models_str = ", ".join(d.get("models", []))
                    print(f"  {ds:<16s} {d.get('n', 0):>6d} {d.get('n_enhancers', 0):>6d} "
                          f"{fmt_float(d.get('mean_gsi')):>10s} {fmt_float(d.get('median_gsi')):>10s} "
                          f"{models_str}")

        # v1 vs v2 comparison
        comp = m1.get("comparison", {})
        if comp:
            sub_header("v1 vs v2 GSI Comparison")
            print(f"  {'Metric':<30s} {'v1':>12s} {'v2':>12s} {'Ratio':>10s}")
            print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
            print(f"  {'Overall mean GSI':<30s} {fmt_float(comp.get('gsi_mean_v1')):>12s} "
                  f"{fmt_float(comp.get('gsi_mean_v2')):>12s} "
                  f"{fmt_float(comp.get('gsi_mean_ratio_v2_over_v1'), 2):>10s}")
            print(f"  {'Overall median GSI':<30s} {fmt_float(comp.get('gsi_median_v1')):>12s} "
                  f"{fmt_float(comp.get('gsi_median_v2')):>12s}")
            print(f"  {'Sample size':<30s} {comp.get('sample_size_v1', 'N/A'):>12} "
                  f"{comp.get('sample_size_v2', 'N/A'):>12}")
            print(f"  {'Frac significant':<30s} "
                  f"{fmt_pct(comp.get('v1_frac_significant_uncorrected')):>12s} "
                  f"{fmt_pct(comp.get('v2_frac_significant_corrected')):>12s}")

            per_ds_comp = comp.get("per_dataset", {})
            if per_ds_comp:
                print()
                print(f"  {'Dataset':<16s} {'v1 Mean':>10s} {'v2 Mean':>10s} "
                      f"{'v1 Median':>10s} {'v2 Median':>10s} {'v1 N':>6s} {'v2 N':>6s}")
                print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6} {'-'*6}")
                for ds in DATASETS:
                    dc = per_ds_comp.get(ds, {})
                    if dc:
                        print(f"  {ds:<16s} {fmt_float(dc.get('v1_mean_gsi')):>10s} "
                              f"{fmt_float(dc.get('v2_mean_gsi')):>10s} "
                              f"{fmt_float(dc.get('v1_median_gsi')):>10s} "
                              f"{fmt_float(dc.get('v2_median_gsi')):>10s} "
                              f"{dc.get('v1_n', 'N/A'):>6} {dc.get('v2_n', 'N/A'):>6}")

    # --- Module 2 ---
    m2 = summary.get("module2", {})
    section_header("3. MODULE 2: GRAMMAR RULES")
    if m2.get("status") == "not_found":
        print("  [NOT AVAILABLE]")
    else:
        for version, label in [("v1", "v1"), ("v2", "v2")]:
            data = m2.get(version, {})
            if data:
                sub_header(f"{label} Rules Summary")
                print(f"  Total rules:              {data.get('n_rules_total', 'N/A')}")
                print(f"  Unique pairs:             {data.get('n_unique_pairs', 'N/A')}")
                print(f"  Mean consensus:           {fmt_float(data.get('mean_consensus'))}")
                print(f"  Orientation agreement:    {fmt_pct(data.get('mean_orientation_agreement'))}")
                print(f"  Spacing correlation:      {fmt_float(data.get('mean_spacing_correlation'))}")
                print(f"  High consensus fraction:  {fmt_pct(data.get('frac_high_consensus'))}")
                print(f"  Mean fold change:         {fmt_float(data.get('mean_fold_change'))}")
                print(f"  +/+ orientation bias:     {fmt_pct(data.get('plus_plus_fraction'))}")

        comp = m2.get("comparison", {})
        if comp:
            sub_header("v1 vs v2 Rules Comparison")
            print(f"  {'Metric':<30s} {'v1':>12s} {'v2':>12s} {'Ratio':>10s}")
            print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
            for key, label in [
                ("n_rules_total", "Total rules"),
                ("mean_consensus", "Mean consensus"),
                ("mean_orientation_agreement", "Orientation agreement"),
                ("mean_spacing_correlation", "Spacing correlation"),
                ("frac_high_consensus", "High consensus frac"),
                ("mean_fold_change", "Mean fold change"),
                ("plus_plus_fraction", "+/+ bias"),
            ]:
                entry = comp.get(key, {})
                if entry:
                    ratio_str = fmt_float(entry.get("ratio_v2_over_v1"), 2) if entry.get("ratio_v2_over_v1") else "N/A"
                    print(f"  {label:<30s} {fmt_float(entry.get('v1')):>12s} "
                          f"{fmt_float(entry.get('v2')):>12s} {ratio_str:>10s}")

    # --- Module 3 ---
    m3 = summary.get("module3", {})
    section_header("4. MODULE 3: COMPOSITIONALITY")
    if m3.get("status") == "not_found":
        print("  [NOT AVAILABLE]")
    else:
        for version, label in [("v1", "v1"), ("v2", "v2")]:
            data = m3.get(version, {})
            if data:
                sub_header(f"{label} Compositionality")
                print(f"  Classification:           {data.get('classification', 'N/A')}")
                print(f"  Mean gap:                 {fmt_float(data.get('mean_gap'))}")
                print(f"  Confidence:               {fmt_float(data.get('confidence'))}")
                print(f"  Best BIC model:           {data.get('best_bic_model', 'N/A')}")
                print(f"  N tests:                  {data.get('n_tests', 'N/A')}")
                per_k = data.get("per_k", {})
                if per_k:
                    print(f"  {'k':<6s} {'N tests':>8s} {'Mean gap':>10s}")
                    for k in sorted(per_k.keys(), key=int):
                        kd = per_k[k]
                        print(f"  {str(k):<6s} {kd.get('n_tests', 0):>8d} "
                              f"{fmt_float(kd.get('mean_gap')):>10s}")

        comp = m3.get("comparison", {})
        if comp:
            sub_header("v1 vs v2 Compositionality Comparison")
            print(f"  Classification v1:  {comp.get('classification_v1', 'N/A')}")
            print(f"  Classification v2:  {comp.get('classification_v2', 'N/A')}")
            print(f"  Classification changed: {comp.get('classification_changed', 'N/A')}")
            print(f"  Mean gap v1:        {fmt_float(comp.get('mean_gap_v1'))}")
            print(f"  Mean gap v2:        {fmt_float(comp.get('mean_gap_v2'))}")

    # --- Module 4 ---
    m4 = summary.get("module4", {})
    section_header("5. MODULE 4: CROSS-SPECIES TRANSFER")
    if m4.get("status") == "not_found":
        print("  [NOT AVAILABLE]")
    else:
        for version, label in [("v1", "v1"), ("v2", "v2")]:
            data = m4.get(version, {})
            if data:
                sub_header(f"{label} Transfer")
                print(f"  Species:                  {data.get('species_list', 'N/A')}")
                print(f"  Mean within-species R2:   {fmt_float(data.get('mean_within_r2'))}")
                print(f"  Mean cross-species R2:    {fmt_float(data.get('mean_cross_r2'))}")
                tr = data.get("transfer_r2", {})
                if tr:
                    for key, val in sorted(tr.items()):
                        print(f"    {key:<25s}  R2 = {fmt_float(val)}")

    # --- Module 5 ---
    m5 = summary.get("module5", {})
    section_header("6. MODULE 5: BIOPHYSICS DECOMPOSITION")
    if m5.get("status") == "not_found":
        print("  [NOT AVAILABLE]")
    else:
        for version, label in [("v1", "v1"), ("v2", "v2")]:
            data = m5.get(version, {})
            if data:
                sub_header(f"{label} Biophysics")
                print(f"  Mean biophysics R2:       {fmt_float(data.get('mean_biophysics_r2'))}")
                per_ds = data.get("per_dataset", {})
                if per_ds:
                    print(f"  {'Dataset':<16s} {'R2':>8s} {'Top Feature':<20s}")
                    print(f"  {'-'*16} {'-'*8} {'-'*20}")
                    for ds in DATASETS:
                        d = per_ds.get(ds, {})
                        if d:
                            print(f"  {ds:<16s} {fmt_float(d.get('biophysics_r2')):>8s} "
                                  f"{d.get('top_feature', 'N/A'):<20s}")

        comp = m5.get("comparison", {})
        if comp:
            sub_header("v1 vs v2 Biophysics Comparison")
            print(f"  {'Dataset':<16s} {'v1 R2':>10s} {'v2 R2':>10s} {'Delta':>10s}")
            print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")
            for ds in DATASETS:
                dc = comp.get(ds, {})
                if dc:
                    print(f"  {ds:<16s} {fmt_float(dc.get('v1_r2')):>10s} "
                          f"{fmt_float(dc.get('v2_r2')):>10s} "
                          f"{fmt_float(dc.get('delta')):>10s}")

    # --- Module 6 ---
    m6 = summary.get("module6", {})
    section_header("7. MODULE 6: GRAMMAR COMPLETENESS")
    if m6.get("status") == "not_found":
        print("  [NOT AVAILABLE]")
    else:
        for version, label in [("v1", "v1"), ("v2", "v2")]:
            data = m6.get(version, {})
            if data:
                sub_header(f"{label} Completeness")
                print(f"  Mean completeness:        {fmt_pct(data.get('mean_grammar_completeness'))}")
                print(f"  Mean grammar contrib:     {fmt_float(data.get('mean_grammar_contribution'))}")
                per_ds = data.get("per_dataset", {})
                if per_ds:
                    print(f"  {'Dataset':<16s} {'Vocab R2':>10s} {'+Gram R2':>10s} "
                          f"{'Contrib':>10s} {'Compl':>10s}")
                    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
                    for ds in DATASETS:
                        d = per_ds.get(ds, {})
                        if d:
                            print(f"  {ds:<16s} {fmt_float(d.get('vocabulary_r2')):>10s} "
                                  f"{fmt_float(d.get('vocab_plus_grammar_r2')):>10s} "
                                  f"{fmt_float(d.get('grammar_contribution')):>10s} "
                                  f"{fmt_pct(d.get('grammar_completeness')):>10s}")

        comp = m6.get("comparison", {})
        if comp:
            sub_header("v1 vs v2 Completeness Comparison")
            print(f"  v1 Mean completeness:  {fmt_pct(comp.get('v1_mean_completeness'))}")
            print(f"  v2 Mean completeness:  {fmt_pct(comp.get('v2_mean_completeness'))}")
            for ds in DATASETS:
                dc = comp.get(ds, {})
                if dc:
                    print(f"  {ds:<16s}  v1={fmt_pct(dc.get('v1_completeness')):>8s}  "
                          f"v2={fmt_pct(dc.get('v2_completeness')):>8s}  "
                          f"delta={fmt_float(dc.get('delta')):>8s}")

    # --- Phase 2 Experiments ---
    section_header("8. PHASE 2: NEW EXPERIMENTS")

    for exp_key, exp_name in [
        ("experiment_b", "Experiment B: Synthetic Grammar Probes"),
        ("experiment_e", "Experiment E: Information Theory Decomposition"),
        ("experiment_f", "Experiment F: Grammar Heterogeneity"),
        ("experiment_g", "Experiment G: Grammar Potential"),
    ]:
        exp = summary.get(exp_key, {})
        sub_header(exp_name)
        if exp.get("status") == "not_found":
            print("  [NOT AVAILABLE]")
        else:
            if exp_key == "experiment_b":
                print(f"  Summary files:            {exp.get('n_summary_files', 0)}")
                print(f"  Parquet files:            {exp.get('n_parquet_files', 0)}")
                if exp.get("mean_confirmation_rate") is not None:
                    print(f"  Mean confirmation rate:   {fmt_pct(exp['mean_confirmation_rate'])}")
            elif exp_key == "experiment_e":
                print(f"  Datasets analyzed:        {exp.get('n_datasets', 0)}")
                if exp.get("mean_grammar_information") is not None:
                    print(f"  Mean grammar info:        {fmt_float(exp['mean_grammar_information'])}")
                if exp.get("mean_vocab_r2") is not None:
                    print(f"  Mean vocab R2:            {fmt_float(exp['mean_vocab_r2'])}")
                per_ds = exp.get("per_dataset", {})
                if per_ds:
                    print(f"  {'Dataset':<16s} {'Vocab R2':>10s} {'Gram Info':>10s}")
                    for ds in DATASETS:
                        d = per_ds.get(ds, {})
                        if d:
                            print(f"  {ds:<16s} {fmt_float(d.get('r2_vocabulary_gb')):>10s} "
                                  f"{fmt_float(d.get('grammar_information')):>10s}")
            elif exp_key == "experiment_f":
                print(f"  Datasets analyzed:        {exp.get('n_datasets', 0)}")
                if exp.get("mean_predictor_r2") is not None:
                    print(f"  Mean predictor R2 (CV):   {fmt_float(exp['mean_predictor_r2'])}")
                per_ds = exp.get("per_dataset", {})
                if per_ds:
                    print(f"  {'Dataset':<16s} {'Rich':>6s} {'Poor':>6s} {'Pred R2':>10s}")
                    for ds in DATASETS:
                        d = per_ds.get(ds, {})
                        if d:
                            print(f"  {ds:<16s} {d.get('n_grammar_rich', 'N/A'):>6} "
                                  f"{d.get('n_grammar_poor', 'N/A'):>6} "
                                  f"{fmt_float(d.get('predictor_r2_cv')):>10s}")
            elif exp_key == "experiment_g":
                print(f"  Datasets analyzed:        {exp.get('n_datasets', 0)}")
                if exp.get("mean_potential") is not None:
                    print(f"  Mean potential:           {fmt_float(exp['mean_potential'])}")
                if exp.get("mean_utilization") is not None:
                    print(f"  Mean utilization:         {fmt_float(exp['mean_utilization'])}")
                per_ds = exp.get("per_dataset", {})
                if per_ds:
                    print(f"  {'Dataset':<16s} {'Potential':>10s} {'Utiliz':>10s}")
                    for ds in DATASETS:
                        d = per_ds.get(ds, {})
                        if d:
                            print(f"  {ds:<16s} {fmt_float(d.get('mean_potential')):>10s} "
                                  f"{fmt_float(d.get('mean_utilization')):>10s}")

    # --- Phase 3 Redesigned Tests ---
    section_header("9. PHASE 3: REDESIGNED TESTS")

    for p3_key, p3_name in [
        ("compositionality_v2", "Compositionality v2 (enhancer-specific)"),
        ("distributional_transfer", "Distributional Transfer"),
        ("attention", "Attention Analysis"),
        ("anova", "ANOVA Decomposition"),
    ]:
        p3 = summary.get(p3_key, {})
        sub_header(p3_name)
        if p3.get("status") == "not_found":
            print("  [NOT AVAILABLE]")
        else:
            if p3_key == "compositionality_v2":
                print(f"  N tests:                  {p3.get('n_tests', 'N/A')}")
                if p3.get("mean_compositionality") is not None:
                    print(f"  Mean compositionality:    {fmt_float(p3['mean_compositionality'])}")
                s = p3.get("summary", {})
                if s:
                    for k, v in s.items():
                        if isinstance(v, (int, float)):
                            print(f"  {k:<30s} {fmt_float(v) if isinstance(v, float) else str(v)}")
            elif p3_key == "distributional_transfer":
                if p3.get("properties_conserved") is not None:
                    print(f"  Properties conserved:     {fmt_pct(p3['properties_conserved'])}")
                data = p3.get("data", {})
                if data:
                    for k, v in data.items():
                        if isinstance(v, (int, float)):
                            print(f"  {k:<30s} {fmt_float(v) if isinstance(v, float) else str(v)}")
            elif p3_key == "attention":
                print(f"  Grammar head files:       {p3.get('n_grammar_head_files', 0)}")
                print(f"  Attention parquets:       {p3.get('n_attention_parquets', 0)}")
                print(f"  Total grammar heads:      {p3.get('total_grammar_heads', 0)}")
                for key, info in p3.get("per_analysis", {}).items():
                    print(f"    {key}: {info.get('n_grammar_heads', 0)} grammar heads "
                          f"/ {info.get('total_heads', '?')} total")
            elif p3_key == "anova":
                print(f"  Datasets analyzed:        {p3.get('n_datasets', 0)}")
                if p3.get("mean_vocab_eta2") is not None:
                    print(f"  Mean vocab eta2:          {fmt_float(p3['mean_vocab_eta2'])}")
                if p3.get("mean_grammar_eta2") is not None:
                    print(f"  Mean grammar eta2:        {fmt_float(p3['mean_grammar_eta2'])}")
                per_ds = p3.get("per_dataset", {})
                if per_ds:
                    print(f"  {'Dataset':<16s} {'Vocab eta2':>12s} {'Gram eta2':>12s}")
                    for ds in DATASETS:
                        d = per_ds.get(ds, {})
                        if d:
                            print(f"  {ds:<16s} {fmt_float(d.get('eta2_vocabulary')):>12s} "
                                  f"{fmt_float(d.get('grammar_total_eta2')):>12s}")

    # --- Key Findings Summary ---
    section_header("10. KEY FINDINGS SUMMARY")
    findings = summary.get("key_findings", {})
    if findings:
        for i, (key, finding) in enumerate(findings.items(), 1):
            print(f"  {i}. {key}: {finding}")
    else:
        print("  [No key findings generated -- most modules not yet available]")

    print()
    print("=" * 80)
    print(f"  Report saved to: {OUTPUT_JSON}")
    print("=" * 80)
    print()


# ---------------------------------------------------------------------------
# Key Findings Extraction
# ---------------------------------------------------------------------------

def extract_key_findings(summary):
    """Extract key findings from the collected data."""
    findings = OrderedDict()

    # Module 1
    m1 = summary.get("module1", {})
    if m1.get("v2"):
        v2 = m1["v2"]
        mean_gsi = v2.get("overall_mean_gsi")
        if mean_gsi is not None:
            findings["gsi_magnitude"] = (
                f"v2 mean GSI = {mean_gsi:.4f} across {v2.get('n_total', '?')} measurements"
            )
        corr_sig = v2.get("corrected_frac_significant")
        if corr_sig is not None:
            findings["significance_correction"] = (
                f"After p-value correction, {corr_sig*100:.1f}% of enhancers show significant "
                f"grammar sensitivity (vs. 100% in v1 with uncorrected F-test)"
            )
        ext = v2.get("extended_analysis", {})
        if ext.get("anova", {}).get("dominant_factor"):
            findings["variance_source"] = (
                f"GSI variance is dominated by {ext['anova']['dominant_factor']} "
                f"(eta2={ext['anova'].get('C(dataset)', {}).get('eta_squared', '?'):.3f}), "
                f"not model choice (eta2={ext['anova'].get('C(model)', {}).get('eta_squared', '?'):.3f})"
            )

    comp = m1.get("comparison", {})
    if comp.get("gsi_mean_ratio_v2_over_v1") is not None:
        ratio = comp["gsi_mean_ratio_v2_over_v1"]
        direction = "higher" if ratio > 1 else "lower"
        findings["v1_v2_gsi_shift"] = (
            f"v2 mean GSI is {ratio:.1f}x {direction} than v1 "
            f"(v1={comp.get('gsi_mean_v1', '?'):.4f}, v2={comp.get('gsi_mean_v2', '?'):.4f}), "
            f"likely due to dataset-specific probes and larger sample"
        )

    # Module 2
    m2 = summary.get("module2", {})
    if m2.get("comparison"):
        c = m2["comparison"]
        n_rules = c.get("n_rules_total", {})
        if n_rules.get("v1") and n_rules.get("v2"):
            findings["rule_count"] = (
                f"v2 extracted {n_rules['v2']:.0f} rules (v1: {n_rules['v1']:.0f})"
            )

    # Module 3
    m3 = summary.get("module3", {})
    if m3.get("comparison"):
        c = m3["comparison"]
        findings["compositionality"] = (
            f"Grammar classification: v1={c.get('classification_v1', '?')}, "
            f"v2={c.get('classification_v2', '?')} "
            f"(changed={c.get('classification_changed', '?')})"
        )

    # Module 4
    m4 = summary.get("module4", {})
    if m4.get("v2"):
        findings["transfer"] = (
            f"Within-species R2={m4['v2'].get('mean_within_r2', '?')}, "
            f"cross-species R2={m4['v2'].get('mean_cross_r2', '?')}"
        )

    # Module 6
    m6 = summary.get("module6", {})
    if m6.get("comparison"):
        c = m6["comparison"]
        findings["completeness"] = (
            f"Grammar completeness: v1={fmt_pct(c.get('v1_mean_completeness'))}, "
            f"v2={fmt_pct(c.get('v2_mean_completeness'))}"
        )

    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Collect all v2 results and produce summary report."""
    print(f"Collecting v2 pipeline results from: {V2_DIR}")
    print(f"Comparing against v1 results in:     {V1_DIR}")
    print()

    if not V2_DIR.exists():
        print(f"ERROR: v2 results directory does not exist: {V2_DIR}")
        print("Has the v2 pipeline been run?")
        sys.exit(1)

    summary = OrderedDict()
    summary["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "project_dir": str(PROJECT_DIR),
        "v1_dir": str(V1_DIR),
        "v2_dir": str(V2_DIR),
        "script": str(Path(__file__).resolve()),
    }

    # 1. Completion status
    print("Checking pipeline completion status...")
    summary["completion_status"] = check_completion_status()

    # 2. Core modules
    print("Collecting Module 1 (GSI Census)...")
    summary["module1"] = collect_module1()

    print("Collecting Module 2 (Grammar Rules)...")
    summary["module2"] = collect_module2()

    print("Collecting Module 3 (Compositionality)...")
    summary["module3"] = collect_module3()

    print("Collecting Module 4 (Cross-Species Transfer)...")
    summary["module4"] = collect_module4()

    print("Collecting Module 5 (Biophysics)...")
    summary["module5"] = collect_module5()

    print("Collecting Module 6 (Completeness)...")
    summary["module6"] = collect_module6()

    # 3. Phase 2 experiments
    print("Collecting Experiment B (Synthetic Probes)...")
    summary["experiment_b"] = collect_experiment_b()

    print("Collecting Experiment E (Information Theory)...")
    summary["experiment_e"] = collect_experiment_e()

    print("Collecting Experiment F (Grammar Heterogeneity)...")
    summary["experiment_f"] = collect_experiment_f()

    print("Collecting Experiment G (Grammar Potential)...")
    summary["experiment_g"] = collect_experiment_g()

    # 4. Phase 3 redesigned tests
    print("Collecting Phase 3: Compositionality v2...")
    summary["compositionality_v2"] = collect_phase3_compositionality_v2()

    print("Collecting Phase 3: Distributional Transfer...")
    summary["distributional_transfer"] = collect_phase3_distributional_transfer()

    print("Collecting Phase 3: Attention Analysis...")
    summary["attention"] = collect_phase3_attention()

    print("Collecting Phase 3: ANOVA Decomposition...")
    summary["anova"] = collect_phase3_anova()

    # 5. Key findings
    print("Extracting key findings...")
    summary["key_findings"] = extract_key_findings(summary)

    # 6. Write JSON
    os.makedirs(V2_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved summary to: {OUTPUT_JSON}")
    print()

    # 7. Print report
    print_report(summary)


if __name__ == "__main__":
    main()
