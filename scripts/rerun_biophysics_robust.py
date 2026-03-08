"""
Re-run Module 5 biophysics analysis using gsi_robust instead of raw GSI.

The v2 pipeline used raw GSI values which have extreme outliers (up to 747)
from near-zero shuffle_mean denominators. gsi_robust stabilizes the denominator:
  gsi_robust = shuffle_std / max(|shuffle_mean|, shuffle_std * 0.1)

This produces meaningful R² values for human datasets that previously showed
negative R² (Agarwal: -9.56, Klein: -19.14, inoue: -0.57).
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.decomposition.biophysics import compute_biophysics_residual
from src.utils.io import load_processed, save_json

RESULTS_DIR = 'results/v2'
GSI_DIR = os.path.join(RESULTS_DIR, 'module1')
OUT_DIR = os.path.join(RESULTS_DIR, 'module5')

DATASETS = ['agarwal', 'inoue', 'jores', 'klein', 'vaishnav']

def main():
    # Load combined GSI results
    all_gsi = pd.read_parquet(os.path.join(GSI_DIR, 'all_gsi_results.parquet'))
    print(f"Loaded {len(all_gsi)} GSI measurements")
    print(f"Columns: {list(all_gsi.columns)}")

    results_summary = {}

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing {ds_name}...")

        # Load processed dataset
        processed_path = f'data/processed/{ds_name}_processed.parquet'
        if not os.path.exists(processed_path):
            print(f"  Could not find {processed_path}, skipping")
            continue
        df = pd.read_parquet(processed_path)
        if df is None or len(df) == 0:
            print(f"  Empty data for {ds_name}, skipping")
            continue

        # Get GSI for this dataset
        ds_gsi = all_gsi[all_gsi['dataset'] == ds_name]
        print(f"  GSI rows: {len(ds_gsi)}")

        if len(ds_gsi) < 20:
            print(f"  Too few GSI rows, skipping")
            continue

        # Use gsi_robust instead of gsi
        gsi_per_seq_robust = ds_gsi.groupby('seq_id')['gsi_robust'].mean()
        gsi_per_seq_raw = ds_gsi.groupby('seq_id')['gsi'].mean()

        merged = df[df['seq_id'].isin(gsi_per_seq_robust.index)]
        print(f"  Merged rows: {len(merged)}")

        if len(merged) < 20:
            print(f"  Too few merged rows, skipping")
            continue

        # Run biophysics with gsi_robust
        grammar_effects_robust = gsi_per_seq_robust.loc[merged['seq_id'].values].values
        grammar_effects_raw = gsi_per_seq_raw.loc[merged['seq_id'].values].values

        print(f"  Raw GSI: mean={np.mean(grammar_effects_raw):.3f}, median={np.median(grammar_effects_raw):.3f}, max={np.max(grammar_effects_raw):.1f}, std={np.std(grammar_effects_raw):.3f}")
        print(f"  Robust GSI: mean={np.mean(grammar_effects_robust):.3f}, median={np.median(grammar_effects_robust):.3f}, max={np.max(grammar_effects_robust):.3f}, std={np.std(grammar_effects_robust):.3f}")

        bio_robust = compute_biophysics_residual(merged, grammar_effects_robust)

        # Read old results for comparison
        old_path = os.path.join(OUT_DIR, f'{ds_name}_biophysics.json')
        old_r2 = None
        if os.path.exists(old_path):
            with open(old_path) as f:
                old = json.load(f)
                old_r2 = old.get('biophysics_r2', None)

        r2_new = bio_robust['biophysics_r2']
        print(f"  Biophysics R² (gsi_robust): {r2_new:.4f} ± {bio_robust['biophysics_r2_std']:.4f}")
        if old_r2 is not None:
            print(f"  Biophysics R² (raw GSI):    {old_r2:.4f}")
            print(f"  Change: {r2_new - old_r2:+.4f}")

        # Top 5 features
        importances = bio_robust['feature_importances']
        top5 = list(importances.items())[:5]
        print(f"  Top features: {', '.join(f'{k}={v:.3f}' for k,v in top5)}")

        # Save corrected results with metadata
        bio_robust['gsi_metric'] = 'gsi_robust'
        bio_robust['original_r2_raw_gsi'] = old_r2
        save_json(bio_robust, os.path.join(OUT_DIR, f'{ds_name}_biophysics_robust.json'))

        results_summary[ds_name] = {
            'r2_raw': old_r2,
            'r2_robust': r2_new,
            'r2_std': bio_robust['biophysics_r2_std'],
            'top_feature': top5[0][0] if top5 else None,
            'top_feature_importance': top5[0][1] if top5 else None,
            'n_samples': bio_robust['n_samples'],
        }

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Biophysics R² Comparison (raw GSI vs gsi_robust)")
    print(f"{'Dataset':<15} {'R² (raw)':<12} {'R² (robust)':<14} {'Change':<10} {'Top Feature'}")
    print("-" * 75)
    for ds_name in DATASETS:
        if ds_name in results_summary:
            r = results_summary[ds_name]
            r2_raw = r['r2_raw']
            r2_rob = r['r2_robust']
            raw_str = f"{r2_raw:.3f}" if r2_raw is not None else "N/A"
            print(f"{ds_name:<15} {raw_str:<12} {r2_rob:<14.3f} {r2_rob - (r2_raw or 0):+.3f}     {r['top_feature']} ({r['top_feature_importance']:.3f})")

    # Save summary
    save_json(results_summary, os.path.join(OUT_DIR, 'biophysics_robust_comparison.json'))
    print(f"\nSaved comparison to {OUT_DIR}/biophysics_robust_comparison.json")
    print("Saved per-dataset results to {ds}_biophysics_robust.json")


if __name__ == '__main__':
    main()
