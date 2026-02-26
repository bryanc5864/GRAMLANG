#!/usr/bin/env python3
"""
GPU-dependent validation analysiss.
Run after resolve_validations.py for CPU-based analyses.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

RESULTS_DIR = Path('results/v3/validation_analysiss')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_cohens_d(diffs):
    """Compute Cohen's d for differences from zero."""
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    if std_diff == 0:
        return 0
    return mean_diff / std_diff


def analysis_1_positive_control_all_models():
    """
    validation 1: Run all 3 models on Georgakopoulos-Soares positive control.
    """
    print("\n" + "="*60)
    print("analysis 1: Positive Control - All Models")
    print("="*60)

    from src.models.model_loader import load_model

    # Load the Georgakopoulos-Soares library
    library_path = Path('data/raw/georgakopoulos_soares/Library_MPRA_TFBSs.txt')
    if not library_path.exists():
        print(f"  ERROR: Library not found at {library_path}")
        return None

    # Parse library - FASTA-like format with >header\nsequence
    sequences = []
    current_header = None
    with open(library_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                current_header = line[1:]  # Remove '>'
            elif current_header is not None:
                sequences.append({'header': current_header, 'sequence': line})
                current_header = None

    print(f"  Loaded {len(sequences)} sequences from library")

    if len(sequences) == 0:
        # Try reading differently
        with open(library_path, 'r') as f:
            content = f.read()
            print(f"  File size: {len(content)} bytes")
            print(f"  First 200 chars: {content[:200]}")
        return None

    # Create pairs (consecutive sequences as orientation variants)
    pairs = []
    for i in range(0, min(1000, len(sequences)-1), 2):
        pairs.append((sequences[i], sequences[i+1]))
    pairs = pairs[:500]
    print(f"  Using {len(pairs)} sequence pairs")

    # Test all 3 models
    models_to_test = ['dnabert2', 'nt', 'hyenadna']
    results = {}

    for model_name in models_to_test:
        print(f"\n  Testing {model_name}...")

        try:
            # Load model with agarwal probe (K562 cell type matches G-S data)
            model = load_model(model_name, dataset_name='agarwal')
            print(f"    Model loaded with agarwal probe")

            # Predict expression for each pair
            diffs = []
            for pair in pairs:
                seq1 = pair[0]['sequence']
                seq2 = pair[1]['sequence']

                try:
                    pred1 = model.predict_expression(seq1)
                    pred2 = model.predict_expression(seq2)
                    # Handle case where prediction returns array
                    if hasattr(pred1, '__len__'):
                        pred1 = float(pred1[0]) if len(pred1) > 0 else float(pred1)
                    if hasattr(pred2, '__len__'):
                        pred2 = float(pred2[0]) if len(pred2) > 0 else float(pred2)
                    diff = abs(float(pred1) - float(pred2))
                    diffs.append(diff)
                except Exception as e:
                    continue

            diffs = np.array(diffs).flatten()

            if len(diffs) < 10:
                print(f"    ERROR: Only {len(diffs)} valid predictions")
                results[model_name] = {'error': 'insufficient_predictions'}
                continue

            # Compute statistics
            mean_diff = np.mean(diffs)
            median_diff = np.median(diffs)
            std_diff = np.std(diffs)

            # Cohen's d
            cohens_d = compute_cohens_d(diffs)

            # t-test against zero
            t_stat, p_value = stats.ttest_1samp(diffs, 0)

            results[model_name] = {
                'n_pairs': int(len(diffs)),
                'mean_abs_diff': float(mean_diff),
                'median_abs_diff': float(median_diff),
                'std_diff': float(std_diff),
                'cohens_d': float(cohens_d),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'frac_gt_0.1': float(np.mean(diffs > 0.1)),
                'frac_gt_0.05': float(np.mean(diffs > 0.05))
            }

            print(f"    n_pairs: {len(diffs)}")
            print(f"    mean |Δ|: {mean_diff:.4f}")
            print(f"    Cohen's d: {cohens_d:.3f}")
            print(f"    p-value: {p_value:.2e}")

            # Clean up GPU memory
            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
            results[model_name] = {'error': str(e)}

    # Save results
    output_path = RESULTS_DIR / 'positive_control_all_models.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_path}")

    return results


def analysis_5_gc_all_datasets():
    """
    validation 7: Run GC correlation analysis for ALL 5 datasets.
    """
    print("\n" + "="*60)
    print("analysis 5: GC Correlation - All 5 Datasets")
    print("="*60)

    from src.models.model_loader import load_model

    datasets = ['agarwal', 'jores', 'de_almeida', 'klein', 'vaishnav']
    model_name = 'dnabert2'
    n_samples = 200

    results = {}

    for dataset in datasets:
        print(f"\n  Dataset: {dataset}")

        # Load data
        data_path = Path(f'data/processed/{dataset}_processed.parquet')
        if not data_path.exists():
            print(f"    ERROR: Data not found")
            results[dataset] = {'error': 'data_not_found'}
            continue

        df = pd.read_parquet(data_path)

        # Load model with dataset-specific probe
        print(f"  Loading {model_name} with {dataset} probe...")
        try:
            model = load_model(model_name, dataset_name=dataset)
        except Exception as e:
            print(f"    ERROR loading model: {e}")
            results[dataset] = {'error': str(e)}
            continue

        # Sample sequences
        if len(df) > n_samples:
            df_sample = df.sample(n=n_samples, random_state=42)
        else:
            df_sample = df

        # Compute GC content and predictions
        gc_contents = []
        predictions = []

        for idx, row in df_sample.iterrows():
            seq = row['sequence'].upper()
            gc = (seq.count('G') + seq.count('C')) / len(seq)
            gc_contents.append(gc)

            try:
                pred = model.predict_expression(seq)
                # Handle array predictions
                if hasattr(pred, '__len__'):
                    pred = float(pred[0]) if len(pred) > 0 else float(pred)
                predictions.append(float(pred))
            except:
                predictions.append(np.nan)

        gc_contents = np.array(gc_contents).flatten()
        predictions = np.array(predictions).flatten()

        # Remove NaN
        valid_mask = ~np.isnan(predictions)
        gc_valid = gc_contents[valid_mask]
        pred_valid = predictions[valid_mask]

        # Compute correlation
        if len(gc_valid) > 10:
            r, p = stats.pearsonr(gc_valid, pred_valid)
            results[dataset] = {
                'n': int(len(gc_valid)),
                'r': float(r),
                'p_value': float(p),
                'direction': 'positive' if r > 0 else 'negative',
                'gc_mean': float(np.mean(gc_valid)),
                'gc_std': float(np.std(gc_valid))
            }
            print(f"    r = {r:+.3f}, p = {p:.3e}")
        else:
            results[dataset] = {'error': 'insufficient_data', 'n_valid': int(len(gc_valid))}

    # Clean up
    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Summary
    print("\n  Summary of GC-Expression Correlations:")
    print("  " + "-"*50)
    for dataset in datasets:
        data = results.get(dataset, {})
        if 'r' in data:
            direction = "+" if data['r'] > 0 else ""
            print(f"    {dataset}: r = {direction}{data['r']:.3f}")

    # Save results
    output_path = RESULTS_DIR / 'gc_correlation_all_datasets.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_path}")

    return results


def main():
    """Run GPU-dependent validation analysiss."""
    print("="*60)
    print("GRAMLANG: GPU-Dependent validation analysiss")
    print("="*60)

    all_results = {}

    # analysis 5: GC correlation all datasets
    print("\n[1/2] GC Correlation All Datasets...")
    try:
        all_results['gc_correlation'] = analysis_5_gc_all_datasets()
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # analysis 1: Positive control all models
    print("\n[2/2] Positive Control All Models...")
    try:
        all_results['positive_control'] = analysis_1_positive_control_all_models()
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()

    # Update master results
    master_path = RESULTS_DIR / 'all_validation_analysiss.json'
    if master_path.exists():
        with open(master_path) as f:
            master = json.load(f)
    else:
        master = {}

    master.update(all_results)

    with open(master_path, 'w') as f:
        json.dump(master, f, indent=2, default=str)

    print("\n" + "="*60)
    print("GPU analysisS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
