#!/usr/bin/env python
"""
Preprocess all MPRA datasets into a standardized format.

Each dataset is standardized to:
  seq_id, sequence, expression, dataset, species, cell_type

Can be run incrementally - processes only datasets with raw files present.
"""

import os
import sys
import gzip
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'data', 'mpra')
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

STATS = {}


def save_dataset(df, name):
    """Save preprocessed dataset."""
    path = os.path.join(PROCESSED_DIR, f'{name}.parquet')
    df.to_parquet(path, index=False)
    csv_path = os.path.join(PROCESSED_DIR, f'{name}.csv.gz')
    df.to_csv(csv_path, index=False, compression='gzip')
    print(f"  Saved {name}: {len(df)} sequences -> {path}")
    STATS[name] = {
        'n_sequences': len(df),
        'seq_length_mean': float(df['sequence'].str.len().mean()),
        'seq_length_std': float(df['sequence'].str.len().std()),
        'expression_mean': float(df['expression'].mean()),
        'expression_std': float(df['expression'].std()),
        'species': df['species'].iloc[0],
        'cell_type': df['cell_type'].iloc[0] if 'cell_type' in df.columns else 'unknown',
    }
    return df


def preprocess_vaishnav():
    """Preprocess Vaishnav et al. 2022 - Yeast random promoters."""
    data_dir = os.path.join(RAW_DIR, 'vaishnav2022')
    train_file = os.path.join(data_dir, 'train_sequences.txt')
    test_file = os.path.join(data_dir, 'test_sequences.txt')

    if not os.path.exists(train_file):
        print("  Vaishnav: No data files found, skipping")
        return None

    print("  Processing Vaishnav 2022 (yeast promoters)...")
    rows = []
    for fpath, split in [(train_file, 'train'), (test_file, 'test')]:
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for i, line in enumerate(f):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    seq, expr = parts[0], float(parts[1])
                    rows.append({
                        'seq_id': f'vaishnav_{split}_{i}',
                        'sequence': seq.upper(),
                        'expression': expr,
                        'dataset': 'vaishnav2022',
                        'species': 'yeast',
                        'cell_type': 'yeast',
                        'split': split,
                    })

    df = pd.DataFrame(rows)

    # Subsample if too large (keep 200K for manageable experiments)
    if len(df) > 200000:
        print(f"  Subsampling from {len(df)} to 200,000 sequences...")
        np.random.seed(42)
        df = df.sample(200000, random_state=42).reset_index(drop=True)

    return save_dataset(df, 'vaishnav2022')


def preprocess_klein():
    """Preprocess Klein et al. 2020 - HepG2 lentiMPRA."""
    data_dir = os.path.join(RAW_DIR, 'klein2020')
    seq_file = os.path.join(data_dir, 'GSE142696_ForwardReverse_sequences_coordinates.tsv.gz')
    activity_file = os.path.join(data_dir, 'GSE142696_ForwardReverse.ActivityRatios.tsv.gz')

    if not os.path.exists(seq_file):
        print("  Klein: No data files found, skipping")
        return None

    print("  Processing Klein 2020 (HepG2 MPRA)...")

    # Read sequences (skip note lines at top)
    seq_rows = []
    with gzip.open(seq_file, 'rt') as f:
        header = None
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('"'):
                continue
            if header is None:
                header = stripped.split('\t')
                continue
            parts = stripped.split('\t')
            if len(parts) >= len(header):
                row = dict(zip(header, parts))
                seq_rows.append(row)

    seq_df = pd.DataFrame(seq_rows)
    # Find and rename the sequence column
    seq_col = [c for c in seq_df.columns if 'sequence' in c.lower()]
    if seq_col:
        seq_df = seq_df.rename(columns={seq_col[0]: 'sequence'})
    print(f"  Sequence df: {len(seq_df)} rows, columns: {list(seq_df.columns)}")

    # Read activity ratios
    act_rows = []
    with gzip.open(activity_file, 'rt') as f:
        header = None
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('"') or stripped.startswith('Note'):
                continue
            if header is None:
                header = stripped.split('\t')
                continue
            parts = stripped.split('\t')
            if len(parts) >= 2:
                row = dict(zip(header, parts))
                act_rows.append(row)

    act_df = pd.DataFrame(act_rows)

    # Merge on name
    print(f"  Activity df: {len(act_df)} rows, columns: {list(act_df.columns)}")

    if 'name' in seq_df.columns and 'name' in act_df.columns:
        # Activity file has _F/_R suffixes; use forward orientation and strip suffix
        act_fwd = act_df[act_df['name'].str.endswith('_F')].copy()
        act_fwd['name_base'] = act_fwd['name'].str[:-2]  # strip _F
        seq_df['name_base'] = seq_df['name'].str.strip()
        merged = seq_df.merge(act_fwd, on='name_base', how='inner', suffixes=('', '_act'))
    else:
        merged = pd.concat([seq_df, act_df], axis=1)

    print(f"  Merged: {len(merged)} rows, columns: {list(merged.columns)[:10]}")

    # Get expression column
    expr_cols = [c for c in merged.columns if 'mean' in c.lower() and ('ratio' in c.lower() or 'rna' in c.lower())]
    if not expr_cols:
        expr_cols = [c for c in merged.columns if 'ratio' in c.lower()]

    if not expr_cols:
        print("  Klein: Could not find expression columns")
        return None

    expr_col = expr_cols[0]
    print(f"  Using expression column: {expr_col}")

    # Build result DataFrame directly
    result = pd.DataFrame({
        'seq_id': [f'klein_{i}' for i in range(len(merged))],
        'sequence': merged['sequence'].str.upper() if 'sequence' in merged.columns else None,
        'expression': pd.to_numeric(merged[expr_col], errors='coerce'),
        'dataset': 'klein2020',
        'species': 'human',
        'cell_type': 'HepG2',
    })

    # Drop rows without sequence or expression
    result = result.dropna(subset=['sequence', 'expression']).reset_index(drop=True)
    # Filter to valid DNA sequences
    result = result[result['sequence'].str.match(r'^[ACGT]+$', na=False)].reset_index(drop=True)
    print(f"  After filtering: {len(result)} sequences")

    if len(result) == 0:
        print("  Klein: No valid sequences after filtering")
        return None

    return save_dataset(result, 'klein2020')


def preprocess_agarwal():
    """Preprocess Agarwal et al. 2025 - lentiMPRA K562/HepG2/WTC11."""
    data_dir = os.path.join(RAW_DIR, 'agarwal2025')
    # Look for supplementary data files
    files = os.listdir(data_dir) if os.path.exists(data_dir) else []
    if not files:
        print("  Agarwal: No data files found, skipping")
        return None

    print(f"  Processing Agarwal 2025 (lentiMPRA). Files: {files}")

    # Handle various possible file formats
    dfs = []
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        if fname.endswith('.tsv') or fname.endswith('.tsv.gz'):
            try:
                df = pd.read_csv(fpath, sep='\t', comment='#')
                dfs.append((fname, df))
            except Exception as e:
                print(f"    Error reading {fname}: {e}")
        elif fname.endswith('.csv') or fname.endswith('.csv.gz'):
            try:
                df = pd.read_csv(fpath, comment='#')
                dfs.append((fname, df))
            except Exception as e:
                print(f"    Error reading {fname}: {e}")
        elif fname.endswith('.txt') or fname.endswith('.txt.gz'):
            try:
                df = pd.read_csv(fpath, sep='\t', comment='#')
                dfs.append((fname, df))
            except Exception as e:
                print(f"    Error reading {fname}: {e}")

    if not dfs:
        print("  Agarwal: Could not parse any files")
        return None

    # Find the one with sequences and expression values
    for fname, df in dfs:
        cols_lower = [c.lower() for c in df.columns]
        has_seq = any('seq' in c for c in cols_lower)
        has_expr = any('expr' in c or 'activity' in c or 'log2' in c or 'ratio' in c for c in cols_lower)
        if has_seq and has_expr:
            print(f"  Using file: {fname} ({len(df)} rows)")
            # Standardize columns
            seq_col = [c for c in df.columns if 'seq' in c.lower()][0]
            expr_col = [c for c in df.columns if any(x in c.lower() for x in ['expr', 'activity', 'log2', 'ratio'])][0]

            result = pd.DataFrame({
                'seq_id': [f'agarwal_{i}' for i in range(len(df))],
                'sequence': df[seq_col].str.upper(),
                'expression': pd.to_numeric(df[expr_col], errors='coerce'),
                'dataset': 'agarwal2025',
                'species': 'human',
                'cell_type': 'K562',  # Default, may have multiple
            })
            result = result.dropna(subset=['expression', 'sequence']).reset_index(drop=True)
            result = result[result['sequence'].str.match(r'^[ACGT]+$', na=False)].reset_index(drop=True)
            return save_dataset(result, 'agarwal2025')

    print("  Agarwal: No suitable data files found")
    return None


def preprocess_kircher():
    """Preprocess Kircher et al. 2019 - Saturation mutagenesis MPRA."""
    data_dir = os.path.join(RAW_DIR, 'kircher2019')
    files = os.listdir(data_dir) if os.path.exists(data_dir) else []
    if not files:
        print("  Kircher: No data files found, skipping")
        return None

    print(f"  Processing Kircher 2019. Files: {files}")
    # Generic processing similar to agarwal
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        try:
            if fname.endswith('.gz'):
                df = pd.read_csv(fpath, sep='\t', compression='gzip', comment='#', nrows=5)
            else:
                df = pd.read_csv(fpath, sep='\t', comment='#', nrows=5)
            print(f"    {fname}: columns={list(df.columns)[:10]}")
        except Exception as e:
            print(f"    {fname}: could not read ({e})")

    return None


def preprocess_dealmeida():
    """Preprocess de Almeida et al. 2022 - Neural differentiation MPRA."""
    data_dir = os.path.join(RAW_DIR, 'dealmeida2022')
    files = os.listdir(data_dir) if os.path.exists(data_dir) else []
    if not files:
        print("  de Almeida: No data files found, skipping")
        return None

    print(f"  Processing de Almeida 2022. Files: {files}")
    return None


def preprocess_jores():
    """Preprocess Jores et al. 2021 - Plant synthetic promoters."""
    data_dir = os.path.join(RAW_DIR, 'jores2021')
    files = os.listdir(data_dir) if os.path.exists(data_dir) else []
    if not files:
        print("  Jores: No data files found, skipping")
        return None

    print(f"  Processing Jores 2021 (plant promoters). Files: {files}")
    # Generic processing
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        try:
            if fname.endswith('.gz'):
                df = pd.read_csv(fpath, sep='\t', compression='gzip', comment='#', nrows=5)
            elif fname.endswith('.csv') or fname.endswith('.csv.gz'):
                df = pd.read_csv(fpath, comment='#', nrows=5)
            else:
                df = pd.read_csv(fpath, sep='\t', comment='#', nrows=5)
            print(f"    {fname}: columns={list(df.columns)[:10]}")
            cols_lower = [c.lower() for c in df.columns]
            has_seq = any('seq' in c for c in cols_lower)
            has_expr = any('expr' in c or 'enrich' in c or 'log' in c for c in cols_lower)
            if has_seq and has_expr:
                # Full read
                if fname.endswith('.gz'):
                    df = pd.read_csv(fpath, sep='\t', compression='gzip', comment='#')
                elif fname.endswith('.csv'):
                    df = pd.read_csv(fpath, comment='#')
                else:
                    df = pd.read_csv(fpath, sep='\t', comment='#')
                seq_col = [c for c in df.columns if 'seq' in c.lower()][0]
                expr_col = [c for c in df.columns if any(x in c.lower() for x in ['expr', 'enrich', 'log'])][0]
                result = pd.DataFrame({
                    'seq_id': [f'jores_{i}' for i in range(len(df))],
                    'sequence': df[seq_col].str.upper(),
                    'expression': pd.to_numeric(df[expr_col], errors='coerce'),
                    'dataset': 'jores2021',
                    'species': 'plant',
                    'cell_type': 'protoplast',
                })
                result = result.dropna(subset=['expression', 'sequence']).reset_index(drop=True)
                result = result[result['sequence'].str.match(r'^[ACGTN]+$', na=False)].reset_index(drop=True)
                return save_dataset(result, 'jores2021')
        except Exception as e:
            print(f"    {fname}: could not read ({e})")

    return None


if __name__ == '__main__':
    print("=" * 60)
    print("GRAMLANG MPRA Data Preprocessing")
    print("=" * 60)

    datasets = [
        ("Vaishnav 2022 (yeast)", preprocess_vaishnav),
        ("Klein 2020 (HepG2)", preprocess_klein),
        ("Agarwal 2025 (lentiMPRA)", preprocess_agarwal),
        ("Kircher 2019 (saturation)", preprocess_kircher),
        ("de Almeida 2022 (neural)", preprocess_dealmeida),
        ("Jores 2021 (plant)", preprocess_jores),
    ]

    for name, fn in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        fn()

    # Save preprocessing stats
    stats_path = os.path.join(PROCESSED_DIR, 'preprocessing_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(STATS, f, indent=2, default=str)

    print(f"\n\n{'='*60}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    for name, stats in STATS.items():
        print(f"  {name}: {stats['n_sequences']} sequences, "
              f"mean_len={stats['seq_length_mean']:.0f}, "
              f"expr_mean={stats['expression_mean']:.3f}")

    print(f"\nStats saved to: {stats_path}")
    print(f"Processed data in: {PROCESSED_DIR}")
