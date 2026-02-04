#!/usr/bin/env python
"""
Prepare processed MPRA data for the full pipeline.

1. Load preprocessed parquet files
2. Subsample for pipeline (configurable)
3. Run motif scanning (FIMO or fallback PWM)
4. Add n_motifs, motif_density columns
5. Save as {name}_processed.parquet + {name}_processed_motif_hits.parquet
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perturbation.motif_scanner import MotifScanner

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MOTIF_DIR = os.path.join(DATA_DIR, 'motifs')

# Mapping from source files to pipeline names and motif databases
DATASET_CONFIG = {
    'vaishnav': {
        'source_file': 'vaishnav2022.parquet',
        'motif_db': 'yeast_motifs.meme',
        'species': 'yeast',
        'subsample': 5000,
    },
    'klein': {
        'source_file': 'klein2020.parquet',
        'motif_db': 'JASPAR2024_vertebrates.meme',  # Klein 2020 is human HepG2
        'species': 'human',
        'subsample': None,  # Only 2275 sequences, use all
    },
    'de_almeida': {
        'source_file': 'de_almeida2024.parquet',
        'motif_db': 'JASPAR2024_vertebrates.meme',
        'species': 'human',
        'subsample': 5000,
    },
    'agarwal': {
        'source_file': 'agarwal2023.parquet',
        'motif_db': 'JASPAR2024_vertebrates.meme',
        'species': 'human',
        'subsample': 5000,
    },
    'kircher': {
        'source_file': 'kircher2019.parquet',
        'motif_db': 'JASPAR2024_vertebrates.meme',
        'species': 'human',
        'subsample': 5000,
    },
    'jores': {
        'source_file': 'jores2021.parquet',
        'motif_db': 'JASPAR2024_plants.meme',
        'species': 'plant',
        'subsample': 5000,
    },
}


def prepare_dataset(name, config, max_motifs_per_db=200, batch_size=500):
    """Prepare a single dataset for the pipeline."""
    source = os.path.join(PROCESSED_DIR, config['source_file'])
    if not os.path.exists(source):
        print(f"  [{name}] Source file not found: {source}, skipping")
        return False

    print(f"\n{'='*60}")
    print(f"Preparing: {name}")
    print(f"{'='*60}")

    # Load
    df = pd.read_parquet(source)
    print(f"  Loaded {len(df)} sequences from {config['source_file']}")

    # Subsample
    if config['subsample'] and len(df) > config['subsample']:
        df = df.sample(config['subsample'], random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {len(df)} sequences")

    # Ensure seq_id column
    if 'seq_id' not in df.columns:
        df['seq_id'] = [f"{name}_{i}" for i in range(len(df))]
    df['seq_id'] = df['seq_id'].astype(str)

    # Ensure sequence column
    if 'sequence' not in df.columns:
        raise ValueError(f"No 'sequence' column in {source}")

    # Load motif database
    motif_db_path = os.path.join(MOTIF_DIR, config['motif_db'])
    if not os.path.exists(motif_db_path):
        print(f"  Motif database not found: {motif_db_path}")
        return False

    # Count motifs in database
    n_motifs_in_db = 0
    with open(motif_db_path) as f:
        for line in f:
            if line.startswith('MOTIF'):
                n_motifs_in_db += 1
    print(f"  Motif database: {config['motif_db']} ({n_motifs_in_db} motifs)")

    # Use a subset of motifs if database is very large
    effective_db = motif_db_path
    if n_motifs_in_db > max_motifs_per_db:
        print(f"  Limiting to first {max_motifs_per_db} motifs for speed")
        effective_db = _subset_meme_file(motif_db_path, max_motifs_per_db)

    # Run motif scanning in batches
    print(f"  Scanning motifs (score_fraction >= 0.65)...")
    scanner = MotifScanner(effective_db, p_threshold=1e-4, score_fraction=0.65)

    all_hits = []
    sequences = df['sequence'].tolist()
    seq_ids = df['seq_id'].tolist()

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        batch_ids = seq_ids[i:i+batch_size]
        hits = scanner.scan_sequences(batch_seqs, batch_ids)
        if len(hits) > 0:
            all_hits.append(hits)
        if (i + batch_size) % 1000 == 0 or i + batch_size >= len(sequences):
            print(f"    Scanned {min(i+batch_size, len(sequences))}/{len(sequences)} sequences "
                  f"({sum(len(h) for h in all_hits)} hits so far)")

    if all_hits:
        motif_hits = pd.concat(all_hits, ignore_index=True)
    else:
        motif_hits = pd.DataFrame(columns=[
            'seq_id', 'motif_id', 'motif_name', 'start', 'end',
            'strand', 'score', 'p_value', 'matched_sequence'
        ])

    print(f"  Total motif hits: {len(motif_hits)}")

    # Add n_motifs and motif_density to dataset
    motif_counts = motif_hits.groupby('seq_id').size().reset_index(name='n_motifs')
    df = df.merge(motif_counts, on='seq_id', how='left')
    df['n_motifs'] = df['n_motifs'].fillna(0).astype(int)
    df['motif_density'] = df['n_motifs'] / df['sequence'].str.len()

    print(f"  Motif stats:")
    print(f"    Sequences with motifs: {(df['n_motifs'] > 0).sum()}/{len(df)}")
    print(f"    Sequences with >=2 motifs: {(df['n_motifs'] >= 2).sum()}/{len(df)}")
    print(f"    Mean motifs per sequence: {df['n_motifs'].mean():.1f}")
    print(f"    Max motifs per sequence: {df['n_motifs'].max()}")

    # Save
    out_data = os.path.join(PROCESSED_DIR, f'{name}_processed.parquet')
    out_hits = os.path.join(PROCESSED_DIR, f'{name}_processed_motif_hits.parquet')

    df.to_parquet(out_data, index=False)
    motif_hits.to_parquet(out_hits, index=False)

    print(f"  Saved: {out_data}")
    print(f"  Saved: {out_hits}")

    # Clean up temp file if created
    if effective_db != motif_db_path and os.path.exists(effective_db):
        os.unlink(effective_db)

    return True


def _subset_meme_file(meme_path, max_motifs):
    """Create a temporary MEME file with only the first N motifs."""
    import tempfile
    out_path = tempfile.mktemp(suffix='.meme')

    motif_count = 0
    header_done = False
    writing = True

    with open(meme_path) as fin, open(out_path, 'w') as fout:
        for line in fin:
            if line.startswith('MOTIF'):
                motif_count += 1
                if motif_count > max_motifs:
                    writing = False
                else:
                    writing = True
                    header_done = True
            if writing:
                fout.write(line)

    return out_path


def main():
    parser = argparse.ArgumentParser(description='Prepare pipeline data')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to prepare (default: all available)')
    parser.add_argument('--max-motifs-db', type=int, default=150,
                        help='Max motifs from database to use')
    args = parser.parse_args()

    datasets = args.datasets or list(DATASET_CONFIG.keys())
    prepared = []

    for name in datasets:
        if name not in DATASET_CONFIG:
            print(f"Unknown dataset: {name}, skipping")
            continue

        success = prepare_dataset(name, DATASET_CONFIG[name],
                                  max_motifs_per_db=args.max_motifs_db)
        if success:
            prepared.append(name)

    print(f"\n{'='*60}")
    print(f"PREPARATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Prepared: {prepared}")
    print(f"  Available for pipeline: {prepared}")


if __name__ == '__main__':
    main()
