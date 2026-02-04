#!/usr/bin/env python
"""
Preprocess remaining MPRA datasets into standardized parquet format.

Handles: Agarwal 2025, Jores 2021, de Almeida (Inoue/Kreimer 2019)
Skips: Kircher 2019 (saturation mutagenesis, not full-sequence MPRA)

Output format: parquet with columns [seq_id, sequence, expression, ...]
"""

import os
import sys
import re
import gzip
import numpy as np
import pandas as pd
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MPRA_DIR = os.path.join(PROJECT_DIR, 'data', 'mpra')
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')

os.makedirs(PROCESSED_DIR, exist_ok=True)


def preprocess_agarwal():
    """
    Agarwal et al. (2025, Nature) - K562 lentiMPRA
    Table S2: sequences (230nt with 15nt adaptors on each end)
    Table S3: expression (log2 RNA/DNA ratios)
    """
    print("\n" + "=" * 60)
    print("Preprocessing: Agarwal 2025 (K562)")
    print("=" * 60)

    s2_path = os.path.join(MPRA_DIR, 'agarwal2025', 'Supplementary_Table_2.xlsx')
    s3_path = os.path.join(MPRA_DIR, 'agarwal2025', 'Supplementary_Table_3.xlsx')

    # Load sequences (K562 large-scale, header at row 1)
    print("  Loading sequences from Table S2...")
    seq_df = pd.read_excel(s2_path, sheet_name='K562 large-scale', skiprows=1)
    # Columns: name, category, chr.hg38, start.hg38, stop.hg38, str.hg38, 230nt sequence
    seq_df.columns = ['name', 'category', 'chr', 'start', 'end', 'strand', 'sequence_230nt']
    print(f"  Loaded {len(seq_df)} sequences")

    # Strip adaptors: 15nt 5' (AGGACCGGATCAACT) + 15nt 3' (CATTGCGTGAACCGA)
    seq_df['sequence'] = seq_df['sequence_230nt'].str[15:-15]
    print(f"  Stripped adaptors: 230bp -> {seq_df['sequence'].str.len().iloc[0]}bp elements")

    # Load expression (K562 summary)
    print("  Loading expression from Table S3...")
    expr_df = pd.read_excel(s3_path, sheet_name='K562_summary_data', header=0)
    # Columns: name, rep1, rep2, rep3, mean
    expr_df.columns = ['name', 'rep1', 'rep2', 'rep3', 'expression']
    expr_df['expression'] = pd.to_numeric(expr_df['expression'], errors='coerce')
    print(f"  Loaded {len(expr_df)} expression values")

    # Merge
    merged = seq_df.merge(expr_df[['name', 'expression']], on='name', how='inner')
    merged = merged.dropna(subset=['sequence', 'expression'])

    # Remove reversed duplicates (keep forward orientation only)
    merged_fwd = merged[~merged['name'].str.endswith('_Reversed:')].copy()
    print(f"  Forward-only elements: {len(merged_fwd)}")

    # Create standardized output
    result = pd.DataFrame({
        'seq_id': merged_fwd['name'].values,
        'sequence': merged_fwd['sequence'].values,
        'expression': merged_fwd['expression'].values,
        'category': merged_fwd['category'].values,
        'chr': merged_fwd['chr'].values,
        'start': merged_fwd['start'].values,
        'end': merged_fwd['end'].values,
    })

    # Remove sequences with non-ACGT characters
    valid = result['sequence'].str.match(r'^[ACGTacgt]+$')
    result = result[valid].reset_index(drop=True)
    print(f"  Valid sequences: {len(result)}")

    out_path = os.path.join(PROCESSED_DIR, 'agarwal2023.parquet')
    result.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Expression range: [{result['expression'].min():.2f}, {result['expression'].max():.2f}]")

    return result


def preprocess_jores():
    """
    Jores et al. (2021, Nature Plants) - Plant promoters
    Table S1: promoter sequences (170bp)
    Table S2: promoter strength (log2, 6 conditions)
    Use 'with enhancer, dark, tobacco leaves' as primary expression.
    """
    print("\n" + "=" * 60)
    print("Preprocessing: Jores 2021 (Plant promoters)")
    print("=" * 60)

    s1_path = os.path.join(MPRA_DIR, 'jores2021', 'Supplementary_Table_1.xlsx')
    s2_path = os.path.join(MPRA_DIR, 'jores2021', 'Supplementary_Table_2.xlsx')

    # Load sequences (header at row 3, 0-indexed)
    print("  Loading sequences from Table S1...")
    seq_df = pd.read_excel(s1_path, skiprows=3)
    # Columns: gene, species, barcodes, type, chromosome, start, end, strand, GC, UTR, mutations, sequence
    print(f"  Loaded {len(seq_df)} promoters")
    print(f"  Species: {seq_df['species'].value_counts().to_dict()}")

    # Load expression
    print("  Loading expression from Table S2...")
    expr_df = pd.read_excel(s2_path, skiprows=3)
    # Use 'with enhancer, dark, tobacco leaves' as primary expression
    expr_cols = expr_df.columns.tolist()
    print(f"  Expression columns: {expr_cols}")

    # Find the tobacco dark with-enhancer column
    tobacco_col = [c for c in expr_cols if 'with enhancer' in c and 'dark' in c and 'tobacco' in c]
    if tobacco_col:
        expr_col = tobacco_col[0]
    else:
        # Fallback: any with-enhancer condition
        enhancer_col = [c for c in expr_cols if 'with enhancer' in c]
        expr_col = enhancer_col[0] if enhancer_col else expr_cols[2]
    print(f"  Using expression column: {expr_col}")

    expr_df = expr_df.rename(columns={
        expr_df.columns[0]: 'gene',
        expr_df.columns[1]: 'species',
        expr_col: 'expression'
    })
    expr_df['expression'] = pd.to_numeric(expr_df['expression'], errors='coerce')

    # Merge on gene + species
    merged = seq_df.merge(
        expr_df[['gene', 'species', 'expression']],
        on=['gene', 'species'],
        how='inner'
    )
    merged = merged.dropna(subset=['sequence', 'expression'])
    print(f"  Merged: {len(merged)} promoters with expression")

    # Create standardized output
    result = pd.DataFrame({
        'seq_id': [f"jores_{i}" for i in range(len(merged))],
        'sequence': merged['sequence'].values,
        'expression': merged['expression'].values,
        'species': merged['species'].values,
        'gene': merged['gene'].values,
        'promoter_type': merged['type'].values,
    })

    # Remove invalid sequences
    valid = result['sequence'].str.match(r'^[ACGTacgt]+$')
    result = result[valid].reset_index(drop=True)
    # Re-assign seq_ids
    result['seq_id'] = [f"jores_{i}" for i in range(len(result))]
    print(f"  Valid sequences: {len(result)}")

    out_path = os.path.join(PROCESSED_DIR, 'jores2021.parquet')
    result.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Expression range: [{result['expression'].min():.2f}, {result['expression'].max():.2f}]")

    return result


def preprocess_dealmeida():
    """
    Inoue & Kreimer et al. (2019, Cell Stem Cell) - Neural induction MPRA
    FASTA: barcoded sequences (multiple barcodes per element)
    Count TSVs: barcode counts per timepoint/replicate
    Compute log2(RNA/DNA) per element, use T48h timepoint.
    """
    print("\n" + "=" * 60)
    print("Preprocessing: de Almeida / Inoue-Kreimer 2019 (Neural induction)")
    print("=" * 60)

    data_dir = os.path.join(MPRA_DIR, 'dealmeida2022')
    fasta_path = os.path.join(data_dir, 'GSE115042_plasmid_library_MPRA.fa.gz')

    # Step 1: Parse FASTA to get element sequences keyed by genomic coordinates
    # FASTA IDs: Half_Array1_seq2_[chr1:2478386-2478556]_barcode1
    # Count IDs: A1_seq1258_[chr5:116095916-116096086]_barcode111430
    # Common key: genomic coordinates [chrN:start-end]
    print("  Parsing FASTA library...")
    coord_to_seq = {}  # coordinate -> core sequence

    with gzip.open(fasta_path, 'rt') as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                header = line[1:]
            else:
                coord_match = re.search(r'\[(.*?)\]', header)
                if coord_match:
                    coord = coord_match.group(1)
                    if coord not in coord_to_seq:
                        # Core sequence: strip 15nt 5' adaptor and last 15nt (barcode + 3' adaptor)
                        core = line[15:-15]
                        coord_to_seq[coord] = core

    print(f"  Unique elements (by coordinate): {len(coord_to_seq)}")

    # Step 2: Parse count files for a timepoint (use T48h for mature response)
    timepoint = 'T48h'
    reps = ['rep1', 'rep2', 'rep3']

    dna_counts = defaultdict(lambda: defaultdict(int))  # coord -> rep -> count
    rna_counts = defaultdict(lambda: defaultdict(int))

    for rep in reps:
        # DNA counts
        dna_file = os.path.join(data_dir, f'{timepoint}_{rep}_DNA.tsv.gz')
        if not os.path.exists(dna_file):
            print(f"  WARNING: Missing {dna_file}")
            continue

        with gzip.open(dna_file, 'rt') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    count, full_id = int(parts[1]), parts[2]
                    coord_match = re.search(r'\[(.*?)\]', full_id)
                    if coord_match:
                        coord = coord_match.group(1)
                        dna_counts[coord][rep] += count

        # RNA counts
        rna_file = os.path.join(data_dir, f'{timepoint}_{rep}_RNA.tsv.gz')
        if not os.path.exists(rna_file):
            print(f"  WARNING: Missing {rna_file}")
            continue

        with gzip.open(rna_file, 'rt') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    count, full_id = int(parts[1]), parts[2]
                    coord_match = re.search(r'\[(.*?)\]', full_id)
                    if coord_match:
                        coord = coord_match.group(1)
                        rna_counts[coord][rep] += count

        print(f"  Loaded {timepoint} {rep}: {len(dna_counts)} DNA, {len(rna_counts)} RNA elements")

    # Step 3: Compute expression as mean log2(RNA/DNA) across replicates
    print("  Computing expression values...")
    print(f"  Coordinates in FASTA: {len(coord_to_seq)}")
    print(f"  Coordinates in DNA counts: {len(dna_counts)}")
    print(f"  Overlap: {len(set(coord_to_seq.keys()) & set(dna_counts.keys()))}")

    records = []
    for coord, sequence in coord_to_seq.items():
        if coord not in dna_counts or coord not in rna_counts:
            continue

        log2_ratios = []
        for rep in reps:
            dna = dna_counts[coord].get(rep, 0)
            rna = rna_counts[coord].get(rep, 0)
            if dna >= 10:  # minimum DNA count threshold
                ratio = (rna + 1) / (dna + 1)  # pseudocount
                log2_ratios.append(np.log2(ratio))

        if len(log2_ratios) >= 2:  # require at least 2 replicates
            # Sanitize seq_id: replace colons/special chars for FIMO compatibility
            safe_id = coord.replace(':', '_').replace('-', '_')
            records.append({
                'seq_id': safe_id,
                'sequence': sequence,
                'expression': np.mean(log2_ratios),
                'expression_std': np.std(log2_ratios),
                'n_replicates': len(log2_ratios),
                'coordinates': coord,
                'timepoint': timepoint,
            })

    result = pd.DataFrame(records)
    print(f"  Elements with expression: {len(result)}")

    # Remove invalid sequences
    valid = result['sequence'].str.match(r'^[ACGTacgt]+$')
    result = result[valid].reset_index(drop=True)
    print(f"  Valid sequences: {len(result)}")

    if len(result) > 0:
        out_path = os.path.join(PROCESSED_DIR, 'de_almeida2024.parquet')
        result.to_parquet(out_path, index=False)
        print(f"  Saved: {out_path}")
        print(f"  Expression range: [{result['expression'].min():.2f}, {result['expression'].max():.2f}]")
    else:
        print("  WARNING: No valid elements found!")

    return result


def main():
    results = {}

    # Agarwal 2025
    try:
        results['agarwal'] = preprocess_agarwal()
    except Exception as e:
        print(f"  ERROR in agarwal: {e}")
        import traceback
        traceback.print_exc()

    # Jores 2021
    try:
        results['jores'] = preprocess_jores()
    except Exception as e:
        print(f"  ERROR in jores: {e}")
        import traceback
        traceback.print_exc()

    # de Almeida 2019
    try:
        results['dealmeida'] = preprocess_dealmeida()
    except Exception as e:
        print(f"  ERROR in dealmeida: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    for name, df in results.items():
        if df is not None and len(df) > 0:
            print(f"  {name}: {len(df)} sequences, "
                  f"seq_len={df['sequence'].str.len().median():.0f}bp, "
                  f"expr=[{df['expression'].min():.2f}, {df['expression'].max():.2f}]")
        else:
            print(f"  {name}: FAILED")

    # Note about Kircher
    print("\n  NOTE: Kircher 2019 skipped - saturation mutagenesis data")
    print("  (single-nucleotide variants, not full-sequence MPRA)")


if __name__ == '__main__':
    main()
