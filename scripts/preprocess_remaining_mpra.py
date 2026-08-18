#!/usr/bin/env python
"""Preprocess the leftover MPRA datasets into parquet.

covers Agarwal 2025, Jores 2021 and Inoue/Kreimer 2019. Kircher 2019 is skipped,
it is saturation mutagenesis rather than full-sequence MPRA.
output columns: seq_id, sequence, expression, plus dataset-specific extras.
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
    """Agarwal 2025 K562 lentiMPRA: table S2 has the 230nt sequences, S3 the log2 RNA/DNA."""
    print("\n" + "=" * 60)
    print("Preprocessing: Agarwal 2025 (K562)")
    print("=" * 60)

    s2_path = os.path.join(MPRA_DIR, 'agarwal2025', 'Supplementary_Table_2.xlsx')
    s3_path = os.path.join(MPRA_DIR, 'agarwal2025', 'Supplementary_Table_3.xlsx')

    # K562 large-scale sheet, header sits on row 1
    print("  Loading sequences from Table S2...")
    seq_df = pd.read_excel(s2_path, sheet_name='K562 large-scale', skiprows=1)
    # columns: name, category, chr.hg38, start.hg38, stop.hg38, str.hg38, 230nt sequence
    seq_df.columns = ['name', 'category', 'chr', 'start', 'end', 'strand', 'sequence_230nt']
    print(f"  Loaded {len(seq_df)} sequences")

    # 15nt adaptor on each end: AGGACCGGATCAACT / CATTGCGTGAACCGA
    seq_df['sequence'] = seq_df['sequence_230nt'].str[15:-15]
    print(f"  Stripped adaptors: 230bp -> {seq_df['sequence'].str.len().iloc[0]}bp elements")

    print("  Loading expression from Table S3...")
    expr_df = pd.read_excel(s3_path, sheet_name='K562_summary_data', header=0)
    # columns: name, rep1, rep2, rep3, mean
    expr_df.columns = ['name', 'rep1', 'rep2', 'rep3', 'expression']
    expr_df['expression'] = pd.to_numeric(expr_df['expression'], errors='coerce')
    print(f"  Loaded {len(expr_df)} expression values")

    merged = seq_df.merge(expr_df[['name', 'expression']], on='name', how='inner')
    merged = merged.dropna(subset=['sequence', 'expression'])

    # forward orientation only
    merged_fwd = merged[~merged['name'].str.endswith('_Reversed:')].copy()
    print(f"  Forward-only elements: {len(merged_fwd)}")

    result = pd.DataFrame({
        'seq_id': merged_fwd['name'].values,
        'sequence': merged_fwd['sequence'].values,
        'expression': merged_fwd['expression'].values,
        'category': merged_fwd['category'].values,
        'chr': merged_fwd['chr'].values,
        'start': merged_fwd['start'].values,
        'end': merged_fwd['end'].values,
    })

    valid = result['sequence'].str.match(r'^[ACGTacgt]+$')
    result = result[valid].reset_index(drop=True)
    print(f"  Valid sequences: {len(result)}")

    out_path = os.path.join(PROCESSED_DIR, 'agarwal2023.parquet')
    result.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Expression range: [{result['expression'].min():.2f}, {result['expression'].max():.2f}]")

    return result


def preprocess_jores():
    """Jores 2021 plant promoters: S1 has 170bp sequences, S2 the log2 strengths."""
    print("\n" + "=" * 60)
    print("Preprocessing: Jores 2021 (Plant promoters)")
    print("=" * 60)

    s1_path = os.path.join(MPRA_DIR, 'jores2021', 'Supplementary_Table_1.xlsx')
    s2_path = os.path.join(MPRA_DIR, 'jores2021', 'Supplementary_Table_2.xlsx')

    # header sits on row 3
    print("  Loading sequences from Table S1...")
    seq_df = pd.read_excel(s1_path, skiprows=3)
    # columns: gene, species, barcodes, type, chromosome, start, end, strand, GC, UTR, mutations, sequence
    print(f"  Loaded {len(seq_df)} promoters")
    print(f"  Species: {seq_df['species'].value_counts().to_dict()}")

    print("  Loading expression from Table S2...")
    expr_df = pd.read_excel(s2_path, skiprows=3)
    # 'with enhancer, dark, tobacco leaves' is our primary condition
    expr_cols = expr_df.columns.tolist()
    print(f"  Expression columns: {expr_cols}")

    tobacco_col = [c for c in expr_cols if 'with enhancer' in c and 'dark' in c and 'tobacco' in c]
    if tobacco_col:
        expr_col = tobacco_col[0]
    else:
        # fall back to any with-enhancer condition
        enhancer_col = [c for c in expr_cols if 'with enhancer' in c]
        expr_col = enhancer_col[0] if enhancer_col else expr_cols[2]
    print(f"  Using expression column: {expr_col}")

    expr_df = expr_df.rename(columns={
        expr_df.columns[0]: 'gene',
        expr_df.columns[1]: 'species',
        expr_col: 'expression'
    })
    expr_df['expression'] = pd.to_numeric(expr_df['expression'], errors='coerce')

    merged = seq_df.merge(
        expr_df[['gene', 'species', 'expression']],
        on=['gene', 'species'],
        how='inner'
    )
    merged = merged.dropna(subset=['sequence', 'expression'])
    print(f"  Merged: {len(merged)} promoters with expression")

    result = pd.DataFrame({
        'seq_id': [f"jores_{i}" for i in range(len(merged))],
        'sequence': merged['sequence'].values,
        'expression': merged['expression'].values,
        'species': merged['species'].values,
        'gene': merged['gene'].values,
        'promoter_type': merged['type'].values,
    })

    valid = result['sequence'].str.match(r'^[ACGTacgt]+$')
    result = result[valid].reset_index(drop=True)
    # renumber after filtering
    result['seq_id'] = [f"jores_{i}" for i in range(len(result))]
    print(f"  Valid sequences: {len(result)}")

    out_path = os.path.join(PROCESSED_DIR, 'jores2021.parquet')
    result.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Expression range: [{result['expression'].min():.2f}, {result['expression'].max():.2f}]")

    return result


def preprocess_dealmeida():
    """Inoue & Kreimer 2019 neural induction MPRA.

    barcoded FASTA plus per-timepoint count TSVs; expression is log2(RNA/DNA) at T48h.
    """
    print("\n" + "=" * 60)
    print("Preprocessing: Inoue / Inoue-Kreimer 2019 (Neural induction)")
    print("=" * 60)

    data_dir = os.path.join(MPRA_DIR, 'dealmeida2022')
    fasta_path = os.path.join(data_dir, 'GSE115042_plasmid_library_MPRA.fa.gz')

    # FASTA ids: Half_Array1_seq2_[chr1:2478386-2478556]_barcode1
    # count ids: A1_seq1258_[chr5:116095916-116096086]_barcode111430
    # the [chrN:start-end] coordinate is the only key they share
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
                        # strip the 15nt 5' adaptor and the barcode + 3' adaptor
                        core = line[15:-15]
                        coord_to_seq[coord] = core

    print(f"  Unique elements (by coordinate): {len(coord_to_seq)}")

    # T48h is the mature response
    timepoint = 'T48h'
    reps = ['rep1', 'rep2', 'rep3']

    dna_counts = defaultdict(lambda: defaultdict(int))  # coord -> rep -> count
    rna_counts = defaultdict(lambda: defaultdict(int))

    for rep in reps:
        # DNA counts
        dna_file = os.path.join(data_dir, f'{timepoint}_{rep}_DNA.tsv.gz')
        if not os.path.exists(dna_file):
            print(f"  missing {dna_file}")
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
            print(f"  missing {rna_file}")
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

    # expression is the mean log2(RNA/DNA) over replicates
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
            # FIMO chokes on colons in sequence ids
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

    valid = result['sequence'].str.match(r'^[ACGTacgt]+$')
    result = result[valid].reset_index(drop=True)
    print(f"  Valid sequences: {len(result)}")

    if len(result) > 0:
        out_path = os.path.join(PROCESSED_DIR, 'inoue2024.parquet')
        result.to_parquet(out_path, index=False)
        print(f"  Saved: {out_path}")
        print(f"  Expression range: [{result['expression'].min():.2f}, {result['expression'].max():.2f}]")
    else:
        print("  no valid elements found")

    return result


def main():
    results = {}

    try:
        results['agarwal'] = preprocess_agarwal()
    except Exception as e:
        print(f"  error in agarwal: {e}")
        import traceback
        traceback.print_exc()

    try:
        results['jores'] = preprocess_jores()
    except Exception as e:
        print(f"  error in jores: {e}")
        import traceback
        traceback.print_exc()

    try:
        results['dealmeida'] = preprocess_dealmeida()
    except Exception as e:
        print(f"  error in dealmeida: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("preprocessing summary")
    print("=" * 60)
    for name, df in results.items():
        if df is not None and len(df) > 0:
            print(f"  {name}: {len(df)} sequences, "
                  f"seq_len={df['sequence'].str.len().median():.0f}bp, "
                  f"expr=[{df['expression'].min():.2f}, {df['expression'].max():.2f}]")
        else:
            print(f"  {name}: failed")

    print("\n  Kircher 2019 skipped: saturation mutagenesis data")
    print("  (single-nucleotide variants, not full-sequence MPRA)")


if __name__ == '__main__':
    main()
