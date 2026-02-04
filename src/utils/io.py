"""Data I/O utilities for GRAMLANG."""

import os
import json
import numpy as np
import pandas as pd
from typing import Any, Optional


def load_raw_mpra(dataset_name: str, raw_path: str) -> pd.DataFrame:
    """
    Load raw MPRA data from various formats.

    Args:
        dataset_name: One of 'agarwal', 'kircher', 'de_almeida', 'vaishnav', 'jores', 'klein'
        raw_path: Path to the raw data directory

    Returns:
        DataFrame with at minimum columns: 'sequence', 'expression'
    """
    if dataset_name == 'agarwal':
        return _load_agarwal(raw_path)
    elif dataset_name == 'kircher':
        return _load_kircher(raw_path)
    elif dataset_name == 'de_almeida':
        return _load_de_almeida(raw_path)
    elif dataset_name == 'vaishnav':
        return _load_vaishnav(raw_path)
    elif dataset_name == 'jores':
        return _load_jores(raw_path)
    elif dataset_name == 'klein':
        return _load_klein(raw_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_agarwal(raw_path: str) -> pd.DataFrame:
    """Load Agarwal et al. MPRA data."""
    # Try common file patterns
    for ext in ['.tsv', '.tsv.gz', '.csv', '.csv.gz', '.txt', '.txt.gz']:
        for prefix in ['agarwal', 'mpra', 'data', 'sequences']:
            fpath = os.path.join(raw_path, prefix + ext)
            if os.path.exists(fpath):
                sep = '\t' if 'tsv' in ext or 'txt' in ext else ','
                compression = 'gzip' if ext.endswith('.gz') else None
                df = pd.read_csv(fpath, sep=sep, compression=compression)
                return _standardize_columns(df)

    # Try loading any file in the directory
    files = os.listdir(raw_path) if os.path.isdir(raw_path) else []
    for f in sorted(files):
        fpath = os.path.join(raw_path, f)
        if f.endswith('.parquet'):
            return _standardize_columns(pd.read_parquet(fpath))
        elif f.endswith(('.tsv', '.tsv.gz', '.txt', '.txt.gz')):
            compression = 'gzip' if f.endswith('.gz') else None
            return _standardize_columns(pd.read_csv(fpath, sep='\t', compression=compression))
        elif f.endswith(('.csv', '.csv.gz')):
            compression = 'gzip' if f.endswith('.gz') else None
            return _standardize_columns(pd.read_csv(fpath, compression=compression))

    raise FileNotFoundError(f"No data files found in {raw_path}")


# Generic loaders for other datasets (same pattern)
_load_kircher = _load_agarwal
_load_de_almeida = _load_agarwal
_load_vaishnav = _load_agarwal
_load_jores = _load_agarwal
_load_klein = _load_agarwal


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names across datasets."""
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    # Sequence column
    for name in ['sequence', 'seq', 'dna', 'dna_sequence', 'enhancer_seq',
                 'oligo', 'oligo_sequence', 'promoter_seq']:
        if name in lower_cols:
            col_map[lower_cols[name]] = 'sequence'
            break

    # Expression column
    for name in ['expression', 'log2_ratio', 'log2fc', 'activity', 'expr',
                 'log2_expression', 'mean_expression', 'score', 'value',
                 'log2_rna_dna', 'rna_dna_log2']:
        if name in lower_cols:
            col_map[lower_cols[name]] = 'expression'
            break

    # Cell type column
    for name in ['cell_type', 'celltype', 'cell', 'tissue', 'condition']:
        if name in lower_cols:
            col_map[lower_cols[name]] = 'cell_type'
            break

    df = df.rename(columns=col_map)

    # Ensure sequence column exists
    if 'sequence' not in df.columns:
        # Try to find the column with DNA-like content
        for col in df.columns:
            sample = str(df[col].iloc[0]).upper()
            if len(sample) > 10 and all(c in 'ACGTN' for c in sample):
                df = df.rename(columns={col: 'sequence'})
                break

    return df


def save_processed(df: pd.DataFrame, output_path: str) -> None:
    """Save processed DataFrame to parquet."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)


def load_processed(path: str) -> pd.DataFrame:
    """Load processed parquet file."""
    return pd.read_parquet(path)


def save_json(data: Any, path: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def get_disk_usage(path: str) -> float:
    """Get disk usage in GB for a path."""
    import subprocess
    result = subprocess.run(['du', '-sb', path], capture_output=True, text=True)
    if result.returncode == 0:
        bytes_used = int(result.stdout.split()[0])
        return bytes_used / (1024**3)
    return 0.0


def check_disk_budget(budget_gb: float = 200.0) -> dict:
    """Check current disk usage against budget."""
    import subprocess
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    usage = get_disk_usage(project_dir)

    return {
        'used_gb': usage,
        'budget_gb': budget_gb,
        'remaining_gb': budget_gb - usage,
        'over_budget': usage > budget_gb,
        'utilization_pct': (usage / budget_gb) * 100
    }
