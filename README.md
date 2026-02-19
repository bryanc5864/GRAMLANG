# GRAMLANG

**Decoding the Computational Grammar of Gene Regulation**

GRAMLANG investigates whether regulatory DNA follows compositional grammar rules using foundation models (DNABERT-2, NT v2-500M, HyenaDNA, Enformer) on five MPRA datasets (Agarwal/K562, Klein/HepG2, de Almeida/neural, Vaishnav/yeast, Jores/plant).

**Key finding (v3):** The standard computational method for measuring grammar (vocabulary-preserving shuffles + expression prediction) is fundamentally confounded by spacer DNA composition effects -- 78-86% of measured variance comes from spacer DNA, not motif arrangement. However, a positive control proves grammar IS real when spacers are controlled (p < 1e-117).

---

## Project Structure

```
src/
  models/        Foundation model wrappers and expression probes (2-layer MLP on frozen embeddings)
  grammar/       Grammar sensitivity index (GSI), rule extraction, compositionality analysis
  perturbation/  Vocabulary-preserving shuffles
  analysis/      Biophysics, transfer, completeness analysis
scripts/         Pipeline scripts (run_full_pipeline.py, train_probes.py,
                 generate_final_figures.py, run_v3_analysis.py)
results/         Module 1-6 results (parquet, JSON) + v3 extension results + figures
data/            MPRA datasets (not included; see Data Acquisition below)
```

## Installation

```bash
# Option A: conda
conda env create -f environment.yml
conda activate gramlang

# Option B: pip
pip install -e .
```

**Requirements:** Python >= 3.10, PyTorch >= 2.1.0, CUDA-capable GPU (tested on 4x A100 80GB, CUDA 12.4, Rocky Linux 9.6).

## Data Acquisition

MPRA datasets must be downloaded separately:

- Agarwal et al. (2023) -- K562 MPRA
- Klein et al. (2020) -- HepG2 MPRA
- de Almeida et al. (2024) -- Neural MPRA
- Vaishnav et al. (2022) -- Yeast MPRA
- Jores et al. (2021) -- Plant MPRA

Motif scanning requires FIMO v5.5.7 and JASPAR 2024 databases.

Place raw data in `data/raw/` and processed data in `data/processed/`.

## Reproduction

### 1. Train probes

```bash
python scripts/train_probes.py \
  --models dnabert2 nt hyenadna \
  --datasets vaishnav2022 klein2020 agarwal2023 jores2021 de_almeida2024
```

### 2. Run main pipeline (Modules 1-6)

```bash
python scripts/run_full_pipeline.py --module 1  # GSI census
python scripts/run_full_pipeline.py --module 2  # Rule extraction
python scripts/run_full_pipeline.py --module 3  # Compositionality
python scripts/run_full_pipeline.py --module 4  # Cross-species transfer
python scripts/run_full_pipeline.py --module 5  # Causal determinants
python scripts/run_full_pipeline.py --module 6  # Completeness
```

### 3. Run v3 extensions

```bash
python scripts/run_v3_analysis.py
```

### 4. Generate figures

```bash
python scripts/generate_final_figures.py
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--n-shuffles` | 50 | Number of vocabulary-preserving shuffles per enhancer |
| `--max-enhancers` | 200 | Enhancers per model-dataset combination |
| `--seed` | 42 | Random seed for reproducibility |

## Results Summary

| Module | Result |
|---|---|
| GSI Census | 7,650 measurements; 8.3% significant (nominal p<0.05), 0.17% FDR-corrected |
| Rules | 9,019 grammar rules; consensus 0.433; orientation agreement 84.8% |
| Compositionality | Gap = 0.989 (context-sensitive in Chomsky hierarchy) |
| Transfer | Zero cross-species grammar transfer |
| Biophysics | R² = 0.06-0.79 across species (v2, robust GSI) |
| Completeness | 6-18% of replicate ceiling |
| v3 Spacer Confound | 78-86% of GSI variance from spacer DNA, not grammar |

## License

MIT License -- see LICENSE file.

## Citation

Cheng, B. (2026). GRAMLANG: Decoding the Computational Grammar of Gene Regulation. [Manuscript in preparation].
