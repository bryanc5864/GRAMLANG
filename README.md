# GRAMLANG

**DNA foundation models encode motif identity but fail to learn regulatory grammar**

## Overview

GRAMLANG investigates what DNA foundation models learn about regulatory sequences. We propose and validate the **billboard model**: foundation models encode which transcription factor binding sites are present (motif identity) but not their relative positions, orientations, or spacing (motif arrangement). Under this model, enhancers function like billboards—what matters is which elements appear, not how they are arranged.

**Key finding:** 89.7% of enhancers (95% CI: 87.7–91.7%) fit the billboard model. However, 363 MPRA sequence pairs with identical motif vocabulary but different arrangements show expression differences up to Δ=6.3—proving grammar matters biologically. Models fail to detect even synthetic grammar rules (p=0.73) or known helical periodicity (p=0.96). The billboard model describes model limitations, not biological reality.

## Methods

**SF-GSI (Spacer-Factored Grammar Sensitivity Index):** A framework that isolates motif arrangement effects from motif identity effects using matched-spacer perturbations and null-normalized testing.

**Models tested:** DNABERT-2, Nucleotide Transformer v2, HyenaDNA, Enformer, PARM

**Datasets:** Agarwal (K562), Klein (HepG2), Inoue (neural), Vaishnav (yeast), Jores (plant) — 7,650 model–enhancer measurements total

## Key Results

| Metric | Value |
|--------|-------|
| Billboard class enhancers | 89.7% (95% CI: 87.7–91.7%) |
| Grammar-sensitive after FDR | 0.12% (9/7,650) |
| Spacer contribution to variance | 78–86% |
| MPRA pairs with identical vocab, different expression | 363 (Δ up to 6.3) |
| Synthetic grammar detection | p = 0.73 (not detected) |
| Helical periodicity detection | p = 0.96 (not detected) |

## Project Structure

```
src/
  models/        Foundation model wrappers and expression probes
  grammar/       Grammar sensitivity index (GSI), SF-GSI, rule extraction
  perturbation/  Vocabulary-preserving shuffles
  analysis/      Biophysics, transfer, completeness analysis
scripts/         Pipeline scripts
results/         Results and figures
data/            MPRA datasets (not included; see below)
```

## Installation

```bash
conda env create -f environment.yml
conda activate gramlang
```

**Requirements:** Python >= 3.10, PyTorch >= 2.1.0, CUDA-capable GPU

## Data Acquisition

MPRA datasets must be downloaded separately:

- Agarwal et al., Nature 639:411, 2025 (K562)
- Klein et al., Nat. Methods 17:1147, 2020 (HepG2)
- Inoue et al., Cell Stem Cell 25:713, 2019 (neural)
- Vaishnav et al., Nature 603:455, 2022 (yeast)
- Jores et al., Nat. Plants 7:842, 2021 (plant)

## Usage

```bash
# Train probes
python scripts/train_probes.py --models dnabert2 nt hyenadna --datasets agarwal klein inoue vaishnav jores

# Run SF-GSI analysis
python scripts/run_sf_gsi.py

# Generate figures
python scripts/generate_final_figures.py
```

## License

MIT License

## Citation

Cheng, B. (2026). DNA foundation models encode motif identity but fail to learn regulatory grammar. ISMB 2026.
