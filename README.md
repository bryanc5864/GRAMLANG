# GRAMLANG

**Anonymized code release for NeurIPS 2026 submission.**

GRAMLANG is a benchmark and diagnostic framework for evaluating whether DNA foundation models (DNABERT-2, Nucleotide Transformer, HyenaDNA, Enformer, PARM) encode regulatory grammar—motif arrangement, orientation, and spacing—on top of motif identity. The central tool is the **Spacer-Factored Grammar Sensitivity Index (SF-GSI)**, which isolates arrangement effects from spacer-composition confounds via matched-spacer perturbations and a per-enhancer null distribution.

This release contains the complete code, intermediate results, and figure-generation pipeline used to produce the headline finding: 89.7% of enhancers fit a strict billboard model in which models are insensitive to motif arrangement, and the grammar-sensitive fraction collapses from 100% to 0.12% (9 / 7,650) after spacer factoring and FDR correction.

## Repository structure

```
.
├── src/                  Library code (importable as the gramlang package)
│   ├── models/           Foundation-model wrappers and expression probes
│   ├── grammar/          GSI / SF-GSI estimators and rule extraction
│   ├── perturbation/     Vocabulary-preserving perturbation operators
│   ├── decomposition/    Variance decomposition (vocab / grammar / embedding)
│   ├── transfer/         Cross-dataset transfer analyses
│   ├── design/           Synthetic-grammar and helical-periodicity controls
│   └── utils/            Shared helpers
├── scripts/              Top-level entry points (see "Reproducing the paper")
├── results/              Pre-computed intermediate outputs (JSON, parquet)
│   └── manuscript_figures/   Six figures used in the submission
├── data/                 Place MPRA datasets here (gitignored; see "Data")
├── environment.yml       Conda environment specification
├── requirements.txt      Pip-installable dependency list
├── pyproject.toml        Package metadata
└── LICENSE               MIT
```

## Installation

**Requirements:** Python ≥ 3.10, a CUDA-capable GPU (tested on a single A100 40 GB).

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate gramlang
```

### Option B — pip

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Data

Five publicly available MPRA datasets are required. Place them under `data/mpra/<dataset>/` after download. The datasets are released by their original authors under their own licenses; we do not redistribute them.

| Dataset  | Citation                              | Source                                                          |
|----------|---------------------------------------|-----------------------------------------------------------------|
| Agarwal  | Agarwal et al., *Nature* 2025         | GEO companion of the original lentiMPRA publication             |
| Klein    | Klein et al., *Nat. Methods* 2020     | Supplementary tables of the original publication                |
| Inoue    | Inoue et al., *Cell Stem Cell* 2019   | GEO accession from the original publication                     |
| Vaishnav | Vaishnav et al., *Nature* 2022        | Supplementary data of the original publication                  |
| Jores    | Jores et al., *Nature Plants* 2021    | Supplementary tables of the original publication                |

Motif calls use **JASPAR2024** profiles scanned with **FIMO** (MEME Suite) at p < 1e-4. JASPAR PWMs can be downloaded from `jaspar.genereg.net`; FIMO is part of MEME Suite (`meme-suite.org`). A preprocessed cache is written to `data/processed/` after the first preprocessing run.

## Reproducing the paper

The full pipeline runs in **under 200 GPU-hours** on a single A100. To reproduce the headline numbers and all six manuscript figures from raw MPRA data:

```bash
python scripts/run_full_pipeline.py
```

This is equivalent to running, in order:

```bash
python scripts/preprocess_mpra.py             # build motif calls + filtered enhancer sets
python scripts/train_probes.py                # train ridge probes on frozen embeddings
python scripts/run_sf_gsi.py                  # compute SF-GSI for 7,650 (model, dataset, enhancer) triples
python scripts/run_factorial_remaining.py     # variance decomposition (Fig. 1)
python scripts/run_positive_control.py        # synthetic grammar + helical periodicity (Fig. 2)
python scripts/run_validation_analyses.py     # transfer + completeness (Figs. 5, 6)
python scripts/generate_manuscript_figures.py # render all six figures
```

To regenerate **only** the figures from the cached intermediate outputs in `results/`:

```bash
python scripts/generate_manuscript_figures.py
```

This produces the six figures referenced in the paper:

| Figure  | File                                                   | Pipeline section it tests          |
|---------|--------------------------------------------------------|------------------------------------|
| Fig. 1  | `results/manuscript_figures/fig1_spacer_confound.pdf`  | Spacer-composition confound        |
| Fig. 2  | `results/manuscript_figures/fig2_positive_control.pdf` | Synthetic grammar + helical scan   |
| Fig. 3  | `results/manuscript_figures/fig3_gsi_census.pdf`       | SF-GSI census + correction cascade |
| Fig. 4  | `results/manuscript_figures/fig4_compositionality.pdf` | Variance decomposition             |
| Fig. 5  | `results/manuscript_figures/fig5_transfer.pdf`         | Cross-dataset transfer             |
| Fig. 6  | `results/manuscript_figures/fig6_completeness.pdf`     | Matched-vocabulary MPRA pairs      |

Headline numerical claims (e.g. 89.7% billboard CI, 0.12% post-FDR, 363 matched-vocabulary pairs) are recoverable from the JSON files under `results/critique/`, `results/sf_gsi/`, and `results/v3/`.

## Reproducibility notes

- Random seeds are fixed; per-enhancer N = 30 permutations for SF-GSI.
- Probe training: Adam, weight decay 1e-3, 5,000 training sequences per dataset. Full hyperparameters in `scripts/train_probes.py`.
- Sensitivity analyses (FIMO threshold, JASPAR version, permutation count) are produced by `scripts/run_validation_analyses.py`; the post-FDR billboard fraction lies in [88.4%, 92.1%] across all settings.

## License

MIT (see `LICENSE`). Code only — MPRA datasets and pretrained foundation-model weights remain under their original licenses.

## Anonymization notice

This repository is the anonymized supplementary code for a NeurIPS 2026 double-blind submission. Author names, affiliations, and identifying URLs have been removed.
