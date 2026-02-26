# GRAMLANG Experiment Log

**Project**: GRAMLANG - Decoding the Computational Grammar of Gene Regulation
**Started**: 2026-02-02
**Status**: **COMPLETE**

---

## Setup

### Environment
- **Conda environment**: `gramlang` (Python 3.10)
- **GPU**: 4x NVIDIA A100 80GB PCIe, CUDA 12.4
- **System**: Rocky Linux 9.6, 7TB RAID (/home), ~1.7TB free
- **Key packages**: PyTorch 2.1.0+cu121, transformers 4.36.0, numpy 1.26.2, pandas 2.1.4

### Directory Structure
```
grammar/
  data/
    raw/           # Raw MPRA downloads
    processed/     # Preprocessed parquet files
    motifs/        # JASPAR + yeast motif databases
    probes/        # Trained expression probes
  src/
    models/        # Model loaders, expression probes
    perturbation/  # Vocabulary-preserving shuffles, motif scanning
    grammar/       # GSI, rule extraction, compositionality
    transfer/      # Cross-species transfer
    decomposition/ # Biophysics, TF structure
  scripts/         # Pipeline runners
  results/         # All experiment outputs
```

---

## Data

### MPRA Datasets

| Dataset | Sequences | Species | Cell Type | Expression Range |
|---------|-----------|---------|-----------|------------------|
| Vaishnav et al. 2022 | 200,000 | Yeast | - | [0.0, 17.0] |
| Klein et al. 2020 | 2,275 | Human | HepG2 | log2 RNA/DNA |
| Agarwal et al. 2025 | 113,386 | Human | K562 | [-1.99, 3.27] |
| de Almeida / Inoue 2019 | 2,453 | Human | hESC neural | [0.39, 4.04] |
| Jores et al. 2021 | 76,177 | Plant | 3 species | [-7.25, 4.94] |
| Georgakopoulos-Soares (positive control) | 209,440 | Human | K562/HepG2 | Controlled |

### Motif Databases
- JASPAR 2024 Vertebrates: 879 motifs
- JASPAR 2024 Plants: 805 motifs
- Yeast motifs (JASPAR fungi + YeTFaSCo): 371 motifs
- Motif scanning: FIMO v5.5.7, p < 1e-4

### Models

| Model | Status | Parameters | Architecture |
|-------|--------|------------|--------------|
| DNABERT-2 | OK | 117M | Transformer |
| NT v2-500M | OK | 498M | Transformer |
| HyenaDNA large-1M | OK | 6.5M | SSM |
| Enformer | OK | 251M | Transformer |
| PARM (K562, HepG2) | OK | - | CNN (MPRA-native) |
| Caduceus | Failed | - | SSM (mamba incompatible) |

---

## Expression Probe Training

### All Model-Dataset Probes

| Model | Dataset | Pearson r | R² | Viable (R²>0.05)? |
|-------|---------|-----------|-----|-------------------|
| DNABERT-2 | agarwal | 0.340 | 0.116 | Yes |
| DNABERT-2 | de_almeida | 0.234 | 0.055 | No |
| DNABERT-2 | jores | **0.580** | **0.336** | Yes |
| DNABERT-2 | klein | 0.260 | 0.068 | No |
| DNABERT-2 | vaishnav | 0.472 | 0.223 | Yes |
| NT v2-500M | agarwal | 0.331 | 0.109 | Yes |
| NT v2-500M | de_almeida | 0.286 | 0.082 | No |
| NT v2-500M | jores | **0.575** | **0.331** | Yes |
| NT v2-500M | klein | 0.331 | 0.110 | Yes |
| NT v2-500M | vaishnav | **0.551** | **0.304** | Yes |
| HyenaDNA | agarwal | 0.168 | 0.028 | No |
| HyenaDNA | de_almeida | 0.174 | 0.030 | No |
| HyenaDNA | jores | 0.507 | 0.257 | Yes |
| HyenaDNA | klein | 0.239 | 0.057 | No |
| HyenaDNA | vaishnav | 0.418 | 0.175 | Yes |

**9/15 probes viable** (R²>0.05). Jores (plant) probes are strongest after vaishnav.

---

## Grammar Sensitivity Census

### GSI Distribution

**7,650 measurements**: 3 foundation models × 5 datasets × 500 enhancers + Enformer × 3 human datasets × 50 enhancers

| Dataset | Species | Median GSI | Mean GSI | Frac Significant (p<0.05) |
|---------|---------|------------|----------|---------------------------|
| Klein | Human (HepG2) | **0.611** | 2.192 | 7.1% |
| Agarwal | Human (K562) | **0.328** | 1.726 | 8.7% |
| Jores | Plant | **0.118** | 0.127 | 10.4% |
| Vaishnav | Yeast | **0.084** | 0.081 | 6.9% |
| de Almeida | Human (neural) | **0.044** | 0.067 | 8.3% |

### Statistical Correction

| Threshold | Count | Percentage |
|-----------|-------|------------|
| GSI > 0 | 7,550/7,550 | 100% |
| z-score p < 0.05 | 625/7,550 | **8.3%** |
| FDR q < 0.05 | 13/7,550 | **0.17%** |

### ANOVA: Sources of Variance

| Factor | η² | F | p-value |
|--------|-----|---|---------|
| **Dataset** | **0.290** | 859.2 | < 1e-300 |
| Model | 0.045 | 264.5 | 1.0e-111 |
| Residual | 0.632 | — | — |

Dataset explains 6.5× more variance than model architecture.

### Cross-Model Agreement

| Dataset | DNABERT-2 vs NT | DNABERT-2 vs HyenaDNA | NT vs HyenaDNA |
|---------|-----------------|----------------------|----------------|
| Agarwal | ρ = 0.90 | ρ = 0.70 | ρ = 0.75 |
| Jores | ρ = 0.89 | ρ = 0.65 | ρ = 0.70 |
| Klein | ρ = 0.88 | ρ = 0.66 | ρ = 0.67 |
| Vaishnav | ρ = 0.56 | ρ = -0.03 | ρ = -0.08 |
| de Almeida | ρ = -0.06 | ρ = -0.16 | ρ = 0.05 |

### Enformer Results

| Dataset | n | Median GSI | Frac Significant |
|---------|---|-----------|-----------------|
| Agarwal | 50 | 0.446 | 14.0% |
| de Almeida | 50 | 0.450 | 2.0% |
| Klein | 50 | 0.434 | 4.0% |

---

## Spacer Confound Discovery

### Factorial Decomposition

Four shuffle types isolating individual factors:

| Factor | Agarwal Var | Jores Var | de Almeida Var |
|--------|-------------|-----------|----------------|
| Position (motif order) | 0.0026 | 0.0373 | 0.0088 |
| Orientation (strand flips) | 0.0015 | 0.0214 | 0.0045 |
| **Spacer (DNA reshuffling)** | **0.0053** | **0.1043** | **0.0166** |
| Full (all combined) | 0.0066 | 0.1296 | 0.0190 |

**Fraction of variance explained:**

| Factor | Agarwal | Jores | de Almeida | Average |
|--------|---------|-------|------------|---------|
| **Spacer** | **82.8%** | **78.4%** | **85.7%** | **82.3%** |
| Position | 42.5% | 28.3% | 47.2% | 39.3% |
| Orientation | 25.6% | 18.6% | 24.2% | 22.8% |

### Spacer Ablation

| Perturbation | Agarwal Δ | Jores Δ | de Almeida Δ |
|--------------|-----------|---------|--------------|
| random_replace | **0.149** | **0.546** | 0.115 |
| gc_shift | 0.102 | 0.346 | **0.121** |
| dinuc_shuffle | 0.045 | 0.239 | 0.113 |
| **motif_only** | **0.034** | **0.089** | **0.062** |

**Motif permutation has smallest effect** (Δ = 0.03-0.09).

### GC-Expression Correlation

| Dataset | r |
|---------|---|
| Jores (plant) | **-0.734** |
| Agarwal (K562) | **+0.658** |
| de Almeida | +0.215 |

Direction **reverses** across species.

---

## Positive Control

Georgakopoulos-Soares data with controlled spacers.

| Metric | Value |
|--------|-------|
| Orientation pairs tested | 500 |
| Mean |Δprediction| | **0.062** |
| Fraction |Δpred| > 0.1 | 17.0% |
| t-test | t = 30.86, **p = 9.54e-118** |

**Models detect grammar when spacers are controlled.**

---

## Feature Decomposition

### What Models Learn

| Dataset | GC Only R² | Dinuc R² | All Features R² |
|---------|------------|----------|-----------------|
| Agarwal | **0.40** | 0.47 | 0.48 |
| Jores | **0.59** | 0.74 | **0.80** |
| de Almeida | 0.08 | 0.11 | 0.16 |

Simple features explain 48-80% of model predictions.

### Variance Decomposition

| Feature Set | Agarwal R² | Jores R² | de Almeida R² |
|------------|-----------|---------|--------------|
| Vocabulary | -0.038 | -0.152 | -0.227 |
| Grammar | -0.083 | -0.153 | -0.223 |
| **DL Embeddings** | **0.079** | **0.265** | **0.026** |

Embeddings capture 16-42% more variance than grammar features.

---

## ANOVA Decomposition

### Vocabulary vs Grammar

| Dataset | Vocab η² | Grammar η² | Unexplained |
|---------|----------|-----------|-------------|
| Klein | **0.224** | 0.000 | 78.7% |
| Jores | 0.121 | **0.016** | 86.3% |
| Agarwal | 0.111 | 0.000 | 90.1% |
| de Almeida | 0.086 | 0.014 | 90.9% |
| Vaishnav | 0.083 | 0.000 | 94.2% |

Vocabulary explains 8-22%; grammar explains 0-1.6%.

### Bag-of-Motifs Baseline

| Dataset | BOM R² | Grammar R² | Increment |
|---------|--------|-----------|-----------|
| Klein | **0.130** | 0.111 | -0.019 |
| Agarwal | **0.095** | 0.076 | -0.018 |
| Jores | **0.097** | 0.089 | -0.009 |
| de Almeida | 0.039 | **0.059** | +0.021 |
| Vaishnav | -0.001 | -0.027 | -0.026 |

Grammar features **decrease** prediction in 4/5 datasets.

---

## PARM Comparison

| Dataset | Cell Type | Median GSI | Significant |
|---------|-----------|------------|-------------|
| Agarwal | K562 | 0.192 | 6.0% |
| Klein | HepG2 | 0.448 | 7.0% |

Matches foundation model significance rates (6-7% vs 8.3%).

---

## Power Analysis

| Dataset | 100 shuffles | 1000 shuffles |
|---------|-------------|---------------|
| Agarwal | 10.0% | **11.0%** |
| Jores | 10.0% | **9.0%** |

10× more shuffles → same significance rate. Grammar rarity is NOT underpowering.

---

## Grammar Rule Extraction

### Rule Database

| Metric | Value |
|--------|-------|
| Total rules | 26,922 |
| Unique motif pairs | 5,032 |
| Mean fold change | 1.622 |
| Helical phasing (>2.0) | 3,632 (13.5%) |

### Cross-Model Consensus

| Metric | Value |
|--------|-------|
| Mean consensus | 0.521 |
| Orientation agreement | 85.1% |
| Spacing correlation | 0.479 |
| High consensus (>0.8) | 7.7% |

---

## Compositionality

| k (motifs) | Mean Gap | Mean R² |
|------------|----------|---------|
| 3 | 0.992 | 0.008 |
| 4 | 0.989 | 0.011 |
| 5 | 0.988 | 0.012 |
| 6 | 0.989 | 0.011 |

**Gap ≈ 0.99**: pairwise rules explain only ~1%. Grammar is context-sensitive.

---

## Cross-Species Transfer

| Source | Target | Transfer R² |
|--------|--------|-------------|
| Human | Human | **0.151** |
| Human | Yeast | 0.000 |
| Human | Plant | 0.000 |
| Plant | Plant | **0.212** |
| Yeast | Yeast | 0.004 |

**Zero transfer between species.**

---

## Biophysical Determinants

| Dataset | Biophysics R² | Top Feature |
|---------|--------------|-------------|
| Jores (plant) | **0.789** | DNA Roll (59%) |
| Klein (HepG2) | **0.375** | Minor Groove Width (16%) |
| Vaishnav (yeast) | 0.218 | CpG content (16%) |
| Agarwal (K562) | 0.062 | CpG content (21%) |
| de Almeida | -0.488 | Not predictable |

---

## Attention Analysis

NT v2-500M attention patterns:

| Metric | Value |
|--------|-------|
| Total heads | 464 |
| Grammar heads | **101 (21.8%)** |
| Mean enrichment | **2.99×** |
| Layer concentration | L15-L28 |

---

## Statistical Validation

| Claim | Test | Result |
|-------|------|--------|
| Grammar (corrected) | z-score | 8.3% significant |
| Grammar (FDR) | Benjamini-Hochberg | 0.17% significant |
| Spacer dominance | Factorial ANOVA | 78-86% of variance |
| Positive control | t-test | p < 1e-117 |
| Context-sensitive | BIC | Gap = 0.989 constant |
| Cross-species | Linear regression | R² = 0.000 |
| Architecture-independent | PARM comparison | 6-7% significant |

---

## Output Files

### Results Directory Structure
```
results/
  module1/           # GSI census
  module2/           # Grammar rules
  module3/           # Compositionality
  module4/           # Cross-species transfer
  module5/           # Biophysics
  module6/           # Completeness
  v3/                # Extension experiments
    bom_baseline/
    factorial_decomposition/
    feature_decomposition/
    parm_comparison/
    positive_control/
    power_analysis/
    spacer_ablation/
    variance_decomposition/
```

---

## Key Conclusions

1. **Grammar is real but masked** by spacer confound in standard methodology
2. **78-86% of GSI variance** is spacer composition, not motif arrangement
3. **GC and dinucleotides explain 40-80%** of model predictions
4. **Only 0.17%** of enhancers survive FDR correction
5. **Billboard model confirmed**: motif identity matters, arrangement does not
6. **Architecture-independent**: CNN and transformers show same patterns
7. **Species-specific**: GC correlation reverses between human and plant

---

*Last Updated: 2026-02-21*
