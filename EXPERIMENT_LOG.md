# GRAMLANG Experiment Log

**Project**: GRAMLANG - Decoding the Computational Grammar of Gene Regulation
**Started**: 2026-02-02
**Last Updated**: 2026-02-05
**Status**: **COMPLETE** (v1: 6 Modules; v2: 14 Components; v3: 8 Experiments)

---

## Setup Phase

### Environment Setup
- **Date**: 2026-02-02
- **Status**: Complete
- **Conda environment**: `gramlang` (Python 3.10)
- **GPU available**: 4x NVIDIA A100 80GB PCIe, CUDA 12.4
- **System**: Rocky Linux 9.6, 7TB RAID (/home), ~1.7TB free
- **Disk budget**: 150-200GB max at any point
- **Current disk usage**: 1.7 GB (1% of budget)
- **Key packages**: PyTorch 2.1.0+cu121, transformers 4.36.0, numpy 1.26.2, pandas 2.1.4

### Directory Structure
```
grammar/
  data/
    raw/           # Raw MPRA downloads
    processed/     # Preprocessed parquet files
    motifs/        # JASPAR + yeast motif databases
    embeddings_cache/  # Cached model embeddings
    probes/        # Trained expression probes
  src/
    models/        # Model loaders, expression probes
    perturbation/  # Vocabulary-preserving shuffles, motif scanning
    grammar/       # GSI, rule extraction, compositionality, complexity
    transfer/      # Cross-species transfer, phylogenetics
    decomposition/ # Biophysics, TF structure, phase diagrams
    design/        # Grammar-optimized design, completeness
    utils/         # Sequence utils, statistics, I/O, visualization
  scripts/         # Pipeline runner, preprocessing, probe training
  results/         # All experiment outputs
```

---

## Data Acquisition

### MPRA Datasets

| Dataset | Status | Sequences | Species | Cell Type | Expression Range | Location |
|---------|--------|-----------|---------|-----------|-----------------|----------|
| Dataset | Status | Sequences | Species | Cell Type | Expression Range | Location |
|---------|--------|-----------|---------|-----------|-----------------|----------|
| Vaishnav et al. 2022 | **Complete** | 200,000 (subsampled) | Yeast | - | [0.0, 17.0] | `data/processed/vaishnav2022.parquet` |
| Klein et al. 2020 | **Complete** | 2,275 | Human | HepG2 | log2 RNA/DNA | `data/processed/klein2020.parquet` |
| Agarwal et al. 2025 | **Complete** | 113,386 | Human | K562 | [-1.99, 3.27] | `data/processed/agarwal2023.parquet` |
| de Almeida / Inoue 2019 | **Complete** | 2,453 | Human | hESC neural | [0.39, 4.04] | `data/processed/de_almeida2024.parquet` |
| Jores et al. 2021 | **Complete** | 76,177 | Plant (3 spp) | - | [-7.25, 4.94] | `data/processed/jores2021.parquet` |
| Kircher et al. 2019 | **Skipped** | ~30K variants | Human | K562 | - | Saturation mutagenesis format |

**Pipeline-Ready Data** (with FIMO v5.5.7 motif annotations, p < 1e-4):

| Dataset | Sequences | With >=2 Motifs | Mean Motifs/Seq | Max Motifs/Seq | Motif DB Used |
|---------|-----------|-----------------|-----------------|----------------|---------------|
| vaishnav | 5,000 | 4,225 (84.5%) | 3.4 | 16 | yeast_motifs.meme (top 200) |
| klein | 2,275 | 2,253 (99.0%) | 14.4 | 72 | JASPAR2024_vertebrates (top 200) |
| agarwal | 5,000 | 4,860 (97.2%) | 9.5 | 56 | JASPAR2024_vertebrates (top 200) |
| de_almeida | 2,453 | 2,425 (98.9%) | 12.3 | 49 | JASPAR2024_vertebrates (top 200) |
| jores | 5,000 | 4,280 (85.6%) | 9.4 | 121 | JASPAR2024_plants (top 200) |

### Motif Databases

| Database | Status | Motifs | Size | Location |
|----------|--------|--------|------|----------|
| JASPAR 2024 Vertebrates | Complete | 879 | 459 KB | `data/motifs/JASPAR2024_vertebrates.meme` |
| JASPAR 2024 Plants | Complete | 805 | 402 KB | `data/motifs/JASPAR2024_plants.meme` |
| JASPAR 2024 Insects | Complete | 286 | 134 KB | `data/motifs/JASPAR2024_insects.meme` |
| Yeast motifs (JASPAR fungi + YeTFaSCo) | Complete | 371 | 198 KB | `data/motifs/yeast_motifs.meme` |

**Motif scanning note**: FIMO v5.5.7 (MEME Suite) compiled from source and installed at `tools/meme/bin/fimo`. Used for all motif scanning with p-value threshold < 1e-4. The auto-detection in `MotifScanner._verify_fimo()` checks `tools/meme/bin/fimo` first, then falls back to PATH. A vectorized numpy-based PWM scanner (score_fraction >= 0.65) is available as fallback when FIMO is unavailable.

**FIMO vs Fallback Scanner Comparison** (re-run 2026-02-03):

| Dataset | Fallback (score_fraction=0.65) | FIMO (p<1e-4) | Change |
|---------|-------------------------------|---------------|--------|
| Vaishnav (yeast) | 46,684 hits (9.3/seq) | 16,895 hits (3.4/seq) | -64% (fewer false positives) |
| Klein (human) | 10,389 hits (4.6/seq) | 32,821 hits (14.4/seq) | +216% (more true positives) |

FIMO gives statistically calibrated p-values and more reliable motif calls, especially for the vertebrate JASPAR database where the score_fraction threshold was either too permissive (yeast) or too restrictive (human).

### Pretrained Models

| Model | Status | Size (FP32) | Params | Architecture | Hidden Dim | Layers | Notes |
|-------|--------|-------------|--------|--------------|-----------|--------|-------|
| Enformer | **OK** | 958 MB | 251M | Transformer | 1536 | 11 | Built-in expression head (CAGE tracks) |
| NT v2-500M | **OK** | 1,901 MB | 498M | Transformer | 1024 | 25 | Expression via trained probe |
| DNABERT-2 | **OK** | 447 MB | 117M | Transformer (BERT) | 768 | 12 | Custom model class, PyTorch attn fallback |
| HyenaDNA large-1M | **OK** | 25 MB | 6.5M | SSM (Hyena) | 256 | 10 | Expression via trained probe |
| Caduceus | **Failed** | - | - | SSM (Mamba) | - | - | mamba_ssm/causal_conv1d incompatible with PyTorch 2.1 |
| GPN | Pending | - | - | CNN | - | - | Not yet tested |
| Borzoi | Deferred | - | - | Transformer | - | - | Complex setup |
| Sei | Deferred | - | - | CNN | - | - | Complex setup |
| Evo 7B | Deferred | - | - | SSM | - | - | Very large |

**Known Issues**:
- **DNABERT-2**: AutoModel.from_pretrained fails (config_class mismatch). Fixed via `get_class_from_dynamic_module('bert_layers.BertModel', ...)`. Triton flash attention disabled due to API incompatibility; uses PyTorch attention fallback.
- **Caduceus**: causal_conv1d 1.6.0 requires `torch.library.custom_op` (PyTorch >= 2.4). causal_conv1d 1.4.0 + mamba_ssm 2.2.2 install pulled in PyTorch 2.10 + numpy 2.x, breaking the environment. Reverted to PyTorch 2.1.0+cu121. Caduceus excluded from current experiments.

---

## Expression Probe Training

**Date**: 2026-02-03
**Method**: 2-layer MLP (input_dim -> 256 -> 1) with ReLU + dropout(0.1), trained on frozen embeddings
**Training**: 80/10/10 split, AdamW (lr=1e-3, wd=1e-4), MSE loss, early stopping (patience=10)
**Dataset**: Vaishnav 2022 (5,000 subsampled sequences, yeast promoters)

| Model | Embedding Dim | Pearson r | Spearman rho | R^2 | Viable (r>0.3)? | Best Epoch | Date |
|-------|--------------|-----------|-------------|-----|-----------------|------------|------|
| NT v2-500M | 1024 | **0.5513** | **0.5595** | **0.3039** | **Yes** | 95 (early stop) | 2026-02-03 |
| DNABERT-2 | 768 | 0.4722 | 0.4760 | 0.2229 | Yes | 100 | 2026-02-03 |
| HyenaDNA | 256 | 0.4181 | 0.4120 | 0.1748 | Yes | 100 | 2026-02-03 |

**Key findings**:
- All 3 foundation models produce embeddings that predict yeast expression (all viable, r > 0.3 threshold met)
- NT v2-500M has the best expression prediction (R^2=0.30), likely due to its larger model size (498M params) and richer embeddings (1024-dim)
- DNABERT-2 is intermediate (R^2=0.22) despite being the smallest model (117M params)
- HyenaDNA has the weakest prediction (R^2=0.17), possibly because its SSM architecture captures different features than transformers, or its small model size (6.5M params) limits representation capacity

**Probe files**: `data/probes/{model}_vaishnav2022_probe.pt` + `*_probe_metrics.json`

---

## Module 1: Grammar Existence & Complexity Classification

**Date**: 2026-02-03 (re-run with FIMO v5.5.7)
**Status**: **COMPLETE**
**Runtime**: ~25 minutes (3 models x 5 datasets x 200 enhancers x 50 shuffles)

### Experiment 1.1: Grammar Sensitivity Census

**Parameters**: n_shuffles=50, min_motifs=2, max_enhancers=200 per (model, dataset) pair
**Motif scanning**: FIMO v5.5.7, p < 1e-4, top 200 motifs per database

**Total GSI measurements**: 3,000 (200 enhancers x 5 datasets x 3 models)

#### Per-Model, Per-Dataset Results

| Model | Dataset | n | Mean GSI | Median GSI | Std GSI | Max GSI | Frac Sig (p<0.05) |
|-------|---------|---|----------|------------|---------|---------|-------------------|
| DNABERT-2 | agarwal (human K562) | 200 | 0.0796 | 0.0793 | 0.0165 | 0.1344 | 100% |
| DNABERT-2 | de_almeida (human neural) | 200 | 0.0747 | 0.0760 | 0.0147 | 0.1122 | 100% |
| DNABERT-2 | jores (plant) | 200 | 0.0757 | 0.0732 | 0.0177 | 0.1469 | 100% |
| DNABERT-2 | klein (human HepG2) | 200 | 0.0802 | 0.0812 | 0.0154 | 0.1249 | 100% |
| DNABERT-2 | vaishnav (yeast) | 200 | 0.0896 | 0.0895 | 0.0133 | 0.1508 | 100% |
| NT v2-500M | agarwal (human K562) | 200 | **0.0904** | **0.0903** | 0.0233 | 0.1612 | 100% |
| NT v2-500M | de_almeida (human neural) | 200 | 0.0782 | 0.0782 | 0.0175 | 0.1339 | 100% |
| NT v2-500M | jores (plant) | 200 | 0.0830 | 0.0777 | 0.0232 | **0.2231** | 100% |
| NT v2-500M | klein (human HepG2) | 200 | **0.1031** | **0.1024** | **0.0295** | 0.2124 | 100% |
| NT v2-500M | vaishnav (yeast) | 200 | 0.0909 | 0.0874 | 0.0158 | 0.1698 | 100% |
| HyenaDNA | agarwal (human K562) | 200 | 0.0442 | 0.0434 | 0.0092 | 0.0723 | 100% |
| HyenaDNA | de_almeida (human neural) | 200 | 0.0456 | 0.0442 | 0.0109 | 0.0939 | 100% |
| HyenaDNA | jores (plant) | 200 | 0.0545 | 0.0476 | 0.0241 | 0.1852 | 100% |
| HyenaDNA | klein (human HepG2) | 200 | 0.0462 | 0.0446 | 0.0101 | 0.0782 | 100% |
| HyenaDNA | vaishnav (yeast) | 200 | 0.0578 | 0.0567 | 0.0103 | 0.1006 | 100% |

#### Per-Dataset Summary (All Models Combined)

| Dataset | Species | Mean GSI | Median GSI | Frac Significant |
|---------|---------|----------|------------|-----------------|
| vaishnav | Yeast | 0.0794 | 0.0808 | 100% |
| klein | Human (HepG2) | 0.0765 | 0.0737 | 100% |
| agarwal | Human (K562) | 0.0714 | 0.0711 | 100% |
| jores | Plant | 0.0711 | 0.0688 | 100% |
| de_almeida | Human (neural) | 0.0662 | 0.0652 | 100% |

**Key observation**: NT v2-500M consistently shows the highest grammar sensitivity across all 5 datasets, while HyenaDNA shows the lowest. This pattern is robust across all 4 species (yeast, human, plant). Grammar sensitivity is remarkably consistent across species (mean GSI 0.066-0.079), suggesting grammar effects are a universal property of regulatory sequences.

#### GSI Distribution (All Models Combined)

| Threshold | Count | Percentage |
|-----------|-------|------------|
| GSI > 0.01 | 3,000/3,000 | **100.0%** |
| GSI > 0.05 | 2,350/3,000 | **78.3%** |
| GSI > 0.10 | 391/3,000 | 13.0% |
| GSI > 0.15 | 23/3,000 | 0.8% |
| GSI > 0.20 | 2/3,000 | 0.1% |
| GSI > 0.30 | 0/3,000 | 0.0% |

**Interpretation**: Grammar effects are **universally detectable** (100% significant at p<0.05 across all 3,000 measurements) but **moderate** in magnitude. The mean GSI of ~0.07 means that vocabulary-preserving shuffles alter predicted expression by ~7% of the mean. This is consistent with a "flexible billboard" model where motif arrangement matters but is not the dominant factor.

According to the research plan decision tree:
- **Not Scenario A** (GSI > 0.3 for >50%): GSI values are well below 0.3
- **Partially Scenario B** (billboard-like): Grammar exists but is weak
- The 100% significance rate means grammar is NOT zero — arrangement genuinely matters, just modestly
- NT achieves GSI > 0.20 in 2 cases (jores plant, klein human), hinting at occasional strong grammar effects

### Experiment 1.2: Grammar Information Content

| Model | Dataset | Bits of Grammar | Entropy (shuffles) | Grammar Specificity |
|-------|---------|----------------|--------------------|--------------------|
| DNABERT-2 | vaishnav | 1.850 | 3.836 | 1.229 |
| NT v2-500M | vaishnav | 1.715 | 3.853 | 0.992 |
| HyenaDNA | vaishnav | 1.309 | 3.862 | 0.628 |
| DNABERT-2 | klein | 1.751 | 3.835 | 1.196 |
| NT v2-500M | klein | 1.509 | 3.854 | 0.816 |
| HyenaDNA | klein | 1.738 | 3.874 | 1.036 |

**Interpretation**: Grammar carries approximately **1.3-1.9 bits of information per enhancer** depending on model and dataset. This means arrangement provides distinguishable information, but the shuffle distribution has high entropy (~3.8 bits), meaning many arrangements produce similar expression levels.

Grammar specificity (z-score of original vs shuffle distribution) ranges from 0.6 to 1.2, indicating the natural arrangement is typically 0.6-1.2 standard deviations above the shuffle mean — detectable but not extreme.

### Experiment 1.3: Formal Complexity Classification
- **Status**: Pending (requires Module 3 compositionality data first)
- Will be computed after Module 3 generates compositionality gap vs. motif count data

### Module 1 Output Files

| File | Description |
|------|-------------|
| `results/module1/all_gsi_results.parquet` | Combined GSI for all 1,200 measurements |
| `results/module1/gsi_summary.json` | Summary statistics per dataset |
| `results/module1/grammar_information.parquet` | Information content metrics |
| `results/module1/fig1_gsi_distribution.pdf` | Figure 1: GSI distribution plot |
| `results/module1/{dataset}_{model}_gsi.parquet` | Per-model, per-dataset GSI files |

---

## Module 2: Cross-Architecture Grammar Rule Extraction

**Date**: 2026-02-03
**Status**: **COMPLETE**
**Runtime**: ~15 minutes

### Experiment 2.1: Causal Grammar Rule Extraction

**Parameters**: models=[dnabert2, nt, hyenadna], max_enhancers_rules=100, spacing_range=2-50bp, 5 datasets
**Method**: For each grammar-sensitive enhancer (GSI > 0.05), extract pairwise motif interaction rules via spacing scan (49 positions), orientation scan (4 configurations), and helical phase scoring (FFT at 10.5bp periodicity).

**Results**: 26,922 grammar rules extracted (6.4× more than initial run with FIMO motif scanning)

| Metric | Value |
|--------|-------|
| Total rules | 26,922 |
| Rules per model | 8,974 each |
| Unique motif pairs | 5,032 |
| Enhancers tested | 500 |
| Mean fold change | 1.622 |
| Mean spacing sensitivity | 1.091 |
| Rules with helical phasing (score > 2.0) | 3,632 (13.5%) |
| Optimal spacing range | 2-50bp |

**Per-Dataset Rule Counts**:
| Dataset | Rules | Notes |
|---------|-------|-------|
| Klein (human HepG2) | 8,688 | Most rules — highest motif density (14.4/seq) |
| de Almeida (human neural) | 6,600 | Many rules despite small dataset (2,453 seqs) |
| Agarwal (human K562) | 5,139 | Good coverage |
| Jores (plant) | 4,473 | Plant grammar rules |
| Vaishnav (yeast) | 2,022 | Fewer with FIMO (fewer yeast motifs pass p-threshold) |

### Experiment 2.2: Cross-Model Grammar Consensus

| Metric | Value |
|--------|-------|
| Mean consensus score | **0.521** |
| Median consensus score | 0.510 |
| High consensus rules (>threshold) | **7.7%** |
| Contested rules | **2.9%** |
| Mean spacing correlation (across models) | **0.479** |
| Orientation agreement | **85.1%** |
| Rules with 3-model consensus | 8,974 |
| Enhancers tested | 500 |

**Interpretation**: Improved cross-architecture consensus with FIMO-based motif scanning (0.521 vs 0.482 previously). Models agree on orientation preferences 85.1% of the time and show better spacing correlation (r=0.479 vs 0.399). Contested rules dropped from 6.3% to 2.9%, suggesting FIMO's statistically calibrated motif calls reduce noise in rule extraction. The 7.7% high-consensus rules represent robust grammar rules agreed upon by all 3 architectures.

### Experiment 2.3: Grammar Representation Geometry
- **Status**: Not started (optional)

### Experiment 2.4: Evo Naturalness Oracle
- **Status**: Not started (Evo model deferred)

### Module 2 Output Files

| File | Description |
|------|-------------|
| `results/module2/grammar_rules_database.parquet` | All 4,218 rules with spacing/orientation profiles |
| `results/module2/consensus_scores.parquet` | Cross-model consensus for 1,406 motif pairs |
| `results/module2/global_consensus.json` | Global consensus summary |

---

## Module 3: Compositionality Testing

**Date**: 2026-02-03
**Status**: **COMPLETE**
**Runtime**: ~6 minutes

### Experiment 3.1: Pairwise -> Higher-Order Prediction

**Parameters**: n_arrangements=100, target_k=[3,4,5,6], max_per_k=20, models=[dnabert2, nt, hyenadna], 5 datasets
**Total tests**: 1,200 (3 models × 5 datasets × 4 k values × 20 enhancers)

#### Compositionality Gap by Motif Count

| k (motifs) | n tests | Mean Gap | Mean Pairwise R² | Mean Pairwise Corr |
|------------|---------|----------|-------------------|-------------------|
| 3 | 135 | 0.992 | 0.008 | 0.012 |
| 4 | 303 | 0.989 | 0.011 | 0.011 |
| 5 | 306 | 0.988 | 0.012 | -0.003 |
| 6 | 321 | 0.989 | 0.011 | -0.001 |
| 7 | 135 | 0.991 | 0.009 | 0.011 |

#### Per-Model Summary

| Model | Mean Gap | Mean R² |
|-------|----------|---------|
| DNABERT-2 | 0.989 | 0.011 |
| NT v2 | 0.990 | 0.010 |
| HyenaDNA | 0.989 | 0.011 |

### Experiment 3.2: Formal Complexity Classification

| Metric | Value |
|--------|-------|
| **Classification** | **Context-Sensitive** |
| Mean compositionality gap | 0.990 |
| Gap range | [0.988, 0.992] |
| Best BIC model | Constant (gap does NOT grow with k) |
| Confidence | 0.530 |

**Interpretation**: The compositionality gap is approximately 0.99 across all motif counts (k=3-7), meaning pairwise grammar rules explain only ~1% of arrangement effects. This is a **strongly non-compositional** result:

1. **Context-sensitive grammar**: Individual pairwise motif interactions (spacing, orientation preferences) do NOT compose to predict higher-order arrangement effects. The full motif context matters.

2. **Constant gap**: The compositionality gap does NOT increase with k (more motifs), suggesting that non-compositionality is a fundamental property, not a scaling artifact. BIC favors the constant model over linear or exponential growth.

3. **Model agreement**: All 3 architectures (DNABERT-2, NT, HyenaDNA) show identical non-compositionality, indicating this is a genuine property of the grammar, not a model artifact.

4. **Chomsky classification**: The constant, near-1 gap with no growth pattern places regulatory grammar at least at the context-sensitive level in the Chomsky hierarchy. Grammar rules are inherently positional and context-dependent.

### Module 3 Output Files

| File | Description |
|------|-------------|
| `results/module3/compositionality_results.parquet` | All 1,200 compositionality tests |
| `results/module3/complexity_classification.json` | Formal complexity classification |
| `results/module3/fig2_compositionality.pdf` | Figure 2: Compositionality gap curve |

### Experiment 3.3: Grammar Tensor Factorization
- **Status**: Not started (optional)

---

## Module 4: Cross-Species Grammar Transfer

**Date**: 2026-02-03
**Status**: **COMPLETE**
**Runtime**: ~2 minutes

### Experiment 4.1: Cross-Species Grammar Transfer

**Method**: Train linear model on grammar rule features (spacing sensitivity, orientation sensitivity, fold change) from one species' rules, predict expression effects in another species. 50 enhancers tested per direction, 100 arrangements each.

| Source | Target | Transfer R² | Transfer Corr | Enhancers |
|--------|--------|-------------|---------------|-----------|
| Human | Human | **0.151** | -0.344 | 50 |
| Human | Yeast | 0.000 | 0.000 | 50 |
| Human | Plant | 0.000 | 0.000 | 50 |
| Yeast | Human | 0.000 | 0.000 | 50 |
| Yeast | Yeast | 0.004 | -0.135 | 50 |
| Yeast | Plant | 0.000 | 0.000 | 50 |
| Plant | Human | 0.000 | 0.000 | 50 |
| Plant | Yeast | 0.000 | 0.000 | 50 |
| Plant | Plant | **0.212** | -0.345 | 50 |

### Experiment 4.2: Grammar Phylogenetics

| Species Pair | Distance |
|-------------|----------|
| Human ↔ Yeast | 1.000 (maximum) |
| Human ↔ Plant | 1.000 (maximum) |
| Yeast ↔ Plant | 1.000 (maximum) |

**Interpretation**: Grammar does NOT transfer between species. All cross-species transfer R² values are exactly 0 across the 3×3 matrix. Within-species transfer varies markedly:

1. **Human within-species (R²=0.151)**: Grammar rules from human enhancers have moderate predictive power for other human enhancers, suggesting some universal grammar patterns in the human regulatory code.

2. **Plant within-species (R²=0.212)**: Strongest within-species transfer, likely because the Jores dataset contains promoters from 3 related plant species (Arabidopsis, maize, sorghum) that share conserved regulatory grammar.

3. **Yeast within-species (R²=0.004)**: Near-zero transfer even within yeast. The Vaishnav dataset contains synthetic random promoters, which may lack the conserved motif pair arrangements found in natural enhancers.

4. **Negative correlations**: All within-species transfers show negative correlations, meaning grammar rules anti-predict direction of effect — the rules capture sensitivity (variance) but not the sign of the arrangement effect.

5. **Zero cross-species**: Grammar is completely species-specific. Rules from one kingdom are entirely uninformative for another.

### Module 4 Output Files

| File | Description |
|------|-------------|
| `results/module4/transfer_matrix.parquet` | 2×2 transfer R² matrix |
| `results/module4/grammar_phylogeny.json` | Species distance matrix and linkage |
| `results/module4/fig4_transfer_heatmap.pdf` | Figure 4: Transfer heatmap |

---

## Module 5: Causal Determinants of Grammar

**Date**: 2026-02-03
**Status**: **COMPLETE**
**Runtime**: ~3 minutes

### Experiment 5.1: Biophysics Residual

**Method**: Random forest regression predicting grammar sensitivity (GSI) from 35 biophysical features: GC content, all 16 dinucleotide frequencies, DNA shape features (MGW, Roll, ProT, HelT — mean, std, min, max), bendability, nucleosome poly-AT. 5-fold cross-validation.

| Dataset | Biophysics R² | Std | Top Feature | Feature Weight |
|---------|--------------|-----|-------------|----------------|
| Vaishnav (yeast) | **0.082** | 0.152 | CG dinucleotide | 0.123 |
| Klein (human) | **0.636** | 0.155 | CG dinucleotide | 0.315 |

**Top-5 Features (Klein)**:
1. CG dinucleotide frequency (0.315)
2. Mean Minor Groove Width (0.143)
3. Mean Propeller Twist (0.098)
4. GC dinucleotide frequency (0.048)
5. MGW standard deviation (0.042)

**Top-5 Features (Vaishnav)**:
1. CG dinucleotide frequency (0.123)
2. Mean Minor Groove Width (0.111)
3. GG dinucleotide frequency (0.057)
4. Roll standard deviation (0.055)
5. Mean Roll (0.054)

**Interpretation**: Biophysics explains strikingly different amounts of grammar across species:
- **Klein (human)**: 63.6% of grammar variance explained by biophysical features, dominated by CpG dinucleotide content and minor groove width. This suggests human enhancer grammar is largely driven by DNA structural properties.
- **Vaishnav (yeast)**: Only 8.2% explained, indicating yeast grammar is primarily determined by factors beyond simple biophysics — likely TF-TF protein interactions, cooperative binding, and chromatin context.

### Experiment 5.2: TF Structural Class Mapping

| Metric | Value |
|--------|-------|
| Classification accuracy | 43.5% |
| Baseline (majority class) | 42.9% |
| Improvement | 0.7% |
| Rules analyzed | 147 |
| Structural classes | Nuclear receptor, Forkhead, Homeodomain, Rel |

**Grammar Type Distribution**:
| Type | Count |
|------|-------|
| Orientation-dependent | 63 (42.9%) |
| Spacing-dependent | 28 (19.0%) |
| Mixed | 29 (19.7%) |
| Insensitive | 13 (8.8%) |
| Helical-phased | 14 (9.5%) |

**Interpretation**: TF structural class does NOT predict grammar type (accuracy barely above baseline). This suggests grammar properties emerge from TF pair interactions rather than individual TF structures.

### Experiment 5.3: Grammar-Strength Interaction

| Dataset | Strength-Grammar Corr | p-value | Interpretation |
|---------|----------------------|---------|----------------|
| Vaishnav | 0.003 | 0.871 | No tradeoff |
| Klein | 0.096 | 0.002 | Weak positive |

**Interpretation**: No strong grammar-strength tradeoff. Weak promoters do not compensate with stricter grammar. In Klein data, there's a weak positive correlation (r=0.096, p=0.002), meaning stronger enhancers actually have slightly more grammar sensitivity — possibly because they have more TF binding sites creating more arrangement possibilities.

### Experiment 5.4: Grammar Phase Diagram

Phase diagrams map (motif count × motif density) → mean GSI for each dataset:

- **No critical threshold detected**: GSI is relatively uniform across the motif count / density space for both datasets. Grammar does not "turn on" at a specific motif count or density threshold.
- **Vaishnav**: GSI ranges 0.066-0.092 across phase space, with slight elevation at high motif counts (>17 motifs, GSI ~0.09)
- **Klein**: GSI ranges 0.073-0.089, with modest increase at higher motif density bins

### Module 5 Output Files

| File | Description |
|------|-------------|
| `results/module5/vaishnav_biophysics.json` | Yeast biophysics residual analysis |
| `results/module5/klein_biophysics.json` | Human biophysics residual analysis |
| `results/module5/structure_grammar_map.parquet` | TF structure → grammar type mapping |
| `results/module5/structure_predicts_grammar.json` | Structure prediction accuracy |
| `results/module5/vaishnav_strength_tradeoff.json` | Yeast strength-grammar interaction |
| `results/module5/klein_strength_tradeoff.json` | Human strength-grammar interaction |
| `results/module5/vaishnav_phase_diagram.json/pdf` | Yeast phase diagram |
| `results/module5/klein_phase_diagram.json/pdf` | Human phase diagram |

---

## Module 6: Grammar-Optimized Design & Completeness

**Date**: 2026-02-03
**Status**: **COMPLETE**
**Runtime**: ~3 minutes (1 model × 5 datasets)
**Model used**: DNABERT-2

### Experiment 6.2: Grammar Completeness Ceiling Test

**Method**: Hierarchical R² decomposition measuring what fraction of expression variance each level of modeling captures:
1. **Vocabulary only**: Predict expression from motif presence/absence (bag-of-motifs)
2. **Vocab + Simple Grammar**: Add pairwise motif interaction features
3. **Vocab + Full Grammar**: Add higher-order grammar features (k=3-6 interactions)
4. **Full Model**: Neural model prediction on actual sequence
5. **MPRA Replicate**: Estimated biological replicate ceiling (R²≈0.85)

| Dataset | Vocab R² | +Grammar R² | Full Model R² | Replicate R² | Grammar Contrib | Completeness |
|---------|----------|-------------|---------------|-------------|----------------|-------------|
| Agarwal (human K562) | 0.151 | 0.145 | 0.037 | 0.85 | -0.009 | 17.7% |
| de Almeida (human neural) | 0.049 | 0.058 | 0.019 | 0.85 | **+0.011** | 5.7% |
| Vaishnav (yeast) | 0.095 | 0.094 | 0.276 | 0.85 | -0.002 | 11.1% |
| Jores (plant) | 0.105 | 0.109 | 0.113 | 0.85 | **+0.005** | 12.4% |
| Klein (human HepG2) | 0.129 | 0.142 | 0.068 | 0.85 | **+0.018** | 16.6% |

**Key Findings**:

1. **Grammar contribution is small but detectable**: With FIMO-based motif scanning and 26,922 rules, grammar adds positive R² above vocabulary for Klein (+0.018), de Almeida (+0.011), and Jores (+0.005). The improved motif calling from FIMO enables detection of real grammar contributions that were invisible with the fallback scanner.

2. **Vocabulary captures most extractable signal**: Motif presence/absence (vocabulary) explains 5-15% of expression variance. Grammar adds at most 1.8% on top.

3. **Model gap**: For Vaishnav, the full neural model (R²=0.276) substantially outperforms the vocabulary model (R²=0.095), meaning the model captures expression-relevant features beyond simple motif vocabulary — likely sequence-level patterns not captured by our motif scanning.

4. **Large ceiling gap**: All grammar completeness scores are 6-18%, meaning our grammar framework captures only a fraction of the explainable expression variance. The remaining 82-94% gap is due to:
   - Vocabulary not fully captured (only top 200 motifs scanned)
   - Higher-order sequence features the model uses
   - Non-motif sequence determinants (spacer sequence, broader context)

5. **Cross-species pattern**: Completeness is similar across species (6-18%), suggesting the limitation is fundamental to the motif-centric grammar framework, not species-specific.

### Module 6 Output Files

| File | Description |
|------|-------------|
| `results/module6/{dataset}_completeness.json` | Per-dataset completeness analysis |
| `results/module6/{dataset}_completeness.pdf` | Completeness barplots |

---

## Disk Usage Tracking

| Date | Total Used | Budget | Notes |
|------|-----------|--------|-------|
| 2026-02-02 | ~0 GB | 200 GB | Project initialized |
| 2026-02-03 | 1.7 GB | 200 GB | Models cached, data preprocessed, Module 1 complete |
| 2026-02-03 | 2.1 GB | 200 GB | FIMO installed, all 6 modules complete, FIMO re-run of Modules 1-2 complete |
| 2026-02-04 | 2.4 GB | 200 GB | v2 pipeline complete (all 14 components), v2 results in results/v2/ |
| 2026-02-04 | 2.5 GB | 200 GB | v2 figures generated (8 PDF + 8 PNG), biophysics corrected with gsi_robust |

---

## Statistical Validation Log

| Claim | Test | Statistic | p-value | Significant? | Date |
|-------|------|-----------|---------|-------------|------|
| Grammar exists (yeast) | F-test (v1, FLAWED) | GSI=0.079 | p<0.05 for 100% | **Artifact** | 2026-02-03 |
| Grammar exists (human K562) | F-test (v1, FLAWED) | GSI=0.071 | p<0.05 for 100% | **Artifact** | 2026-02-03 |
| Grammar exists (human HepG2) | F-test (v1, FLAWED) | GSI=0.077 | p<0.05 for 100% | **Artifact** | 2026-02-03 |
| Grammar exists (human neural) | F-test (v1, FLAWED) | GSI=0.066 | p<0.05 for 100% | **Artifact** | 2026-02-03 |
| Grammar exists (plant) | F-test (v1, FLAWED) | GSI=0.071 | p<0.05 for 100% | **Artifact** | 2026-02-03 |
| Grammar (v2 corrected, all) | z-score test | median z=0.77 | p<0.05 for 8.3% | **Minority** | 2026-02-04 |
| Grammar (v2, vaishnav/dnabert2) | z-score test | median z=0.99 | p<0.05 for 15.6% | **Yes (subset)** | 2026-02-04 |
| Grammar (v2, vaishnav/hyenadna) | z-score test | median z=0.48 | p<0.05 for 0.6% | **No** | 2026-02-04 |
| NT embeds predict expression | Pearson correlation | r=0.551 | p<<0.001 | **Yes** | 2026-02-03 |
| DNABERT-2 embeds predict expression | Pearson correlation | r=0.472 | p<<0.001 | **Yes** | 2026-02-03 |
| HyenaDNA embeds predict expression | Pearson correlation | r=0.418 | p<<0.001 | **Yes** | 2026-02-03 |
| Grammar is context-sensitive | BIC model comparison | Gap=0.990 constant | BIC favors constant | **Yes** | 2026-02-03 |
| Cross-species grammar transfer (H→Y) | Linear regression | R²=0.000 | p=1.0 | **No transfer** | 2026-02-03 |
| Cross-species grammar transfer (Y→H) | Linear regression | R²=0.000 | p=1.0 | **No transfer** | 2026-02-03 |
| Klein biophysics explains grammar | Random forest CV | R²=0.636 | - | **Yes** | 2026-02-03 |
| Vaishnav biophysics explains grammar | Random forest CV | R²=0.082 | - | **Weak** | 2026-02-03 |
| TF structure predicts grammar type | Classification | Acc=43.5% vs 42.9% baseline | - | **No** | 2026-02-03 |
| Strength-grammar tradeoff (yeast) | Spearman correlation | r=0.003 | p=0.871 | **No** | 2026-02-03 |
| Strength-grammar tradeoff (human) | Spearman correlation | r=0.096 | p=0.002 | **Weak** | 2026-02-03 |

---

## Final Conclusions

### Primary Findings

1. **Grammar exists in a minority of enhancers** *(REVISED in v2)*: v1 claimed 100% significance, but this was an artifact of the F-test with zero noise variance. With corrected z-score p-values, only **8.3% of enhancers** show that their natural arrangement is significantly different from random rearrangements (p<0.05). Grammar (shuffling affects expression) is detectable via non-zero GSI in most enhancers, but the natural arrangement is typically "average" — not specifically optimized. The "flexible billboard" model remains apt, but the billboard is even more flexible than v1 suggested.

2. **Grammar is context-sensitive**: Pairwise grammar rules explain only ~1% of higher-order arrangement effects (compositionality gap = 0.99). Grammar does NOT compose — knowing how motif pairs interact does not predict how triplets or larger combinations behave. By the Chomsky hierarchy, regulatory grammar is at least context-sensitive.

3. **Grammar does not transfer across species**: Zero cross-species grammar transfer between human and yeast (R² = 0.000). Grammar rules are species-specific and possibly enhancer-specific. Even within-species transfer is weak (R² < 0.02).

4. **Grammar has different biophysical bases across species**: In human enhancers (Klein), 63.6% of grammar is explained by DNA biophysical features (CpG content, minor groove width). In yeast, only 8.2% is biophysically determined — yeast grammar likely depends more on protein-protein interactions and chromatin context.

5. **Grammar completeness ceiling is low**: Our motif-centric grammar framework captures only 6-18% of explainable expression variance across datasets. The vocabulary (motif presence/absence) explains most of what grammar captures; adding pairwise or higher-order rules provides negligible improvement.

### Secondary Findings

6. **Architecture matters**: DNABERT-2 and NT v2 (transformers) show ~2× higher grammar sensitivity than HyenaDNA (SSM). Attention mechanisms may encode arrangement information more explicitly than state-space models.

7. **Cross-model consensus is moderate**: Models agree on orientation preferences (82.5%) but diverge on spacing profiles (r=0.399). Only 6.7% of rules show strong cross-architecture consensus.

8. **No grammar-strength tradeoff**: Weak promoters do not compensate with stricter grammar. Grammar sensitivity is approximately independent of expression strength.

9. **TF structural class does not predict grammar type**: The grammar properties of a motif pair depend on the pair interaction, not on the structural families of the individual TFs.

10. **No grammar phase transition**: Grammar does not "turn on" at a critical motif count or density threshold — it is relatively uniform across the motif count/density space.

### Interpretation

The overarching conclusion is that regulatory grammar exists as a **real but secondary contributor** to gene expression. The dominant factor is **vocabulary** — which transcription factor binding sites are present. Grammar (arrangement, spacing, orientation) modulates expression but does not fundamentally control it. This grammar is:
- **Non-compositional**: Cannot be decomposed into independent pairwise rules
- **Species-specific**: Not conserved across evolutionary distances
- **Partially biophysical**: Depends on DNA structure in human enhancers, but on other factors in yeast
- **Architecture-dependent**: Perceived differently by different model architectures

These findings suggest that the "grammar" metaphor may be somewhat misleading for regulatory sequences. Unlike human language, where syntax dramatically alters meaning, regulatory "syntax" fine-tunes an already vocabulary-determined expression program. The regulatory code is more like a **billboard with flexible formatting** than a language with strict grammar.

---

## Final Figures

All figures saved to `results/figures/` in both PDF (vector, 300 DPI) and PNG (raster) formats.

| Figure | Description | Key Finding |
|--------|-------------|-------------|
| `fig1_grammar_existence.pdf` | GSI distribution, per-model/dataset breakdown, information content, max disruption | Grammar exists in 100% of enhancers (mean GSI ~0.08) |
| `fig2_compositionality.pdf` | Compositionality gap vs k, per-model, pairwise R² | Gap=0.99 constant across k: context-sensitive grammar |
| `fig3_rules_consensus.pdf` | Rule strength, spacing sensitivity, consensus, helical phasing | 4,218 rules, 82.5% orientation agreement, 13.2% helical |
| `fig4_transfer.pdf` | Cross-species transfer R² and correlation heatmaps | Zero transfer between human and yeast |
| `fig5_biophysics.pdf` | Biophysics R², feature importances, phase diagrams | Human grammar 63.6% biophysical, yeast only 8.2% |
| `fig6_completeness.pdf` | Completeness ceiling by dataset, grammar contribution | Grammar completeness only 6-18% of ceiling |
| `fig7_summary.pdf` | Grand summary of all key findings | Comprehensive overview panel |

---

## Technical Notes

### Model Loading Workflow
1. Foundation models (DNABERT-2, NT, HyenaDNA) require expression probes for `predict_expression()`
2. Probes are auto-loaded from `data/probes/` when `load_model()` is called
3. Enformer has a built-in CAGE track expression head (no probe needed)
4. Models are loaded one at a time to manage GPU memory

### Vocabulary-Preserving Shuffle Method
1. Scan sequence for TF binding motifs (PWM-based, score_fraction >= 0.65)
2. Extract motif sequences and inter-motif spacers
3. Randomly permute motif order
4. Randomly flip motif orientation (50% chance per motif)
5. Randomly redistribute spacer lengths
6. Fill spacers with dinucleotide-shuffled DNA (preserves dinucleotide frequencies)
7. Assemble new sequence at exact original length

### Known Limitations
- Only 150 of 879 vertebrate motifs / 371 yeast motifs / 805 plant motifs used for motif scanning (speed constraint)
- Vaishnav dataset is synthetic random promoters — may not represent natural enhancer architecture
- Caduceus excluded due to PyTorch version incompatibility (mamba_ssm requires PyTorch >= 2.4)
- Expression probes trained only on Vaishnav (yeast) — may not generalize perfectly to other species/cell types
- Kircher 2019 dataset skipped — saturation mutagenesis format (single-nucleotide variants) incompatible with full-sequence pipeline
- Jores 2021 uses Arabidopsis/Maize/Sorghum promoters — plant motif database is less comprehensive
- Module 6 completeness run with DNABERT-2 only (1 of 3 models) due to runtime constraints
- Only 2 species (human, yeast) for transfer analysis — plant datasets lack established grammar rules
- GSI computed with 50 shuffles per enhancer — more shuffles would improve statistical power
- Max 200 enhancers per (model, dataset) pair for Module 1 — larger samples would better characterize tails

---

## Complete File Manifest

### Source Code (`src/`)
| File | Purpose |
|------|---------|
| `src/models/foundation.py` | Model loading (DNABERT-2, NT, HyenaDNA, Enformer) |
| `src/models/expression_probe.py` | 2-layer MLP probes for expression prediction |
| `src/perturbation/shuffle.py` | Vocabulary-preserving shuffle engine |
| `src/perturbation/motif_scanner.py` | FIMO + fallback PWM motif scanning |
| `src/grammar/gsi.py` | Grammar Sensitivity Index computation |
| `src/grammar/rule_extraction.py` | Pairwise motif interaction rule extraction |
| `src/grammar/compositionality.py` | Compositionality testing (pairwise → higher-order) |
| `src/grammar/complexity.py` | Chomsky hierarchy classification |
| `src/transfer/cross_species.py` | Cross-species grammar transfer |
| `src/transfer/phylogenetics.py` | Grammar phylogenetics and distance matrices |
| `src/decomposition/biophysics.py` | Biophysics residual analysis |
| `src/decomposition/tf_structure.py` | TF structural class mapping |
| `src/decomposition/strength_tradeoff.py` | Grammar-strength interaction |
| `src/decomposition/phase_diagram.py` | Grammar phase diagram |
| `src/design/grammar_optimization.py` | Grammar-aware sequence design |
| `src/design/completeness.py` | Grammar completeness ceiling test |
| `src/utils/sequence.py` | Sequence utilities |
| `src/utils/statistics.py` | Statistical helper functions |
| `src/utils/io.py` | I/O utilities |
| `src/utils/visualization.py` | Plotting functions |

### Scripts (`scripts/`)
| File | Purpose |
|------|---------|
| `scripts/run_full_pipeline.py` | Main pipeline runner (all 6 modules) |
| `scripts/train_probes.py` | Expression probe training script |
| `scripts/prepare_pipeline_data.py` | Data preprocessing with motif scanning |
| `scripts/preprocess_remaining_mpra.py` | Raw MPRA → parquet for Agarwal, Jores, de Almeida |
| `scripts/generate_final_figures.py` | v1 summary figure generation |
| `scripts/generate_v2_figures.py` | v2 publication figure generation (8 figures) |
| `scripts/rerun_biophysics_robust.py` | Re-run Module 5 biophysics with gsi_robust |
| `scripts/collect_v2_results.py` | Collect v2 results into summary JSON |
| `scripts/validation_experiment_j.py` | Evolutionary conservation analysis |

### Results (`results/`)
| Directory | Key Files |
|-----------|-----------|
| `results/module1/` | `all_gsi_results.parquet`, `gsi_summary.json`, `grammar_information.parquet`, `fig1_gsi_distribution.pdf` |
| `results/module2/` | `grammar_rules_database.parquet`, `consensus_scores.parquet`, `global_consensus.json` |
| `results/module3/` | `compositionality_results.parquet`, `complexity_classification.json`, `fig2_compositionality.pdf` |
| `results/module4/` | `transfer_matrix.parquet`, `grammar_phylogeny.json`, `fig4_transfer_heatmap.pdf` |
| `results/module5/` | `*_biophysics.json`, `*_phase_diagram.json/pdf`, `*_strength_tradeoff.json`, `structure_*.json/parquet` |
| `results/module6/` | `*_completeness.json/pdf` |
| `results/figures/` | `fig1-fig7` (PDF + PNG, v1 summary figures) |
| `results/v2/figures/` | `v2_fig1-fig8` (PDF + PNG, v2 publication figures) |
| `results/v2/module5/` | `*_biophysics_robust.json` (corrected with gsi_robust), `biophysics_robust_comparison.json` |

---

## GRAMLANG v2: Comprehensive Improvements

**Date**: 2026-02-03 to 2026-02-04
**Status**: **COMPLETE** (All 14 components finished, 2026-02-04 ~10:14 UTC, 12.4h runtime)
**Motivation**: Critical analysis identified key methodological flaws in v1 and proposed improvements.

### P0 Issues Addressed

1. **Expression probes trained only on vaishnav (yeast)**: Applied to all species in v1. Fixed by training species-specific probes for all 15 (model, dataset) combinations.
2. **Enformer not used**: Added Enformer with native CAGE expression head (no probe needed) for human datasets.
3. **Compositionality gap of 0.99 likely artifact**: Redesigned as enhancer-specific ANOVA.
4. **Cross-species transfer tested wrong thing**: Redesigned as distributional comparison.

### v2 Species-Specific Probe Training

**Date**: 2026-02-03
**Method**: Same architecture as v1 (2-layer MLP), trained on dataset-specific data
**Script**: `scripts/train_all_probes.py`

| Model | Dataset | Species | Pearson r | R² | Viable (R²>0.05)? |
|-------|---------|---------|-----------|-----|-------------------|
| DNABERT-2 | agarwal | human | 0.3404 | 0.1159 | Yes |
| DNABERT-2 | de_almeida | human | 0.2336 | 0.0546 | No |
| DNABERT-2 | jores | plant | **0.5796** | **0.3359** | Yes |
| DNABERT-2 | klein | human | 0.2602 | 0.0677 | No |
| DNABERT-2 | vaishnav | yeast | 0.4722 | 0.2229 | Yes |
| NT v2-500M | agarwal | human | 0.3305 | 0.1092 | Yes |
| NT v2-500M | de_almeida | human | 0.2863 | 0.0820 | No |
| NT v2-500M | jores | plant | **0.5752** | **0.3309** | Yes |
| NT v2-500M | klein | human | 0.3309 | 0.1095 | Yes |
| NT v2-500M | vaishnav | yeast | **0.5513** | **0.3039** | Yes |
| HyenaDNA | agarwal | human | 0.1681 | 0.0283 | No |
| HyenaDNA | de_almeida | human | 0.1738 | 0.0302 | No |
| HyenaDNA | jores | plant | 0.5067 | 0.2567 | Yes |
| HyenaDNA | klein | human | 0.2390 | 0.0571 | No |
| HyenaDNA | vaishnav | yeast | 0.4181 | 0.1748 | Yes |

**Key findings**:
- Jores (plant) probes are the best after vaishnav, with R²=0.26-0.34 — much better than cross-species vaishnav probes (R²=0.17-0.22), validating the critique
- de_almeida is the hardest dataset (R²<0.09 for all models), likely due to small size (2,453 sequences) and neural cell type complexity
- HyenaDNA probes are weak for non-yeast datasets (R²<0.06), suggesting SSM architecture captures fewer expression-relevant features
- 9/15 probes meet viability threshold (R²>0.05); the remaining 6 will produce noisier GSI estimates

### v2 Module 1: GSI Census with Species-Specific Probes

**Date**: 2026-02-03
**Status**: Foundation models COMPLETE (15/15 files), Enformer IN PROGRESS
**Parameters**: n_shuffles=100, max_enhancers=500 (vs v1: n_shuffles=50, max_enhancers=200)

#### v1 vs v2 GSI Comparison (Foundation Models)

**7,500 measurements** (500 enhancers × 3 models × 5 datasets) vs v1's 3,000.

| Dataset | Species | Mean Cohen's d | Median Cohen's d | Significant (3 models) | Key Finding |
|---------|---------|----------------|------------------|----------------------|-------------|
| jores | plant | **+1.20** | +1.51 | 3/3 *** | Strongest consistent improvement |
| klein | human | +0.25 | +0.27 | 3/3 *** | Median GSI 5-15× higher |
| agarwal | human | +0.13 | +0.11 | 3/3 *** | Large positive but outlier-driven |
| vaishnav | yeast | +0.11 | +0.14 | 2/3 * | Negligible change (expected) |
| de_almeida | human | **-1.49** | -3.01 | 1/3 (DNABERT2 only) | v2 WORSE for HyenaDNA, NT |

**Overall**: 12/15 pairs significantly improved (Mann-Whitney p<0.05). Mean Cohen's d = +0.04 (misleadingly low due to de_almeida offsets). Median Cohen's d = +0.14.

**Threshold analysis** (fraction of enhancers exceeding GSI thresholds):
- GSI > 0.10: v1 avg 13.0% → v2 avg 59.2% (+46.2 pp)
- GSI > 0.20: v1 avg 0.1% → v2 avg 33.8% (+33.7 pp)

**Interpretation**: Species-specific probes dramatically improve grammar sensitivity detection for datasets where the probe is viable (R²>0.05), particularly jores (plant) and klein (human). However, weak probes (de_almeida with HyenaDNA and NT) can degrade performance, adding noise rather than signal. Vaishnav is unchanged because v1 already used yeast probes.

**Outlier analysis**: 52 enhancers (0.7%) have GSI > 10, caused by near-zero shuffle mean inflating the CV formula. These are flagged with `gsi_extreme=True` and a robust metric `gsi_robust` is provided.

#### Critical P-Value Correction

**Date**: 2026-02-04
**Issue**: v1 and initial v2 pipeline used F-test with noise_var estimated from 20 redundant predictions of the same sequence. Since models are deterministic, noise_var ≈ 0, producing F-stat → ∞ and p ≈ 0 for ALL enhancers. This made the "100% significant" claim in v1 an artifact.

**Fix**: Re-computed p-values using z-score approach: z = |original_expr - shuffle_mean| / shuffle_std, then p = 2(1 - Φ(|z|)). This tests whether the natural arrangement's expression is significantly different from the shuffle distribution.

| Before (v1/v2 uncorrected) | After (z-score corrected) |
|---------------------------|--------------------------|
| 100% significant (p<0.05) | **8.3% significant** (625/7,500) |
| 100% significant (p<0.01) | **2.6% significant** (196/7,500) |
| Median p ≈ 1e-16 | Median p ≈ 0.44 |

**Per-dataset significance rates (corrected p<0.05)**:
| Dataset/Model | Sig % | Median z | Interpretation |
|---------------|-------|----------|----------------|
| vaishnav/dnabert2 | 15.6% | 0.99 | Best: yeast probes are strong |
| jores/hyenadna | 12.8% | 0.91 | Plant probes effective |
| agarwal/dnabert2 | 12.2% | 0.85 | Good signal |
| de_almeida/dnabert2 | 11.0% | 0.78 | Moderate despite weak probes |
| vaishnav/hyenadna | 0.6% | 0.48 | HyenaDNA weak for yeast |
| vaishnav/nt | 4.4% | 0.75 | NT moderate for yeast |

**Revised interpretation**: Grammar is NOT universally present. Only ~8% of enhancers show evidence that their natural arrangement is specifically tuned (original expression significantly different from random rearrangements). Most enhancers have some GSI > 0 (shuffling does create expression variation), but the natural arrangement is typically "average" — not optimized for expression. This means:
- Arrangement CAN matter (high GSI), but usually the natural arrangement isn't specifically selected for its position in the expression landscape
- The ~8% with significant z-scores may represent genuinely grammar-sensitive regulatory elements
- This revises the v1 claim from "grammar exists everywhere" to "grammar is detectable in a minority of enhancers"

Also saved: `gsi_robust` (denominator-stabilized GSI), `z_score`, `p_value_corrected`, `gsi_extreme` flag to all 15 parquet files.

#### v2 Module 1: Comprehensive GSI Analysis

**Date**: 2026-02-04
**Script output**: `results/v2/module1/v2_gsi_analysis.json`

| Finding | Result |
|---------|--------|
| Most grammar-detecting model | **DNABERT-2** (median GSI=0.167, 81.4% > 0.10) |
| Least grammar-detecting model | HyenaDNA (median GSI=0.065, 34.2% > 0.10) |
| Strongest grammar dataset | **Klein** (median GSI=0.611, 96.1% > 0.10) |
| Weakest grammar dataset | de_almeida (median GSI=0.044, 33.3% > 0.10) |
| Motif density → GSI | NO correlation (mean rho=-0.059) |
| Expression → GSI | Weak positive (mean rho=+0.10, dataset-dependent) |
| Dataset explains GSI variance | **29.0%** (η² from two-way ANOVA on log-GSI) |
| Model explains GSI variance | **4.5%** (η²) |
| Residual | **63.2%** |

**Cross-model agreement** (Spearman rho of GSI between model pairs on same enhancers):
| Dataset | DNABERT2↔NT | DNABERT2↔HyenaDNA | NT↔HyenaDNA |
|---------|------------|-------------------|-------------|
| agarwal | 0.90*** | 0.70*** | 0.75*** |
| jores | 0.89*** | 0.65*** | 0.70*** |
| klein | 0.88*** | 0.66*** | 0.67*** |
| vaishnav | 0.56*** | -0.03 ns | -0.08 ns |
| de_almeida | -0.06 ns | -0.16*** | 0.05 ns |

**Interpretation**: Dataset identity dominates GSI variance (29% vs 4.5% for model). Models agree strongly on agarwal/jores/klein but disagree on vaishnav and de_almeida, likely because HyenaDNA probes are weak for these datasets. The strong cross-model agreement for well-probed datasets supports GSI as a robust measure.

#### Validation Experiment I: Cross-Dataset Replication

**Date**: 2026-02-04
**Question**: Do within-species GSI distributions resemble each other more than cross-species?
**Script output**: `results/v2/module1/cross_dataset_validation.json`

| Metric | Within-species | Cross-species | Ratio |
|--------|---------------|--------------|-------|
| Mean |Cohen's d| | **0.955** (n=9) | **1.888** (n=21) | **1.98x** |
| Permutation p-value | — | — | **0.035** |

**Per-model within/cross ratios**:
| Model | Within |d| | Cross |d| | Ratio |
|-------|-----------|----------|-------|
| DNABERT-2 | 1.10 | 1.64 | 1.49x |
| NT v2 | 1.00 | 1.96 | 1.95x |
| HyenaDNA | 0.77 | 2.07 | 2.69x |

**Result**: **SUPPORTED** — within-species GSI distributions are significantly more similar than cross-species (permutation p=0.035). This provides evidence that GSI captures species-specific regulatory grammar properties. However, within-species |d| is still ~1.0, reflecting genuine heterogeneity across MPRA experimental designs.

#### v1 Rules Database Analysis

**Date**: 2026-02-04
**Script output**: `results/v2/v1_rules_analysis.json`

| Metric | Value |
|--------|-------|
| Total rules | 26,922 rows |
| Unique motif pairs | 5,032 |
| Unique motifs | 398 |
| Weak rules (FC < 1.5) | **32.2%** |
| Full cross-model consensus | **10.5%** (642/6,090 pairs) |
| Orientation agreement | 52.4% |
| Spacing agreement (std≤5bp) | 15.7% |
| +/+ orientation bias | **83.3%** (potential extraction artifact) |
| Cross-species shared pairs | **0** (disjoint motif databases) |
| Helical periodicity (>2.0) | 13.5% |

**Key issues for v2 Module 2**:
1. 32% of rules have fold_change < 1.5 — need stricter filtering
2. Only 10.5% have full consensus — cross-model agreement should be a quality gate
3. No p-value column — v2 should add significance testing
4. 83.3% +/+ orientation may be extraction bias, not biology
5. 42% of rules classified as "weak" — poor signal-to-noise ratio

### v2 New Experiment Modules Implemented

| Experiment | Module | Description | Status |
|------------|--------|-------------|--------|
| B | `src/grammar/synthetic_probes.py` | Synthetic grammar probes with controlled spacing/orientation | Code complete, awaiting pipeline |
| C | `src/grammar/attention_grammar.py` | Attention-based grammar extraction from transformers | Code complete, awaiting pipeline |
| E | `src/grammar/information_theory.py` | Information-theoretic R² decomposition (vocab vs grammar) | Code complete, awaiting pipeline |
| F | `src/grammar/heterogeneity.py` | Grammar heterogeneity analysis (grammar-rich enhancers) | Code complete, awaiting pipeline |
| G | `src/grammar/counterfactual.py` | Counterfactual grammar potential (max-min across shuffles) | Code complete, awaiting pipeline |
| — | `src/grammar/compositionality_v2.py` | Enhancer-specific factorial compositionality | Code complete, awaiting pipeline |
| — | `src/transfer/distributional_transfer.py` | Distributional grammar property comparison | Code complete, awaiting pipeline |
| — | `src/grammar/anova_decomposition.py` | ANOVA variance decomposition + power analysis | Code complete, awaiting pipeline |

### v2 Pipeline Structure

**Script**: `scripts/run_v2_pipeline.py`
**Phases**:
- Phase 1: Core Modules 1-6 re-run with species-specific probes + Enformer
- Phase 2: New experiments (B, E, F, G)
- Phase 3: Redesigned tests (compositionality v2, distributional transfer, attention analysis, ANOVA)

**Results directory**: `results/v2/`

### v2 Files

| File | Purpose |
|------|---------|
| `scripts/train_all_probes.py` | Train probes for all 15 (model, dataset) combinations |
| `scripts/run_v2_pipeline.py` | Full v2 pipeline (3 phases) |
| `src/grammar/synthetic_probes.py` | Experiment B: controlled spacing/orientation probes |
| `src/grammar/attention_grammar.py` | Experiment C: attention pattern analysis |
| `src/grammar/information_theory.py` | Experiment E: R² decomposition |
| `src/grammar/heterogeneity.py` | Experiment F: grammar heterogeneity |
| `src/grammar/counterfactual.py` | Experiment G: grammar potential |
| `src/grammar/compositionality_v2.py` | Redesigned compositionality |
| `src/transfer/distributional_transfer.py` | Redesigned cross-species transfer |
| `src/grammar/anova_decomposition.py` | ANOVA + power analysis |

---

## v2 Pipeline Results (Complete)

**Pipeline**: `scripts/run_v2_pipeline.py`
**Started**: 2026-02-03 ~21:47 UTC | **Completed**: 2026-02-04 ~10:14 UTC
**Duration**: ~12.5 hours | **GPU**: A100 80GB (GPU 0)
**Process**: PID 3227868

### Module 1: GSI Census (v2)

**Scale**: 7,700 measurements (7,500 foundation model + 150 Enformer + 50 per Enformer dataset)
**Models**: DNABERT-2, NT v2-500M, HyenaDNA (500 enhancers each), Enformer (50 each, 3 datasets)

#### Foundation Model Results (after p-value correction)

| Dataset | Median GSI | Frac Significant (p<0.05) |
|---------|-----------|--------------------------|
| Klein (HepG2) | 0.611 | 7.1% |
| Agarwal (K562) | 0.333 | 9.0% |
| Jores (plant) | 0.118 | 10.4% |
| Vaishnav (yeast) | 0.084 | 6.9% |
| de Almeida (neural) | 0.044 | 8.3% |

#### Enformer Results (after p-value correction)

| Dataset | n | Median GSI | Frac Significant |
|---------|---|-----------|-----------------|
| Agarwal (K562) | 50 | 0.446 | 14.0% |
| de Almeida (neural) | 50 | 0.450 | 2.0% |
| Klein (HepG2) | 50 | 0.434 | 4.0% |

**Key finding**: Enformer shows higher median GSI (~0.44) than foundation models (~0.13), but lower significance rate. Enformer's 196,608bp context window captures broader regulatory context but with noisier grammar signal.

#### ANOVA of GSI Variance

| Source | η² | p-value |
|--------|-----|---------|
| Dataset | 0.290 | <1e-300 |
| Model | 0.045 | 1.0e-111 |
| Dataset × Model | 0.033 | 1.2e-77 |
| Residual | 0.632 | — |

**Dominant factor**: DATASET (29% of variance), not model architecture (4.5%).

### Module 2: Grammar Rules (v2)

| Metric | v1 | v2 | Change |
|--------|-----|-----|--------|
| Total rules | 26,922 | 9,019 | -67% |
| Mean consensus | 0.521 | 0.433 | -17% |
| Orientation agreement | 85.1% | 84.8% | ~same |
| Spacing correlation | 0.479 | 0.289 | -40% |
| High consensus (>0.8) | 7.7% | 2.4% | -69% |
| Contested rules | 2.9% | 12.1% | +4.2× |

**Interpretation**: With species-specific probes and 500 enhancers (vs 200), fewer rules are extracted and they show lower cross-model agreement. This suggests v1's higher consensus was partly inflated by the shared yeast probe introducing systematic bias. The higher contested fraction (12.1%) indicates genuine model disagreement on grammar rules.

### Module 3: Compositionality (v2)

| Metric | v1 | v2 |
|--------|-----|-----|
| Classification | Context-sensitive | Context-sensitive |
| Mean gap | 0.991 | 0.989 |
| Best BIC model | Linear | Linear |
| Linear R² | low | 0.480 |

**Confirmed**: Grammar remains context-sensitive (non-compositional). The compositionality gap is ~0.99 — pairwise rules explain only ~1% of higher-order effects. This is stable across v1 and v2, confirming non-compositionality is a fundamental property of regulatory grammar.

### Module 4: Cross-Species Transfer (v2)

All cross-species grammar distances remain 1.0 (maximum). Grammar is completely species-specific. This holds with species-specific probes (confirming it's not an artifact of probe mismatch).

### Module 5: Causal Determinants (v2, CORRECTED)

#### Original Results (raw GSI — BROKEN for human datasets)

| Dataset | Biophysics R² (v1) | Biophysics R² (v2, raw) | Issue |
|---------|--------------------|-----------------------|-------|
| Agarwal (K562) | 0.649 | -9.56 | Extreme GSI outliers (max 195) |
| Klein (HepG2) | 0.624 | -19.14 | Extreme GSI outliers (max 250) |
| de Almeida (neural) | 0.524 | -0.57 | Moderate outliers |
| Jores (plant) | 0.482 | 0.79 | No outliers |
| Vaishnav (yeast) | 0.117 | 0.20 | No outliers |

#### Corrected Results (gsi_robust — denominator-stabilized GSI)

Re-run with `gsi_robust = shuffle_std / max(|shuffle_mean|, shuffle_std × 0.1)` which caps extreme values:

| Dataset | R² (raw GSI) | R² (gsi_robust) | Change | Top Feature (importance) |
|---------|-------------|----------------|--------|--------------------------|
| **Jores** (plant) | 0.792 | **0.789** | -0.003 | shape_Roll_std (0.586) |
| **Klein** (HepG2) | -19.14 | **0.375** | +19.51 | shape_MGW_mean (0.158) |
| **Vaishnav** (yeast) | 0.197 | **0.218** | +0.021 | dinuc_CG (0.160) |
| **Agarwal** (K562) | -9.56 | **0.062** | +9.62 | dinuc_CG (0.212) |
| de Almeida (neural) | -0.573 | **-0.488** | +0.085 | dinuc_CA (0.170) |

**Revised interpretation**: With stabilized GSI, biophysics explains grammar across a wide spectrum:
- **Jores (79%)**: DNA Roll flexibility dominates — grammar in plant promoters is primarily a structural property
- **Klein (38%)**: Minor Groove Width and Propeller Twist drive grammar — HepG2 enhancers have biophysically predictable grammar, overturning the v2 initial conclusion
- **Vaishnav (22%)**: CpG dinucleotide content is key — improved from v1 (12%) with species-specific probes
- **Agarwal (6%)**: Weak but positive — K562 grammar is mostly non-biophysical
- **de Almeida (-49%)**: Still negative — grammar signal too weak for any predictor

The v1 results (all positive, R²=0.12-0.65) likely benefited from the shared yeast probe smoothing GSI distributions. The corrected v2 results show genuine species-specific biophysics patterns.

### Module 6: Completeness Ceiling (v2)

| Dataset | Vocab R² | Grammar Contribution | Completeness | v1 Completeness |
|---------|---------|---------------------|-------------|-----------------|
| Agarwal | 0.151 | -0.005 | 17.3% | 17.0% |
| Klein | 0.129 | -0.008 | 14.5% | 16.7% |
| Jores | 0.105 | +0.001 | 12.5% | 12.8% |
| Vaishnav | 0.095 | +0.001 | 11.3% | 11.0% |
| de Almeida | 0.049 | +0.013 | 7.0% | 6.8% |

**Stable result**: Grammar completeness is 7-17% across datasets, essentially unchanged from v1. Grammar rules contribute at most 1.3% additional R² beyond vocabulary. The 83-93% gap to replicate ceiling remains.

### Experiment B: Synthetic Grammar Probes

Ran for all 5 datasets × 2 models (DNABERT-2, NT). Results in `results/v2/experiment_b/`. Synthetic probes with controlled spacing and orientation provide ground truth for validating grammar detection sensitivity.

### Experiment E: Information-Theoretic Decomposition

| Dataset | Vocab R² | Grammar Fraction | Unexplained |
|---------|---------|-----------------|-------------|
| Agarwal | 0.053 | 36.6% | 91.6% |

Grammar features explain 36.6% of the explained variance (which is only 8.4% total). Vocabulary explains the majority (63.4%). 91.6% of expression variance remains unexplained by the motif-centric framework.

### Experiment F: Grammar Heterogeneity

Grammar sensitivity is highly heterogeneous across enhancers:
- GSI skewness: 14.2, kurtosis: 215 (heavy-tailed distribution)
- Top 5% "grammar-rich" enhancers have significantly higher GC content (0.60 vs 0.51, p=7.7e-7) and CpG ratio (0.59 vs 0.30, p=1.0e-6)
- Grammar-rich enhancers cannot be predicted from sequence features alone (CV R² = -3.46)

### Experiment G: Grammar Potential

| Dataset | Mean Potential | Utilization | Headroom |
|---------|---------------|-------------|----------|
| Agarwal | 0.449 | 0.511 | 0.220 |

Natural enhancer arrangements achieve 51.1% of their grammar potential. 13% of enhancers are in the top 10th percentile of potential, and grammar potential anticorrelates with motif count (r = -0.245) — enhancers with fewer motifs have more room for grammar optimization.

### Compositionality v2 (Enhancer-Specific Factorial)

| Metric | Value |
|--------|-------|
| Tests | 984 across 188 enhancers |
| Non-additive fraction | 77.5% |
| Additive fraction | 6.7% |
| Median compositionality | 0.0 |
| Mean interaction effect | 0.40 |
| Mean abs interaction | 0.73 |

**Confirmed**: 77.5% of motif pairs show non-additive (interactive) effects on expression. The median compositionality of 0.0 means half of all motif pair interactions have zero additive component — pure epistasis. This is stronger than the v1 Module 3 finding (gap=0.99), corroborating that motif arrangement effects are inherently non-linear.

### Distributional Transfer

Grammar properties are species-specific but with one conserved property:

| Cross-species pair | Mean Cohen's d |
|-------------------|----------------|
| Human vs Plant | 3.56 (spacing), 2.74 (orientation) |
| Human vs Yeast | 11.3 (spacing), 6.58 (orientation) |
| Plant vs Yeast | 5.20 (spacing), 2.72 (orientation) |

- **Conserved**: Helical phasing rate (~13.4% ± 0.5% across all species) — this is the only universally conserved grammar property
- **Divergent**: Spacing sensitivity (yeast 10× higher), orientation sensitivity (yeast 10× higher), grammar type distribution (yeast = 57% "mixed", human = 62% "insensitive")
- Grammar properties conserved: 16.7% (only helical phasing)

### ANOVA Variance Decomposition

| Dataset | Vocab η² | Grammar η² | Vocab Dominance |
|---------|---------|-----------|----------------|
| Agarwal | 0.111 | 0.000 | 100% |
| Klein | 0.224 | 0.000 | 100% |

Grammar explains **0%** of expression variance in the ANOVA framework. Vocabulary dominance = 100% for both tested datasets.

**Power analysis** (Agarwal): Observed grammar effect = 0.001, power = 29.3% (underpowered). Minimum detectable effect at 80% power = 0.004. The analysis is not adequately powered to detect the observed tiny grammar effect.

### Attention Analysis (Experiment C)

NT v2-500M attention heads analyzed for grammar-aware patterns:

| Metric | Value |
|--------|-------|
| Total heads | 464 |
| Grammar-sensitive heads | 101 (21.8%) |
| Mean enrichment (all) | 1.49× |
| Mean enrichment (grammar) | 2.99× |
| Top head | Layer 16 Head 6 (4.31× enrichment) |

Grammar-sensitive heads are concentrated in layers 15-19 and 24-28 (middle and late layers). The top 10 heads all show >4× motif-pair attention enrichment, suggesting genuine learned grammar representations.

### Orientation Bias Investigation

**Verdict: PRIMARILY AN EXTRACTION ARTIFACT**

The 83.3% +/+ orientation bias in v1 Module 2 rules is explained by 3 compounding bugs in `rule_extraction.py`:

1. **Spacing optimization uses only +/+ orientation** — spacing is tuned for +/+ and then applied to all orientations, giving +/+ an inherent advantage
2. **argmax favors first element** — when orientation sensitivity is low, `np.argmax` selects +/+ (first in list) disproportionately
3. **Fallback defaults to +/+** — failed orientation tests silently become +/+

Evidence:
- +/+ fraction = 73.1% in low-sensitivity quartile vs 89.0% in high-sensitivity quartile
- Bias is consistent across datasets (78-85%) and models (82-85%)
- +/+ rules have the same fold change as non-+/+ rules (median 1.59 vs 1.58)

**Results**: `results/v2/orientation_bias_investigation.json`

### Enformer vs Foundation Model GSI Correlations

Enformer (native CAGE head, no trained probe) provides an independent grammar measurement against the 3 foundation models (which use trained expression probes).

| Dataset | vs DNABERT-2 | vs NT v2-500M | vs HyenaDNA | Interpretation |
|---------|-------------|---------------|-------------|----------------|
| Agarwal | ρ=+0.30 (p=0.04) | ρ=+0.45 (p=0.001) | ρ=+0.35 (p=0.01) | **Consistent**: grammar signal validated |
| de Almeida | ρ=+0.10 (p=0.48) | ρ=+0.06 (p=0.67) | ρ=-0.23 (p=0.12) | **Uncorrelated**: weak grammar signal |
| Klein | ρ=-0.40 (p=0.004) | ρ=-0.43 (p=0.002) | ρ=-0.45 (p=0.001) | **Anti-correlated**: conflicting signals |

**Key finding**: Klein's extreme GSI values from foundation models (median=0.611, highest of all datasets) **anti-correlate** with Enformer's independent measurement. This casts doubt on Klein's grammar signal and suggests probe-dependent artifacts. Agarwal's grammar signal is validated by Enformer agreement. de Almeida's signal is too weak for either architecture to reliably detect.

### v2 Rules Orientation Bias Check

The v2 rules database (27,057 rules) shows the **same +/+ bias** as v1 (83.4% vs 83.3%), confirming the extraction artifact persists since the same `rule_extraction.py` code was used. The bias is consistent across v2 datasets (79-85%) and models (82-85%).

Additionally, v2 rules have extreme fold change outliers: mean FC = -1.77 billion (!) while median FC = 2.055. These outliers come from near-zero expression denominators. 39.8% of rules are weak (FC<1.5), 51.1% are strong (FC>2.0).

### Validation Experiment J: Evolutionary Conservation of Grammar Properties

**Script**: `scripts/validation_experiment_j.py`
**Results**: `results/v2/module1/validation_experiment_j.json`

#### J1: Within-Species Grammar Consistency

- **Motif pair overlap**: Human datasets share 16.5% of motif pairs (Jaccard), vs zero cross-species overlap (permutation p=0.005). About 30-37% of pairs in each human dataset appear in at least one other human dataset.
- **Rule property agreement**: For shared motif pairs, spacing sensitivity correlation is near zero (mean r=-0.025). Orientation sensitivity is also near zero (mean r=-0.009). Optimal spacing differs by ~15-16bp for the same pair across datasets.
- **Orientation agreement**: ~80-83% for shared pairs (well above 25% chance), but inflated by the +/+ bias artifact.

#### J2: Helical Phasing Universality

- Helical phasing is **UNIVERSAL** across species: human=13.8%, plant=13.4%, yeast=14.3% (CV=0.029).
- **Not above null**: Null model (FFT on random spacings) predicts 23.1% would pass threshold by chance. Observed rates (~22-24%) are indistinguishable from random (binomial p>0.29, Cohen's h<0.013).
- Within-human rates are homogeneous (chi2=1.84, p=0.40).

#### J3: Grammar-Expression Coupling Conservation

- GSI-expression coupling is **NOT conserved** within human datasets:
  - Agarwal: consistently positive (r=+0.28 to +0.35)
  - Klein: consistently negative (r=-0.23 to -0.36)
  - de Almeida: weak/null
- Fisher Z-tests show all pairwise human correlations differ **highly significantly** (all p<0.01, most p<1e-6).
- The relationship between grammar sensitivity and expression strength is context-dependent, not a species property.

#### J4: Motif Pair Grammar Conservation

- Spacing sensitivity: only 2/9 comparisons significant (both de_almeida-klein)
- Orientation sensitivity: 1/9 significant
- Agarwal-Klein comparisons show **anti-correlated** spacing sensitivity (r=-0.13 to -0.32), meaning the same motif pair has opposite spacing preferences in different human contexts.

#### Overall Conclusion

Within-species grammar is **partially conserved at the vocabulary level** (shared motif pairs) but **not conserved at the rule level** (specific spacing/orientation preferences). Helical phasing is universal but indistinguishable from random. Grammar rules are enhancer-context-dependent, not fixed by species identity. The "alphabet" (motif pairs) is partially shared; the "grammar" (arrangement rules) is written anew for each regulatory element.

---

## GRAMLANG v3: Extension Experiments

**Date**: 2026-02-05
**Status**: **COMPLETE** (P0.1-P0.3, P1.1-P1.3, P2.4, P3.3 all finished)
**Motivation**: v2 identified critical gaps: GSI conflates grammar with spacer effects, 100 shuffles may be underpowered, no bag-of-motifs baseline, 78-94% variance unexplained.

### P0.1: Rule Extraction Bug Fixes (Code Changes)
Fixed 3 orientation bias bugs in `rule_extraction.py`:
1. Spacing now optimized independently per orientation (was +/+ only)
2. Orientation testing order randomized (was always +/+ first)
3. Low-sensitivity rules now classified as "undetermined" (was defaulting to +/+)

### P0.2: Redefined Grammar Metrics (Code Changes)
Added to `sensitivity.py`:
- Grammar Effect Size (GES): robust z-score using median/MAD
- Grammar Practical Effect (GPE): (max - min shuffle) / |median|
- Robust GSI: σ / max(|μ|, σ×0.1)
- z-score p-values: 2×(1 - Φ(z))

### P0.3: Power Analysis with 1000 Shuffles

| Dataset | 100 shuf | 250 shuf | 500 shuf | 1000 shuf |
|---------|----------|----------|----------|-----------|
| Agarwal | 10.0% | 11.0% | 11.0% | 11.0% |
| Jores | 10.0% | 9.0% | 9.0% | 9.0% |

**Conclusion**: Grammar rarity is NOT an underpowering artifact. Median z-score stabilizes at ~0.97 — well below 1.96 significance threshold. Most enhancers genuinely have sub-significant grammar effects.

### P1.1: Factorial Shuffle Decomposition (CRITICAL FINDING)

| Factor | Fraction of Full Shuffle Variance |
|--------|----------------------------------|
| | Agarwal | Jores | de Almeida |
| **Spacer DNA** | **82.8%** | **78.4%** | **85.7%** |
| Position | 42.5% | 28.3% | 47.2% |
| Orientation | 25.6% | 18.6% | 24.2% |

**CRITICAL FINDING**: Spacer DNA changes account for 78-86% of total expression variance from vocabulary-preserving shuffles. What we called "grammar sensitivity" is primarily a spacer effect, not a motif arrangement effect. This fundamentally undermines GSI as a grammar metric.

### P1.2: Bag-of-Motifs Baseline

| Dataset | BOM R² (RF) | Grammar R² (RF) | Increment |
|---------|-------------|-----------------|-----------|
| Klein | 0.130 | 0.111 | -0.019 |
| Agarwal | 0.095 | 0.076 | -0.018 |
| Jores | 0.097 | 0.089 | -0.009 |
| de Almeida | 0.039 | 0.059 | +0.021 |
| Vaishnav | -0.001 | -0.027 | -0.026 |

Grammar features decrease prediction in 4/5 datasets. Billboard model confirmed.

### P1.3: Variance Decomposition

| Feature Set | Agarwal R² | Jores R² | de Almeida R² |
|------------|-----------|---------|--------------|
| Vocabulary | -0.038 | -0.152 | -0.227 |
| Grammar | -0.083 | -0.153 | -0.223 |
| **DL Embeddings** | **0.079** | **0.265** | **0.026** |

DL embeddings capture 16-42% more variance than grammar features. The model knows something our motif-centric features completely miss.

### P2.4: PARM Model Comparison

PARM (Nature 2026) is an MPRA-trained CNN that directly predicts promoter activity without needing expression probes. Tested on K562 (Agarwal) and HepG2 (Klein).

| Dataset | Cell Type | Median GSI | Median z | Significant |
|---------|-----------|------------|----------|-------------|
| Agarwal | K562 | 0.192 | 0.55 | 6.0% |
| Klein | HepG2 | 0.448 | 0.62 | 7.0% |

**Conclusion**: PARM shows 6-7% significance rate — matching foundation models (8.3%). The low significance rate is NOT a probe artifact; it's architecture-independent.

### P3.3: Detailed Spacer Ablation (DEFINITIVE BILLBOARD TEST)

Four perturbation types tested on 100 enhancers per dataset:

| Perturbation | Agarwal Δexpr | Jores Δexpr | de Almeida Δexpr |
|--------------|---------------|-------------|------------------|
| random_replace | **0.149** | **0.546** | 0.115 |
| gc_shift | 0.102 | 0.346 | **0.121** |
| dinuc_shuffle | 0.045 | 0.239 | 0.113 |
| **motif_only** | **0.034** | **0.089** | **0.062** |

**CRITICAL FINDING**: Motif permutation (motif_only) has the **smallest effect** across all datasets (Δ = 0.03-0.09). Complete spacer replacement has 2-6× larger effect. This is the definitive billboard confirmation: motif order contributes minimally.

**GC-expression correlation (spacer)**:
- Jores (plant): r = **-0.73** (negative)
- Agarwal (K562): r = **+0.66** (positive)
- Direction reverses across species, explaining zero cross-species transfer.

### v3 Summary: Narrative-Changing Conclusions

1. **GSI measures spacer, not grammar**: 78-86% of GSI variance from spacer changes
2. **Billboard confirmed**: Motif permutation has smallest effect when spacers fixed
3. **Grammar hurts prediction**: Adding grammar features decreases R² in 4/5 datasets
4. **Models learn non-grammar**: DL embeddings capture GC, k-mers, DNA shape — not syntax
5. **Architecture-independent**: PARM (CNN) shows same significance rate as transformers

### v3 Scripts & Results
| Script | Output |
|--------|--------|
| `scripts/run_v3_analysis.py` | `results/v3/power_analysis/`, `factorial_decomposition/`, etc. |
| `scripts/run_spacer_ablation.py` | `results/v3/spacer_ablation/` |
| `scripts/run_parm_comparison.py` | `results/v3/parm_comparison/` |
| Modified: `src/grammar/sensitivity.py` | Added GES, GPE, robust GSI |
| Modified: `src/grammar/rule_extraction.py` | Fixed orientation bias |
| Modified: `src/perturbation/vocabulary_preserving.py` | Added factorial shuffle types |
