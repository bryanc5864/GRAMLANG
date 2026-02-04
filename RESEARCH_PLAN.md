# GRAMLANG: Decoding the Computational Grammar of Gene Regulation

## A Complete Research Plan for Extracting, Testing, and Applying Regulatory Grammar from Pretrained Genomic Models

**Version**: 2.0
**Date**: February 2026
**Scope**: Purely computational — all experiments use pretrained models and published MPRA datasets
**Estimated Duration**: 24 weeks
**Compute Requirements**: 4x A100 80GB GPUs (or equivalent), ~500 GPU-hours total

---

# TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Background & Motivation](#2-background--motivation)
3. [Foundational Questions](#3-foundational-questions)
4. [Resources: Models & Datasets](#4-resources-models--datasets)
5. [Environment Setup](#5-environment-setup)
6. [Module 1: Grammar Existence & Complexity Classification](#6-module-1-grammar-existence--complexity-classification)
7. [Module 2: Cross-Architecture Grammar Extraction](#7-module-2-cross-architecture-grammar-rule-extraction)
8. [Module 3: Compositionality Testing](#8-module-3-compositionality-testing)
9. [Module 4: Cross-Species Grammar Transfer](#9-module-4-cross-species-grammar-transfer)
10. [Module 5: Causal Determinants of Grammar](#10-module-5-causal-determinants-of-grammar)
11. [Module 6: Grammar-Optimized Design & Completeness](#11-module-6-grammar-optimized-design--completeness)
12. [Statistical Framework](#12-statistical-framework)
13. [Output Specifications](#13-output-specifications)
14. [Timeline & Milestones](#14-timeline--milestones)
15. [Expected Results & Decision Tree](#15-expected-results--decision-tree)
16. [Publication Strategy](#16-publication-strategy)

---

# 1. Executive Summary

GRAMLANG is a purely computational project that uses 9 pretrained genomic deep learning models and 6 published MPRA datasets to answer a fundamental question: **does regulatory DNA follow compositional grammar rules, and if so, what is the formal structure of that grammar?**

The project has no wet-lab component. All experiments are in silico perturbation studies: we take real enhancer sequences from MPRA datasets, systematically modify their arrangement (spacing, orientation, ordering of motifs) while holding motif content constant, run these modifications through pretrained models, and analyze the predicted expression changes to extract grammar rules.

The key novelty is NOT in the individual perturbation methods (shuffling, spacing scans — these are established) but in the ANALYTICAL FRAMEWORK applied to the results:

1. **Formal complexity classification** — Where does regulatory grammar sit on the Chomsky hierarchy?
2. **Cross-architecture rule extraction and comparison** — Do 9 different model architectures agree on grammar RULES (not just predictions)?
3. **Compositionality testing** — Do pairwise rules compose to predict higher-order arrangements?
4. **Functional cross-species transfer** — Do grammar rules learned from one species predict arrangement effects in another?
5. **Causal decomposition** — How much grammar is biophysics in disguise vs. irreducible abstract rules?
6. **Grammar-aware sequence design** — Does explicit grammar optimization improve synthetic enhancer design?

---

# 2. Background & Motivation

## 2.1 The Regulatory Grammar Problem

Gene expression is controlled by cis-regulatory elements (enhancers, promoters, silencers) that contain clusters of transcription factor binding sites (TFBS). The "regulatory grammar" hypothesis posits that the arrangement of these binding sites — their spacing, orientation, and ordering — matters for function, analogous to how word order matters in natural language.

Two competing models dominate the field:

**Billboard Model**: Enhancers act as platforms where TF binding sites contribute roughly additively. What matters is WHICH motifs are present and their binding AFFINITY, not their precise arrangement. This predicts that shuffling motifs within an enhancer should have minimal effect on activity.

**Enhanceosome Model**: Enhancers require specific motif arrangements, spacing constraints, and helical phasing. The classic example is the IFN-beta enhanceosome, where the exact arrangement of NF-kB, IRF, ATF-2/c-Jun, and HMGI(Y) binding sites is critical for cooperative assembly.

Recent consensus (2020-2025) is that most enhancers fall on a SPECTRUM between these extremes, with additional models (TF collective, flexible billboard) occupying intermediate positions. However, this "spectrum" characterization is descriptive, not explanatory. It says "it depends" without specifying what it depends ON.

## 2.2 What's Been Done (and Is Therefore Not Novel)

| Analysis | Status | Key References |
|----------|--------|----------------|
| Motif shuffling in enhancers via MPRA | Standard | de Almeida et al. 2023 Nature; Avsec et al. 2021 Nature |
| Pairwise motif spacing/orientation scans | Extensively studied | Yanez-Cuna et al. 2013; Ng et al. 2014; Grossman et al. 2017 |
| Attention/attribution analysis on genomic models | Well-trodden | Avsec et al. 2021; Linder et al. 2023; Novakovsky et al. 2023 |
| Conservation at regulatory positions | Hundreds of papers | PhyloP/PhastCons standard workflow |
| Variant depletion at functional sites (gnomAD) | Standard | Multiple groups |
| Cell-type enhancer specificity | Thoroughly characterized | Gosai et al. 2024 Nature; Taskiran et al. 2024 Nature |
| DNA shape at regulatory elements | Routine | Zhou et al. 2015; DNAshapeR |
| Synthetic enhancer design with DL | Already published | Gosai et al. 2024 Nature; Taskiran et al. 2024 Nature |
| Billboard vs. enhanceosome characterization | Consensus exists | Vaishnav et al. 2022 Nature; Long et al. 2016 |
| Enhancer-promoter compatibility | Multiplicative model (78-91% var.) | Bergman et al. 2022 Nature |
| gReLU framework for sequence modeling | Published 2025 | Linder et al. 2025 Nature Methods |
| SEAM mechanistic attribution | Published 2025 | Chen et al. 2025 bioRxiv |

## 2.3 What's Genuinely Unknown (Our Territory)

1. **Formal grammar complexity**: Where regulatory grammar sits on the Chomsky hierarchy
2. **Cross-model grammar convergence**: Whether different architectures learn the same grammar RULES
3. **Compositionality**: Whether pairwise rules predict higher-order arrangement effects
4. **Functional grammar transfer across species**: Not conservation, but rule transferability
5. **Causal grammar decomposition**: Biophysics residual after accounting for all known physical mechanisms
6. **Grammar phase transitions**: Critical thresholds where arrangement starts to matter
7. **Grammar-optimized design**: Whether explicit grammar optimization improves over expression-only optimization

## 2.4 The Dream

The long-term goal (beyond this project, but what this project contributes to) is: for any ~200bp enhancer in any cell type, predict the quantitative effect of every possible sequence modification on regulatory activity. A secondary goal is to design optimal synthetic enhancers for any target context — or a universal enhancer active across all contexts. This contributes to the virtual cell vision.

---

# 3. Foundational Questions

## Q1: Does Regulatory Grammar Exist as a Learnable, Rule-Based System?

**What a definitive answer looks like**: A formal characterization of grammar complexity (regular? context-free? context-sensitive?) with quantitative bounds on how much expression variance grammar explains beyond vocabulary. A number — "grammar carries X bits of information per enhancer" — that the field can build on.

## Q2: How Does Grammar Vary Across Species?

**What a definitive answer looks like**: Not "are grammar positions conserved?" but "do grammar RULES transfer functionally?" A grammar transfer matrix showing how well rules from Species A predict arrangement effects in Species B. A grammar phylogeny showing how grammar similarity relates to evolutionary distance.

## Q3: What Do Models Actually Learn About Grammar?

**What a definitive answer looks like**: Explicit extraction of grammar rules from each of 9 architectures. Proof of whether they converge. Identification of which model's grammar is closest to MPRA ground truth. Knowledge of WHERE in each model grammar is encoded and WHETHER it's encoded linearly.

## Q4: What Determines Grammar?

**What a definitive answer looks like**: Causal decomposition showing: X% is biophysics (DNA shape, nucleosome competition, cooperativity energetics), Y% is TF identity (structural class determines grammar type), Z% is binding strength (weak sites need precise grammar, strong sites don't), W% is irreducible higher-order context. Plus a phase diagram showing when grammar "turns on" as a function of motif count and density.

---

# 4. Resources: Models & Datasets

## 4.1 Pretrained Models

### Model 1: Enformer
- **Architecture**: Transformer with dilated convolutions
- **Input**: 196,608 bp DNA sequence
- **Output**: 896 bins x 5,313 tracks (human)
- **HuggingFace**: `EleutherAI/enformer-official-rough`
- **Key property**: Long-range attention can encode distant motif dependencies

### Model 2: Borzoi
- **Architecture**: Transformer, larger than Enformer
- **Input**: 524,288 bp DNA sequence
- **Output**: 6,144 bins x 7,611 tracks
- **Repository**: `https://github.com/calico/borzoi`
- **Note**: Requires ~40GB VRAM

### Model 3: Sei
- **Architecture**: Deep CNN (no attention)
- **Input**: 4,096 bp DNA sequence
- **Output**: 21,907 chromatin profiles -> 40 sequence classes
- **Repository**: `https://github.com/FunctionLab/sei-framework`

### Model 4: Nucleotide Transformer (NT)
- **Architecture**: Transformer foundation model
- **Version**: v2-500M
- **HuggingFace**: `InstaDeepAI/nucleotide-transformer-v2-500m-multi-species`

### Model 5: DNABERT-2
- **Architecture**: BERT-style transformer with BPE tokenization
- **HuggingFace**: `zhihan1996/DNABERT-2-117M`

### Model 6: HyenaDNA
- **Architecture**: Hyena operator (long convolution, sub-quadratic)
- **HuggingFace**: `LongSafari/hyenadna-large-1m-seqlen-hf`

### Model 7: Evo
- **Architecture**: StripedHyena (7B parameters)
- **HuggingFace**: `togethercomputer/evo-1-131k-base`
- **Note**: Requires ~28GB VRAM

### Model 8: Caduceus
- **Architecture**: BiMamba (bidirectional SSM)
- **HuggingFace**: `kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16`

### Model 9: GPN
- **Architecture**: Convolutional + masked language model
- **HuggingFace**: `songlab/gpn-msa-sapiens`

### Model Categorization

| Model | Type | Has Expression Head | Grammar Analysis Strategy |
|-------|------|-------------------|--------------------------|
| Enformer | Seq->Activity | Yes (5313 tracks) | Direct CAGE predictions |
| Borzoi | Seq->Activity | Yes (7611 tracks) | Direct CAGE predictions |
| Sei | Seq->ChromatinClass | Yes (40 classes) | Sequence class vector |
| NT | Foundation (MLM) | No | Linear probe |
| DNABERT-2 | Foundation (MLM) | No | Linear probe |
| HyenaDNA | Foundation (LM) | No | Linear probe |
| Evo | Foundation (AR) | No | Likelihood + probe |
| Caduceus | Foundation (biSSM) | No | Linear probe |
| GPN | Foundation (MLM) | No | Likelihood + probe |

## 4.2 MPRA Datasets

| Dataset | Species | Cell Types | Scale | Key Value |
|---------|---------|-----------|-------|-----------|
| Agarwal et al. 2023 | Human | K562, HepG2 | ~700K sequences | Large-scale human |
| Kircher et al. 2019 | Human | HepG2, K562 | ~20K variants | Saturation mutagenesis |
| de Almeida et al. 2024 | Human | Neural progenitors | ~2K perturbations | Differentiation context |
| Vaishnav et al. 2022 | Yeast | Standard conditions | 100M sequences | Massive scale |
| Jores et al. 2021 | Plant | Leaf protoplasts | ~100K sequences | Cross-kingdom |
| Klein et al. 2020 | Drosophila | S2 cells | ~5K variants | Invertebrate |

### Cross-Species Coverage

| Species | Kingdom | Divergence from Human |
|---------|---------|----------------------|
| Human | Animal (Mammal) | 0 |
| Drosophila | Animal (Insect) | ~800 million years |
| Yeast | Fungi | ~1 billion years |
| Arabidopsis/Maize | Plant | ~1.5 billion years |

---

# 5. Environment Setup

## 5.1 Conda Environment

Python 3.10 with PyTorch 2.1, transformers 4.36, enformer-pytorch, biopython, pyfaidx, MEME suite, tensorly, scikit-learn, statsmodels, matplotlib, seaborn, etc.

## 5.2 Directory Structure

```
gramlang/
├── RESEARCH_PLAN.md
├── EXPERIMENT_LOG.md
├── RESULTS.md
├── environment.yml
├── config/
├── data/{raw,processed,motifs,genome}/
├── src/{models,perturbation,grammar,transfer,decomposition,design,utils}/
├── notebooks/
├── scripts/
├── results/{module1-6,figures}/
└── manuscript/
```

## 5.3 Memory Budget (150-200GB cap)

| Component | Max Size | Strategy |
|-----------|----------|----------|
| Model weights (all 9) | ~30GB | Sequential loading, not simultaneous |
| Raw MPRA data | ~50GB | Vaishnav subsampled to 1M |
| Processed data | ~20GB | Parquet format |
| Embedding caches | ~40GB | Per-model, delete after use |
| Grammar Rule Database | ~10GB | Compressed |
| Results & figures | ~5GB | |
| **Total** | **~155GB** | Under 200GB cap |

---

# 6. Module 1: Grammar Existence & Complexity Classification

**Duration**: Weeks 4-6 | **Compute**: ~80 GPU-hours

## Experiment 1.1: Grammar Sensitivity Census
- Compute Grammar Sensitivity Index (GSI) for every enhancer in every dataset
- GSI = Var(expression across vocabulary-preserving shuffles) / original expression
- 100 shuffles per enhancer per model
- Filter: enhancers with >=2 motifs

## Experiment 1.2: Grammar Information Content
- Quantify bits of information grammar carries per enhancer
- Entropy of shuffle distribution, grammar specificity metric

## Experiment 1.3: Formal Complexity Classification
- Classify grammar on Chomsky hierarchy (regular / context-free / context-sensitive)
- Based on compositionality gap growth rate vs motif count
- Constant growth -> Regular, Linear -> Context-free, Exponential -> Context-sensitive

---

# 7. Module 2: Cross-Architecture Grammar Rule Extraction

**Duration**: Weeks 5-8 | **Compute**: ~150 GPU-hours

## Experiment 2.1: Causal Grammar Rule Extraction
- For each grammar-sensitive enhancer, extract pairwise rules
- Spacing scan (2-50bp), orientation scan (4 configurations)
- Compute helical phase score (10.5bp periodicity)

## Experiment 2.2: Cross-Model Grammar Consensus
- Correlation of spacing profiles across models
- Agreement on optimal spacing (+-2bp tolerance)
- Agreement on optimal orientation
- Overall consensus score

## Experiment 2.3: Grammar Representation Geometry
- Find "grammar direction" in embedding space per model
- Test linearity with logistic probe (correct vs shuffled)
- Layer sweep: which layer encodes grammar most strongly
- Cross-model alignment via Procrustes analysis

## Experiment 2.4: Evo Naturalness Oracle
- Test whether Evo's sequence likelihood correlates with grammar quality
- Compare log-likelihood of original vs shuffled sequences

---

# 8. Module 3: Compositionality Testing

**Duration**: Weeks 9-11 | **Compute**: ~60 GPU-hours

## Experiment 3.1: Pairwise -> Higher-Order Prediction
- Core compositionality test
- Generate 500 random arrangements per enhancer (k>=3 motifs)
- Compare pairwise-predicted expression to model prediction
- Compositionality gap = 1 - R^2(pairwise, model)

## Experiment 3.2: Grammar Tensor Factorization
- CP decomposition of arrangement->expression tensor
- Find minimum rank for >90% explained variance
- Grammar rank = intrinsic complexity of the grammar

---

# 9. Module 4: Cross-Species Grammar Transfer

**Duration**: Weeks 10-12 | **Compute**: ~80 GPU-hours

## Experiment 4.1: Cross-Species Grammar Transfer
- Build predictive grammar model from source species rules
- Apply to predict arrangement effects in target species
- Compute full 4x4 transfer matrix

## Experiment 4.2: Grammar Phylogenetics
- Convert transfer matrix to distance matrix
- Build grammar similarity tree (UPGMA)
- Compare to known sequence phylogeny

---

# 10. Module 5: Causal Determinants of Grammar

**Duration**: Weeks 13-16 | **Compute**: ~80 GPU-hours

## Experiment 5.1: The Biophysics Residual
- Compute biophysical features: DNA shape, nucleosome affinity, GC content, dinucleotide frequencies
- Fit gradient boosting model: biophysics -> grammar effect
- Report explained fraction

## Experiment 5.2: TF Structural Class as Grammar Predictor
- Map motifs to TF structural classes (bHLH, bZIP, zinc finger, etc.)
- Test whether structural class pair predicts grammar type
- Random forest classification

## Experiment 5.3: Grammar-Strength Interaction
- Test: do weak motifs need more precise grammar?
- Stratify by PWM score quartiles, compare GSI

## Experiment 5.4: Grammar Phase Diagram
- 2D grid: motif count x motif density
- Map mean GSI across grid
- Find critical thresholds

---

# 11. Module 6: Grammar-Optimized Design & Completeness

**Duration**: Weeks 17-20 | **Compute**: ~50 GPU-hours

## Experiment 6.1: Grammar-Aware Sequence Design
- Three strategies: expression-only, grammar-only, joint
- Compare predicted expression across strategies
- 1000 candidates per strategy

## Experiment 6.2: Grammar Completeness - The Ceiling Test
- Five levels: vocabulary only, +simple grammar, +full grammar, full model, MPRA replicate
- Grammar contribution = (level3 - level1) / (level5 - level1)

---

# 12. Statistical Framework

## Permutation Tests
- All claims tested against rigorous null models
- 10,000 permutations minimum
- Benjamini-Hochberg FDR correction at q=0.05

## Tests Per Claim

| Claim | Null Hypothesis | Test | Threshold |
|-------|----------------|------|-----------|
| Grammar exists | Expression invariant to arrangement | Permutation test on GSI | p < 0.001 (Bonferroni) |
| Grammar is compositional | Higher-order terms = noise | Bootstrap CI on gap | 95% CI excludes 0 |
| Models agree | Rules are model-specific | Permutation on model labels | p < 0.01 |
| Grammar transfers | Species labels uninformative | Permutation on species labels | p < 0.01 |
| Biophysics explains | Features uninformative | Permutation on features | p < 0.01 |
| Structure predicts grammar | Structure uninformative | Permutation on labels | p < 0.01 |
| Grammar-strength tradeoff | Independence | Partial correlation | p < 0.01 |
| Grammar optimization helps | Same as random | Two-sample t-test | p < 0.05 |

---

# 13. Output Specifications

## Key Figures (7 main figures for publication)

| Figure | Title | Module |
|--------|-------|--------|
| Fig 1 | Grammar Sensitivity Census | Module 1 |
| Fig 2 | Grammar Complexity Classification | Modules 1,3 |
| Fig 3 | Cross-Architecture Grammar Consensus | Module 2 |
| Fig 4 | Grammar Representation Geometry | Module 2 |
| Fig 5 | Cross-Species Grammar Transfer | Module 4 |
| Fig 6 | Causal Determinants of Grammar | Module 5 |
| Fig 7 | Grammar Completeness & Design | Module 6 |

## Supplementary Tables (S1-S8)

Full GSI scores, Grammar Rule Database, consensus scores, transfer matrix, biophysical importances, TF structure map, designed sequences, statistical tests.

---

# 14. Timeline & Milestones

| Week | Phase | Milestone |
|------|-------|-----------|
| 1-3 | Setup | All models running, data preprocessed, probes trained |
| 4-6 | Module 1 | GSI census complete, complexity classification |
| 5-8 | Module 2 | Grammar Rule Database, consensus scores |
| 9-11 | Module 3 | Compositionality gap curve, grammar rank |
| 10-12 | Module 4 | Transfer matrix, grammar phylogeny |
| 13-16 | Module 5 | Causal decomposition complete |
| 17-20 | Module 6 | Design comparison, completeness metric |
| 21-24 | Integration | Figures, manuscript, code release |

---

# 15. Expected Results & Decision Tree

## Scenario A: Grammar Is Strong and Universal
GSI > 0.3 for >50%, consensus > 0.7, gap < 0.2, transfer > 0.5
-> Transformative: "periodic table" of regulatory grammar

## Scenario B: Grammar Is Weak and Model-Dependent
GSI < 0.1 for >70%, consensus < 0.3, gap > 0.5
-> Valuable negative: billboard model validated

## Scenario C: Grammar Is Strong but Non-Compositional
GSI > 0.3, gap > 0.5, grammar order > 4
-> Context-sensitive: deep learning fundamentally needed

## Scenario D: Grammar Is Species-Specific
Transfer < 0.1 cross-species, within-species > 0.5
-> Universal rules don't exist

## Scenario E: Grammar Is Biophysics
Biophysics fraction > 0.9
-> Grammar is physics, not abstract rules

---

# 16. Publication Strategy

**Target**: Nature Methods (primary), Genome Research, Nature Genetics, Genome Biology (alternatives)

**Title**: "The Formal Structure of Regulatory Grammar: Complexity, Compositionality, and Universality Across Species and Models"

**Key claims to support**:
1. Grammar is [regular/context-free/context-sensitive], carrying X bits per enhancer
2. Nine architectures converge on X% of rules
3. Grammar from species A predicts Y% in species B
4. Z% explained by biophysics, W% irreducible
5. Grammar-aware design improves by V%
