# GRAMLANG Manuscript Outline

**Working Title**: The Standard Computational Method for Measuring Regulatory Grammar Is Confounded by Spacer DNA Composition

**Narrative Strategy**: v3-first — lead with the spacer confound discovery, then present v1/v2 results as context.

---

## Abstract (~250 words)

- Problem: Whether regulatory DNA follows compositional grammar rules is debated. Vocabulary-preserving shuffles are the standard computational approach.
- Approach: Systematic study across 3 foundation models (DNABERT-2, NT v2-500M, HyenaDNA) + Enformer, 5 MPRA datasets, 6 analysis modules.
- Key finding: 78-86% of grammar sensitivity index (GSI) variance comes from spacer DNA composition changes, not motif arrangement.
- Positive control: When spacers are controlled, models DO detect grammar (p < 1e-117).
- Implication: The field needs spacer-controlled experimental designs to study regulatory grammar computationally.

---

## 1. Introduction (~800 words)

- Regulatory grammar hypothesis: motif arrangement affects expression
- Prior computational approaches: vocabulary-preserving shuffles (de Almeida et al., Agarwal et al.)
- Foundation models as expression predictors
- Gap: No systematic evaluation of what VP shuffles actually measure
- Our contribution: The standard method is confounded; grammar IS real but needs better measurement

---

## 2. Results

### 2.1 Spacer DNA Dominates Grammar Sensitivity Measurements
**[Figure 1]**

- Factorial decomposition (P1.1): position, orientation, spacer, interaction
- Spacer accounts for 78-86% of full shuffle variance (Agarwal: 83%, Jores: 78%)
- Motif position: 28-43%; Orientation: 19-26%; these overlap heavily with spacer
- Spacer ablation (P2.4): GC shift > dinucleotide shuffle > motif rearrangement
- Feature decomposition (P3.3): GC content + dinucleotides explain 40-80% of model predictions
- GC-expression correlation reverses across species (+0.63 human, -0.73 plant)

### 2.2 Grammar IS Real: Positive Control
**[Figure 2]**

- Georgakopoulos-Soares orientation data with controlled spacers
- DNABERT-2 detects orientation effects (p < 1e-117, mean |delta| = 0.062)
- 17% of pairs show > 0.1 absolute difference
- Conclusion: models ARE grammar-sensitive; the VP shuffle method obscures this

### 2.3 Grammar Sensitivity Census (Corrected)
**[Figure 3]**

- v2 census: 7,650 GSI measurements across 3 models x 5 datasets x 500 enhancers
- v1 artifact: 100% significant (F-test with zero noise) -> corrected to 8.3% nominal, 0.17% FDR
- Dataset hierarchy: Klein > Agarwal > Jores > Vaishnav > de Almeida
- Cross-model agreement: Spearman rho = 0.65-0.90 for 3/5 datasets
- ANOVA: Dataset explains 29% of variance, Model 4.5%
- Enformer anti-correlates with foundation models for Klein (rho = -0.40 to -0.45)

### 2.4 Grammar Is Non-Compositional
**[Figure 4]**

- Compositionality gap ~0.989: pairwise rules explain ~1% of higher-order effects
- Constant across k=3-7 (BIC favors constant model)
- 77.5% of interactions are non-additive (v2 factorial design)
- Classification: context-sensitive in Chomsky hierarchy

### 2.5 Grammar Does Not Transfer Across Species
**[Figure 5]**

- 3x3 transfer matrix: all cross-species R^2 = 0.000
- Within-species: human R^2 = 0.151, plant R^2 = 0.212, yeast R^2 = 0.004
- Phylogenetic distances all = 1.0 (maximum)
- Within-species GSI distributions 2x more similar than cross-species (p = 0.035)
- Only helical phasing is conserved (d ~ 0.01-0.03)

### 2.6 Grammar Completeness Ceiling
**[Figure 6]**

- Vocabulary explains 5-15% of expression variance
- Grammar adds at most 1.8% (Klein)
- Completeness: 6-18% of replicate ceiling
- 82-94% gap due to features beyond motif-centric grammar

---

## 3. Discussion (~1000 words)

### 3.1 The Spacer Confound
- VP shuffles change spacer composition as a side effect
- Foundation models are highly sensitive to GC content and dinucleotides
- Measured "grammar" is primarily a spacer composition artifact
- Prior studies using VP shuffles should be reinterpreted

### 3.2 Implications for the Field
- Need spacer-controlled shuffle designs (swap motifs, keep spacers identical)
- Positive control demonstrates grammar IS real when properly measured
- GC sensitivity reversal across species (-0.76 plant vs +0.63 human) is biologically meaningful

### 3.3 What Grammar Does Exist
- Non-compositional (context-sensitive): higher-order interactions dominate
- Species-specific: no transfer between kingdoms
- Weak contributor: 6-18% of replicate ceiling
- "Flexible billboard" model: motif identity >> arrangement

### 3.4 Limitations
- Only 3 foundation models + Enformer (Caduceus, GPN, Evo not tested)
- Probes are weak (median R^2 = 0.17) — may underestimate grammar in some datasets
- 100 shuffles may underpower individual-enhancer significance tests
- Power analysis shows significance rate saturates at ~11% even with 1000 shuffles
- Only FIMO-defined motifs; non-canonical binding sites not captured

---

## 4. Methods

### 4.1 MPRA Datasets
- Agarwal et al. 2023 (K562, human), Klein et al. 2020 (HepG2, human), de Almeida et al. 2024 (neural, human), Vaishnav et al. 2022 (yeast), Jores et al. 2021 (plant)
- Preprocessing: sequence extraction, expression normalization

### 4.2 Foundation Models
- DNABERT-2 (117M params, 768-dim, 12 layers)
- Nucleotide Transformer v2-500M (498M, 1024-dim, 25 layers)
- HyenaDNA (6.5M, 256-dim, 10 layers)
- Enformer (251M, native CAGE head, 196,608bp context)
- PARM (MPRA-trained CNN, comparison only)

### 4.3 Expression Probes
- 2-layer MLP: input_dim -> 256 -> ReLU -> Dropout(0.1) -> 1
- Trained on frozen embeddings, 80/10/10 split
- AdamW optimizer, lr=1e-3, weight_decay=1e-4, MSE loss
- Early stopping with patience=10
- Viability threshold: Pearson r > 0.3

### 4.4 Vocabulary-Preserving Shuffles
- Motif scanning: FIMO v5.5.7, p < 1e-4, JASPAR 2024
- Shuffle: random reassignment of motif positions/orientations
- Spacer fill: dinucleotide-shuffled DNA
- 100 shuffles per enhancer (50 default in pipeline, extended for v2)
- v3 factorial variants: position-only, orientation-only, spacer-only

### 4.5 Grammar Sensitivity Index
- GSI = std(expression across shuffles) / |mean(expression)|
- v2 correction: z-score-based p-values (not F-test)
- v3 robust variants: GES (median/MAD), GPE (dynamic range)

### 4.6 Statistical Framework
- Permutation-based p-values for individual enhancers
- Benjamini-Hochberg FDR correction
- Bootstrap 95% CIs (10,000 resamples)
- Two-way ANOVA for variance decomposition
- Spearman rank correlations for cross-model agreement

---

## Figure Plan

| Figure | Title | Panels | Data Source |
|--------|-------|--------|-------------|
| **Fig 1** | Spacer Confound Discovery | (A) Factorial decomposition bar chart, (B) Spacer ablation effect sizes, (C) Feature decomposition R^2 by feature class, (D) GC-expression correlation reversal | v3/factorial_decomposition/, v3/spacer_ablation/, v3/feature_decomposition/ |
| **Fig 2** | Positive Control: Grammar Is Real | (A) Orientation effect distribution, (B) p-value / effect size, (C) Comparison: VP shuffle vs controlled design | v3/positive_control/ |
| **Fig 3** | GSI Census (Corrected) | (A) GSI distribution by model, (B) GSI by dataset, (C) Significance correction (100% -> 8.3% -> 0.17%), (D) Cross-model agreement heatmap | module1/, v2 corrections |
| **Fig 4** | Compositionality & Complexity | (A) Gap vs k, (B) Chomsky classification, (C) Interaction strength distribution | module3/ |
| **Fig 5** | Cross-Species Transfer | (A) Transfer R^2 heatmap, (B) Phylogenetic distance, (C) Within vs cross-species similarity | module4/ |
| **Fig 6** | Completeness Ceiling | (A) Hierarchical R^2 decomposition, (B) Completeness percentages | module6/ |
| **Fig S1** | Probe Quality & PARM Comparison | (A) Probe R^2 by model x dataset, (B) PARM vs foundation model GSI | probes/, v3/parm_comparison/ |
| **Fig S2** | Power Analysis | (A) Significance rate vs n_shuffles, (B) BOM null model comparison | v3/power_analysis/, v3/bom_baseline/ |

---

## Supplementary Tables

| Table | Content | Source |
|-------|---------|--------|
| **Table S1** | Full GSI census (per model x dataset) | module1/ |
| **Table S2** | Probe training metrics (all 18 combinations) | probes/ |
| **Table S3** | v2 significance rates per combination | RESULTS.md v2 erratum |
| **Table S4** | Grammar rules database summary | module2/ |
| **Table S5** | Biophysics R^2 (raw vs robust GSI) | module5/, v2 corrections |
| **Table S6** | v3 factorial decomposition full results | v3/factorial_decomposition/ |

---

## Key Sentences for Abstract/Discussion

1. "The standard computational approach to measuring regulatory grammar is fundamentally confounded: 78-86% of grammar sensitivity variance comes from spacer DNA composition, not motif arrangement."
2. "A positive control on experimentally designed sequences with controlled spacers demonstrates that foundation models ARE sensitive to grammar (p < 1e-117)."
3. "After correcting statistical artifacts, only 8.3% of enhancers show nominally significant grammar sensitivity (0.17% FDR-corrected), down from 100% in the initial analysis."
4. "Grammar is non-compositional (compositionality gap = 0.989), species-specific (zero cross-species transfer), and explains at most 6-18% of the replicate ceiling."
