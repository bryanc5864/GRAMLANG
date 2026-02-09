# GRAMLANG v3: Extension Plan — Redefining Regulatory Grammar

**Date**: 2026-02-05
**Status**: PROPOSED
**Goal**: Address fundamental gaps in v1/v2 to produce a definitive computational characterization of regulatory grammar

---

## Part 1: Critical Gap Analysis

### A. Fundamental Conceptual Issues

#### Gap 1: GSI Measures the Wrong Thing
**Problem**: GSI = σ_shuffle / |μ_shuffle| measures the *coefficient of variation* of shuffled expressions, not how special the natural arrangement is. A high GSI means shuffles vary a lot relative to their mean — but says nothing about whether the natural arrangement is optimal.

The z-score (distance of natural arrangement from shuffle mean) is the actual grammar signal. And z-scores are significant for only 8.3% of enhancers (0.17% after FDR). We've been reporting GSI as if it measures grammar strength, but it measures arrangement *sensitivity* — a subtly different thing.

**Impact**: All Module 1 findings are framed around GSI when they should be framed around z-scores and effect sizes. The "median GSI = 0.13" finding means shuffles vary by 13%, not that grammar contributes 13%.

**Fix**: Redefine the primary metric. Report (a) effect size: |original_expr - shuffle_mean| / shuffle_std (z-score), (b) fraction of expression variance attributable to arrangement (partial η²), and (c) the practical effect: how much expression changes from best-to-worst arrangement (grammar potential from Experiment G).

#### Gap 2: Vocabulary-Preserving Shuffles Don't Purely Isolate Grammar
**Problem**: The shuffle procedure simultaneously changes:
1. Motif positions (grammar — intended)
2. Motif orientations (grammar — intended, but random 50% flip)
3. Spacer DNA sequences (NOT grammar — dinucleotide-shuffled spacers)
4. Local sequence context around motifs (NOT grammar — flanking changes)

The expression change after shuffling conflates grammar effects with spacer/context effects. If a model uses spacer sequence (which transformers certainly can), the measured "grammar sensitivity" includes non-grammar signal.

**Impact**: GSI is an upper bound on grammar effect, not a precise measurement. True grammar contribution could be substantially lower.

**Fix**: Design controlled shuffles that isolate individual factors:
- **Position-only shuffles**: Permute motif positions, keep orientations and spacer sequences fixed (shift motifs within existing spacer DNA)
- **Orientation-only shuffles**: Flip motif orientations without changing positions
- **Spacer-only shuffles**: Keep motif positions/orientations fixed, only reshuffle spacer DNA
- **Factorial decomposition**: Measure each contribution independently and their interactions

#### Gap 3: Expression Probes Are Too Weak
**Problem**: Only 9/15 probes are viable (R² > 0.05). Even the best probe (R² = 0.34) captures only 34% of expression variance. We measure grammar through a very noisy lens. The 2-layer MLP with frozen embeddings is the simplest possible architecture.

**Impact**: Weak probes add noise that could either inflate GSI (noisy predictions → variable shuffles) or mask real grammar (true signal drowned out). The zero correlation between probe R² and GSI (ρ = -0.03) is reassuring but doesn't rule out systematic bias.

**Fix**:
- Train deeper probes (3-4 layers, attention pooling instead of mean pooling)
- Fine-tune the last 2-4 layers of foundation models on expression data (parameter-efficient with LoRA)
- Use PARM-style MPRA-native models as an additional architecture (avoids the probe problem entirely)
- Compare GSI from probes with different R² to bound the probe noise contribution

#### Gap 4: 100 Shuffles Is Underpowered
**Problem**: The ANOVA power analysis shows 3/5 datasets are underpowered (29.3% power). With 100 shuffles, the standard error of the shuffle mean is σ/√100 = σ/10. For a z-score of 1.96 to be significant, we need |orig - μ| > 1.96 × σ/√100. The effective power to detect a 0.5σ grammar effect is only ~50%.

**Impact**: Explains why only 8.3% of enhancers are significant — we may simply lack power, not grammar.

**Fix**: Run 1,000 shuffles for a subset of enhancers (100 per dataset) to establish the power curve. Calculate the required sample size for 80% power at each observed effect size. If 1,000 shuffles yields 30-50% significance, the "grammar is rare" conclusion changes to "grammar is common but small."

#### Gap 5: Motif Scanning Is Incomplete
**Problem**: Only ~150 of 879 vertebrate JASPAR motifs are scanned (speed constraint). FIMO p < 1e-4 is stringent. We may be missing many relevant motifs, which means:
- "Vocabulary-preserving" shuffles only preserve a subset of actual vocabulary
- Motif pairs driving grammar interactions may be absent from our analysis
- Enhancers classified as "single-motif" (excluded from analysis) may actually have multiple undetected motifs

**Impact**: Potentially large — if key interacting TF pairs are missed, grammar will appear weaker than it is.

**Fix**: Run full JASPAR scan (all 879 vertebrate + species-specific databases). Use relaxed p-threshold (1e-3) with post-hoc FDR correction. Compare GSI with full vs. partial motif scanning to quantify the impact.

#### Gap 6: No Experimental Validation (and Synthetic Probes FAIL)
**Problem**: All findings are in silico. Experiment B (synthetic grammar probes) performed WORSE than random baseline (accuracy 0.331 vs 0.355), which is a red flag that extracted grammar rules don't capture real biology.

**Impact**: If rules don't predict even synthetic probe outcomes, the entire grammar rule framework may be capturing model artifacts rather than biology.

**Fix**:
- Compare in silico grammar predictions against actual MPRA measurements from the same datasets (we have Agarwal, Klein, de Almeida, Vaishnav, Jores — all with measured expression)
- For enhancers with multiple measured variants in MPRA data, test whether arrangement-similar variants have similar expression
- Cross-validate: train grammar model on 80% of MPRA data, predict expression of remaining 20% from grammar features alone

#### Gap 7: Grammar Is Model-Dependent
**Problem**: Different architectures detect different grammar. For de_almeida, cross-model correlation is ρ = -0.06 (complete disagreement). Klein anti-correlates with Enformer (ρ = -0.45). If grammar were a real sequence property, all models should agree.

**Impact**: Much of what we call "grammar" may be architecture-specific learned features, not biological signal.

**Fix**:
- Focus on the **intersection** of model predictions: enhancers significant in ALL models (currently 0-1 per dataset — need more power first)
- Use ensemble agreement as the grammar metric: GSI_consensus = mean(GSI_model1, GSI_model2, GSI_model3) weighted by probe quality
- Add PARM (Nature 2026) and ChromBPNet as additional independent architectures
- Test whether consensus-significant enhancers have different biological properties than model-specific ones

#### Gap 8: We Don't Decompose the 78-94% Unexplained Variance
**Problem**: ANOVA shows vocabulary explains 8-22%, grammar 0-1.6%, and 78-94% is unexplained. What IS that unexplained variance? Without understanding it, we can't assess whether grammar is genuinely small or whether we're measuring it wrong.

**Impact**: The "grammar is weak" conclusion rests on this number. If the unexplained variance is due to higher-order motif interactions (which are technically grammar), then grammar is actually large but our pairwise framework can't capture it.

**Fix**:
- Decompose unexplained variance into: (a) spacer sequence effects, (b) higher-order motif interactions (k≥3), (c) motif-flanking context effects, (d) long-range sequence features, (e) technical noise
- Use deep learning model embeddings as "full grammar" features in ANOVA (compare to our hand-crafted grammar features)
- The gap between "grammar features" and "full model" (0.04-0.28 R²) represents grammar-like information the model captures but our features don't — characterize what this is

---

### B. Methodological Gaps

#### Gap 9: Rule Extraction Is Buggy
**Problem**: The +/+ orientation artifact (83.3%) corrupts all downstream analyses. The spacing optimization uses only +/+ orientation, argmax favors first element on ties, fallback defaults to +/+. This is a known bug from the orientation bias investigation.

**Impact**: Modules 2, 3, 5, 6 all use rules from the buggy extractor. The compositionality gap, TF structure prediction, and completeness ceiling all inherit this bias.

**Fix**: Rewrite rule_extraction.py:
1. Optimize spacing independently per orientation (4× more compute but correct)
2. Use permutation test instead of argmax for orientation selection
3. Randomize testing order to eliminate first-element bias
4. Remove +/+ fallback — use "undetermined" when sensitivity is below threshold
5. Re-run Modules 2-6 with corrected rules

#### Gap 10: No Comparison to Bag-of-Motifs Baseline
**Problem**: The BOM paper (2025, Nature Communications) shows simple unordered motif counts outperform complex deep learning models for predicting cell-type-specific enhancers across multiple species. We never test whether our grammar model outperforms this trivial baseline.

**Impact**: If bag-of-motifs matches or exceeds grammar model performance, it's definitive evidence that grammar doesn't matter for expression prediction — which is actually a key finding to report.

**Fix**: Implement bag-of-motifs baseline:
- Feature vector: [count(motif_1), count(motif_2), ..., count(motif_N)]
- Train same regression model (random forest) on expression
- Compare R² of: vocabulary-only, +grammar features, +bag-of-motifs
- If BOM ≥ grammar features: "grammar adds nothing beyond vocabulary" is confirmed with proper baseline

#### Gap 11: No Comparison to PARM/ChromBPNet/HERMES
**Problem**: PARM (Nature 2026) claims "gene regulation is far more predictable than previously believed" and can design synthetic promoters. HERMES (bioRxiv 2026) distills an "interpretable rulebook." ChromBPNet discovers compact TF motif lexicons and cooperative syntax. We don't compare to any of these.

**Impact**: If PARM captures grammar that our models miss, our low completeness (7-17%) reflects probe limitations, not biology. If PARM also shows low grammar completeness, it's converging evidence.

**Fix**:
- Download PARM model, run grammar analysis pipeline on it (it's MPRA-trained, so expression probes are unnecessary)
- Compare PARM GSI distribution to foundation model GSI
- If PARM shows higher grammar sensitivity, the problem is our probes; if similar, grammar is genuinely weak
- Compare HERMES "rulebook" grammar rules to our extracted rules

#### Gap 12: Missing Condition-Dependent Grammar
**Problem**: Literature shows grammar is condition-dependent:
- Plant enhancers: cooperative in light, additive in dark (Plant Cell 2024)
- Drosophila: housekeeping = additive, developmental = super-additive (Nature Communications 2024)
- PARM tests 10 cell types + stimuli

Our analysis uses a single condition per dataset. Grammar might be stimulus-responsive.

**Impact**: By measuring grammar in a single state, we may miss context-dependent rules that are strong in specific conditions.

**Fix**: Use Enformer's multi-track output to compare grammar across cell types. Enformer predicts 5,313 CAGE tracks (different cell types/conditions). For each enhancer, compute GSI using different CAGE tracks and test whether grammar sensitivity varies across cell types.

---

### C. Missing Analyses (New Experiments)

#### Gap 13: No Per-Motif-Pair Grammar Profiling
**Problem**: We analyze grammar at the enhancer level (GSI across all shuffles) but the literature shows grammar is motif-pair-specific. ETS-AP-1 pairs synergize at short distances; other pairs are independent. We need to know WHICH specific pairs have grammar.

**Fix**: For each of the ~5,000 unique motif pairs in our database:
- Compute pair-specific grammar effect (expression change when this specific pair's arrangement changes)
- Build a "grammar matrix" of TF × TF interaction strengths
- Identify the top 50 "grammar pairs" that drive most grammar effects
- Cross-reference with known cooperative TF pairs (literature validation)

#### Gap 14: No Active Learning / Targeted Probe Design
**Problem**: Friedman et al. (2023) showed active learning doubles grammar model performance. We passively analyze existing sequences rather than designing informative ones.

**Fix**: Implement active learning loop:
1. Identify 100 maximally grammar-uncertain enhancers
2. For each, generate 50 systematically varied arrangements (not random shuffles — designed factorial experiments)
3. Predict expression for all arrangements
4. Use the variance across designed arrangements (not random shuffles) as grammar signal
5. This gives much more informative grammar measurements than random shuffling

#### Gap 15: No Chromatin Context Integration
**Problem**: Grammar might depend on chromatin state. An enhancer in open chromatin might follow different grammar rules than one in closed chromatin. We ignore this entirely.

**Fix**: Use Enformer's predicted chromatin accessibility tracks. For each enhancer:
- Predict ATAC-seq/DNase-seq signal in addition to expression
- Test whether grammar sensitivity correlates with predicted accessibility
- Stratify enhancers by predicted chromatin state and compare grammar within each stratum

#### Gap 16: The Spacer Problem
**Problem**: We treat spacer DNA as inert filler, but models likely use spacer sequence features. Dinucleotide-shuffled spacers change the sequence context around motifs, potentially creating or destroying binding sites, altering DNA shape, or disrupting higher-order features.

**Fix**: Design "spacer ablation" experiment:
1. Natural arrangement with natural spacers (baseline)
2. Natural arrangement with shuffled spacers (isolates spacer contribution)
3. Shuffled arrangement with natural spacers (isolates position contribution)
4. Full factorial: measure spacer × position interaction
5. This directly quantifies how much of "grammar sensitivity" is actually spacer sensitivity

---

## Part 2: Extension Plan (Prioritized)

### Priority 0: Critical Fixes (Must-Do Before Any New Analysis)

#### P0.1: Fix Rule Extraction Bugs
- Rewrite `rule_extraction.py` to fix orientation bias
- Independent spacing optimization per orientation
- Permutation-based orientation selection
- Randomized testing order
- "Undetermined" category for low-sensitivity rules
- **Effort**: 1 day coding, 1 day re-running Module 2
- **Impact**: Fixes corruption in Modules 2-6

#### P0.2: Redefine Primary Grammar Metric
- Shift from GSI (coefficient of variation) to a multi-metric framework:
  - **Grammar Effect Size (GES)**: |original_expr - shuffle_median| / MAD(shuffles) — robust z-score
  - **Grammar Variance Fraction (GVF)**: η² from ANOVA (what fraction of expression variance is arrangement)
  - **Grammar Practical Effect (GPE)**: (max_shuffle - min_shuffle) / |median_shuffle| — potential dynamic range
- Report all three; GES is the primary significance test, GVF is the population-level summary, GPE is the biological interpretation
- **Effort**: 0.5 days
- **Impact**: Fixes the conceptual confusion between sensitivity and effect size

#### P0.3: Increase Shuffle Count for Power
- Run 1,000 shuffles on 100 enhancers per dataset (500 total)
- Compare significance rates at 100 vs 500 vs 1,000 shuffles
- Compute power curves and determine minimum shuffles needed for 80% power
- If significance rate rises substantially with more shuffles, the "grammar is rare" conclusion is wrong
- **Effort**: ~4 hours GPU time (500 enhancers × 1,000 shuffles × 5 models = 2.5M forward passes)
- **Impact**: Directly addresses whether grammar is rare or underpowered

### Priority 1: Decomposition Experiments (High Impact)

#### P1.1: Factorial Shuffle Decomposition
- Design 4 shuffle types isolating individual grammar factors:
  1. Position-only: permute motif positions, fix orientations, keep spacer DNA
  2. Orientation-only: flip orientations, fix positions, keep spacer DNA
  3. Spacer-only: keep motifs fixed, reshuffle only spacer regions
  4. Full shuffle (current approach): all factors simultaneously
- For each: compute expression change relative to original
- ANOVA: expression ~ position + orientation + spacer + position×orientation + position×spacer + orientation×spacer
- Run on 200 enhancers × 5 datasets × 3 models with 100 shuffles each type
- **Effort**: 2 days coding, ~8 hours GPU time
- **Impact**: GAME-CHANGER — directly answers "how much is position vs orientation vs spacer vs interaction?" This is the key experiment to redefine what grammar means.

#### P1.2: Bag-of-Motifs Baseline
- Implement BOM feature vector: [count(motif_i) for i in 1..N_motifs]
- Train random forest regression on expression (same CV scheme as completeness analysis)
- Compare R² of: BOM-only, BOM+pairwise grammar, BOM+full grammar features, full DL model
- **Effort**: 0.5 days
- **Impact**: Establishes whether grammar adds anything beyond simple motif counts

#### P1.3: Unexplained Variance Decomposition
- Extract DL model embeddings (768-dim for DNABERT-2) for all enhancers
- Train linear regression on expression using: (a) vocabulary features, (b) grammar features, (c) embeddings
- The gap R²(embeddings) - R²(grammar) = "grammar information captured by DL but not by our features"
- Apply sparse autoencoders or NMF to identify what the embedding dimensions represent
- **Effort**: 1 day
- **Impact**: Explains what the 78-94% unexplained variance is, and whether it's "grammar we can't capture" or "non-grammar signal"

### Priority 2: Validation & Robustness (Required for Publication)

#### P2.1: MPRA Cross-Validation
- For each MPRA dataset, identify enhancer pairs that share motif vocabulary but differ in arrangement
- Test whether our grammar model predicts their measured expression difference
- This is the gold standard: does in silico grammar match experimentally measured grammar?
- **Effort**: 1 day (data mining + analysis, no GPU needed)
- **Impact**: First experimental validation of computational grammar predictions

#### P2.2: Full Motif Scan
- Run FIMO with all 879 vertebrate motifs + all species-specific databases
- Relaxed threshold (p < 1e-3) with BH FDR correction
- Re-compute GSI with full motif vocabulary
- Compare to current results to quantify impact of missing motifs
- **Effort**: ~2 hours FIMO + 1 day re-analysis
- **Impact**: Bounds the "missing motif" problem

#### P2.3: Probe Improvement
- Train 3-layer probes with attention pooling (instead of mean pooling + 2-layer MLP)
- LoRA fine-tuning of last 2 layers of foundation models on expression data
- Compare GSI distributions between weak probes, improved probes, and Enformer (probe-free)
- **Effort**: 1 day training, 0.5 day analysis
- **Impact**: Directly addresses probe noise concern. If improved probes change GSI distributions substantially, the "grammar is weak" conclusion needs revision.

#### P2.4: PARM Comparison
- Download PARM model from vansteensellab/PARM (GitHub)
- Run grammar pipeline on PARM predictions (no expression probe needed — PARM directly predicts promoter activity)
- Compare PARM grammar sensitivity to foundation model grammar sensitivity
- PARM was trained on actual MPRA data, so it should capture genuine regulatory grammar better than foundation models + probes
- **Effort**: 1 day setup, ~4 hours GPU, 1 day analysis
- **Impact**: If PARM shows stronger grammar, the problem is our probes. If PARM shows similar weakness, grammar is genuinely small.

### Priority 3: Novel Experiments (High Novelty)

#### P3.1: TF-Pair Grammar Atlas
- For all ~5,000 unique motif pairs, compute pair-specific grammar metrics:
  - Pair-specific GSI contribution (how much does THIS pair's arrangement drive expression change?)
  - Pair-specific distance dependence (expression vs. spacing profile)
  - Pair-specific orientation effect (fold change across 4 orientations)
- Build TF × TF grammar interaction matrix (sparse, ~5000 entries)
- Identify "grammar hotspot" pairs (top 5%) and "grammar-inert" pairs (bottom 50%)
- Cross-reference grammar hotspot pairs with:
  - Known physical TF-TF interactions (BioGRID, STRING)
  - Co-occupancy from ChIP-seq (if available)
  - Literature-validated cooperative pairs (ETS-AP-1, NANOG-OCT4, etc.)
- **Effort**: 2 days coding, ~8 hours GPU
- **Impact**: Shifts from "does grammar exist?" to "WHICH specific interactions constitute grammar?" — much more publishable and actionable

#### P3.2: Condition-Dependent Grammar via Enformer Multi-Track
- For 200 enhancers, compute GSI using 20 different Enformer CAGE tracks (different cell types)
- Test: does grammar sensitivity vary across cell types?
- Identify "constitutive grammar" (arrangement matters in all cell types) vs "conditional grammar" (arrangement matters only in specific contexts)
- **Effort**: 1 day coding, ~6 hours GPU (200 enhancers × 20 tracks × 100 shuffles × 196,608 bp)
- **Impact**: First evidence for condition-dependent grammar. If grammar is condition-specific, it explains why static analysis finds weak signal.

#### P3.3: Spacer Ablation Experiment
- For 200 enhancers, test 4 conditions:
  1. Natural motifs + natural spacers (baseline)
  2. Natural motifs + shuffled spacers
  3. Shuffled motifs + natural spacers
  4. Shuffled motifs + shuffled spacers
- Full factorial ANOVA: expression ~ motif_arrangement × spacer_identity
- **Effort**: 1 day coding, ~4 hours GPU
- **Impact**: Directly quantifies spacer contribution. If spacer_identity has large η², our "grammar sensitivity" is substantially spacer-driven.

#### P3.4: Grammar-Aware Sequence Design with Validation
- Take top 50 "grammar hotspot" enhancers (from P3.1)
- For each, computationally optimize arrangement (greedy search over spacing/orientation permutations)
- Compare: random arrangement → natural arrangement → optimized arrangement
- Measure expression gain from grammar optimization
- If gain is <5%, grammar optimization is biologically irrelevant
- If gain is >20%, grammar-aware design is practically useful
- **Effort**: 1 day
- **Impact**: Answers the applied question: "can we use grammar knowledge to design better enhancers?"

### Priority 4: Framework Extensions (Conceptual Depth)

#### P4.1: Formal Grammar Redefition
Based on all the above experiments, write a formal mathematical definition of regulatory grammar:

```
Definition: The regulatory grammar G of a cis-regulatory element e is a tuple:
  G(e) = (V, S, C, I) where:

  V = vocabulary effect (expression attributable to motif identity/count)
  S = syntax effect (expression attributable to motif arrangement)
  C = context effect (expression attributable to spacer/flanking sequence)
  I = interaction effect (V×S, V×C, S×C cross-terms)

  such that: Var(expression) = V + S + C + I + ε

Grammar strength: g(e) = S / (V + S + C + I)
Grammar type:
  - Billboard: g(e) < 0.05 (arrangement contributes <5%)
  - Soft grammar: 0.05 ≤ g(e) < 0.20
  - Strong grammar: 0.20 ≤ g(e) < 0.50
  - Enhanceosome: g(e) ≥ 0.50
```

This replaces the vague "grammar exists" / "billboard model" dichotomy with a quantitative continuous measure per enhancer, and introduces the missing "context effect" that current analyses conflate with grammar.

#### P4.2: Chromsky Hierarchy Revisited
The current classification (context-sensitive) is based on compositionality gap alone. Strengthen by:
- Testing specific predictions of each grammar class
- Regular: expression = f(motif_counts) — test with BOM baseline
- Context-free: expression = f(pairwise_interactions) — test with pairwise model
- Context-sensitive: expression = f(full_arrangement) — test with k≥3 interactions
- Measure the fraction of enhancers in each class (not all enhancers need the same grammar class)

#### P4.3: Per-Enhancer Grammar Classification
Instead of classifying all regulatory grammar as one type, classify each enhancer individually:
- For each enhancer, compute g(e) from P4.1
- Classify: billboard (<0.05), soft (0.05-0.20), strong (0.20-0.50), enhanceosome (>0.50)
- Characterize what biological features distinguish each class
- Do grammar-strong enhancers have different evolutionary conservation, chromatin states, or TF compositions?

---

## Part 3: Implementation Timeline

### Phase A: Critical Fixes (Week 1)
- P0.1: Fix rule extraction ← DAY 1-2
- P0.2: Redefine metrics ← DAY 2
- P0.3: Power analysis (1000 shuffles) ← DAY 3-4
- Write up corrected Module 1 results ← DAY 5

### Phase B: Decomposition (Week 2)
- P1.1: Factorial shuffle decomposition ← DAY 1-3
- P1.2: Bag-of-motifs baseline ← DAY 3
- P1.3: Unexplained variance decomposition ← DAY 4-5
- Update RESULTS.md with decomposition findings

### Phase C: Validation (Week 3)
- P2.1: MPRA cross-validation ← DAY 1-2
- P2.2: Full motif scan ← DAY 2-3
- P2.3: Probe improvement ← DAY 3-4
- P2.4: PARM comparison ← DAY 4-5

### Phase D: Novel Experiments (Week 4-5)
- P3.1: TF-pair grammar atlas ← DAY 1-3
- P3.2: Condition-dependent grammar ← DAY 3-4
- P3.3: Spacer ablation ← DAY 5
- P3.4: Grammar-aware design ← DAY 6

### Phase E: Framework (Week 6)
- P4.1: Formal grammar redefinition
- P4.2: Chomsky hierarchy revisited
- P4.3: Per-enhancer classification
- Final RESULTS.md and manuscript draft

---

## Part 4: Expected Outcomes & Narrative Arc

### If Grammar Is Genuinely Weak (Billboard Confirmed)
The paper becomes: "Regulatory sequences follow a flexible billboard model: a comprehensive computational dissection."
- Vocabulary dominates (8-22% η²), grammar is marginal (0-1.6% η²)
- Factorial decomposition shows spacer effects ≥ grammar effects
- BOM baseline matches grammar model
- Per-enhancer classification: >90% billboard
- **Narrative**: We tried harder than anyone to find grammar and it's not there. The field should stop looking.

### If Grammar Is Real But Poorly Measured (v2 Was Underpowered)
The paper becomes: "Regulatory grammar is pervasive but subtle: corrected measurements reveal widespread motif arrangement effects."
- 1,000 shuffles → significance rises from 8.3% to 30-50%
- Improved probes increase grammar η² from 0-1.6% to 5-10%
- Factorial decomposition shows position effects > spacer effects
- PARM shows stronger grammar than foundation models
- Per-enhancer classification: 60% billboard, 30% soft grammar, 10% strong grammar
- **Narrative**: Grammar exists everywhere but requires sensitive measurement. It's a fine-tuning mechanism, not the main signal.

### If Grammar Is Condition-Dependent (Most Interesting Outcome)
The paper becomes: "Regulatory grammar is context-activated: condition-specific motif arrangement effects explain enhancer specificity."
- Static GSI is weak, but condition-specific GSI varies dramatically across cell types
- "Grammar pairs" (from P3.1) correspond to known cooperative TFs
- Grammar-strong enhancers are developmental/cell-type-specific; grammar-weak are housekeeping
- Factorial decomposition reveals position×context interaction as the dominant term
- **Narrative**: Grammar is the mechanism of regulatory specificity. Vocabulary determines basal activity; grammar determines when and where.

---

## Part 5: Compute Budget

| Experiment | GPU Hours (est.) | Disk (est.) |
|-----------|-----------------|-------------|
| P0.3: 1000 shuffles | 4h | 500 MB |
| P1.1: Factorial decomposition | 8h | 2 GB |
| P1.3: Embedding extraction | 2h | 5 GB |
| P2.2: Full motif scan | 0h (CPU) | 200 MB |
| P2.3: Probe improvement | 4h | 1 GB |
| P2.4: PARM comparison | 4h | 2 GB |
| P3.1: TF-pair atlas | 8h | 1 GB |
| P3.2: Condition-dependent | 6h | 500 MB |
| P3.3: Spacer ablation | 4h | 500 MB |
| P3.4: Grammar-aware design | 2h | 200 MB |
| **Total** | **~42h** | **~13 GB** |

Well within the 4×A100 budget and 200 GB disk budget.

---

## Part 6: Key Literature to Position Against

| Paper | Claim | Our Response |
|-------|-------|-------------|
| **BOM (2025, Nat Commun)** | Simple motif counts beat DL | We formally test this with P1.2 and quantify the grammar increment |
| **PARM (2026, Nature)** | Gene regulation is far more predictable than believed | We test whether PARM's predictability comes from grammar or vocabulary |
| **Georgakopoulos-Soares (2023)** | Orientation/order are "major drivers" (+7.7%) | We decompose this with factorial shuffles (P1.1) to separate orientation from position |
| **Avsec (2021, BPNet)** | Soft motif syntax, helical periodicity | We test helical phasing universality and find it indistinguishable from random |
| **de Almeida (2022, DeepSTARR)** | Context-dependent motif function | Our per-enhancer grammar classification (P4.3) quantifies this |
| **HERMES (2026, bioRxiv)** | Interpretable EP grammar rulebook | We compare their rulebook to our cross-architecture consensus rules |
| **Malik (2024, Nat Genet)** | Most TF motifs are pairwise independent | Our TF-pair atlas (P3.1) tests this at scale |

---

## Summary: The Three Core Questions v3 Must Answer

1. **How much of "grammar sensitivity" is actually grammar?** (Factorial decomposition P1.1 + spacer ablation P3.3)
2. **Is grammar rare because it's genuinely weak, or because we can't measure it?** (Power analysis P0.3 + probe improvement P2.3 + PARM comparison P2.4)
3. **Is grammar a static property or a context-dependent regulatory mechanism?** (Condition-dependent grammar P3.2 + per-enhancer classification P4.3)

The answers to these three questions determine whether the paper's conclusion is "billboard confirmed," "grammar is subtle," or "grammar is conditional" — and each is a strong, publishable finding.
