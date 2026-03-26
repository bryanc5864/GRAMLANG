# GRAMLANG → NeurIPS 2026 Main Track Plan

## Current Status: Strong Finding, Weak Contribution

**What we have:**
- Spacer confound discovery (78-86% of GSI variance)
- Positive control proving grammar exists (p < 1e-117)
- Architecture-independent validation (transformers + CNN)
- Billboard model confirmation

**What NeurIPS needs:**
- Novel method/algorithm
- Positive results (not just "X doesn't work")
- ML contribution (not just biology insight)
- Practical utility

---

## Proposed Contribution: DISENTANGLED REGULATORY GRAMMAR

**Title:** "Disentangling Motif Grammar from Sequence Composition in Regulatory DNA"

**One-sentence pitch:** We discover that foundation models confound grammar with spacer composition, then introduce Spacer-Factored Grammar Networks (SFGNs) that disentangle these effects and reveal the true syntactic rules of gene regulation.

---

## Three-Part Contribution

### Part 1: The Problem (Existing Work, Reframed)
**Section 2: The Spacer Confound**

Current approach: vocabulary-preserving shuffles change BOTH motif arrangement AND spacer DNA → measured "grammar sensitivity" is 78-86% spacer effects.

Key experiments (already done):
- Factorial decomposition: spacer >> position >> orientation
- Positive control: grammar IS real when spacers controlled
- GC correlation reversal across species

**ML Framing:** Current foundation models learn a compositionally confounded representation — they cannot distinguish syntax from lexical statistics.

### Part 2: The Method (NEW WORK NEEDED)
**Section 3: Spacer-Factored Grammar Networks (SFGN)**

#### 2.1 Architecture Innovation

```
Input sequence
    ↓
┌─────────────────────────────────────┐
│  Motif Encoder (frozen DNABERT-2)   │
│  → Extract motif-only embeddings    │
└─────────────────────────────────────┘
    ↓                    ↓
┌─────────────┐    ┌─────────────────┐
│ Grammar     │    │ Composition     │
│ Module      │    │ Module          │
│ (learnable) │    │ (learnable)     │
│             │    │                 │
│ - Pairwise  │    │ - GC content    │
│   attention │    │ - k-mer freq    │
│ - Position  │    │ - DNA shape     │
│   encoding  │    │                 │
└─────────────┘    └─────────────────┘
    ↓                    ↓
┌─────────────────────────────────────┐
│  Disentangled Fusion Layer          │
│  y = α·g(grammar) + β·c(composition)│
│  with orthogonality constraint      │
└─────────────────────────────────────┘
    ↓
Expression prediction
```

**Key innovations:**
1. **Explicit disentanglement** — grammar and composition are separate pathways
2. **Orthogonality loss** — L_orth = |corr(grammar_emb, composition_emb)|
3. **Interpretable α/β weights** — quantify grammar vs composition contribution
4. **Motif-centric attention** — attend over motif positions, not raw sequence

#### 2.2 Spacer-Factored GSI (SF-GSI)

New metric that isolates true grammar sensitivity:

```
SF-GSI = GSI_full - GSI_spacer_only

where:
- GSI_full = σ(full_shuffles) / |μ(full_shuffles)|
- GSI_spacer_only = σ(spacer_shuffles) / |μ(spacer_shuffles)|
  (spacer_shuffles: reshuffle spacers, keep motifs fixed)
```

This removes the spacer confound by subtraction.

#### 2.3 Grammar Rule Extraction with Controls

For each motif pair (A, B), measure:
- Δ_spacing = effect of changing A-B distance (spacers controlled)
- Δ_order = effect of swapping A↔B order (spacers controlled)
- Δ_orientation = effect of flipping A or B strand (spacers controlled)

Output: **Controlled Grammar Rulebook** — rules that survive spacer factorization.

### Part 3: The Discovery (NEW RESULTS NEEDED)
**Section 4: What Grammar Actually Exists**

With SFGN and SF-GSI, re-analyze all datasets:

#### 3.1 Quantifying True Grammar Contribution

| Dataset | GSI (confounded) | SF-GSI (controlled) | True Grammar % |
|---------|------------------|---------------------|----------------|
| Agarwal | X% | Y% | Z% |
| ... | ... | ... | ... |

**Hypothesis:** True grammar is 5-15% of expression variance (not 0%, not 80%)

#### 3.2 Universal vs Species-Specific Rules

- Which grammar rules transfer across species?
- Which are species-specific?
- Does the GC reversal affect grammar rules too?

#### 3.3 Grammar-Guided Enhancer Design

**Application:** Use discovered rules to design synthetic enhancers

Experiment:
1. Generate enhancers with RANDOM motif arrangement
2. Generate enhancers with GRAMMAR-GUIDED arrangement (using SFGN rules)
3. Compare predicted expression (and ideally, experimental validation)

**Expected result:** Grammar-guided > random, but effect size is modest (5-15%)

---

## Experimental Plan

### Phase 1: Method Development (4 weeks)

**Week 1-2: SFGN Architecture**
- [ ] Implement motif encoder (freeze DNABERT-2, extract motif positions)
- [ ] Implement grammar module (pairwise attention over motifs)
- [ ] Implement composition module (k-mer, GC, shape features)
- [ ] Implement disentangled fusion with orthogonality loss

**Week 3-4: Training & Validation**
- [ ] Train on Agarwal (largest dataset)
- [ ] Validate disentanglement: α/β weights, orthogonality
- [ ] Compare to baselines: DNABERT-2 probe, BOM, full grammar features

### Phase 2: SF-GSI & Controlled Analysis (3 weeks)

**Week 5-6: SF-GSI Implementation**
- [ ] Implement spacer-only shuffle function
- [ ] Compute SF-GSI for all datasets
- [ ] Validate: SF-GSI should be lower than GSI, closer to positive control effect sizes

**Week 7: Controlled Rule Extraction**
- [ ] Re-run rule extraction with spacer controls
- [ ] Compare to confounded rules: which survive?
- [ ] Quantify: % of rules that are actually grammar vs spacer artifacts

### Phase 3: Discovery & Application (3 weeks)

**Week 8-9: Cross-Dataset Analysis**
- [ ] SF-GSI census across all 5 datasets
- [ ] Transfer analysis: which rules are universal?
- [ ] SFGN α/β decomposition: grammar contribution per dataset

**Week 10: Enhancer Design Application**
- [ ] Generate synthetic enhancers: random vs grammar-guided
- [ ] Predict with SFGN and baselines
- [ ] (Stretch) Collaborate for experimental validation

### Phase 4: Paper Writing (2 weeks)

**Week 11-12: Paper**
- [ ] Main paper (8 pages + references)
- [ ] Appendix with all experimental details
- [ ] Code release preparation

---

## Expected Results & Story

### Main Claims

1. **Foundation models confound grammar with composition** (existing finding, reframed)
   - 78-86% of GSI variance is spacer effects
   - Positive control proves grammar exists

2. **SFGN disentangles grammar from composition** (new method)
   - Orthogonal representations
   - Interpretable α/β weights
   - Better expression prediction than confounded models

3. **True regulatory grammar is modest but real** (new finding)
   - SF-GSI: 5-15% of expression variance (not 0%, not 80%)
   - ~100 rules survive spacer control (vs ~9000 confounded)
   - Some rules transfer across species, most don't

4. **Grammar-guided design improves enhancer engineering** (application)
   - 5-15% improvement over random arrangement
   - Practical utility for synthetic biology

### Figure Plan

| Figure | Content |
|--------|---------|
| Fig 1 | The spacer confound: factorial decomposition showing spacer dominance |
| Fig 2 | SFGN architecture and disentanglement validation |
| Fig 3 | SF-GSI census: true grammar contribution across datasets |
| Fig 4 | Controlled grammar rulebook: universal vs species-specific |
| Fig 5 | Application: grammar-guided enhancer design |

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| SFGN doesn't disentangle well | Medium | Try multiple architectures (attention, GNN, transformer) |
| True grammar is ~0% | Low | Positive control proves it exists; question is magnitude |
| True grammar is still >50% | Low | Would be surprising given factorial results; still interesting |
| No improvement in enhancer design | Medium | Focus on interpretability even if application is weak |
| Reviewers say "obvious" | Medium | Emphasize that nobody has done this; cite papers using confounded GSI |

---

## Comparison to Baselines

| Method | Grammar | Composition | Disentangled | Expression R² |
|--------|---------|-------------|--------------|---------------|
| DNABERT-2 + probe | Confounded | Confounded | No | 0.30 |
| Bag-of-Motifs | No | Implicit | N/A | 0.10 |
| Grammar features | Confounded | No | No | 0.08 |
| **SFGN (ours)** | **Controlled** | **Explicit** | **Yes** | **0.35?** |

---

## Timeline to NeurIPS 2026

- **Submission deadline:** ~May 2026
- **Current date:** March 2026
- **Time remaining:** ~8 weeks

| Week | Milestone |
|------|-----------|
| Mar 25-31 | SFGN architecture implementation |
| Apr 1-7 | SFGN training & validation |
| Apr 8-14 | SF-GSI implementation & census |
| Apr 15-21 | Controlled rule extraction |
| Apr 22-28 | Cross-dataset analysis & transfer |
| Apr 29-May 5 | Enhancer design application |
| May 6-12 | Paper writing |
| May 13-19 | Paper polishing & submission |

---

## Alternative Venues (Backup)

If NeurIPS doesn't work out:

| Venue | Deadline | Fit |
|-------|----------|-----|
| ICLR 2027 | Oct 2026 | Good (more open to negative results) |
| ICML 2027 | Jan 2027 | Good |
| NeurIPS ML4Bio Workshop | Sep 2026 | Excellent (lower bar) |
| Nature Methods | Rolling | Excellent (methodology focus) |
| Genome Research | Rolling | Good (biology audience) |

---

## Summary

**Current state:** Strong negative finding, weak ML contribution

**Proposed addition:**
1. SFGN architecture (disentangles grammar from composition)
2. SF-GSI metric (spacer-factored grammar sensitivity)
3. Controlled grammar rulebook (true rules, not artifacts)
4. Enhancer design application (practical utility)

**Expected story:** "Foundation models confound grammar with composition. We fix this with SFGN, revealing that true regulatory grammar contributes 5-15% of expression variance — modest but real, and useful for enhancer design."

**Feasibility:** 8 weeks is tight but doable if SFGN works on first try.
