#!/usr/bin/env python
"""
GRAMLANG v2 Pipeline Runner

Improvements over v1:
1. Dataset-specific expression probes (not vaishnav-only)
2. Enformer with native CAGE tracks for human datasets
3. Increased sample sizes
4. New experiments: synthetic probes, heterogeneity, counterfactual potential,
   info-theoretic decomposition, redesigned compositionality & transfer,
   attention analysis, ANOVA decomposition
5. Power analysis for null results

Usage:
    python scripts/run_v2_pipeline.py [--phase N] [--models model1,model2]
"""

import os
import sys
import gc
import json
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model
from src.models.expression_probes import load_probe
from src.perturbation.motif_scanner import MotifScanner
from src.grammar.sensitivity import run_gsi_census, compute_grammar_information
from src.grammar.rule_extraction import GrammarRuleExtractor
from src.grammar.consensus import compute_grammar_consensus, compute_global_consensus
from src.grammar.compositionality import run_compositionality_sweep
from src.grammar.complexity import classify_grammar_complexity
from src.transfer.cross_species import compute_transfer_matrix
from src.transfer.phylogenetics import build_grammar_phylogeny
from src.decomposition.biophysics import compute_biophysics_residual
from src.decomposition.tf_structure import build_structure_grammar_map, test_structure_predicts_grammar
from src.decomposition.strength_tradeoff import compute_grammar_strength_tradeoff
from src.decomposition.phase_diagram import compute_grammar_phase_diagram
from src.design.completeness import compute_grammar_completeness
from src.utils.io import load_processed, save_json, check_disk_budget
from src.utils import visualization as viz

# New v2 modules
from src.grammar.synthetic_probes import run_synthetic_grammar_probes, get_top_motif_pairs, summarize_synthetic_probes
from src.grammar.heterogeneity import analyze_grammar_heterogeneity
from src.grammar.counterfactual import compute_grammar_potential, summarize_grammar_potential
from src.grammar.information_theory import compute_information_decomposition
from src.grammar.compositionality_v2 import run_enhancer_specific_compositionality, summarize_compositionality_v2
from src.transfer.distributional_transfer import compute_distributional_transfer
from src.grammar.attention_grammar import extract_attention_patterns, identify_grammar_heads
from src.grammar.anova_decomposition import compute_anova_decomposition, compute_power_analysis

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'v2')
PROBES_DIR = os.path.join(PROJECT_DIR, 'data', 'probes')

# Foundation models (probe-based)
FOUNDATION_MODELS = ['dnabert2', 'nt', 'hyenadna']

ALL_DATASETS = {
    'agarwal': {'species': 'human', 'cell_type': 'K562'},
    'inoue': {'species': 'human', 'cell_type': None},
    'vaishnav': {'species': 'yeast', 'cell_type': None},
    'jores': {'species': 'plant', 'cell_type': None},
    'klein': {'species': 'human', 'cell_type': 'HepG2'},
}

SPECIES_MAP = {
    'human': ['agarwal', 'inoue', 'klein'],
    'yeast': ['vaishnav'],
    'plant': ['jores'],
}

# Enformer-compatible datasets (human only, with CAGE tracks)
ENFORMER_DATASETS = {
    'agarwal': 'K562',
    'klein': 'HepG2',
    'inoue': 'K562',  # Use K562 as best proxy
}


def log_msg(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def swap_probe(model, model_name, ds_name, device='cuda'):
    """Swap the expression probe on an already-loaded model (no model reload)."""
    if model_name == 'enformer':
        return  # Enformer uses built-in CAGE head

    if not hasattr(model, 'set_probe'):
        return

    # Try dataset-specific probe first, fall back to vaishnav
    for probe_name in [f'{model_name}_{ds_name}', f'{model_name}_vaishnav', f'{model_name}_vaishnav2022']:
        probe_path = os.path.join(PROBES_DIR, f'{probe_name}_probe.pt')
        if os.path.exists(probe_path):
            probe = load_probe(PROBES_DIR, probe_name, model.hidden_dim, device=device)
            model.set_probe(probe)
            log_msg(f"    Swapped probe: {probe_name}")
            return
    log_msg(f"    WARNING: No probe found for {model_name}/{ds_name}")


def load_model_with_probe(model_name, ds_name, device='cuda'):
    """Load model with dataset-specific probe."""
    if model_name == 'enformer':
        return load_model('enformer', device=device)

    model = load_model(model_name, device=device, dataset_name='__dummy__')
    swap_probe(model, model_name, ds_name, device=device)
    return model


def get_available_datasets():
    available = {}
    for name, info in ALL_DATASETS.items():
        path = os.path.join(DATA_DIR, 'processed', f'{name}_processed.parquet')
        if os.path.exists(path):
            available[name] = info
    return available


# ============================================================
# Phase 1: Core Modules (re-run with dataset-specific probes)
# ============================================================

def run_phase1_module1(models, datasets, args):
    """Module 1: GSI Census with dataset-specific probes + Enformer.

    Optimized: loads each model once, swaps probes per dataset.
    """
    log_msg("=" * 60)
    log_msg("PHASE 1 / MODULE 1: Grammar Sensitivity Census (v2)")
    log_msg("=" * 60)

    outdir = os.path.join(RESULTS_DIR, 'module1')
    os.makedirs(outdir, exist_ok=True)
    all_results = []

    # Pre-load all dataset data
    dataset_data = {}
    for ds_name, ds_info in datasets.items():
        mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
        if not os.path.exists(mh_path):
            continue
        dataset_data[ds_name] = {
            'df': load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet')),
            'motif_hits': pd.read_parquet(mh_path),
            'info': ds_info,
        }

    # Foundation models: load once, swap probes per dataset
    for model_name in models:
        log_msg(f"  Loading model: {model_name}")
        model = load_model(model_name, device='cuda', dataset_name='__dummy__')

        for ds_name, dd in dataset_data.items():
            # Skip if result already exists
            result_path = os.path.join(outdir, f'{ds_name}_{model_name}_gsi.parquet')
            if os.path.exists(result_path):
                log_msg(f"    {ds_name} already done, loading")
                all_results.append(pd.read_parquet(result_path))
                continue

            swap_probe(model, model_name, ds_name)
            log_msg(f"    Running GSI: {ds_name}")

            gsi = run_gsi_census(
                dataset=dd['df'], model=model, motif_hits=dd['motif_hits'],
                n_shuffles=args.n_shuffles, min_motifs=2,
                cell_type=dd['info']['cell_type'],
                max_enhancers=args.max_enhancers,
            )
            gsi['dataset'] = ds_name
            gsi['species'] = dd['info']['species']
            gsi.to_parquet(result_path)
            all_results.append(gsi)

        model.unload()
        gc.collect()
        torch.cuda.empty_cache()

    # Enformer for human datasets (separate because it's slow)
    enformer_datasets = {k: v for k, v in dataset_data.items() if k in ENFORMER_DATASETS}
    if enformer_datasets:
        log_msg(f"  Loading Enformer...")
        try:
            enf_model = load_model('enformer', device='cuda')
            for ds_name, dd in enformer_datasets.items():
                result_path = os.path.join(outdir, f'{ds_name}_enformer_gsi.parquet')
                if os.path.exists(result_path):
                    all_results.append(pd.read_parquet(result_path))
                    continue

                cell = ENFORMER_DATASETS[ds_name]
                log_msg(f"    Running GSI: {ds_name} (Enformer, cell={cell})")
                gsi = run_gsi_census(
                    dataset=dd['df'], model=enf_model, motif_hits=dd['motif_hits'],
                    n_shuffles=args.n_shuffles, min_motifs=2,
                    cell_type=cell,
                    max_enhancers=min(args.max_enhancers, 50),
                )
                gsi['dataset'] = ds_name
                gsi['species'] = dd['info']['species']
                gsi.to_parquet(result_path)
                all_results.append(gsi)
            enf_model.unload()
        except Exception as e:
            log_msg(f"    Enformer failed: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_parquet(os.path.join(outdir, 'all_gsi_results.parquet'))

        summary = {}
        for ds in combined['dataset'].unique():
            ds_data = combined[combined['dataset'] == ds]
            summary[ds] = {
                'n_enhancers': int(ds_data['seq_id'].nunique()),
                'mean_gsi': float(ds_data['gsi'].mean()),
                'median_gsi': float(ds_data['gsi'].median()),
                'frac_significant': float((ds_data['p_value'] < 0.05).mean()),
                'models': list(ds_data['model'].unique()),
            }
        save_json(summary, os.path.join(outdir, 'gsi_summary.json'))
        log_msg(f"  Module 1 complete: {len(combined)} GSI measurements")
        return combined

    return pd.DataFrame()


def run_phase1_module2(models, datasets, gsi_results, args):
    """Module 2: Rule Extraction with dataset-specific probes."""
    log_msg("=" * 60)
    log_msg("PHASE 1 / MODULE 2: Grammar Rule Extraction (v2)")
    log_msg("=" * 60)

    outdir = os.path.join(RESULTS_DIR, 'module2')
    os.makedirs(outdir, exist_ok=True)
    all_rules = []

    sensitive = gsi_results[gsi_results['gsi'] > 0.05]['seq_id'].unique() if len(gsi_results) > 0 else []

    # Pre-load dataset data
    dataset_data = {}
    for ds_name, ds_info in datasets.items():
        mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
        if not os.path.exists(mh_path):
            continue
        dataset_data[ds_name] = {
            'df': load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet')),
            'motif_hits': pd.read_parquet(mh_path),
            'info': ds_info,
        }

    # Load each model once, iterate datasets
    for model_name in models:
        log_msg(f"  Loading model: {model_name}")
        model = load_model(model_name, device='cuda', dataset_name='__dummy__')

        for ds_name, dd in dataset_data.items():
            swap_probe(model, model_name, ds_name)
            log_msg(f"    Extracting rules: {ds_name}")

            df = dd['df']
            motif_hits = dd['motif_hits']
            ds_info = dd['info']

            df_filtered = df[df['seq_id'].isin(sensitive)] if len(sensitive) > 0 else df[df['n_motifs'] >= 2]
            if len(df_filtered) == 0:
                continue

            sample = df_filtered.sample(
                n=min(args.max_enhancers_rules, len(df_filtered)), random_state=42
            )

            extractor = GrammarRuleExtractor(model, cell_type=ds_info['cell_type'])
            for _, row in sample.iterrows():
                seq_id = str(row['seq_id'])
                seq = row['sequence']
                seq_motifs = motif_hits[motif_hits['seq_id'] == seq_id]
                annotation = {'sequence': seq, 'motifs': seq_motifs.to_dict('records')}
                try:
                    rules = extractor.extract_pairwise_rules(seq, annotation)
                    for pair_key, rule in rules.items():
                        all_rules.append({
                            'seq_id': seq_id, 'dataset': ds_name, 'model': model_name,
                            'pair': pair_key,
                            'motif_a': rule['motif_a_name'], 'motif_b': rule['motif_b_name'],
                            'optimal_spacing': rule['optimal_spacing'],
                            'spacing_sensitivity': rule['spacing_sensitivity'],
                            'optimal_orientation': rule['optimal_orientation'],
                            'orientation_sensitivity': rule['orientation_sensitivity'],
                            'helical_phase_score': rule['helical_phase_score'],
                            'fold_change': rule['fold_change'],
                            'spacing_profile': rule['spacing_profile'],
                            'spacings': rule['spacings'],
                        })
                except Exception:
                    continue

        model.unload()
        gc.collect()
        torch.cuda.empty_cache()

    if all_rules:
        rules_df = pd.DataFrame(all_rules)
        rules_df.to_parquet(os.path.join(outdir, 'grammar_rules_database.parquet'))

        consensus = compute_grammar_consensus(rules_df)
        consensus.to_parquet(os.path.join(outdir, 'consensus_scores.parquet'))
        global_cons = compute_global_consensus(consensus)
        save_json(global_cons, os.path.join(outdir, 'global_consensus.json'))

        log_msg(f"  Module 2 complete: {len(rules_df)} rules")
        return rules_df
    return pd.DataFrame()


def run_phase1_modules3456(models, datasets, gsi_results, rules_df, args):
    """Modules 3-6 with dataset-specific probes."""
    # Module 3: Compositionality (original)
    log_msg("=" * 60)
    log_msg("PHASE 1 / MODULE 3: Compositionality (v2)")
    log_msg("=" * 60)
    outdir3 = os.path.join(RESULTS_DIR, 'module3')
    os.makedirs(outdir3, exist_ok=True)

    all_comp = []
    for ds_name, ds_info in datasets.items():
        df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
        mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
        if not os.path.exists(mh_path):
            continue
        motif_hits = pd.read_parquet(mh_path)

        for model_name in models[:3]:
            model = load_model_with_probe(model_name, ds_name)
            comp = run_compositionality_sweep(
                dataset=df, motif_hits=motif_hits, model=model,
                cell_type=ds_info['cell_type'],
                target_ks=[3, 4, 5, 6],
                max_per_k=args.max_per_k,
                n_arrangements=args.n_arrangements,
            )
            comp['dataset'] = ds_name
            all_comp.append(comp)
            model.unload()

    if all_comp:
        comp_df = pd.concat(all_comp, ignore_index=True)
        comp_df.to_parquet(os.path.join(outdir3, 'compositionality_results.parquet'))
        gap_by_k = comp_df.groupby('n_motifs')['compositionality_gap'].agg(['mean', 'std']).reset_index()
        classification = classify_grammar_complexity(gap_by_k['n_motifs'].values, gap_by_k['mean'].values)
        save_json(classification, os.path.join(outdir3, 'complexity_classification.json'))
        log_msg(f"  Module 3 complete: {classification['classification']}")

    # Module 4: Transfer
    log_msg("=" * 60)
    log_msg("PHASE 1 / MODULE 4: Cross-Species Transfer (v2)")
    log_msg("=" * 60)
    outdir4 = os.path.join(RESULTS_DIR, 'module4')
    os.makedirs(outdir4, exist_ok=True)

    species_rules = {}
    species_datasets = {}
    species_motif_hits = {}
    for species, ds_names in SPECIES_MAP.items():
        for ds_name in ds_names:
            path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet')
            mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
            if os.path.exists(path) and os.path.exists(mh_path):
                ds_rules = rules_df[rules_df['dataset'] == ds_name] if len(rules_df) > 0 else pd.DataFrame()
                if len(ds_rules) > 0 or species not in species_datasets:
                    species_datasets[species] = load_processed(path)
                    species_motif_hits[species] = pd.read_parquet(mh_path)
                    if len(ds_rules) > 0:
                        species_rules[species] = ds_rules
                        break

    available_species = [s for s in species_datasets if s in species_rules]
    if len(available_species) >= 2:
        model = load_model_with_probe(models[0], list(datasets.keys())[0])
        transfer_df = compute_transfer_matrix(
            species_rules, species_datasets, species_motif_hits,
            model, species_list=available_species,
        )
        transfer_df.to_parquet(os.path.join(outdir4, 'transfer_matrix.parquet'))
        phylo = build_grammar_phylogeny(transfer_df, available_species)
        save_json(phylo, os.path.join(outdir4, 'grammar_phylogeny.json'))
        model.unload()
        log_msg(f"  Module 4 complete: {len(available_species)} species")

    # Module 5: Causal determinants
    log_msg("=" * 60)
    log_msg("PHASE 1 / MODULE 5: Causal Determinants (v2)")
    log_msg("=" * 60)
    outdir5 = os.path.join(RESULTS_DIR, 'module5')
    os.makedirs(outdir5, exist_ok=True)

    for ds_name in datasets:
        df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
        ds_gsi = gsi_results[gsi_results['dataset'] == ds_name] if len(gsi_results) > 0 else pd.DataFrame()
        if len(ds_gsi) > 20:
            gsi_per_seq = ds_gsi.groupby('seq_id')['gsi'].mean()
            merged = df[df['seq_id'].isin(gsi_per_seq.index)]
            if len(merged) > 20:
                grammar_effects = gsi_per_seq.loc[merged['seq_id'].values].values
                bio = compute_biophysics_residual(merged, grammar_effects)
                save_json(bio, os.path.join(outdir5, f'{ds_name}_biophysics.json'))
                log_msg(f"  {ds_name} biophysics R²: {bio.get('biophysics_r2', 0):.3f}")

    if len(rules_df) > 0:
        structure_map = build_structure_grammar_map(rules_df)
        structure_map.to_parquet(os.path.join(outdir5, 'structure_grammar_map.parquet'))
        save_json(test_structure_predicts_grammar(rules_df),
                  os.path.join(outdir5, 'structure_predicts_grammar.json'))

        for ds_name in datasets:
            mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
            if os.path.exists(mh_path):
                motif_hits = pd.read_parquet(mh_path)
                ds_rules = rules_df[rules_df['dataset'] == ds_name]
                if len(ds_rules) > 20:
                    tradeoff = compute_grammar_strength_tradeoff(ds_rules, motif_hits)
                    save_json(tradeoff, os.path.join(outdir5, f'{ds_name}_strength_tradeoff.json'))

    if len(gsi_results) > 0:
        for ds_name in datasets:
            df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
            ds_gsi = gsi_results[gsi_results['dataset'] == ds_name]
            if len(ds_gsi) > 20:
                phase = compute_grammar_phase_diagram(ds_gsi, df)
                save_json(phase, os.path.join(outdir5, f'{ds_name}_phase_diagram.json'))

    log_msg("  Module 5 complete")

    # Module 6: Completeness
    log_msg("=" * 60)
    log_msg("PHASE 1 / MODULE 6: Grammar Completeness (v2)")
    log_msg("=" * 60)
    outdir6 = os.path.join(RESULTS_DIR, 'module6')
    os.makedirs(outdir6, exist_ok=True)

    for ds_name, ds_info in datasets.items():
        df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
        mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
        if not os.path.exists(mh_path):
            continue
        motif_hits = pd.read_parquet(mh_path)

        model = load_model_with_probe(models[0], ds_name)
        grammar_features = rules_df[rules_df['dataset'] == ds_name] if len(rules_df) > 0 else pd.DataFrame()

        completeness = compute_grammar_completeness(
            df, motif_hits, grammar_features, model,
            cell_type=ds_info['cell_type']
        )
        save_json(completeness, os.path.join(outdir6, f'{ds_name}_completeness.json'))
        if 'error' not in completeness:
            log_msg(f"  {ds_name}: vocab={completeness['vocabulary_r2']:.3f}, "
                    f"grammar={completeness['vocab_plus_full_grammar_r2']:.3f}")
        model.unload()

    log_msg("  Module 6 complete")


# ============================================================
# Phase 2: New Experiments
# ============================================================

def run_phase2_new_experiments(models, datasets, gsi_results, rules_df, args):
    """Phase 2: New experiments (B, E, F, G)."""

    # Experiment B: Synthetic Grammar Probes
    log_msg("=" * 60)
    log_msg("PHASE 2 / EXPERIMENT B: Synthetic Grammar Probes")
    log_msg("=" * 60)
    outdir_b = os.path.join(RESULTS_DIR, 'experiment_b')
    os.makedirs(outdir_b, exist_ok=True)

    for ds_name, ds_info in datasets.items():
        if len(rules_df) == 0:
            break
        ds_rules = rules_df[rules_df['dataset'] == ds_name]
        if len(ds_rules) < 10:
            continue

        df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
        mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
        if not os.path.exists(mh_path):
            continue
        motif_hits = pd.read_parquet(mh_path)

        pairs = get_top_motif_pairs(ds_rules, motif_hits, df, n_pairs=20)
        if not pairs:
            continue

        for model_name in models[:2]:  # Use 2 models for speed
            model = load_model_with_probe(model_name, ds_name)
            target_len = int(df['sequence'].str.len().median())

            probe_results = run_synthetic_grammar_probes(
                model, pairs, target_length=target_len,
                cell_type=ds_info['cell_type'],
            )
            probe_results['dataset'] = ds_name
            probe_results.to_parquet(os.path.join(outdir_b, f'{ds_name}_{model_name}_synthetic.parquet'))

            probe_summary = summarize_synthetic_probes(probe_results)
            save_json(probe_summary, os.path.join(outdir_b, f'{ds_name}_{model_name}_summary.json'))
            log_msg(f"  {ds_name}/{model_name}: {len(probe_results)} measurements, "
                    f"{len(pairs)} pairs")
            model.unload()

    # Experiment F: Grammar Heterogeneity
    log_msg("=" * 60)
    log_msg("PHASE 2 / EXPERIMENT F: Grammar Heterogeneity")
    log_msg("=" * 60)
    outdir_f = os.path.join(RESULTS_DIR, 'experiment_f')
    os.makedirs(outdir_f, exist_ok=True)

    for ds_name in datasets:
        if len(gsi_results) == 0:
            break
        df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
        mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
        if not os.path.exists(mh_path):
            continue
        motif_hits = pd.read_parquet(mh_path)

        ds_gsi = gsi_results[gsi_results['dataset'] == ds_name]
        if len(ds_gsi) < 20:
            continue

        het = analyze_grammar_heterogeneity(ds_gsi, df, motif_hits)
        save_json(het, os.path.join(outdir_f, f'{ds_name}_heterogeneity.json'))
        if 'error' not in het:
            log_msg(f"  {ds_name}: {het['n_grammar_rich']} grammar-rich, "
                    f"predictor R²={het['predictor_r2_cv']:.3f}")

    # Experiment G: Counterfactual Grammar Potential
    log_msg("=" * 60)
    log_msg("PHASE 2 / EXPERIMENT G: Grammar Potential")
    log_msg("=" * 60)
    outdir_g = os.path.join(RESULTS_DIR, 'experiment_g')
    os.makedirs(outdir_g, exist_ok=True)

    for ds_name, ds_info in datasets.items():
        df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
        mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
        if not os.path.exists(mh_path):
            continue
        motif_hits = pd.read_parquet(mh_path)

        model = load_model_with_probe(models[0], ds_name)
        potential = compute_grammar_potential(
            df, model, motif_hits,
            n_shuffles=200, max_enhancers=200,
            cell_type=ds_info['cell_type'],
        )
        potential.to_parquet(os.path.join(outdir_g, f'{ds_name}_potential.parquet'))
        summary = summarize_grammar_potential(potential)
        save_json(summary, os.path.join(outdir_g, f'{ds_name}_potential_summary.json'))
        if 'error' not in summary:
            log_msg(f"  {ds_name}: mean potential={summary['mean_potential']:.4f}, "
                    f"utilization={summary['mean_utilization']:.2f}")
        model.unload()

    # Experiment E: Information-Theoretic Decomposition
    log_msg("=" * 60)
    log_msg("PHASE 2 / EXPERIMENT E: Information Decomposition")
    log_msg("=" * 60)
    outdir_e = os.path.join(RESULTS_DIR, 'experiment_e')
    os.makedirs(outdir_e, exist_ok=True)

    for ds_name, ds_info in datasets.items():
        df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
        mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
        if not os.path.exists(mh_path):
            continue
        motif_hits = pd.read_parquet(mh_path)

        model = load_model_with_probe(models[0], ds_name)
        info = compute_information_decomposition(
            df, motif_hits, model, cell_type=ds_info['cell_type'],
        )
        save_json(info, os.path.join(outdir_e, f'{ds_name}_information.json'))
        if 'error' not in info:
            log_msg(f"  {ds_name}: vocab R²={info['r2_vocabulary_gb']:.3f}, "
                    f"grammar info={info['grammar_information']:.3f}")
        model.unload()

    log_msg("  Phase 2 experiments complete")


# ============================================================
# Phase 3: Redesigned Tests & Attention Analysis
# ============================================================

def run_phase3(models, datasets, gsi_results, rules_df, args):
    """Phase 3: Redesigned compositionality, transfer, attention, ANOVA."""

    # Redesigned Compositionality (enhancer-specific)
    log_msg("=" * 60)
    log_msg("PHASE 3 / COMPOSITIONALITY V2 (enhancer-specific)")
    log_msg("=" * 60)
    outdir_cv2 = os.path.join(RESULTS_DIR, 'compositionality_v2')
    os.makedirs(outdir_cv2, exist_ok=True)

    all_comp_v2 = []
    for ds_name, ds_info in datasets.items():
        df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
        mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
        if not os.path.exists(mh_path):
            continue
        motif_hits = pd.read_parquet(mh_path)

        for model_name in models[:2]:
            model = load_model_with_probe(model_name, ds_name)
            comp = run_enhancer_specific_compositionality(
                df, motif_hits, model, cell_type=ds_info['cell_type'],
                max_enhancers=50, n_perturbations=20,
            )
            if len(comp) > 0:
                comp['dataset'] = ds_name
                all_comp_v2.append(comp)
            model.unload()

    if all_comp_v2:
        comp_v2_df = pd.concat(all_comp_v2, ignore_index=True)
        comp_v2_df.to_parquet(os.path.join(outdir_cv2, 'compositionality_v2_results.parquet'))
        summary = summarize_compositionality_v2(comp_v2_df)
        save_json(summary, os.path.join(outdir_cv2, 'compositionality_v2_summary.json'))
        log_msg(f"  Compositionality v2: {summary.get('mean_compositionality', 0):.3f} "
                f"({summary.get('n_tests', 0)} tests)")

    # Distributional Transfer
    log_msg("=" * 60)
    log_msg("PHASE 3 / DISTRIBUTIONAL TRANSFER")
    log_msg("=" * 60)
    outdir_dt = os.path.join(RESULTS_DIR, 'distributional_transfer')
    os.makedirs(outdir_dt, exist_ok=True)

    if len(rules_df) > 0 and len(gsi_results) > 0:
        dist_transfer = compute_distributional_transfer(rules_df, gsi_results, SPECIES_MAP)
        save_json(dist_transfer, os.path.join(outdir_dt, 'distributional_transfer.json'))
        log_msg(f"  Distributional transfer: {dist_transfer.get('grammar_properties_conserved', 0):.1%} "
                f"properties conserved")

    # Attention Analysis (transformers only)
    log_msg("=" * 60)
    log_msg("PHASE 3 / ATTENTION ANALYSIS")
    log_msg("=" * 60)
    outdir_attn = os.path.join(RESULTS_DIR, 'attention')
    os.makedirs(outdir_attn, exist_ok=True)

    for ds_name, ds_info in list(datasets.items())[:2]:  # First 2 datasets
        df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
        mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
        if not os.path.exists(mh_path):
            continue
        motif_hits = pd.read_parquet(mh_path)

        # Select grammar-sensitive sequences
        ds_gsi = gsi_results[gsi_results['dataset'] == ds_name] if len(gsi_results) > 0 else pd.DataFrame()
        if len(ds_gsi) > 0:
            top_seq_ids = ds_gsi.nlargest(20, 'gsi')['seq_id'].unique()
        else:
            top_seq_ids = df['seq_id'].head(20).values

        sample = df[df['seq_id'].isin(top_seq_ids)]
        sequences = sample['sequence'].tolist()[:20]
        annotations = []
        for _, row in sample.head(20).iterrows():
            hits = motif_hits[motif_hits['seq_id'] == row['seq_id']]
            annotations.append({'motifs': hits.to_dict('records')})

        for model_name in ['dnabert2', 'nt']:
            if model_name not in models:
                continue
            model = load_model_with_probe(model_name, ds_name)
            attn_df = extract_attention_patterns(model, sequences, annotations,
                                                  cell_type=ds_info['cell_type'])
            if isinstance(attn_df, pd.DataFrame) and len(attn_df) > 0:
                attn_df.to_parquet(os.path.join(outdir_attn, f'{ds_name}_{model_name}_attention.parquet'))
                grammar_heads = identify_grammar_heads(attn_df, ds_gsi)
                save_json(grammar_heads, os.path.join(outdir_attn, f'{ds_name}_{model_name}_grammar_heads.json'))
                log_msg(f"  {ds_name}/{model_name}: {grammar_heads.get('n_grammar_heads', 0)} grammar heads")
            model.unload()

    # ANOVA Decomposition
    log_msg("=" * 60)
    log_msg("PHASE 3 / ANOVA DECOMPOSITION")
    log_msg("=" * 60)
    outdir_anova = os.path.join(RESULTS_DIR, 'anova')
    os.makedirs(outdir_anova, exist_ok=True)

    for ds_name, ds_info in datasets.items():
        df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
        mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
        if not os.path.exists(mh_path):
            continue
        motif_hits = pd.read_parquet(mh_path)

        model = load_model_with_probe(models[0], ds_name)
        anova = compute_anova_decomposition(
            df, motif_hits, model, cell_type=ds_info['cell_type'],
        )
        save_json(anova, os.path.join(outdir_anova, f'{ds_name}_anova.json'))
        if 'error' not in anova:
            log_msg(f"  {ds_name}: vocab η²={anova['eta2_vocabulary']:.3f}, "
                    f"grammar η²={anova['grammar_total_eta2']:.3f}")

            # Power analysis for null results
            for test_name, effect in [
                ('grammar_eta2', anova['grammar_total_eta2']),
            ]:
                power = compute_power_analysis(max(effect, 0.001), anova['n_sequences'])
                save_json(power, os.path.join(outdir_anova, f'{ds_name}_{test_name}_power.json'))
        model.unload()

    log_msg("  Phase 3 complete")


def main():
    parser = argparse.ArgumentParser(description='GRAMLANG v2 Pipeline')
    parser.add_argument('--phase', type=int, default=0, help='Run specific phase (1-3), 0=all')
    parser.add_argument('--models', type=str, default=None, help='Comma-separated model names')
    parser.add_argument('--n-shuffles', type=int, default=100, help='Shuffles for GSI (v2 default: 100)')
    parser.add_argument('--max-enhancers', type=int, default=500, help='Max enhancers for GSI (v2 default: 500)')
    parser.add_argument('--max-enhancers-rules', type=int, default=100, help='Max for rule extraction')
    parser.add_argument('--max-per-k', type=int, default=30, help='Max per motif count')
    parser.add_argument('--n-arrangements', type=int, default=100, help='Arrangements per enhancer')
    args = parser.parse_args()

    models = args.models.split(',') if args.models else FOUNDATION_MODELS
    datasets = get_available_datasets()

    if not datasets:
        log_msg("ERROR: No preprocessed datasets found")
        return

    log_msg(f"Models: {models}")
    log_msg(f"Datasets: {list(datasets.keys())}")
    log_msg(f"v2 improvements: dataset-specific probes, Enformer, new experiments")

    gsi_results = pd.DataFrame()
    rules_df = pd.DataFrame()

    if args.phase in [0, 1]:
        gsi_results = run_phase1_module1(models, datasets, args)
        rules_df = run_phase1_module2(models, datasets, gsi_results, args)
        run_phase1_modules3456(models, datasets, gsi_results, rules_df, args)
    else:
        # Load existing results
        gsi_path = os.path.join(RESULTS_DIR, 'module1', 'all_gsi_results.parquet')
        rules_path = os.path.join(RESULTS_DIR, 'module2', 'grammar_rules_database.parquet')
        if os.path.exists(gsi_path):
            gsi_results = pd.read_parquet(gsi_path)
        if os.path.exists(rules_path):
            rules_df = pd.read_parquet(rules_path)

    if args.phase in [0, 2]:
        run_phase2_new_experiments(models, datasets, gsi_results, rules_df, args)

    if args.phase in [0, 3]:
        run_phase3(models, datasets, gsi_results, rules_df, args)

    log_msg("=" * 60)
    log_msg("v2 Pipeline complete!")
    log_msg("=" * 60)


if __name__ == '__main__':
    main()
