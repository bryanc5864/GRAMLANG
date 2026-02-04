#!/usr/bin/env python
"""
GRAMLANG Full Pipeline Runner

Orchestrates all 6 modules sequentially, managing GPU memory by
loading one model at a time. Documents all results.

Usage:
    python scripts/run_full_pipeline.py [--module N] [--models model1,model2]
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
from src.models.expression_probes import (
    ExpressionProbe, train_expression_probe,
    extract_and_cache_embeddings, save_probe, load_probe
)
from src.perturbation.motif_scanner import MotifScanner
from src.grammar.sensitivity import run_gsi_census, compute_grammar_information
from src.grammar.rule_extraction import GrammarRuleExtractor
from src.grammar.consensus import compute_grammar_consensus, compute_global_consensus
from src.grammar.compositionality import run_compositionality_sweep
from src.grammar.complexity import classify_grammar_complexity
from src.grammar.geometry import (
    compute_grammar_direction, layer_sweep_grammar, compute_cross_model_alignment
)
from src.transfer.cross_species import compute_transfer_matrix
from src.transfer.phylogenetics import build_grammar_phylogeny
from src.decomposition.biophysics import compute_biophysics_residual
from src.decomposition.tf_structure import (
    build_structure_grammar_map, test_structure_predicts_grammar
)
from src.decomposition.strength_tradeoff import compute_grammar_strength_tradeoff
from src.decomposition.phase_diagram import compute_grammar_phase_diagram
from src.design.grammar_optimizer import design_grammar_optimized_enhancer
from src.design.completeness import compute_grammar_completeness
from src.utils.io import load_processed, save_json, check_disk_budget
from src.utils.statistics import correct_pvalues
from src.utils import visualization as viz

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

# Models to run (ordered by memory requirement, smallest first)
ALL_MODELS = ['dnabert2', 'caduceus', 'nt', 'hyenadna', 'enformer']
# 'gpn', 'borzoi', 'sei', 'evo' added when verified working

# Datasets
ALL_DATASETS = {
    'agarwal': {'species': 'human', 'cell_type': 'K562'},
    'kircher': {'species': 'human', 'cell_type': 'K562'},
    'de_almeida': {'species': 'human', 'cell_type': None},
    'vaishnav': {'species': 'yeast', 'cell_type': None},
    'jores': {'species': 'plant', 'cell_type': None},
    'klein': {'species': 'human', 'cell_type': 'HepG2'},  # Klein 2020 is human HepG2
}

SPECIES_MAP = {
    'human': ['agarwal', 'kircher', 'de_almeida', 'klein'],
    'yeast': ['vaishnav'],
    'plant': ['jores'],
}


def log_msg(msg):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def check_disk():
    """Check disk budget."""
    usage = check_disk_budget(200.0)
    log_msg(f"Disk: {usage['used_gb']:.1f} GB / {usage['budget_gb']} GB "
            f"({usage['utilization_pct']:.0f}%)")
    if usage['over_budget']:
        log_msg("WARNING: Over disk budget!")
    return usage


def get_available_datasets():
    """Check which datasets are available (preprocessed)."""
    available = {}
    for name, info in ALL_DATASETS.items():
        path = os.path.join(DATA_DIR, 'processed', f'{name}_processed.parquet')
        if os.path.exists(path):
            available[name] = info
    return available


def run_module1(models, datasets, args):
    """Module 1: Grammar Existence & Complexity Classification."""
    log_msg("=" * 60)
    log_msg("MODULE 1: Grammar Existence & Complexity Classification")
    log_msg("=" * 60)

    os.makedirs(os.path.join(RESULTS_DIR, 'module1'), exist_ok=True)
    all_gsi_results = []

    for model_name in models:
        log_msg(f"\n  Loading model: {model_name}")
        model = load_model(model_name, device='cuda')

        for ds_name, ds_info in datasets.items():
            log_msg(f"  Dataset: {ds_name}")
            df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
            mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
            if os.path.exists(mh_path):
                motif_hits = pd.read_parquet(mh_path)
            else:
                log_msg(f"    No motif hits for {ds_name}, skipping")
                continue

            gsi = run_gsi_census(
                dataset=df,
                model=model,
                motif_hits=motif_hits,
                n_shuffles=args.n_shuffles,
                min_motifs=2,
                cell_type=ds_info['cell_type'],
                max_enhancers=args.max_enhancers,
            )
            gsi['dataset'] = ds_name
            gsi['species'] = ds_info['species']

            gsi.to_parquet(os.path.join(
                RESULTS_DIR, 'module1', f'{ds_name}_{model_name}_gsi.parquet'
            ))
            all_gsi_results.append(gsi)

        model.unload()

    if all_gsi_results:
        combined = pd.concat(all_gsi_results, ignore_index=True)
        combined.to_parquet(os.path.join(RESULTS_DIR, 'module1', 'all_gsi_results.parquet'))

        # Summary
        summary = {}
        for ds in combined['dataset'].unique():
            ds_data = combined[combined['dataset'] == ds]
            summary[ds] = {
                'n_enhancers': int(ds_data['seq_id'].nunique()),
                'mean_gsi': float(ds_data['gsi'].mean()),
                'median_gsi': float(ds_data['gsi'].median()),
                'frac_significant': float((ds_data['p_value'] < 0.05).mean()),
            }
        save_json(summary, os.path.join(RESULTS_DIR, 'module1', 'gsi_summary.json'))

        # Grammar information
        info_results = []
        for _, row in combined.iterrows():
            if 'shuffle_mean' in row and pd.notna(row.get('shuffle_std', None)):
                info = compute_grammar_information(
                    row['original_expression'],
                    np.random.normal(row['shuffle_mean'], max(row['shuffle_std'], 1e-6), 100)
                )
                info_results.append({
                    'seq_id': row['seq_id'],
                    'model': row['model'],
                    'dataset': row['dataset'],
                    **info
                })
        if info_results:
            pd.DataFrame(info_results).to_parquet(
                os.path.join(RESULTS_DIR, 'module1', 'grammar_information.parquet')
            )

        # Plot
        viz.plot_gsi_distribution(
            combined,
            os.path.join(RESULTS_DIR, 'module1', 'fig1_gsi_distribution.pdf')
        )

        log_msg(f"  Module 1 complete: {len(combined)} GSI measurements")
        return combined
    return pd.DataFrame()


def run_module2(models, datasets, gsi_results, args):
    """Module 2: Cross-Architecture Grammar Rule Extraction."""
    log_msg("=" * 60)
    log_msg("MODULE 2: Cross-Architecture Grammar Rule Extraction")
    log_msg("=" * 60)

    os.makedirs(os.path.join(RESULTS_DIR, 'module2'), exist_ok=True)
    all_rules = []

    # Select grammar-sensitive enhancers
    if len(gsi_results) > 0:
        sensitive = gsi_results[gsi_results['gsi'] > 0.05]['seq_id'].unique()
        log_msg(f"  {len(sensitive)} grammar-sensitive enhancers (GSI > 0.05)")
    else:
        sensitive = []

    for model_name in models:
        log_msg(f"\n  Loading model: {model_name}")
        model = load_model(model_name, device='cuda')

        for ds_name, ds_info in datasets.items():
            log_msg(f"  Dataset: {ds_name}")
            df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
            mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
            if not os.path.exists(mh_path):
                continue
            motif_hits = pd.read_parquet(mh_path)

            # Filter to sensitive enhancers
            if len(sensitive) > 0:
                df_filtered = df[df['seq_id'].isin(sensitive)]
            else:
                df_filtered = df[df['n_motifs'] >= 2]

            if len(df_filtered) == 0:
                continue

            # Sample
            sample = df_filtered.sample(
                n=min(args.max_enhancers_rules, len(df_filtered)),
                random_state=42
            )

            extractor = GrammarRuleExtractor(model, cell_type=ds_info['cell_type'])
            for _, row in sample.iterrows():
                seq_id = str(row['seq_id'])
                seq = row['sequence']
                seq_motifs = motif_hits[motif_hits['seq_id'] == seq_id]
                annotation = {
                    'sequence': seq,
                    'motifs': seq_motifs.to_dict('records'),
                }
                try:
                    rules = extractor.extract_pairwise_rules(seq, annotation)
                    for pair_key, rule in rules.items():
                        all_rules.append({
                            'seq_id': seq_id,
                            'dataset': ds_name,
                            'model': model_name,
                            'pair': pair_key,
                            'motif_a': rule['motif_a_name'],
                            'motif_b': rule['motif_b_name'],
                            'optimal_spacing': rule['optimal_spacing'],
                            'spacing_sensitivity': rule['spacing_sensitivity'],
                            'optimal_orientation': rule['optimal_orientation'],
                            'orientation_sensitivity': rule['orientation_sensitivity'],
                            'helical_phase_score': rule['helical_phase_score'],
                            'fold_change': rule['fold_change'],
                            'spacing_profile': rule['spacing_profile'],
                            'spacings': rule['spacings'],
                        })
                except Exception as e:
                    continue

        model.unload()

    if all_rules:
        rules_df = pd.DataFrame(all_rules)
        rules_df.to_parquet(os.path.join(RESULTS_DIR, 'module2', 'grammar_rules_database.parquet'))

        # Consensus
        consensus = compute_grammar_consensus(rules_df)
        consensus.to_parquet(os.path.join(RESULTS_DIR, 'module2', 'consensus_scores.parquet'))

        global_consensus = compute_global_consensus(consensus)
        save_json(global_consensus, os.path.join(RESULTS_DIR, 'module2', 'global_consensus.json'))

        log_msg(f"  Module 2 complete: {len(rules_df)} rules, "
                f"consensus={global_consensus.get('mean_consensus', 0):.3f}")
        return rules_df
    return pd.DataFrame()


def run_module3(models, datasets, rules_df, args):
    """Module 3: Compositionality Testing."""
    log_msg("=" * 60)
    log_msg("MODULE 3: Compositionality Testing")
    log_msg("=" * 60)

    os.makedirs(os.path.join(RESULTS_DIR, 'module3'), exist_ok=True)
    all_comp = []

    for model_name in models[:3]:  # Use top 3 models for speed
        model = load_model(model_name, device='cuda')

        for ds_name, ds_info in datasets.items():
            df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
            mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
            if not os.path.exists(mh_path):
                continue
            motif_hits = pd.read_parquet(mh_path)

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
        comp_df.to_parquet(os.path.join(RESULTS_DIR, 'module3', 'compositionality_results.parquet'))

        # Complexity classification
        gap_by_k = comp_df.groupby('n_motifs')['compositionality_gap'].agg(['mean', 'std']).reset_index()
        classification = classify_grammar_complexity(
            gap_by_k['n_motifs'].values,
            gap_by_k['mean'].values
        )
        save_json(classification, os.path.join(RESULTS_DIR, 'module3', 'complexity_classification.json'))

        viz.plot_compositionality_curve(
            comp_df,
            os.path.join(RESULTS_DIR, 'module3', 'fig2_compositionality.pdf')
        )

        log_msg(f"  Module 3 complete: classification={classification['classification']}")
        return comp_df
    return pd.DataFrame()


def run_module4(models, datasets, rules_df, args):
    """Module 4: Cross-Species Grammar Transfer."""
    log_msg("=" * 60)
    log_msg("MODULE 4: Cross-Species Grammar Transfer")
    log_msg("=" * 60)

    os.makedirs(os.path.join(RESULTS_DIR, 'module4'), exist_ok=True)

    # Organize rules by species
    species_rules = {}
    species_datasets = {}
    species_motif_hits = {}

    for species, ds_names in SPECIES_MAP.items():
        for ds_name in ds_names:
            path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet')
            mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
            if os.path.exists(path) and os.path.exists(mh_path):
                # Only use this dataset if it has rules OR we haven't found one yet
                ds_rules = rules_df[rules_df['dataset'] == ds_name] if len(rules_df) > 0 else pd.DataFrame()
                if len(ds_rules) > 0 or species not in species_datasets:
                    species_datasets[species] = load_processed(path)
                    species_motif_hits[species] = pd.read_parquet(mh_path)
                    if len(ds_rules) > 0:
                        species_rules[species] = ds_rules
                        log_msg(f"  {species}: using {ds_name} ({len(ds_rules)} rules)")
                        break  # Found dataset with rules for this species

    available_species = [s for s in species_datasets if s in species_rules and len(species_rules[s]) > 0]

    if len(available_species) < 2:
        log_msg("  Not enough species for transfer analysis")
        return pd.DataFrame()

    model = load_model(models[0], device='cuda')
    transfer_df = compute_transfer_matrix(
        species_rules, species_datasets, species_motif_hits,
        model, species_list=available_species,
    )
    transfer_df.to_parquet(os.path.join(RESULTS_DIR, 'module4', 'transfer_matrix.parquet'))

    # Phylogeny
    phylo = build_grammar_phylogeny(transfer_df, available_species)
    save_json(phylo, os.path.join(RESULTS_DIR, 'module4', 'grammar_phylogeny.json'))

    viz.plot_transfer_heatmap(
        transfer_df,
        os.path.join(RESULTS_DIR, 'module4', 'fig5_transfer_heatmap.pdf')
    )

    model.unload()
    log_msg(f"  Module 4 complete: {len(available_species)} species tested")
    return transfer_df


def run_module5(gsi_results, rules_df, datasets, args):
    """Module 5: Causal Determinants of Grammar."""
    log_msg("=" * 60)
    log_msg("MODULE 5: Causal Determinants of Grammar")
    log_msg("=" * 60)

    os.makedirs(os.path.join(RESULTS_DIR, 'module5'), exist_ok=True)

    # 5.1 Biophysics residual
    for ds_name in datasets:
        df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
        ds_gsi = gsi_results[gsi_results['dataset'] == ds_name] if len(gsi_results) > 0 else pd.DataFrame()

        if len(ds_gsi) > 0:
            # Merge GSI into dataset
            gsi_per_seq = ds_gsi.groupby('seq_id')['gsi'].mean()
            merged = df[df['seq_id'].isin(gsi_per_seq.index)]

            if len(merged) > 20:
                grammar_effects = gsi_per_seq.loc[merged['seq_id'].values].values
                bio_result = compute_biophysics_residual(merged, grammar_effects)
                save_json(bio_result, os.path.join(RESULTS_DIR, 'module5', f'{ds_name}_biophysics.json'))
                log_msg(f"  {ds_name} biophysics R2: {bio_result.get('biophysics_r2', 0):.3f}")

    # 5.2 TF structure mapping
    if len(rules_df) > 0:
        structure_map = build_structure_grammar_map(rules_df)
        structure_map.to_parquet(os.path.join(RESULTS_DIR, 'module5', 'structure_grammar_map.parquet'))

        structure_test = test_structure_predicts_grammar(rules_df)
        save_json(structure_test, os.path.join(RESULTS_DIR, 'module5', 'structure_predicts_grammar.json'))

        # 5.3 Strength tradeoff
        for ds_name in datasets:
            mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
            if os.path.exists(mh_path):
                motif_hits = pd.read_parquet(mh_path)
                ds_rules = rules_df[rules_df['dataset'] == ds_name]
                if len(ds_rules) > 20:
                    tradeoff = compute_grammar_strength_tradeoff(ds_rules, motif_hits)
                    save_json(tradeoff, os.path.join(RESULTS_DIR, 'module5', f'{ds_name}_strength_tradeoff.json'))

    # 5.4 Phase diagram
    if len(gsi_results) > 0:
        for ds_name in datasets:
            df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
            ds_gsi = gsi_results[gsi_results['dataset'] == ds_name]
            if len(ds_gsi) > 20:
                phase = compute_grammar_phase_diagram(ds_gsi, df)
                save_json(phase, os.path.join(RESULTS_DIR, 'module5', f'{ds_name}_phase_diagram.json'))

                if 'error' not in phase:
                    viz.plot_phase_diagram(
                        phase,
                        os.path.join(RESULTS_DIR, 'module5', f'{ds_name}_phase_diagram.pdf')
                    )

    log_msg("  Module 5 complete")


def run_module6(models, datasets, rules_df, args):
    """Module 6: Grammar-Optimized Design & Completeness."""
    log_msg("=" * 60)
    log_msg("MODULE 6: Grammar-Optimized Design & Completeness")
    log_msg("=" * 60)

    os.makedirs(os.path.join(RESULTS_DIR, 'module6'), exist_ok=True)
    model = load_model(models[0], device='cuda')

    for ds_name, ds_info in datasets.items():
        df = load_processed(os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed.parquet'))
        mh_path = os.path.join(DATA_DIR, 'processed', f'{ds_name}_processed_motif_hits.parquet')
        if not os.path.exists(mh_path):
            continue
        motif_hits = pd.read_parquet(mh_path)

        # Grammar features from rules
        rules_path = os.path.join(RESULTS_DIR, 'module2', 'grammar_rules_database.parquet')
        if os.path.exists(rules_path):
            grammar_features = pd.read_parquet(rules_path)
            grammar_features = grammar_features[grammar_features['dataset'] == ds_name]
        else:
            grammar_features = pd.DataFrame()

        # Completeness test
        completeness = compute_grammar_completeness(
            df, motif_hits, grammar_features, model,
            cell_type=ds_info['cell_type']
        )
        save_json(completeness, os.path.join(RESULTS_DIR, 'module6', f'{ds_name}_completeness.json'))

        if 'error' not in completeness:
            viz.plot_completeness_barplot(
                completeness,
                os.path.join(RESULTS_DIR, 'module6', f'{ds_name}_completeness.pdf')
            )
            log_msg(f"  {ds_name} completeness: vocab={completeness['vocabulary_r2']:.3f}, "
                    f"grammar={completeness['vocab_plus_full_grammar_r2']:.3f}, "
                    f"model={completeness['full_model_r2']:.3f}")

    model.unload()
    log_msg("  Module 6 complete")


def main():
    parser = argparse.ArgumentParser(description='GRAMLANG Full Pipeline')
    parser.add_argument('--module', type=int, default=0,
                        help='Run specific module (1-6), 0=all')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model names')
    parser.add_argument('--n-shuffles', type=int, default=50,
                        help='Number of shuffles for GSI')
    parser.add_argument('--max-enhancers', type=int, default=200,
                        help='Max enhancers for GSI census')
    parser.add_argument('--max-enhancers-rules', type=int, default=100,
                        help='Max enhancers for rule extraction')
    parser.add_argument('--max-per-k', type=int, default=30,
                        help='Max enhancers per motif count for compositionality')
    parser.add_argument('--n-arrangements', type=int, default=100,
                        help='Arrangements per enhancer for compositionality')
    args = parser.parse_args()

    models = args.models.split(',') if args.models else ALL_MODELS
    datasets = get_available_datasets()

    if not datasets:
        log_msg("ERROR: No preprocessed datasets found. Run preprocessing first.")
        return

    log_msg(f"Models: {models}")
    log_msg(f"Datasets: {list(datasets.keys())}")
    check_disk()

    # Run modules
    gsi_results = pd.DataFrame()
    rules_df = pd.DataFrame()

    if args.module in [0, 1]:
        gsi_results = run_module1(models, datasets, args)
    elif os.path.exists(os.path.join(RESULTS_DIR, 'module1', 'all_gsi_results.parquet')):
        gsi_results = pd.read_parquet(os.path.join(RESULTS_DIR, 'module1', 'all_gsi_results.parquet'))

    if args.module in [0, 2]:
        rules_df = run_module2(models, datasets, gsi_results, args)
    elif os.path.exists(os.path.join(RESULTS_DIR, 'module2', 'grammar_rules_database.parquet')):
        rules_df = pd.read_parquet(os.path.join(RESULTS_DIR, 'module2', 'grammar_rules_database.parquet'))

    if args.module in [0, 3]:
        run_module3(models, datasets, rules_df, args)

    if args.module in [0, 4]:
        run_module4(models, datasets, rules_df, args)

    if args.module in [0, 5]:
        run_module5(gsi_results, rules_df, datasets, args)

    if args.module in [0, 6]:
        run_module6(models, datasets, rules_df, args)

    check_disk()
    log_msg("Pipeline complete!")


if __name__ == '__main__':
    main()
