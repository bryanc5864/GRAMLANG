#!/usr/bin/env python3
"""
Critique experiments to strengthen the billboard model paper.

Experiments:
1. Synthetic grammar test - Can models detect KNOWN grammar rules?
2. Known grammar validation - Test BPNet's Nanog 10.5bp periodicity
3. Grammar-positive deep dive - What's special about the 10% with grammar?
4. Bootstrap confidence intervals - Statistical rigor
5. Natural MPRA variants - Find pairs differing only in arrangement
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model
from src.grammar.sensitivity import compute_gsi


def create_synthetic_grammar_sequences(n_sequences=100, seq_length=200):
    """
    Create synthetic sequences with KNOWN grammar rules.

    Rule 1: Motif A followed by Motif B within 20bp = HIGH expression
    Rule 2: Motif B followed by Motif A = LOW expression
    Rule 3: Spacing matters: 10bp spacing = HIGH, 50bp spacing = LOW
    """
    # Simple motifs (not real TF binding sites, but recognizable patterns)
    MOTIF_A = "GATAAGAT"  # GATA-like
    MOTIF_B = "CACGTGAC"  # E-box-like

    sequences = []
    labels = []  # 'high_grammar', 'low_grammar', 'no_grammar'
    grammar_rules = []

    rng = np.random.default_rng(42)
    bases = ['A', 'C', 'G', 'T']

    for i in range(n_sequences):
        # Random background
        bg = ''.join(rng.choice(bases, seq_length))

        if i < n_sequences // 3:
            # HIGH grammar: A then B, close spacing (10-20bp)
            pos_a = 50
            spacing = rng.integers(10, 21)
            pos_b = pos_a + len(MOTIF_A) + spacing
            seq = bg[:pos_a] + MOTIF_A + bg[pos_a+len(MOTIF_A):pos_b] + MOTIF_B + bg[pos_b+len(MOTIF_B):]
            seq = seq[:seq_length]
            sequences.append(seq)
            labels.append('high_grammar')
            grammar_rules.append({'rule': 'A_then_B_close', 'spacing': int(spacing)})

        elif i < 2 * n_sequences // 3:
            # LOW grammar: B then A (reversed order)
            pos_b = 50
            spacing = rng.integers(10, 21)
            pos_a = pos_b + len(MOTIF_B) + spacing
            seq = bg[:pos_b] + MOTIF_B + bg[pos_b+len(MOTIF_B):pos_a] + MOTIF_A + bg[pos_a+len(MOTIF_A):]
            seq = seq[:seq_length]
            sequences.append(seq)
            labels.append('low_grammar')
            grammar_rules.append({'rule': 'B_then_A', 'spacing': int(spacing)})

        else:
            # NO grammar: A and B far apart (>80bp)
            pos_a = 30
            pos_b = 150
            seq = bg[:pos_a] + MOTIF_A + bg[pos_a+len(MOTIF_A):pos_b] + MOTIF_B + bg[pos_b+len(MOTIF_B):]
            seq = seq[:seq_length]
            sequences.append(seq)
            labels.append('no_grammar')
            grammar_rules.append({'rule': 'far_apart', 'spacing': pos_b - pos_a - len(MOTIF_A)})

    return sequences, labels, grammar_rules, MOTIF_A, MOTIF_B


def experiment_1_synthetic_grammar(model_name='dnabert2', output_dir='results/critique'):
    """
    Test if models can detect KNOWN synthetic grammar rules.

    If models CAN'T detect these simple rules, interpretation B is supported
    (models fail to learn grammar).
    """
    print("\n=== Experiment 1: Synthetic Grammar Detection ===")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create synthetic sequences
    sequences, labels, rules, motif_a, motif_b = create_synthetic_grammar_sequences(n_sequences=150)

    # Load model (no probe needed - just embeddings)
    model = load_model(model_name, dataset_name='agarwal')

    # Get embeddings for all sequences
    print("  Getting embeddings...")
    embeddings = model.get_embeddings(sequences)

    # Also get expression predictions
    print("  Getting predictions...")
    predictions = model.predict_expression(sequences)

    # Analyze by grammar label
    results_by_label = defaultdict(list)
    for i, label in enumerate(labels):
        results_by_label[label].append({
            'embedding': embeddings[i],
            'prediction': predictions[i],
            'rule': rules[i]
        })

    # Statistical tests
    high_preds = [r['prediction'] for r in results_by_label['high_grammar']]
    low_preds = [r['prediction'] for r in results_by_label['low_grammar']]
    no_preds = [r['prediction'] for r in results_by_label['no_grammar']]

    # T-tests
    t_high_vs_low, p_high_vs_low = stats.ttest_ind(high_preds, low_preds)
    t_high_vs_no, p_high_vs_no = stats.ttest_ind(high_preds, no_preds)

    # Effect sizes (Cohen's d)
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx-1)*np.std(x)**2 + (ny-1)*np.std(y)**2) / (nx+ny-2))
        return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0

    d_high_vs_low = cohens_d(high_preds, low_preds)
    d_high_vs_no = cohens_d(high_preds, no_preds)

    # Embedding similarity analysis
    high_embs = np.array([r['embedding'] for r in results_by_label['high_grammar']])
    low_embs = np.array([r['embedding'] for r in results_by_label['low_grammar']])

    # Mean embedding distance between classes
    high_centroid = np.mean(high_embs, axis=0)
    low_centroid = np.mean(low_embs, axis=0)
    centroid_distance = 1 - cosine(high_centroid, low_centroid)

    results = {
        'experiment': 'synthetic_grammar',
        'model': model_name,
        'n_sequences': len(sequences),
        'motif_a': motif_a,
        'motif_b': motif_b,
        'predictions': {
            'high_grammar_mean': float(np.mean(high_preds)),
            'high_grammar_std': float(np.std(high_preds)),
            'low_grammar_mean': float(np.mean(low_preds)),
            'low_grammar_std': float(np.std(low_preds)),
            'no_grammar_mean': float(np.mean(no_preds)),
            'no_grammar_std': float(np.std(no_preds)),
        },
        'statistical_tests': {
            'high_vs_low_t': float(t_high_vs_low),
            'high_vs_low_p': float(p_high_vs_low),
            'high_vs_low_d': float(d_high_vs_low),
            'high_vs_no_t': float(t_high_vs_no),
            'high_vs_no_p': float(p_high_vs_no),
            'high_vs_no_d': float(d_high_vs_no),
        },
        'embedding_analysis': {
            'centroid_similarity': float(centroid_distance),
        },
        'interpretation': 'Model detects grammar' if p_high_vs_low < 0.05 else 'Model does NOT detect grammar'
    }

    # Save
    with open(output_path / 'exp1_synthetic_grammar.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results:")
    print(f"    High grammar pred: {np.mean(high_preds):.3f} ± {np.std(high_preds):.3f}")
    print(f"    Low grammar pred:  {np.mean(low_preds):.3f} ± {np.std(low_preds):.3f}")
    print(f"    High vs Low: t={t_high_vs_low:.2f}, p={p_high_vs_low:.4f}, d={d_high_vs_low:.3f}")
    print(f"    Interpretation: {results['interpretation']}")

    return results


def experiment_2_helical_periodicity(model_name='dnabert2', output_dir='results/critique'):
    """
    Test if models detect helical periodicity (10.5bp spacing).

    BPNet found Nanog binding is sensitive to 10.5bp periodicity.
    We create sequences with motif pairs at different spacings and test
    if 10-11bp shows different predictions than other spacings.
    """
    print("\n=== Experiment 2: Helical Periodicity Detection ===")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    MOTIF = "GATAAG"  # GATA motif

    rng = np.random.default_rng(42)
    bases = ['A', 'C', 'G', 'T']

    # Create sequences with different spacings
    spacings = list(range(5, 51, 1))  # 5bp to 50bp
    sequences_by_spacing = {}

    n_per_spacing = 20

    for spacing in spacings:
        seqs = []
        for _ in range(n_per_spacing):
            bg = ''.join(rng.choice(bases, 200))
            pos1 = 50
            pos2 = pos1 + len(MOTIF) + spacing
            seq = bg[:pos1] + MOTIF + bg[pos1+len(MOTIF):pos2] + MOTIF + bg[pos2+len(MOTIF):]
            seqs.append(seq[:200])
        sequences_by_spacing[spacing] = seqs

    # Load model
    model = load_model(model_name, dataset_name='agarwal')

    # Get predictions for each spacing
    print("  Getting predictions for each spacing...")
    predictions_by_spacing = {}
    for spacing, seqs in tqdm(sequences_by_spacing.items()):
        preds = model.predict_expression(seqs)
        predictions_by_spacing[spacing] = preds

    # Analyze periodicity
    mean_preds = {s: float(np.mean(p)) for s, p in predictions_by_spacing.items()}
    std_preds = {s: float(np.std(p)) for s, p in predictions_by_spacing.items()}

    # Check for 10.5bp periodicity
    # Compare 10-11bp vs 15-16bp (half period away)
    helical_spacings = [10, 11, 21, 22, 31, 32]  # Helical (n * 10.5)
    non_helical_spacings = [15, 16, 26, 27, 36, 37]  # Non-helical

    helical_preds = [p for s in helical_spacings if s in predictions_by_spacing
                     for p in predictions_by_spacing[s]]
    non_helical_preds = [p for s in non_helical_spacings if s in predictions_by_spacing
                         for p in predictions_by_spacing[s]]

    t_stat, p_value = stats.ttest_ind(helical_preds, non_helical_preds)

    # Autocorrelation to detect periodicity
    mean_series = [mean_preds[s] for s in sorted(mean_preds.keys())]

    results = {
        'experiment': 'helical_periodicity',
        'model': model_name,
        'motif': MOTIF,
        'spacings_tested': spacings,
        'n_per_spacing': n_per_spacing,
        'mean_predictions_by_spacing': mean_preds,
        'std_predictions_by_spacing': std_preds,
        'helical_analysis': {
            'helical_spacings': helical_spacings,
            'non_helical_spacings': non_helical_spacings,
            'helical_mean': float(np.mean(helical_preds)),
            'non_helical_mean': float(np.mean(non_helical_preds)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
        },
        'interpretation': 'Model detects helical periodicity' if p_value < 0.05 else 'Model does NOT detect helical periodicity'
    }

    with open(output_path / 'exp2_helical_periodicity.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results:")
    print(f"    Helical (10-11bp) mean: {np.mean(helical_preds):.3f}")
    print(f"    Non-helical (15-16bp) mean: {np.mean(non_helical_preds):.3f}")
    print(f"    t={t_stat:.2f}, p={p_value:.4f}")
    print(f"    Interpretation: {results['interpretation']}")

    return results


def experiment_3_grammar_positive_analysis(output_dir='results/critique'):
    """
    Deep dive into the ~10% of enhancers that DO show grammar.

    Questions:
    - What motifs are enriched?
    - What's different about their sequence composition?
    - Are they from specific genomic regions?
    """
    print("\n=== Experiment 3: Grammar-Positive Deep Dive ===")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load classification results
    classification_files = list(Path('results/confirmatory').glob('*_enhancer_classification.json'))

    if not classification_files:
        print("  No classification results found. Run confirmatory experiments first.")
        return None

    all_grammar_positive = []
    all_billboard = []

    for f in classification_files:
        with open(f) as file:
            data = json.load(file)

        dataset = data['dataset']
        for result in data.get('per_enhancer_results', []):
            entry = {
                'dataset': dataset,
                'seq_id': result['seq_id'],
                'gsi': result['gsi'],
                'p_value': result['p_value'],
                'classification': result['classification'],
                'n_motifs': result['n_motifs']
            }

            if result['classification'] in ['moderate', 'strong']:
                all_grammar_positive.append(entry)
            elif result['classification'] == 'billboard':
                all_billboard.append(entry)

    print(f"  Found {len(all_grammar_positive)} grammar-positive enhancers")
    print(f"  Found {len(all_billboard)} billboard enhancers")

    if not all_grammar_positive:
        print("  No grammar-positive enhancers found.")
        return None

    # Analysis
    gp_gsi = [e['gsi'] for e in all_grammar_positive]
    bb_gsi = [e['gsi'] for e in all_billboard]

    gp_motifs = [e['n_motifs'] for e in all_grammar_positive]
    bb_motifs = [e['n_motifs'] for e in all_billboard]

    # Statistical comparison
    t_motifs, p_motifs = stats.ttest_ind(gp_motifs, bb_motifs)

    # Dataset distribution
    gp_datasets = defaultdict(int)
    for e in all_grammar_positive:
        gp_datasets[e['dataset']] += 1

    results = {
        'experiment': 'grammar_positive_analysis',
        'n_grammar_positive': len(all_grammar_positive),
        'n_billboard': len(all_billboard),
        'grammar_positive_stats': {
            'mean_gsi': float(np.mean(gp_gsi)),
            'std_gsi': float(np.std(gp_gsi)),
            'mean_n_motifs': float(np.mean(gp_motifs)),
            'std_n_motifs': float(np.std(gp_motifs)),
        },
        'billboard_stats': {
            'mean_gsi': float(np.mean(bb_gsi)),
            'std_gsi': float(np.std(bb_gsi)),
            'mean_n_motifs': float(np.mean(bb_motifs)),
            'std_n_motifs': float(np.std(bb_motifs)),
        },
        'motif_count_comparison': {
            't_statistic': float(t_motifs),
            'p_value': float(p_motifs),
            'interpretation': 'Grammar-positive have MORE motifs' if np.mean(gp_motifs) > np.mean(bb_motifs) and p_motifs < 0.05 else 'No significant difference'
        },
        'dataset_distribution': dict(gp_datasets),
        'top_grammar_positive': sorted(all_grammar_positive, key=lambda x: x['gsi'], reverse=True)[:10]
    }

    with open(output_path / 'exp3_grammar_positive_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results:")
    print(f"    Grammar-positive: {len(all_grammar_positive)} ({len(all_grammar_positive)/(len(all_grammar_positive)+len(all_billboard))*100:.1f}%)")
    print(f"    GP mean motifs: {np.mean(gp_motifs):.1f} vs BB mean motifs: {np.mean(bb_motifs):.1f}")
    print(f"    Motif count difference: p={p_motifs:.4f}")
    print(f"    Dataset distribution: {dict(gp_datasets)}")

    return results


def experiment_4_bootstrap_confidence_intervals(output_dir='results/critique'):
    """
    Add bootstrap confidence intervals to key findings.
    """
    print("\n=== Experiment 4: Bootstrap Confidence Intervals ===")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load classification results
    classification_files = list(Path('results/confirmatory').glob('*_enhancer_classification.json'))

    results = {}

    for f in classification_files:
        with open(f) as file:
            data = json.load(file)

        dataset = data['dataset']
        per_enhancer = data.get('per_enhancer_results', [])

        if not per_enhancer:
            continue

        # Bootstrap the billboard percentage
        classifications = [r['classification'] for r in per_enhancer]
        n_bootstrap = 1000
        bootstrap_billboard_pcts = []

        rng = np.random.default_rng(42)

        for _ in range(n_bootstrap):
            sample = rng.choice(classifications, size=len(classifications), replace=True)
            billboard_pct = sum(1 for c in sample if c == 'billboard') / len(sample) * 100
            bootstrap_billboard_pcts.append(billboard_pct)

        ci_lower = np.percentile(bootstrap_billboard_pcts, 2.5)
        ci_upper = np.percentile(bootstrap_billboard_pcts, 97.5)

        results[dataset] = {
            'n_enhancers': len(per_enhancer),
            'billboard_pct': data['percentages'].get('billboard', 0),
            'bootstrap_mean': float(np.mean(bootstrap_billboard_pcts)),
            'bootstrap_std': float(np.std(bootstrap_billboard_pcts)),
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper),
        }

        print(f"  {dataset}: {results[dataset]['billboard_pct']:.1f}% [{ci_lower:.1f}%, {ci_upper:.1f}%]")

    # Overall bootstrap
    all_classifications = []
    for f in classification_files:
        with open(f) as file:
            data = json.load(file)
        for r in data.get('per_enhancer_results', []):
            all_classifications.append(r['classification'])

    bootstrap_overall = []
    for _ in range(1000):
        sample = rng.choice(all_classifications, size=len(all_classifications), replace=True)
        billboard_pct = sum(1 for c in sample if c == 'billboard') / len(sample) * 100
        bootstrap_overall.append(billboard_pct)

    results['overall'] = {
        'n_enhancers': len(all_classifications),
        'billboard_pct': sum(1 for c in all_classifications if c == 'billboard') / len(all_classifications) * 100,
        'bootstrap_mean': float(np.mean(bootstrap_overall)),
        'bootstrap_std': float(np.std(bootstrap_overall)),
        'ci_95_lower': float(np.percentile(bootstrap_overall, 2.5)),
        'ci_95_upper': float(np.percentile(bootstrap_overall, 97.5)),
    }

    print(f"\n  Overall: {results['overall']['billboard_pct']:.1f}% [{results['overall']['ci_95_lower']:.1f}%, {results['overall']['ci_95_upper']:.1f}%]")

    with open(output_path / 'exp4_bootstrap_ci.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def experiment_5_mpra_grammar_variants(model_name='dnabert2', output_dir='results/critique'):
    """
    Find natural pairs in MPRA data that differ primarily in motif arrangement.

    This provides experimental validation if such pairs exist and have
    measured expression differences.
    """
    print("\n=== Experiment 5: Natural MPRA Grammar Variants ===")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = ['agarwal', 'jores', 'klein']
    all_results = {}

    for dataset in datasets:
        print(f"\n  Processing {dataset}...")

        data_path = Path('data/processed') / f'{dataset}_processed.parquet'
        motif_path = Path('data/processed') / f'{dataset}_processed_motif_hits.parquet'

        if not data_path.exists():
            print(f"    Dataset not found")
            continue

        df = pd.read_parquet(data_path)
        motif_df = pd.read_parquet(motif_path)

        # Group motifs by sequence
        motif_groups = motif_df.groupby('seq_id')

        # Create motif vocabulary signature for each sequence
        vocab_signatures = {}
        for seq_id in df['seq_id']:
            if seq_id in motif_groups.groups:
                motifs = motif_groups.get_group(seq_id)
                # Vocabulary = sorted list of motif names
                vocab = tuple(sorted(motifs['motif_name'].tolist()))
                if vocab not in vocab_signatures:
                    vocab_signatures[vocab] = []
                vocab_signatures[vocab].append(seq_id)

        # Find pairs with same vocabulary
        same_vocab_pairs = []
        for vocab, seq_ids in vocab_signatures.items():
            if len(seq_ids) >= 2 and len(vocab) >= 2:
                # Get expression values
                for i, id1 in enumerate(seq_ids[:10]):  # Limit to first 10
                    for id2 in seq_ids[i+1:i+11]:
                        expr1 = df[df['seq_id'] == id1]['expression'].values[0]
                        expr2 = df[df['seq_id'] == id2]['expression'].values[0]
                        same_vocab_pairs.append({
                            'seq_id_1': id1,
                            'seq_id_2': id2,
                            'vocabulary': list(vocab),
                            'n_motifs': len(vocab),
                            'expr_1': float(expr1),
                            'expr_2': float(expr2),
                            'expr_diff': float(abs(expr1 - expr2))
                        })

        if same_vocab_pairs:
            # Sort by expression difference
            same_vocab_pairs = sorted(same_vocab_pairs, key=lambda x: x['expr_diff'], reverse=True)

            expr_diffs = [p['expr_diff'] for p in same_vocab_pairs]

            all_results[dataset] = {
                'n_same_vocab_pairs': len(same_vocab_pairs),
                'mean_expr_diff': float(np.mean(expr_diffs)),
                'max_expr_diff': float(np.max(expr_diffs)),
                'frac_large_diff': float(np.mean([d > 0.5 for d in expr_diffs])),
                'top_pairs': same_vocab_pairs[:20],
            }

            print(f"    Found {len(same_vocab_pairs)} same-vocabulary pairs")
            print(f"    Mean expression diff: {np.mean(expr_diffs):.3f}")
            print(f"    Max expression diff: {np.max(expr_diffs):.3f}")
            print(f"    Fraction with |Δexpr| > 0.5: {np.mean([d > 0.5 for d in expr_diffs])*100:.1f}%")
        else:
            all_results[dataset] = {'n_same_vocab_pairs': 0}
            print(f"    No same-vocabulary pairs found")

    with open(output_path / 'exp5_mpra_grammar_variants.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run critique experiments')
    parser.add_argument('--experiments', nargs='+', type=int, default=[1,2,3,4,5],
                       help='Which experiments to run (1-5)')
    parser.add_argument('--model', type=str, default='dnabert2')
    parser.add_argument('--output-dir', type=str, default='results/critique')
    args = parser.parse_args()

    np.random.seed(42)

    results = {}

    if 1 in args.experiments:
        results['exp1'] = experiment_1_synthetic_grammar(args.model, args.output_dir)

    if 2 in args.experiments:
        results['exp2'] = experiment_2_helical_periodicity(args.model, args.output_dir)

    if 3 in args.experiments:
        results['exp3'] = experiment_3_grammar_positive_analysis(args.output_dir)

    if 4 in args.experiments:
        results['exp4'] = experiment_4_bootstrap_confidence_intervals(args.output_dir)

    if 5 in args.experiments:
        results['exp5'] = experiment_5_mpra_grammar_variants(args.model, args.output_dir)

    print("\n=== All Critique Experiments Complete ===")
    print(f"Results saved to {args.output_dir}/")

    # Summary
    print("\n=== SUMMARY ===")
    if 'exp1' in results and results['exp1']:
        print(f"Exp 1 (Synthetic): {results['exp1']['interpretation']}")
    if 'exp2' in results and results['exp2']:
        print(f"Exp 2 (Helical): {results['exp2']['interpretation']}")
    if 'exp3' in results and results['exp3']:
        print(f"Exp 3 (GP Analysis): {results['exp3']['n_grammar_positive']} grammar-positive enhancers")
    if 'exp4' in results and results['exp4']:
        overall = results['exp4'].get('overall', {})
        print(f"Exp 4 (Bootstrap): {overall.get('billboard_pct', 0):.1f}% [{overall.get('ci_95_lower', 0):.1f}%, {overall.get('ci_95_upper', 0):.1f}%]")
    if 'exp5' in results and results['exp5']:
        total_pairs = sum(r.get('n_same_vocab_pairs', 0) for r in results['exp5'].values())
        print(f"Exp 5 (MPRA variants): {total_pairs} same-vocabulary pairs found")


if __name__ == '__main__':
    main()
