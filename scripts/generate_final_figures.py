"""Generate all final summary figures for GRAMLANG project."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = '/home/bcheng/grammar/results'
FIGURES_DIR = '/home/bcheng/grammar/results/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

PALETTE = sns.color_palette("Set2", 10)
MODEL_COLORS = {
    'dnabert2': PALETTE[4],
    'nt': PALETTE[3],
    'hyenadna': PALETTE[5],
}


def fig1_gsi_overview():
    """Figure 1: Grammar Existence - GSI Distribution and Summary."""
    gsi_df = pd.read_parquet(os.path.join(RESULTS_DIR, 'module1', 'all_gsi_results.parquet'))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (A) GSI distribution by model
    ax = axes[0, 0]
    for model in ['dnabert2', 'nt', 'hyenadna']:
        data = gsi_df[gsi_df['model'] == model]['gsi']
        ax.hist(data, bins=40, alpha=0.45, label=model,
                color=MODEL_COLORS.get(model, 'gray'))
    ax.set_xlabel('Grammar Sensitivity Index (GSI)')
    ax.set_ylabel('Count')
    ax.set_title('(A) GSI Distribution by Model')
    ax.legend()
    ax.axvline(x=0.05, color='red', linestyle=':', alpha=0.5, label='p<0.05 threshold')

    # (B) GSI by dataset
    ax = axes[0, 1]
    if 'dataset' in gsi_df.columns:
        datasets = sorted(gsi_df['dataset'].unique())
        positions = np.arange(len(datasets))
        width = 0.25
        for idx, model in enumerate(['dnabert2', 'nt', 'hyenadna']):
            means = []
            stds = []
            for ds in datasets:
                vals = gsi_df[(gsi_df['model'] == model) & (gsi_df['dataset'] == ds)]['gsi']
                means.append(vals.mean())
                stds.append(vals.std())
            ax.bar(positions + idx * width, means, width, yerr=stds,
                   label=model, color=MODEL_COLORS[model], alpha=0.8, capsize=3)
        ax.set_xticks(positions + width)
        ax.set_xticklabels(datasets)
        ax.set_ylabel('Mean GSI')
        ax.set_title('(B) GSI by Dataset and Model')
        ax.legend()

    # (C) Grammar information content
    ax = axes[1, 0]
    info_df = pd.read_parquet(os.path.join(RESULTS_DIR, 'module1', 'grammar_information.parquet'))
    models = info_df['model'].unique()
    x = np.arange(len(models))
    width = 0.35
    for i, dataset in enumerate(sorted(info_df['dataset'].unique())):
        subset = info_df[info_df['dataset'] == dataset]
        bits = [subset[subset['model'] == m]['bits_of_grammar'].values[0] if len(subset[subset['model'] == m]) > 0 else 0 for m in models]
        ax.bar(x + i * width, bits, width, label=dataset, alpha=0.8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models)
    ax.set_ylabel('Bits of Grammar')
    ax.set_title('(C) Grammar Information Content')
    ax.legend()

    # (D) Max disruption by model
    ax = axes[1, 1]
    for model in ['dnabert2', 'nt', 'hyenadna']:
        data = gsi_df[gsi_df['model'] == model]['max_disruption']
        ax.hist(data, bins=40, alpha=0.45, label=model,
                color=MODEL_COLORS.get(model, 'gray'))
    ax.set_xlabel('Maximum Expression Disruption')
    ax.set_ylabel('Count')
    ax.set_title('(D) Maximum Disruption Distribution')
    ax.legend()

    plt.suptitle('Figure 1: Grammar Existence & Sensitivity Census', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_grammar_existence.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_grammar_existence.png'))
    plt.close()
    print("  Fig 1: Grammar existence complete")


def fig2_compositionality():
    """Figure 2: Compositionality and Chomsky Classification."""
    comp_df = pd.read_parquet(os.path.join(RESULTS_DIR, 'module3', 'compositionality_results.parquet'))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (A) Compositionality gap vs k
    ax = axes[0]
    gap_by_k = comp_df.groupby('n_motifs')['compositionality_gap'].agg(
        ['mean', 'std', 'count']
    ).reset_index()
    ax.errorbar(gap_by_k['n_motifs'], gap_by_k['mean'],
                yerr=gap_by_k['std'] / np.sqrt(gap_by_k['count']),
                fmt='o-', color='steelblue', capsize=4, markersize=8, linewidth=2)
    ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Regular grammar')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Context-free')
    ax.axhline(y=0.99, color='red', linestyle='--', alpha=0.5, label='Context-sensitive')
    ax.set_xlabel('Number of Motifs (k)')
    ax.set_ylabel('Compositionality Gap (1 - R²)')
    ax.set_title('(A) Compositionality Gap vs k')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # (B) Per-model compositionality
    ax = axes[1]
    for model in ['dnabert2', 'nt', 'hyenadna']:
        model_data = comp_df[comp_df['model'] == model]
        gap_by_k_m = model_data.groupby('n_motifs')['compositionality_gap'].mean().reset_index()
        ax.plot(gap_by_k_m['n_motifs'], gap_by_k_m['compositionality_gap'],
                'o-', label=model, color=MODEL_COLORS[model], markersize=6)
    ax.set_xlabel('Number of Motifs (k)')
    ax.set_ylabel('Compositionality Gap')
    ax.set_title('(B) Per-Model Compositionality')
    ax.legend()
    ax.set_ylim(0.95, 1.0)

    # (C) Pairwise R² distribution
    ax = axes[2]
    ax.hist(comp_df['pairwise_r2'], bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=comp_df['pairwise_r2'].mean(), color='red', linestyle='--',
               label=f"Mean R²={comp_df['pairwise_r2'].mean():.3f}")
    ax.set_xlabel('Pairwise Prediction R²')
    ax.set_ylabel('Count')
    ax.set_title('(C) Pairwise Rule Prediction Quality')
    ax.legend()

    plt.suptitle('Figure 2: Compositionality & Chomsky Classification', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_compositionality.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_compositionality.png'))
    plt.close()
    print("  Fig 2: Compositionality complete")


def fig3_rules_consensus():
    """Figure 3: Grammar Rules & Cross-Model Consensus."""
    rules_df = pd.read_parquet(os.path.join(RESULTS_DIR, 'module2', 'grammar_rules_database.parquet'))
    consensus_df = pd.read_parquet(os.path.join(RESULTS_DIR, 'module2', 'consensus_scores.parquet'))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (A) Rule fold change distribution
    ax = axes[0, 0]
    ax.hist(rules_df['fold_change'], bins=50, color='coral', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=rules_df['fold_change'].mean(), color='red', linestyle='--',
               label=f"Mean={rules_df['fold_change'].mean():.2f}")
    ax.set_xlabel('Fold Change (Best / Worst Arrangement)')
    ax.set_ylabel('Count')
    ax.set_title('(A) Grammar Rule Strength Distribution')
    ax.legend()

    # (B) Spacing sensitivity distribution
    ax = axes[0, 1]
    ax.hist(rules_df['spacing_sensitivity'], bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Spacing Sensitivity (CV across spacings)')
    ax.set_ylabel('Count')
    ax.set_title('(B) Spacing Sensitivity Distribution')

    # (C) Consensus score distribution
    ax = axes[1, 0]
    if 'consensus_score' in consensus_df.columns:
        ax.hist(consensus_df['consensus_score'], bins=40, color='mediumpurple',
                alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.axvline(x=consensus_df['consensus_score'].mean(), color='red', linestyle='--',
                   label=f"Mean={consensus_df['consensus_score'].mean():.3f}")
        ax.set_xlabel('Consensus Score')
        ax.set_ylabel('Count')
        ax.set_title('(C) Cross-Model Consensus Distribution')
        ax.legend()

    # (D) Helical phasing score
    ax = axes[1, 1]
    if 'helical_phase_score' in rules_df.columns:
        ax.hist(rules_df['helical_phase_score'], bins=50, color='forestgreen',
                alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.axvline(x=2.0, color='red', linestyle='--', label='Helical threshold (2.0)')
        n_helical = (rules_df['helical_phase_score'] > 2.0).sum()
        ax.set_xlabel('Helical Phase Score (10.5bp periodicity)')
        ax.set_ylabel('Count')
        ax.set_title(f'(D) Helical Phasing ({n_helical} rules > threshold)')
        ax.legend()

    plt.suptitle('Figure 3: Grammar Rules & Cross-Architecture Consensus', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_rules_consensus.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_rules_consensus.png'))
    plt.close()
    print("  Fig 3: Rules & consensus complete")


def fig4_transfer():
    """Figure 4: Cross-Species Grammar Transfer."""
    transfer_df = pd.read_parquet(os.path.join(RESULTS_DIR, 'module4', 'transfer_matrix.parquet'))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (A) Transfer R² heatmap
    ax = axes[0]
    species = sorted(transfer_df['source'].unique())
    n = len(species)
    matrix = np.zeros((n, n))
    for _, row in transfer_df.iterrows():
        i = species.index(row['source'])
        j = species.index(row['target'])
        matrix[i, j] = row['transfer_r2']
    sns.heatmap(matrix, xticklabels=species, yticklabels=species,
                annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=0.05, ax=ax,
                cbar_kws={'label': 'Transfer R²'})
    ax.set_xlabel('Target Species')
    ax.set_ylabel('Source Species')
    ax.set_title('(A) Cross-Species Grammar Transfer (R²)')

    # (B) Transfer correlation heatmap
    ax = axes[1]
    corr_matrix = np.zeros((n, n))
    for _, row in transfer_df.iterrows():
        i = species.index(row['source'])
        j = species.index(row['target'])
        corr_matrix[i, j] = row['transfer_corr']
    sns.heatmap(corr_matrix, xticklabels=species, yticklabels=species,
                annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-0.3, vmax=0.3, ax=ax,
                cbar_kws={'label': 'Transfer Correlation'})
    ax.set_xlabel('Target Species')
    ax.set_ylabel('Source Species')
    ax.set_title('(B) Cross-Species Grammar Transfer (Correlation)')

    plt.suptitle('Figure 4: Cross-Species Grammar Transfer', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_transfer.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_transfer.png'))
    plt.close()
    print("  Fig 4: Transfer complete")


def fig5_biophysics():
    """Figure 5: Causal Determinants - Biophysics and Phase Diagrams."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (A) Biophysics R² comparison
    ax = axes[0, 0]
    datasets_bio = ['Vaishnav\n(Yeast)', 'Klein\n(Human)']
    r2_vals = [0.082, 0.636]
    colors = ['#3498db', '#e74c3c']
    bars = ax.bar(datasets_bio, r2_vals, color=colors, edgecolor='black', linewidth=0.5, width=0.5)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('Biophysics R² (5-fold CV)')
    ax.set_title('(A) Biophysical Explanation of Grammar')
    ax.set_ylim(0, 0.85)

    # (B) Top features comparison
    ax = axes[0, 1]
    with open(os.path.join(RESULTS_DIR, 'module5', 'klein_biophysics.json')) as f:
        klein_bio = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'module5', 'vaishnav_biophysics.json')) as f:
        vaish_bio = json.load(f)

    top_features = ['dinuc_CG', 'shape_MGW_mean', 'shape_ProT_mean', 'dinuc_GC', 'shape_MGW_std',
                    'shape_Roll_std', 'shape_Roll_mean', 'dinuc_GG', 'gc_content', 'dinuc_CT']
    klein_imp = [klein_bio['feature_importances'].get(f, 0) for f in top_features]
    vaish_imp = [vaish_bio['feature_importances'].get(f, 0) for f in top_features]

    x = np.arange(len(top_features))
    width = 0.35
    ax.barh(x - width/2, klein_imp, width, label='Klein (Human)', color='#e74c3c', alpha=0.8)
    ax.barh(x + width/2, vaish_imp, width, label='Vaishnav (Yeast)', color='#3498db', alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels([f.replace('dinuc_', '').replace('shape_', '').replace('_', ' ') for f in top_features], fontsize=9)
    ax.set_xlabel('Feature Importance')
    ax.set_title('(B) Top Biophysical Feature Importances')
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # (C) Vaishnav phase diagram
    ax = axes[1, 0]
    with open(os.path.join(RESULTS_DIR, 'module5', 'vaishnav_phase_diagram.json')) as f:
        vaish_phase = json.load(f)
    grid = np.array(vaish_phase['phase_grid'])
    grid[grid == 0] = np.nan
    im = ax.imshow(grid.T, origin='lower', aspect='auto',
                   extent=[vaish_phase['motif_count_edges'][0], vaish_phase['motif_count_edges'][-1],
                           vaish_phase['density_edges'][0], vaish_phase['density_edges'][-1]],
                   cmap='hot', vmin=0.05, vmax=0.10)
    plt.colorbar(im, ax=ax, label='Mean GSI')
    ax.set_xlabel('Motif Count')
    ax.set_ylabel('Motif Density (motifs/bp)')
    ax.set_title('(C) Phase Diagram: Vaishnav (Yeast)')

    # (D) Klein phase diagram
    ax = axes[1, 1]
    with open(os.path.join(RESULTS_DIR, 'module5', 'klein_phase_diagram.json')) as f:
        klein_phase = json.load(f)
    grid = np.array(klein_phase['phase_grid'])
    grid[grid == 0] = np.nan
    im = ax.imshow(grid.T, origin='lower', aspect='auto',
                   extent=[klein_phase['motif_count_edges'][0], klein_phase['motif_count_edges'][-1],
                           klein_phase['density_edges'][0], klein_phase['density_edges'][-1]],
                   cmap='hot', vmin=0.05, vmax=0.10)
    plt.colorbar(im, ax=ax, label='Mean GSI')
    ax.set_xlabel('Motif Count')
    ax.set_ylabel('Motif Density (motifs/bp)')
    ax.set_title('(D) Phase Diagram: Klein (Human)')

    plt.suptitle('Figure 5: Causal Determinants of Grammar', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_biophysics.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_biophysics.png'))
    plt.close()
    print("  Fig 5: Biophysics & phase diagrams complete")


def fig6_completeness():
    """Figure 6: Grammar Completeness Ceiling."""
    datasets = ['agarwal', 'de_almeida', 'vaishnav', 'jores', 'klein']
    labels = ['Agarwal\n(K562)', 'de Almeida\n(Neural)', 'Vaishnav\n(Yeast)', 'Jores\n(Plant)', 'Klein\n(HepG2)']

    completeness_data = {}
    for ds in datasets:
        path = os.path.join(RESULTS_DIR, 'module6', f'{ds}_completeness.json')
        if os.path.exists(path):
            with open(path) as f:
                completeness_data[ds] = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (A) Stacked R² comparison across datasets
    ax = axes[0]
    x = np.arange(len(datasets))
    vocab_r2 = [completeness_data[ds]['vocabulary_r2'] for ds in datasets]
    grammar_add = [max(0, completeness_data[ds]['vocab_plus_full_grammar_r2'] - completeness_data[ds]['vocabulary_r2']) for ds in datasets]
    model_r2 = [completeness_data[ds]['full_model_r2'] for ds in datasets]
    replicate_r2 = [completeness_data[ds]['replicate_r2'] for ds in datasets]

    width = 0.2
    ax.bar(x - 1.5*width, vocab_r2, width, label='Vocabulary', color='#e74c3c', alpha=0.8)
    ax.bar(x - 0.5*width, [v + g for v, g in zip(vocab_r2, grammar_add)], width,
           label='Vocab + Grammar', color='#f1c40f', alpha=0.8)
    ax.bar(x + 0.5*width, model_r2, width, label='Full Model', color='#2ecc71', alpha=0.8)
    ax.bar(x + 1.5*width, replicate_r2, width, label='MPRA Replicate', color='#3498db', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('R² (Expression Prediction)')
    ax.set_title('(A) Expression Prediction by Model Level')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)

    # (B) Grammar completeness percentages
    ax = axes[1]
    completeness_pct = [completeness_data[ds]['grammar_completeness'] * 100 for ds in datasets]
    grammar_contrib = [completeness_data[ds]['grammar_contribution'] for ds in datasets]

    colors_bar = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71', '#9b59b6']
    bars = ax.bar(x, completeness_pct, color=colors_bar, edgecolor='black', linewidth=0.5, width=0.6)
    for bar, val in zip(bars, completeness_pct):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Grammar Completeness (%)')
    ax.set_title('(B) Grammar Completeness Ceiling')
    ax.set_ylim(0, 25)

    plt.suptitle('Figure 6: Grammar Completeness Ceiling', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_completeness.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_completeness.png'))
    plt.close()
    print("  Fig 6: Completeness complete")


def fig7_summary():
    """Figure 7: Grand Summary - Key Findings Overview."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (A) Key numbers
    ax = axes[0, 0]
    ax.axis('off')
    summary_text = (
        "GRAMLANG: Key Numbers\n"
        "=" * 35 + "\n\n"
        "Models tested:          3\n"
        "Datasets:               5\n"
        "GSI measurements:     1,200\n"
        "Grammar rules:        4,218\n"
        "Compositionality tests: 1,200\n\n"
        "Mean GSI:              0.078\n"
        "Frac significant:      100%\n"
        "Mean consensus:        0.482\n"
        "Orientation agreement: 82.5%\n"
        "Compositionality gap:  0.990\n"
        "Cross-species transfer: 0.000\n"
        "Grammar completeness: 6-18%"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title('(A) Summary Statistics')

    # (B) GSI by model (simplified)
    ax = axes[0, 1]
    models = ['DNABERT-2', 'NT v2', 'HyenaDNA']
    mean_gsi = [0.0875, 0.0952, 0.0479]
    colors = [MODEL_COLORS['dnabert2'], MODEL_COLORS['nt'], MODEL_COLORS['hyenadna']]
    bars = ax.bar(models, mean_gsi, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, mean_gsi):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean GSI')
    ax.set_title('(B) Grammar Sensitivity by Model')
    ax.set_ylim(0, 0.13)

    # (C) Compositionality gap summary
    ax = axes[0, 2]
    k_vals = [3, 4, 5, 6, 7]
    gaps = [0.992, 0.989, 0.988, 0.989, 0.991]
    ax.plot(k_vals, gaps, 'o-', color='steelblue', markersize=10, linewidth=2)
    ax.fill_between(k_vals, 0.985, 0.995, alpha=0.1, color='steelblue')
    ax.set_xlabel('Number of Motifs (k)')
    ax.set_ylabel('Compositionality Gap')
    ax.set_title('(C) Non-Compositionality')
    ax.set_ylim(0.98, 1.0)
    ax.axhline(y=0.99, color='red', linestyle=':', alpha=0.3)

    # (D) Transfer matrix
    ax = axes[1, 0]
    transfer_matrix = np.array([[0.015, 0.000], [0.000, 0.020]])
    sns.heatmap(transfer_matrix, xticklabels=['Human', 'Yeast'],
                yticklabels=['Human', 'Yeast'], annot=True, fmt='.3f',
                cmap='YlOrRd', vmin=0, vmax=0.05, ax=ax,
                cbar_kws={'label': 'R²'})
    ax.set_title('(D) Cross-Species Transfer')

    # (E) Biophysics R²
    ax = axes[1, 1]
    species = ['Yeast', 'Human']
    bio_r2 = [0.082, 0.636]
    remaining = [1 - r for r in bio_r2]
    ax.barh(species, bio_r2, color='#e74c3c', label='Biophysics-explained')
    ax.barh(species, remaining, left=bio_r2, color='#3498db', alpha=0.4, label='Unexplained')
    ax.set_xlabel('Fraction of Grammar Variance')
    ax.set_title('(E) Biophysical Basis of Grammar')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    # (F) Completeness
    ax = axes[1, 2]
    ds_labels = ['Agarwal', 'de Almeida', 'Vaishnav', 'Jores', 'Klein']
    completeness = [17.7, 5.7, 11.1, 12.4, 16.6]
    colors_ds = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71', '#9b59b6']
    bars = ax.bar(ds_labels, completeness, color=colors_ds, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, completeness):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Grammar Completeness (%)')
    ax.set_title('(F) Grammar Completeness Ceiling')
    ax.set_ylim(0, 25)
    ax.tick_params(axis='x', rotation=30)

    plt.suptitle('Figure 7: GRAMLANG - Complete Results Summary', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_summary.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig7_summary.png'))
    plt.close()
    print("  Fig 7: Summary complete")


if __name__ == '__main__':
    print("Generating final figures...")
    fig1_gsi_overview()
    fig2_compositionality()
    fig3_rules_consensus()
    fig4_transfer()
    fig5_biophysics()
    fig6_completeness()
    fig7_summary()
    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print("Files generated:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        size = os.path.getsize(os.path.join(FIGURES_DIR, f))
        print(f"  {f}: {size/1024:.1f} KB")
