"""
Generate compact manuscript-ready figures for the GRAMLANG paper.
Sized for NeurIPS format (5.5in text width). No suptitles — captions in LaTeX.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'manuscript_figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# NeurIPS-compact defaults
plt.rcParams.update({
    'font.size': 7,
    'font.family': 'sans-serif',
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.4,
    'ytick.major.width': 0.4,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'lines.linewidth': 1.0,
    'lines.markersize': 3,
})

PALETTE = sns.color_palette("Set2", 10)
MODEL_COLORS = {'dnabert2': PALETTE[4], 'nt': PALETTE[3], 'hyenadna': PALETTE[5]}
DS_COLORS = ['#e74c3c', '#2ecc71', '#e67e22']


def load_json(path):
    with open(os.path.join(RESULTS_DIR, path)) as f:
        return json.load(f)


def fig1_spacer_confound():
    """Figure 1: Spacer Confound (2x2, compact)."""
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.8))
    plt.subplots_adjust(hspace=0.55, wspace=0.4)

    datasets = ['agarwal', 'jores', 'de_almeida']
    ds_labels = ['Agarwal', 'Jores', 'de Almeida']

    # (A) Factorial decomposition
    ax = axes[0, 0]
    components = ['position', 'orientation', 'spacer']
    comp_colors = ['#3498db', '#e67e22', '#e74c3c']
    comp_labels = ['Position', 'Orientation', 'Spacer']
    x = np.arange(len(datasets))
    width = 0.22
    for i, (comp, color, label) in enumerate(zip(components, comp_colors, comp_labels)):
        fracs = []
        for ds in datasets:
            data = load_json(f'v3/factorial_decomposition/{ds}_dnabert2_factorial_summary.json')
            fracs.append(data['effect_sizes'][comp]['median_fraction_of_full'] * 100)
        ax.bar(x + i * width, fracs, width, label=label, color=color, alpha=0.85,
               edgecolor='black', linewidth=0.2)
    ax.set_xticks(x + width)
    ax.set_xticklabels(ds_labels, fontsize=6)
    ax.set_ylabel('% of Full Variance')
    ax.set_title('(A) Factorial Decomposition')
    ax.legend(fontsize=5, loc='upper right', frameon=False)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)

    # (B) Spacer ablation
    ax = axes[0, 1]
    ablation_data = load_json('v3/spacer_ablation/spacer_ablation_summary.json')
    effects = ['motif_only', 'dinuc_shuffle', 'gc_shift', 'random_replace']
    eff_labels = ['Motif', 'Dinuc', 'GC', 'Random']
    for ds_idx, (ds, dl) in enumerate(zip(datasets, ds_labels)):
        deltas = [ablation_data[ds]['variant_effects'][e]['median_delta'] for e in effects]
        ax.plot(range(len(effects)), deltas, 'o-', label=dl, color=DS_COLORS[ds_idx], markersize=3)
    ax.set_xticks(range(len(effects)))
    ax.set_xticklabels(eff_labels, fontsize=6)
    ax.set_ylabel('Median |$\\Delta$ Expr|')
    ax.set_title('(B) Spacer Ablation')
    ax.legend(fontsize=5, frameon=False)

    # (C) Feature decomposition R^2
    ax = axes[1, 0]
    feat_data = load_json('v3/feature_decomposition/feature_decomposition_summary.json')
    fcs = ['gc_only', 'dinuc_only', 'shape_only', 'trinuc_only', 'all_features']
    fc_labels = ['GC', 'Dinuc', 'Shape', 'Trinuc', 'All']
    x = np.arange(len(fcs))
    width = 0.25
    for ds_idx, (ds, dl) in enumerate(zip(datasets, ds_labels)):
        r2s = [feat_data[ds][fc]['mean_r2'] for fc in fcs]
        ax.bar(x + ds_idx * width, r2s, width, label=dl, color=DS_COLORS[ds_idx],
               alpha=0.85, edgecolor='black', linewidth=0.2)
    ax.set_xticks(x + width)
    ax.set_xticklabels(fc_labels, fontsize=6)
    ax.set_ylabel('$R^2$')
    ax.set_title('(C) Feature Prediction of Expression')
    ax.legend(fontsize=5, frameon=False)

    # (D) GC-expression correlation reversal
    ax = axes[1, 1]
    gc_corrs = [feat_data[ds]['correlations']['gc_content']['r'] for ds in datasets]
    colors_bar = ['#e74c3c' if v > 0 else '#3498db' for v in gc_corrs]
    bars = ax.bar(ds_labels, gc_corrs, color=colors_bar, edgecolor='black', linewidth=0.3, width=0.5)
    for bar, val in zip(bars, gc_corrs):
        y = bar.get_height() + (0.03 if val > 0 else -0.08)
        ax.text(bar.get_x() + bar.get_width() / 2., y, f'{val:+.2f}',
                ha='center', va='bottom' if val > 0 else 'top', fontsize=6, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.4)
    ax.set_ylabel('GC-Expr Correlation ($r$)')
    ax.set_title('(D) GC Sensitivity Reversal')
    ax.set_ylim(-1, 1)

    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_spacer_confound.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_spacer_confound.png'))
    plt.close()
    print("  Fig 1 done")


def fig2_positive_control():
    """Figure 2: Positive Control (1x2, compact)."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.0))
    plt.subplots_adjust(wspace=0.4)

    pc_data = load_json('v3/positive_control/dnabert2_positive_control.json')
    orient = pc_data['orientation']

    # (A) Effect size distribution
    ax = axes[0]
    thresholds = ['Any', '>0.05', '>0.10', '>0.15', '>0.20']
    pcts = [100, 50, 17, 10, 3]
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
    bars = ax.bar(thresholds, pcts, color=colors, edgecolor='black', linewidth=0.3, width=0.6)
    ax.set_ylabel('% of Pairs')
    ax.set_xlabel('$|\\Delta|$ threshold')
    ax.set_title('(A) Effect Size Distribution')
    ax.text(0.95, 0.95, f'$p = {orient["p_value"]:.0e}$\n$n = {orient["n_pairs"]}$ pairs',
            transform=ax.transAxes, ha='right', va='top', fontsize=5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, linewidth=0.3))

    # (B) Method comparison
    ax = axes[1]
    methods = ['VP Shuffle\n(confounded)', 'Controlled\nDesign']
    sig = [8.3, 100]
    colors_c = ['#e74c3c', '#2ecc71']
    bars = ax.bar(methods, sig, color=colors_c, edgecolor='black', linewidth=0.3, width=0.5)
    for bar, val in zip(bars, sig):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1.5,
                f'{val}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('(B) Method Comparison')
    ax.set_ylim(0, 118)

    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_positive_control.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_positive_control.png'))
    plt.close()
    print("  Fig 2 done")


def fig3_gsi_census():
    """Figure 3: GSI Census (2x2, compact)."""
    gsi_df = pd.read_parquet(os.path.join(RESULTS_DIR, 'module1', 'all_gsi_results.parquet'))
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.8))
    plt.subplots_adjust(hspace=0.55, wspace=0.4)

    # (A) GSI distribution by model
    ax = axes[0, 0]
    for model in ['dnabert2', 'nt', 'hyenadna']:
        data = gsi_df[gsi_df['model'] == model]['gsi'].clip(upper=gsi_df['gsi'].quantile(0.95))
        ax.hist(data, bins=40, alpha=0.45, label=model, color=MODEL_COLORS.get(model, 'gray'))
    ax.set_xlabel('GSI')
    ax.set_ylabel('Count')
    ax.set_title('(A) GSI Distribution')
    ax.legend(fontsize=5, frameon=False)

    # (B) GSI by dataset
    ax = axes[0, 1]
    if 'dataset' in gsi_df.columns:
        dsets = sorted(gsi_df['dataset'].unique())
        pos = np.arange(len(dsets))
        w = 0.25
        for idx, model in enumerate(['dnabert2', 'nt', 'hyenadna']):
            meds = [gsi_df[(gsi_df['model'] == model) & (gsi_df['dataset'] == ds)]['gsi'].median() for ds in dsets]
            ax.bar(pos + idx * w, meds, w, label=model, color=MODEL_COLORS[model], alpha=0.8)
        ax.set_xticks(pos + w)
        ax.set_xticklabels([d[:5] for d in dsets], fontsize=5, rotation=20)
        ax.set_ylabel('Median GSI')
        ax.set_title('(B) GSI by Dataset')
        ax.legend(fontsize=5, frameon=False)

    # (C) Significance cascade
    ax = axes[1, 0]
    stages = ['v1 (F-test)', 'v2 (z-score)', 'v2 (FDR)']
    rates = [100, 8.3, 0.17]
    colors_c = ['#e74c3c', '#f39c12', '#2ecc71']
    bars = ax.bar(stages, rates, color=colors_c, edgecolor='black', linewidth=0.3, width=0.55)
    for bar, val in zip(bars, rates):
        lbl = f'{val:.1f}%' if val >= 1 else f'{val:.2f}%'
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() * 1.15,
                lbl, ha='center', va='bottom', fontsize=6, fontweight='bold')
    ax.set_ylabel('Significant (%)')
    ax.set_title('(C) Correction Cascade')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 200)
    ax.tick_params(axis='x', labelsize=5)

    # (D) Cross-model agreement
    ax = axes[1, 1]
    agreement = {
        'Agar.': [0.902, 0.702, 0.750], 'Klein': [0.879, 0.657, 0.671],
        'Jores': [0.894, 0.645, 0.695], 'Vaish.': [0.565, -0.030, -0.076],
        'deAlm.': [-0.064, -0.164, 0.050],
    }
    names = list(agreement.keys())
    avgs = [np.mean(v) for v in agreement.values()]
    colors_d = ['#e74c3c', '#9b59b6', '#2ecc71', '#3498db', '#e67e22']
    bars = ax.bar(names, avgs, color=colors_d, edgecolor='black', linewidth=0.3)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2., max(val, 0) + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=5)
    ax.axhline(y=0, color='black', linewidth=0.4)
    ax.set_ylabel('Mean $\\rho$')
    ax.set_title('(D) Cross-Model Agreement')
    ax.set_ylim(-0.2, 1.0)
    ax.tick_params(axis='x', labelsize=5)

    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_gsi_census.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_gsi_census.png'))
    plt.close()
    print("  Fig 3 done")


def fig4_compositionality():
    """Figure 4: Compositionality (single panel + inset text)."""
    comp_df = pd.read_parquet(os.path.join(RESULTS_DIR, 'module3', 'compositionality_results.parquet'))
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.2))

    gap_by_k = comp_df.groupby('n_motifs')['compositionality_gap'].agg(['mean', 'std', 'count']).reset_index()
    ax.errorbar(gap_by_k['n_motifs'], gap_by_k['mean'],
                yerr=gap_by_k['std'] / np.sqrt(gap_by_k['count']),
                fmt='o-', color='steelblue', capsize=3, markersize=4, linewidth=1.2)
    ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.4, linewidth=0.6, label='Regular')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.4, linewidth=0.6, label='Context-free')
    ax.axhline(y=0.99, color='red', linestyle='--', alpha=0.4, linewidth=0.6, label='Context-sensitive')
    ax.set_xlabel('Number of Motifs ($k$)')
    ax.set_ylabel('Compositionality Gap ($1 - R^2$)')
    ax.legend(fontsize=5, frameon=False, loc='center right')
    ax.set_ylim(0, 1.05)
    ax.text(0.03, 0.15, 'Gap = 0.989\n77.5% non-additive\nClassification: Context-sensitive',
            transform=ax.transAxes, fontsize=5, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8, linewidth=0.3))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_compositionality.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_compositionality.png'))
    plt.close()
    print("  Fig 4 done")


def fig5_transfer():
    """Figure 5: Transfer (1x2, compact)."""
    transfer_df = pd.read_parquet(os.path.join(RESULTS_DIR, 'module4', 'transfer_matrix.parquet'))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))
    plt.subplots_adjust(wspace=0.45)

    # (A) Heatmap
    ax = axes[0]
    species = sorted(transfer_df['source'].unique())
    n = len(species)
    matrix = np.zeros((n, n))
    for _, row in transfer_df.iterrows():
        i = species.index(row['source'])
        j = species.index(row['target'])
        matrix[i, j] = row['transfer_r2']
    sns.heatmap(matrix, xticklabels=species, yticklabels=species,
                annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=0.25, ax=ax,
                cbar_kws={'label': '$R^2$', 'shrink': 0.8}, annot_kws={'size': 5})
    ax.set_xlabel('Target', fontsize=6)
    ax.set_ylabel('Source', fontsize=6)
    ax.set_title('(A) Transfer $R^2$')
    ax.tick_params(labelsize=5)

    # (B) Within vs cross
    ax = axes[1]
    cats = ['Within', 'Cross']
    vals = [0.955, 1.888]
    colors_b = ['#2ecc71', '#e74c3c']
    bars = ax.bar(cats, vals, color=colors_b, edgecolor='black', linewidth=0.3, width=0.45)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', va='bottom', fontsize=6, fontweight='bold')
    ax.set_ylabel("Cohen's $d$")
    ax.set_title("(B) Within vs Cross-Species")
    ax.text(0.5, 0.88, '$2\\times$ ratio ($p = 0.035$)', transform=ax.transAxes,
            ha='center', fontsize=5, color='gray', style='italic')

    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_transfer.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_transfer.png'))
    plt.close()
    print("  Fig 5 done")


def fig6_completeness():
    """Figure 6: Completeness (1x2, compact)."""
    datasets = ['agarwal', 'de_almeida', 'vaishnav', 'jores', 'klein']
    labels = ['Agar.', 'deAlm.', 'Vaish.', 'Jores', 'Klein']
    cdata = {}
    for ds in datasets:
        path = os.path.join(RESULTS_DIR, 'module6', f'{ds}_completeness.json')
        if os.path.exists(path):
            with open(path) as f:
                cdata[ds] = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))
    plt.subplots_adjust(wspace=0.4)

    # (A) Hierarchical R^2
    ax = axes[0]
    x = np.arange(len(datasets))
    w = 0.18
    vocab = [cdata[d]['vocabulary_r2'] for d in datasets]
    gram = [cdata[d].get('vocab_plus_full_grammar_r2', cdata[d]['vocabulary_r2']) for d in datasets]
    model = [cdata[d]['full_model_r2'] for d in datasets]
    repl = [cdata[d]['replicate_r2'] for d in datasets]
    ax.bar(x - 1.5*w, vocab, w, label='Vocab', color='#e74c3c', alpha=0.8)
    ax.bar(x - 0.5*w, gram, w, label='+Grammar', color='#f1c40f', alpha=0.8)
    ax.bar(x + 0.5*w, model, w, label='Full Model', color='#2ecc71', alpha=0.8)
    ax.bar(x + 1.5*w, repl, w, label='Replicate', color='#3498db', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel('$R^2$')
    ax.set_title('(A) Hierarchical $R^2$ Decomposition')
    ax.legend(fontsize=4.5, frameon=False, ncol=2)
    ax.set_ylim(0, 1.0)

    # (B) Completeness %
    ax = axes[1]
    pcts = [cdata[d]['grammar_completeness'] * 100 for d in datasets]
    colors_bar = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71', '#9b59b6']
    bars = ax.bar(x, pcts, color=colors_bar, edgecolor='black', linewidth=0.3, width=0.55)
    for bar, val in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel('Completeness (%)')
    ax.set_title('(B) Grammar Completeness')
    ax.set_ylim(0, 25)

    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_completeness.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig6_completeness.png'))
    plt.close()
    print("  Fig 6 done")


if __name__ == '__main__':
    print("Generating compact manuscript figures...")
    fig1_spacer_confound()
    fig2_positive_control()
    fig3_gsi_census()
    fig4_compositionality()
    fig5_transfer()
    fig6_completeness()
    print(f"\nAll figures saved to {FIGURES_DIR}/")
