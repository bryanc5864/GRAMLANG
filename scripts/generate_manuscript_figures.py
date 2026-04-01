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

# --- Unified color palette ---
# Dataset colors (consistent across all figures)
DS_PALETTE = {
    'agarwal': '#c0392b',   # red
    'jores':   '#27ae60',   # green
    'inoue':   '#2980b9',   # blue
    'klein':   '#8e44ad',   # purple
    'vaishnav':'#d35400',   # orange
}
DS_COLORS_LIST = [DS_PALETTE['agarwal'], DS_PALETTE['jores'], DS_PALETTE['inoue']]

# Model colors (consistent across all figures)
MODEL_PALETTE = {
    'dnabert2':  '#2980b9',  # blue
    'nt':        '#c0392b',  # red
    'hyenadna':  '#27ae60',  # green
}

# Semantic colors
C_POS    = '#27ae60'  # positive / good / within
C_NEG    = '#c0392b'  # negative / bad / cross
C_WARN   = '#e67e22'  # warning / moderate
C_ACCENT = '#2980b9'  # accent / neutral emphasis

# Component colors for factorial decomposition
COMP_COLORS = {
    'position':    '#2980b9',  # blue
    'orientation': '#e67e22',  # orange
    'spacer':      '#c0392b',  # red
}

# Correction cascade
CASCADE_COLORS = ['#c0392b', '#e67e22', '#27ae60']  # red, orange, green

# ISMB / Bioinformatics style: Times New Roman
plt.rcParams.update({
    'font.size': 7,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 7,
    'axes.titlesize': 8,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5.5,
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


def load_json(path):
    with open(os.path.join(RESULTS_DIR, path)) as f:
        return json.load(f)


def _legend(ax, **kwargs):
    """Compact legend with consistent spacing."""
    defaults = dict(
        fontsize=5.5, frameon=False, handlelength=1.2,
        handletextpad=0.4, columnspacing=0.8, labelspacing=0.3,
        borderaxespad=0.3,
    )
    defaults.update(kwargs)
    return ax.legend(**defaults)


def fig1_spacer_confound():
    """Figure 1: Spacer Confound (2x2, compact)."""
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.8))
    plt.subplots_adjust(hspace=0.55, wspace=0.4)

    datasets = ['agarwal', 'jores', 'inoue']
    ds_labels = ['Agarwal', 'Jores', 'Inoue']

    # (A) Factorial decomposition
    ax = axes[0, 0]
    components = ['position', 'orientation', 'spacer']
    comp_labels = ['Position', 'Orientation', 'Spacer']
    x = np.arange(len(datasets))
    width = 0.22
    for i, (comp, label) in enumerate(zip(components, comp_labels)):
        fracs = []
        for ds in datasets:
            data = load_json(f'v3/factorial_decomposition/{ds}_dnabert2_factorial_summary.json')
            fracs.append(data['effect_sizes'][comp]['median_fraction_of_full'] * 100)
        ax.bar(x + i * width, fracs, width, label=label, color=COMP_COLORS[comp],
               alpha=0.85, edgecolor='black', linewidth=0.2)
    ax.set_xticks(x + width)
    ax.set_xticklabels(ds_labels, fontsize=6)
    ax.set_ylabel('% of Full Variance')
    ax.set_title('(A) Factorial Decomposition')
    _legend(ax, loc='upper right')
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)

    # (B) Spacer ablation
    ax = axes[0, 1]
    ablation_data = load_json('v3/spacer_ablation/spacer_ablation_summary.json')
    effects = ['motif_only', 'dinuc_shuffle', 'gc_shift', 'random_replace']
    eff_labels = ['Motif', 'Dinuc', 'GC', 'Random']
    for ds, dl in zip(datasets, ds_labels):
        deltas = [ablation_data[ds]['variant_effects'][e]['median_delta'] for e in effects]
        ax.plot(range(len(effects)), deltas, 'o-', label=dl,
                color=DS_PALETTE[ds], markersize=3)
    ax.set_xticks(range(len(effects)))
    ax.set_xticklabels(eff_labels, fontsize=6)
    ax.set_ylabel('Median |$\\Delta$ Expr|')
    ax.set_title('(B) Spacer Ablation')
    _legend(ax)

    # (C) Feature decomposition R^2
    ax = axes[1, 0]
    feat_data = load_json('v3/feature_decomposition/feature_decomposition_summary.json')
    fcs = ['gc_only', 'dinuc_only', 'shape_only', 'trinuc_only', 'all_features']
    fc_labels = ['GC', 'Dinuc', 'Shape', 'Trinuc', 'All']
    x = np.arange(len(fcs))
    width = 0.25
    for ds, dl in zip(datasets, ds_labels):
        r2s = [feat_data[ds][fc]['mean_r2'] for fc in fcs]
        ax.bar(x + datasets.index(ds) * width, r2s, width, label=dl,
               color=DS_PALETTE[ds], alpha=0.85, edgecolor='black', linewidth=0.2)
    ax.set_xticks(x + width)
    ax.set_xticklabels(fc_labels, fontsize=6)
    ax.set_ylabel('$R^2$')
    ax.set_title('(C) Feature Prediction of Expression')
    _legend(ax)

    # (D) GC-expression correlation reversal
    ax = axes[1, 1]
    gc_corrs = [feat_data[ds]['correlations']['gc_content']['r'] for ds in datasets]
    colors_bar = [C_NEG if v > 0 else C_ACCENT for v in gc_corrs]
    bars = ax.bar(ds_labels, gc_corrs, color=colors_bar, edgecolor='black',
                  linewidth=0.3, width=0.5)
    for bar, val in zip(bars, gc_corrs):
        y = bar.get_height() + (0.03 if val > 0 else -0.08)
        ax.text(bar.get_x() + bar.get_width() / 2., y, f'{val:+.2f}',
                ha='center', va='bottom' if val > 0 else 'top',
                fontsize=6, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.4)
    ax.set_ylabel('GC\u2013Expr Correlation ($r$)')
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
    colors = [C_POS, '#229954', C_WARN, '#d35400', C_NEG]
    bars = ax.bar(thresholds, pcts, color=colors, edgecolor='black',
                  linewidth=0.3, width=0.6)
    ax.set_ylabel('% of Pairs')
    ax.set_xlabel('$|\\Delta|$ threshold')
    ax.set_title('(A) Effect Size Distribution')
    ax.text(0.95, 0.95, f'$p = {orient["p_value"]:.0e}$\n$n = {orient["n_pairs"]}$ pairs',
            transform=ax.transAxes, ha='right', va='top', fontsize=5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.8, linewidth=0.3))

    # (B) Method comparison
    ax = axes[1]
    methods = ['VP Shuffle\n(confounded)', 'Controlled\nDesign']
    sig = [8.3, 100]
    colors_c = [C_NEG, C_POS]
    bars = ax.bar(methods, sig, color=colors_c, edgecolor='black',
                  linewidth=0.3, width=0.5)
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

    models = ['dnabert2', 'nt', 'hyenadna']
    model_labels = {'dnabert2': 'DNABERT-2', 'nt': 'NT v2', 'hyenadna': 'HyenaDNA'}

    # (A) GSI distribution by model
    ax = axes[0, 0]
    for model in models:
        data = gsi_df[gsi_df['model'] == model]['gsi'].clip(upper=gsi_df['gsi'].quantile(0.95))
        ax.hist(data, bins=40, alpha=0.45, label=model_labels[model],
                color=MODEL_PALETTE[model])
    ax.set_xlabel('GSI')
    ax.set_ylabel('Count')
    ax.set_title('(A) GSI Distribution')
    _legend(ax)

    # (B) GSI by dataset
    ax = axes[0, 1]
    if 'dataset' in gsi_df.columns:
        dsets = sorted(gsi_df['dataset'].unique())
        pos = np.arange(len(dsets))
        w = 0.25
        for idx, model in enumerate(models):
            meds = [gsi_df[(gsi_df['model'] == model) & (gsi_df['dataset'] == ds)]['gsi'].median()
                    for ds in dsets]
            ax.bar(pos + idx * w, meds, w, label=model_labels[model],
                   color=MODEL_PALETTE[model], alpha=0.8)
        ax.set_xticks(pos + w)
        ax.set_xticklabels([d.capitalize()[:5] + '.' for d in dsets],
                           fontsize=5, rotation=20)
        ax.set_ylabel('Median GSI')
        ax.set_title('(B) GSI by Dataset')
        _legend(ax)

    # (C) Significance cascade
    ax = axes[1, 0]
    stages = ['v1 (F-test)', 'v2 (z-score)', 'v2 (FDR)']
    rates = [100, 8.3, 0.17]
    bars = ax.bar(stages, rates, color=CASCADE_COLORS, edgecolor='black',
                  linewidth=0.3, width=0.55)
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
    # Map to unified dataset palette
    ds_key_map = {'Agar.': 'agarwal', 'Klein': 'klein', 'Jores': 'jores',
                  'Vaish.': 'vaishnav', 'deAlm.': 'inoue'}
    names = list(agreement.keys())
    avgs = [np.mean(v) for v in agreement.values()]
    colors_d = [DS_PALETTE[ds_key_map[n]] for n in names]
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
                fmt='o-', color=C_ACCENT, capsize=3, markersize=4, linewidth=1.2)
    ax.axhline(y=0.2, color=C_POS, linestyle='--', alpha=0.4, linewidth=0.6, label='Regular')
    ax.axhline(y=0.5, color=C_WARN, linestyle='--', alpha=0.4, linewidth=0.6, label='Context-free')
    ax.axhline(y=0.99, color=C_NEG, linestyle='--', alpha=0.4, linewidth=0.6, label='Context-sensitive')
    ax.set_xlabel('Number of Motifs ($k$)')
    ax.set_ylabel('Compositionality Gap ($1 - R^2$)')
    _legend(ax, loc='center right')
    ax.set_ylim(0, 1.05)
    ax.text(0.03, 0.15, 'Gap = 0.989\n77.5% non-additive\nClassification: Context-sensitive',
            transform=ax.transAxes, fontsize=5, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      alpha=0.8, linewidth=0.3))

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
    species_labels = [s.capitalize() for s in species]
    sns.heatmap(matrix, xticklabels=species_labels, yticklabels=species_labels,
                annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=0.25, ax=ax,
                cbar_kws={'label': '$R^2$', 'shrink': 0.8}, annot_kws={'size': 5})
    ax.set_xlabel('Target', fontsize=6)
    ax.set_ylabel('Source', fontsize=6)
    ax.set_title('(A) Transfer $R^2$')
    ax.tick_params(labelsize=5)

    # (B) Within vs cross-species
    ax = axes[1]
    cats = ['Within-Species', 'Cross-Species']
    vals = [0.955, 1.888]
    colors_b = [C_POS, C_NEG]
    bars = ax.bar(cats, vals, color=colors_b, edgecolor='black',
                  linewidth=0.3, width=0.45)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', va='bottom', fontsize=6, fontweight='bold')
    ax.set_ylabel("Cohen's $d$")
    ax.set_title("(B) Within vs Cross-Species")
    ax.text(0.5, 0.88, '$2\\times$ ratio ($p = 0.035$)', transform=ax.transAxes,
            ha='center', fontsize=5, color='#555555', style='italic')
    ax.tick_params(axis='x', labelsize=5.5)

    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_transfer.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_transfer.png'))
    plt.close()
    print("  Fig 5 done")


def fig6_completeness():
    """Figure 6: Completeness (1x2, compact)."""
    datasets = ['agarwal', 'inoue', 'vaishnav', 'jores', 'klein']
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
    ax.bar(x - 1.5*w, vocab, w, label='Vocab', color=C_NEG, alpha=0.8)
    ax.bar(x - 0.5*w, gram, w, label='+Grammar', color='#f1c40f', alpha=0.8)
    ax.bar(x + 0.5*w, model, w, label='Full Model', color=C_POS, alpha=0.8)
    ax.bar(x + 1.5*w, repl, w, label='Replicate', color=C_ACCENT, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel('$R^2$')
    ax.set_title('(A) Hierarchical $R^2$ Decomposition')
    _legend(ax, fontsize=4.5, ncol=2)
    ax.set_ylim(0, 1.0)

    # (B) Completeness %
    ax = axes[1]
    pcts = [cdata[d]['grammar_completeness'] * 100 for d in datasets]
    colors_bar = [DS_PALETTE[d] for d in datasets]
    bars = ax.bar(x, pcts, color=colors_bar, edgecolor='black',
                  linewidth=0.3, width=0.55)
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
