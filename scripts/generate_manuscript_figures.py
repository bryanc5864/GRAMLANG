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
    'de_almeida': '#2980b9',# blue
    'jores':   '#27ae60',   # green
    'inoue':   '#2980b9',   # blue
    'klein':   '#8e44ad',   # purple
    'vaishnav':'#d35400',   # orange
}
DS_COLORS_LIST = [DS_PALETTE['agarwal'], DS_PALETTE['jores'], DS_PALETTE['inoue']]

DS_LABELS = {
    'agarwal': 'Agarwal',
    'de_almeida': 'Inoue',
    'inoue': 'Inoue',
    'jores': 'Jores',
    'klein': 'Klein',
    'vaishnav': 'Vaishnav',
}

DS_SHORT_LABELS = {
    'agarwal': 'Agar.',
    'de_almeida': 'Inoue',
    'inoue': 'Inoue',
    'jores': 'Jores',
    'klein': 'Klein',
    'vaishnav': 'Vaish.',
}

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
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
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
        fontsize=8, frameon=False, handlelength=1.2,
        handletextpad=0.4, columnspacing=0.8, labelspacing=0.3,
        borderaxespad=0.3,
    )
    defaults.update(kwargs)
    return ax.legend(**defaults)


def fig1_spacer_confound():
    """Figure 1: Spacer Confound (1x4, compact)."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
    plt.subplots_adjust(wspace=0.45)

    datasets = ['agarwal', 'jores', 'inoue']
    ds_labels = [DS_LABELS[d] for d in datasets]

    # (A) Factorial decomposition
    ax = axes[0]
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
    ax.set_xticklabels(ds_labels, fontsize=9)
    ax.set_ylabel('% of Full Variance')
    ax.set_title('(A) Factorial Decomposition')
    _legend(ax, loc='upper right', bbox_to_anchor=(1.0, -0.15), ncol=3)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)

    # (B) Spacer ablation
    ax = axes[1]
    ablation_data = load_json('v3/spacer_ablation/spacer_ablation_summary.json')
    effects = ['motif_only', 'dinuc_shuffle', 'gc_shift', 'random_replace']
    eff_labels = ['Motif', 'Dinuc', 'GC', 'Random']
    for ds, dl in zip(datasets, ds_labels):
        deltas = [ablation_data[ds]['variant_effects'][e]['median_delta'] for e in effects]
        ax.plot(range(len(effects)), deltas, 'o-', label=dl,
                color=DS_PALETTE[ds], markersize=3)
    ax.set_xticks(range(len(effects)))
    ax.set_xticklabels(eff_labels, fontsize=9)
    ax.set_ylabel('Median |$\\Delta$ Expr|')
    ax.set_title('(B) Spacer Ablation')
    _legend(ax, loc='upper left')

    # (C) Feature decomposition R^2
    ax = axes[2]
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
    ax.set_xticklabels(fc_labels, fontsize=9, rotation=30, ha='right')
    ax.set_ylabel('$R^2$')
    ax.set_title('(C) Feature Prediction of Expression')
    _legend(ax, loc='upper right', bbox_to_anchor=(1.0, -0.15), ncol=3)

    # (D) GC-expression correlation reversal
    ax = axes[3]
    gc_corrs = [feat_data[ds]['correlations']['gc_content']['r'] for ds in datasets]
    colors_bar = [C_NEG if v > 0 else C_ACCENT for v in gc_corrs]
    bars = ax.bar(ds_labels, gc_corrs, color=colors_bar, edgecolor='black',
                  linewidth=0.3, width=0.5)
    for bar, val in zip(bars, gc_corrs):
        y = bar.get_height() + (0.03 if val > 0 else -0.08)
        ax.text(bar.get_x() + bar.get_width() / 2., y, f'{val:+.2f}',
                ha='center', va='bottom' if val > 0 else 'top',
                fontsize=9, fontweight='bold')
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

    # (A) Supported summary statistics from DNABERT-2 positive control
    ax = axes[0]
    stats = ['Median $|\\Delta|$', 'Mean $|\\Delta|$', 'Frac $>0.1$']
    vals = [
        orient['median_abs_diff'],
        orient['mean_abs_diff'],
        orient['frac_diff_gt_0.1'],
    ]
    colors = [C_ACCENT, C_WARN, C_POS]
    bars = ax.bar(stats, vals, color=colors, edgecolor='black',
                  linewidth=0.3, width=0.6)
    for idx, (bar, val) in enumerate(zip(bars, vals)):
        label = f'{val:.3f}'
        if idx == 2:
            label = f'{val * 100:.0f}%'
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('Value')
    ax.set_title('(A) DNABERT-2 Positive Control')
    ax.tick_params(axis='x', labelsize=9)
    ax.set_ylim(0, 0.22)
    ax.text(0.95, 0.95, f'$p = {orient["p_value"]:.0e}$\n$n = {orient["n_pairs"]}$ pairs',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.8, linewidth=0.3))

    # (B) Method comparison (computed from v2 data)
    ax = axes[1]
    methods = ['VP Shuffle\n(confounded)', 'Controlled\nDesign']
    # VP shuffle rate from v2 z-score correction; controlled = positive control detection (p < 0.05)
    _v2_corr = load_json('v2/module1/p_value_correction_summary.json')
    _vp_rate = 100.0 * sum(v['n_sig'] for v in _v2_corr.values()) / (500 * len(_v2_corr))
    # Positive control: all models detect orientation (p << 0.05), so 100% detection
    _pc_all = load_json('v3/validation_analyses/positive_control_all_models.json')
    _pc_detected = sum(1 for v in _pc_all.values() if v['p_value'] < 0.05)
    _ctrl_rate = 100.0 * _pc_detected / len(_pc_all)
    sig = [round(_vp_rate, 1), round(_ctrl_rate, 1)]
    colors_c = [C_NEG, C_POS]
    bars = ax.bar(methods, sig, color=colors_c, edgecolor='black',
                  linewidth=0.3, width=0.5)
    for bar, val in zip(bars, sig):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1.5,
                f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_title('(B) Confounded vs Controlled')
    ax.set_ylim(0, 118)

    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_positive_control.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_positive_control.png'))
    plt.close()
    print("  Fig 2 done")


def fig3_gsi_census():
    """Figure 3: GSI census redesign with cleaner statistical storytelling."""
    gsi_df = pd.read_parquet(os.path.join(RESULTS_DIR, 'module1', 'all_gsi_results.parquet'))
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.3))
    plt.subplots_adjust(wspace=0.45, top=0.82)

    models = ['dnabert2', 'nt', 'hyenadna']
    model_labels = {'dnabert2': 'DNABERT-2', 'nt': 'NT v2', 'hyenadna': 'HyenaDNA'}
    gsi_df = gsi_df.copy()
    gsi_df['dataset_label'] = gsi_df['dataset'].map(DS_SHORT_LABELS)
    gsi_df['model_label'] = gsi_df['model'].map(model_labels)
    dsets = ['agarwal', 'de_almeida', 'jores', 'klein', 'vaishnav']
    dset_labels = [DS_SHORT_LABELS[d] for d in dsets]

    # (A) Distribution by model: violins + median points
    ax = axes[0]
    sns.violinplot(
        data=gsi_df,
        x='model_label',
        y='gsi',
        hue='model_label',
        order=[model_labels[m] for m in models],
        palette=[MODEL_PALETTE[m] for m in models],
        inner=None,
        cut=0,
        linewidth=0.5,
        legend=False,
        ax=ax,
    )
    medians = gsi_df.groupby('model_label')['gsi'].median().reindex([model_labels[m] for m in models])
    ax.scatter(range(len(medians)), medians.values, s=26, color='white', edgecolors='black', zorder=3, linewidths=0.6)
    for idx, val in enumerate(medians.values):
        ax.text(idx, val + 0.004, f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('GSI')
    ax.set_title('(A) Model-Wise GSI Distribution')
    ax.set_ylim(0.02, 0.12)

    # (B) Dataset medians with connected model profiles
    ax = axes[1]
    xpos = np.arange(len(dsets))
    for model in models:
        meds = [gsi_df[(gsi_df['model'] == model) & (gsi_df['dataset'] == ds)]['gsi'].median()
                for ds in dsets]
        ax.plot(xpos, meds, marker='o', color=MODEL_PALETTE[model], label=model_labels[model], linewidth=1.3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(dset_labels, rotation=18)
    ax.set_ylabel('Median GSI')
    ax.set_title('(B) Dataset Profiles')
    ax.set_ylim(0.035, 0.108)
    _legend(ax, loc='upper left')

    # (C) Correction cascade as connected log-scale line (computed from v2 data)
    ax = axes[2]
    stages = ['v1 (F-test)', 'v2 (z-score)', 'v2 (FDR)']
    v2_gsi = pd.read_parquet(os.path.join(RESULTS_DIR, 'v2', 'module1', 'all_gsi_results.parquet'))
    from scipy.stats import norm as _norm
    from statsmodels.stats.multitest import multipletests as _mt
    _n_total = len(v2_gsi)
    _pct_raw = 100.0 * (v2_gsi['p_value'] < 0.05).sum() / _n_total
    _z_pvals = 2 * (1 - _norm.cdf(v2_gsi['z_score'].abs()))
    _pct_zscore = 100.0 * (_z_pvals < 0.05).sum() / _n_total
    _reject, _, _, _ = _mt(_z_pvals, alpha=0.05, method='fdr_bh')
    _pct_fdr = 100.0 * _reject.sum() / _n_total
    rates = [round(_pct_raw, 1), round(_pct_zscore, 1), round(_pct_fdr, 2)]
    x = np.arange(len(stages))
    ax.plot(x, rates, color='#444444', linewidth=1.0, zorder=1)
    ax.scatter(x, rates, s=[90, 70, 56], color=CASCADE_COLORS, edgecolors='black', linewidths=0.4, zorder=2)
    for xi, val in zip(x, rates):
        lbl = f'{val:.1f}%' if val >= 1 else f'{val:.2f}%'
        ax.text(xi + 0.15, val, lbl, ha='left', va='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel('Significant (%)')
    ax.set_title('(C) Correction Cascade')
    ax.set_yscale('log')
    ax.set_ylim(0.05, 300)
    ax.tick_params(axis='x', labelsize=9)

    # (D) Pairwise correlations by dataset, computed from data
    ax = axes[3]
    pair_labels = ['B2 vs NT', 'B2 vs Hyena', 'NT vs Hyena']
    pair_colors = ['#1f4e79', '#6c8ebf', '#9fbad6']
    _model_pairs = [('dnabert2', 'nt'), ('dnabert2', 'hyenadna'), ('nt', 'hyenadna')]
    _ds_order = ['agarwal', 'klein', 'jores', 'vaishnav', 'de_almeida']
    _ds_short = {'agarwal': 'Agar.', 'klein': 'Klein', 'jores': 'Jores',
                 'vaishnav': 'Vaish.', 'de_almeida': 'de Alm.'}
    agreement = {}
    for _ds in _ds_order:
        _ds_data = gsi_df[gsi_df['dataset'] == _ds]
        _pivot = _ds_data.pivot_table(index='seq_id', columns='model', values='gsi')
        _corrs = []
        for _m1, _m2 in _model_pairs:
            if _m1 in _pivot.columns and _m2 in _pivot.columns:
                _r = _pivot[[_m1, _m2]].dropna().corr(method='spearman').iloc[0, 1]
                _corrs.append(round(_r, 3))
            else:
                _corrs.append(0.0)
        agreement[_ds_short[_ds]] = _corrs
    names = list(agreement.keys())
    y = np.arange(len(names))
    offsets = [-0.18, 0.0, 0.18]
    ax.axvline(0, color='#999999', linewidth=0.6, linestyle='--', zorder=0)
    for yi, vals in enumerate(agreement.values()):
        ax.hlines(yi, min(vals), max(vals), color='#d0d0d0', linewidth=1.0, zorder=1)
    for idx, (label, color, offset) in enumerate(zip(pair_labels, pair_colors, offsets)):
        vals = [agreement[name][idx] for name in names]
        ax.scatter(vals, y + offset, s=22, color=color, edgecolors='white', linewidths=0.3, label=label, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel('Pairwise $\\rho$')
    ax.set_title('(D) Agreement Structure')
    ax.set_xlim(-0.25, 0.85)
    _legend(ax, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=7)

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
    ax.axhline(y=0.2, color=C_POS, linestyle='--', alpha=0.4, linewidth=0.6)
    ax.axhline(y=0.5, color=C_WARN, linestyle='--', alpha=0.4, linewidth=0.6)
    ax.axhline(y=0.99, color=C_NEG, linestyle='--', alpha=0.4, linewidth=0.6)
    ax.set_xlabel('Number of Motifs ($k$)')
    ax.set_ylabel('Compositionality Gap ($1 - R^2$)')
    ax.set_ylim(0.97, 1.002)
    ax.text(0.97, 0.70, 'Context-sensitive threshold', transform=ax.transAxes,
            fontsize=9, va='center', ha='right', color=C_NEG)
    ax.text(0.03, 0.08, 'Mean gap = 0.989\n77.5% non-additive',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                      alpha=0.8, linewidth=0.3))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_compositionality.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_compositionality.png'))
    plt.close()
    print("  Fig 4 done")


def fig5_transfer():
    """Figure 5: Transfer heatmap + variance decomposition lollipop."""
    transfer_df = pd.read_parquet(os.path.join(RESULTS_DIR, 'module4', 'transfer_matrix.parquet'))

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2),
                             gridspec_kw={'width_ratios': [1, 1.4]})
    plt.subplots_adjust(wspace=0.5)

    # ── Panel A: Transfer R² heatmap (restyled) ──
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
                cmap='Blues', vmin=0, vmax=0.25, ax=ax, annot=False,
                linewidths=1.0, linecolor='white',
                cbar_kws={'label': '$R^2$', 'shrink': 0.75, 'aspect': 12})

    # Custom annotations: bold diagonal, gray off-diagonal
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if i == j:
                ax.text(j + 0.5, i + 0.5, f'{val:.3f}',
                        ha='center', va='center', fontsize=10,
                        fontweight='bold', color='white' if val > 0.12 else 'black')
                # Highlight diagonal with border
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                             edgecolor='black', linewidth=1.5))
            else:
                ax.text(j + 0.5, i + 0.5, f'{val:.3f}',
                        ha='center', va='center', fontsize=9, color='#888888')

    ax.set_xlabel('Target', fontsize=10)
    ax.set_ylabel('Source', fontsize=10)
    ax.set_title('(A) Cross-Species Transfer $R^2$', fontsize=11)
    ax.tick_params(labelsize=9)

    # ── Panel B: Variance decomposition lollipop ──
    ax = axes[1]
    ds_order = ['agarwal', 'inoue', 'vaishnav', 'jores', 'klein']
    ds_labels = [DS_SHORT_LABELS[d] for d in ds_order]
    cdata = {}
    for ds in ds_order:
        path = os.path.join(RESULTS_DIR, 'module6', f'{ds}_completeness.json')
        if os.path.exists(path):
            with open(path) as f:
                cdata[ds] = json.load(f)

    levels = [
        ('vocabulary_r2',                'Vocab',      C_NEG,    'o',  5),
        ('vocab_plus_full_grammar_r2',   '+Grammar',   C_WARN,   's',  5),
        ('full_model_r2',                'Full Model', C_POS,    'D',  4.5),
        ('replicate_r2',                 'Replicate',  C_ACCENT, 'o',  6),
    ]

    y_pos = np.arange(len(ds_order))

    # Draw connecting lines and gap shading per dataset
    for yi, ds in enumerate(ds_order):
        d = cdata[ds]
        vocab = d['vocabulary_r2']
        repl = d['replicate_r2']
        model = d['full_model_r2']

        # Gray shaded gap: full model → replicate
        ax.fill_betweenx([yi - 0.15, yi + 0.15], model, repl,
                         color='#e0e0e0', alpha=0.5, zorder=0)
        # Connecting line
        ax.plot([vocab, repl], [yi, yi], color='#aaaaaa', linewidth=0.8,
                zorder=1, solid_capstyle='round')

    # Plot markers per level (so legend groups correctly)
    for key, label, color, marker, ms in levels:
        vals = [cdata[ds][key] for ds in ds_order]
        ax.scatter(vals, y_pos, color=color, marker=marker, s=ms**2,
                   label=label, zorder=3, edgecolors='white', linewidths=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ds_labels, fontsize=9)
    ax.set_xlabel('$R^2$', fontsize=7)
    ax.set_title('(B) Variance Decomposition', fontsize=11)
    ax.set_xlim(-0.02, 0.95)
    ax.invert_yaxis()

    # Replicate ceiling line
    ax.axvline(x=0.85, color=C_ACCENT, linewidth=0.6, linestyle=':', alpha=0.5)

    _legend(ax, loc='upper center', bbox_to_anchor=(0.5, -0.12),
            fontsize=9, ncol=2, markerscale=0.9)

    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_transfer.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_transfer.png'))
    plt.close()
    print("  Fig 5 done")


def fig6_completeness():
    """Figure 6: Completeness (1x2, compact)."""
    datasets = ['agarwal', 'inoue', 'vaishnav', 'jores', 'klein']
    labels = [DS_SHORT_LABELS[d] for d in datasets]
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
    _legend(ax, loc='upper center', bbox_to_anchor=(0.5, 1.01), fontsize=8, ncol=2)
    ax.set_ylim(0, 1.0)

    # (B) Completeness %
    ax = axes[1]
    pcts = [cdata[d]['grammar_completeness'] * 100 for d in datasets]
    colors_bar = [DS_PALETTE[d] for d in datasets]
    bars = ax.bar(x, pcts, color=colors_bar, edgecolor='black',
                  linewidth=0.3, width=0.55)
    for bar, val in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
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
