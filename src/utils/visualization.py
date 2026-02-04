"""Visualization utilities for GRAMLANG."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from typing import Optional, List, Dict

# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

PALETTE = sns.color_palette("Set2", 10)
MODEL_COLORS = {
    'enformer': PALETTE[0],
    'borzoi': PALETTE[1],
    'sei': PALETTE[2],
    'nt': PALETTE[3],
    'dnabert2': PALETTE[4],
    'hyenadna': PALETTE[5],
    'evo': PALETTE[6],
    'caduceus': PALETTE[7],
    'gpn': PALETTE[8],
}


def plot_gsi_distribution(gsi_df: pd.DataFrame, output_path: str,
                          title: str = "Grammar Sensitivity Index Distribution"):
    """Plot distribution of GSI across enhancers."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (A) Overall distribution
    ax = axes[0]
    for model in sorted(gsi_df['model'].unique()):
        data = gsi_df[gsi_df['model'] == model]['gsi']
        ax.hist(data, bins=50, alpha=0.4, label=model,
                color=MODEL_COLORS.get(model, 'gray'))
    ax.set_xlabel('Grammar Sensitivity Index (GSI)')
    ax.set_ylabel('Count')
    ax.set_title('(A) GSI Distribution by Model')
    ax.legend(fontsize=8, ncol=2)

    # (B) GSI by motif count
    ax = axes[1]
    gsi_by_motifs = gsi_df.groupby(['n_motifs', 'model'])['gsi'].mean().reset_index()
    for model in sorted(gsi_df['model'].unique()):
        subset = gsi_by_motifs[gsi_by_motifs['model'] == model]
        ax.plot(subset['n_motifs'], subset['gsi'], 'o-', label=model,
                color=MODEL_COLORS.get(model, 'gray'), markersize=4, alpha=0.7)
    ax.set_xlabel('Number of Motifs')
    ax.set_ylabel('Mean GSI')
    ax.set_title('(B) GSI vs Motif Count')

    # (C) GSI by dataset
    ax = axes[2]
    if 'dataset' in gsi_df.columns:
        sns.boxplot(data=gsi_df, x='dataset', y='gsi', ax=ax)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('GSI')
        ax.set_title('(C) GSI by Species/Dataset')
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_consensus_heatmap(consensus_df: pd.DataFrame, output_path: str):
    """Plot model-pair consensus heatmap."""
    models = sorted(set(consensus_df['model_1'].unique()) |
                    set(consensus_df['model_2'].unique()))
    n = len(models)
    matrix = np.ones((n, n))

    for _, row in consensus_df.iterrows():
        i = models.index(row['model_1'])
        j = models.index(row['model_2'])
        matrix[i, j] = row.get('mean_spacing_correlation', row.get('cosine_similarity', 0))
        matrix[j, i] = matrix[i, j]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=models, yticklabels=models,
                annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                vmin=-1, vmax=1, ax=ax)
    ax.set_title('Cross-Model Grammar Consensus')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_compositionality_curve(comp_df: pd.DataFrame, output_path: str):
    """Plot compositionality gap vs motif count."""
    fig, ax = plt.subplots(figsize=(8, 6))

    gap_by_k = comp_df.groupby('n_motifs')['compositionality_gap'].agg(
        ['mean', 'std', 'count']
    ).reset_index()

    ax.errorbar(gap_by_k['n_motifs'], gap_by_k['mean'],
                yerr=gap_by_k['std'] / np.sqrt(gap_by_k['count']),
                fmt='o-', color='steelblue', capsize=4, markersize=8)

    ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Regular threshold')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Context-free threshold')

    ax.set_xlabel('Number of Motifs')
    ax.set_ylabel('Compositionality Gap (1 - R²)')
    ax.set_title('Compositionality Gap vs Grammar Order')
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_transfer_heatmap(transfer_df: pd.DataFrame, output_path: str):
    """Plot cross-species grammar transfer matrix."""
    species = sorted(transfer_df['source'].unique())
    n = len(species)
    matrix = np.zeros((n, n))

    for _, row in transfer_df.iterrows():
        i = species.index(row['source'])
        j = species.index(row['target'])
        matrix[i, j] = row['transfer_r2']

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, xticklabels=species, yticklabels=species,
                annot=True, fmt='.3f', cmap='YlOrRd', vmin=0, vmax=1, ax=ax)
    ax.set_xlabel('Target Species')
    ax.set_ylabel('Source Species')
    ax.set_title('Cross-Species Grammar Transfer (R²)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_phase_diagram(phase_data: dict, output_path: str):
    """Plot grammar phase diagram (motif count x density)."""
    fig, ax = plt.subplots(figsize=(8, 7))

    grid = np.array(phase_data['phase_grid'])
    count_edges = phase_data['motif_count_edges']
    density_edges = phase_data['density_edges']

    im = ax.imshow(grid.T, origin='lower', aspect='auto',
                   extent=[count_edges[0], count_edges[-1],
                           density_edges[0], density_edges[-1]],
                   cmap='hot')
    plt.colorbar(im, ax=ax, label='Mean GSI')

    if phase_data.get('critical_motif_count'):
        ax.axvline(x=phase_data['critical_motif_count'], color='cyan',
                   linestyle='--', label=f"Critical count: {phase_data['critical_motif_count']:.0f}")
    if phase_data.get('critical_density'):
        ax.axhline(y=phase_data['critical_density'], color='lime',
                   linestyle='--', label=f"Critical density: {phase_data['critical_density']:.3f}")

    ax.set_xlabel('Motif Count')
    ax.set_ylabel('Motif Density (motifs/bp)')
    ax.set_title('Grammar Phase Diagram')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_completeness_barplot(completeness: dict, output_path: str):
    """Plot the grammar completeness ceiling analysis."""
    levels = ['Vocabulary\nOnly', 'Vocab +\nSimple Grammar',
              'Vocab +\nFull Grammar', 'Full\nModel', 'MPRA\nReplicate']
    values = [
        completeness['vocabulary_r2'],
        completeness['vocab_plus_simple_grammar_r2'],
        completeness['vocab_plus_full_grammar_r2'],
        completeness['full_model_r2'],
        completeness['replicate_r2']
    ]
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(levels, values, color=colors, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('R² (Expression Prediction)')
    ax.set_title('Grammar Completeness: Vocabulary → Grammar → Model')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=completeness['replicate_r2'], color='blue', linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
