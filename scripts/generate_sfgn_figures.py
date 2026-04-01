#!/usr/bin/env python3
"""
Generate publication-quality figures for SFGN NeurIPS submission.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = Path('results/sfgn/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ['agarwal', 'jores', 'klein', 'vaishnav']
DATASET_LABELS = {
    'agarwal': 'Agarwal (K562)',
    'jores': 'Jores (Plant)',
    'klein': 'Klein (mESC)',
    'vaishnav': 'Vaishnav (Yeast)'
}


def load_training_history():
    """Load training history for all datasets."""
    histories = {}
    for dataset in DATASETS:
        path = Path(f'results/sfgn/{dataset}_sfgn_history.json')
        if path.exists():
            with open(path) as f:
                histories[dataset] = json.load(f)
    return histories


def load_metrics():
    """Load final metrics for all datasets."""
    metrics = {}
    for dataset in DATASETS:
        path = Path(f'results/sfgn/{dataset}_sfgn_metrics.json')
        if path.exists():
            with open(path) as f:
                metrics[dataset] = json.load(f)
    return metrics


def load_sf_gsi_results():
    """Load SF-GSI results."""
    results = {}
    for dataset in DATASETS:
        path = Path(f'results/sf_gsi/{dataset}_sf_gsi.json')
        if path.exists():
            with open(path) as f:
                results[dataset] = json.load(f)
    return results


def fig1_alpha_trajectory(histories):
    """Figure 1: Grammar weight (α) trajectory during training."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for i, (dataset, history) in enumerate(histories.items()):
        epochs = [h['epoch'] for h in history]
        alphas = [h['mean_alpha'] for h in history]
        ax.plot(epochs, alphas, 'o-', label=DATASET_LABELS.get(dataset, dataset),
                color=colors[i], linewidth=2, markersize=4)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Grammar Weight (α)')
    ax.set_title('SFGN Learns to Increase Grammar Reliance')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal weighting')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_alpha_trajectory.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig1_alpha_trajectory.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig1_alpha_trajectory")


def fig2_final_alpha_vs_r2(metrics):
    """Figure 2: Final α vs R² scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 5))

    datasets = []
    alphas = []
    r2s = []

    for dataset, m in metrics.items():
        datasets.append(DATASET_LABELS.get(dataset, dataset))
        alphas.append(m['final_metrics']['mean_alpha'])
        r2s.append(m['final_metrics']['r2'])

    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    for i, (d, a, r) in enumerate(zip(datasets, alphas, r2s)):
        ax.scatter(a, r, s=150, c=[colors[i]], label=d, edgecolors='black', linewidth=1)

    ax.set_xlabel('Final Grammar Weight (α)')
    ax.set_ylabel('Validation R²')
    ax.set_title('High Grammar Weight ≠ Better Prediction')
    ax.legend(loc='best')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    # Add annotation
    ax.annotate('Higher α, but\nlower/similar R²',
                xy=(0.85, -0.15), fontsize=10, style='italic',
                ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_alpha_vs_r2.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig2_alpha_vs_r2.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig2_alpha_vs_r2")


def fig3_sf_gsi_decomposition(sf_gsi_results):
    """Figure 3: SF-GSI decomposition showing spacer contribution."""
    fig, ax = plt.subplots(figsize=(9, 5))

    datasets = []
    gsi_values = []
    sf_gsi_values = []
    spacer_contrib = []

    for dataset in DATASETS:
        if dataset in sf_gsi_results:
            res = sf_gsi_results[dataset]['results']
            # Use regression method
            if 'regression' in res and 'error' not in res['regression']:
                datasets.append(DATASET_LABELS.get(dataset, dataset))
                gsi_values.append(res['regression']['gsi'])
                sf_gsi_values.append(res['regression']['sf_gsi'])
                spacer_contrib.append(res['regression']['spacer_contribution'] * 100)

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, gsi_values, width, label='GSI (Total)', color='steelblue')
    bars2 = ax.bar(x + width/2, sf_gsi_values, width, label='SF-GSI (Grammar only)', color='coral')

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Sensitivity Index')
    ax.set_title('Spacer-Factored GSI: Most Sensitivity is Spacer-Driven')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend()

    # Add spacer contribution annotations
    for i, (d, sc) in enumerate(zip(datasets, spacer_contrib)):
        ax.annotate(f'{sc:.0f}% spacer', xy=(i, max(gsi_values[i], sf_gsi_values[i]) + 0.05),
                    ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_sf_gsi_decomposition.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig3_sf_gsi_decomposition.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig3_sf_gsi_decomposition")


def fig4_training_loss(histories):
    """Figure 4: Training loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    # Left: Training MSE
    ax = axes[0]
    for i, (dataset, history) in enumerate(histories.items()):
        epochs = [h['epoch'] for h in history]
        mse = [h['train_mse'] for h in history]
        ax.plot(epochs, mse, 'o-', label=DATASET_LABELS.get(dataset, dataset),
                color=colors[i], linewidth=2, markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training MSE')
    ax.set_title('Training Loss Decreases')
    ax.legend(loc='upper right')

    # Right: Validation R²
    ax = axes[1]
    for i, (dataset, history) in enumerate(histories.items()):
        epochs = [h['epoch'] for h in history]
        r2 = [h['val_r2'] for h in history]
        ax.plot(epochs, r2, 'o-', label=DATASET_LABELS.get(dataset, dataset),
                color=colors[i], linewidth=2, markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation R²')
    ax.set_title('Validation R² Remains Low')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_training_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig4_training_curves.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig4_training_curves")


def fig5_summary_table(metrics, sf_gsi_results):
    """Figure 5: Summary table as a figure."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    # Prepare data
    rows = []
    for dataset in DATASETS:
        row = [DATASET_LABELS.get(dataset, dataset)]

        if dataset in metrics:
            m = metrics[dataset]['final_metrics']
            row.extend([
                f"{m['mean_alpha']:.2f}",
                f"{m['mean_beta']:.2f}",
                f"{m['r2']:.3f}",
                f"{m['pearson_r']:.3f}"
            ])
        else:
            row.extend(['-', '-', '-', '-'])

        if dataset in sf_gsi_results:
            res = sf_gsi_results[dataset]['results']
            if 'regression' in res and 'error' not in res['regression']:
                row.append(f"{res['regression']['spacer_contribution']*100:.0f}%")
            else:
                row.append('-')
        else:
            row.append('-')

        rows.append(row)

    columns = ['Dataset', 'α (Grammar)', 'β (Comp.)', 'Val R²', 'Pearson r', 'Spacer %']

    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D6DCE4')

    ax.set_title('SFGN Results Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_summary_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig5_summary_table.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig5_summary_table")


def fig6_billboard_model(metrics):
    """Figure 6: Visual explanation of billboard model finding."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create conceptual diagram
    ax.text(0.5, 0.95, 'The Billboard Model of Regulatory DNA',
            ha='center', va='top', fontsize=14, fontweight='bold',
            transform=ax.transAxes)

    ax.text(0.5, 0.85, 'SFGN learns high grammar weights (α → 0.7-1.0)',
            ha='center', va='top', fontsize=11, transform=ax.transAxes)
    ax.text(0.5, 0.78, 'BUT validation R² stays near zero',
            ha='center', va='top', fontsize=11, transform=ax.transAxes)

    ax.text(0.5, 0.65, '↓', ha='center', va='top', fontsize=16, transform=ax.transAxes)

    ax.text(0.5, 0.55, 'Interpretation:', ha='center', va='top',
            fontsize=12, fontweight='bold', transform=ax.transAxes)

    # Two columns
    ax.text(0.25, 0.42, 'Grammar is learnable\n(motif positions matter\nduring training)',
            ha='center', va='top', fontsize=10, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    ax.text(0.75, 0.42, 'Grammar is not predictive\n(doesn\'t generalize to\nnew sequences)',
            ha='center', va='top', fontsize=10, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    ax.text(0.5, 0.18, '↓', ha='center', va='top', fontsize=16, transform=ax.transAxes)

    ax.text(0.5, 0.08, '"Billboard Model": Motif identity (vocabulary) matters,\nbut arrangement (grammar) adds noise rather than signal',
            ha='center', va='top', fontsize=11, style='italic', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_billboard_model.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig6_billboard_model.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig6_billboard_model")


def main():
    print("Loading data...")
    histories = load_training_history()
    metrics = load_metrics()
    sf_gsi_results = load_sf_gsi_results()

    print(f"Found histories for: {list(histories.keys())}")
    print(f"Found metrics for: {list(metrics.keys())}")
    print(f"Found SF-GSI for: {list(sf_gsi_results.keys())}")

    print("\nGenerating figures...")

    if histories:
        fig1_alpha_trajectory(histories)
        fig4_training_loss(histories)

    if metrics:
        fig2_final_alpha_vs_r2(metrics)

    if sf_gsi_results:
        fig3_sf_gsi_decomposition(sf_gsi_results)

    if metrics or sf_gsi_results:
        fig5_summary_table(metrics, sf_gsi_results)

    if metrics:
        fig6_billboard_model(metrics)

    print(f"\nFigures saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
