"""Grammar phase diagram: when does grammar "turn on"?"""

import numpy as np
import pandas as pd


def compute_grammar_phase_diagram(
    gsi_results: pd.DataFrame,
    dataset: pd.DataFrame,
    bins_count: int = 8,
    bins_density: int = 8,
) -> dict:
    """
    Map the 2D phase diagram of grammar:
    (motif_count, motif_density) -> mean GSI.

    Returns:
        Dict with phase grid and critical thresholds
    """
    # Use n_motifs/motif_density from GSI results if available, else merge from dataset
    if 'n_motifs' in gsi_results.columns and 'motif_density' in gsi_results.columns:
        merged = gsi_results
    else:
        merged = gsi_results.merge(
            dataset[['seq_id', 'n_motifs', 'motif_density']],
            on='seq_id', how='left'
        )

    # Average GSI across models
    enhancer_gsi = merged.groupby('seq_id').agg(
        mean_gsi=('gsi', 'mean'),
        n_motifs=('n_motifs', 'first'),
        motif_density=('motif_density', 'first')
    ).reset_index().dropna()

    if len(enhancer_gsi) < 10:
        return {'error': 'Too few data points'}

    count_edges = np.linspace(
        enhancer_gsi['n_motifs'].min(),
        enhancer_gsi['n_motifs'].max() + 1,
        bins_count + 1
    )
    density_edges = np.linspace(
        enhancer_gsi['motif_density'].min(),
        enhancer_gsi['motif_density'].max() + 1e-6,
        bins_density + 1
    )

    phase_grid = np.zeros((bins_count, bins_density))
    count_grid = np.zeros((bins_count, bins_density))

    for i in range(bins_count):
        for j in range(bins_density):
            mask = (
                (enhancer_gsi['n_motifs'] >= count_edges[i]) &
                (enhancer_gsi['n_motifs'] < count_edges[i+1]) &
                (enhancer_gsi['motif_density'] >= density_edges[j]) &
                (enhancer_gsi['motif_density'] < density_edges[j+1])
            )
            if mask.sum() > 0:
                phase_grid[i, j] = enhancer_gsi.loc[mask, 'mean_gsi'].mean()
                count_grid[i, j] = mask.sum()

    # Find critical thresholds
    critical_count = None
    for i in range(bins_count):
        if np.max(phase_grid[i, :]) > 0.1:
            critical_count = float((count_edges[i] + count_edges[i+1]) / 2)
            break

    critical_density = None
    for j in range(bins_density):
        if np.max(phase_grid[:, j]) > 0.1:
            critical_density = float((density_edges[j] + density_edges[j+1]) / 2)
            break

    return {
        'phase_grid': phase_grid.tolist(),
        'count_grid': count_grid.tolist(),
        'motif_count_edges': count_edges.tolist(),
        'density_edges': density_edges.tolist(),
        'critical_motif_count': critical_count,
        'critical_density': critical_density,
        'n_total_enhancers': len(enhancer_gsi),
    }
