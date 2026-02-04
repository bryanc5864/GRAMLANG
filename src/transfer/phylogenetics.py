"""Grammar phylogenetics: build grammar similarity trees."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def build_grammar_phylogeny(transfer_df: pd.DataFrame, species_list: list) -> dict:
    """
    Build a grammar similarity tree from the transfer matrix.

    Args:
        transfer_df: Transfer matrix from compute_transfer_matrix
        species_list: List of species names

    Returns:
        Dict with distance matrix, linkage matrix
    """
    n = len(species_list)
    dist_matrix = np.zeros((n, n))

    for i, sp_i in enumerate(species_list):
        for j, sp_j in enumerate(species_list):
            if i == j:
                dist_matrix[i, j] = 0
            else:
                ij = transfer_df[
                    (transfer_df['source'] == sp_i) &
                    (transfer_df['target'] == sp_j)
                ]['transfer_r2'].values
                ji = transfer_df[
                    (transfer_df['source'] == sp_j) &
                    (transfer_df['target'] == sp_i)
                ]['transfer_r2'].values

                mean_transfer = 0
                if len(ij) > 0 and len(ji) > 0:
                    mean_transfer = (ij[0] + ji[0]) / 2

                dist_matrix[i, j] = 1 - mean_transfer

    # Symmetrize
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(dist_matrix, 0)

    # Ensure valid distance matrix
    dist_matrix = np.clip(dist_matrix, 0, 2)

    # UPGMA clustering
    condensed = squareform(dist_matrix)
    linkage_matrix = linkage(condensed, method='average')

    return {
        'distance_matrix': dist_matrix.tolist(),
        'linkage_matrix': linkage_matrix.tolist(),
        'species_list': species_list,
    }
