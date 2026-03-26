"""
Composition Module: Encode sequence composition features.

Captures non-grammar features that foundation models learn:
- GC content (global and local)
- K-mer frequencies (1-mer to 4-mer)
- Dinucleotide composition
- DNA shape proxies (via dinucleotide patterns)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from itertools import product


class KmerEncoder(nn.Module):
    """
    Encode k-mer frequencies as features.
    """

    def __init__(self, k_range: tuple = (1, 4)):
        super().__init__()
        self.k_range = k_range

        # Compute total number of k-mer features
        self.n_features = 0
        self.kmer_indices = {}
        idx = 0
        for k in range(k_range[0], k_range[1] + 1):
            kmers = [''.join(p) for p in product('ACGT', repeat=k)]
            for kmer in kmers:
                self.kmer_indices[kmer] = idx
                idx += 1
            self.n_features += 4 ** k

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """
        Compute k-mer frequency features.

        Args:
            sequences: List of DNA sequences

        Returns:
            features: (batch, n_features) k-mer frequencies
        """
        batch_size = len(sequences)
        features = torch.zeros(batch_size, self.n_features)

        for i, seq in enumerate(sequences):
            seq = seq.upper()
            for k in range(self.k_range[0], self.k_range[1] + 1):
                n_kmers = len(seq) - k + 1
                if n_kmers <= 0:
                    continue
                for j in range(n_kmers):
                    kmer = seq[j:j+k]
                    if kmer in self.kmer_indices:
                        features[i, self.kmer_indices[kmer]] += 1
                # Normalize by count
                start_idx = sum(4**kk for kk in range(self.k_range[0], k))
                end_idx = start_idx + 4**k
                if n_kmers > 0:
                    features[i, start_idx:end_idx] /= n_kmers

        return features


class GCEncoder(nn.Module):
    """
    Encode GC content features (global and windowed).
    """

    def __init__(self, window_sizes: List[int] = [50, 100, 200]):
        super().__init__()
        self.window_sizes = window_sizes
        # Features: global GC, GC variance, GC per window size (mean, std, max, min)
        self.n_features = 2 + len(window_sizes) * 4

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """
        Compute GC content features.

        Args:
            sequences: List of DNA sequences

        Returns:
            features: (batch, n_features)
        """
        batch_size = len(sequences)
        features = torch.zeros(batch_size, self.n_features)

        for i, seq in enumerate(sequences):
            seq = seq.upper()
            gc_count = seq.count('G') + seq.count('C')
            gc_global = gc_count / max(len(seq), 1)

            # Global GC
            features[i, 0] = gc_global

            # Windowed GC statistics
            idx = 1
            for window_size in self.window_sizes:
                gc_windows = []
                step = window_size // 2
                for j in range(0, len(seq) - window_size + 1, step):
                    window = seq[j:j + window_size]
                    wgc = (window.count('G') + window.count('C')) / window_size
                    gc_windows.append(wgc)

                if len(gc_windows) > 0:
                    gc_windows = np.array(gc_windows)
                    features[i, idx] = gc_windows.mean()
                    features[i, idx + 1] = gc_windows.std()
                    features[i, idx + 2] = gc_windows.max()
                    features[i, idx + 3] = gc_windows.min()
                else:
                    features[i, idx] = gc_global
                    features[i, idx + 1] = 0
                    features[i, idx + 2] = gc_global
                    features[i, idx + 3] = gc_global
                idx += 4

            # GC variance (already computed above for first window)
            features[i, -1] = features[i, 2]  # std of smallest window

        return features


class DNAShapeEncoder(nn.Module):
    """
    Encode DNA shape features via dinucleotide proxies.

    Uses established dinucleotide → shape parameter mappings.
    """

    def __init__(self):
        super().__init__()

        # Dinucleotide shape parameters (simplified from literature)
        # Values are approximate averages for each shape feature
        self.shape_params = {
            # (Roll, Twist, Slide, Rise) - simplified values
            'AA': (0.0, 35.6, -0.1, 3.3),
            'AC': (4.0, 34.4, -0.5, 3.4),
            'AG': (4.5, 36.0, -0.2, 3.3),
            'AT': (-5.4, 32.7, -0.2, 3.3),
            'CA': (4.0, 34.5, 0.5, 3.4),
            'CC': (0.5, 33.7, 0.0, 3.4),
            'CG': (1.2, 36.1, 0.4, 3.5),
            'CT': (4.5, 36.0, -0.2, 3.3),
            'GA': (-1.0, 36.9, 0.2, 3.4),
            'GC': (-0.1, 34.4, 0.0, 3.4),
            'GG': (0.5, 33.7, 0.0, 3.4),
            'GT': (4.0, 34.4, -0.5, 3.4),
            'TA': (2.5, 36.0, 0.0, 3.4),
            'TC': (-1.0, 36.9, 0.2, 3.4),
            'TG': (4.0, 34.5, 0.5, 3.4),
            'TT': (0.0, 35.6, -0.1, 3.3),
        }

        # Features: mean and std for each shape parameter (4 params × 2 stats = 8)
        self.n_features = 8

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """
        Compute DNA shape features.

        Args:
            sequences: List of DNA sequences

        Returns:
            features: (batch, n_features)
        """
        batch_size = len(sequences)
        features = torch.zeros(batch_size, self.n_features)

        for i, seq in enumerate(sequences):
            seq = seq.upper()
            shape_values = [[], [], [], []]  # Roll, Twist, Slide, Rise

            for j in range(len(seq) - 1):
                dinuc = seq[j:j+2]
                if dinuc in self.shape_params:
                    params = self.shape_params[dinuc]
                    for k, val in enumerate(params):
                        shape_values[k].append(val)

            # Compute mean and std for each shape parameter
            idx = 0
            for k in range(4):
                if len(shape_values[k]) > 0:
                    vals = np.array(shape_values[k])
                    features[i, idx] = vals.mean()
                    features[i, idx + 1] = vals.std()
                idx += 2

        return features


class CompositionModule(nn.Module):
    """
    Full composition module: encodes sequence composition into representation.

    Combines:
    - K-mer frequencies
    - GC content features
    - DNA shape features
    - Optional: learned embeddings from frozen model

    Projects to output_dim with optional learned refinement.
    """

    def __init__(
        self,
        output_dim: int = 128,
        k_range: tuple = (1, 4),
        window_sizes: List[int] = [50, 100, 200],
        hidden_dim: int = 256,
        use_sequence_embedding: bool = False,
        embedding_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.use_sequence_embedding = use_sequence_embedding

        # Feature encoders
        self.kmer_encoder = KmerEncoder(k_range)
        self.gc_encoder = GCEncoder(window_sizes)
        self.shape_encoder = DNAShapeEncoder()

        # Total input features
        self.n_handcrafted = (
            self.kmer_encoder.n_features +
            self.gc_encoder.n_features +
            self.shape_encoder.n_features
        )

        input_dim = self.n_handcrafted
        if use_sequence_embedding:
            input_dim += embedding_dim

        # Projection network
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        sequences: List[str],
        sequence_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute composition features.

        Args:
            sequences: List of DNA sequences
            sequence_embeddings: Optional (batch, embedding_dim) from foundation model

        Returns:
            composition_vector: (batch, output_dim)
        """
        device = next(self.projector.parameters()).device

        # Handcrafted features
        kmer_features = self.kmer_encoder(sequences).to(device)
        gc_features = self.gc_encoder(sequences).to(device)
        shape_features = self.shape_encoder(sequences).to(device)

        # Concatenate
        features = torch.cat([kmer_features, gc_features, shape_features], dim=1)

        # Add sequence embedding if provided
        if self.use_sequence_embedding and sequence_embeddings is not None:
            features = torch.cat([features, sequence_embeddings], dim=1)

        # Project to output dimension
        composition_vector = self.projector(features)

        return composition_vector

    def get_handcrafted_features(self, sequences: List[str]) -> torch.Tensor:
        """
        Get only handcrafted features (for analysis).
        """
        device = next(self.projector.parameters()).device
        kmer_features = self.kmer_encoder(sequences).to(device)
        gc_features = self.gc_encoder(sequences).to(device)
        shape_features = self.shape_encoder(sequences).to(device)
        return torch.cat([kmer_features, gc_features, shape_features], dim=1)
