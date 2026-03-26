"""
Grammar Module: Learn syntactic relationships between motifs.

Uses pairwise attention over motif embeddings to capture:
- Spacing effects (distance between motifs)
- Ordering effects (which motif comes first)
- Orientation effects (strand of each motif)
- Combinatorial effects (which motifs co-occur)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for motif positions.
    Encodes both absolute position and relative distances.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Sinusoidal encoding (fixed)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # Learnable position embedding
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_motifs) integer positions (0-511)

        Returns:
            pos_encoding: (batch, n_motifs, d_model)
        """
        # Clamp positions
        positions = positions.clamp(0, 511)

        # Combine fixed sinusoidal + learnable
        fixed = self.pe[positions]  # (batch, n_motifs, d_model)
        learned = self.pos_embed(positions)  # (batch, n_motifs, d_model)

        return self.dropout(fixed + learned)


class StrandEncoding(nn.Module):
    """Encode motif strand orientation."""

    def __init__(self, d_model: int):
        super().__init__()
        self.strand_embed = nn.Embedding(2, d_model)  # 0 = '+', 1 = '-'

    def forward(self, strands: torch.Tensor) -> torch.Tensor:
        """
        Args:
            strands: (batch, n_motifs) with values 0 or 1

        Returns:
            strand_encoding: (batch, n_motifs, d_model)
        """
        return self.strand_embed(strands)


class PairwiseDistance(nn.Module):
    """
    Compute pairwise distance features between motifs.
    """

    def __init__(self, d_model: int, n_distance_bins: int = 32):
        super().__init__()
        self.n_bins = n_distance_bins
        self.distance_embed = nn.Embedding(n_distance_bins, d_model)

        # Bin edges: log-spaced from 1 to 1000bp
        self.register_buffer(
            'bin_edges',
            torch.logspace(0, 3, n_distance_bins - 1)  # 1 to 1000
        )

    def _discretize_distance(self, distances: torch.Tensor) -> torch.Tensor:
        """Bin continuous distances into discrete buckets."""
        bins = torch.bucketize(distances.abs(), self.bin_edges)
        return bins.clamp(0, self.n_bins - 1)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_motifs) motif start positions

        Returns:
            distance_features: (batch, n_motifs, n_motifs, d_model)
        """
        # Pairwise distances
        pos_i = positions.unsqueeze(-1)  # (batch, n_motifs, 1)
        pos_j = positions.unsqueeze(-2)  # (batch, 1, n_motifs)
        distances = pos_j - pos_i  # (batch, n_motifs, n_motifs)

        # Discretize and embed
        bins = self._discretize_distance(distances)
        return self.distance_embed(bins)


class GrammarAttention(nn.Module):
    """
    Multi-head attention over motifs with grammar-aware biases.

    Incorporates:
    - Pairwise distance information
    - Relative position (before/after)
    - Strand compatibility
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        n_distance_bins: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Distance bias
        self.distance_bias = nn.Embedding(n_distance_bins, n_heads)
        self.register_buffer(
            'bin_edges',
            torch.logspace(0, 3, n_distance_bins - 1)
        )
        self.n_distance_bins = n_distance_bins

        # Order bias (before vs after)
        self.order_bias = nn.Parameter(torch.zeros(n_heads))

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, n_motifs, d_model) motif embeddings
            positions: (batch, n_motifs) motif positions
            mask: (batch, n_motifs) attention mask

        Returns:
            output: (batch, n_motifs, d_model)
        """
        batch_size, n_motifs, _ = x.shape

        # QKV
        q = self.q_proj(x).view(batch_size, n_motifs, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, n_motifs, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, n_motifs, self.n_heads, self.d_head)

        # Attention scores
        q = q.transpose(1, 2)  # (batch, heads, n_motifs, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch, heads, n_motifs, n_motifs)

        # Add distance bias
        pos_i = positions.unsqueeze(-1)
        pos_j = positions.unsqueeze(-2)
        distances = (pos_j - pos_i).abs()
        dist_bins = torch.bucketize(distances, self.bin_edges).clamp(0, self.n_distance_bins - 1)
        dist_bias = self.distance_bias(dist_bins)  # (batch, n_motifs, n_motifs, n_heads)
        dist_bias = dist_bias.permute(0, 3, 1, 2)  # (batch, n_heads, n_motifs, n_motifs)
        scores = scores + dist_bias

        # Add order bias (positive if j comes after i)
        order_sign = (pos_j > pos_i).float() - 0.5  # -0.5 or +0.5
        order_bias = order_sign.unsqueeze(1) * self.order_bias.view(1, -1, 1, 1)
        scores = scores + order_bias

        # Mask
        if mask is not None:
            # Create 2D mask: (batch, 1, n_motifs, n_motifs)
            # mask_2d[b, :, i, j] = 1 if both i and j are valid motifs
            mask_2d = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)
            # Use large negative instead of -inf to avoid NaN in softmax
            scores = scores.masked_fill(mask_2d == 0, -1e9)

        # Softmax and apply
        attn = F.softmax(scores, dim=-1)

        # Handle NaN from all-masked rows (replace with uniform or zero)
        attn = torch.nan_to_num(attn, nan=0.0)

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, heads, n_motifs, d_head)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_motifs, self.d_model)

        # Zero out masked positions in output
        if mask is not None:
            out = out * mask.unsqueeze(-1)

        return self.out_proj(out)


class GrammarModule(nn.Module):
    """
    Full grammar module: encodes motif arrangement into grammar representation.

    Architecture:
    1. Add positional and strand encodings to motif embeddings
    2. Apply multiple layers of grammar attention
    3. Pool to fixed-size grammar vector
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_motifs: int = 50,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional and strand encodings
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=512, dropout=dropout)
        self.strand_encoding = StrandEncoding(hidden_dim)

        # Grammar attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': GrammarAttention(hidden_dim, n_heads, dropout),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                ),
                'norm2': nn.LayerNorm(hidden_dim),
            })
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Pooling: attention-based
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        motif_embeddings: torch.Tensor,
        motif_positions: torch.Tensor,
        motif_strands: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            motif_embeddings: (batch, n_motifs, input_dim)
            motif_positions: (batch, n_motifs) positions in sequence (0-indexed)
            motif_strands: (batch, n_motifs) strand (0='+', 1='-')
            mask: (batch, n_motifs) valid motif mask

        Returns:
            grammar_vector: (batch, output_dim)
        """
        batch_size, n_motifs, _ = motif_embeddings.shape
        device = motif_embeddings.device

        # Handle edge case: no motifs at all
        if n_motifs == 0 or (mask is not None and mask.sum() == 0):
            return torch.zeros(batch_size, self.output_dim, device=device)

        # Project input
        x = self.input_proj(motif_embeddings)

        # Add position and strand encodings
        pos_enc = self.pos_encoding(motif_positions)
        strand_enc = self.strand_encoding(motif_strands)
        x = x + pos_enc + strand_enc

        # Apply grammar attention layers
        for layer in self.layers:
            # Self-attention with residual
            attn_out = layer['attention'](layer['norm1'](x), motif_positions, mask)
            x = x + attn_out

            # FFN with residual
            x = x + layer['ffn'](layer['norm2'](x))

        # Pool to single vector via attention
        query = self.pool_query.expand(batch_size, -1, -1)

        if mask is not None:
            key_padding_mask = ~mask.bool()
        else:
            key_padding_mask = None

        pooled, _ = self.pool_attn(query, x, x, key_padding_mask=key_padding_mask)
        pooled = pooled.squeeze(1)  # (batch, hidden_dim)

        # Handle potential NaN from all-masked sequences
        pooled = torch.nan_to_num(pooled, nan=0.0)

        # Project to output
        grammar_vector = self.output_proj(pooled)

        return grammar_vector
