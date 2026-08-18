"""Readout heads over frozen token-level DNA-LM embeddings.

The submitted paper had one readout: ridge on mean-pooled embeddings. Mean
pooling is permutation-invariant, so a null could be blamed on the readout
rather than the representation. Reviewers bg4P (Q1), asH5 and SvHh (Q3) all
asked for the same disambiguation, hence a ladder of heads differing only in
how much arrangement they can express:

    mean_linear   mean-pool -> linear            (baseline; perm-invariant)
    mean_mlp      mean-pool -> 2-layer MLP       (perm-invariant)
    attn_pos      pos emb -> attention pooling -> linear   (order-aware)
    cnn1d         dilated 1D-CNN -> max+mean pool          (order-aware)
    transformer   pos emb -> 2-layer encoder -> CLS        (order-aware)

order-aware heads recovering what mean_linear misses would mean the claim is
about the readout, not the representation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _masked_mean(h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe)

    def forward(self, h):
        return h + self.pe[: h.shape[1]].unsqueeze(0)


class MeanLinear(nn.Module):
    """Submitted baseline: mean pool then linear (ridge = L2 weight decay)."""

    perm_invariant = True

    def __init__(self, d_in: int, **kw):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.fc = nn.Linear(d_in, 1)

    def forward(self, h, m):
        return self.fc(self.norm(_masked_mean(h, m)))


class MeanMLP(nn.Module):
    perm_invariant = True

    def __init__(self, d_in: int, d_hidden: int = 256, dropout: float = 0.1, **kw):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, h, m):
        return self.net(self.norm(_masked_mean(h, m)))


class AttnPos(nn.Module):
    """Positional embeddings + single-query attention pooling.

    the positional encoding is what makes this order-aware; content-only
    attention pooling is still permutation-invariant."""

    perm_invariant = False

    def __init__(self, d_in: int, d_hidden: int = 256, dropout: float = 0.1, **kw):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_hidden)
        self.pos = PositionalEncoding(d_hidden)
        self.score = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.Tanh(), nn.Linear(d_hidden, 1)
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_hidden, 1)

    def forward(self, h, m):
        x = self.pos(self.proj(self.norm(h)))
        s = self.score(x).squeeze(-1)
        s = s.masked_fill(m < 0.5, float("-inf"))
        a = torch.softmax(s, dim=1).unsqueeze(-1)
        return self.fc(self.drop((x * a).sum(1)))


class CNN1D(nn.Module):
    """Dilated 1D convs over tokens, then max+mean pooling."""

    perm_invariant = False

    def __init__(self, d_in: int, d_hidden: int = 192, dropout: float = 0.1, **kw):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Conv1d(d_in, d_hidden, 1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_hidden, d_hidden, 5, padding=2 * d, dilation=d),
                    nn.BatchNorm1d(d_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for d in (1, 2, 4)
            ]
        )
        self.fc = nn.Sequential(nn.Linear(2 * d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))

    def forward(self, h, m):
        x = self.proj(self.norm(h).transpose(1, 2))
        for b in self.blocks:
            x = x + b(x)
        mm = m.unsqueeze(1)
        x = x * mm
        mx = x.masked_fill(mm < 0.5, float("-inf")).max(dim=2).values
        mx = torch.nan_to_num(mx, neginf=0.0)
        av = x.sum(2) / mm.sum(2).clamp(min=1.0)
        return self.fc(torch.cat([mx, av], dim=1))


class ShallowTransformer(nn.Module):
    perm_invariant = False

    def __init__(self, d_in: int, d_hidden: int = 192, dropout: float = 0.1, n_layers: int = 2, **kw):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Linear(d_in, d_hidden)
        self.pos = PositionalEncoding(d_hidden)
        self.cls = nn.Parameter(torch.randn(1, 1, d_hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_hidden, nhead=4, dim_feedforward=2 * d_hidden,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.fc = nn.Linear(d_hidden, 1)

    def forward(self, h, m):
        x = self.pos(self.proj(self.norm(h)))
        B = x.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)
        pad = torch.cat([torch.ones(B, 1, device=m.device), m], dim=1) < 0.5
        x = self.enc(x, src_key_padding_mask=pad)
        return self.fc(x[:, 0])


HEADS = {
    "mean_linear": MeanLinear,
    "mean_mlp": MeanMLP,
    "attn_pos": AttnPos,
    "cnn1d": CNN1D,
    "transformer": ShallowTransformer,
}

ORDER_AWARE = ["attn_pos", "cnn1d", "transformer"]


def build_head(kind: str, d_in: int, **kw) -> nn.Module:
    return HEADS[kind](d_in, **kw)
