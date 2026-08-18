"""Spacer-Factored Grammar Networks (SFGN).

Separates grammar from composition when predicting expression from regulatory
DNA. A frozen foundation model gives motif-level embeddings; the grammar module
reads their arrangement, the composition module reads GC/k-mer/shape content,
and the fusion head combines them as y = alpha*f(g) + beta*f(c), with an
orthogonality penalty pushing the two vectors apart.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass

from .motif_encoder import MotifEncoder, SequenceEncoder
from .grammar_module import GrammarModule
from .composition_module import CompositionModule


@dataclass
class SFGNConfig:
    """SFGN hyperparameters."""

    foundation_model: str = 'dnabert2'  # model_loader names: dnabert2, nt, hyenadna
    freeze_foundation: bool = True

    grammar_hidden_dim: int = 256
    grammar_output_dim: int = 128
    grammar_n_heads: int = 8
    grammar_n_layers: int = 3
    max_motifs: int = 50

    composition_output_dim: int = 128
    composition_hidden_dim: int = 256
    k_range: tuple = (1, 4)
    gc_window_sizes: tuple = (50, 100, 200)
    use_sequence_embedding: bool = True

    fusion_hidden_dim: int = 128
    fusion_dropout: float = 0.1

    orthogonality_weight: float = 0.1

    output_dim: int = 1

    dropout: float = 0.1


class SFGNOutput(NamedTuple):
    """What SFGN.forward returns."""
    prediction: torch.Tensor          # (batch,) expression prediction
    grammar_vector: torch.Tensor      # (batch, grammar_dim) grammar representation
    composition_vector: torch.Tensor  # (batch, comp_dim) composition representation
    alpha: torch.Tensor               # (batch,) grammar weight
    beta: torch.Tensor                # (batch,) composition weight
    orthogonality_loss: torch.Tensor  # scalar orthogonality penalty


class DisentangledFusion(nn.Module):
    """Combines the grammar and composition vectors with learned alpha/beta
    weights, plus an orthogonality penalty to keep them separate."""

    def __init__(
        self,
        grammar_dim: int,
        composition_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.grammar_dim = grammar_dim
        self.composition_dim = composition_dim

        # both pathways into one space
        self.grammar_transform = nn.Sequential(
            nn.Linear(grammar_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.composition_transform = nn.Sequential(
            nn.Linear(composition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),  # alpha + beta = 1
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        grammar_vector: torch.Tensor,
        composition_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (prediction, alpha, beta, orthogonality_loss)."""
        grammar_vector = torch.nan_to_num(grammar_vector, nan=0.0, posinf=0.0, neginf=0.0)
        composition_vector = torch.nan_to_num(composition_vector, nan=0.0, posinf=0.0, neginf=0.0)

        g_transformed = self.grammar_transform(grammar_vector)
        c_transformed = self.composition_transform(composition_vector)

        combined = torch.cat([g_transformed, c_transformed], dim=-1)
        weights = self.weight_net(combined)
        alpha = weights[:, 0]  # grammar
        beta = weights[:, 1]   # composition

        fused = alpha.unsqueeze(-1) * g_transformed + beta.unsqueeze(-1) * c_transformed

        prediction = self.predictor(fused).squeeze(-1)

        # cosine similarity as a stand-in for correlation between g and c
        g_norm = F.normalize(g_transformed, dim=-1, eps=1e-8)
        c_norm = F.normalize(c_transformed, dim=-1, eps=1e-8)
        cosine_sim = (g_norm * c_norm).sum(dim=-1)  # (batch,)
        orthogonality_loss = cosine_sim.abs().mean()

        if torch.isnan(orthogonality_loss):
            orthogonality_loss = torch.tensor(0.0, device=orthogonality_loss.device)

        return prediction, alpha, beta, orthogonality_loss


class SFGN(nn.Module):
    """Spacer-factored grammar network: grammar effects split from
    composition effects in gene regulation."""

    def __init__(self, config: SFGNConfig = None, device: str = 'cuda'):
        super().__init__()

        if config is None:
            config = SFGNConfig()

        self.config = config
        self.device = device

        self.motif_encoder = MotifEncoder(
            model_name=config.foundation_model,
            pool_strategy='mean',
            device=device,
            freeze=config.freeze_foundation,
        )

        if config.use_sequence_embedding:
            self.sequence_encoder = SequenceEncoder(
                model_name=config.foundation_model,
                device=device,
                freeze=config.freeze_foundation,
            )
        else:
            self.sequence_encoder = None

        self.grammar_module = GrammarModule(
            input_dim=self.motif_encoder.hidden_dim,
            hidden_dim=config.grammar_hidden_dim,
            output_dim=config.grammar_output_dim,
            n_heads=config.grammar_n_heads,
            n_layers=config.grammar_n_layers,
            dropout=config.dropout,
            max_motifs=config.max_motifs,
        ).to(device)

        self.composition_module = CompositionModule(
            output_dim=config.composition_output_dim,
            k_range=config.k_range,
            window_sizes=list(config.gc_window_sizes),
            hidden_dim=config.composition_hidden_dim,
            use_sequence_embedding=config.use_sequence_embedding,
            embedding_dim=self.motif_encoder.hidden_dim if config.use_sequence_embedding else 0,
            dropout=config.dropout,
        ).to(device)

        self.fusion = DisentangledFusion(
            grammar_dim=config.grammar_output_dim,
            composition_dim=config.composition_output_dim,
            hidden_dim=config.fusion_hidden_dim,
            output_dim=config.output_dim,
            dropout=config.fusion_dropout,
        ).to(device)

        self.orthogonality_weight = config.orthogonality_weight

    def forward(
        self,
        sequences: List[str],
        motif_annotations: List[List[Dict]],
    ) -> SFGNOutput:
        """Motif_annotations is one list of motif dicts per sequence."""
        batch_size = len(sequences)
        device = self.device

        # grammar pathway
        motif_data = []
        max_motifs = 0
        for seq, motifs in zip(sequences, motif_annotations):
            if len(motifs) > 0:
                emb, meta = self.motif_encoder(seq, motifs)
                motif_data.append((emb, meta))
                max_motifs = max(max_motifs, len(motifs))
            else:
                motif_data.append((None, []))

        # pad to a common length
        if max_motifs == 0:
            max_motifs = 1

        padded_embeddings = torch.zeros(
            batch_size, max_motifs, self.motif_encoder.hidden_dim, device=device
        )
        positions = torch.zeros(batch_size, max_motifs, dtype=torch.long, device=device)
        strands = torch.zeros(batch_size, max_motifs, dtype=torch.long, device=device)
        mask = torch.zeros(batch_size, max_motifs, device=device)

        for i, (emb, meta) in enumerate(motif_data):
            if emb is not None and len(meta) > 0:
                n = min(len(meta), max_motifs)
                padded_embeddings[i, :n] = emb[:n]
                for j, m in enumerate(meta[:n]):
                    positions[i, j] = min(int(m['start']), 511)
                    strands[i, j] = 0 if m['strand'] == '+' else 1
                mask[i, :n] = 1

        grammar_vector = self.grammar_module(
            padded_embeddings, positions, strands, mask
        )

        # composition pathway
        if self.sequence_encoder is not None:
            seq_embeddings = self.sequence_encoder(sequences)
        else:
            seq_embeddings = None

        composition_vector = self.composition_module(sequences, seq_embeddings)

        prediction, alpha, beta, orth_loss = self.fusion(
            grammar_vector, composition_vector
        )

        return SFGNOutput(
            prediction=prediction,
            grammar_vector=grammar_vector,
            composition_vector=composition_vector,
            alpha=alpha,
            beta=beta,
            orthogonality_loss=orth_loss,
        )

    def compute_loss(
        self,
        output: SFGNOutput,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """MSE plus the orthogonality penalty; returns (loss, metrics)."""
        mse_loss = F.mse_loss(output.prediction, targets)

        orth_loss = output.orthogonality_loss * self.orthogonality_weight

        total_loss = mse_loss + orth_loss

        metrics = {
            'mse_loss': mse_loss.item(),
            'orthogonality_loss': output.orthogonality_loss.item(),
            'total_loss': total_loss.item(),
            'mean_alpha': output.alpha.mean().item(),
            'mean_beta': output.beta.mean().item(),
        }

        return total_loss, metrics

    def predict_expression(self, sequences: List[str], motif_annotations: List[List[Dict]]) -> torch.Tensor:
        """Forward() without grads."""
        self.eval()
        with torch.no_grad():
            output = self.forward(sequences, motif_annotations)
        return output.prediction

    def decompose(
        self,
        sequences: List[str],
        motif_annotations: List[List[Dict]],
    ) -> Dict[str, torch.Tensor]:
        """Prediction plus both vectors and their alpha/beta weights."""
        self.eval()
        with torch.no_grad():
            output = self.forward(sequences, motif_annotations)

        return {
            'prediction': output.prediction,
            'grammar_vector': output.grammar_vector,
            'composition_vector': output.composition_vector,
            'alpha': output.alpha,
            'beta': output.beta,
        }
