"""Motif-level embeddings pulled out of a frozen foundation model.

given a sequence plus motif annotations, pool the token embeddings that fall
inside each motif span.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple

# model_loader is the only thing that loads DNABERT-2 correctly
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class MotifEncoder(nn.Module):
    """Per-motif embeddings: token embeddings pooled over each motif span,
    returned with the motif metadata."""

    def __init__(
        self,
        model_name: str = 'dnabert2',
        pool_strategy: str = 'mean',  # 'mean', 'max', 'cls'
        device: str = 'cuda',
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.pool_strategy = pool_strategy
        self.device = device
        self._freeze = freeze

        from src.models.model_loader import load_model
        self._base_model = load_model(model_name, dataset_name='__dummy__')

        self.model = self._base_model.model
        self.tokenizer = self._base_model.tokenizer

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.hidden_dim = self._base_model.hidden_dim

    def _get_token_embeddings(self, sequence: str) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """(seq_len, hidden_dim) token embeddings + per-token char offsets."""
        encoding = self.tokenizer(
            sequence,
            return_tensors='pt',
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )

        input_ids = encoding['input_ids'].to(self.device)
        offset_mapping = encoding['offset_mapping'][0].tolist()  # (n_tokens, 2)

        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state[0]  # (n_tokens, hidden_dim)
            else:
                embeddings = outputs[0][0]  # tuple output

        return embeddings, offset_mapping

    def _pool_motif_tokens(
        self,
        embeddings: torch.Tensor,
        offset_mapping: List[Tuple[int, int]],
        motif_start: int,
        motif_end: int
    ) -> torch.Tensor:
        """Pool token embeddings inside one motif span."""
        motif_token_indices = []
        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start is None or tok_end is None:
                continue
            if tok_start < motif_end and tok_end > motif_start:
                motif_token_indices.append(idx)

        if len(motif_token_indices) == 0:
            # no token overlaps the span
            return torch.zeros(self.hidden_dim, device=self.device)

        motif_embeddings = embeddings[motif_token_indices]  # (n_motif_tokens, hidden_dim)

        if self.pool_strategy == 'mean':
            return motif_embeddings.mean(dim=0)
        elif self.pool_strategy == 'max':
            return motif_embeddings.max(dim=0)[0]
        elif self.pool_strategy == 'cls':
            return motif_embeddings[0]
        else:
            return motif_embeddings.mean(dim=0)

    def forward(
        self,
        sequence: str,
        motif_annotations: List[Dict]
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """Motif_annotations: dicts with 'start', 'end', 'motif_name', 'strand'.

        returns (n_motifs, hidden_dim) embeddings + matching metadata dicts."""
        if len(motif_annotations) == 0:
            return torch.zeros(0, self.hidden_dim, device=self.device), []

        token_embeddings, offset_mapping = self._get_token_embeddings(sequence)

        motif_embeddings = []
        motif_metadata = []

        for motif in sorted(motif_annotations, key=lambda m: m['start']):
            emb = self._pool_motif_tokens(
                token_embeddings,
                offset_mapping,
                motif['start'],
                motif['end']
            )
            motif_embeddings.append(emb)

            motif_metadata.append({
                'start': motif['start'],
                'end': motif['end'],
                'name': motif.get('motif_name', 'unknown'),
                'strand': motif.get('strand', '+'),
                'length': motif['end'] - motif['start'],
                'relative_position': motif['start'] / len(sequence),
            })

        motif_embeddings = torch.stack(motif_embeddings)  # (n_motifs, hidden_dim)

        return motif_embeddings, motif_metadata

    def forward_batch(
        self,
        sequences: List[str],
        motif_annotations_batch: List[List[Dict]],
    ) -> List[Tuple[torch.Tensor, List[Dict]]]:
        """Forward() over a batch, one sequence at a time."""
        results = []
        for seq, motifs in zip(sequences, motif_annotations_batch):
            results.append(self.forward(seq, motifs))
        return results


class SequenceEncoder(nn.Module):
    """Whole-sequence pooled embedding, used for the composition features
    (GC and k-mer content end up encoded implicitly)."""

    def __init__(
        self,
        model_name: str = 'dnabert2',
        device: str = 'cuda',
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device

        from src.models.model_loader import load_model
        self._base_model = load_model(model_name, dataset_name='__dummy__')

        self.model = self._base_model.model
        self.tokenizer = self._base_model.tokenizer

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.hidden_dim = self._base_model.hidden_dim

    def forward(self, sequences: List[str]) -> torch.Tensor:
        """Mean-pooled embeddings, (batch_size, hidden_dim)."""
        encoding = self.tokenizer(
            sequences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoding, output_hidden_states=True)
            if hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs[0]

            mask = encoding['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return pooled
