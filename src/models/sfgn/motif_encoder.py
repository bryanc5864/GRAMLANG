"""
Motif Encoder: Extract motif-level embeddings from frozen foundation model.

Given a sequence and motif annotations, extracts embeddings at motif positions
and pools them into motif-level representations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple

# Use the existing model loader that handles DNABERT-2 correctly
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class MotifEncoder(nn.Module):
    """
    Extracts motif-level embeddings from a frozen foundation model.

    For each motif in the sequence:
    1. Get token embeddings from foundation model
    2. Pool tokens within motif span → motif embedding
    3. Return list of motif embeddings + their metadata
    """

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

        # Use existing model loader
        from src.models.model_loader import load_model
        self._base_model = load_model(model_name, dataset_name='__dummy__')

        # Access internal model and tokenizer
        self.model = self._base_model.model
        self.tokenizer = self._base_model.tokenizer

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.hidden_dim = self._base_model.hidden_dim

    def _get_token_embeddings(self, sequence: str) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Get token-level embeddings and character-to-token mapping.

        Returns:
            embeddings: (seq_len, hidden_dim) token embeddings
            char_to_token: list of (start_char, end_char) for each token
        """
        # Tokenize with offset mapping
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
            # Use last hidden state
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state[0]  # (n_tokens, hidden_dim)
            else:
                embeddings = outputs[0][0]  # Fallback for tuple output

        return embeddings, offset_mapping

    def _pool_motif_tokens(
        self,
        embeddings: torch.Tensor,
        offset_mapping: List[Tuple[int, int]],
        motif_start: int,
        motif_end: int
    ) -> torch.Tensor:
        """
        Pool token embeddings within a motif span.
        """
        # Find tokens that overlap with motif
        motif_token_indices = []
        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start is None or tok_end is None:
                continue
            # Check overlap
            if tok_start < motif_end and tok_end > motif_start:
                motif_token_indices.append(idx)

        if len(motif_token_indices) == 0:
            # Fallback: return zero embedding
            return torch.zeros(self.hidden_dim, device=self.device)

        motif_embeddings = embeddings[motif_token_indices]  # (n_motif_tokens, hidden_dim)

        if self.pool_strategy == 'mean':
            return motif_embeddings.mean(dim=0)
        elif self.pool_strategy == 'max':
            return motif_embeddings.max(dim=0)[0]
        elif self.pool_strategy == 'cls':
            return motif_embeddings[0]  # First token
        else:
            return motif_embeddings.mean(dim=0)

    def forward(
        self,
        sequence: str,
        motif_annotations: List[Dict]
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Extract motif embeddings from sequence.

        Args:
            sequence: DNA sequence string
            motif_annotations: List of dicts with 'start', 'end', 'motif_name', 'strand'

        Returns:
            motif_embeddings: (n_motifs, hidden_dim) tensor
            motif_metadata: List of dicts with position/name info
        """
        if len(motif_annotations) == 0:
            return torch.zeros(0, self.hidden_dim, device=self.device), []

        # Get token embeddings
        token_embeddings, offset_mapping = self._get_token_embeddings(sequence)

        # Pool for each motif
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
        """
        Batch forward pass.
        """
        results = []
        for seq, motifs in zip(sequences, motif_annotations_batch):
            results.append(self.forward(seq, motifs))
        return results


class SequenceEncoder(nn.Module):
    """
    Full sequence encoder for composition features.
    Returns pooled sequence embedding (for GC, k-mer info implicitly encoded).
    """

    def __init__(
        self,
        model_name: str = 'dnabert2',
        device: str = 'cuda',
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device

        # Use existing model loader
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
        """
        Get pooled sequence embeddings.

        Args:
            sequences: List of DNA sequences

        Returns:
            embeddings: (batch_size, hidden_dim)
        """
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

            # Mean pool over sequence length
            mask = encoding['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return pooled
