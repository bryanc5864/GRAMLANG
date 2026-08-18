"""Token-level backbone access.

model_loader.GrammarModel only gives mean-pooled embeddings, which is the
readout bottleneck itself. Here we expose token-level hidden states plus the
attention mask so sequence-aware heads (attention pooling, CNN, MLP) train on
the same frozen representations as the mean-pool baseline. Backbones stay
frozen; only heads train.
"""

from __future__ import annotations

import gc
from typing import List, Tuple

import numpy as np
import torch


def _dnabert2_disable_triton():
    """DNABERT-2's Triton flash-attn kernel no longer compiles; nulling the
    symbol makes bert_layers fall back to plain PyTorch attention."""
    import sys

    for name, mod in list(sys.modules.items()):
        if name.endswith("bert_layers") and hasattr(mod, "flash_attn_qkvpacked_func"):
            mod.flash_attn_qkvpacked_func = None


class Backbone:
    """Frozen DNA language model exposing padded token-level hidden states."""

    def __init__(self, name: str, device: str = "cuda", dtype=torch.float32):
        self.name = name
        self.device = device
        self.dtype = dtype
        self._load()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _load(self):
        from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

        if self.name == "dnabert2":
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            repo = "zhihan1996/DNABERT-2-117M"
            self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            ModelClass = get_class_from_dynamic_module("bert_layers.BertModel", repo)
            _dnabert2_disable_triton()
            self.model = ModelClass.from_pretrained(repo).to(self.device)
            _dnabert2_disable_triton()
            self.hidden_dim = 768
            self.max_length = 512

        elif self.name == "nt":
            repo = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
            self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            self.model = AutoModelForMaskedLM.from_pretrained(
                repo, trust_remote_code=True
            ).to(self.device)
            self.hidden_dim = 1024
            self.max_length = 512

        elif self.name == "hyenadna":
            repo = "LongSafari/hyenadna-large-1m-seqlen-hf"
            self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(
                self.device
            )
            self.hidden_dim = 256
            self.max_length = 1024

        else:
            raise ValueError(f"unknown backbone {self.name!r}")

    @torch.no_grad()
    def _forward_batch(self, seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}
        mask = tok.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(tok["input_ids"])

        if self.name == "dnabert2":
            out = self.model(**tok)
            hidden = out[0]
            # unpadded path can hand back (total_nnz, D)
            if hidden.dim() == 2:
                hidden = _repad(hidden, mask)
        elif self.name == "nt":
            out = self.model(**tok, output_hidden_states=True)
            hidden = out.hidden_states[-1]
        else:
            out = self.model(**tok, output_hidden_states=True)
            hidden = out.hidden_states[-1]

        return hidden.float(), mask.float()

    @torch.no_grad()
    def token_embeddings(
        self, seqs: List[str], batch_size: int = 64, max_tokens: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(N, L, D) float16 hidden states + (N, L) mask, padded to max_tokens
        (default: longest sequence seen)."""
        chunks, masks = [], []
        for i in range(0, len(seqs), batch_size):
            h, m = self._forward_batch(seqs[i : i + batch_size])
            chunks.append(h.cpu())
            masks.append(m.cpu())
        L = max_tokens or max(c.shape[1] for c in chunks)
        D = chunks[0].shape[2]
        N = len(seqs)
        H = torch.zeros(N, L, D)
        M = torch.zeros(N, L)
        off = 0
        for h, m in zip(chunks, masks):
            n, l, _ = h.shape
            l = min(l, L)
            H[off : off + n, :l] = h[:, :l]
            M[off : off + n, :l] = m[:, :l]
            off += n
        return H.numpy().astype(np.float16), M.numpy().astype(np.float32)

    @torch.no_grad()
    def mean_embeddings(self, seqs: List[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(seqs), batch_size):
            h, m = self._forward_batch(seqs[i : i + batch_size])
            pooled = (h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            out.append(pooled.cpu().numpy())
        return np.concatenate(out, 0)

    @torch.no_grad()
    def predict_with_head(
        self, seqs: List[str], head, batch_size: int = 64
    ) -> np.ndarray:
        """Backbone + trained head, end to end."""
        out = []
        for i in range(0, len(seqs), batch_size):
            h, m = self._forward_batch(seqs[i : i + batch_size])
            out.append(head(h, m).squeeze(-1).float().cpu().numpy())
        return np.concatenate(out, 0)

    def unload(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def _repad(flat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(total_nnz, D) -> (B, L, D)."""
    B, L = mask.shape
    D = flat.shape[-1]
    out = flat.new_zeros(B, L, D)
    out[mask.bool()] = flat
    return out
