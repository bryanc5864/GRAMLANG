"""
Experiment C: Attention-Based Grammar Extraction.

For transformer models (DNABERT-2, NT v2-500M), analyze attention patterns
to identify grammar-encoding heads and motif-to-motif attention.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union


def extract_attention_patterns(
    model,
    sequences: List[str],
    motif_annotations: List[Dict],
    cell_type: str = None,
) -> Union[Dict, pd.DataFrame]:
    """
    Extract attention patterns between motif positions.

    For each sequence:
    1. Run forward pass capturing all attention weights
    2. Identify attention between motif positions
    3. Score motif-to-motif attention strength per head

    Only works for transformer models (DNABERT-2, NT v2-500M).
    """
    if model.architecture_type != 'transformer':
        return {'error': f'Attention analysis requires transformer, got {model.architecture_type}'}

    if model.name == 'enformer':
        return {'error': 'Enformer attention extraction not implemented (complex architecture)'}

    results = []

    for seq, annot in zip(sequences, motif_annotations):
        motifs = annot.get('motifs', [])
        if len(motifs) < 2:
            continue

        # Get attention weights
        attention = _get_attention_weights(model, seq)
        if attention is None:
            continue

        # attention shape: (n_layers, n_heads, seq_len, seq_len)
        n_layers, n_heads, seq_len, _ = attention.shape

        # Map motif positions to token positions
        motif_token_ranges = _map_motifs_to_tokens(model, seq, motifs)

        if len(motif_token_ranges) < 2:
            continue

        # Compute motif-to-motif attention for each (layer, head)
        for li in range(n_layers):
            for hi in range(n_heads):
                attn = attention[li, hi]  # (seq_len, seq_len)

                # For each motif pair, average attention between their token positions
                for mi, (name_i, start_i, end_i) in enumerate(motif_token_ranges):
                    for mj, (name_j, start_j, end_j) in enumerate(motif_token_ranges):
                        if mi >= mj:
                            continue

                        # Average attention from motif i tokens to motif j tokens
                        if end_i <= seq_len and end_j <= seq_len:
                            attn_ij = attn[start_i:end_i, start_j:end_j].mean()
                            attn_ji = attn[start_j:end_j, start_i:end_i].mean()

                            # Background: average attention to random positions
                            bg_attn = attn.mean()

                            results.append({
                                'motif_a': name_i,
                                'motif_b': name_j,
                                'layer': li,
                                'head': hi,
                                'attn_ab': float(attn_ij),
                                'attn_ba': float(attn_ji),
                                'attn_mean': float((attn_ij + attn_ji) / 2),
                                'background_attn': float(bg_attn),
                                'attn_enrichment': float((attn_ij + attn_ji) / 2 / max(bg_attn, 1e-10)),
                            })

    return pd.DataFrame(results)


def identify_grammar_heads(
    attention_df: pd.DataFrame,
    gsi_results: pd.DataFrame,
    enrichment_threshold: float = 2.0,
) -> Dict:
    """
    Identify attention heads that encode grammar.

    Grammar heads are defined as heads where:
    - Motif-to-motif attention is enriched above background
    - Attention enrichment correlates with grammar sensitivity
    """
    if len(attention_df) == 0:
        return {'error': 'No attention data'}

    # Average enrichment per (layer, head)
    head_stats = attention_df.groupby(['layer', 'head']).agg(
        mean_enrichment=('attn_enrichment', 'mean'),
        max_enrichment=('attn_enrichment', 'max'),
        n_pairs=('motif_a', 'count'),
    ).reset_index()

    # Identify enriched heads
    grammar_heads = head_stats[head_stats['mean_enrichment'] > enrichment_threshold]

    return {
        'n_heads_total': len(head_stats),
        'n_grammar_heads': len(grammar_heads),
        'grammar_head_fraction': float(len(grammar_heads) / max(len(head_stats), 1)),
        'grammar_heads': grammar_heads.to_dict('records') if len(grammar_heads) > 0 else [],
        'top_heads': head_stats.nlargest(10, 'mean_enrichment').to_dict('records'),
        'mean_enrichment_all': float(head_stats['mean_enrichment'].mean()),
        'mean_enrichment_grammar': float(grammar_heads['mean_enrichment'].mean()) if len(grammar_heads) > 0 else 0,
    }


def compare_attention_original_vs_shuffled(
    model,
    original_seqs: List[str],
    shuffled_seqs: List[str],
    motif_annotations: List[Dict],
    cell_type: str = None,
) -> Dict:
    """
    Compare attention patterns between original and shuffled sequences.

    For grammar-sensitive enhancers, which attention heads change most
    between original and shuffled arrangements?
    """
    results = []

    for orig_seq, shuf_seq, annot in zip(original_seqs, shuffled_seqs, motif_annotations):
        # Get attention for both
        orig_attn = _get_attention_weights(model, orig_seq)
        shuf_attn = _get_attention_weights(model, shuf_seq)

        if orig_attn is None or shuf_attn is None:
            continue

        # Ensure same shape (they should be since same length)
        min_len = min(orig_attn.shape[2], shuf_attn.shape[2])
        orig_attn = orig_attn[:, :, :min_len, :min_len]
        shuf_attn = shuf_attn[:, :, :min_len, :min_len]

        # Compute per-head difference
        n_layers, n_heads = orig_attn.shape[:2]
        for li in range(n_layers):
            for hi in range(n_heads):
                diff = np.abs(orig_attn[li, hi] - shuf_attn[li, hi]).mean()
                results.append({
                    'layer': li,
                    'head': hi,
                    'attention_change': float(diff),
                })

    if not results:
        return {'error': 'No attention comparison data'}

    df = pd.DataFrame(results)
    head_changes = df.groupby(['layer', 'head'])['attention_change'].agg(['mean', 'std']).reset_index()

    return {
        'n_comparisons': len(original_seqs),
        'top_changing_heads': head_changes.nlargest(10, 'mean').to_dict('records'),
        'mean_attention_change': float(head_changes['mean'].mean()),
    }


def _get_attention_weights(model, sequence: str) -> Optional[np.ndarray]:
    """Extract attention weights from a transformer model."""
    try:
        if model.name == 'dnabert2':
            return _get_dnabert2_attention(model, sequence)
        elif model.name == 'nt':
            return _get_nt_attention(model, sequence)
        else:
            return None
    except Exception:
        return None


def _get_dnabert2_attention(model, sequence: str) -> Optional[np.ndarray]:
    """Extract attention from DNABERT-2."""
    tokens = model.tokenizer(
        [sequence], return_tensors="pt", padding=True,
        truncation=True, max_length=512
    ).to(model.device)

    with torch.no_grad():
        # Try output_attentions first (may work with custom BertModel)
        try:
            out = model.model(**tokens, output_attentions=True)
            if hasattr(out, 'attentions') and out.attentions:
                attn_list = [a[0].cpu().numpy() for a in out.attentions]
                return np.array(attn_list)
            # Some custom models return tuple: check for attentions in tuple
            if isinstance(out, tuple) and len(out) > 2:
                # Try index 2 (common for: hidden_states, pooled, attentions)
                candidate = out[2]
                if isinstance(candidate, (list, tuple)) and len(candidate) > 0:
                    attn_list = [a[0].cpu().numpy() if isinstance(a, torch.Tensor) else a for a in candidate]
                    return np.array(attn_list)
        except TypeError:
            pass  # output_attentions not supported

        # Fallback: hook into self-attention modules to capture QK attention
        attentions = []
        handles = []

        if not hasattr(model.model, 'encoder') or not hasattr(model.model.encoder, 'layer'):
            return None

        for layer in model.model.encoder.layer:
            # Hook into the self-attention sub-module, not the attention wrapper
            attn_self = getattr(layer.attention, 'self', layer.attention)
            def make_hook(storage):
                def hook_fn(module, inp, out):
                    if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                        storage.append(out[1].detach().cpu().numpy())
                return hook_fn
            hook_list = []
            attentions.append(hook_list)
            handles.append(attn_self.register_forward_hook(make_hook(hook_list)))

        try:
            model.model(**tokens)
        finally:
            for h in handles:
                h.remove()

        attn_array = []
        for layer_attns in attentions:
            if layer_attns:
                attn_array.append(layer_attns[0][0])  # Remove batch dim

        if attn_array:
            return np.array(attn_array)
        return None


def _get_nt_attention(model, sequence: str) -> Optional[np.ndarray]:
    """Extract attention from NT v2."""
    tokens = model.tokenizer(
        [sequence], return_tensors="pt", padding=True,
        truncation=True, max_length=1000
    ).to(model.device)

    with torch.no_grad():
        out = model.model(**tokens, output_attentions=True)

    if hasattr(out, 'attentions') and out.attentions:
        attn_list = [a[0].cpu().numpy() for a in out.attentions]
        return np.array(attn_list)
    return None


def _map_motifs_to_tokens(model, sequence: str, motifs: List[Dict]) -> List[Tuple[str, int, int]]:
    """Map motif character positions to token positions."""
    # This is an approximation - exact mapping depends on tokenizer
    if model.name == 'dnabert2':
        # DNABERT-2 uses BPE tokenization, ~6 chars per token
        char_per_token = 6
    elif model.name == 'nt':
        # NT uses 6-mer tokenization
        char_per_token = 6
    else:
        char_per_token = 1

    result = []
    for m in motifs:
        name = m.get('motif_name', 'unknown')
        start = int(m.get('start', 0))
        end = int(m.get('end', 0))

        # Approximate token positions (add 1 for CLS token)
        token_start = start // char_per_token + 1
        token_end = max(token_start + 1, end // char_per_token + 1)

        result.append((name, token_start, token_end))

    return result
