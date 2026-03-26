#!/usr/bin/env python3
"""Debug NaN issues in SFGN training."""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.sfgn import SFGN, SFGNConfig


def check_tensor(name, tensor):
    """Check tensor for NaN/Inf."""
    if tensor is None:
        print(f"  {name}: None")
        return False
    if isinstance(tensor, (list, tuple)):
        print(f"  {name}: list/tuple of {len(tensor)} items")
        return False
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    min_val = tensor.min().item() if tensor.numel() > 0 else 0
    max_val = tensor.max().item() if tensor.numel() > 0 else 0
    mean_val = tensor.float().mean().item() if tensor.numel() > 0 else 0

    status = "OK" if not (has_nan or has_inf) else "PROBLEM!"
    if has_nan:
        status = "HAS NaN!"
    if has_inf:
        status = "HAS Inf!"

    print(f"  {name}: shape={list(tensor.shape)}, min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f} [{status}]")
    return has_nan or has_inf


def load_dataset(dataset_name: str, data_dir: str = 'data/processed'):
    """Load dataset and motif annotations."""
    data_path = Path(data_dir) / f'{dataset_name}_processed.parquet'
    motif_path = Path(data_dir) / f'{dataset_name}_processed_motif_hits.parquet'

    df = pd.read_parquet(data_path)
    motif_df = pd.read_parquet(motif_path)

    motif_groups = motif_df.groupby('seq_id')

    motif_annotations = []
    for seq_id in df['seq_id']:
        if seq_id in motif_groups.groups:
            group = motif_groups.get_group(seq_id)
            motifs = []
            for _, row in group.iterrows():
                motifs.append({
                    'start': int(row['start']),
                    'end': int(row['end']),
                    'motif_name': row.get('motif_name', 'unknown'),
                    'strand': row.get('strand', '+'),
                })
            motif_annotations.append(motifs)
        else:
            motif_annotations.append([])

    return df, motif_annotations


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    df, motif_annotations = load_dataset('agarwal')

    # Sample a small batch
    indices = [0, 1, 2, 3]
    sequences = [df.iloc[i]['sequence'] for i in indices]
    expressions = torch.tensor([df.iloc[i]['expression'] for i in indices], dtype=torch.float32).to(device)
    motifs = [motif_annotations[i] for i in indices]

    print(f"\n--- Batch info ---")
    for i, (seq, m) in enumerate(zip(sequences, motifs)):
        print(f"  Seq {i}: len={len(seq)}, n_motifs={len(m)}")

    # Create model
    config = SFGNConfig()
    model = SFGN(config, device=device)

    print(f"\n--- Forward pass ---")

    # Manual forward with debug
    print("\n1. Motif encoding...")
    batch_size = len(sequences)
    motif_data = []
    max_motifs = 0

    for seq, m in zip(sequences, motifs):
        if len(m) > 0:
            emb, meta = model.motif_encoder(seq, m)
            check_tensor(f"motif_emb (n={len(m)})", emb)
            motif_data.append((emb, meta))
            max_motifs = max(max_motifs, len(m))
        else:
            motif_data.append((None, []))

    print(f"\n2. Building padded tensors (max_motifs={max_motifs})...")

    if max_motifs == 0:
        max_motifs = 1

    padded_embeddings = torch.zeros(batch_size, max_motifs, model.motif_encoder.hidden_dim, device=device)
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

    check_tensor("padded_embeddings", padded_embeddings)
    check_tensor("positions", positions)
    check_tensor("strands", strands)
    check_tensor("mask", mask)

    print(f"\n3. Grammar module forward...")
    grammar_vector = model.grammar_module(padded_embeddings, positions, strands, mask)
    problem = check_tensor("grammar_vector", grammar_vector)

    if problem:
        print("\n  [!] Grammar module output has issues. Debugging internals...")

        # Debug grammar module internals
        x = model.grammar_module.input_proj(padded_embeddings)
        check_tensor("  after input_proj", x)

        pos_enc = model.grammar_module.pos_encoding(positions)
        check_tensor("  pos_encoding", pos_enc)

        strand_enc = model.grammar_module.strand_encoding(strands)
        check_tensor("  strand_encoding", strand_enc)

        x = x + pos_enc + strand_enc
        check_tensor("  after adding encodings", x)

        for layer_idx, layer in enumerate(model.grammar_module.layers):
            x_normed = layer['norm1'](x)
            check_tensor(f"  layer {layer_idx} norm1", x_normed)

            attn_out = layer['attention'](x_normed, positions, mask)
            check_tensor(f"  layer {layer_idx} attention", attn_out)

            x = x + attn_out
            check_tensor(f"  layer {layer_idx} after attn residual", x)

            x = x + layer['ffn'](layer['norm2'](x))
            check_tensor(f"  layer {layer_idx} after ffn", x)

    print(f"\n4. Sequence encoding...")
    if model.sequence_encoder is not None:
        seq_embeddings = model.sequence_encoder(sequences)
        check_tensor("seq_embeddings", seq_embeddings)
    else:
        seq_embeddings = None

    print(f"\n5. Composition module forward...")
    composition_vector = model.composition_module(sequences, seq_embeddings)
    check_tensor("composition_vector", composition_vector)

    print(f"\n6. Fusion...")
    prediction, alpha, beta, orth_loss = model.fusion(grammar_vector, composition_vector)
    check_tensor("prediction", prediction)
    check_tensor("alpha", alpha)
    check_tensor("beta", beta)
    check_tensor("orth_loss", orth_loss.unsqueeze(0))

    print(f"\n7. Full forward pass...")
    output = model(sequences, motifs)
    check_tensor("output.prediction", output.prediction)
    check_tensor("output.grammar_vector", output.grammar_vector)
    check_tensor("output.composition_vector", output.composition_vector)

    print(f"\n8. Loss computation...")
    loss, metrics = model.compute_loss(output, expressions)
    print(f"  loss = {loss.item():.4f}")
    print(f"  metrics = {metrics}")

    print(f"\n9. Backward pass...")
    loss.backward()

    # Check gradients
    print("\n10. Gradient check...")
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            if has_nan or has_inf:
                print(f"  {name}: HAS NaN/Inf in gradient!")

    print("\n--- Second batch (different sequences) ---")
    indices2 = [100, 101, 102, 103]
    sequences2 = [df.iloc[i]['sequence'] for i in indices2]
    expressions2 = torch.tensor([df.iloc[i]['expression'] for i in indices2], dtype=torch.float32).to(device)
    motifs2 = [motif_annotations[i] for i in indices2]

    for i, (seq, m) in enumerate(zip(sequences2, motifs2)):
        print(f"  Seq {i}: len={len(seq)}, n_motifs={len(m)}")

    model.zero_grad()
    output2 = model(sequences2, motifs2)
    check_tensor("output2.prediction", output2.prediction)
    check_tensor("output2.grammar_vector", output2.grammar_vector)

    loss2, metrics2 = model.compute_loss(output2, expressions2)
    print(f"  loss2 = {loss2.item():.4f}")

    loss2.backward()

    # Check gradients again
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            if has_nan or has_inf:
                print(f"  {name}: HAS NaN/Inf in gradient after batch 2!")

    print("\n=== Debug complete ===")


if __name__ == '__main__':
    main()
