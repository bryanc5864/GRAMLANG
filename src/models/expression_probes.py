"""
Expression prediction probes for foundation models.

Foundation models (NT, DNABERT-2, HyenaDNA, Evo, Caduceus, GPN) don't have
built-in expression heads. We train lightweight linear probes on top of their
frozen embeddings to predict MPRA expression.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


class ExpressionProbe(nn.Module):
    """Linear probe on frozen foundation model embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, hidden_dim) pooled embeddings
        Returns:
            (batch,) predicted expression
        """
        return self.head(embeddings).squeeze(-1)


def train_expression_probe(
    embeddings: np.ndarray,
    expressions: np.ndarray,
    input_dim: int,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = 'cuda',
    seed: int = 42,
) -> Tuple[ExpressionProbe, dict]:
    """
    Train an expression probe on pre-extracted embeddings.

    Args:
        embeddings: (n_samples, hidden_dim) array
        expressions: (n_samples,) array of expression values
        input_dim: Hidden dimension of embeddings
        hidden_dim: Probe hidden dimension
        lr: Learning rate
        weight_decay: L2 regularization
        batch_size: Training batch size
        max_epochs: Maximum epochs
        patience: Early stopping patience
        device: CUDA device
        seed: Random seed

    Returns:
        (trained_probe, metrics_dict)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        embeddings, expressions, test_size=0.2, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed
    )

    # Create data loaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # Initialize probe
    probe = ExpressionProbe(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        # Train
        probe.train()
        epoch_loss = 0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = probe(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)

        # Validate
        probe.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = probe(X_batch)
                val_loss += criterion(pred, y_batch).item()
                n_val += 1
        val_loss /= n_val
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    probe.load_state_dict(best_state)
    probe.to(device)
    probe.eval()

    # Evaluate on test set
    all_preds = []
    all_true = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = probe(X_batch).cpu().numpy()
            all_preds.extend(pred)
            all_true.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    # Compute metrics
    pearson_r, pearson_p = pearsonr(all_preds, all_true)
    spearman_r, spearman_p = spearmanr(all_preds, all_true)
    r_squared = pearson_r ** 2
    mse = np.mean((all_preds - all_true) ** 2)

    metrics = {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_rho': float(spearman_r),
        'spearman_p': float(spearman_p),
        'r_squared': float(r_squared),
        'mse': float(mse),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'best_epoch': max_epochs - epochs_no_improve,
        'viable': pearson_r > 0.3,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }

    return probe, metrics


def extract_and_cache_embeddings(
    model,
    sequences: list,
    cache_path: str,
    batch_size: int = 32,
    layer: int = -1
) -> np.ndarray:
    """
    Extract embeddings and cache to disk.

    Args:
        model: GrammarModel instance
        sequences: List of DNA sequences
        cache_path: Path to save/load embeddings
        batch_size: Processing batch size
        layer: Layer to extract from

    Returns:
        (n_sequences, hidden_dim) array
    """
    if os.path.exists(cache_path):
        print(f"  Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"  Extracting embeddings for {len(sequences)} sequences...")
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), batch_size)):
        batch = sequences[i:i + batch_size]
        embs = model.get_embeddings(batch, layer=layer)
        all_embeddings.append(embs)

    embeddings = np.concatenate(all_embeddings, axis=0)

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"  Saved embeddings to {cache_path} ({embeddings.nbytes / 1e6:.1f} MB)")

    return embeddings


def save_probe(probe: ExpressionProbe, metrics: dict, save_dir: str, model_name: str):
    """Save trained probe and metrics."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(probe.state_dict(), os.path.join(save_dir, f'{model_name}_probe.pt'))

    import json

    def make_serializable(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (list, tuple)):
            return [make_serializable(x) for x in v]
        return v

    metrics_save = {k: make_serializable(v) for k, v in metrics.items()}

    with open(os.path.join(save_dir, f'{model_name}_probe_metrics.json'), 'w') as f:
        json.dump(metrics_save, f, indent=2)


def load_probe(save_dir: str, model_name: str, input_dim: int,
               hidden_dim: int = 256, device: str = 'cuda') -> ExpressionProbe:
    """Load a trained probe."""
    probe = ExpressionProbe(input_dim, hidden_dim)
    state = torch.load(os.path.join(save_dir, f'{model_name}_probe.pt'),
                       map_location=device)
    probe.load_state_dict(state)
    probe.to(device)
    probe.eval()
    return probe
