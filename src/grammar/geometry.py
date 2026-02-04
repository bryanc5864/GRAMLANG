"""
Grammar representation geometry analysis.

Finds the geometric structure of grammar in model embedding spaces.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cosine
from itertools import combinations


def compute_grammar_direction(
    model,
    correct_sequences: list,
    shuffled_sequences: list,
    layer: int = -1
) -> dict:
    """
    Find the "grammar direction" in embedding space.

    Args:
        model: GrammarModel instance
        correct_sequences: Natural arrangement, high expression
        shuffled_sequences: Vocabulary-matched shuffled versions
        layer: Which layer

    Returns:
        Dict with grammar direction, linearity metrics
    """
    correct_embeds = model.get_embeddings(correct_sequences, layer=layer)
    shuffled_embeds = model.get_embeddings(shuffled_sequences, layer=layer)

    # Grammar direction: mean difference
    grammar_dir = correct_embeds.mean(axis=0) - shuffled_embeds.mean(axis=0)
    norm = np.linalg.norm(grammar_dir)
    grammar_dir_norm = grammar_dir / max(norm, 1e-10)

    # Test linearity: linear probe to distinguish correct vs shuffled
    X = np.concatenate([correct_embeds, shuffled_embeds], axis=0)
    y = np.array([1] * len(correct_embeds) + [0] * len(shuffled_embeds))

    if len(X) >= 10:
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        scores = cross_val_score(clf, X, y, cv=min(5, len(X) // 2), scoring='accuracy')
        linearity_acc = float(scores.mean())
    else:
        linearity_acc = 0.5

    return {
        'grammar_direction': grammar_dir_norm,
        'grammar_direction_raw': grammar_dir,
        'linearity_accuracy': linearity_acc,
        'linearity_r2': 2 * (linearity_acc - 0.5),
        'direction_norm': float(norm),
        'layer': layer,
        'n_samples': len(correct_sequences),
    }


def layer_sweep_grammar(
    model,
    correct_sequences: list,
    shuffled_sequences: list
) -> pd.DataFrame:
    """Find which layer encodes grammar most strongly."""
    results = []
    for layer_idx in range(model.num_layers):
        try:
            result = compute_grammar_direction(
                model, correct_sequences, shuffled_sequences, layer=layer_idx
            )
            results.append({
                'layer': layer_idx,
                'linearity_accuracy': result['linearity_accuracy'],
                'linearity_r2': result['linearity_r2'],
                'direction_norm': result['direction_norm'],
            })
        except Exception as e:
            results.append({
                'layer': layer_idx,
                'linearity_accuracy': 0.5,
                'linearity_r2': 0.0,
                'direction_norm': 0.0,
            })
    return pd.DataFrame(results)


def compute_cross_model_alignment(
    grammar_directions: dict,
    model_names: list
) -> pd.DataFrame:
    """Compute alignment of grammar directions across models."""
    results = []

    for m1, m2 in combinations(model_names, 2):
        if m1 not in grammar_directions or m2 not in grammar_directions:
            continue

        dir1 = grammar_directions[m1]['grammar_direction']
        dir2 = grammar_directions[m2]['grammar_direction']

        # If dimensions match, compute cosine similarity directly
        if len(dir1) == len(dir2):
            cos_sim = 1 - cosine(dir1, dir2)
        else:
            # Random projection to shared space
            target_dim = min(len(dir1), len(dir2), 128)
            rng = np.random.default_rng(42)
            proj1 = rng.standard_normal((len(dir1), target_dim)) / np.sqrt(target_dim)
            proj2 = rng.standard_normal((len(dir2), target_dim)) / np.sqrt(target_dim)
            dir1_proj = dir1 @ proj1
            dir2_proj = dir2 @ proj2
            cos_sim = 1 - cosine(dir1_proj, dir2_proj)

        results.append({
            'model_1': m1,
            'model_2': m2,
            'cosine_similarity': float(cos_sim),
            'dim_1': len(dir1),
            'dim_2': len(dir2),
        })

    return pd.DataFrame(results)
