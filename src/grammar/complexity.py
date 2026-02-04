"""
Formal grammar complexity classification on the Chomsky hierarchy.

Classifies regulatory grammar as regular, context-free, or context-sensitive
based on how the compositionality gap grows with motif count.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict


def classify_grammar_complexity(
    motif_counts: np.ndarray,
    compositionality_gaps: np.ndarray,
    gap_errors: np.ndarray = None,
) -> dict:
    """
    Classify grammar complexity based on compositionality gap growth.

    Regular (constant): gap = c
    Context-free (linear): gap = a*k + b
    Context-sensitive (exponential): gap = a*exp(b*k) + c

    Args:
        motif_counts: Array of motif counts (3, 4, 5, ...)
        compositionality_gaps: Mean compositionality gap per count
        gap_errors: Standard errors (optional)

    Returns:
        Dict with classification, confidence, fit results
    """
    k = motif_counts.astype(float)
    g = compositionality_gaps.astype(float)
    n = len(k)

    if n < 3:
        return {
            'classification': 'insufficient_data',
            'confidence': 0.0,
            'note': f'Only {n} motif count levels; need at least 3'
        }

    results = {}

    # Constant fit (regular grammar)
    c_mean = np.mean(g)
    ss_res_const = np.sum((g - c_mean) ** 2)
    results['constant'] = {
        'params': [float(c_mean)],
        'ss_res': float(ss_res_const),
        'n_params': 1
    }

    # Linear fit (context-free)
    try:
        coeffs = np.polyfit(k, g, 1)
        g_pred = np.polyval(coeffs, k)
        ss_res_lin = np.sum((g - g_pred) ** 2)
        results['linear'] = {
            'params': coeffs.tolist(),
            'ss_res': float(ss_res_lin),
            'n_params': 2,
            'slope': float(coeffs[0]),
            'intercept': float(coeffs[1])
        }
    except Exception:
        results['linear'] = {'params': [0, 0], 'ss_res': float('inf'), 'n_params': 2}

    # Exponential fit (context-sensitive)
    try:
        def exponential(x, a, b, c):
            return a * np.exp(b * x) + c
        popt, _ = curve_fit(exponential, k, g, p0=[0.01, 0.3, 0], maxfev=10000,
                            bounds=([0, 0, -np.inf], [np.inf, 2, np.inf]))
        g_pred = exponential(k, *popt)
        ss_res_exp = np.sum((g - g_pred) ** 2)
        results['exponential'] = {
            'params': popt.tolist(),
            'ss_res': float(ss_res_exp),
            'n_params': 3
        }
    except Exception:
        results['exponential'] = {'params': [0, 0, 0], 'ss_res': float('inf'), 'n_params': 3}

    # Model selection via BIC
    ss_total = np.sum((g - np.mean(g)) ** 2)
    bic = {}
    r_squared = {}

    for model_name, r in results.items():
        kp = r['n_params']
        ss = r['ss_res']
        if ss > 0 and ss_total > 0:
            bic[model_name] = n * np.log(ss / n) + kp * np.log(n)
            r_squared[model_name] = 1 - ss / ss_total
        else:
            bic[model_name] = float('inf')
            r_squared[model_name] = 0

    best_model = min(bic, key=bic.get)

    classification_map = {
        'constant': 'regular',
        'linear': 'context-free',
        'exponential': 'context-sensitive'
    }

    # Confidence from BIC difference
    bic_sorted = sorted(bic.values())
    if len(bic_sorted) >= 2 and bic_sorted[0] != float('inf'):
        bic_diff = bic_sorted[1] - bic_sorted[0]
        confidence = 1 - np.exp(-bic_diff / 2)
    else:
        confidence = 0.5

    # Additional checks
    # If gap is consistently near 0, grammar is minimal (billboard)
    if np.mean(g) < 0.05:
        classification = 'minimal'
        note = 'Compositionality gap near zero — grammar is minimal'
    # If gap is consistently near 1, grammar is maximal (no compositionality)
    elif np.mean(g) > 0.9:
        classification = 'context-sensitive'
        note = 'Compositionality gap near 1 — strongly non-compositional'
    else:
        classification = classification_map.get(best_model, 'unknown')
        note = f'Best BIC model: {best_model}'

    return {
        'classification': classification,
        'best_bic_model': best_model,
        'confidence': float(confidence),
        'bic': {k: float(v) for k, v in bic.items()},
        'r_squared': {k: float(v) for k, v in r_squared.items()},
        'fit_results': results,
        'mean_gap': float(np.mean(g)),
        'gap_range': [float(np.min(g)), float(np.max(g))],
        'note': note,
    }
