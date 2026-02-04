"""Statistical testing framework for GRAMLANG."""

import numpy as np
from scipy import stats
from typing import Callable, Optional
from statsmodels.stats.multitest import multipletests


def permutation_test(
    observed_statistic: float,
    null_distribution: np.ndarray,
    alternative: str = 'greater'
) -> dict:
    """
    Standard permutation test.

    Args:
        observed_statistic: Test statistic on real data
        null_distribution: Test statistics under null hypothesis
        alternative: 'greater', 'less', or 'two-sided'

    Returns:
        Dict with p_value, z_score, effect_size
    """
    null_distribution = np.asarray(null_distribution)

    if alternative == 'greater':
        p_value = np.mean(null_distribution >= observed_statistic)
    elif alternative == 'less':
        p_value = np.mean(null_distribution <= observed_statistic)
    else:
        p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_statistic))

    # Ensure p > 0 (use 1/n as floor)
    if p_value == 0:
        p_value = 1.0 / (len(null_distribution) + 1)

    null_mean = np.mean(null_distribution)
    null_std = np.std(null_distribution)
    z_score = (observed_statistic - null_mean) / max(null_std, 1e-10)

    return {
        'p_value': float(p_value),
        'z_score': float(z_score),
        'observed': float(observed_statistic),
        'null_mean': float(null_mean),
        'null_std': float(null_std),
        'effect_size': float(z_score),
        'n_permutations': len(null_distribution)
    }


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn: Callable,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> dict:
    """Compute bootstrap confidence interval for any statistic."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data)

    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_fn(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return {
        'point_estimate': float(statistic_fn(data)),
        'lower': float(lower),
        'upper': float(upper),
        'confidence': confidence,
        'se': float(np.std(bootstrap_stats)),
        'n_bootstrap': n_bootstrap
    }


def correct_pvalues(p_values: np.ndarray, method: str = 'fdr_bh',
                    alpha: float = 0.05) -> tuple:
    """
    Apply multiple testing correction.

    Args:
        p_values: Array of p-values
        method: Correction method ('fdr_bh', 'bonferroni', etc.)
        alpha: Significance level

    Returns:
        (corrected_p_values, rejected_mask)
    """
    p_values = np.asarray(p_values)

    # Handle edge cases
    if len(p_values) == 0:
        return np.array([]), np.array([], dtype=bool)
    if len(p_values) == 1:
        return p_values, p_values < alpha

    rejected, corrected_p, _, _ = multipletests(
        p_values, alpha=alpha, method=method
    )
    return corrected_p, rejected


def f_test_variance_ratio(var1: float, var2: float,
                          n1: int, n2: int) -> dict:
    """F-test for comparing two variances."""
    if var2 == 0:
        return {'f_stat': float('inf'), 'p_value': 0.0}

    f_stat = var1 / var2
    p_value = 1 - stats.f.cdf(f_stat, n1 - 1, n2 - 1)

    return {
        'f_stat': float(f_stat),
        'p_value': float(p_value),
        'df1': n1 - 1,
        'df2': n2 - 1
    }


def correlation_with_ci(x: np.ndarray, y: np.ndarray,
                        method: str = 'pearson',
                        n_bootstrap: int = 5000) -> dict:
    """Compute correlation with bootstrap confidence interval."""
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    if len(x) < 3:
        return {'r': 0.0, 'p_value': 1.0, 'ci_lower': 0.0, 'ci_upper': 0.0}

    if method == 'pearson':
        r, p = stats.pearsonr(x, y)
    elif method == 'spearman':
        r, p = stats.spearmanr(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot_rs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(x), size=len(x), replace=True)
        if method == 'pearson':
            br, _ = stats.pearsonr(x[idx], y[idx])
        else:
            br, _ = stats.spearmanr(x[idx], y[idx])
        boot_rs.append(br)

    boot_rs = np.array(boot_rs)

    return {
        'r': float(r),
        'p_value': float(p),
        'ci_lower': float(np.percentile(boot_rs, 2.5)),
        'ci_upper': float(np.percentile(boot_rs, 97.5)),
        'n': len(x)
    }
