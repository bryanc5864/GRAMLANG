"""
Redesigned Cross-Species Transfer: Distributional Comparison.

Instead of testing whether specific rules transfer (which fails because
species don't share TFs), tests whether abstract grammar *properties*
are conserved: spacing distributions, orientation preferences, helical
phasing rates, grammar type distributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import wasserstein_distance, ks_2samp, entropy
from scipy.spatial.distance import jensenshannon


def compute_distributional_transfer(
    rules_df: pd.DataFrame,
    gsi_results: pd.DataFrame,
    species_map: Dict[str, List[str]],
) -> Dict:
    """
    Compare grammar property distributions across species.

    Tests grammar *universals* rather than specific rule transfer:
    - Spacing sensitivity distributions
    - Orientation effect distributions
    - Helical phasing rate
    - Grammar type proportions
    - GSI distributions

    Returns dict with pairwise species comparisons.
    """
    # Map datasets to species
    dataset_to_species = {}
    for species, datasets in species_map.items():
        for ds in datasets:
            dataset_to_species[ds] = species

    # Add species column
    rules_with_species = rules_df.copy()
    rules_with_species['species'] = rules_with_species['dataset'].map(dataset_to_species)
    rules_with_species = rules_with_species.dropna(subset=['species'])

    species_list = sorted(rules_with_species['species'].unique())

    if len(species_list) < 2:
        return {'error': 'Need at least 2 species for distributional transfer'}

    # Extract per-species distributions
    species_distributions = {}
    for species in species_list:
        sp_rules = rules_with_species[rules_with_species['species'] == species]
        sp_gsi = gsi_results[gsi_results['dataset'].isin(
            [ds for ds, sp in dataset_to_species.items() if sp == species]
        )]

        species_distributions[species] = {
            'spacing_sensitivity': sp_rules['spacing_sensitivity'].values,
            'orientation_sensitivity': sp_rules['orientation_sensitivity'].values,
            'helical_phase_score': sp_rules['helical_phase_score'].values,
            'fold_change': sp_rules['fold_change'].values,
            'gsi': sp_gsi['gsi'].values if len(sp_gsi) > 0 else np.array([]),
            'n_rules': len(sp_rules),
            'n_gsi': len(sp_gsi),
        }

        # Grammar type proportions
        types = _classify_rule_types(sp_rules)
        species_distributions[species]['grammar_types'] = types

    # Pairwise comparisons
    comparisons = {}
    for i, sp1 in enumerate(species_list):
        for j, sp2 in enumerate(species_list):
            if i >= j:
                continue

            d1 = species_distributions[sp1]
            d2 = species_distributions[sp2]
            pair_key = f"{sp1}_vs_{sp2}"

            comparison = {}

            # Compare each distribution
            for prop in ['spacing_sensitivity', 'orientation_sensitivity',
                        'helical_phase_score', 'fold_change', 'gsi']:
                v1 = d1[prop]
                v2 = d2[prop]
                if len(v1) < 5 or len(v2) < 5:
                    continue

                # Wasserstein distance (Earth Mover's Distance)
                emd = wasserstein_distance(v1, v2)

                # KS test
                ks_stat, ks_pval = ks_2samp(v1, v2)

                # Effect size: difference in means / pooled std
                pooled_std = np.sqrt(
                    (np.var(v1) * len(v1) + np.var(v2) * len(v2)) /
                    (len(v1) + len(v2))
                )
                cohens_d = abs(np.mean(v1) - np.mean(v2)) / max(pooled_std, 1e-10)

                comparison[prop] = {
                    'mean_1': float(np.mean(v1)),
                    'mean_2': float(np.mean(v2)),
                    'std_1': float(np.std(v1)),
                    'std_2': float(np.std(v2)),
                    'wasserstein_distance': float(emd),
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pval),
                    'cohens_d': float(cohens_d),
                }

            # Compare grammar type proportions (Jensen-Shannon divergence)
            types1 = d1['grammar_types']
            types2 = d2['grammar_types']
            all_types = sorted(set(list(types1.keys()) + list(types2.keys())))
            p1 = np.array([types1.get(t, 0) for t in all_types], dtype=float)
            p2 = np.array([types2.get(t, 0) for t in all_types], dtype=float)
            if p1.sum() > 0 and p2.sum() > 0:
                p1 /= p1.sum()
                p2 /= p2.sum()
                js_div = float(jensenshannon(p1, p2))
            else:
                js_div = 1.0

            comparison['grammar_type_js_divergence'] = js_div
            comparison['grammar_types_1'] = {k: float(v) for k, v in types1.items()}
            comparison['grammar_types_2'] = {k: float(v) for k, v in types2.items()}

            comparisons[pair_key] = comparison

    # Overall summary
    summary = {
        'species': species_list,
        'n_species': len(species_list),
        'species_distributions': {
            sp: {
                'n_rules': d['n_rules'],
                'n_gsi': d['n_gsi'],
                'mean_spacing_sensitivity': float(d['spacing_sensitivity'].mean()) if len(d['spacing_sensitivity']) > 0 else 0,
                'mean_fold_change': float(d['fold_change'].mean()) if len(d['fold_change']) > 0 else 0,
                'helical_phasing_rate': float((d['helical_phase_score'] > 2.0).mean()) if len(d['helical_phase_score']) > 0 else 0,
                'mean_gsi': float(d['gsi'].mean()) if len(d['gsi']) > 0 else 0,
                'grammar_types': {k: float(v) for k, v in d['grammar_types'].items()},
            }
            for sp, d in species_distributions.items()
        },
        'pairwise_comparisons': comparisons,
    }

    # Compute overall conservation score
    # Average KS p-value across all property comparisons
    all_pvals = []
    all_cohens_d = []
    for pair_comp in comparisons.values():
        for prop in ['spacing_sensitivity', 'orientation_sensitivity', 'helical_phase_score', 'fold_change']:
            if prop in pair_comp:
                all_pvals.append(pair_comp[prop]['ks_pvalue'])
                all_cohens_d.append(pair_comp[prop]['cohens_d'])

    if all_pvals:
        summary['mean_ks_pvalue'] = float(np.mean(all_pvals))
        summary['mean_cohens_d'] = float(np.mean(all_cohens_d))
        summary['grammar_properties_conserved'] = float(np.mean([p > 0.05 for p in all_pvals]))
    else:
        summary['grammar_properties_conserved'] = 0

    return summary


def _classify_rule_types(rules: pd.DataFrame) -> Dict[str, float]:
    """Classify grammar rules into types based on their properties."""
    types = {
        'orientation_dependent': 0,
        'spacing_dependent': 0,
        'helical_phased': 0,
        'mixed': 0,
        'insensitive': 0,
    }

    for _, rule in rules.iterrows():
        orient_sens = rule.get('orientation_sensitivity', 0)
        spacing_sens = rule.get('spacing_sensitivity', 0)
        helical = rule.get('helical_phase_score', 0)

        if helical > 2.0:
            types['helical_phased'] += 1
        elif orient_sens > spacing_sens * 1.5:
            types['orientation_dependent'] += 1
        elif spacing_sens > orient_sens * 1.5:
            types['spacing_dependent'] += 1
        elif orient_sens > 0.5 and spacing_sens > 0.5:
            types['mixed'] += 1
        else:
            types['insensitive'] += 1

    # Normalize
    total = sum(types.values())
    if total > 0:
        types = {k: v / total for k, v in types.items()}

    return types
