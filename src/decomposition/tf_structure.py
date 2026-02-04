"""
TF structural class as grammar predictor.

Maps motifs to TF structural families and tests whether
structural class predicts grammar type.
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

TF_STRUCTURAL_CLASSES = {
    'bHLH': ['MYC', 'MAX', 'USF1', 'TCF3', 'TWIST1', 'HAND2', 'MYOD1', 'NEUROD', 'HEY', 'HES'],
    'bZIP': ['JUN', 'FOS', 'CREB1', 'ATF1', 'BATF', 'CEBPA', 'CEBPB', 'BACH', 'MAF', 'NFE2'],
    'Zinc_finger_C2H2': ['SP1', 'KLF4', 'EGR1', 'CTCF', 'ZNF143', 'REST', 'SNAI', 'ZIC'],
    'Homeodomain': ['POU5F1', 'NANOG', 'HOX', 'PAX6', 'CDX2', 'OTX2', 'DLX', 'LHX', 'PBX'],
    'ETS': ['ETS1', 'ELF1', 'GABPA', 'SPI1', 'ERG', 'FLI1', 'ETV', 'ELK'],
    'GATA': ['GATA1', 'GATA2', 'GATA3', 'GATA4', 'GATA6'],
    'Nuclear_receptor': ['ESR1', 'AR', 'PPARG', 'RXRA', 'NR3C1', 'RARA', 'VDR', 'HNF4'],
    'Forkhead': ['FOXA1', 'FOXA2', 'FOXP3', 'FOXO1', 'FOXD', 'FOXL'],
    'HMG': ['SOX2', 'SOX9', 'SOX17', 'LEF1', 'TCF7', 'SRY'],
    'Rel': ['RELA', 'NFKB1', 'REL', 'NFKB2', 'NFAT'],
    'STAT': ['STAT1', 'STAT3', 'STAT5A', 'STAT5B'],
    'IRF': ['IRF1', 'IRF3', 'IRF4', 'IRF8'],
}


def assign_structural_class(motif_name: str) -> str:
    """Map a motif name to its TF structural class."""
    name = motif_name.upper()
    for cls, members in TF_STRUCTURAL_CLASSES.items():
        for member in members:
            if member in name:
                return cls
    return 'Unknown'


def build_structure_grammar_map(rules_df: pd.DataFrame) -> pd.DataFrame:
    """Build mapping from TF structural class pairs to grammar types."""
    rules = rules_df.copy()
    rules['class_a'] = rules['motif_a'].apply(assign_structural_class)
    rules['class_b'] = rules['motif_b'].apply(assign_structural_class)

    # Canonical pair name
    rules['class_pair'] = rules.apply(
        lambda r: '_'.join(sorted([r['class_a'], r['class_b']])), axis=1
    )

    return rules.groupby('class_pair').agg(
        mean_spacing_sensitivity=('spacing_sensitivity', 'mean'),
        std_spacing_sensitivity=('spacing_sensitivity', 'std'),
        mean_orientation_sensitivity=('orientation_sensitivity', 'mean'),
        mean_helical_phase=('helical_phase_score', 'mean'),
        mean_fold_change=('fold_change', 'mean'),
        n_rules=('pair', 'count')
    ).reset_index()


def test_structure_predicts_grammar(rules_df: pd.DataFrame) -> dict:
    """Test whether TF structural class predicts grammar type."""
    rules = rules_df.copy()
    rules['class_a'] = rules['motif_a'].apply(assign_structural_class)
    rules['class_b'] = rules['motif_b'].apply(assign_structural_class)

    # Remove unknown
    rules = rules[(rules['class_a'] != 'Unknown') & (rules['class_b'] != 'Unknown')]

    if len(rules) < 50:
        return {'error': 'Too few rules with known classes', 'n_rules': len(rules)}

    # Classify grammar type
    sp_med = rules['spacing_sensitivity'].median()
    or_med = rules['orientation_sensitivity'].median()
    sp_q25 = rules['spacing_sensitivity'].quantile(0.25)
    or_q25 = rules['orientation_sensitivity'].quantile(0.25)

    def classify(row):
        if row.get('helical_phase_score', 0) > 3:
            return 'helical_phased'
        elif row['spacing_sensitivity'] > sp_med and row['spacing_sensitivity'] > row['orientation_sensitivity']:
            return 'spacing_dependent'
        elif row['orientation_sensitivity'] > or_med:
            return 'orientation_dependent'
        elif row['spacing_sensitivity'] < sp_q25 and row['orientation_sensitivity'] < or_q25:
            return 'insensitive'
        else:
            return 'mixed'

    rules['grammar_type'] = rules.apply(classify, axis=1)

    # Encode features
    le_a = LabelEncoder().fit(rules['class_a'])
    le_b = LabelEncoder().fit(rules['class_b'])
    X = np.column_stack([le_a.transform(rules['class_a']),
                          le_b.transform(rules['class_b'])])

    le_g = LabelEncoder().fit(rules['grammar_type'])
    y = le_g.transform(rules['grammar_type'])

    # Random forest
    clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

    baseline = Counter(y).most_common(1)[0][1] / len(y)

    return {
        'accuracy': float(scores.mean()),
        'accuracy_std': float(scores.std()),
        'baseline_accuracy': float(baseline),
        'improvement': float(scores.mean() - baseline),
        'n_rules': len(rules),
        'grammar_type_dist': dict(Counter(rules['grammar_type'])),
        'structural_classes': list(set(rules['class_a'].unique()) | set(rules['class_b'].unique())),
    }
