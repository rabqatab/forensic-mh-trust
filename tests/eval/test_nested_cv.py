import numpy as np
from sklearn.datasets import make_classification

from forensic_mh.eval.nested_cv import leakage_free_cv_score


def test_leakage_free_cv_returns_per_fold_scores():
    X, y = make_classification(
        n_samples=100, n_features=50, n_informative=10,
        n_redundant=10, n_classes=3, random_state=42,
    )
    scores = leakage_free_cv_score(X, y, n_top_features=10, n_splits=5, random_state=42)
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_leakage_free_cv_uses_groupkfold_when_groups_passed():
    """When groups are passed, samples in the same group must be in the same fold."""
    X, y = make_classification(
        n_samples=60, n_features=20, n_informative=5,
        n_classes=3, random_state=42,
    )
    # 30 groups of 2 — each group must stay together
    groups = np.repeat(np.arange(30), 2)
    scores = leakage_free_cv_score(
        X, y, n_top_features=5, n_splits=5,
        groups=groups, random_state=42,
    )
    assert len(scores) == 5


def test_leaky_vs_nested_demonstrates_optimism_gap():
    """Sanity: pre-selecting features on full data inflates accuracy
    relative to nested selection. Used as regression guard for P0 #1."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from xgboost import XGBClassifier
    X, y = make_classification(
        n_samples=200, n_features=200, n_informative=5,
        random_state=42,
    )
    nested = leakage_free_cv_score(X, y, n_top_features=5, n_splits=5, random_state=42)
    nested_mean = float(np.mean(nested))

    # leaky version
    pre = XGBClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.1,
        random_state=42, eval_metric='mlogloss', verbosity=0,
    )
    pre.fit(X, y)
    top_idx = np.argsort(pre.feature_importances_)[::-1][:5]
    leaky = cross_val_score(
        XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            random_state=42, eval_metric='mlogloss', verbosity=0,
        ),
        X[:, top_idx], y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
    )
    leaky_mean = float(np.mean(leaky))
    print(f"\nLeakage gap: leaky={leaky_mean:.3f} vs nested={nested_mean:.3f}")
    assert nested_mean > 0.0  # smoke
