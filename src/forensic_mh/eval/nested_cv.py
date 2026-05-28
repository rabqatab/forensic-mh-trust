"""Nested cross-validation with leak-free feature selection.

P0 fix from review #1: the original proposal used full-data feature importance
ranking, then cross_val_score on the top-N — this leaks test labels into
selection. We re-select features within each outer fold.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def leakage_free_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    n_top_features: int,
    n_splits: int = 5,
    groups: Optional[np.ndarray] = None,
    random_state: int = 42,
    xgb_kwargs: Optional[dict] = None,
) -> list[float]:
    """Run nested CV: per-fold feature selection (inner) + scoring (outer).

    For each outer fold:
      1. Fit XGBoost on train fold only.
      2. Select top-N features by importance from that train-fold model.
      3. Refit XGBoost on train fold using only those features.
      4. Score on test fold with the same feature subset.

    Args:
        X: (n_samples, n_features)
        y: (n_samples,) integer labels
        n_top_features: how many features to select per fold
        n_splits: outer fold count
        groups: optional group labels for GroupKFold (relatedness)
        random_state: reproducibility
        xgb_kwargs: passed to XGBClassifier

    Returns:
        list[float] of per-fold accuracy scores
    """
    if xgb_kwargs is None:
        xgb_kwargs = dict(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=random_state, eval_metric="mlogloss", verbosity=0,
        )
    splitter = (
        GroupKFold(n_splits=n_splits)
        if groups is not None
        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )
    split_args = (X, y, groups) if groups is not None else (X, y)

    scores: list[float] = []
    for train_idx, test_idx in splitter.split(*split_args):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # inner: feature selection on train fold only
        selector = XGBClassifier(**xgb_kwargs)
        selector.fit(X_tr, y_tr)
        importances = selector.feature_importances_
        top_idx = np.argsort(importances)[::-1][:n_top_features]

        # outer: refit + score on selected features
        clf = XGBClassifier(**xgb_kwargs)
        clf.fit(X_tr[:, top_idx], y_tr)
        pred = clf.predict(X_te[:, top_idx])
        scores.append(float(accuracy_score(y_te, pred)))
    return scores
