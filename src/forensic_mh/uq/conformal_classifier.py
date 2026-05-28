"""Model-agnostic conformal wrapper.

Wraps any fitted-able estimator exposing predict_proba. The SAME class wraps
the XGBoost baseline (Plan 2) and the SSL foundation model (Plan 3): expose a
predict_proba on the FM and this layer is unchanged.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from forensic_mh.uq.conformal import (
    build_prediction_sets,
    conformal_quantile,
    mondrian_quantiles,
)


class ConformalClassifier:
    def __init__(self, base_estimator, alpha: float = 0.1,
                 mondrian: bool = True, cal_size: float = 0.3,
                 random_state: int = 42):
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.mondrian = mondrian
        self.cal_size = cal_size
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X, y, test_size=self.cal_size,
            stratify=y, random_state=self.random_state,
        )
        self.model_ = clone(self.base_estimator).fit(X_tr, y_tr)
        cal_probs = self.model_.predict_proba(X_cal)
        cal_true_scores = 1.0 - cal_probs[np.arange(len(y_cal)), y_cal]
        if self.mondrian:
            self.quantiles_ = mondrian_quantiles(
                cal_true_scores, y_cal, self.n_classes_, self.alpha)
        else:
            q = conformal_quantile(cal_true_scores, self.alpha)
            self.quantiles_ = {k: q for k in range(self.n_classes_)}
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(np.asarray(X))

    def predict_set(self, X) -> list[list[int]]:
        return build_prediction_sets(self.predict_proba(X), self.quantiles_)
