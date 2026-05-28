import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from forensic_mh.uq.conformal_classifier import ConformalClassifier


def _data():
    X, y = make_classification(
        n_samples=900, n_features=20, n_informative=10,
        n_classes=3, random_state=0,
    )
    return X[:600], y[:600], X[600:], y[600:]


def test_fit_calibrate_then_predict_sets_cover_target():
    Xtr, ytr, Xte, yte = _data()
    cc = ConformalClassifier(LogisticRegression(max_iter=500), alpha=0.1, mondrian=True)
    cc.fit(Xtr, ytr)            # internally splits train vs calibration
    sets = cc.predict_set(Xte)
    cov = np.mean([y in s for s, y in zip(sets, yte)])
    assert cov >= 0.85          # ~0.90 target


def test_predict_set_returns_list_of_lists():
    Xtr, ytr, Xte, yte = _data()
    cc = ConformalClassifier(LogisticRegression(max_iter=500), alpha=0.2).fit(Xtr, ytr)
    sets = cc.predict_set(Xte[:5])
    assert len(sets) == 5
    assert all(isinstance(s, list) for s in sets)


def test_proba_passthrough_shape():
    Xtr, ytr, Xte, _ = _data()
    cc = ConformalClassifier(LogisticRegression(max_iter=500)).fit(Xtr, ytr)
    p = cc.predict_proba(Xte)
    assert p.shape == (len(Xte), 3)
