import numpy as np

from forensic_mh.fm.sklearn_api import ForensicFMClassifier
from forensic_mh.uq.conformal_classifier import ConformalClassifier


def _data():
    rng = np.random.default_rng(0)
    N, M, K = 120, 5, 4
    X = rng.integers(0, K, size=(N, M))
    y = rng.integers(0, 3, size=N)
    X[:, 0] = y                      # learnable signal
    return X, y


def test_predict_proba_shape_and_normalised():
    X, y = _data()
    clf = ForensicFMClassifier(k=4, d_model=16, n_layers=1, n_heads=2,
                               pretrain_epochs=2, finetune_epochs=10, seed=0).fit(X, y)
    p = clf.predict_proba(X)
    assert p.shape == (len(X), 3)
    assert np.allclose(p.sum(1), 1.0, atol=1e-5)


def test_embed_returns_fixed_width_vectors():
    X, y = _data()
    clf = ForensicFMClassifier(k=4, d_model=16, n_layers=1, n_heads=2,
                               pretrain_epochs=1, finetune_epochs=2, seed=0).fit(X, y)
    assert clf.embed(X).shape == (len(X), 16)


def test_conformal_classifier_wraps_fm_unchanged():
    X, y = _data()
    base = ForensicFMClassifier(k=4, d_model=16, n_layers=1, n_heads=2,
                                pretrain_epochs=1, finetune_epochs=10, seed=0)
    cc = ConformalClassifier(base, alpha=0.1, mondrian=True).fit(X, y)
    sets = cc.predict_set(X[:10])
    assert len(sets) == 10 and all(isinstance(s, list) for s in sets)
