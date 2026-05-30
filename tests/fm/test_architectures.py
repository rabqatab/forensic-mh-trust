"""Shape/contract tests for the diverse DL architectures."""
import numpy as np
import pytest

from forensic_mh.fm.architectures import TorchArchClassifier


@pytest.mark.parametrize("arch", ["embmlp", "cnn1d", "supae", "fttransformer", "resnettab", "rescnn"])
def test_fit_predict_proba_shape_and_normalization(arch):
    rng = np.random.RandomState(0)
    M, n, k, n_classes = 30, 50, 8, 4
    X = rng.randint(0, k, size=(n, M))
    y = rng.randint(0, n_classes, size=n)
    clf = TorchArchClassifier(arch=arch, k=k, d=16, epochs=6, patience=4, device="cpu").fit(X, y)
    Xte = rng.randint(0, k, size=(7, M))
    proba = clf.predict_proba(Xte)
    assert proba.shape == (7, n_classes)
    assert np.allclose(proba.sum(1), 1.0, atol=1e-4)
    assert set(clf.predict(Xte)).issubset(set(np.unique(y)))


def test_codes_at_vocab_upper_bound_do_not_index_error():
    # code == k-1 (the FMVocab OTHER slot) must be a valid embedding index
    rng = np.random.RandomState(1)
    M, n, k = 12, 24, 8
    X = np.full((n, M), k - 1)
    X[:, : M // 2] = rng.randint(0, k, size=(n, M // 2))
    y = rng.randint(0, 3, size=n)
    clf = TorchArchClassifier(arch="embmlp", k=k, d=8, epochs=3, patience=3, device="cpu").fit(X, y)
    assert clf.predict_proba(X).shape == (n, 3)
