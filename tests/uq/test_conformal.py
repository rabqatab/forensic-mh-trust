import numpy as np
import pytest

from forensic_mh.uq.conformal import (
    build_prediction_sets,
    class_conditional_coverage,
    conformal_quantile,
    empirical_coverage,
    mean_set_size,
    mondrian_quantiles,
)


def test_quantile_is_kth_order_statistic():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    # alpha=0.5 → k = ceil((4+1)*0.5) = 3 → 3rd smallest = 0.3
    assert conformal_quantile(scores, alpha=0.5) == pytest.approx(0.3)


def test_quantile_empty_calibration_returns_one():
    # no calibration info → include everything (conservative)
    assert conformal_quantile(np.array([]), alpha=0.1) == 1.0


def test_quantile_alpha_too_small_for_n_returns_one():
    # n=3, alpha=0.05 → k = ceil(4*0.95) = 4 > n → 1.0 (include all)
    assert conformal_quantile(np.array([0.1, 0.2, 0.3]), alpha=0.05) == 1.0


def test_marginal_coverage_meets_target_on_calibrated_probs():
    rng = np.random.default_rng(0)
    n, K = 4000, 3
    y = rng.integers(0, K, size=n)
    logits = rng.normal(size=(n, K))
    logits[np.arange(n), y] += 1.5
    probs = np.exp(logits)
    probs /= probs.sum(1, keepdims=True)
    cal, te = slice(0, n // 2), slice(n // 2, n)
    true_scores = 1 - probs[cal][np.arange(n // 2), y[cal]]
    q = conformal_quantile(true_scores, alpha=0.1)
    sets = build_prediction_sets(probs[te], {k: q for k in range(K)})
    cov = empirical_coverage(sets, y[te])
    assert cov >= 0.88  # ~0.90 target, finite-sample slack


def test_mondrian_gives_per_class_coverage():
    rng = np.random.default_rng(1)
    n, K = 6000, 3
    y = rng.integers(0, K, size=n)
    logits = rng.normal(size=(n, K))
    logits[np.arange(n), y] += 1.2
    probs = np.exp(logits)
    probs /= probs.sum(1, keepdims=True)
    cal, te = slice(0, n // 2), slice(n // 2, n)
    cal_true_scores = 1 - probs[cal][np.arange(n // 2), y[cal]]
    q = mondrian_quantiles(cal_true_scores, y[cal], n_classes=K, alpha=0.1)
    sets = build_prediction_sets(probs[te], q)
    per_class = class_conditional_coverage(sets, y[te], n_classes=K)
    for k in range(K):
        assert per_class[k] >= 0.85


def test_mean_set_size_between_zero_and_k():
    sets = [[0], [0, 1], [], [2]]
    assert mean_set_size(sets) == pytest.approx((1 + 2 + 0 + 1) / 4)
