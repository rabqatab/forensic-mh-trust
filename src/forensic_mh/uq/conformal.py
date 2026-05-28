"""Split-conformal prediction (LAC score) with Mondrian per-class option.

Pure numpy. Nonconformity of a label k at x is s = 1 - p_hat(k|x).
"""
from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Rigorous split-conformal threshold = k-th smallest score,
    k = ⌈(n+1)(1-alpha)⌉.

    Use the order statistic directly — NOT np.quantile, whose (n-1)-spacing
    convention shifts the threshold by one position for small n and can drop
    coverage below 1-alpha. Empty calibration or k>n → 1.0 (scores ∈ [0,1],
    so a threshold of 1.0 includes every label; maximally conservative)."""
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return 1.0
    return float(np.sort(scores)[k - 1])


def mondrian_quantiles(
    cal_true_scores: np.ndarray,
    cal_labels: np.ndarray,
    n_classes: int,
    alpha: float,
) -> dict[int, float]:
    """Per-class quantile from calibration points grouped by TRUE label.

    cal_true_scores[i] = 1 - p_hat(y_i | x_i)."""
    cal_true_scores = np.asarray(cal_true_scores, dtype=float)
    cal_labels = np.asarray(cal_labels)
    return {
        k: conformal_quantile(cal_true_scores[cal_labels == k], alpha)
        for k in range(n_classes)
    }


def build_prediction_sets(
    probs: np.ndarray,
    quantiles: dict[int, float],
) -> list[list[int]]:
    """For each row, include label k iff (1 - p[k]) <= quantiles[k]."""
    sets: list[list[int]] = []
    for row in probs:
        s = [k for k in range(row.shape[0]) if (1.0 - row[k]) <= quantiles[k]]
        sets.append(s)
    return sets


def empirical_coverage(sets: list[list[int]], y_true: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    return float(np.mean([y in s for s, y in zip(sets, y_true)]))


def class_conditional_coverage(
    sets: list[list[int]],
    y_true: np.ndarray,
    n_classes: int,
) -> dict[int, float]:
    y_true = np.asarray(y_true)
    out: dict[int, float] = {}
    for k in range(n_classes):
        mask = y_true == k
        if not mask.any():
            out[k] = float("nan")
            continue
        out[k] = float(np.mean([k in s for s, m in zip(sets, mask) if m]))
    return out


def mean_set_size(sets: list[list[int]]) -> float:
    return float(np.mean([len(s) for s in sets])) if sets else 0.0
