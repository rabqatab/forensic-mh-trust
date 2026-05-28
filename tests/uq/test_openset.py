import numpy as np
import pytest

from forensic_mh.uq.openset import (
    fpr_at_tpr,
    msp_score,
    ood_auroc,
    open_set_decision,
    reject_rate,
)


def test_msp_score_is_one_minus_max_prob():
    probs = np.array([[0.7, 0.2, 0.1], [0.34, 0.33, 0.33]])
    np.testing.assert_allclose(msp_score(probs), [0.3, 0.66])


def test_ood_auroc_perfect_separation():
    s_in = np.array([0.0, 0.1, 0.2])
    s_ood = np.array([0.8, 0.9, 1.0])
    assert ood_auroc(s_in, s_ood) == pytest.approx(1.0)


def test_fpr_at_95_tpr_perfect():
    s_in = np.array([0.0, 0.1, 0.2])
    s_ood = np.array([0.8, 0.9, 1.0])
    assert fpr_at_tpr(s_in, s_ood, tpr=0.95) == pytest.approx(0.0)


def test_reject_rate_counts_empty_sets():
    sets = [[0], [], [1, 2], []]
    assert reject_rate(sets) == pytest.approx(0.5)


# --- open_set_decision: YOUR contribution (Task 4 Step 3b) ---
# These two endpoints are the spec your reject rule must satisfy. They are
# xfail until you implement open_set_decision (remove the markers when done).

@pytest.mark.xfail(reason="awaiting user implementation of open_set_decision", strict=False)
def test_open_set_decision_rejects_empty_set():
    # empty conformal set must reject regardless of MSP
    d = open_set_decision(pred_set=[], max_prob=0.99, msp_threshold=0.0)
    assert d == "reject"


@pytest.mark.xfail(reason="awaiting user implementation of open_set_decision", strict=False)
def test_open_set_decision_accepts_confident_nonempty():
    d = open_set_decision(pred_set=[1], max_prob=0.9, msp_threshold=0.5)
    assert d == [1]
