"""Open-set recognition utilities.

OSR signal couples to conformal: an EMPTY prediction set ⇒ unknown. A
max-softmax-probability (MSP) baseline is provided for comparison and as an
orthogonal, tunable second signal.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def msp_score(probs: np.ndarray) -> np.ndarray:
    """OOD-ness score = 1 - max softmax prob (higher = more OOD-like)."""
    return 1.0 - np.max(probs, axis=1)


def ood_auroc(scores_in: np.ndarray, scores_ood: np.ndarray) -> float:
    y = np.concatenate([np.zeros(len(scores_in)), np.ones(len(scores_ood))])
    s = np.concatenate([scores_in, scores_ood])
    return float(roc_auc_score(y, s))


def fpr_at_tpr(scores_in: np.ndarray, scores_ood: np.ndarray, tpr: float = 0.95) -> float:
    """FPR (in-dist wrongly flagged OOD) at the threshold achieving `tpr`
    on true OOD samples."""
    thr = np.quantile(scores_ood, 1.0 - tpr)
    return float(np.mean(scores_in >= thr))


def reject_rate(sets: list[list[int]]) -> float:
    return float(np.mean([len(s) == 0 for s in sets])) if sets else 0.0


def open_set_decision(pred_set: list[int], max_prob: float,
                      msp_threshold: float):
    """Decide accept-with-set vs reject-as-unknown for one sample.

    Inputs:
      pred_set      : the conformal prediction set (possibly empty)
      max_prob      : the model's top class probability for this sample
      msp_threshold : MSP cutoff in [0, 1]; below it the sample looks unknown

    Return either the string "reject" or the accepted prediction set (list[int]).

    TODO(you): decide how the two signals combine. Consider:
      - Empty conformal set ⇒ reject (distribution-free, but α-coupled).
      - max_prob < msp_threshold ⇒ also reject? (catches confident-looking but
        diffuse cases the set may still populate at small α.)
      - Do you reject if EITHER fires (safer, more false rejects) or only if
        the set is empty (cleaner theory, ignores MSP)?
    Forensic framing: a false *accept* of an unknown-population sample is the
    costly error in court. That argues for the more conservative rule — but
    over-rejecting destroys utility. Your call shapes the OSR operating point.
    """
    # TODO(you): 3-6 lines implementing your chosen combination.
    raise NotImplementedError
