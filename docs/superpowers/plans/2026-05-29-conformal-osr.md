# Plan 2 — Conformal Prediction + Open-Set Recognition (Trust Layer) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a **model-agnostic trust layer** — distribution-free prediction sets via Mondrian (class-conditional) split-conformal prediction, plus open-set recognition that rejects unknown populations — and validate it on the leakage-free EAS MH baseline (Plan 1), using 1000G non-EAS superpopulations and leave-one-population-out as the "unknown" test sets. No external/Korean data required.

**Architecture:** The trust layer wraps *any* fitted classifier exposing `predict_proba` (XGBoost baseline now; the SSL foundation model in Plan 3 later — same interface). Conformal calibration is implemented directly in numpy (mapie 1.4 is installed but its API churns across versions; a ~40-line split-conformal core is version-proof and transparent). Open-set reject is tied to the conformal machinery: **an empty prediction set ⇒ "unknown"**, with a max-softmax-probability (MSP) baseline for comparison. Evaluation uses two complementary unknown sources: far-OOD (non-EAS superpopulations EUR/AFR/SAS/AMR) and near-OOD (leave-one-EAS-population-out).

**Tech Stack:**
- Python 3.11 (existing uv venv)
- `numpy`, `scikit-learn`, `xgboost`, `pandas` (already installed)
- `cyvcf2` for OOD-sample VCF extraction (reuse `forensic_mh.data.vcf_io`)
- `pytest` (TDD)
- mapie 1.4.0 present — used only as an optional cross-check, never load-bearing

**Why this is not throwaway work:** the conformal + OSR layer is built against `predict_proba`, so the identical layer wraps the Plan 3 SSL foundation model. Plan 2's output *is* the trustworthiness half of the final Trustworthy Forensic Model.

---

## Background: the conformal math (so the implementer needs zero prior context)

**Split-conformal, LAC ("least ambiguous") score.** Hold out a calibration set the model never trained on. For each calibration point `(xᵢ, yᵢ)` compute the nonconformity of its *true* label:

```
sᵢ = 1 − p̂(yᵢ | xᵢ)
```

**Marginal quantile** at miscoverage `α` — the rigorous conformal threshold is an **order statistic**, not a `np.quantile` call (the latter's `(n-1)`-spacing convention is off by one position for small `n` and can break the guarantee):

```
q = the k-th smallest of {sᵢ},  where  k = ⌈(n+1)(1−α)⌉
    if k > n  (α too small for this calibration size) → q = 1.0, i.e. include every label
```

**Prediction set** for a new `x`:

```
C(x) = { k : 1 − p̂(k | x) ≤ q }  =  { k : p̂(k | x) ≥ 1 − q }
```

This guarantees marginal coverage `P(Y ∈ C(X)) ≥ 1 − α` (finite-sample, distribution-free, exchangeability only).

**Mondrian / class-conditional.** Compute a separate quantile `q_k` from calibration points whose *true* label is `k`. Test candidate label `k` with its own `q_k`:

```
C(x) = { k : 1 − p̂(k | x) ≤ q_k }
```

This upgrades the guarantee to **per-class**: `P(Y ∈ C(X) | Y = k) ≥ 1 − α` for every `k` — important in forensics where you must not silently under-cover a minority population.

**Open-set via empty sets.** If `C(x) = ∅`, no label is plausible at confidence `1−α` ⇒ declare **unknown/reject**. Larger `α` ⇒ smaller sets ⇒ more empties ⇒ more rejects: the accept/reject behaviour is `α`-controlled, giving a tunable OSR operating curve "for free."

---

## File Structure

```
mh-eas-panel/
├── src/forensic_mh/
│   ├── data/
│   │   └── encoding.py            # NEW — shared diplotype→int encoder (aligns EAS/OOD codes)
│   ├── uq/                        # NEW package — the trust layer
│   │   ├── __init__.py            # NEW
│   │   ├── conformal.py           # NEW — quantiles, prediction sets, coverage metrics (pure fns)
│   │   ├── conformal_classifier.py# NEW — model-agnostic wrapper (fit/calibrate/predict_set)
│   │   └── openset.py             # NEW — MSP score, reject decision (USER contribution), OOD metrics
│   └── eval/
│       └── lopo.py                # NEW — leave-one-population-out splits
├── scripts/
│   ├── 10_conformal_curve.py      # NEW — coverage vs set-size curve on EAS MH
│   ├── 11_openset_ood.py          # NEW — non-EAS far-OOD extraction + reject eval
│   └── 12_lopo_nearood.py         # NEW — leave-one-EAS-pop-out near-OOD eval
├── tests/
│   ├── data/test_encoding.py      # NEW
│   ├── uq/__init__.py             # NEW
│   ├── uq/test_conformal.py       # NEW
│   ├── uq/test_conformal_classifier.py  # NEW
│   ├── uq/test_openset.py         # NEW
│   └── eval/test_lopo.py          # NEW
└── results/conformal/             # gitignored — curves, OOD metrics, LOPO tables
```

**Responsibility boundaries:**
- `uq/conformal.py`: pure functions, numpy only — no model, no I/O. Easiest to test exhaustively.
- `uq/conformal_classifier.py`: orchestration around a base estimator; the public surface Plan 3 reuses.
- `uq/openset.py`: reject logic + OOD scoring metrics.
- `data/encoding.py`: the one piece of data plumbing Plan 2 needs (consistent codes across sample groups).

---

## Task 1: Shared diplotype encoder (aligns codes across EAS and OOD)

**Why first:** `pipelines.baseline.build_diplotype_matrix` fits a fresh `LabelEncoder` per column on whatever samples it's given, so an EAS-trained model can't be applied to non-EAS samples — the same diplotype string would get different integer codes. We need an encoder fit on **training diplotypes** that maps unseen strings (common in OOD) to a reserved code.

**Files:**
- Create: `src/forensic_mh/data/encoding.py`
- Test: `tests/data/test_encoding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_encoding.py
import numpy as np
from forensic_mh.data.encoding import DiplotypeEncoder


def test_encoder_assigns_stable_codes_to_seen_strings():
    enc = DiplotypeEncoder()
    train = [["A-T|G-C", "A-A|A-A"], ["A-T|G-C", "T-T|T-T"]]  # 2 samples, 2 markers
    X = enc.fit_transform(train)
    assert X.shape == (2, 2)
    # same string in column 0 → same code
    assert X[0, 0] == X[1, 0]


def test_encoder_maps_unseen_to_reserved_code():
    enc = DiplotypeEncoder(unseen_code=-1)
    enc.fit([["A-T|G-C"], ["A-T|G-C"]])
    X = enc.transform([["NOVEL|HAP"]])  # unseen in training
    assert X[0, 0] == -1


def test_encoder_unseen_fraction_reported():
    enc = DiplotypeEncoder()
    enc.fit([["A|A"], ["A|A"]])
    enc.transform([["A|A"], ["B|B"]])
    assert enc.last_unseen_fraction == 0.5  # one of two cells unseen
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/data/test_encoding.py -v`
Expected: ImportError (no `encoding` module)

- [ ] **Step 3: Write minimal implementation**

```python
# src/forensic_mh/data/encoding.py
"""Shared per-marker diplotype→int encoder.

Fit on TRAINING diplotype strings so the same string maps to the same code
across any sample group. Unseen strings (frequent in OOD / non-EAS samples)
map to a reserved code — itself an open-set signal."""
from __future__ import annotations
import numpy as np


class DiplotypeEncoder:
    def __init__(self, unseen_code: int = -1):
        self.unseen_code = unseen_code
        self.maps_: list[dict[str, int]] = []   # one dict per marker column
        self.last_unseen_fraction: float = 0.0

    def fit(self, rows: list[list[str]]) -> "DiplotypeEncoder":
        n_markers = len(rows[0])
        self.maps_ = []
        for j in range(n_markers):
            uniq = sorted({row[j] for row in rows})
            self.maps_.append({s: i for i, s in enumerate(uniq)})
        return self

    def transform(self, rows: list[list[str]]) -> np.ndarray:
        n, m = len(rows), len(self.maps_)
        X = np.empty((n, m), dtype=np.int64)
        unseen = 0
        for i, row in enumerate(rows):
            for j in range(m):
                code = self.maps_[j].get(row[j], self.unseen_code)
                if code == self.unseen_code:
                    unseen += 1
                X[i, j] = code
        self.last_unseen_fraction = unseen / (n * m) if n * m else 0.0
        return X

    def fit_transform(self, rows: list[list[str]]) -> np.ndarray:
        return self.fit(rows).transform(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_encoding.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/forensic_mh/data/encoding.py tests/data/test_encoding.py
git commit -m "feat(data): shared diplotype encoder with unseen-code (OOD-aware)"
```

---

## Task 2: Conformal core — quantiles, prediction sets, coverage (pure functions)

**Files:**
- Create: `src/forensic_mh/uq/__init__.py` (empty), `src/forensic_mh/uq/conformal.py`
- Create: `tests/uq/__init__.py` (empty)
- Test: `tests/uq/test_conformal.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/uq/test_conformal.py
import numpy as np
import pytest
from forensic_mh.uq.conformal import (
    conformal_quantile, mondrian_quantiles,
    build_prediction_sets, empirical_coverage,
    mean_set_size, class_conditional_coverage,
)


def test_quantile_is_conservative_higher_interpolation():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    # alpha=0.5 → level = ceil(5*0.5)/4 = 3/4 → 0.75 quantile ("higher") = 0.3
    assert conformal_quantile(scores, alpha=0.5) == pytest.approx(0.3)


def test_quantile_empty_calibration_returns_one():
    # no calibration info → include everything (conservative)
    assert conformal_quantile(np.array([]), alpha=0.1) == 1.0


def test_marginal_coverage_meets_target_on_calibrated_probs():
    rng = np.random.default_rng(0)
    n, K = 4000, 3
    y = rng.integers(0, K, size=n)
    # well-calibrated-ish probs: true class gets a boost
    logits = rng.normal(size=(n, K))
    logits[np.arange(n), y] += 1.5
    probs = np.exp(logits); probs /= probs.sum(1, keepdims=True)
    # split half cal / half test
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
    logits = rng.normal(size=(n, K)); logits[np.arange(n), y] += 1.2
    probs = np.exp(logits); probs /= probs.sum(1, keepdims=True)
    cal, te = slice(0, n // 2), slice(n // 2, n)
    cal_true_scores = 1 - probs[cal][np.arange(n // 2), y[cal]]
    q = mondrian_quantiles(cal_true_scores, y[cal], n_classes=K, alpha=0.1)
    sets = build_prediction_sets(probs[te], q)
    per_class = class_conditional_coverage(sets, y[te], n_classes=K)
    for k in range(K):
        assert per_class[k] >= 0.85  # each class individually covered


def test_mean_set_size_between_zero_and_k():
    sets = [[0], [0, 1], [], [2]]
    assert mean_set_size(sets) == pytest.approx((1 + 2 + 0 + 1) / 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/uq/test_conformal.py -v`
Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/forensic_mh/uq/conformal.py
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
    cal_true_scores: np.ndarray, cal_labels: np.ndarray,
    n_classes: int, alpha: float,
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
    probs: np.ndarray, quantiles: dict[int, float],
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
    sets: list[list[int]], y_true: np.ndarray, n_classes: int,
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/uq/test_conformal.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/forensic_mh/uq/__init__.py src/forensic_mh/uq/conformal.py tests/uq/__init__.py tests/uq/test_conformal.py
git commit -m "feat(uq): split-conformal core (Mondrian quantiles, sets, coverage)"
```

---

## Task 3: Model-agnostic ConformalClassifier wrapper

**Files:**
- Create: `src/forensic_mh/uq/conformal_classifier.py`
- Test: `tests/uq/test_conformal_classifier.py`

The public surface Plan 3 reuses. It wraps any estimator with `fit` + `predict_proba`. Calibration uses a held-out split disjoint from training.

- [ ] **Step 1: Write the failing test**

```python
# tests/uq/test_conformal_classifier.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/uq/test_conformal_classifier.py -v`
Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/forensic_mh/uq/conformal_classifier.py
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
    conformal_quantile, mondrian_quantiles, build_prediction_sets,
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/uq/test_conformal_classifier.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/forensic_mh/uq/conformal_classifier.py tests/uq/test_conformal_classifier.py
git commit -m "feat(uq): model-agnostic ConformalClassifier wrapper"
```

---

## Task 4: Open-set scoring + reject decision (⭐ YOUR design contribution) + OOD metrics

**Files:**
- Create: `src/forensic_mh/uq/openset.py`
- Test: `tests/uq/test_openset.py`

This task contains **one decision that genuinely shapes the system** — see Step 3b.

- [ ] **Step 1: Write the failing test**

```python
# tests/uq/test_openset.py
import numpy as np
import pytest
from forensic_mh.uq.openset import (
    msp_score, ood_auroc, fpr_at_tpr, reject_rate, open_set_decision,
)


def test_msp_score_is_one_minus_max_prob():
    probs = np.array([[0.7, 0.2, 0.1], [0.34, 0.33, 0.33]])
    np.testing.assert_allclose(msp_score(probs), [0.3, 0.66])


def test_ood_auroc_perfect_separation():
    # in-dist scores low, ood scores high → AUROC = 1.0
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


def test_open_set_decision_rejects_empty_set():
    # empty conformal set must reject regardless of MSP
    d = open_set_decision(pred_set=[], max_prob=0.99, msp_threshold=0.0)
    assert d == "reject"


def test_open_set_decision_accepts_confident_nonempty():
    d = open_set_decision(pred_set=[1], max_prob=0.9, msp_threshold=0.5)
    assert d == [1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/uq/test_openset.py -v`
Expected: ImportError

- [ ] **Step 3a: Write the scoring + metric functions (no design choice here)**

```python
# src/forensic_mh/uq/openset.py
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
```

- [ ] **Step 3b: ⭐ Implement the reject rule — YOUR decision**

I've set up the file and the two passing tests above (`test_open_set_decision_rejects_empty_set`, `test_open_set_decision_accepts_confident_nonempty`) pin the two endpoints. The behaviour *between* them is a real design choice with forensic consequences. Add this function to `src/forensic_mh/uq/openset.py`:

```python
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
```

Guidance: the recommended starting point is **reject if the set is empty OR `max_prob < msp_threshold`, else return `pred_set`** — but try the empty-set-only variant too and compare reject curves in Task 6/7. Whichever you pick, the two endpoint tests must stay green; add a test for your chosen middle behaviour.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/uq/test_openset.py -v`
Expected: 6 passed (after you implement Step 3b)

- [ ] **Step 5: Commit**

```bash
git add src/forensic_mh/uq/openset.py tests/uq/test_openset.py
git commit -m "feat(uq): open-set reject (empty-set + MSP) and OOD metrics"
```

---

## Task 5: Coverage vs set-size curve on EAS MH

**Files:**
- Create: `scripts/10_conformal_curve.py`

Sweep `α` and record marginal coverage, per-class coverage, and mean set size on the EAS MH baseline. This is the first headline figure of the trust layer.

- [ ] **Step 1: Write the script**

```python
# scripts/10_conformal_curve.py
"""Coverage vs set-size trade-off curve for Mondrian conformal on EAS MH."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from forensic_mh.pipelines.baseline import build_diplotype_matrix, load_eas_labels
from forensic_mh.uq.conformal_classifier import ConformalClassifier
from forensic_mh.uq.conformal import (
    empirical_coverage, class_conditional_coverage, mean_set_size,
)

VCF = "data/eas/EAS_chr22.vcf.gz"
PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"


def main():
    X, sids, markers = build_diplotype_matrix(VCF, "chr22")
    y, pops = load_eas_labels(PANEL, sids)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    base = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         eval_metric="mlogloss", verbosity=0, random_state=42)
    rows = []
    for alpha in [0.30, 0.20, 0.10, 0.05]:
        cc = ConformalClassifier(base, alpha=alpha, mondrian=True).fit(X_tr, y_tr)
        sets = cc.predict_set(X_te)
        cc_cov = class_conditional_coverage(sets, y_te, len(pops))
        rows.append({
            "alpha": alpha,
            "target_coverage": round(1 - alpha, 3),
            "marginal_coverage": round(empirical_coverage(sets, y_te), 4),
            "mean_set_size": round(mean_set_size(sets), 4),
            "per_class_coverage": {pops[k]: round(cc_cov[k], 4) for k in range(len(pops))},
        })
        print(f"alpha={alpha}: cov={rows[-1]['marginal_coverage']} "
              f"size={rows[-1]['mean_set_size']}")

    out = Path("results/conformal/coverage_curve.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_markers": len(markers), "populations": pops,
                               "curve": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/10_conformal_curve.py`
Expected: marginal coverage ≈ `1−α` at each level; set size shrinks as `α` grows. `results/conformal/coverage_curve.json` written.

⚠️ With chr22-only (near-chance base model) sets will be large (often all 5 labels) at small `α` — that's *correct* conformal behaviour: an uninformative model must output big sets to keep its coverage promise. The story sharpens once genome-wide markers (Plan 1 Task 10 revision) land; the script is marker-set agnostic.

- [ ] **Step 3: Commit**

```bash
git add scripts/10_conformal_curve.py
git commit -m "feat(uq): coverage vs set-size curve on EAS MH"
```

---

## Task 6: Far-OOD — non-EAS superpopulations

**Files:**
- Create: `scripts/11_openset_ood.py`

Extract non-EAS superpopulation samples (EUR/AFR/SAS/AMR) at the **same MH markers**, encode with the EAS-fit `DiplotypeEncoder`, and measure how often the trust layer rejects them (empty set) vs accepts in-dist EAS test samples. The `ALL.chr22…vcf.gz` already downloaded contains these samples — no new download.

- [ ] **Step 1: Write the script**

```python
# scripts/11_openset_ood.py
"""Far-OOD open-set eval: non-EAS superpopulations as unknown."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from forensic_mh.data.markers import load_mh_markers, filter_by_chromosome, parse_positions
from forensic_mh.data.vcf_io import extract_diplotypes_for_locus
from forensic_mh.data.encoding import DiplotypeEncoder
from forensic_mh.pipelines.baseline import load_eas_labels
from forensic_mh.uq.conformal_classifier import ConformalClassifier
from forensic_mh.uq.openset import msp_score, ood_auroc, fpr_at_tpr, reject_rate

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
VCF_ALL = "data/1000g/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
VCF_EAS = "data/eas/EAS_chr22.vcf.gz"
BUILD = "hg19"


def _rows(vcf, sample_ids, markers):
    """Return list[list[str]] of 'h0|h1' per marker, aligned to markers order."""
    per_marker = {}
    for _, mh in markers.iterrows():
        pos = parse_positions(mh, build=BUILD)
        if not pos:
            continue
        d = extract_diplotypes_for_locus(vcf, "22", pos, sample_ids)
        per_marker[mh["Name"]] = d
    names = list(per_marker.keys())
    rows, kept = [], []
    for s in sample_ids:
        if all(s in per_marker[m] for m in names):
            rows.append([f"{per_marker[m][s][0]}|{per_marker[m][s][1]}" for m in names])
            kept.append(s)
    return rows, kept, names


def main():
    panel = pd.read_csv(PANEL, sep="\t")
    eas_ids = panel[panel.super_pop == "EAS"]["sample"].tolist()
    ood_ids = panel[panel.super_pop != "EAS"]["sample"].sample(
        n=300, random_state=42).tolist()   # subsample for speed

    markers = filter_by_chromosome(load_mh_markers(), "chr22")
    eas_rows, eas_kept, names = _rows(VCF_EAS, eas_ids, markers)
    ood_rows, _, _ = _rows(VCF_ALL, ood_ids, markers)

    enc = DiplotypeEncoder()
    X_eas = enc.fit_transform(eas_rows)
    y, pops = load_eas_labels(PANEL, eas_kept)
    X_ood = enc.transform(ood_rows)
    print(f"OOD unseen-diplotype fraction: {enc.last_unseen_fraction:.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_eas, y, test_size=0.3, stratify=y, random_state=42)
    base = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         eval_metric="mlogloss", verbosity=0, random_state=42)

    report = {"populations": pops, "n_markers": len(names),
              "ood_unseen_fraction": round(enc.last_unseen_fraction, 4), "by_alpha": []}
    for alpha in [0.20, 0.10, 0.05]:
        cc = ConformalClassifier(base, alpha=alpha, mondrian=True).fit(X_tr, y_tr)
        sets_in = cc.predict_set(X_te)
        sets_ood = cc.predict_set(X_ood)
        s_in = msp_score(cc.predict_proba(X_te))
        s_ood = msp_score(cc.predict_proba(X_ood))
        report["by_alpha"].append({
            "alpha": alpha,
            "reject_rate_in_dist": round(reject_rate(sets_in), 4),   # want low
            "reject_rate_ood": round(reject_rate(sets_ood), 4),      # want high
            "msp_auroc": round(ood_auroc(s_in, s_ood), 4),
            "msp_fpr@95tpr": round(fpr_at_tpr(s_in, s_ood), 4),
        })
        print(report["by_alpha"][-1])

    out = Path("results/conformal/openset_ood.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/11_openset_ood.py`
Expected: a printed unseen-diplotype fraction (non-EAS should show many novel haplotypes), plus per-`α` reject rates and MSP AUROC. `results/conformal/openset_ood.json` written.

- [ ] **Step 3: Commit**

```bash
git add scripts/11_openset_ood.py
git commit -m "feat(uq): far-OOD open-set eval with non-EAS superpopulations"
```

---

## Task 7: Near-OOD — leave-one-EAS-population-out (LOPO)

**Files:**
- Create: `src/forensic_mh/eval/lopo.py`
- Create: `scripts/12_lopo_nearood.py`
- Test: `tests/eval/test_lopo.py`

Hold out one EAS population entirely; treat its samples as an unknown the model was never trained on. Harder than far-OOD because the held-out population is genetically close — the real test of whether OSR catches a *novel but related* group (the honest stand-in for "an unrepresented population like Korean").

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_lopo.py
import numpy as np
from forensic_mh.eval.lopo import leave_one_population_out


def test_lopo_yields_one_split_per_population():
    y = np.array([0, 0, 1, 1, 2, 2])
    pops = ["CDX", "CHB", "CHS"]
    splits = list(leave_one_population_out(y, pops))
    assert len(splits) == 3
    for held_name, in_idx, out_idx in splits:
        # held population absent from in-distribution indices
        assert held_name in pops
        assert len(set(y[in_idx]) & {pops.index(held_name)}) == 0
        assert all(y[i] == pops.index(held_name) for i in out_idx)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_lopo.py -v`
Expected: ImportError

- [ ] **Step 3: Write minimal implementation**

```python
# src/forensic_mh/eval/lopo.py
"""Leave-one-population-out splits for near-OOD open-set evaluation."""
from __future__ import annotations
import numpy as np
from typing import Iterator


def leave_one_population_out(
    y: np.ndarray, pop_names: list[str],
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (held_population_name, in_dist_idx, held_out_idx) for each label.

    in_dist_idx: samples of all OTHER populations (model trains on these).
    held_out_idx: samples of the held population (unknown at test time)."""
    y = np.asarray(y)
    for k, name in enumerate(pop_names):
        out_idx = np.where(y == k)[0]
        in_idx = np.where(y != k)[0]
        yield name, in_idx, out_idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_lopo.py -v`
Expected: 1 passed

- [ ] **Step 5: Write the LOPO eval script**

```python
# scripts/12_lopo_nearood.py
"""Near-OOD: leave one EAS population out, measure reject rate on the unknown."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from forensic_mh.pipelines.baseline import build_diplotype_matrix, load_eas_labels
from forensic_mh.eval.lopo import leave_one_population_out
from forensic_mh.uq.conformal_classifier import ConformalClassifier
from forensic_mh.uq.openset import reject_rate

VCF = "data/eas/EAS_chr22.vcf.gz"
PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"


def main():
    X, sids, markers = build_diplotype_matrix(VCF, "chr22")
    y, pops = load_eas_labels(PANEL, sids)
    base = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         eval_metric="mlogloss", verbosity=0, random_state=42)

    alpha = 0.10
    rows = []
    for held, in_idx, out_idx in leave_one_population_out(y, pops):
        # remap remaining 4 labels to 0..3 for the classifier
        y_in = y[in_idx]
        remap = {old: new for new, old in enumerate(sorted(set(y_in)))}
        y_in_r = np.array([remap[v] for v in y_in])
        X_in = X[in_idx]
        Xtr, Xte, ytr, yte = train_test_split(
            X_in, y_in_r, test_size=0.3, stratify=y_in_r, random_state=42)
        cc = ConformalClassifier(base, alpha=alpha, mondrian=True).fit(Xtr, ytr)
        rr_in = reject_rate(cc.predict_set(Xte))        # known pops — want low
        rr_held = reject_rate(cc.predict_set(X[out_idx]))  # unknown pop — want high
        rows.append({"held_out": held,
                     "reject_rate_known": round(rr_in, 4),
                     "reject_rate_held_out": round(rr_held, 4)})
        print(rows[-1])

    out = Path("results/conformal/lopo_nearood.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"alpha": alpha, "results": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run it**

Run: `python scripts/12_lopo_nearood.py`
Expected: per-held-population reject rates. Honest expectation on chr22-only: separation may be weak (near-chance base model) — interpret rejection *gap* (held-out minus known), not absolutes. Sharpens genome-wide.

- [ ] **Step 7: Commit**

```bash
git add src/forensic_mh/eval/lopo.py tests/eval/test_lopo.py scripts/12_lopo_nearood.py
git commit -m "feat(eval): LOPO near-OOD open-set evaluation"
```

---

## Task 8: Plan 2 RESULTS document

**Files:**
- Create: `docs/superpowers/plans/2026-05-29-conformal-osr-RESULTS.md`

- [ ] **Step 1: Write the results summary**

```markdown
# Plan 2 — Conformal + OSR RESULTS

**완료일**: 2026-MM-DD (실제 채움)

## 산출 코드
- `src/forensic_mh/uq/{conformal,conformal_classifier,openset}.py` — model-agnostic trust layer
- `src/forensic_mh/data/encoding.py` — OOD-aware shared encoder
- `src/forensic_mh/eval/lopo.py` — near-OOD splits
- `scripts/10_conformal_curve.py`, `11_openset_ood.py`, `12_lopo_nearood.py`

## 결과 (results/conformal/)
- `coverage_curve.json` — marginal·per-class coverage vs mean set size (α∈{.30,.20,.10,.05})
- `openset_ood.json` — 비-EAS far-OOD reject rate + MSP AUROC + FPR@95TPR
- `lopo_nearood.json` — LOPO reject-rate gap (held-out − known)

## 핵심 주장 검증
- [ ] marginal coverage ≥ 1−α (분포-free 보장 실증)
- [ ] per-class(Mondrian) coverage가 모든 5집단에서 ≥ 1−α
- [ ] 비-EAS OOD reject rate > in-dist reject rate (open-set 작동)
- [ ] LOPO에서 held-out reject > known reject (near-OOD 민감도)

## 한계 (논문에 명시)
- chr22-only면 base model이 near-chance → set이 큼. genome-wide 필요 (Plan 1 Task 10).
- KOR 데이터 부재: open-set unknown은 비-EAS superpop + LOPO로 대체. Korean 자체 검증 불가 — limitation.
- 1000G EAS는 사실상 unrelated이라 GroupKFold 영향 미미하나, 일반화 시 relatedness 처리 필요.

## Trustworthy Forensic Model 진척
이 trust layer는 `predict_proba` 인터페이스에만 의존 → Plan 3 SSL FM에 그대로 wrapping (Plan 4 통합).
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-05-29-conformal-osr-RESULTS.md
git commit -m "docs: Plan 2 conformal + OSR results summary"
```

---

## Self-Review

### Spec coverage (sprint Plan 2 항목 → task 매핑)

| sprint Plan 2 항목 | task |
|---|---|
| Mondrian conformal (numpy 직접) | Task 2, 3 |
| coverage vs set-size curve | Task 5 |
| empty-set reject OSR + MSP baseline | Task 4 |
| 비-EAS superpopulation OOD | Task 6 |
| LOPO near-OOD | Task 7 |
| model-agnostic interface (FM 재사용) | Task 3 (`predict_proba` only) |
| OOD 코드 정합성 (encoding) | Task 1 |
| RESULTS | Task 8 |

전부 매핑됨. KOR 외부 검증 항목은 의도적으로 부재 (scope 제외).

### Placeholder scan
- Task 4 Step 3b의 `open_set_decision`는 **의도된 사용자 기여 지점** (NotImplementedError + TODO). 그 외 placeholder 없음 — 모든 코드 inline.

### Type consistency
- `build_prediction_sets(probs, quantiles: dict[int,float]) -> list[list[int]]` — Task 2 정의, Task 3/5/6/7에서 일관 사용.
- `ConformalClassifier(base, alpha, mondrian).fit().predict_set()/predict_proba()` — Task 3 정의, 모든 스크립트 동일 호출.
- `DiplotypeEncoder.fit_transform/transform` + `last_unseen_fraction` — Task 1 정의, Task 6 사용.
- `msp_score/ood_auroc/fpr_at_tpr/reject_rate` 시그니처 — Task 4 정의, Task 6/7 사용.

### Critical risk
- chr22-only base model이 near-chance라 OSR 분리력이 약할 수 있음 — 절대값이 아니라 in-dist vs OOD **격차**로 해석. genome-wide marker(Plan 1 Task 10) 후 재실행이 정식 결과.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-29-conformal-osr.md`.

다음 단계 실행 방식:
1. **Subagent-Driven (recommended)** — task별 fresh subagent + review (Task 4 Step 3b는 사용자 직접 구현 지점이므로 그 task는 inline로).
2. **Inline Execution** — 이 세션에서 task 단위 batch.
