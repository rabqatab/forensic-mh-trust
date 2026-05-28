"""Shared per-marker diplotype→int encoder.

Fit on TRAINING diplotype strings so the same string maps to the same code
across any sample group. Unseen strings (frequent in OOD / non-EAS samples)
map to a reserved code — itself an open-set signal."""
from __future__ import annotations

import numpy as np


class DiplotypeEncoder:
    def __init__(self, unseen_code: int = -1):
        self.unseen_code = unseen_code
        self.maps_: list[dict[str, int]] = []  # one dict per marker column
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
