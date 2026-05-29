"""Per-marker capped diplotype vocabulary for the FM.

Each marker keeps its top (k-1) most frequent diplotype strings; everything
else (rare or unseen) maps to OTHER (index k-1). Slot k is the MASK token.
Uniform (k+1)-slot layout per marker → vectorised masked prediction.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


class FMVocab:
    def __init__(self, rows: list[list[str]], k: int = 16):
        if not rows:
            raise ValueError("FMVocab requires at least one row")
        self.k = k
        self.OTHER = k - 1
        self.MASK = k
        self.n_slots = k + 1
        self.n_markers = len(rows[0])
        # per-marker {string: code} for the top (k-1) strings
        self.maps_: list[dict[str, int]] = []
        for j in range(self.n_markers):
            counts = Counter(row[j] for row in rows)
            top = sorted(counts, key=lambda s: (-counts[s], s))[: k - 1]
            self.maps_.append({s: i for i, s in enumerate(top)})

    def encode(self, rows: list[list[str]]) -> np.ndarray:
        out = np.empty((len(rows), self.n_markers), dtype=np.int64)
        for i, row in enumerate(rows):
            for j in range(self.n_markers):
                out[i, j] = self.maps_[j].get(row[j], self.OTHER)
        return out
