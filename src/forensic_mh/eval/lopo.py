"""Leave-one-population-out splits for near-OOD open-set evaluation."""
from __future__ import annotations

from typing import Iterator

import numpy as np


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
