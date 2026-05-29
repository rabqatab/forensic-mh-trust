"""Dataset producing one masked-modeling view + two contrastive views per sample.

Contrastive augmentations operate at the STRING level so allele-dropout (ADO,
het→hom) is faithful, then re-encode with the shared FMVocab.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from forensic_mh.fm.vocab import FMVocab


def _ado(row: list[str], rng: np.random.Generator, p: float) -> list[str]:
    """Collapse heterozygous 'h0|h1' (h0!=h1) to a homozygote with prob p."""
    out = []
    for cell in row:
        h0, h1 = cell.split("|", 1)
        if h0 != h1 and rng.random() < p:
            keep = h0 if rng.random() < 0.5 else h1
            out.append(f"{keep}|{keep}")
        else:
            out.append(cell)
    return out


class MHMatrixDataset(Dataset):
    """Masked-modeling + contrastive dataset for microhaplotype matrices.

    Augmentation is stochastic — each ``__getitem__`` draws fresh samples from
    ``self.rng``.  Because ``self.rng`` is shared state, DataLoader must use
    ``num_workers=0`` (or a ``worker_init_fn`` that reseeds the RNG) to avoid
    identical augmentations across forked workers.
    """

    def __init__(self, rows: list[list[str]], vocab: FMVocab,
                 mask_frac: float = 0.15, ado_prob: float = 0.1,
                 drop_prob: float = 0.15, seed: int = 0):
        self.rows = rows
        self.vocab = vocab
        self.mask_frac = mask_frac
        self.ado_prob = ado_prob
        self.drop_prob = drop_prob
        self.base = vocab.encode(rows)           # (N, M) int64
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def _view(self, i: int) -> torch.Tensor:
        row = _ado(self.rows[i], self.rng, self.ado_prob)
        codes = self.vocab.encode([row])[0].copy()
        drop = self.rng.random(self.vocab.n_markers) < self.drop_prob
        codes[drop] = self.vocab.MASK
        return torch.from_numpy(codes)

    def __getitem__(self, i: int) -> dict:
        base = self.base[i].copy()
        target = torch.from_numpy(base.copy())
        mask_pos = torch.from_numpy(self.rng.random(self.vocab.n_markers) < self.mask_frac)
        inp = torch.from_numpy(base)
        inp[mask_pos] = self.vocab.MASK
        return {
            "input": inp, "target": target, "mask_pos": mask_pos,
            "view1": self._view(i), "view2": self._view(i),
        }
