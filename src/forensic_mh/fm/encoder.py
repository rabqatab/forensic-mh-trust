"""Small marker-transformer. Per-marker value embeddings are weight-tied to the
masked-prediction head (einsum), so variable diplotype semantics per marker are
handled with a single uniform (K+1)-slot table."""
from __future__ import annotations

import torch
import torch.nn as nn


class MHTransformer(nn.Module):
    def __init__(self, n_markers: int, k: int, d_model: int = 128,
                 n_layers: int = 3, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.M = n_markers
        self.k = k
        self.slots = k + 1                       # K value classes + MASK
        self.d = d_model
        self.value_emb = nn.Embedding(n_markers * self.slots, d_model)
        self.marker_pos = nn.Embedding(n_markers, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.register_buffer("offsets", torch.arange(n_markers) * self.slots)

    def _hidden(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, M) per-marker codes in [0, slots)
        N = x.shape[0]
        tok = self.value_emb(x + self.offsets) + self.marker_pos(
            torch.arange(self.M, device=x.device))
        seq = torch.cat([self.cls.expand(N, -1, -1), tok], dim=1)  # (N, M+1, d)
        return self.enc(seq)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self._hidden(x)[:, 0]              # CLS → (N, d)

    def masked_logits(self, x: torch.Tensor) -> torch.Tensor:
        h = self._hidden(x)[:, 1:]                # per-marker hidden (N, M, d)
        W = self.value_emb.weight.view(self.M, self.slots, self.d)[:, : self.k]
        return torch.einsum("nmd,mkd->nmk", h, W)  # (N, M, K)
