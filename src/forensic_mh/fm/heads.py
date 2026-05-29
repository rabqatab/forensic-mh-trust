"""Task heads on the shared FM embedding."""
from __future__ import annotations

import torch
import torch.nn as nn


class AncestryHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int = 5):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)


class SexHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)


class KinshipHead(nn.Module):
    """Pairwise: symmetric features (|a-b|, a*b) → relatedness logit."""
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([torch.abs(a - b), a * b], dim=1)
        return self.fc(feat)
