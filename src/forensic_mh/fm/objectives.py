"""SSL objectives: masked-marker cross-entropy + NT-Xent contrastive."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_marker_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask_pos: torch.Tensor
) -> torch.Tensor:
    """Mean CE over masked marker positions only.

    logits: (N, M, K); targets: (N, M) in [0,K); mask_pos: (N, M) bool.
    Returns 0 if nothing is masked.
    """
    if mask_pos.sum() == 0:
        return logits.sum() * 0.0
    sel = logits[mask_pos]            # (n_masked, K)
    tgt = targets[mask_pos]           # (n_masked,)
    return F.cross_entropy(sel, tgt)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy (SimCLR) for a batch.

    z1, z2: (B, d) embeddings of two views. Positive pair = (z1[i], z2[i]).
    Requires batch size B >= 2; with B=1 there are no negatives and the loss degenerates to ~0.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                # (2B, d)
    sim = z @ z.t() / tau                          # (2B, 2B)
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, targets)
