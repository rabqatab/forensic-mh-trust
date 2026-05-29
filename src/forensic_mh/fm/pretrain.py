"""SSL pretraining: masked-marker + NT-Xent contrastive on MHMatrixDataset."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from forensic_mh.fm.objectives import masked_marker_loss, nt_xent


def ssl_pretrain(encoder, dataset, epochs: int = 50, batch_size: int = 64,
                 lr: float = 1e-3, lambda_contrastive: float = 0.5,
                 weight_decay: float = 1e-2, seed: int = 0,
                 device: str = "cpu") -> list[dict]:
    torch.manual_seed(seed)
    encoder.to(device).train()
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    history = []
    for _ in range(epochs):
        tot = 0.0
        for b in loader:
            inp = b["input"].to(device)
            logits = encoder.masked_logits(inp)
            l_mask = masked_marker_loss(logits, b["target"].to(device),
                                        b["mask_pos"].to(device))
            z1 = encoder.embed(b["view1"].to(device))
            z2 = encoder.embed(b["view2"].to(device))
            l_con = nt_xent(z1, z2) if z1.shape[0] > 1 else z1.sum() * 0.0
            loss = l_mask + lambda_contrastive * l_con
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.detach().item()
        history.append({"loss": tot / len(loader)})
    return history
