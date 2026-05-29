"""Multi-task fine-tuning of encoder + ancestry/sex heads (joint loss)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def multitask_finetune(encoder, heads, X, y_anc, y_sex, epochs: int = 50,
                       batch_size: int = 64, lr: float = 1e-3,
                       lambda_sex: float = 0.5, weight_decay: float = 1e-2,
                       seed: int = 0, device: str = "cpu") -> list[dict]:
    torch.manual_seed(seed)
    encoder.to(device).train()
    anc, sex = heads["ancestry"].to(device), heads["sex"].to(device)
    params = list(encoder.parameters()) + list(anc.parameters()) + list(sex.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(X, y_anc, y_sex),
                        batch_size=batch_size, shuffle=True)
    history = []
    for _ in range(epochs):
        correct = tot = 0
        for xb, ya, ys in loader:
            emb = encoder.embed(xb.to(device))
            la = anc(emb)
            loss = F.cross_entropy(la, ya.to(device))
            loss = loss + lambda_sex * F.binary_cross_entropy_with_logits(
                sex(emb).squeeze(1), ys.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            correct += int((la.argmax(1).cpu() == ya).sum()); tot += len(ya)
        history.append({"ancestry_acc": correct / tot})
    return history
