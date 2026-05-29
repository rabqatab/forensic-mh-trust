"""scikit-learn-style adapter so the Plan 2 trust layer wraps the FM unchanged.

Accepts integer-encoded code matrices (values in [0, k)); MASK slot is k.
Runs a short SSL pretrain on the input matrix (self-supervised, label-free)
then multi-task fine-tune of the ancestry head, exposing predict_proba/embed.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, TensorDataset

from forensic_mh.fm.encoder import MHTransformer
from forensic_mh.fm.heads import AncestryHead
from forensic_mh.fm.objectives import masked_marker_loss


class ForensicFMClassifier(BaseEstimator):
    def __init__(self, k: int = 16, d_model: int = 128, n_layers: int = 3,
                 n_heads: int = 4, pretrain_epochs: int = 30,
                 finetune_epochs: int = 30, lr: float = 1e-3,
                 mask_frac: float = 0.15, seed: int = 42, device: str = "cpu",
                 weight_decay: float = 1e-2, batch_size: int = 64):
        self.k = k
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.mask_frac = mask_frac
        self.seed = seed
        self.device = device
        self.weight_decay = weight_decay
        self.batch_size = batch_size

    def _mask(self, x, gen):
        m = torch.rand(x.shape, generator=gen) < self.mask_frac
        inp = x.clone(); inp[m] = self.k          # MASK slot
        return inp, m

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        X = torch.as_tensor(np.asarray(X), dtype=torch.long)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y_idx = torch.as_tensor(np.searchsorted(self.classes_, y), dtype=torch.long)
        M = X.shape[1]
        self.encoder_ = MHTransformer(M, self.k, self.d_model,
                                      self.n_layers, self.n_heads).to(self.device)
        gen = torch.Generator().manual_seed(self.seed)
        # SSL pretrain (masked-marker; label-free)
        opt = torch.optim.AdamW(self.encoder_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        dl = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=True)
        self.encoder_.train()
        for _ in range(self.pretrain_epochs):
            for (xb,) in dl:
                inp, m = self._mask(xb, gen)
                loss = masked_marker_loss(self.encoder_.masked_logits(inp.to(self.device)),
                                          xb.to(self.device), m.to(self.device))
                opt.zero_grad(); loss.backward(); opt.step()
        # multi-task fine-tune (ancestry head)
        self.head_ = AncestryHead(self.d_model, self.n_classes_).to(self.device)
        params = list(self.encoder_.parameters()) + list(self.head_.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        dl = DataLoader(TensorDataset(X, y_idx), batch_size=self.batch_size, shuffle=True)
        for _ in range(self.finetune_epochs):
            for xb, yb in dl:
                logits = self.head_(self.encoder_.embed(xb.to(self.device)))
                loss = torch.nn.functional.cross_entropy(logits, yb.to(self.device))
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        self.encoder_.eval(); self.head_.eval()
        X = torch.as_tensor(np.asarray(X), dtype=torch.long).to(self.device)
        logits = self.head_(self.encoder_.embed(X))
        return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    @torch.no_grad()
    def embed(self, X):
        self.encoder_.eval()
        X = torch.as_tensor(np.asarray(X), dtype=torch.long).to(self.device)
        return self.encoder_.embed(X).cpu().numpy()
