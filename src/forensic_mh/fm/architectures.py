"""Diverse DL architectures for MH ancestry — grounded in popgen DL literature.

All share a per-marker (marker, diplotype-code) embedding (as in the FM encoder)
and consume integer code matrices from FMVocab (values in [0, k)).

- EmbMLP   : entity-embedding MLP — categorical-embedding / Diet-Networks family
             (Romero et al. 2017, ICLR; Guo & Berkhahn 2016) for p>>n SNP data.
- CNN1D    : 1D-CNN over genome-ordered markers — popgen-CNN family
             (Flagel et al. 2019 MBE; genomatnn; Korfmann et al. 2023 Nat Rev Genet).
- SupAE    : supervised autoencoder with a bottleneck + reconstruction aux loss —
             autoencoder family (Neural ADMIXTURE, Mantes et al. 2023 Nat Comp Sci;
             popVAE, Battey et al. 2021).

(The attention/Transformer arm reuses fm.MHTransformer via ForensicFMClassifier.)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader, TensorDataset


class EmbMLP(nn.Module):
    def __init__(self, M, k, d, n_classes, hidden=256, p=0.3):
        super().__init__()
        self.value_emb = nn.Embedding(M * k, d)
        self.pos = nn.Embedding(M, d)
        self.register_buffer("off", torch.arange(M) * k)
        self.register_buffer("idx", torch.arange(M))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(hidden, n_classes))

    def forward(self, x):
        e = self.value_emb(x + self.off) + self.pos(self.idx)   # (B, M, d)
        return self.head(e.mean(1))


class CNN1D(nn.Module):
    def __init__(self, M, k, d, n_classes, ch=64, p=0.3):
        super().__init__()
        self.value_emb = nn.Embedding(M * k, d)
        self.register_buffer("off", torch.arange(M) * k)
        self.conv = nn.Sequential(
            nn.Conv1d(d, ch, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(ch, ch, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(ch, ch, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(ch, n_classes))

    def forward(self, x):
        e = self.value_emb(x + self.off)             # (B, M, d)
        h = self.conv(e.transpose(1, 2)).squeeze(-1)  # (B, ch)
        return self.head(h)


class SupAE(nn.Module):
    def __init__(self, M, k, d, n_classes, bottleneck=32, p=0.3):
        super().__init__()
        self.value_emb = nn.Embedding(M * k, d)
        self.register_buffer("off", torch.arange(M) * k)
        self.enc = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.ReLU(),
                                 nn.Linear(128, bottleneck))
        self.dec = nn.Linear(bottleneck, d)
        self.clf = nn.Sequential(nn.Dropout(p), nn.Linear(bottleneck, n_classes))

    def forward(self, x):
        e = self.value_emb(x + self.off).mean(1)     # (B, d) bag-of-markers
        z = self.enc(e)
        return self.clf(z), self.dec(z), e


class FTTransformer(nn.Module):
    """Feature-Tokenizer Transformer (Gorishniy et al. 2021) for categorical features:
    per-feature value embedding as tokens + [CLS] + pre-norm transformer -> CLS head.
    """
    def __init__(self, M, k, d, n_classes, n_layers=3, n_heads=4, p=0.1):
        super().__init__()
        self.value_emb = nn.Embedding(M * k, d)
        self.register_buffer("off", torch.arange(M) * k)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=4 * d,
                                           dropout=p, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def forward(self, x):
        tok = self.value_emb(x + self.off)                       # (B, M, d)
        seq = torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], dim=1)
        return self.head(self.enc(seq)[:, 0])


class ResNetTab(nn.Module):
    """ResNet for tabular data (Gorishniy et al. 2021) — the co-SOTA with FT-Transformer.
    Bag-of-marker embedding -> residual channel-MLP blocks -> head."""
    def __init__(self, M, k, d, n_classes, n_blocks=3, hidden=256, p=0.2):
        super().__init__()
        self.value_emb = nn.Embedding(M * k, d)
        self.pos = nn.Embedding(M, d)
        self.register_buffer("off", torch.arange(M) * k)
        self.register_buffer("idx", torch.arange(M))
        self.inp = nn.Linear(d, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU(),
                          nn.Dropout(p), nn.Linear(hidden, hidden)) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, x):
        e = (self.value_emb(x + self.off) + self.pos(self.idx)).mean(1)   # (B, d)
        h = self.inp(e)
        for blk in self.blocks:
            h = h + blk(h)
        return self.head(h)


class ResCNN(nn.Module):
    """Deeper residual 1D-CNN over genome-ordered markers (genomatnn/Flagel-faithful)."""
    def __init__(self, M, k, d, n_classes, ch=64, n_blocks=3, p=0.2):
        super().__init__()
        self.value_emb = nn.Embedding(M * k, d)
        self.register_buffer("off", torch.arange(M) * k)
        self.stem = nn.Conv1d(d, ch, 7, padding=3)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
                          nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch)) for _ in range(n_blocks)])
        self.pool = nn.MaxPool1d(2)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(ch, n_classes))

    def forward(self, x):
        h = torch.relu(self.stem(self.value_emb(x + self.off).transpose(1, 2)))   # (B, ch, M)
        for blk in self.blocks:
            h = self.pool(torch.relu(h + blk(h)))
        return self.head(h.mean(-1))


_ARCH = {"embmlp": EmbMLP, "cnn1d": CNN1D, "supae": SupAE, "fttransformer": FTTransformer,
         "resnettab": ResNetTab, "rescnn": ResCNN}


class TorchArchClassifier(BaseEstimator):
    """sklearn-style wrapper around the architectures above (consumes int codes)."""

    def __init__(self, arch="embmlp", k=8, d=32, epochs=150, lr=1e-3, batch=64,
                 weight_decay=1e-3, recon_w=0.3, patience=25, seed=0, device=None):
        self.arch = arch
        self.k = k
        self.d = d
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.weight_decay = weight_decay
        self.recon_w = recon_w
        self.patience = patience
        self.seed = seed
        self.device = device

    def _dev(self):
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        dev = self._dev()
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        yi = np.searchsorted(self.classes_, y)
        M = X.shape[1]
        # internal validation split for early stopping (train data only — leakage-free)
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(len(X))
        nval = max(self.n_classes_, int(0.12 * len(X)))
        va, tr = idx[:nval], idx[nval:]
        Xt = torch.as_tensor(X[tr], dtype=torch.long)
        yt = torch.as_tensor(yi[tr], dtype=torch.long)
        Xv = torch.as_tensor(X[va], dtype=torch.long).to(dev)
        yv = torch.as_tensor(yi[va], dtype=torch.long).to(dev)
        self.model_ = _ARCH[self.arch](M, self.k, self.d, self.n_classes_).to(dev)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        dl = DataLoader(TensorDataset(Xt, yt), batch_size=self.batch, shuffle=True)
        best, best_state, bad = 1e9, None, 0
        for _ in range(self.epochs):
            self.model_.train()
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                out = self.model_(xb)
                if self.arch == "supae":
                    logits, recon, target = out
                    loss = F.cross_entropy(logits, yb) + self.recon_w * F.mse_loss(recon, target)
                else:
                    loss = F.cross_entropy(out, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                vo = self.model_(Xv)
                vlogits = vo[0] if self.arch == "supae" else vo
                vl = F.cross_entropy(vlogits, yv).item()
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {kk: vv.detach().clone() for kk, vv in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state:
            self.model_.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        dev = self._dev()
        self.model_.eval()
        X = torch.as_tensor(np.asarray(X), dtype=torch.long).to(dev)
        out = self.model_(X)
        logits = out[0] if self.arch == "supae" else out
        return torch.softmax(logits, 1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
