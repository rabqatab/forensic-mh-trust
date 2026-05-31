"""(d) CREE core — variance-as-nonconformity open-set (the distinctive claim).

A VARIATIONAL random-effects embedding: each (marker, diplotype) embedding is a
Gaussian N(mu, var) shrunk toward a per-marker prior. A sample's posterior
variance (mean over its markers) is an INTRINSIC open-set score: OOD / unseen
diplotypes map to high-variance (rare/OTHER) codes. We test whether this
variance score beats MSP for open-set AUROC (target: LinearSVC 0.957, §24.4),
while keeping accuracy + conformal coverage. 5-seed. GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import collect_genome_wide_strings, discover_chrom_vcfs, load_eas_labels
from forensic_mh.uq.openset import ood_auroc

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
K, D, EPOCHS, LR, KL_W = 8, 32, 250, 1e-3, 0.5
SEEDS = list(range(5))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class VRE(nn.Module):
    def __init__(self, M, k, d, n_classes, p=0.3):
        super().__init__()
        self.M, self.k, self.d = M, k, d
        self.mu = nn.Embedding(M * k, d); nn.init.normal_(self.mu.weight, std=0.05)
        self.logvar = nn.Embedding(M * k, d); nn.init.constant_(self.logvar.weight, -4.0)
        self.register_buffer("off", torch.arange(M) * k)
        self.log_sig = nn.Parameter(torch.zeros(M))          # per-marker prior log-std
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))
        self.drop = nn.Dropout(p)

    def forward(self, x, sample=True):
        idx = x + self.off
        mu, lv = self.mu(idx), self.logvar(idx)
        e = mu + torch.randn_like(mu) * (0.5 * lv).exp() if sample else mu
        return self.head(self.drop(e.mean(1)))

    def kl(self):
        mu = self.mu.weight.view(self.M, self.k, self.d)
        lv = self.logvar.weight.view(self.M, self.k, self.d)
        ps = (self.log_sig.exp() ** 2).view(self.M, 1, 1) + 1e-6
        return (0.5 * (lv.exp() / ps + mu ** 2 / ps - 1 - lv + ps.log())).mean()

    @torch.no_grad()
    def var_score(self, x):                                  # per-sample posterior variance (higher = OOD)
        return self.logvar(x + self.off).exp().mean((1, 2)).cpu().numpy()


def fit(X, y, M, nc, seed):
    torch.manual_seed(seed)
    m = VRE(M, K, D, nc).to(DEV)
    Xt = torch.as_tensor(X, dtype=torch.long).to(DEV); yt = torch.as_tensor(y).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        loss = F.cross_entropy(m(Xt), yt) + KL_W * m.kl()
        loss.backward(); opt.step()
    m.eval()
    return m


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    y, _ = load_eas_labels(PANEL, sids)
    classes = np.unique(y); y = np.searchsorted(classes, y)
    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    ood_rows = [[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)]
    M = len(names)
    print(f"panel={M} EAS={len(sids)} OOD={len(ood_rows)} device={DEV}", flush=True)

    vocab = FMVocab(eas_rows, k=K)
    X = vocab.encode(eas_rows); Xood = vocab.encode(ood_rows)
    rec = {"acc": [], "auroc_var": [], "auroc_msp": []}
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        m = fit(Xtr, ytr, M, len(classes), seed)
        with torch.no_grad():
            pte = F.softmax(m(torch.as_tensor(Xte, dtype=torch.long).to(DEV), sample=False), 1).cpu().numpy()
            poo = F.softmax(m(torch.as_tensor(Xood, dtype=torch.long).to(DEV), sample=False), 1).cpu().numpy()
        from forensic_mh.uq.openset import msp_score
        var_in = m.var_score(torch.as_tensor(Xte, dtype=torch.long).to(DEV))
        var_oo = m.var_score(torch.as_tensor(Xood, dtype=torch.long).to(DEV))
        rec["acc"].append(float((pte.argmax(1) == yte).mean()))
        # ood_auroc wants an OOD-ness score (higher = more OOD)
        rec["auroc_msp"].append(ood_auroc(msp_score(pte), msp_score(poo)))   # 1 - max softmax
        rec["auroc_var"].append(ood_auroc(var_in, var_oo))                   # higher posterior variance = OOD
        print(f"seed {seed}: acc={rec['acc'][-1]:.3f} AUROC var={rec['auroc_var'][-1]:.3f} msp={rec['auroc_msp'][-1]:.3f}", flush=True)

    out = {"model": "CREE (variational random-effects)", **{k: {"mean": round(float(np.mean(v)), 4),
           "std": round(float(np.std(v)), 4)} for k, v in rec.items()},
           "ref": {"LinearSVC_OSR": 0.957, "LogReg_OSR": 0.863}}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/cree_variance.json").write_text(json.dumps(out, indent=2))
    print(f"\nCREE acc={out['acc']['mean']*100:.1f}  open-set AUROC: variance={out['auroc_var']['mean']:.3f} "
          f"vs MSP={out['auroc_msp']['mean']:.3f}  (ref LinearSVC 0.957)", flush=True)
    print("saved results/baseline/cree_variance.json", flush=True)
    print("CREE_VARIANCE_DONE", flush=True)


if __name__ == "__main__":
    main()
