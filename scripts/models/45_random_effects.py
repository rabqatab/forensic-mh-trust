"""Embedding approach C — random-effects embedding (LMMNN-style), GPU.

Simchoni & Rosset (2021, NeurIPS) model high-cardinality categorical features as
random EFFECTS (Gaussian with a learned variance) instead of free embeddings, giving
shrinkage toward the mean — well-suited to small n. Our diplotypes are exactly
high-cardinality (<=278 per marker). Here each (marker, diplotype) embedding is a
random effect e ~ N(0, sigma_m^2) with a per-marker learned variance; the Gaussian
penalty adapts shrinkage. Same protocol: genome-wide, leakage-free 5-fold,
acc + far-OOD MSP AUROC. Compare to LogReg 79.6% / EmbMLP 29.7%.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import collect_genome_wide_strings, discover_chrom_vcfs, load_eas_labels
from forensic_mh.uq.openset import msp_score, ood_auroc

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K, D, EPOCHS, LR, RE_W = 8, 32, 250, 1e-3, 1.0


class RandEffEmb(nn.Module):
    def __init__(self, M, k, d, n_classes, p=0.3):
        super().__init__()
        self.M, self.k = M, k
        self.emb = nn.Embedding(M * k, d)
        nn.init.normal_(self.emb.weight, std=0.05)
        self.register_buffer("off", torch.arange(M) * k)
        self.log_sig = nn.Parameter(torch.zeros(M))        # per-marker random-effect log-std
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))
        self.drop = nn.Dropout(p)

    def forward(self, x):                                  # x: (B, M)
        pooled = self.drop(self.emb(x + self.off).mean(1))  # (B, d)
        return self.head(pooled)

    def re_penalty(self):
        W = self.emb.weight.view(self.M, self.k, D)         # (M, k, d)
        var = (self.log_sig.exp() ** 2).view(self.M, 1, 1) + 1e-6
        return ((W ** 2) / (2 * var) + 0.5 * self.log_sig.view(self.M, 1, 1)).mean()


def fit_eval(Xtr, ytr, Xte, Xood, M, n_classes, seed=0):
    torch.manual_seed(seed)
    model = RandEffEmb(M, K, D, n_classes).to(DEV)
    Xt = torch.as_tensor(Xtr, dtype=torch.long).to(DEV)
    yt = torch.as_tensor(ytr, dtype=torch.long).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    n = len(Xt); rng = np.random.RandomState(seed); idx = rng.permutation(n)
    va, tr = idx[: max(n // 8, n_classes)], idx[n // 8:]
    best, best_state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(Xt[tr]), yt[tr]) + RE_W * model.re_penalty()
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(Xt[va]), yt[va]).item()
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 30:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pte = F.softmax(model(torch.as_tensor(Xte, dtype=torch.long).to(DEV)), 1).cpu().numpy()
        poo = F.softmax(model(torch.as_tensor(Xood, dtype=torch.long).to(DEV)), 1).cpu().numpy()
    return pte, poo


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

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    accs, aurocs = [], []
    for fi, (tr, te) in enumerate(cv.split(eas_rows, y)):
        vocab = FMVocab([eas_rows[i] for i in tr], k=K)
        Xtr = vocab.encode([eas_rows[i] for i in tr])
        Xte = vocab.encode([eas_rows[i] for i in te])
        Xood = vocab.encode(ood_rows)
        pte, poo = fit_eval(Xtr, y[tr], Xte, Xood, M, len(classes), seed=fi)
        accs.append(float((pte.argmax(1) == y[te]).mean()))
        aurocs.append(ood_auroc(msp_score(pte), msp_score(poo)))
        print(f"fold {fi+1}: acc={accs[-1]:.3f} far-OOD AUROC={aurocs[-1]:.3f}", flush=True)

    out = {"model": "RandomEffectEmb", "panel": M,
           "acc_mean": round(float(np.mean(accs)), 4), "acc_std": round(float(np.std(accs)), 4),
           "auroc_mean": round(float(np.mean(aurocs)), 4),
           "reference": {"LogReg": 0.796, "EmbMLP": 0.297}}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/random_effects.json").write_text(json.dumps(out, indent=2))
    print(f"\nRandomEffectEmb acc={out['acc_mean']*100:.1f}±{out['acc_std']*100:.1f}  "
          f"far-OOD AUROC={out['auroc_mean']:.3f}  | ref LogReg 79.6, EmbMLP 29.7", flush=True)
    print("saved results/baseline/random_effects.json", flush=True)
    print("RANDEFF_DONE", flush=True)


if __name__ == "__main__":
    main()
