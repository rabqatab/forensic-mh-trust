"""Embedding approach A — Diet Networks (Romero et al. 2017, ICLR), GPU.

The real Diet Network for p>>n: each one-hot feature is embedded by its profile
across training samples (column of X^T), and an auxiliary network phi PREDICTS the
first-layer weight for that feature -> the fat (D x h) weight matrix costs only
phi's tiny parameters. We only proxied this with EmbMLP (29.7%, §24.2); here is
the faithful version. Same protocol: genome-wide one-hot, leakage-free 5-fold,
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
from sklearn.preprocessing import OneHotEncoder

from forensic_mh.pipelines.baseline import collect_genome_wide_strings, discover_chrom_vcfs, load_eas_labels
from forensic_mh.uq.openset import msp_score, ood_auroc

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, EPOCHS, LR = 128, 200, 1e-3


class DietNet(nn.Module):
    def __init__(self, E_train, h, n_classes, p=0.3):
        super().__init__()
        self.register_buffer("E", E_train)                 # (D, n_train) feature profiles
        n_train = E_train.shape[1]
        self.phi = nn.Sequential(nn.Linear(n_train, 128), nn.ReLU(), nn.Linear(128, h))
        self.b = nn.Parameter(torch.zeros(h))
        self.drop = nn.Dropout(p)
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, n_classes))

    def forward(self, x):                                  # x: (B, D) dense
        W = self.phi(self.E)                               # (D, h) predicted weights
        return self.head(self.drop(torch.relu(x @ W + self.b)))


def fit_eval(Xtr, ytr, Xte, Xood, n_classes, seed=0):
    torch.manual_seed(seed)
    E = torch.as_tensor(Xtr.T, dtype=torch.float32)
    E = (E - E.mean(1, keepdim=True)) / (E.std(1, keepdim=True) + 1e-6)
    model = DietNet(E.to(DEV), H, n_classes).to(DEV)
    Xt = torch.as_tensor(Xtr, dtype=torch.float32).to(DEV)
    yt = torch.as_tensor(ytr, dtype=torch.long).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    # internal val for early stop
    n = len(Xt); rng = np.random.RandomState(seed); idx = rng.permutation(n)
    va, tr = idx[: max(n // 8, n_classes)], idx[n // 8:]
    best, best_state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(Xt[tr]), yt[tr])
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
        pte = F.softmax(model(torch.as_tensor(Xte, dtype=torch.float32).to(DEV)), 1).cpu().numpy()
        poo = F.softmax(model(torch.as_tensor(Xood, dtype=torch.float32).to(DEV)), 1).cpu().numpy()
    return pte, poo


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    Xstr = np.array([[rows[s].get(m, "N|N") for m in names] for s in sids], dtype=object)
    y, _ = load_eas_labels(PANEL, sids)
    classes = np.unique(y); y = np.searchsorted(classes, y)
    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    Ostr = np.array([[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)], dtype=object)
    print(f"panel={len(names)} EAS={len(sids)} OOD={Ostr.shape[0]} device={DEV}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    accs, aurocs = [], []
    for fi, (tr, te) in enumerate(cv.split(Xstr, y)):
        ohe = OneHotEncoder(handle_unknown="ignore").fit(Xstr[tr])
        Xtr = ohe.transform(Xstr[tr]).toarray().astype(np.float32)
        Xte = ohe.transform(Xstr[te]).toarray().astype(np.float32)
        Xood = ohe.transform(Ostr).toarray().astype(np.float32)
        pte, poo = fit_eval(Xtr, y[tr], Xte, Xood, len(classes), seed=fi)
        accs.append(float((pte.argmax(1) == y[te]).mean()))
        aurocs.append(ood_auroc(msp_score(pte), msp_score(poo)))
        print(f"fold {fi+1}: D={Xtr.shape[1]} acc={accs[-1]:.3f} far-OOD AUROC={aurocs[-1]:.3f}", flush=True)

    out = {"model": "DietNet", "panel": len(names),
           "acc_mean": round(float(np.mean(accs)), 4), "acc_std": round(float(np.std(accs)), 4),
           "auroc_mean": round(float(np.mean(aurocs)), 4),
           "reference": {"LogReg": 0.796, "EmbMLP": 0.297}}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/dietnet.json").write_text(json.dumps(out, indent=2))
    print(f"\nDietNet acc={out['acc_mean']*100:.1f}±{out['acc_std']*100:.1f}  "
          f"far-OOD AUROC={out['auroc_mean']:.3f}  | ref LogReg 79.6, EmbMLP 29.7", flush=True)
    print("saved results/baseline/dietnet.json", flush=True)
    print("DIETNET_DONE", flush=True)


if __name__ == "__main__":
    main()
