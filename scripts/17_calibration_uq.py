"""Thread 1 — calibration (ECE) + Deep Ensembles / MC-Dropout UQ.

Quantifies the 'XGBoost is overconfident' claim with ECE/reliability across the
model zoo, and adds aleatoric/epistemic UQ (Deep Ensemble of MLPs + MC-Dropout)
— novel for forensic ancestry. Compares ECE, accuracy, and OSR (MSP AUROC; plus
ensemble epistemic-MI AUROC) on genome-wide EAS vs non-EAS OOD.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.openset import msp_score, ood_auroc

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
SEED, K_ENS, MC_T = 42, 5, 20


def ece(proba, y, n_bins=10):
    conf, pred = proba.max(1), proba.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(y)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(e)


class MLP(nn.Module):
    def __init__(self, d_in, n_cls, hidden=64, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(hidden, n_cls))

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, n_cls, seed, epochs=60):
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[1], n_cls)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(Xt), yt)
        loss.backward(); opt.step()
    return m


def proba_mlp(m, X, train_mode=False):
    (m.train if train_mode else m.eval)()
    with torch.no_grad():
        return torch.softmax(m(torch.tensor(X, dtype=torch.float32)), 1).numpy()


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    Xstr = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    y, pops = load_eas_labels(PANEL, sids)
    enc = FMVocab(Xstr, k=16)
    Xord = enc.encode(Xstr)
    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    Ostr = [[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)]
    Oord = enc.encode(Ostr)
    print(f"panel={len(names)} EAS={len(sids)} OOD={len(Ostr)}", flush=True)

    n_cls = len(pops)
    out = {"panel": len(names), "models": {}}

    # --- tree / linear models (ordinal or one-hot) ---
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(Xstr)
    Xoh, Ooh = oh.transform(Xstr), oh.transform(Ostr)
    itr, ite = train_test_split(np.arange(len(sids)), test_size=0.3, stratify=y, random_state=SEED)

    for name, X, Xo in [("RandomForest", Xord, Oord), ("XGBoost", Xord, Oord), ("LogReg", Xoh, Ooh)]:
        clf = (RandomForestClassifier(n_estimators=400, random_state=SEED) if name == "RandomForest"
               else XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                  eval_metric="mlogloss", verbosity=0, random_state=SEED) if name == "XGBoost"
               else LogisticRegression(max_iter=2000))
        clf.fit(X[itr], y[itr])
        p_te, p_ood = clf.predict_proba(X[ite]), clf.predict_proba(Xo)
        out["models"][name] = {
            "accuracy": round(float((p_te.argmax(1) == y[ite]).mean()), 4),
            "ECE": round(ece(p_te, y[ite]), 4),
            "msp_auroc": round(ood_auroc(msp_score(p_te), msp_score(p_ood)), 4)}
        print(f"[{name}] {out['models'][name]}", flush=True)

    # --- Deep Ensemble (K MLPs on one-hot) + epistemic MI OSR ---
    Xtr_oh = Xoh[itr]
    members_te = np.stack([proba_mlp(train_mlp(Xtr_oh, y[itr], n_cls, s), Xoh[ite]) for s in range(K_ENS)])
    members_ood = np.stack([proba_mlp(train_mlp(Xtr_oh, y[itr], n_cls, s), Ooh) for s in range(K_ENS)])
    pe_te, pe_ood = members_te.mean(0), members_ood.mean(0)

    def mi(stack, mean):  # epistemic = H(mean) - mean_k H(p_k)
        H = -(mean * np.log(mean + 1e-9)).sum(1)
        Hk = -(stack * np.log(stack + 1e-9)).sum(2).mean(0)
        return H - Hk
    out["models"]["DeepEnsemble"] = {
        "accuracy": round(float((pe_te.argmax(1) == y[ite]).mean()), 4),
        "ECE": round(ece(pe_te, y[ite]), 4),
        "msp_auroc": round(ood_auroc(msp_score(pe_te), msp_score(pe_ood)), 4),
        "epistemic_auroc": round(ood_auroc(mi(members_te, pe_te), mi(members_ood, pe_ood)), 4)}
    print(f"[DeepEnsemble] {out['models']['DeepEnsemble']}", flush=True)

    # --- MC-Dropout (1 MLP, T stochastic passes) ---
    m = train_mlp(Xtr_oh, y[itr], n_cls, SEED)
    mc_te = np.stack([proba_mlp(m, Xoh[ite], train_mode=True) for _ in range(MC_T)])
    mc_ood = np.stack([proba_mlp(m, Ooh, train_mode=True) for _ in range(MC_T)])
    mt, mo = mc_te.mean(0), mc_ood.mean(0)
    out["models"]["MC-Dropout"] = {
        "accuracy": round(float((mt.argmax(1) == y[ite]).mean()), 4),
        "ECE": round(ece(mt, y[ite]), 4),
        "msp_auroc": round(ood_auroc(msp_score(mt), msp_score(mo)), 4),
        "epistemic_auroc": round(ood_auroc(mi(mc_te, mt), mi(mc_ood, mo)), 4)}
    print(f"[MC-Dropout] {out['models']['MC-Dropout']}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/calibration_uq.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/calibration_uq.json", flush=True)


if __name__ == "__main__":
    main()
