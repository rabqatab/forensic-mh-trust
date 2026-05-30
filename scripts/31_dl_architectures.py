"""Diverse DL architectures benchmark (RQ1/RQ3) — literature-grounded, GPU.

Same protocol as the classical/MLP zoo (§24): genome-wide 3,042 markers,
leakage-free 5-fold CV, accuracy + far-OOD MSP AUROC. Architectures span the
deep-learning-for-population-genetics literature:
  - EmbMLP    : entity-embedding MLP (Diet Networks, Romero 2017; entity emb.)
  - CNN1D     : 1D-CNN over genome-ordered markers (Flagel 2019; genomatnn)
  - SupAE     : supervised autoencoder (Neural ADMIXTURE 2023; popVAE 2021)
  - Transformer (supervised)      : MHTransformer, no pretraining
  - Transformer (SSL + finetune)  : MHTransformer with masked-marker pretraining (the FM)
All consume FMVocab integer codes; encoding fit inside each fold (leakage-free).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from forensic_mh.fm.architectures import TorchArchClassifier
from forensic_mh.fm.sklearn_api import ForensicFMClassifier
from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.openset import msp_score, ood_auroc

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
K = 8
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build(name):
    if name == "Transformer (supervised)":
        return ForensicFMClassifier(k=K, d_model=64, n_layers=2, n_heads=4,
                                    pretrain_epochs=0, finetune_epochs=60,
                                    batch_size=32, device=DEV)
    if name == "Transformer (SSL+ft)":
        return ForensicFMClassifier(k=K, d_model=64, n_layers=2, n_heads=4,
                                    pretrain_epochs=25, finetune_epochs=60,
                                    batch_size=32, device=DEV)
    arch = {"EmbMLP (Diet-Net)": "embmlp", "CNN1D (popgen-CNN)": "cnn1d",
            "SupAE (Neural-ADMIXTURE)": "supae"}[name]
    return TorchArchClassifier(arch=arch, k=K, d=32, epochs=150, device=DEV)


MODELS = ["EmbMLP (Diet-Net)", "CNN1D (popgen-CNN)", "SupAE (Neural-ADMIXTURE)",
          "Transformer (supervised)", "Transformer (SSL+ft)"]


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    y, _ = load_eas_labels(PANEL, sids)
    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    ood_rows = [[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)]
    y = np.asarray(y)
    print(f"panel={len(names)} EAS={len(sids)} OOD={len(ood_rows)} device={DEV}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    out = {"panel": len(names), "device": DEV, "models": {}}
    folds = list(cv.split(eas_rows, y))
    for name in MODELS:
        accs, aurocs = [], []
        for fi, (tr, te) in enumerate(folds):
            tr_rows = [eas_rows[i] for i in tr]
            vocab = FMVocab(tr_rows, k=K)                  # fit on train only
            Xtr = vocab.encode(tr_rows)
            Xte = vocab.encode([eas_rows[i] for i in te])
            Xood = vocab.encode(ood_rows)
            clf = build(name).fit(Xtr, y[tr])
            accs.append(float((clf.predict(Xte) == y[te]).mean()))
            s_in = msp_score(clf.predict_proba(Xte))
            s_ood = msp_score(clf.predict_proba(Xood))
            aurocs.append(ood_auroc(s_in, s_ood))
        out["models"][name] = {"acc_mean": round(float(np.mean(accs)), 4),
                               "acc_std": round(float(np.std(accs)), 4),
                               "auroc_mean": round(float(np.mean(aurocs)), 4),
                               "auroc_std": round(float(np.std(aurocs)), 4)}
        r = out["models"][name]
        print(f"{name:28s} acc={r['acc_mean']*100:5.1f}±{r['acc_std']*100:4.1f}  "
              f"far-OOD AUROC={r['auroc_mean']:.3f}±{r['auroc_std']:.3f}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/dl_architectures.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/dl_architectures.json", flush=True)
    print("DL_ARCH_DONE", flush=True)


if __name__ == "__main__":
    main()
