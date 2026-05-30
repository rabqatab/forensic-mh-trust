"""Modern tabular-DL SOTA (RQ3) — FT-Transformer + TabNet, GPU.

Our earlier transformer/MLP might undersell DL. Here the canonical tabular-DL
SOTA baselines:
- FT-Transformer (Gorishniy et al. 2021): feature tokenizer + [CLS] + transformer.
- TabNet (Arik & Pfister 2021): sequential attentive feature selection.
Same protocol: genome-wide, leakage-free 5-fold, acc + far-OOD MSP AUROC.
If even these lose to LogReg(one-hot) 79.6%, RQ3 (simplicity wins) is bulletproof.
GPU job — submit via sparkq.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from forensic_mh.fm.architectures import TorchArchClassifier
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


def run_tabnet(Xtr, ytr, Xte, Xood, M, n_classes):
    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.12, stratify=ytr, random_state=0)
    clf = TabNetClassifier(cat_idxs=list(range(M)), cat_dims=[K] * M, cat_emb_dim=4,
                           device_name=DEV, seed=0, verbose=0)
    clf.fit(Xt, yt, eval_set=[(Xv, yv)], max_epochs=120, patience=20,
            batch_size=128, virtual_batch_size=64)
    return clf.predict(Xte), clf.predict_proba(Xte), clf.predict_proba(Xood)


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
    res = {"FT-Transformer": ([], []), "TabNet": ([], [])}
    for fi, (tr, te) in enumerate(cv.split(eas_rows, y)):
        tr_rows = [eas_rows[i] for i in tr]
        vocab = FMVocab(tr_rows, k=K)
        Xtr = vocab.encode(tr_rows)
        Xte = vocab.encode([eas_rows[i] for i in te])
        Xood = vocab.encode(ood_rows)

        # FT-Transformer
        ft = TorchArchClassifier(arch="fttransformer", k=K, d=32, epochs=150, device=DEV).fit(Xtr, y[tr])
        pte, pood = ft.predict_proba(Xte), ft.predict_proba(Xood)
        res["FT-Transformer"][0].append(float((pte.argmax(1) == y[te]).mean()))
        res["FT-Transformer"][1].append(ood_auroc(msp_score(pte), msp_score(pood)))

        # TabNet
        pred, ptte, ptood = run_tabnet(Xtr, y[tr], Xte, Xood, M, len(classes))
        res["TabNet"][0].append(float((np.asarray(pred).ravel() == y[te]).mean()))
        res["TabNet"][1].append(ood_auroc(msp_score(ptte), msp_score(ptood)))
        print(f"fold {fi+1}: FT={res['FT-Transformer'][0][-1]:.3f} TabNet={res['TabNet'][0][-1]:.3f}", flush=True)

    out = {"panel": M, "device": DEV, "models": {}}
    for name, (a, u) in res.items():
        out["models"][name] = {"acc_mean": round(float(np.mean(a)), 4),
                               "acc_std": round(float(np.std(a)), 4),
                               "auroc_mean": round(float(np.mean(u)), 4)}
        r = out["models"][name]
        print(f"{name:16s} acc={r['acc_mean']*100:5.1f}±{r['acc_std']*100:4.1f}  "
              f"far-OOD AUROC={r['auroc_mean']:.3f}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/sota_dl.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/sota_dl.json", flush=True)
    print("SOTA_DL_DONE", flush=True)


if __name__ == "__main__":
    main()
