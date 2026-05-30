"""More DL paradigms (RQ3) — ResNet-tabular + deep residual CNN, GPU.

Fills two gaps in §24.2: ResNet-tabular (Gorishniy et al. 2021 — the co-SOTA
paired with FT-Transformer) and a deeper residual 1D-CNN (genomatnn/Flagel-
faithful, vs our shallow CNN1D). Same protocol: genome-wide, leakage-free 5-fold,
acc + far-OOD MSP AUROC. GPU job (sparkq).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from forensic_mh.fm.architectures import TorchArchClassifier
from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import collect_genome_wide_strings, discover_chrom_vcfs, load_eas_labels
from forensic_mh.uq.openset import msp_score, ood_auroc

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
K = 8
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {"ResNet-tabular": "resnettab", "ResCNN (deep popgen-CNN)": "rescnn"}


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    y, _ = load_eas_labels(PANEL, sids)
    y = np.asarray(y)
    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    ood_rows = [[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)]
    print(f"panel={len(names)} EAS={len(sids)} OOD={len(ood_rows)} device={DEV}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = list(cv.split(eas_rows, y))
    out = {"panel": len(names), "device": DEV, "models": {}}
    for name, arch in MODELS.items():
        accs, aurocs = [], []
        for fi, (tr, te) in enumerate(folds):
            tr_rows = [eas_rows[i] for i in tr]
            vocab = FMVocab(tr_rows, k=K)
            Xtr, Xte, Xood = (vocab.encode(tr_rows),
                              vocab.encode([eas_rows[i] for i in te]),
                              vocab.encode(ood_rows))
            clf = TorchArchClassifier(arch=arch, k=K, d=32, epochs=150, device=DEV).fit(Xtr, y[tr])
            pte, pood = clf.predict_proba(Xte), clf.predict_proba(Xood)
            accs.append(float((pte.argmax(1) == y[te]).mean()))
            aurocs.append(ood_auroc(msp_score(pte), msp_score(pood)))
        out["models"][name] = {"acc_mean": round(float(np.mean(accs)), 4),
                               "acc_std": round(float(np.std(accs)), 4),
                               "auroc_mean": round(float(np.mean(aurocs)), 4)}
        r = out["models"][name]
        print(f"{name:26s} acc={r['acc_mean']*100:5.1f}±{r['acc_std']*100:4.1f}  "
              f"far-OOD AUROC={r['auroc_mean']:.3f}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/dl_resnet_cnn.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/dl_resnet_cnn.json", flush=True)
    print("DL_RESNET_CNN_DONE", flush=True)


if __name__ == "__main__":
    main()
