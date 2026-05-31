"""RQ7 — CLEAN in-callset external validation (gnomAD HGDP+1KG, GRCh38).

The preliminary RQ7 (scripts/23) trained on 1000G EAS (hg19) and tested on HGDP
(hg38), suffering 43% unseen diplotypes from the build mismatch + only 510
markers. Here BOTH cohorts come from the SAME harmonized gnomAD callset (GRCh38,
full 3,042 markers) -> no liftover, no build mismatch. 3-class mapping (as in
scripts/23): train 1KG {CHB+CHS->Han, JPT->Japanese, CDX->Dai} (KHV dropped),
test HGDP {Han, Japanese, Dai}. LogReg(one-hot). Reports transfer accuracy,
per-class recall, confusion, unseen-diplotype fraction (should be ~0 now),
within-HGDP CV.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from forensic_mh.data.encoding import DiplotypeEncoder
from forensic_mh.pipelines.baseline import collect_genome_wide_strings, discover_chrom_vcfs

META = "data/hgdp/gnomad_meta_v1.tsv"
KMAP = {"CHB": "Han", "CHS": "Han", "JPT": "Japanese", "CDX": "Dai"}   # KHV dropped
HGDP3 = {"Han", "Japanese", "Dai"}


def main() -> None:
    rmap, names = collect_genome_wide_strings(discover_chrom_vcfs("data/hgdp1kg", prefix="HGDP1KG_chr"), build="hg38")
    sids = sorted(rmap)
    pop = {r["s"]: r["hgdp_tgp_meta.Population"] for r in csv.DictReader(open(META), delimiter="\t")}

    tr_idx, ytr, te_idx, yte = [], [], [], []
    for i, s in enumerate(sids):
        p = pop.get(s)
        if not s.startswith("HGDP") and p in KMAP:          # 1KG-EAS train (mapped)
            tr_idx.append(i); ytr.append(KMAP[p])
        elif s.startswith("HGDP") and p in HGDP3:           # HGDP-EAS test
            te_idx.append(i); yte.append(p)
    M = len(names)
    rows = lambda idx: np.array([[rmap[sids[i]].get(m, "N|N") for m in names] for i in idx], dtype=object)
    Xtr, Xte = rows(tr_idx), rows(te_idx)
    ytr, yte = np.array(ytr), np.array(yte)
    print(f"markers={M} (full, hg38, in-callset)", flush=True)
    print(f"train 1KG n={len(ytr)} {dict(zip(*np.unique(ytr, return_counts=True)))}", flush=True)
    print(f"test HGDP n={len(yte)} {dict(zip(*np.unique(yte, return_counts=True)))}", flush=True)

    # unseen-diplotype fraction (should be ~0 in-callset, vs 43% cross-build)
    enc = DiplotypeEncoder(); enc.fit([list(r) for r in Xtr]); enc.transform([list(r) for r in Xte])
    unseen = round(enc.last_unseen_fraction, 4)

    pipe = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=3000)).fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    acc = float((pred == yte).mean())
    labels = sorted(set(ytr) | set(yte))
    cm = confusion_matrix(yte, pred, labels=labels)
    per_class = {labels[i]: round(cm[i, i] / max(cm[i].sum(), 1), 4) for i in range(len(labels))}

    cv = StratifiedKFold(3, shuffle=True, random_state=42)
    wh = cross_val_score(make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                       LogisticRegression(max_iter=3000)), Xte, yte, cv=cv, scoring="accuracy")

    out = {"in_callset": True, "build": "hg38", "markers": M, "n_train_1kg": len(ytr), "n_test_hgdp": len(yte),
           "classes": labels, "transfer_accuracy": round(acc, 4), "per_class_recall": per_class,
           "confusion_matrix": cm.tolist(), "unseen_diplotype_fraction": unseen,
           "within_hgdp_cv": {"mean": round(float(wh.mean()), 4), "std": round(float(wh.std()), 4)},
           "vs_crossbuild": {"acc_510markers": 0.824, "unseen": 0.43}}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/rq7_incallset.json").write_text(json.dumps(out, indent=2))
    print(f"\nTRANSFER (in-callset, 3042 markers) acc={acc:.4f}  per-class={per_class}", flush=True)
    print(f"unseen-diplotype fraction={unseen}  (cross-build was 0.43)", flush=True)
    print(f"within-HGDP 3-fold CV={out['within_hgdp_cv']}", flush=True)
    print("confusion (rows=true):", labels, cm.tolist(), flush=True)
    print("saved results/baseline/rq7_incallset.json", flush=True)
    print("RQ7_INCALLSET_DONE", flush=True)


if __name__ == "__main__":
    main()
