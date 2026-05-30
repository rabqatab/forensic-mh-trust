"""External-cohort generalization: train LogReg(one-hot) on 1000G EAS, test on HGDP.

3-class mapping (only overlapping pops): CHB+CHS->Han, JPT->Japanese, CDX->Dai
(KHV dropped — no HGDP match). 1000G is hg19, HGDP is hg38; markers matched by
NAME (common MicroHapDB markers), alleles compared as diplotype strings — the
OneHotEncoder(handle_unknown='ignore') absorbs build/allele mismatches, and we
report the unseen-diplotype fraction as a harmonization-quality metric.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from forensic_mh.data.encoding import DiplotypeEncoder
from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
MAP3 = {"CHB": "Han", "CHS": "Han", "JPT": "Japanese", "CDX": "Dai"}  # KHV dropped
SEED = 42


def main() -> None:
    # 1000G EAS (hg19), remap to 3 classes
    rk, names_k = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sk = sorted(rk)
    yk_int, pops = load_eas_labels(PANEL, sk)
    lab_k, keep_k = [], []
    for i, s in enumerate(sk):
        pn = pops[yk_int[i]]
        if pn in MAP3:
            lab_k.append(MAP3[pn]); keep_k.append(s)

    # HGDP (hg38)
    rh, names_h = collect_genome_wide_strings(discover_chrom_vcfs("data/hgdp", prefix="HGDP_chr"), build="hg38")
    sh = sorted(rh)
    popmap = dict(l.split("\t") for l in Path("data/hgdp/HGDP_eas_pop.tsv").read_text().split("\n") if l)
    lab_h = [popmap[s] for s in sh]

    common = [m for m in names_k if m in set(names_h)]
    Xk = np.array([[rk[s].get(m, "N|N") for m in common] for s in keep_k], dtype=object)
    yk = np.array(lab_k)
    Xh = np.array([[rh[s].get(m, "N|N") for m in common] for s in sh], dtype=object)
    yh = np.array(lab_h)
    print(f"common markers={len(common)} (1000G {len(names_k)} / HGDP {len(names_h)})", flush=True)
    print(f"1000G train n={len(yk)} {dict(zip(*np.unique(yk, return_counts=True)))}", flush=True)
    print(f"HGDP test  n={len(yh)} {dict(zip(*np.unique(yh, return_counts=True)))}", flush=True)

    # build-harmonization quality: HGDP diplotype cells unseen in 1000G
    enc = DiplotypeEncoder(); enc.fit([list(r) for r in Xk]); enc.transform([list(r) for r in Xh])
    unseen = round(enc.last_unseen_fraction, 4)

    pipe = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=3000))
    pipe.fit(Xk, yk)
    pred = pipe.predict(Xh)
    acc = float((pred == yh).mean())
    labels = sorted(set(yk) | set(yh))
    cm = confusion_matrix(yh, pred, labels=labels)
    per_class = {labels[i]: round(cm[i, i] / max(cm[i].sum(), 1), 4) for i in range(len(labels))}

    # within-HGDP CV (3-fold; Dai n=5)
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    wh = cross_val_score(make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                        LogisticRegression(max_iter=3000)), Xh, yh, cv=cv, scoring="accuracy")

    out = {"common_markers": len(common), "n_train_1000g": len(yk), "n_test_hgdp": len(yh),
           "classes": labels, "transfer_accuracy": round(acc, 4),
           "per_class_recall": per_class, "confusion_matrix": cm.tolist(),
           "unseen_diplotype_fraction": unseen,
           "within_hgdp_cv": {"mean": round(float(wh.mean()), 4), "std": round(float(wh.std()), 4)}}
    print(f"TRANSFER acc={acc:.4f}  per-class={per_class}  unseen={unseen}", flush=True)
    print(f"within-HGDP 3-fold CV={out['within_hgdp_cv']}", flush=True)
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/hgdp_transfer.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/hgdp_transfer.json", flush=True)


if __name__ == "__main__":
    main()
