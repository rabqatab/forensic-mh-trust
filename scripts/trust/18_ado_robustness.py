"""Thread 2 — degraded-DNA (ADO) robustness of the trust layer.

Train RandomForest + Mondrian conformal on CLEAN EAS data, then evaluate on a
test set with simulated allele-dropout (ADO: het→hom) at increasing rates.
Reports how accuracy, conformal coverage, set-size, and far-OOD MSP AUROC
degrade — the forensic-realism angle (degraded samples) competitors ignore.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from forensic_mh.data.encoding import DiplotypeEncoder
from forensic_mh.fm.dataset import _ado  # het->hom allele dropout (module-level)
from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.conformal import empirical_coverage, mean_set_size
from forensic_mh.uq.conformal_classifier import ConformalClassifier
from forensic_mh.uq.openset import msp_score, ood_auroc

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
ALPHA, SEED = 0.10, 42
RATES = [0.0, 0.1, 0.2, 0.3, 0.5]


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    enc = DiplotypeEncoder()
    enc.fit(eas_rows)
    y, pops = load_eas_labels(PANEL, sids)

    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    Xood = enc.transform([[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)])
    print(f"panel={len(names)} EAS={len(sids)} OOD={len(sorted(orows))}", flush=True)

    # split on string rows so we can apply ADO to the raw test diplotypes
    idx = np.arange(len(sids))
    tr, te = train_test_split(idx, test_size=0.3, stratify=y, random_state=SEED)
    Xtr = enc.transform([eas_rows[i] for i in tr])
    ytr, yte = y[tr], y[te]
    test_rows = [eas_rows[i] for i in te]

    cc = ConformalClassifier(RandomForestClassifier(n_estimators=400, random_state=SEED),
                             alpha=ALPHA, mondrian=True).fit(Xtr, ytr)
    rng = np.random.default_rng(SEED)

    out = {"panel": len(names), "alpha": ALPHA, "rates": []}
    for p in RATES:
        ado_rows = [_ado(r, rng, p) for r in test_rows]
        Xte = enc.transform(ado_rows)
        proba = cc.predict_proba(Xte)
        acc = float((cc.classes_[proba.argmax(1)] == yte).mean())
        sets = cc.predict_set(Xte)
        auroc = ood_auroc(msp_score(proba), msp_score(cc.predict_proba(Xood)))
        rec = {"ado_rate": p, "accuracy": round(acc, 4),
               "coverage": round(empirical_coverage(sets, yte), 4),
               "set_size": round(mean_set_size(sets), 4),
               "msp_auroc_vs_ood": round(auroc, 4)}
        out["rates"].append(rec)
        print(f"[ADO {p}] {rec}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/ado_robustness.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/ado_robustness.json", flush=True)


if __name__ == "__main__":
    main()
