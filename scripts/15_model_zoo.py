"""Model zoo benchmark vs XGBoost on the FULL genome-wide EAS panel.

Classical ML approaches (LogReg, kNN, SVM-RBF, RandomForest) + XGBoost, all on
the same genome-wide MH panel. Tree models use ordinal FMVocab codes; linear/
distance models use a one-hot Pipeline (their correct categorical encoding).
Accuracy via leakage-free StratifiedKFold(5); coverage + far-OOD via
ConformalClassifier on a 70/30 split. FM is compared separately (scripts/14).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.conformal import empirical_coverage, mean_set_size
from forensic_mh.uq.conformal_classifier import ConformalClassifier
from forensic_mh.uq.openset import msp_score, ood_auroc

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
K_VOCAB, ALPHA, SEED = 16, 0.10, 42


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    Xstr = np.array([[rows[s].get(m, "N|N") for m in names] for s in sids], dtype=object)
    y, pops = load_eas_labels(PANEL, sids)
    enc = FMVocab([list(r) for r in Xstr], k=K_VOCAB)
    Xord = enc.encode([list(r) for r in Xstr])

    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    osids = sorted(orows)
    Ostr = np.array([[orows[s].get(m, "N|N") for m in names] for s in osids], dtype=object)
    Oord = enc.encode([list(r) for r in Ostr])
    print(f"panel={len(names)} EAS={len(sids)} OOD={len(osids)}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    def oh(model):
        return make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                             StandardScaler(with_mean=False), model)

    specs = {
        "XGBoost":      (XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                       eval_metric="mlogloss", verbosity=0, random_state=SEED), "ord"),
        "RandomForest": (RandomForestClassifier(n_estimators=400, random_state=SEED), "ord"),
        "LogReg":       (oh(LogisticRegression(max_iter=2000)), "str"),
        "kNN":          (oh(KNeighborsClassifier(n_neighbors=15)), "str"),
        "SVM-RBF":      (oh(SVC(kernel="rbf", probability=True, random_state=SEED)), "str"),
    }

    out = {"panel": len(names), "n_eas": len(sids), "n_ood": len(osids), "alpha": ALPHA, "models": {}}
    for name, (model, rep) in specs.items():
        Xacc = Xord if rep == "ord" else Xstr
        Xood = Oord if rep == "ord" else Ostr
        acc = cross_val_score(model, Xacc, y, cv=cv, scoring="accuracy")
        Xtr, Xte, ytr, yte = train_test_split(Xacc, y, test_size=0.30, stratify=y, random_state=SEED)
        cc = ConformalClassifier(model, alpha=ALPHA, mondrian=True).fit(Xtr, ytr)
        sets = cc.predict_set(Xte)
        s_in, s_ood = msp_score(cc.predict_proba(Xte)), msp_score(cc.predict_proba(Xood))
        out["models"][name] = {
            "cv_acc_mean": round(float(acc.mean()), 4), "cv_acc_std": round(float(acc.std()), 4),
            "coverage": round(empirical_coverage(sets, yte), 4),
            "set_size": round(mean_set_size(sets), 4),
            "msp_auroc": round(ood_auroc(s_in, s_ood), 4)}
        print(f"[{name}] {out['models'][name]}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/model_zoo.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/model_zoo.json", flush=True)


if __name__ == "__main__":
    main()
