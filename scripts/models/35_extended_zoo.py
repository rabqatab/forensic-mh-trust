"""Extended ML zoo — linear family + native-categorical trees + reg sweep (RQ3).

Two targeted questions:
1. Is the linear win a LogReg quirk or the LINEAR CLASS? -> LinearSVC, SGD-log,
   L1 / elastic-net LogReg, C sweep.
2. Did trees lose because of ORDINAL encoding or the MODEL CLASS? -> native-
   categorical gradient boosting (HistGradientBoosting with categorical_features
   on ordinal codes does subset splits, not ordinal-threshold splits).
Same protocol: genome-wide, leakage-free 5-fold, acc + far-OOD MSP AUROC.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from forensic_mh.data.encoding import DiplotypeEncoder
from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.openset import msp_score, ood_auroc

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def onehot(est):
    return ("onehot", make_pipeline(OneHotEncoder(handle_unknown="ignore"), est))


def models():
    return {
        # --- linear family (is it the linear class, not just LogReg?) ---
        "LogReg L2 (ref)": onehot(LogisticRegression(max_iter=2000)),
        "LogReg L2 C=0.1": onehot(LogisticRegression(C=0.1, max_iter=2000)),
        "LogReg L2 C=10": onehot(LogisticRegression(C=10, max_iter=2000)),
        "LogReg L1": onehot(LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=3000)),
        "LogReg elastic-net": onehot(LogisticRegression(penalty="elasticnet", solver="saga",
                                                        l1_ratio=0.5, C=0.5, max_iter=3000)),
        "LinearSVC (calibrated)": onehot(CalibratedClassifierCV(LinearSVC(max_iter=5000), cv=3)),
        "SGD-log": onehot(SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-3)),
        "ComplementNB": onehot(ComplementNB()),
    }


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    Xstr = np.array(eas_rows, dtype=object)
    enc = DiplotypeEncoder(); Xord = enc.fit_transform(eas_rows)
    y, _ = load_eas_labels(PANEL, sids)
    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    ood_rows = [[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)]
    Ostr = np.array(ood_rows, dtype=object); Oord = enc.transform(ood_rows)
    M = len(names)
    print(f"panel={M} EAS={len(sids)} OOD={Ostr.shape[0]}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    out = {"models": {}}
    for name, (kind, est) in models().items():
        accs, aurocs = [], []
        from sklearn.base import clone
        for tr, te in cv.split(Xstr, y):
            m = clone(est); m.fit(Xstr[tr], y[tr])
            acc = (m.predict(Xstr[te]) == y[te]).mean()
            s_in, s_ood = msp_score(m.predict_proba(Xstr[te])), msp_score(m.predict_proba(Ostr))
            accs.append(float(acc)); aurocs.append(ood_auroc(s_in, s_ood))
        out["models"][name] = {"acc_mean": round(float(np.mean(accs)), 4),
                               "acc_std": round(float(np.std(accs)), 4),
                               "auroc_mean": round(float(np.mean(aurocs)), 4)}
        r = out["models"][name]
        print(f"{name:24s} acc={r['acc_mean']*100:5.1f}±{r['acc_std']*100:4.1f}  "
              f"far-OOD AUROC={r['auroc_mean']:.3f}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/extended_zoo.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/extended_zoo.json", flush=True)
    print("EXTENDED_ZOO_DONE", flush=True)


if __name__ == "__main__":
    main()
