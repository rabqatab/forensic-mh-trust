"""Broader linear family (RQ3) — is it really the LINEAR CLASS, not just LogReg?

§24.4 had LogReg variants + LinearSVC + SGD. Here we add the rest of the linear
family with DISTINCT loss/decision rules: Ridge (squared loss), Passive-Aggressive
& Perceptron (margin/mistake-driven), Nearest-Centroid (Euclidean to class means),
Multinomial NB (log-linear), and an attempt at shrinkage-LDA (Fisher discriminant,
the classic popgen linear method). Same protocol: one-hot, leakage-free 5-fold,
acc + far-OOD MSP AUROC. Non-probabilistic models are wrapped in
CalibratedClassifierCV for proba (Nearest-Centroid: accuracy only).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.openset import msp_score, ood_auroc

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def oh(est):
    return make_pipeline(OneHotEncoder(handle_unknown="ignore"), est)


def cal(est):
    return make_pipeline(OneHotEncoder(handle_unknown="ignore"), CalibratedClassifierCV(est, cv=3))


def models():
    return {
        "Ridge (squared loss)": ("proba", cal(RidgeClassifier())),
        "Passive-Aggressive": ("proba", cal(PassiveAggressiveClassifier(max_iter=2000, tol=1e-3))),
        "Perceptron": ("proba", cal(Perceptron(max_iter=2000, tol=1e-3))),
        "Multinomial NB": ("proba", oh(MultinomialNB())),
        "Nearest-Centroid": ("acc", oh(NearestCentroid())),
        "LDA (shrinkage)": ("lda", None),   # attempted with dense + guard
    }


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    Xstr = np.array(eas_rows, dtype=object)
    y, _ = load_eas_labels(PANEL, sids)
    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    Ostr = np.array([[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)], dtype=object)
    print(f"panel={len(names)} EAS={len(sids)} OOD={Ostr.shape[0]}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    out = {"models": {}}
    from sklearn.base import clone
    for name, (kind, est) in models().items():
        accs, aurocs = [], []
        try:
            for tr, te in cv.split(Xstr, y):
                if kind == "lda":
                    oh_e = OneHotEncoder(handle_unknown="ignore").fit(Xstr[tr])
                    Xtr, Xte, Xo = (oh_e.transform(Xstr[tr]).toarray(),
                                    oh_e.transform(Xstr[te]).toarray(), oh_e.transform(Ostr).toarray())
                    m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, y[tr])
                    accs.append(float((m.predict(Xte) == y[te]).mean()))
                    aurocs.append(ood_auroc(msp_score(m.predict_proba(Xte)), msp_score(m.predict_proba(Xo))))
                else:
                    m = clone(est).fit(Xstr[tr], y[tr])
                    accs.append(float((m.predict(Xstr[te]) == y[te]).mean()))
                    if kind == "proba":
                        aurocs.append(ood_auroc(msp_score(m.predict_proba(Xstr[te])), msp_score(m.predict_proba(Ostr))))
        except Exception as e:
            out["models"][name] = {"status": f"infeasible: {type(e).__name__}: {str(e)[:80]}"}
            print(f"{name:24s} INFEASIBLE ({type(e).__name__})", flush=True)
            continue
        r = {"acc_mean": round(float(np.mean(accs)), 4), "acc_std": round(float(np.std(accs)), 4)}
        if aurocs:
            r["auroc_mean"] = round(float(np.mean(aurocs)), 4)
        out["models"][name] = r
        au = f"far-OOD AUROC={r['auroc_mean']:.3f}" if aurocs else "(no proba)"
        print(f"{name:24s} acc={r['acc_mean']*100:5.1f}±{r['acc_std']*100:.1f}  {au}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/linear_family.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/linear_family.json", flush=True)
    print("LINEAR_FAMILY_DONE", flush=True)


if __name__ == "__main__":
    main()
