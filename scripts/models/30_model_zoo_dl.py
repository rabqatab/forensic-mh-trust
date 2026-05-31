"""Unified trust-aware model zoo — classical baselines + DL architectures.

Earlier comparisons were unfair: classical models were 5-fold one-hot (§13) but
DL was evaluated under different protocols (MLP ensemble on a 70/30 split; the
SSL transformer on a 256-marker panel). Here EVERY model runs the SAME protocol:
genome-wide one-hot, leakage-free 5-fold CV, reporting accuracy + far-OOD MSP
AUROC (open-set separation). Adds DL (MLP 1/2/deep layers) and extra baselines
(BernoulliNB, ExtraTrees) to LogReg/kNN/RF/XGBoost. Ties to RQ1 (does each
architecture separate OOD?) and RQ3 (does complexity beat the linear model?).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.openset import msp_score, ood_auroc

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def models():
    def mlp(layers):
        return MLPClassifier(hidden_layer_sizes=layers, max_iter=300, early_stopping=True,
                             n_iter_no_change=15, random_state=0)
    return {
        # --- baselines ---
        "LogReg(one-hot)": LogisticRegression(max_iter=2000),
        "BernoulliNB": BernoulliNB(),
        "kNN": KNeighborsClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=0),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=400, random_state=0),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                 eval_metric="mlogloss", verbosity=0, random_state=0),
        # --- DL architectures (on one-hot) ---
        "MLP-1 (256)": mlp((256,)),
        "MLP-2 (256,128)": mlp((256, 128)),
        "MLP-deep (512,256,128)": mlp((512, 256, 128)),
    }


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    Xstr = np.array([[rows[s].get(m, "N|N") for m in names] for s in sids], dtype=object)
    y, _ = load_eas_labels(PANEL, sids)
    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    Ostr = np.array([[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)], dtype=object)
    print(f"panel={len(names)} EAS={len(sids)} OOD={Ostr.shape[0]} chance=0.20", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    out = {"panel": len(names), "n_eas": len(sids), "n_ood": int(Ostr.shape[0]), "models": {}}
    for name, est in models().items():
        accs, aurocs = [], []
        for tr, te in cv.split(Xstr, y):
            pipe = make_pipeline(OneHotEncoder(handle_unknown="ignore"), est)
            pipe.fit(Xstr[tr], y[tr])
            accs.append(float((pipe.predict(Xstr[te]) == y[te]).mean()))
            s_in = msp_score(pipe.predict_proba(Xstr[te]))
            s_ood = msp_score(pipe.predict_proba(Ostr))
            aurocs.append(ood_auroc(s_in, s_ood))
        out["models"][name] = {"acc_mean": round(float(np.mean(accs)), 4),
                               "acc_std": round(float(np.std(accs)), 4),
                               "auroc_mean": round(float(np.mean(aurocs)), 4),
                               "auroc_std": round(float(np.std(aurocs)), 4)}
        r = out["models"][name]
        print(f"{name:24s} acc={r['acc_mean']*100:5.1f}±{r['acc_std']*100:.1f}  "
              f"far-OOD AUROC={r['auroc_mean']:.3f}±{r['auroc_std']:.3f}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/model_zoo_dl.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/model_zoo_dl.json", flush=True)
    print("MODEL_ZOO_DL_DONE", flush=True)


if __name__ == "__main__":
    main()
