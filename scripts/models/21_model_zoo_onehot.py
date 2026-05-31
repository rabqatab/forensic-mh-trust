"""Clean model zoo — ALL models on the CORRECT encoding (one-hot, NO scaler).

Earlier comparison was confounded: trees used ordinal codes, linear/distance
used one-hot + StandardScaler (Appendix A). Here every model uses an in-pipeline
OneHotEncoder with no scaler, leakage-free 5-fold CV. Gives the true ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
SEED = 42


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    Xstr = np.array([[rows[s].get(m, "N|N") for m in names] for s in sids], dtype=object)
    y, pops = load_eas_labels(PANEL, sids)
    print(f"panel={len(names)} EAS={len(sids)}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    def oh(model):
        return make_pipeline(OneHotEncoder(handle_unknown="ignore"), model)  # NO scaler

    models = {
        "LogReg": LogisticRegression(max_iter=3000),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                 eval_metric="mlogloss", verbosity=0, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=SEED),
        "SVM-RBF": SVC(kernel="rbf", random_state=SEED),
        "kNN": KNeighborsClassifier(n_neighbors=15),
    }
    out = {"panel": len(names), "n": len(sids), "encoding": "one-hot (no scaler)", "results": {}}
    for name, model in models.items():
        s = cross_val_score(oh(model), Xstr, y, cv=cv, scoring="accuracy")
        out["results"][name] = {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4)}
        print(f"[{name}] {out['results'][name]['mean']*100:.1f}% +/- {out['results'][name]['std']*100:.1f}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/model_zoo_onehot.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/model_zoo_onehot.json", flush=True)


if __name__ == "__main__":
    main()
