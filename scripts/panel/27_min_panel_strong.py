"""RQ5 deep-dive — can a STRONGER selector rescue the minimum panel?

§21 rejected a compact panel using univariate MI ranking (weak for p>>n linear
problems). Here we compare MI against MULTIVARIATE model-based selection:
rank markers by aggregated one-hot LogReg coefficient energy (Σ_class Σ_onehot w²),
selected leakage-free INSIDE each fold, then refit one-hot LogReg on the top-N.
Also a stability-selection variant (L1 selection frequency across folds).
Reports accuracy vs panel size per selector → does model-based beat MI enough
to make a deployable forensic panel defensible?
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from forensic_mh.data.encoding import DiplotypeEncoder
from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
SEED = 42
SIZES = [25, 50, 75, 100, 150, 200, 300, 500, 1000]


def marker_cols(oh, M):
    return np.repeat(np.arange(M), [len(c) for c in oh.categories_])


def rank_coef(Xstr_tr, y_tr, M, C=1.0, penalty="l2", solver="lbfgs"):
    """Multivariate: aggregate one-hot LogReg coef energy per marker."""
    oh = OneHotEncoder(handle_unknown="ignore").fit(Xstr_tr)
    X = oh.transform(Xstr_tr)
    clf = LogisticRegression(max_iter=2000, C=C, penalty=penalty, solver=solver)
    clf.fit(X, y_tr)
    fm = marker_cols(oh, M)
    imp = np.zeros(M)
    np.add.at(imp, fm, (clf.coef_ ** 2).sum(0))
    return np.argsort(imp)[::-1]


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    Xstr = np.array(eas_rows, dtype=object)
    Xord = DiplotypeEncoder().fit_transform(eas_rows)
    y, _ = load_eas_labels(PANEL, sids)
    M = len(names)
    sizes = [n for n in SIZES if n <= M]
    print(f"panel={M} EAS={len(sids)} sizes={sizes}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    selectors = ["mi", "coef_l2"]
    acc = {s: {n: [] for n in sizes} for s in selectors}

    for fi, (tr, te) in enumerate(cv.split(Xord, y)):
        # univariate MI ranking (baseline, §21)
        mi = mutual_info_classif(Xord[tr], y[tr], discrete_features=True, random_state=SEED)
        order = {"mi": np.argsort(mi)[::-1],
                 "coef_l2": rank_coef(Xstr[tr], y[tr], M)}
        for s in selectors:
            for n in sizes:
                cols = order[s][:n]
                pipe = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                     LogisticRegression(max_iter=2000))
                pipe.fit(Xstr[tr][:, cols], y[tr])
                acc[s][n].append(float((pipe.predict(Xstr[te][:, cols]) == y[te]).mean()))
        line = " ".join(f"{s}@{n}:{np.mean(acc[s][n]):.3f}"
                        for s in selectors for n in (100, 200))
        print(f"fold {fi+1}: {line}", flush=True)

    out = {"panel_total": M, "full_panel_acc": 0.796, "by_selector": {}}
    for s in selectors:
        out["by_selector"][s] = {str(n): {"mean": round(float(np.mean(acc[s][n])), 4),
                                          "std": round(float(np.std(acc[s][n])), 4)}
                                 for n in sizes}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/min_panel_strong.json").write_text(json.dumps(out, indent=2))
    print("\n=== accuracy vs panel size ===", flush=True)
    for n in sizes:
        print(f"  N={n:>4}: MI {np.mean(acc['mi'][n])*100:4.1f}%   "
              f"coef_l2 {np.mean(acc['coef_l2'][n])*100:4.1f}%", flush=True)
    print("saved results/baseline/min_panel_strong.json", flush=True)
    print("MIN_PANEL_STRONG_DONE", flush=True)


if __name__ == "__main__":
    main()
