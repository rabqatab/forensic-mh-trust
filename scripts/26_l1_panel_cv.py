"""B-followup — does L1-LogReg's jointly-selected sparse panel recover full accuracy?

§21 showed MI-top-N (individually ranked) has no plateau: 327 markers ~57%.
But L1 selects markers *jointly* (interaction-aware). This measures, leakage-free,
(a) the CV accuracy of one-hot L1-LogReg and (b) how many markers it keeps per fold
— answering whether a ~10x reduced panel reaches ~79%.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
SEED = 42
CS = [0.1, 0.2, 0.5]   # L1 strengths (smaller C = sparser)


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    Xstr = np.array(eas_rows, dtype=object)
    y, _ = load_eas_labels(PANEL, sids)
    M = len(names)
    print(f"panel={M} EAS={len(sids)}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out = {"panel_total": M, "by_C": {}}
    for C in CS:
        accs, nmarkers = [], []
        for fi, (tr, te) in enumerate(cv.split(Xstr, y)):
            oh = OneHotEncoder(handle_unknown="ignore").fit(Xstr[tr])
            Xtr, Xte = oh.transform(Xstr[tr]), oh.transform(Xstr[te])
            feat_marker = np.repeat(np.arange(M), [len(c) for c in oh.categories_])
            clf = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=3000)
            clf.fit(Xtr, y[tr])
            accs.append(float((clf.predict(Xte) == y[te]).mean()))
            used = np.unique(feat_marker[(np.abs(clf.coef_).sum(0) > 1e-8)])
            nmarkers.append(int(len(used)))
            print(f"  C={C} fold{fi+1}: acc={accs[-1]:.3f} markers={nmarkers[-1]}", flush=True)
        out["by_C"][str(C)] = {
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "markers_mean": round(float(np.mean(nmarkers)), 1),
        }
        r = out["by_C"][str(C)]
        print(f"C={C}: acc={r['accuracy_mean']}±{r['accuracy_std']} "
              f"markers~{r['markers_mean']}/{M}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/l1_panel_cv.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/l1_panel_cv.json", flush=True)
    print("L1_PANEL_DONE", flush=True)


if __name__ == "__main__":
    main()
