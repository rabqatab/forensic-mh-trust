"""B — minimum MH panel with one-hot LogReg (leakage-free).

The proposal's core deliverable ("N markers for X% accuracy"), redone with the
winning model. Marker importance via mutual information (computed inside each
train fold on the ordinal matrix → leakage-free), then top-N markers one-hot ->
LogReg. Reports accuracy vs panel size + the smallest N reaching key thresholds.
Also a complementary L1-LogReg natural-sparsity point.
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
PANEL_SIZES = [10, 20, 50, 100, 200, 500, 1000]


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    Xstr = np.array(eas_rows, dtype=object)
    Xord = DiplotypeEncoder().fit_transform(eas_rows)
    y, pops = load_eas_labels(PANEL, sids)
    M = len(names)
    print(f"panel={M} EAS={len(sids)}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    sizes = [n for n in PANEL_SIZES if n <= M] + [M]
    acc = {n: [] for n in sizes}
    for fi, (tr, te) in enumerate(cv.split(Xord, y)):
        mi = mutual_info_classif(Xord[tr], y[tr], discrete_features=True, random_state=SEED)
        order = np.argsort(mi)[::-1]
        for n in sizes:
            cols = order[:n]
            pipe = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                 LogisticRegression(max_iter=2000))
            pipe.fit(Xstr[tr][:, cols], y[tr])
            acc[n].append(float((pipe.predict(Xstr[te][:, cols]) == y[te]).mean()))
        print(f"fold {fi+1}: " + " ".join(f"{n}:{np.mean(acc[n]):.3f}" for n in sizes), flush=True)

    curve = {str(n): {"mean": round(float(np.mean(acc[n])), 4), "std": round(float(np.std(acc[n])), 4)} for n in sizes}
    # smallest N reaching thresholds
    thr = {}
    for t in [0.70, 0.75, 0.78]:
        hit = [n for n in sizes if np.mean(acc[n]) >= t]
        thr[str(t)] = min(hit) if hit else None

    # complementary: L1-LogReg natural sparsity (markers with any nonzero one-hot weight)
    oh = OneHotEncoder(handle_unknown="ignore").fit(Xstr)
    Xoh = oh.transform(Xstr)
    feat_marker = np.repeat(np.arange(M), [len(c) for c in oh.categories_])
    l1 = LogisticRegression(penalty="l1", solver="saga", C=0.2, max_iter=3000)
    l1.fit(Xoh, y)
    used = np.unique(feat_marker[(np.abs(l1.coef_).sum(0) > 1e-8)])
    print(f"L1(C=0.2) uses {len(used)} of {M} markers (full-data fit, indicative)", flush=True)

    out = {"panel_total": M, "curve": curve, "min_markers_for_threshold": thr,
           "l1_markers_used": int(len(used))}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/min_panel_logreg.json").write_text(json.dumps(out, indent=2))
    print("min-panel curve:", {n: curve[str(n)]["mean"] for n in sizes}, flush=True)
    print("min markers for [70/75/78]%:", thr, flush=True)
    print("saved results/baseline/min_panel_logreg.json", flush=True)


if __name__ == "__main__":
    main()
