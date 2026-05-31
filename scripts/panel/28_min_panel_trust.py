"""RQ5 — the minimum FORENSIC panel: trust metrics vs panel size.

§27 showed multivariate model-based selection front-loads accuracy (25 markers
→ 52%, 1000 → 77%). A forensic panel's value, though, is TRUST, not top-1.
Here, for each panel size (model-based selection, leakage-free), we report
conformal coverage + mean set size + far-OOD AUROC + empty-set reject + accuracy
(multi-seed), and emit a FIXED deployable panel (top-N markers ranked on all EAS)
as the actual deliverable. Defines the minimum panel as the smallest N meeting a
forensic trust spec (coverage >= 0.90 at acceptable set size).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.conformal import empirical_coverage, mean_set_size
from forensic_mh.uq.conformal_classifier import ConformalClassifier
from forensic_mh.uq.openset import msp_score, ood_auroc, reject_rate

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
ALPHA = 0.10
SIZES = [25, 50, 100, 200, 300, 500]
SEEDS = list(range(5))


def rank_coef(Xstr_tr, y_tr, M):
    oh = OneHotEncoder(handle_unknown="ignore").fit(Xstr_tr)
    X = oh.transform(Xstr_tr)
    fm = np.repeat(np.arange(M), [len(c) for c in oh.categories_])
    clf = LogisticRegression(max_iter=2000).fit(X, y_tr)
    imp = np.zeros(M)
    np.add.at(imp, fm, (clf.coef_ ** 2).sum(0))
    return np.argsort(imp)[::-1]


def pipe():
    return make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                         LogisticRegression(max_iter=2000))


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    Xstr = np.array([[rows[s].get(m, "N|N") for m in names] for s in sids], dtype=object)
    y, _ = load_eas_labels(PANEL, sids)
    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    Ostr = np.array([[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)], dtype=object)
    M = len(names)
    print(f"panel={M} EAS={len(sids)} OOD={Ostr.shape[0]}", flush=True)

    # 3-way split to keep marker SELECTION disjoint from conformal CALIBRATION:
    #   Xsel (select markers) | Xfc (estimator train + calibration, internal) | Xte (test).
    # Selecting on the same data conformal calibrates on leaks cal labels into the
    # score function and breaks coverage (under-covers worse as N grows) — see docs/06.
    frontier = {n: {k: [] for k in ["acc", "coverage", "set_size", "msp_auroc", "reject_ood"]}
                for n in SIZES}
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(Xstr, y, test_size=0.25, stratify=y, random_state=seed)
        Xsel, Xfc, ysel, yfc = train_test_split(Xtr, ytr, test_size=2/3, stratify=ytr, random_state=seed)
        order = rank_coef(Xsel, ysel, M)          # selection on design split only
        for n in SIZES:
            cols = order[:n]
            cc = ConformalClassifier(pipe(), alpha=ALPHA, mondrian=True).fit(Xfc[:, cols], yfc)
            proba = cc.predict_proba(Xte[:, cols])
            sets_in = cc.predict_set(Xte[:, cols])
            sets_ood = cc.predict_set(Ostr[:, cols])
            s_in, s_ood = msp_score(proba), msp_score(cc.predict_proba(Ostr[:, cols]))
            pred = cc.classes_[proba.argmax(1)]
            frontier[n]["acc"].append(float((pred == yte).mean()))
            frontier[n]["coverage"].append(empirical_coverage(sets_in, yte))
            frontier[n]["set_size"].append(mean_set_size(sets_in))
            frontier[n]["msp_auroc"].append(ood_auroc(s_in, s_ood))
            frontier[n]["reject_ood"].append(reject_rate(sets_ood))
        print(f"seed {seed} done", flush=True)

    # fixed deployable panels (ranked on ALL EAS) — the actual deliverable
    order_full = rank_coef(Xstr, y, M)
    panels = {str(n): [names[i] for i in order_full[:n]] for n in (50, 100, 200)}

    out = {"panel_total": M, "alpha": ALPHA, "n_seeds": len(SEEDS),
           "selector": "model-based one-hot LogReg coef energy",
           "frontier": {str(n): {k: {"mean": round(float(np.mean(v)), 4),
                                     "std": round(float(np.std(v)), 4)}
                                 for k, v in frontier[n].items()} for n in SIZES},
           "fixed_panels": panels}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/min_panel_trust.json").write_text(json.dumps(out, indent=2))
    print("\n=== forensic frontier (model-based selection, alpha=0.10) ===", flush=True)
    print(f'{"N":>5} | {"acc":>6} | {"cover":>6} | {"set":>5} | {"AUROC":>6} | {"rejOOD":>6}', flush=True)
    for n in SIZES:
        r = out["frontier"][str(n)]
        print(f'{n:>5} | {r["acc"]["mean"]*100:5.1f}% | {r["coverage"]["mean"]:.3f} | '
              f'{r["set_size"]["mean"]:.2f} | {r["msp_auroc"]["mean"]:.3f} | {r["reject_ood"]["mean"]:.3f}',
              flush=True)
    print("saved results/baseline/min_panel_trust.json", flush=True)
    print("MIN_PANEL_TRUST_DONE", flush=True)


if __name__ == "__main__":
    main()
