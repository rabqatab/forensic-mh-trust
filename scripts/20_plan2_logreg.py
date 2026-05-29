"""Plan 2 metrics with LogReg(one-hot) base vs RF/XGBoost (side-by-side).

LogReg consumes raw diplotype strings via an in-pipeline OneHotEncoder (NO
StandardScaler — see Appendix A); tree models consume ordinal DiplotypeEncoder
codes. Same genome-wide data and same outer split. Reports conformal coverage
curve, far-OOD (MSP AUROC + empty-set reject + FPR@95), and LOPO near-OOD.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from forensic_mh.data.encoding import DiplotypeEncoder
from forensic_mh.eval.lopo import leave_one_population_out
from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.conformal import empirical_coverage, mean_set_size
from forensic_mh.uq.conformal_classifier import ConformalClassifier
from forensic_mh.uq.openset import fpr_at_tpr, msp_score, ood_auroc, reject_rate

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
SEED = 42


def make(name):
    if name == "XGBoost":
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             eval_metric="mlogloss", verbosity=0, random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=400, random_state=SEED)
    return make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                         LogisticRegression(max_iter=3000))


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    Xstr = np.array(eas_rows, dtype=object)
    enc = DiplotypeEncoder()
    Xord = enc.fit_transform(eas_rows)
    y, pops = load_eas_labels(PANEL, sids)

    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    ood_rows = [[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)]
    Ostr = np.array(ood_rows, dtype=object)
    Oord = enc.transform(ood_rows)
    print(f"panel={len(names)} EAS={len(sids)} OOD={len(ood_rows)}", flush=True)

    itr, ite = train_test_split(np.arange(len(sids)), test_size=0.3, stratify=y, random_state=SEED)
    out = {"panel": len(names), "models": {}}

    for name in ["LogReg(one-hot)", "RandomForest", "XGBoost"]:
        rep_str = name.startswith("LogReg")
        X = Xstr if rep_str else Xord
        Xood = Ostr if rep_str else Oord
        res = {"coverage_curve": [], "far_ood": [], "lopo": []}
        for alpha in [0.30, 0.20, 0.10, 0.05]:
            cc = ConformalClassifier(make(name), alpha=alpha, mondrian=True).fit(X[itr], y[itr])
            sets_in, sets_ood = cc.predict_set(X[ite]), cc.predict_set(Xood)
            s_in, s_ood = msp_score(cc.predict_proba(X[ite])), msp_score(cc.predict_proba(Xood))
            res["coverage_curve"].append({"alpha": alpha,
                "coverage": round(empirical_coverage(sets_in, y[ite]), 4),
                "set_size": round(mean_set_size(sets_in), 4)})
            res["far_ood"].append({"alpha": alpha,
                "reject_in": round(reject_rate(sets_in), 4),
                "reject_ood": round(reject_rate(sets_ood), 4),
                "msp_auroc": round(ood_auroc(s_in, s_ood), 4),
                "fpr@95tpr": round(fpr_at_tpr(s_in, s_ood), 4)})
        # accuracy on the held-out test (from alpha=0.1 fit's proba is fine; refit once for clarity)
        cc = ConformalClassifier(make(name), alpha=0.10, mondrian=True).fit(X[itr], y[itr])
        acc = float((cc.classes_[cc.predict_proba(X[ite]).argmax(1)] == y[ite]).mean())
        res["test_accuracy"] = round(acc, 4)
        for held, in_idx, out_idx in leave_one_population_out(y, pops):
            yin = y[in_idx]
            remap = {o: n for n, o in enumerate(sorted(set(yin)))}
            yin_r = np.array([remap[v] for v in yin])
            Xi = X[in_idx]
            xtr, xte, ya, _ = train_test_split(Xi, yin_r, test_size=0.3, stratify=yin_r, random_state=SEED)
            cc = ConformalClassifier(make(name), alpha=0.10, mondrian=True).fit(xtr, ya)
            res["lopo"].append({"held_out": held,
                "reject_known": round(reject_rate(cc.predict_set(xte)), 4),
                "reject_held_out": round(reject_rate(cc.predict_set(X[out_idx])), 4)})
        out["models"][name] = res
        c10 = [c for c in res["coverage_curve"] if c["alpha"] == 0.10][0]
        f10 = [f for f in res["far_ood"] if f["alpha"] == 0.10][0]
        gap = float(np.mean([l["reject_held_out"] - l["reject_known"] for l in res["lopo"]]))
        print(f"[{name}] acc={res['test_accuracy']} cov@0.1={c10['coverage']} set={c10['set_size']} "
              f"MSP_AUROC={f10['msp_auroc']} reject_ood@0.3="
              f"{[f['reject_ood'] for f in res['far_ood'] if f['alpha']==0.3][0]} "
              f"LOPO_gap={round(gap,4)}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/plan2_logreg.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/plan2_logreg.json", flush=True)


if __name__ == "__main__":
    main()
