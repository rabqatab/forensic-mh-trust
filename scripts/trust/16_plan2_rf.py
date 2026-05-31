"""Plan 2 metrics re-run with RandomForest base vs XGBoost (side-by-side).

Same genome-wide panel, same DiplotypeEncoder encoding, same splits — only the
base_estimator changes. Reports conformal coverage curve, far-OOD (MSP AUROC +
empty-set reject), and LOPO near-OOD for both models.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from forensic_mh.data.encoding import DiplotypeEncoder
from forensic_mh.eval.lopo import leave_one_population_out
from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.conformal import (
    class_conditional_coverage,
    empirical_coverage,
    mean_set_size,
)
from forensic_mh.uq.conformal_classifier import ConformalClassifier
from forensic_mh.uq.openset import fpr_at_tpr, msp_score, ood_auroc, reject_rate

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
SEED = 42


def make(name):
    if name == "XGBoost":
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             eval_metric="mlogloss", verbosity=0, random_state=SEED)
    return RandomForestClassifier(n_estimators=400, random_state=SEED)


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    enc = DiplotypeEncoder()
    X = enc.fit_transform(eas_rows)
    y, pops = load_eas_labels(PANEL, sids)

    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    osids = sorted(orows)
    Xood = enc.transform([[orows[s].get(m, "N|N") for m in names] for s in osids])
    print(f"panel={len(names)} EAS={len(sids)} OOD={len(osids)}", flush=True)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    out = {"panel": len(names), "models": {}}

    for name in ["XGBoost", "RandomForest"]:
        res = {"coverage_curve": [], "far_ood": [], "lopo": []}
        # coverage + far-OOD across alpha (one fit per alpha)
        for alpha in [0.30, 0.20, 0.10, 0.05]:
            cc = ConformalClassifier(make(name), alpha=alpha, mondrian=True).fit(Xtr, ytr)
            sets_in = cc.predict_set(Xte)
            sets_ood = cc.predict_set(Xood)
            s_in, s_ood = msp_score(cc.predict_proba(Xte)), msp_score(cc.predict_proba(Xood))
            res["coverage_curve"].append({
                "alpha": alpha,
                "coverage": round(empirical_coverage(sets_in, yte), 4),
                "set_size": round(mean_set_size(sets_in), 4)})
            res["far_ood"].append({
                "alpha": alpha,
                "reject_in": round(reject_rate(sets_in), 4),
                "reject_ood": round(reject_rate(sets_ood), 4),
                "msp_auroc": round(ood_auroc(s_in, s_ood), 4),
                "fpr@95tpr": round(fpr_at_tpr(s_in, s_ood), 4)})
        # LOPO near-OOD (alpha=0.10)
        for held, in_idx, out_idx in leave_one_population_out(y, pops):
            yin = y[in_idx]
            remap = {o: n for n, o in enumerate(sorted(set(yin)))}
            yin_r = np.array([remap[v] for v in yin])
            Xi = X[in_idx]
            xtr, xte, ya, _ = train_test_split(Xi, yin_r, test_size=0.3, stratify=yin_r, random_state=SEED)
            cc = ConformalClassifier(make(name), alpha=0.10, mondrian=True).fit(xtr, ya)
            rr_known = reject_rate(cc.predict_set(xte))
            rr_held = reject_rate(cc.predict_set(X[out_idx]))
            res["lopo"].append({"held_out": held, "reject_known": round(rr_known, 4),
                                "reject_held_out": round(rr_held, 4), "gap": round(rr_held - rr_known, 4)})
        out["models"][name] = res
        cov10 = [c for c in res["coverage_curve"] if c["alpha"] == 0.10][0]
        auroc10 = [f for f in res["far_ood"] if f["alpha"] == 0.10][0]
        print(f"[{name}] cov@0.1={cov10['coverage']} set={cov10['set_size']} "
              f"MSP_AUROC={auroc10['msp_auroc']} reject_ood@0.05="
              f"{[f['reject_ood'] for f in res['far_ood'] if f['alpha']==0.05][0]} "
              f"LOPO_gap_mean={round(float(np.mean([l['gap'] for l in res['lopo']])),4)}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/plan2_rf_vs_xgb.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/plan2_rf_vs_xgb.json", flush=True)


if __name__ == "__main__":
    main()
