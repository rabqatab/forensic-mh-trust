"""A — statistical rigor for the central claim (base model governs OSR).

Repeat coverage / set-size / far-OOD (AUROC, FPR@95, empty-set reject) across
N_SEEDS train/test splits for LogReg(one-hot), RandomForest, XGBoost → mean ± std
(error bars on the OSR-base-model gap). OOD set fixed; alpha=0.10.
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
from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.conformal import empirical_coverage, mean_set_size
from forensic_mh.uq.conformal_classifier import ConformalClassifier
from forensic_mh.uq.openset import fpr_at_tpr, msp_score, ood_auroc, reject_rate

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
ALPHA = 0.10
SEEDS = list(range(10))


def make(name):
    if name == "XGBoost":
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             eval_metric="mlogloss", verbosity=0, random_state=0)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=400, random_state=0)
    return make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=2000))


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    Xstr = np.array(eas_rows, dtype=object)
    enc = DiplotypeEncoder(); Xord = enc.fit_transform(eas_rows)
    y, pops = load_eas_labels(PANEL, sids)
    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    ood_rows = [[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)]
    Ostr = np.array(ood_rows, dtype=object); Oord = enc.transform(ood_rows)
    print(f"panel={len(names)} EAS={len(sids)} OOD={len(ood_rows)} seeds={len(SEEDS)}", flush=True)

    out = {"alpha": ALPHA, "n_seeds": len(SEEDS), "models": {}}
    for name in ["LogReg(one-hot)", "RandomForest", "XGBoost"]:
        rep_str = name.startswith("LogReg")
        X = Xstr if rep_str else Xord
        Xood = Ostr if rep_str else Oord
        m = {k: [] for k in ["coverage", "set_size", "msp_auroc", "fpr95", "reject_ood"]}
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
            cc = ConformalClassifier(make(name), alpha=ALPHA, mondrian=True).fit(Xtr, ytr)
            sets_in = cc.predict_set(Xte); sets_ood = cc.predict_set(Xood)
            s_in, s_ood = msp_score(cc.predict_proba(Xte)), msp_score(cc.predict_proba(Xood))
            m["coverage"].append(empirical_coverage(sets_in, yte))
            m["set_size"].append(mean_set_size(sets_in))
            m["msp_auroc"].append(ood_auroc(s_in, s_ood))
            m["fpr95"].append(fpr_at_tpr(s_in, s_ood))
            m["reject_ood"].append(reject_rate(sets_ood))
        out["models"][name] = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
                               for k, v in m.items()}
        r = out["models"][name]
        print(f"[{name}] AUROC={r['msp_auroc']['mean']}±{r['msp_auroc']['std']} "
              f"cov={r['coverage']['mean']}±{r['coverage']['std']} "
              f"set={r['set_size']['mean']} FPR95={r['fpr95']['mean']} "
              f"reject_ood={r['reject_ood']['mean']}±{r['reject_ood']['std']}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/trust_rigor.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/trust_rigor.json", flush=True)


if __name__ == "__main__":
    main()
