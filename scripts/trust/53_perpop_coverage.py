"""Per-population (Mondrian) conformal coverage for the RECOMMENDED base (LogReg).

Forensic reliability must hold PER population, not just on average. Marginal
coverage can mask subgroup under-coverage. We report class-conditional coverage
for the recommended LogReg(one-hot) base with Mondrian split-conformal, across
10 seeds (per-class test sets are small — esp. KHV — so a single split is noisy).
Honest output: where the per-population guarantee binds and where it does not.
CPU. results/conformal/perpop_coverage_logreg.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from forensic_mh.pipelines.baseline import collect_genome_wide_strings, discover_chrom_vcfs, load_eas_labels
from forensic_mh.uq.conformal import class_conditional_coverage, empirical_coverage, mean_set_size
from forensic_mh.uq.conformal_classifier import ConformalClassifier

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
ALPHAS = [0.20, 0.10, 0.05]
SEEDS = list(range(10))


def make():
    return make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=3000))


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    Xstr = np.array(eas_rows, dtype=object)
    yi, pops = load_eas_labels(PANEL, sids)   # yi: int labels 0..4; pops: ordered names [CDX..KHV]
    nC = len(pops)
    print(f"panel={len(names)} EAS={len(sids)} classes={pops}", flush=True)
    counts = {pops[k]: int((yi == k).sum()) for k in range(nC)}
    print(f"per-class N (total): {counts}", flush=True)

    agg = {a: {"marginal": [], "set": [], "per_class": {p: [] for p in pops}} for a in ALPHAS}
    for seed in SEEDS:
        itr, ite = train_test_split(np.arange(len(sids)), test_size=0.3, stratify=yi, random_state=seed)
        for a in ALPHAS:
            cc = ConformalClassifier(make(), alpha=a, mondrian=True).fit(Xstr[itr], yi[itr])
            sets = cc.predict_set(Xstr[ite])
            agg[a]["marginal"].append(empirical_coverage(sets, yi[ite]))
            agg[a]["set"].append(mean_set_size(sets))
            ccov = class_conditional_coverage(sets, yi[ite], nC)
            for k in range(nC):
                if not np.isnan(ccov[k]):
                    agg[a]["per_class"][pops[k]].append(ccov[k])
        print(f"seed {seed} done", flush=True)

    def ms(v):
        return {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} if v else None

    out = {"base": "LogReg(one-hot) + Mondrian split-conformal", "panel": len(names),
           "seeds": len(SEEDS), "per_class_N_total": counts, "by_alpha": {}}
    for a in ALPHAS:
        out["by_alpha"][str(a)] = {
            "target_coverage": round(1 - a, 3),
            "marginal_coverage": ms(agg[a]["marginal"]),
            "mean_set_size": ms(agg[a]["set"]),
            "per_class_coverage": {p: ms(agg[a]["per_class"][p]) for p in pops},
        }
    Path("results/conformal").mkdir(parents=True, exist_ok=True)
    Path("results/conformal/perpop_coverage_logreg.json").write_text(json.dumps(out, indent=2))

    print("\n=== LogReg per-population coverage (10-seed mean) ===", flush=True)
    for a in ALPHAS:
        r = out["by_alpha"][str(a)]
        pc = "  ".join(f"{p} {r['per_class_coverage'][p]['mean']:.2f}" for p in pops)
        print(f"alpha={a} (target {1-a:.2f}): marginal {r['marginal_coverage']['mean']:.3f} "
              f"set {r['mean_set_size']['mean']:.2f} | {pc}", flush=True)
    print("saved results/conformal/perpop_coverage_logreg.json", flush=True)
    print("PERPOP_DONE", flush=True)


if __name__ == "__main__":
    main()
