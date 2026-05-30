"""Native-categorical trees (RQ3) — was the tree loss an ENCODING or MODEL-CLASS issue?

Trees lost under ordinal encoding (§13/Appendix A) and reached only 57-60% with
one-hot. Here trees handle the diplotypes as NATIVE categoricals:
- HistGradientBoosting: capped FMVocab codes (k=200<=255) + categorical_features
  (subset splits, not ordinal-threshold).
- CatBoost: raw diplotype STRINGS as categorical features (ordered target stats).
If native handling still lags LogReg(one-hot) 79.6%, the tree loss is model-class,
not encoding. Same protocol: leakage-free 5-fold, acc + far-OOD MSP AUROC.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.openset import msp_score, ood_auroc

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    Xstr = np.array(eas_rows, dtype=object)
    y, _ = load_eas_labels(PANEL, sids)
    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    ood_rows = [[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)]
    Ostr = np.array(ood_rows, dtype=object)
    M = len(names)
    cat_idx = list(range(M))
    print(f"panel={M} EAS={len(sids)} OOD={Ostr.shape[0]}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    out = {"models": {}}
    res = {"HistGBDT (native-cat)": ([], []), "CatBoost (native-cat str)": ([], [])}
    for tr, te in cv.split(Xstr, y):
        tr_rows = [eas_rows[i] for i in tr]
        # HistGBDT — capped categorical codes
        vocab = FMVocab(tr_rows, k=200)
        Ctr, Cte, Cood = vocab.encode(tr_rows), vocab.encode([eas_rows[i] for i in te]), vocab.encode(ood_rows)
        hgb = HistGradientBoostingClassifier(categorical_features=cat_idx, max_iter=300,
                                             learning_rate=0.1, random_state=0)
        hgb.fit(Ctr, y[tr])
        res["HistGBDT (native-cat)"][0].append(float((hgb.predict(Cte) == y[te]).mean()))
        res["HistGBDT (native-cat)"][1].append(
            ood_auroc(msp_score(hgb.predict_proba(Cte)), msp_score(hgb.predict_proba(Cood))))
        # CatBoost — raw strings as categoricals
        cb = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.1,
                                loss_function="MultiClass", random_seed=0, verbose=False)
        cb.fit(Xstr[tr], y[tr], cat_features=cat_idx)
        res["CatBoost (native-cat str)"][0].append(float((cb.predict(Xstr[te]).ravel() == y[te]).mean()))
        res["CatBoost (native-cat str)"][1].append(
            ood_auroc(msp_score(cb.predict_proba(Xstr[te])), msp_score(cb.predict_proba(Ostr))))
        print("  fold done", flush=True)

    for name, (a, u) in res.items():
        out["models"][name] = {"acc_mean": round(float(np.mean(a)), 4),
                               "acc_std": round(float(np.std(a)), 4),
                               "auroc_mean": round(float(np.mean(u)), 4)}
        r = out["models"][name]
        print(f"{name:26s} acc={r['acc_mean']*100:5.1f}±{r['acc_std']*100:4.1f}  "
              f"far-OOD AUROC={r['auroc_mean']:.3f}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/native_cat_trees.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/native_cat_trees.json", flush=True)
    print("NATIVE_CAT_DONE", flush=True)


if __name__ == "__main__":
    main()
