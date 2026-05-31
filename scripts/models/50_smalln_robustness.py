"""(e) CREE — extreme small-n robustness (where a shrinkage embedding can win).

The random-effects embedding's empirical-Bayes shrinkage should degrade more
gracefully than a from-scratch linear model as labels shrink. Fixed 30% test;
subsample train to n in {50,100,150,200,full}; compare RandEffClassifier(emb)
vs LogReg(one-hot). 5 seeds. If the embedding's curve is flatter at small n,
that is a CREE capability the linear baseline lacks. GPU+CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from forensic_mh.fm.architectures import RandEffClassifier
from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import collect_genome_wide_strings, discover_chrom_vcfs, load_eas_labels

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
K = 8
SIZES = [50, 100, 150, 200, None]   # None = full train
SEEDS = list(range(5))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    Xstr = np.array(eas_rows, dtype=object)
    y, _ = load_eas_labels(PANEL, sids)
    classes = np.unique(y); y = np.searchsorted(classes, y)
    M = len(names)
    vocab = FMVocab(eas_rows, k=K)
    Xcode = vocab.encode(eas_rows)
    print(f"panel={M} EAS={len(sids)} device={DEV}", flush=True)

    res = {str(n if n else "full"): {"RandEff": [], "LogReg": []} for n in SIZES}
    sss = StratifiedShuffleSplit(n_splits=len(SEEDS), test_size=0.3, random_state=0)
    for seed, (tr_full, te) in enumerate(sss.split(Xstr, y)):
        rng = np.random.RandomState(seed)
        for n in SIZES:
            if n is None:
                tr = tr_full
            else:
                # stratified subsample of size ~n from tr_full
                per = max(1, n // len(np.unique(y)))
                tr = np.concatenate([rng.choice(tr_full[y[tr_full] == c], min(per, (y[tr_full] == c).sum()), replace=False)
                                     for c in np.unique(y)])
            key = str(n if n else "full")
            # RandEff (embedding)
            re = RandEffClassifier(k=K, d=32, epochs=200, seed=seed, device=DEV).fit(Xcode[tr], y[tr])
            res[key]["RandEff"].append(float((re.predict(Xcode[te]) == y[te]).mean()))
            # LogReg (one-hot)
            lr = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=2000)).fit(Xstr[tr], y[tr])
            res[key]["LogReg"].append(float((lr.predict(Xstr[te]) == y[te]).mean()))
        print(f"seed {seed} done", flush=True)

    out = {"sizes": [n if n else "full" for n in SIZES], "curves": {}}
    for n in SIZES:
        key = str(n if n else "full")
        out["curves"][key] = {m: {"mean": round(float(np.mean(res[key][m])), 4),
                                  "std": round(float(np.std(res[key][m])), 4)} for m in ("RandEff", "LogReg")}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/smalln.json").write_text(json.dumps(out, indent=2))
    print("\n=== accuracy vs train-n (RandEff emb vs LogReg) ===", flush=True)
    for n in SIZES:
        key = str(n if n else "full")
        r = out["curves"][key]
        print(f"  n={key:>4}: RandEff {r['RandEff']['mean']*100:4.1f}  LogReg {r['LogReg']['mean']*100:4.1f}  "
              f"(Δ {(r['RandEff']['mean']-r['LogReg']['mean'])*100:+.1f})", flush=True)
    print("saved results/baseline/smalln.json", flush=True)
    print("SMALLN_DONE", flush=True)


if __name__ == "__main__":
    main()
