"""Verify the one-hot accuracy result with leakage-free 5-fold CV.

Thread-1 single-split showed LogReg/MLP on one-hot MH ~73% — far above the
~57% 'ceiling' (which was tree-models-on-ordinal). Confirm with proper CV
(OneHotEncoder fit inside each fold via Pipeline). Also isolates the
StandardScaler handicap that depressed LogReg in the earlier model-zoo.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
SEED = 42


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    Xstr = np.array([[rows[s].get(m, "N|N") for m in names] for s in sids], dtype=object)
    y, pops = load_eas_labels(PANEL, sids)
    print(f"panel={len(names)} EAS={len(sids)}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pipes = {
        "LogReg_onehot_noscaler": make_pipeline(
            OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=3000)),
        "LogReg_onehot_scaler": make_pipeline(
            OneHotEncoder(handle_unknown="ignore"), StandardScaler(with_mean=False),
            LogisticRegression(max_iter=3000)),
    }
    out = {"panel": len(names), "n": len(sids), "results": {}}
    for name, pipe in pipes.items():
        s = cross_val_score(pipe, Xstr, y, cv=cv, scoring="accuracy")
        out["results"][name] = {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4),
                                "folds": [round(float(x), 4) for x in s]}
        print(f"[{name}] {out['results'][name]['mean']*100:.1f}% +/- {out['results'][name]['std']*100:.1f}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/onehot_cv.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/onehot_cv.json", flush=True)


if __name__ == "__main__":
    main()
