"""Near-OOD: leave one EAS population out, measure reject rate on the unknown."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from forensic_mh.eval.lopo import leave_one_population_out
from forensic_mh.pipelines.baseline import (
    build_genome_wide_matrix,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.conformal_classifier import ConformalClassifier
from forensic_mh.uq.openset import reject_rate

EAS_DIR = "data/eas"
PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"


def main() -> None:
    vcfs = discover_chrom_vcfs(EAS_DIR)
    print(f"genome-wide EAS VCFs: {len(vcfs)} chroms {sorted(vcfs, key=int)}")
    X, sids, markers = build_genome_wide_matrix(vcfs)
    print(f"X: {X.shape[0]} samples × {X.shape[1]} markers")
    y, pops = load_eas_labels(PANEL, sids)
    base = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         eval_metric="mlogloss", verbosity=0, random_state=42)

    alpha = 0.10
    rows = []
    for held, in_idx, out_idx in leave_one_population_out(y, pops):
        y_in = y[in_idx]
        remap = {old: new for new, old in enumerate(sorted(set(y_in)))}
        y_in_r = np.array([remap[v] for v in y_in])
        X_in = X[in_idx]
        Xtr, Xte, ytr, yte = train_test_split(
            X_in, y_in_r, test_size=0.3, stratify=y_in_r, random_state=42)
        cc = ConformalClassifier(base, alpha=alpha, mondrian=True).fit(Xtr, ytr)
        rr_in = reject_rate(cc.predict_set(Xte))           # known pops — want low
        rr_held = reject_rate(cc.predict_set(X[out_idx]))  # unknown pop — want high
        rows.append({"held_out": held,
                     "reject_rate_known": round(rr_in, 4),
                     "reject_rate_held_out": round(rr_held, 4),
                     "gap": round(rr_held - rr_in, 4)})
        print(rows[-1])

    out = Path("results/conformal/lopo_nearood.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"alpha": alpha, "results": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
