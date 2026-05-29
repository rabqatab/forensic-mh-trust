"""Coverage vs set-size trade-off curve for Mondrian conformal on EAS MH."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from forensic_mh.pipelines.baseline import build_diplotype_matrix, load_eas_labels
from forensic_mh.uq.conformal import (
    class_conditional_coverage,
    empirical_coverage,
    mean_set_size,
)
from forensic_mh.uq.conformal_classifier import ConformalClassifier

VCF = "data/eas/EAS_chr22.vcf.gz"
PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"


def main() -> None:
    X, sids, markers = build_diplotype_matrix(VCF, "chr22")
    y, pops = load_eas_labels(PANEL, sids)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

    base = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         eval_metric="mlogloss", verbosity=0, random_state=42)
    rows = []
    for alpha in [0.30, 0.20, 0.10, 0.05]:
        cc = ConformalClassifier(base, alpha=alpha, mondrian=True).fit(X_tr, y_tr)
        sets = cc.predict_set(X_te)
        cc_cov = class_conditional_coverage(sets, y_te, len(pops))
        rows.append({
            "alpha": alpha,
            "target_coverage": round(1 - alpha, 3),
            "marginal_coverage": round(empirical_coverage(sets, y_te), 4),
            "mean_set_size": round(mean_set_size(sets), 4),
            "per_class_coverage": {pops[k]: round(cc_cov[k], 4) for k in range(len(pops))},
        })
        print(f"alpha={alpha}: cov={rows[-1]['marginal_coverage']} "
              f"size={rows[-1]['mean_set_size']}")

    out = Path("results/conformal/coverage_curve.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_markers": len(markers), "populations": pops,
                               "curve": rows}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
