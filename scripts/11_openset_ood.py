"""Far-OOD open-set eval: non-EAS superpopulations as unknown (genome-wide).

EAS (in-dist) and OOD (non-EAS) diplotypes are collected over the SAME set of
chromosomes (the intersection of what scripts/06 has extracted so far) so the
marker columns align, then encoded with one shared DiplotypeEncoder fit on EAS.
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from forensic_mh.data.encoding import DiplotypeEncoder
from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.conformal_classifier import ConformalClassifier
from forensic_mh.uq.openset import fpr_at_tpr, msp_score, ood_auroc, reject_rate

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
EAS_DIR = "data/eas"
OOD_DIR = "data/ood"


def _to_lists(rows, sids, names):
    return [[rows[s].get(m, "N|N") for m in names] for s in sids]


def main() -> None:
    eas_vcfs = discover_chrom_vcfs(EAS_DIR, prefix="EAS_chr")
    ood_vcfs = discover_chrom_vcfs(OOD_DIR, prefix="OOD_chr")
    common = sorted(set(eas_vcfs) & set(ood_vcfs), key=int)
    if not common:
        raise SystemExit("No chromosome has BOTH EAS and OOD subsets yet "
                         "(scripts/06 still running). Re-run when overlap exists.")
    print(f"genome-wide chroms with EAS+OOD: {common}")

    eas_rows, names = collect_genome_wide_strings({c: eas_vcfs[c] for c in common})
    ood_rows, _ = collect_genome_wide_strings({c: ood_vcfs[c] for c in common})
    eas_sids, ood_sids = sorted(eas_rows), sorted(ood_rows)

    enc = DiplotypeEncoder()
    X_eas = enc.fit_transform(_to_lists(eas_rows, eas_sids, names))
    X_ood = enc.transform(_to_lists(ood_rows, ood_sids, names))
    print(f"markers={len(names)}  EAS={len(eas_sids)}  OOD={len(ood_sids)}  "
          f"OOD unseen-diplotype fraction={enc.last_unseen_fraction:.3f}")

    y, pops = load_eas_labels(PANEL, eas_sids)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_eas, y, test_size=0.3, stratify=y, random_state=42)
    base = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         eval_metric="mlogloss", verbosity=0, random_state=42)

    report = {"chroms": common, "n_markers": len(names),
              "ood_unseen_fraction": round(enc.last_unseen_fraction, 4), "by_alpha": []}
    for alpha in [0.20, 0.10, 0.05]:
        cc = ConformalClassifier(base, alpha=alpha, mondrian=True).fit(X_tr, y_tr)
        sets_in = cc.predict_set(X_te)
        sets_ood = cc.predict_set(X_ood)
        s_in = msp_score(cc.predict_proba(X_te))
        s_ood = msp_score(cc.predict_proba(X_ood))
        report["by_alpha"].append({
            "alpha": alpha,
            "reject_rate_in_dist": round(reject_rate(sets_in), 4),
            "reject_rate_ood": round(reject_rate(sets_ood), 4),
            "msp_auroc": round(ood_auroc(s_in, s_ood), 4),
            "msp_fpr@95tpr": round(fpr_at_tpr(s_in, s_ood), 4),
        })
        print(report["by_alpha"][-1])

    out = Path("results/conformal/openset_ood.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
