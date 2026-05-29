"""Far-OOD open-set eval: non-EAS superpopulations as unknown."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from forensic_mh.data.encoding import DiplotypeEncoder
from forensic_mh.data.markers import filter_by_chromosome, load_mh_markers, parse_positions
from forensic_mh.data.vcf_io import extract_diplotypes_for_locus
from forensic_mh.pipelines.baseline import load_eas_labels
from forensic_mh.uq.conformal_classifier import ConformalClassifier
from forensic_mh.uq.openset import fpr_at_tpr, msp_score, ood_auroc, reject_rate

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
VCF_ALL = "data/1000g/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
VCF_EAS = "data/eas/EAS_chr22.vcf.gz"
BUILD = "hg19"


def _rows(vcf, sample_ids, markers):
    """Return (rows, kept_sample_ids, marker_names): 'h0|h1' per marker."""
    per_marker = {}
    for _, mh in markers.iterrows():
        pos = parse_positions(mh, build=BUILD)
        if not pos:
            continue
        per_marker[mh["Name"]] = extract_diplotypes_for_locus(vcf, "22", pos, sample_ids)
    names = list(per_marker.keys())
    rows, kept = [], []
    for s in sample_ids:
        if all(s in per_marker[m] for m in names):
            rows.append([f"{per_marker[m][s][0]}|{per_marker[m][s][1]}" for m in names])
            kept.append(s)
    return rows, kept, names


def main() -> None:
    panel = pd.read_csv(PANEL, sep="\t")
    eas_ids = panel[panel.super_pop == "EAS"]["sample"].tolist()
    ood_ids = panel[panel.super_pop != "EAS"]["sample"].sample(
        n=300, random_state=42).tolist()

    markers = filter_by_chromosome(load_mh_markers(), "chr22")
    eas_rows, eas_kept, names = _rows(VCF_EAS, eas_ids, markers)
    ood_rows, _, _ = _rows(VCF_ALL, ood_ids, markers)

    enc = DiplotypeEncoder()
    X_eas = enc.fit_transform(eas_rows)
    y, pops = load_eas_labels(PANEL, eas_kept)
    X_ood = enc.transform(ood_rows)
    print(f"OOD unseen-diplotype fraction: {enc.last_unseen_fraction:.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_eas, y, test_size=0.3, stratify=y, random_state=42)
    base = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         eval_metric="mlogloss", verbosity=0, random_state=42)

    report = {"populations": pops, "n_markers": len(names),
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
