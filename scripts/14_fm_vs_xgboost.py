"""Plan 3b-core: SSL FM vs XGBoost on an identical high-Ae EAS panel.

Compares 5-pop accuracy, conformal coverage/set-size, and far-OOD MSP AUROC,
using the SAME FMVocab-encoded marker matrix for both models (isolates the
model difference). Single stratified 70/30 split; one ConformalClassifier fit
per model serves accuracy + coverage + OSR.
"""
from __future__ import annotations

import json
from pathlib import Path

import microhapdb
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from forensic_mh.fm.sklearn_api import ForensicFMClassifier
from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.conformal import empirical_coverage, mean_set_size
from forensic_mh.uq.conformal_classifier import ConformalClassifier
from forensic_mh.uq.openset import msp_score, ood_auroc, reject_rate

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
K_VOCAB = 16
N_PANEL = 256
ALPHA = 0.10
SEED = 42


def top_ae_markers(n: int) -> set[str]:
    m = microhapdb.markers
    auto = m[m["Chrom"].astype(str).str.replace("chr", "").isin([str(i) for i in range(1, 23)])]
    auto = auto[auto["Positions37"].notna() & auto["Ae"].notna()]
    return set(auto.sort_values("Ae", ascending=False)["Name"].head(n))


def main() -> None:
    panel = top_ae_markers(N_PANEL)
    eas_rows, eas_names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    names = [m for m in eas_names if m in panel]
    eas_sids = sorted(eas_rows)
    eas_X = [[eas_rows[s].get(m, "N|N") for m in names] for s in eas_sids]

    ood_rows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    ood_sids = sorted(ood_rows)
    ood_X = [[ood_rows[s].get(m, "N|N") for m in names] for s in ood_sids]
    print(f"panel markers used={len(names)} EAS={len(eas_sids)} OOD={len(ood_sids)}", flush=True)

    enc = FMVocab(eas_X, k=K_VOCAB)
    X = enc.encode(eas_X)
    Xood = enc.encode(ood_X)
    y, pops = load_eas_labels(PANEL, eas_sids)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)

    def evaluate(base, label):
        cc = ConformalClassifier(base, alpha=ALPHA, mondrian=True).fit(Xtr, ytr)
        proba = cc.predict_proba(Xte)
        acc = float((cc.classes_[proba.argmax(1)] == yte).mean())
        sets = cc.predict_set(Xte)
        s_in, s_ood = msp_score(proba), msp_score(cc.predict_proba(Xood))
        r = {"accuracy": round(acc, 4),
             "coverage": round(empirical_coverage(sets, yte), 4),
             "set_size": round(mean_set_size(sets), 4),
             "msp_auroc": round(ood_auroc(s_in, s_ood), 4),
             "ood_reject": round(reject_rate(cc.predict_set(Xood)), 4)}
        print(f"[{label}] {r}", flush=True)
        return r

    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                        eval_metric="mlogloss", verbosity=0, random_state=SEED)
    fm = ForensicFMClassifier(k=K_VOCAB, d_model=64, n_layers=2, n_heads=2,
                              pretrain_epochs=20, finetune_epochs=40, seed=SEED, device="cpu")

    out = {"panel_markers": len(names), "n_eas": len(eas_sids), "n_ood": len(ood_sids),
           "alpha": ALPHA, "test_frac": 0.30,
           "xgboost": evaluate(xgb, "XGBoost"), "fm": evaluate(fm, "FM")}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/fm_vs_xgboost.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/fm_vs_xgboost.json", flush=True)


if __name__ == "__main__":
    main()
