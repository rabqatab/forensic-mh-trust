"""PCA-feature ablation (Chen et al. 2025 동기): raw ordinal 인코딩 vs PCA류(one-hot→SVD)
피처가 동아시아 5집단 분류 정확도를 올리는지 leakage-free로 비교.

- Arm A: per-marker ordinal 정수 인코딩 → XGBoost (현재 baseline 표현)
- Arm B: one-hot(diplotype) → StandardScaler → TruncatedSVD(K) → XGBoost/LogReg
  (SVD는 sparse one-hot용 PCA 등가물. 모든 변환은 sklearn Pipeline으로 fold 내부에서만 fit → 누수 없음)
동일 StratifiedKFold(5, seed=42)로 두 arm 비교.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from forensic_mh.pipelines.baseline import (
    _encode,
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"


def _xgb():
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         eval_metric="mlogloss", verbosity=0, random_state=42)


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    X_str = np.array([[rows[s].get(m, "N|N") for m in names] for s in sids], dtype=object)
    Xord, sids2 = _encode(rows, names)
    assert sids2 == sids
    y, pops = load_eas_labels(PANEL, sids)
    print(f"samples={len(sids)} markers={len(names)} pops={pops}", flush=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    out = {"n_markers": len(names), "n_samples": len(sids), "populations": pops, "results": {}}

    # Arm A: raw ordinal -> XGBoost
    a = cross_val_score(_xgb(), Xord, y, cv=cv, scoring="accuracy")
    out["results"]["raw_ordinal_xgb"] = {"mean": float(a.mean()), "std": float(a.std())}
    print(f"[A] raw ordinal + XGB: {a.mean()*100:.1f}% +/- {a.std()*100:.1f}", flush=True)

    # Arm B: one-hot -> SVD(K) -> {XGB, LogReg}
    for K in [10, 20, 50, 100]:
        pre = [OneHotEncoder(handle_unknown="ignore"),
               StandardScaler(with_mean=False),
               TruncatedSVD(n_components=K, random_state=42)]
        for clf_name, clf in [("xgb", _xgb()),
                              ("logreg", LogisticRegression(max_iter=2000, C=1.0))]:
            pipe = make_pipeline(*pre, clf)
            s = cross_val_score(pipe, X_str, y, cv=cv, scoring="accuracy")
            out["results"][f"pca{K}_{clf_name}"] = {"mean": float(s.mean()), "std": float(s.std())}
            print(f"[B] one-hot->SVD({K})->{clf_name}: {s.mean()*100:.1f}% +/- {s.std()*100:.1f}", flush=True)

    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/pca_ablation.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/pca_ablation.json", flush=True)


if __name__ == "__main__":
    main()
