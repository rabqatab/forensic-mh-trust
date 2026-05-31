"""Day 1-2 산출: chr22-only leakage-free baseline.

Plan 1의 첫 가시적 결과. chr22만 사용하므로 정확도는 chance(~20%) 위·90% 아래의
중간 값이 정상 — 핵심은 leakage-free pipeline이 끝까지 작동하는 것.

Full chromosome panel은 Plan 2 후반에서.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from forensic_mh.eval.nested_cv import leakage_free_cv_score
from forensic_mh.pipelines.baseline import build_diplotype_matrix, load_eas_labels


def main() -> None:
    vcf = "data/eas/EAS_chr22.vcf.gz"
    panel = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"

    print("[1/3] Building diplotype matrix for chr22 (hg19)...")
    X, sids, marker_names = build_diplotype_matrix(vcf, "chr22", build="hg19")
    print(f"  X shape: {X.shape}  ({len(sids)} samples × {len(marker_names)} markers)")

    print("[2/3] Loading EAS labels...")
    y, pops = load_eas_labels(panel, sids)
    counts = {p: int(np.sum(y == i)) for i, p in enumerate(pops)}
    print(f"  labels: {counts}")

    print("[3/3] Nested CV (leakage-free) at multiple panel sizes...")
    n_markers = len(marker_names)
    candidates = sorted({n for n in [5, 10, 20, 30, n_markers] if n <= n_markers})
    acc_results = {}
    for n in candidates:
        scores = leakage_free_cv_score(
            X, y, n_top_features=n, n_splits=5, random_state=42,
        )
        mean, std = float(np.mean(scores)), float(np.std(scores))
        acc_results[n] = {"mean": mean, "std": std, "scores": list(map(float, scores))}
        print(f"  MH {n:3d}개: 정확도 {mean:.3f} +/- {std:.3f}")

    out = Path("results/baseline")
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "chromosome": "chr22",
        "build": "hg19",
        "n_samples": int(X.shape[0]),
        "n_markers_total": int(X.shape[1]),
        "populations": pops,
        "label_counts": counts,
        "panel_size_accuracy": acc_results,
    }
    out_path = out / "chr22_baseline.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
