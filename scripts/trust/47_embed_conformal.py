"""(b) Embedding-DL + conformal trust layer — RandEffEmb wrapped (RQ1 extension).

The model-agnostic ConformalClassifier wraps the random-effects embedding (73%,
§27): does the embedding-DL get valid coverage + open-set rejection like the
linear base? 5-seed train/test splits, alpha=0.10, far-OOD = non-EAS. Reports
coverage / set size / far-OOD AUROC / empty-set reject, vs LogReg(one-hot) (§20).
GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from forensic_mh.fm.architectures import RandEffClassifier
from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import collect_genome_wide_strings, discover_chrom_vcfs, load_eas_labels
from forensic_mh.uq.conformal import empirical_coverage, mean_set_size
from forensic_mh.uq.conformal_classifier import ConformalClassifier
from forensic_mh.uq.openset import fpr_at_tpr, msp_score, ood_auroc, reject_rate

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
ALPHA, SEEDS, K = 0.10, list(range(5)), 8
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    y, _ = load_eas_labels(PANEL, sids)
    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    ood_rows = [[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)]
    M = len(names)
    print(f"panel={M} EAS={len(sids)} OOD={len(ood_rows)} device={DEV}", flush=True)

    vocab = FMVocab(eas_rows, k=K)
    X = vocab.encode(eas_rows); Xood = vocab.encode(ood_rows); y = np.asarray(y)
    rec = {k: [] for k in ["coverage", "set_size", "msp_auroc", "fpr95", "reject_ood"]}
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        cc = ConformalClassifier(RandEffClassifier(k=K, seed=seed, device=DEV), alpha=ALPHA, mondrian=True).fit(Xtr, ytr)
        sets_in, sets_ood = cc.predict_set(Xte), cc.predict_set(Xood)
        s_in, s_ood = msp_score(cc.predict_proba(Xte)), msp_score(cc.predict_proba(Xood))
        yte_idx = np.searchsorted(cc.classes_, yte)
        rec["coverage"].append(empirical_coverage(sets_in, yte_idx))
        rec["set_size"].append(mean_set_size(sets_in))
        rec["msp_auroc"].append(ood_auroc(s_in, s_ood))
        rec["fpr95"].append(fpr_at_tpr(s_in, s_ood))
        rec["reject_ood"].append(reject_rate(sets_ood))
        print(f"seed {seed}: cov={rec['coverage'][-1]:.3f} set={rec['set_size'][-1]:.2f} "
              f"AUROC={rec['msp_auroc'][-1]:.3f} reject={rec['reject_ood'][-1]:.3f}", flush=True)

    out = {"model": "RandEffEmb + Mondrian conformal", "alpha": ALPHA, "n_seeds": len(SEEDS),
           **{k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} for k, v in rec.items()},
           "ref_LogReg_§20": {"auroc": 0.840, "coverage": 0.916, "set": 1.72}}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/embed_conformal.json").write_text(json.dumps(out, indent=2))
    print(f"\nRandEffEmb+conformal: cov={out['coverage']['mean']:.3f} set={out['set_size']['mean']:.2f} "
          f"far-OOD AUROC={out['msp_auroc']['mean']:.3f} reject_ood={out['reject_ood']['mean']:.3f}", flush=True)
    print("saved results/baseline/embed_conformal.json", flush=True)
    print("EMBED_CONFORMAL_DONE", flush=True)


if __name__ == "__main__":
    main()
