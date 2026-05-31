"""TabPFN on the top-200 panel (RQ3) — does small-data tabular SOTA beat linear?

TabPFN (Hollmann et al. 2023) is the prior-fitted-transformer SOTA for SMALL
tabular classification — n=504 is exactly its regime. We use the top-200
model-based panel (selected leakage-free inside each fold, as in §23).

Inference runs on the Prior Labs CLOUD via `tabpfn-client` (no local weights/GPU).
The cloud client exposes no categorical-feature flag, so the 200 markers are fed
as capped ordinal codes (numerical) — a mild ordinal handicap, documented.
Fair comparison: TabPFN@200 vs LogReg(one-hot)@200 (§23.1 = 63.9%); LogReg@full
= 79.6% (TabPFN cannot use all 3,042 due to its feature limit).

REPRODUCIBILITY: cloud inference — the server model can change over time. We
record the tabpfn-client version + model identifiers in the output JSON.
Auth via env var TABPFN_TOKEN (never hard-coded / committed).
"""
from __future__ import annotations

import json
import os
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from tabpfn_client import TabPFNClassifier, set_access_token

from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)
from forensic_mh.uq.openset import msp_score, ood_auroc

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
N_PANEL = 200
VK = 64                        # capped categorical cardinality
N_EST = 8


def rank_coef(Xstr_tr, y_tr, M):
    oh = OneHotEncoder(handle_unknown="ignore").fit(Xstr_tr)
    X = oh.transform(Xstr_tr)
    fm = np.repeat(np.arange(M), [len(c) for c in oh.categories_])
    clf = LogisticRegression(max_iter=2000).fit(X, y_tr)
    imp = np.zeros(M)
    np.add.at(imp, fm, (clf.coef_ ** 2).sum(0))
    return np.argsort(imp)[::-1]


def _load_token() -> str:
    """TABPFN_TOKEN from env, else from the gitignored .env (no key on the CLI)."""
    tok = os.environ.get("TABPFN_TOKEN")
    if not tok and Path(".env").exists():
        for ln in Path(".env").read_text().splitlines():
            if ln.strip().startswith("TABPFN_TOKEN="):
                tok = ln.split("=", 1)[1].strip().strip('"').strip("'")
    if not tok:
        raise SystemExit("TABPFN_TOKEN not set (env or .env)")
    return tok


def main() -> None:
    set_access_token(_load_token())
    cli_ver = pkg_version("tabpfn-client")
    print(f"tabpfn-client {cli_ver} (cloud inference)", flush=True)

    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    eas_rows = [[rows[s].get(m, "N|N") for m in names] for s in sids]
    Xstr = np.array(eas_rows, dtype=object)
    y, _ = load_eas_labels(PANEL, sids)
    orows, _ = collect_genome_wide_strings(discover_chrom_vcfs("data/ood", prefix="OOD_chr"))
    ood_rows = [[orows[s].get(m, "N|N") for m in names] for s in sorted(orows)]
    M = len(names)
    print(f"panel={M} EAS={len(sids)} OOD={len(ood_rows)} N_PANEL={N_PANEL}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    accs, aurocs, model_ids = [], [], set()
    for fi, (tr, te) in enumerate(cv.split(Xstr, y)):
        cols = rank_coef(Xstr[tr], y[tr], M)[:N_PANEL]
        tr_rows = [[eas_rows[i][c] for c in cols] for i in tr]
        te_rows = [[eas_rows[i][c] for c in cols] for i in te]
        oo_rows = [[r[c] for c in cols] for r in ood_rows]
        vocab = FMVocab(tr_rows, k=VK)
        Xtr = vocab.encode(tr_rows).astype(float)
        Xte = vocab.encode(te_rows).astype(float)
        Xoo = vocab.encode(oo_rows).astype(float)
        clf = TabPFNClassifier(n_estimators=N_EST, ignore_pretraining_limits=True, random_state=0)
        clf.fit(Xtr, y[tr])
        model_ids.add(str(getattr(clf, "model_path", "default")))
        pte, poo = clf.predict_proba(Xte), clf.predict_proba(Xoo)
        accs.append(float((clf.classes_[pte.argmax(1)] == y[te]).mean()))
        aurocs.append(ood_auroc(msp_score(pte), msp_score(poo)))
        print(f"fold {fi+1}: acc={accs[-1]:.3f} far-OOD AUROC={aurocs[-1]:.3f}", flush=True)

    out = {"model": "TabPFN (cloud)", "tabpfn_client_version": cli_ver,
           "server_model": sorted(model_ids), "n_estimators": N_EST,
           "encoding": "capped ordinal codes (k=%d), numerical to TabPFN" % VK,
           "panel": M, "n_panel": N_PANEL,
           "acc_mean": round(float(np.mean(accs)), 4), "acc_std": round(float(np.std(accs)), 4),
           "auroc_mean": round(float(np.mean(aurocs)), 4), "auroc_std": round(float(np.std(aurocs)), 4),
           "reference": {"LogReg_onehot@200": 0.639, "LogReg_onehot@full": 0.796}}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/tabpfn_panel.json").write_text(json.dumps(out, indent=2))
    print(f"\nTabPFN@{N_PANEL} acc={out['acc_mean']*100:.1f}±{out['acc_std']*100:.1f}  "
          f"far-OOD AUROC={out['auroc_mean']:.3f}  | ref LogReg@200 63.9, @full 79.6", flush=True)
    print(f"client={cli_ver} server_model={out['server_model']}", flush=True)
    print("saved results/baseline/tabpfn_panel.json", flush=True)
    print("TABPFN_PANEL_DONE", flush=True)


if __name__ == "__main__":
    main()
