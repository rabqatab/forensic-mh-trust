"""RQ5 confirmation — marker-level RFE (recursive feature elimination).

§23.1 used one-shot model-based ranking (rank once, take top-N). RFE is the
stronger wrapper: refit and re-rank after each elimination, so it can recover
when removing a redundant marker changes the others' importance. We eliminate
at the MARKER level (drop whole markers, not individual one-hot columns):
repeatedly fit one-hot LogReg on the surviving markers, aggregate per-marker
coef energy, drop down to the next target size, refit, evaluate. Leakage-free
(elimination runs inside each CV fold's train). Confirms the minimum panel
against an independent, stronger selector.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from forensic_mh.pipelines.baseline import (
    collect_genome_wide_strings,
    discover_chrom_vcfs,
    load_eas_labels,
)

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
SEED = 42
# descending elimination schedule (refit+eval at each level); includes the
# §23.1 comparison sizes 1000/500/300/200/100/50/25
SCHEDULE = [2000, 1000, 500, 300, 200, 100, 50, 25]
REPORT = [25, 50, 100, 200, 300, 500, 1000]


def coef_rank(Xstr_tr, y_tr, keep):
    """Per-marker coef energy on the surviving markers; returns local order (best first)."""
    oh = OneHotEncoder(handle_unknown="ignore").fit(Xstr_tr[:, keep])
    X = oh.transform(Xstr_tr[:, keep])
    clf = LogisticRegression(max_iter=2000).fit(X, y_tr)
    fm = np.repeat(np.arange(len(keep)), [len(c) for c in oh.categories_])
    imp = np.zeros(len(keep))
    np.add.at(imp, fm, (clf.coef_ ** 2).sum(0))
    return np.argsort(imp)[::-1]


def eval_panel(Xstr_tr, y_tr, Xstr_te, y_te, cols):
    pipe = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                         LogisticRegression(max_iter=2000))
    pipe.fit(Xstr_tr[:, cols], y_tr)
    return float((pipe.predict(Xstr_te[:, cols]) == y_te).mean())


def main() -> None:
    rows, names = collect_genome_wide_strings(discover_chrom_vcfs("data/eas"))
    sids = sorted(rows)
    Xstr = np.array([[rows[s].get(m, "N|N") for m in names] for s in sids], dtype=object)
    y, _ = load_eas_labels(PANEL, sids)
    M = len(names)
    print(f"panel={M} EAS={len(sids)} schedule={SCHEDULE}", flush=True)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rfe_acc = {n: [] for n in SCHEDULE}
    for fi, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y)):
        keep = np.arange(M)                       # recursive elimination on train only
        for target in SCHEDULE:
            order = coef_rank(Xstr[tr], y[tr], keep)
            keep = keep[order[:target]]           # drop weakest down to target, refit next loop
            acc = eval_panel(Xstr[tr], y[tr], Xstr[te], y[te], keep)
            rfe_acc[target].append(acc)
            print(f"fold {fi+1} N={target}: acc={acc:.3f}", flush=True)

    out = {"panel_total": M, "schedule": SCHEDULE,
           "rfe": {str(n): {"mean": round(float(np.mean(rfe_acc[n])), 4),
                            "std": round(float(np.std(rfe_acc[n])), 4)} for n in SCHEDULE}}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/min_panel_rfe.json").write_text(json.dumps(out, indent=2))
    print("\n=== RFE accuracy vs panel size (vs §23.1 one-shot coef_l2 / MI) ===", flush=True)
    oneshot = {25: 52.2, 50: 54.6, 100: 61.1, 200: 63.9, 300: 67.5, 500: 70.0, 1000: 76.8}
    mi = {25: 32.9, 50: 39.1, 100: 49.4, 200: 54.8, 300: 59.9, 500: 60.3, 1000: 68.0}
    print(f'{"N":>5} | {"RFE":>12} | {"one-shot":>9} | {"MI":>6}', flush=True)
    for n in REPORT:
        r = out["rfe"].get(str(n))
        rs = f'{r["mean"]*100:.1f}±{r["std"]*100:.1f}' if r else "—"
        print(f'{n:>5} | {rs:>12} | {oneshot.get(n,"—"):>9} | {mi.get(n,"—"):>6}', flush=True)
    print("saved results/baseline/min_panel_rfe.json", flush=True)
    print("MIN_PANEL_RFE_DONE", flush=True)


if __name__ == "__main__":
    main()
