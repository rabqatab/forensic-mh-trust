"""near-OOD recognition — the project's biggest open hurdle (KOR-proxy via HGDP).

The forensic risk: an unrepresented but genetically CLOSE population (e.g. Korean)
gets a confident in-panel label. We have no KOR data, but the gnomAD HGDP+1KG
callset gives a data-backed proxy: HGDP East-Asian-region populations that do NOT
match our 5-class panel (Yakut, Uygur, Mongola, Oroqen, Cambodian, Lahu, Yi, Miao,
She, Naxi, Tu, Tujia, Hezhen, Daur, Xibo) are genuine NEAR-OOD; HGDP non-EAS is
FAR-OOD; held-out 1KG-EAS-5 is in-dist. Same callset/markers → no build confound.

Three analyses off one shared train/split (5 seeds, LogReg + VRE on the same data):
  (1) near-OOD AUROC for every score: MSP, conformal set-size, CREE variance,
      logit-Mahalanobis, embedding-Mahalanobis, embedding-kNN — vs the easy far-OOD.
  (2) distance-based scores (Mahalanobis / kNN) vs the saturating MSP.
  (3) set-size soft-reject Pareto: reject if set-size >= tau -> (near reject, in-dist
      false-reject, far reject) trade-off. GPU (VRE).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import collect_genome_wide_strings, discover_chrom_vcfs
from forensic_mh.uq.conformal import build_prediction_sets, mondrian_quantiles
from forensic_mh.uq.openset import msp_score, ood_auroc

META = "data/hgdp/gnomad_meta_v1.tsv"
EAS5 = {"CHB", "CHS", "JPT", "KHV", "CDX"}
MATCHED = {"Han", "Japanese", "Dai"}      # HGDP-EAS that DO map to the panel -> in-dist
K, D, EPOCHS, LR, KL_W, ALPHA = 8, 32, 250, 1e-3, 0.5, 0.10
SEEDS = list(range(5))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class VRE(nn.Module):
    def __init__(self, M, k, d, nc, p=0.3):
        super().__init__()
        self.M, self.k, self.d = M, k, d
        self.mu = nn.Embedding(M * k, d); nn.init.normal_(self.mu.weight, std=0.05)
        self.logvar = nn.Embedding(M * k, d); nn.init.constant_(self.logvar.weight, -4.0)
        self.register_buffer("off", torch.arange(M) * k)
        self.log_sig = nn.Parameter(torch.zeros(M))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, nc))
        self.drop = nn.Dropout(p)

    def forward(self, x, sample=True):
        idx = x + self.off; mu, lv = self.mu(idx), self.logvar(idx)
        e = mu + torch.randn_like(mu) * (0.5 * lv).exp() if sample else mu
        return self.head(self.drop(e.mean(1)))

    def kl(self):
        mu = self.mu.weight.view(self.M, self.k, self.d); lv = self.logvar.weight.view(self.M, self.k, self.d)
        ps = (self.log_sig.exp() ** 2).view(self.M, 1, 1) + 1e-6
        return (0.5 * (lv.exp() / ps + mu ** 2 / ps - 1 - lv + ps.log())).mean()

    @torch.no_grad()
    def var_score(self, x):
        return self.logvar(x + self.off).exp().mean((1, 2)).cpu().numpy()

    @torch.no_grad()
    def embed(self, x):
        return self.mu(x + self.off).mean(1).cpu().numpy()


def fit_vre(X, y, M, nc, seed):
    torch.manual_seed(seed); m = VRE(M, K, D, nc).to(DEV)
    Xt = torch.as_tensor(X, dtype=torch.long).to(DEV); yt = torch.as_tensor(y).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        (F.cross_entropy(m(Xt), yt) + KL_W * m.kl()).backward(); opt.step()
    m.eval(); return m


def maha_scorer(feat_tr, y_tr, nc):
    """Class-conditional Mahalanobis: min over classes of (x-mu_c) Σ^-1 (x-mu_c)."""
    mus = np.stack([feat_tr[y_tr == c].mean(0) for c in range(nc)])
    cov = EmpiricalCovariance().fit(np.concatenate([feat_tr[y_tr == c] - mus[c] for c in range(nc)]))
    P = cov.precision_
    def score(F_):
        d = np.stack([((F_ - mus[c]) @ P * (F_ - mus[c])).sum(1) for c in range(nc)], 1)
        return d.min(1)
    return score


def knn_scorer(feat_tr, k=10):
    from sklearn.neighbors import NearestNeighbors
    nn_ = NearestNeighbors(n_neighbors=k).fit(feat_tr)
    return lambda F_: nn_.kneighbors(F_)[0][:, -1]   # distance to k-th neighbor


def auroc_block(score_in, score_near, score_far):
    return {"near": round(ood_auroc(score_in, score_near), 4),
            "far": round(ood_auroc(score_in, score_far), 4)}


def main() -> None:
    rmap, names = collect_genome_wide_strings(discover_chrom_vcfs("data/hgdp1kg", prefix="HGDP1KG_chr"), build="hg38")
    sids = sorted(rmap); rows = [[rmap[s].get(m, "N|N") for m in names] for s in sids]
    M = len(names)
    meta = {r["s"]: r for r in csv.DictReader(open(META), delimiter="\t")}
    tgp = lambda s: meta.get(s, {}).get("subsets.tgp") == "true"
    hgdp = lambda s: meta.get(s, {}).get("subsets.hgdp") == "true"
    pop = lambda s: meta.get(s, {}).get("hgdp_tgp_meta.Population", "")
    reg = lambda s: meta.get(s, {}).get("hgdp_tgp_meta.Genetic.region", "")

    classes = sorted(EAS5); cmap = {c: i for i, c in enumerate(classes)}; nc = len(classes)
    a_idx = [i for i, s in enumerate(sids) if tgp(s) and pop(s) in EAS5]
    ya = np.array([cmap[pop(sids[i])] for i in a_idx])
    near_idx = [i for i, s in enumerate(sids) if hgdp(s) and reg(s) == "EAS" and pop(s) not in MATCHED]   # KOR-proxy
    matched_idx = [i for i, s in enumerate(sids) if hgdp(s) and reg(s) in ("EAS",) and pop(s) in MATCHED]  # sanity in-dist
    far_idx = [i for i, s in enumerate(sids) if hgdp(s) and reg(s) not in ("EAS", "")]

    vocab = FMVocab(rows, k=K); codes = vocab.encode(rows)
    Xstr = np.array(rows, dtype=object)
    near_pops = sorted({pop(sids[i]) for i in near_idx})
    print(f"M={M} | train+test 1KG-EAS5={len(a_idx)} | NEAR-OOD(unmatched HGDP-EAS)={len(near_idx)} "
          f"{near_pops} | matched-EAS(sanity)={len(matched_idx)} | FAR-OOD={len(far_idx)} | {DEV}", flush=True)

    rec = {k: {"near": [], "far": []} for k in
           ["msp", "setsize", "cree_var", "maha_logit", "maha_emb", "knn_emb"]}
    pareto = {str(t): {"near": [], "far": [], "false_reject": []} for t in [2, 3, 4, 5]}

    idx_all = np.array(a_idx)
    for seed in SEEDS:
        # 3-way split (fit / cal / test) — conformal calibration must be disjoint from fit
        tr_l, te_l = train_test_split(np.arange(len(idx_all)), test_size=0.3, stratify=ya, random_state=seed)
        fit_l, cal_l = train_test_split(tr_l, test_size=0.3, stratify=ya[tr_l], random_state=seed)
        fitI, calI, teI = idx_all[fit_l], idx_all[cal_l], idx_all[te_l]
        yfit, ycal = ya[fit_l], ya[cal_l]

        # --- LogReg(one-hot): MSP, conformal set-size, logit-Mahalanobis ---
        lr = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=3000)).fit(Xstr[fitI], yfit)
        def proba(idx): return lr.predict_proba(Xstr[idx])
        p_te, p_near, p_far = proba(teI), proba(near_idx), proba(far_idx)
        # conformal (Mondrian) set sizes — quantiles from the disjoint cal split
        cal_scores = 1.0 - lr.predict_proba(Xstr[calI])[np.arange(len(ycal)), ycal]
        q = mondrian_quantiles(cal_scores, ycal, nc, ALPHA)
        ssize = lambda P: np.array([len(s) for s in build_prediction_sets(P, q)], dtype=float)
        s_te, s_near, s_far = ssize(p_te), ssize(p_near), ssize(p_far)
        # logit-space Mahalanobis (fit on the fit split)
        lg = lambda idx: lr.decision_function(Xstr[idx])
        mlog = maha_scorer(lg(fitI), yfit, nc)

        # --- VRE (CREE): variance, embedding Mahalanobis + kNN ---
        m = fit_vre(codes[fitI], yfit, M, nc, seed)
        tcode = lambda idx: torch.as_tensor(codes[idx], dtype=torch.long).to(DEV)
        v_te, v_near, v_far = (m.var_score(tcode(teI)), m.var_score(tcode(near_idx)), m.var_score(tcode(far_idx)))
        emb = lambda idx: m.embed(tcode(idx))
        memb = maha_scorer(emb(fitI), yfit, nc); kemb = knn_scorer(emb(fitI))

        rec["msp"]["near"].append(ood_auroc(msp_score(p_te), msp_score(p_near)))
        rec["msp"]["far"].append(ood_auroc(msp_score(p_te), msp_score(p_far)))
        rec["setsize"]["near"].append(ood_auroc(s_te, s_near)); rec["setsize"]["far"].append(ood_auroc(s_te, s_far))
        rec["cree_var"]["near"].append(ood_auroc(v_te, v_near)); rec["cree_var"]["far"].append(ood_auroc(v_te, v_far))
        rec["maha_logit"]["near"].append(ood_auroc(mlog(lg(teI)), mlog(lg(near_idx))))
        rec["maha_logit"]["far"].append(ood_auroc(mlog(lg(teI)), mlog(lg(far_idx))))
        rec["maha_emb"]["near"].append(ood_auroc(memb(emb(teI)), memb(emb(near_idx))))
        rec["maha_emb"]["far"].append(ood_auroc(memb(emb(teI)), memb(emb(far_idx))))
        rec["knn_emb"]["near"].append(ood_auroc(kemb(emb(teI)), kemb(emb(near_idx))))
        rec["knn_emb"]["far"].append(ood_auroc(kemb(emb(teI)), kemb(emb(far_idx))))

        # --- (3) set-size soft-reject Pareto ---
        for t in [2, 3, 4, 5]:
            pareto[str(t)]["false_reject"].append(float((s_te >= t).mean()))
            pareto[str(t)]["near"].append(float((s_near >= t).mean()))
            pareto[str(t)]["far"].append(float((s_far >= t).mean()))
        print(f"seed {seed}: MSP near={rec['msp']['near'][-1]:.3f} | var near={rec['cree_var']['near'][-1]:.3f} | "
              f"maha_emb near={rec['maha_emb']['near'][-1]:.3f} | knn near={rec['knn_emb']['near'][-1]:.3f}", flush=True)

    ms = lambda v: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
    out = {"markers": M, "n": {"train_test_1KG_EAS5": len(a_idx), "near_OOD": len(near_idx),
           "near_pops": near_pops, "matched_sanity": len(matched_idx), "far_OOD": len(far_idx)},
           "auroc": {k: {"near": ms(rec[k]["near"]), "far": ms(rec[k]["far"])} for k in rec},
           "setsize_pareto": {t: {kk: ms(v) for kk, v in d.items()} for t, d in pareto.items()}}
    Path("results/conformal").mkdir(parents=True, exist_ok=True)
    Path("results/conformal/near_ood.json").write_text(json.dumps(out, indent=2))

    print("\n=== near-OOD AUROC (in-dist vs unmatched HGDP-EAS) | far-OOD for gradient ===", flush=True)
    for k in rec:
        a = out["auroc"][k]
        print(f"  {k:12s}: near {a['near']['mean']:.3f}±{a['near']['std']:.3f}   far {a['far']['mean']:.3f}", flush=True)
    print("\n=== set-size soft-reject Pareto (reject if size>=tau) ===", flush=True)
    for t in [2, 3, 4, 5]:
        p = out["setsize_pareto"][str(t)]
        print(f"  tau={t}: near-reject {p['near']['mean']:.2f}  far-reject {p['far']['mean']:.2f}  "
              f"in-dist false-reject {p['false_reject']['mean']:.2f}", flush=True)
    print("saved results/conformal/near_ood.json\nNEAR_OOD_DONE", flush=True)


if __name__ == "__main__":
    main()
