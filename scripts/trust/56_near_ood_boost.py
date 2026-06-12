"""Push near-OOD further — can we beat single-CREE-variance 0.782, generalizing
to UNSEEN close populations?

Forensic reality: you can tune a near-OOD detector on some close populations, but
deploy it on one you've never seen (KOR). So every method is evaluated with a
held-out POPULATION split: the combiner sees fit-pops, is tested on disjoint
test-pops. 3 outer seeds (each reshuffles the 15 HGDP near-pops into fit/test).

Methods vs the single-VRE variance baseline:
  - var_ens     : 5-VRE variance ensemble (KL_W=2.0, the marginally-best setting)
  - maha / rmaha: embedding Mahalanobis and RELATIVE Mahalanobis (Ren 2021, near-OOD)
  - fusion_unsup: z-normalized average of {var_ens, rmaha, msp}
  - fusion_sup  : logistic combiner fit on (in-dist-cal vs fit-pops), tested on
                  (in-dist-test vs test-pops) — does a learned detector transfer
                  to populations it was never tuned on? GPU.
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import collect_genome_wide_strings, discover_chrom_vcfs
from forensic_mh.uq.openset import msp_score, ood_auroc

META = "data/hgdp/gnomad_meta_v1.tsv"
EAS5 = {"CHB", "CHS", "JPT", "KHV", "CDX"}
MATCHED = {"Han", "Japanese", "Dai"}
K, D, EPOCHS, LR, KL_W, NENS = 8, 32, 250, 1e-3, 2.0, 5
SEEDS = list(range(3))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class VRE(nn.Module):
    def __init__(self, M, k, d, nc, p=0.3):
        super().__init__()
        self.M, self.k, self.d = M, k, d
        self.mu = nn.Embedding(M * k, d); nn.init.normal_(self.mu.weight, std=0.05)
        self.logvar = nn.Embedding(M * k, d); nn.init.constant_(self.logvar.weight, -4.0)
        self.register_buffer("off", torch.arange(M) * k)
        self.log_sig = nn.Parameter(torch.zeros(M))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, nc)); self.drop = nn.Dropout(p)

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


def rel_maha(feat_tr, y_tr, nc):
    """Relative Mahalanobis (Ren 2021): min_c MD_c - MD_global -> near-OOD-targeted."""
    mus = np.stack([feat_tr[y_tr == c].mean(0) for c in range(nc)])
    P = EmpiricalCovariance().fit(np.concatenate([feat_tr[y_tr == c] - mus[c] for c in range(nc)])).precision_
    mu0 = feat_tr.mean(0); P0 = EmpiricalCovariance().fit(feat_tr - mu0).precision_
    def score(Fx):
        mdc = np.stack([((Fx - mus[c]) @ P * (Fx - mus[c])).sum(1) for c in range(nc)], 1).min(1)
        md0 = ((Fx - mu0) @ P0 * (Fx - mu0)).sum(1)
        return mdc - md0
    return score


def maha(feat_tr, y_tr, nc):
    mus = np.stack([feat_tr[y_tr == c].mean(0) for c in range(nc)])
    P = EmpiricalCovariance().fit(np.concatenate([feat_tr[y_tr == c] - mus[c] for c in range(nc)])).precision_
    return lambda Fx: np.stack([((Fx - mus[c]) @ P * (Fx - mus[c])).sum(1) for c in range(nc)], 1).min(1)


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

    a_idx = np.array([i for i, s in enumerate(sids) if tgp(s) and pop(s) in EAS5])
    ya = np.array([cmap[pop(sids[i])] for i in a_idx])
    near_by_pop = {}
    for i, s in enumerate(sids):
        if hgdp(s) and reg(s) == "EAS" and pop(s) not in MATCHED:
            near_by_pop.setdefault(pop(s), []).append(i)
    near_pops = sorted(near_by_pop)
    vocab = FMVocab(rows, k=K); codes = vocab.encode(rows)
    Xstr = np.array(rows, dtype=object)
    print(f"M={M} EAS5={len(a_idx)} near-pops={len(near_pops)} {near_pops} {DEV}", flush=True)

    METHODS = ["var_single", "var_ens", "maha", "rmaha", "msp", "fusion_unsup", "fusion_sup"]
    rec = {m: [] for m in METHODS}
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        # in-dist 3-way: fit / cal / test
        trv, te = train_test_split(np.arange(len(a_idx)), test_size=0.3, stratify=ya, random_state=seed)
        fit_l, cal_l = train_test_split(trv, test_size=0.3, stratify=ya[trv], random_state=seed)
        fitI, calI, teI = a_idx[fit_l], a_idx[cal_l], a_idx[te]; yfit = ya[fit_l]
        # near-pop split: fit-pops (tune combiner) vs test-pops (held-out eval)
        sh = rng.permutation(near_pops); fit_pops, test_pops = sh[:8], sh[8:]
        nfitI = np.array([i for p in fit_pops for i in near_by_pop[p]])
        ntestI = np.array([i for p in test_pops for i in near_by_pop[p]])

        # base models
        lr = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=3000)).fit(Xstr[fitI], yfit)
        ms = lambda idx: msp_score(lr.predict_proba(Xstr[idx]))
        vres = [fit_vre(codes[fitI], yfit, M, nc, seed * 10 + j) for j in range(NENS)]
        tc = lambda idx: torch.as_tensor(codes[idx], dtype=torch.long).to(DEV)
        vsingle = lambda idx: vres[0].var_score(tc(idx))
        vens = lambda idx: np.mean([v.var_score(tc(idx)) for v in vres], 0)
        emb = lambda idx: vres[0].embed(tc(idx))
        mh = maha(emb(fitI), yfit, nc); rmh = rel_maha(emb(fitI), yfit, nc)

        scorers = {"var_single": vsingle, "var_ens": vens, "msp": ms,
                   "maha": lambda idx: mh(emb(idx)), "rmaha": lambda idx: rmh(emb(idx))}
        S = {k: {"cal": f(calI), "te": f(teI), "nfit": f(nfitI), "ntest": f(ntestI)} for k, f in scorers.items()}

        for k in ["var_single", "var_ens", "maha", "rmaha", "msp"]:
            rec[k].append(ood_auroc(S[k]["te"], S[k]["ntest"]))   # held-out test-pops
        # unsupervised z-fusion of {var_ens, rmaha, msp} (normalize by cal stats)
        comp = ["var_ens", "rmaha", "msp"]
        def zfuse(split):
            return np.mean([(S[k][split] - S[k]["cal"].mean()) / (S[k]["cal"].std() + 1e-9) for k in comp], 0)
        rec["fusion_unsup"].append(ood_auroc(zfuse("te"), zfuse("ntest")))
        # supervised logistic combiner: fit on (cal in-dist=0, fit-pops=1), test on (te=0, test-pops=1)
        feats = lambda split: np.stack([S[k][split] for k in scorers], 1)
        Xtr = np.concatenate([feats("cal"), feats("nfit")]); ytr = np.r_[np.zeros(len(calI)), np.ones(len(nfitI))]
        comb = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
        p_te = comb.predict_proba(feats("te"))[:, 1]; p_nt = comb.predict_proba(feats("ntest"))[:, 1]
        rec["fusion_sup"].append(ood_auroc(p_te, p_nt))
        print(f"seed {seed}: test-pops={list(test_pops)} | var_single={rec['var_single'][-1]:.3f} "
              f"var_ens={rec['var_ens'][-1]:.3f} rmaha={rec['rmaha'][-1]:.3f} "
              f"fusion_unsup={rec['fusion_unsup'][-1]:.3f} fusion_sup={rec['fusion_sup'][-1]:.3f}", flush=True)

    agg = {m: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} for m, v in rec.items()}
    out = {"baseline_single_var": 0.782, "held_out_pop_eval": True, "n_ens": NENS, "kl_w": KL_W, "auroc": agg}
    Path("results/conformal").mkdir(parents=True, exist_ok=True)
    Path("results/conformal/near_ood_boost.json").write_text(json.dumps(out, indent=2))
    print("\n=== near-OOD AUROC on HELD-OUT test-populations (vs single-var 0.782 baseline) ===", flush=True)
    for m in METHODS:
        print(f"  {m:14s}: {agg[m]['mean']:.3f} ± {agg[m]['std']:.3f}", flush=True)
    print("saved results/conformal/near_ood_boost.json\nNEAR_BOOST_DONE", flush=True)


if __name__ == "__main__":
    main()
