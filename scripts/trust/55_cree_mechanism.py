"""CREE deep-dive — WHY does posterior variance become a near-OOD signal?

Falsifiable claim: the random-effects KL shrinkage is what turns variance into an
open-set score. KL pulls each N(mu,var) toward a per-marker prior N(0, sig_m^2);
codes WITH training data have their variance driven DOWN by the CE loss, while
codes the model never saw are shaped ONLY by KL and stay at the (higher) prior
variance -> a "familiarity gradient". Prediction: with KL_W=0 the unseen codes
keep their low init variance and near-OOD AUROC collapses toward chance.

Three probes off the same near-OOD setup as script 54 (1KG-EAS-5 train, unmatched
HGDP-EAS = near-OOD, non-EAS = far-OOD):
  (A) KL ablation: KL_W in {0, 0.1, 0.5, 2.0} -> accuracy + variance AUROC (near/far).
  (B) per-(marker,code) logvar vs training frequency (Spearman) -> is variance
      learned familiarity?
  (C) OTHER-slot ablation: recompute the score excluding OTHER-slot positions ->
      does near-OOD survive without the trivial rare/OTHER-counting? GPU. 3 seeds.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import collect_genome_wide_strings, discover_chrom_vcfs
from forensic_mh.uq.openset import ood_auroc

META = "data/hgdp/gnomad_meta_v1.tsv"
EAS5 = {"CHB", "CHS", "JPT", "KHV", "CDX"}
MATCHED = {"Han", "Japanese", "Dai"}
K, D, EPOCHS, LR = 8, 32, 250, 1e-3
KL_GRID = [0.0, 0.1, 0.5, 2.0]
SEEDS = list(range(3))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OTHER = K - 1


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
    def per_pos_var(self, x):                      # (n, M) per-position posterior variance
        return self.logvar(x + self.off).exp().mean(2).cpu().numpy()


def fit_vre(X, y, M, nc, seed, kl_w):
    torch.manual_seed(seed); m = VRE(M, K, D, nc).to(DEV)
    Xt = torch.as_tensor(X, dtype=torch.long).to(DEV); yt = torch.as_tensor(y).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        (F.cross_entropy(m(Xt), yt) + kl_w * m.kl()).backward(); opt.step()
    m.eval(); return m


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
    near_idx = [i for i, s in enumerate(sids) if hgdp(s) and reg(s) == "EAS" and pop(s) not in MATCHED]
    far_idx = [i for i, s in enumerate(sids) if hgdp(s) and reg(s) not in ("EAS", "")]
    vocab = FMVocab(rows, k=K); codes = vocab.encode(rows)
    print(f"M={M} EAS5={len(a_idx)} near={len(near_idx)} far={len(far_idx)} {DEV}", flush=True)

    def score(m, idx, exclude_other=False):
        x = torch.as_tensor(codes[idx], dtype=torch.long).to(DEV)
        pv = m.per_pos_var(x)                       # (n, M)
        if exclude_other:
            mask = (codes[idx] != OTHER)
            return (pv * mask).sum(1) / np.maximum(mask.sum(1), 1)
        return pv.mean(1)

    # (A) KL ablation
    ablation = {str(w): {"acc": [], "near": [], "far": [], "near_noOther": []} for w in KL_GRID}
    # (B/C) mechanism on KL_W=0.5
    spear, logvar_seen, logvar_unseen = [], [], []
    for seed in SEEDS:
        tr, te = train_test_split(np.arange(len(a_idx)), test_size=0.3, stratify=ya, random_state=seed)
        fitI, teI, yfit = a_idx[tr], a_idx[te], ya[tr]
        # training frequency per (marker,code) slot
        freq = np.zeros(M * K)
        flat = (codes[fitI] + (np.arange(M) * K)).ravel()
        np.add.at(freq, flat, 1)
        for w in KL_GRID:
            m = fit_vre(codes[fitI], yfit, M, nc, seed, w)
            with torch.no_grad():
                acc = float((F.softmax(m(torch.as_tensor(codes[teI], dtype=torch.long).to(DEV), sample=False), 1)
                             .argmax(1).cpu().numpy() == ya[te]).mean())
            s_te, s_near, s_far = score(m, teI), score(m, near_idx), score(m, far_idx)
            ablation[str(w)]["acc"].append(acc)
            ablation[str(w)]["near"].append(ood_auroc(s_te, s_near))
            ablation[str(w)]["far"].append(ood_auroc(s_te, s_far))
            ablation[str(w)]["near_noOther"].append(
                ood_auroc(score(m, teI, True), score(m, near_idx, True)))
            if w == 0.5:
                lv = m.logvar.weight.mean(1).detach().cpu().numpy()   # (M*K,) mean logvar per slot
                seen = freq > 0
                spear.append(spearmanr(freq[seen], lv[seen]).statistic)
                logvar_seen.append(float(lv[seen].mean())); logvar_unseen.append(float(lv[~seen].mean()))
        print(f"seed {seed}: KL0 near={ablation['0.0']['near'][-1]:.3f} | "
              f"KL0.5 near={ablation['0.5']['near'][-1]:.3f} | spearman(freq,logvar)={spear[-1]:.3f}", flush=True)

    ms = lambda v: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
    out = {"kl_ablation": {w: {k: ms(v) for k, v in d.items()} for w, d in ablation.items()},
           "mechanism": {"spearman_freq_vs_logvar": ms(spear),
                         "mean_logvar_seen_codes": ms(logvar_seen),
                         "mean_logvar_unseen_codes": ms(logvar_unseen)}}
    Path("results/conformal").mkdir(parents=True, exist_ok=True)
    Path("results/conformal/cree_mechanism.json").write_text(json.dumps(out, indent=2))

    print("\n=== (A) KL shrinkage ablation: variance near-OOD AUROC ===", flush=True)
    print(f"{'KL_W':>6} {'acc':>6} {'near':>14} {'far':>8} {'near(no-OTHER)':>16}", flush=True)
    for w in KL_GRID:
        a = out["kl_ablation"][str(w)]
        print(f"{w:>6} {a['acc']['mean']*100:>5.1f}% {a['near']['mean']:>8.3f}±{a['near']['std']:.3f} "
              f"{a['far']['mean']:>8.3f} {a['near_noOther']['mean']:>14.3f}", flush=True)
    me = out["mechanism"]
    print(f"\n=== (B/C) mechanism @KL_W=0.5 ===", flush=True)
    print(f"  Spearman(train-freq, logvar) = {me['spearman_freq_vs_logvar']['mean']:.3f} "
          f"(음수 = 자주 본 코드일수록 분산 낮음 = 학습된 친숙도)", flush=True)
    print(f"  mean logvar: seen codes {me['mean_logvar_seen_codes']['mean']:.3f} "
          f"vs unseen codes {me['mean_logvar_unseen_codes']['mean']:.3f}", flush=True)
    print("saved results/conformal/cree_mechanism.json\nCREE_MECH_DONE", flush=True)


if __name__ == "__main__":
    main()
