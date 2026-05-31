"""(f) CREE cross-cohort transfer — does intrinsic variance open-set transfer? GPU.

CREE's selling point (docs/07) is a TRANSFERABLE representation whose posterior
variance is an intrinsic open-set score. (d) showed variance >> MSP for open-set
WITHIN one cohort (1KG). Here we test the harder, honest claim: train the
variational random-effects embedding on cohort A (1000 Genomes EAS-5), then
ZERO-SHOT transfer to cohort B (HGDP — never seen, different platform/sampling,
18 novel East-Asian populations incl. Yakut/Uygur/Mongola). Does the variance
score, learned on A, still rank HGDP-EAS (in-dist) below HGDP-nonEAS (OOD)
across the cohort boundary? Compare variance-AUROC vs MSP-AUROC, cross vs within.

Both cohorts share the SAME hg38 harmonized marker panel (gnomAD HGDP+1KG),
so transfer is not confounded by build/marker mismatch (RQ7 §22). 5 seeds.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import collect_genome_wide_strings, discover_chrom_vcfs
from forensic_mh.uq.openset import msp_score, ood_auroc

META = "data/hgdp/gnomad_meta_v1.tsv"
EAS5 = {"CHB", "CHS", "JPT", "KHV", "CDX"}
K, D, EPOCHS, LR, KL_W = 8, 32, 250, 1e-3, 0.5
SEEDS = list(range(5))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class VRE(nn.Module):
    """Variational random-effects embedding (same as scripts/trust/49)."""

    def __init__(self, M, k, d, n_classes, p=0.3):
        super().__init__()
        self.M, self.k, self.d = M, k, d
        self.mu = nn.Embedding(M * k, d); nn.init.normal_(self.mu.weight, std=0.05)
        self.logvar = nn.Embedding(M * k, d); nn.init.constant_(self.logvar.weight, -4.0)
        self.register_buffer("off", torch.arange(M) * k)
        self.log_sig = nn.Parameter(torch.zeros(M))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))
        self.drop = nn.Dropout(p)

    def forward(self, x, sample=True):
        idx = x + self.off
        mu, lv = self.mu(idx), self.logvar(idx)
        e = mu + torch.randn_like(mu) * (0.5 * lv).exp() if sample else mu
        return self.head(self.drop(e.mean(1)))

    def kl(self):
        mu = self.mu.weight.view(self.M, self.k, self.d)
        lv = self.logvar.weight.view(self.M, self.k, self.d)
        ps = (self.log_sig.exp() ** 2).view(self.M, 1, 1) + 1e-6
        return (0.5 * (lv.exp() / ps + mu ** 2 / ps - 1 - lv + ps.log())).mean()

    @torch.no_grad()
    def var_score(self, x):                                  # per-sample posterior variance (higher = OOD)
        return self.logvar(x + self.off).exp().mean((1, 2)).cpu().numpy()


def fit(X, y, M, nc, seed):
    torch.manual_seed(seed)
    m = VRE(M, K, D, nc).to(DEV)
    Xt = torch.as_tensor(X, dtype=torch.long).to(DEV); yt = torch.as_tensor(y).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        loss = F.cross_entropy(m(Xt), yt) + KL_W * m.kl()
        loss.backward(); opt.step()
    m.eval()
    return m


def softmax_np(m, codes):
    with torch.no_grad():
        return F.softmax(m(torch.as_tensor(codes, dtype=torch.long).to(DEV), sample=False), 1).cpu().numpy()


def vscore(m, codes):
    return m.var_score(torch.as_tensor(codes, dtype=torch.long).to(DEV))


def main() -> None:
    rmap, names = collect_genome_wide_strings(discover_chrom_vcfs("data/hgdp1kg", prefix="HGDP1KG_chr"), build="hg38")
    sids = sorted(rmap)
    rows = [[rmap[s].get(m, "N|N") for m in names] for s in sids]
    M = len(names)
    meta = {r["s"]: r for r in csv.DictReader(open(META), delimiter="\t")}
    tgp = {s for s in sids if meta.get(s, {}).get("subsets.tgp") == "true"}
    hgdp = {s for s in sids if meta.get(s, {}).get("subsets.hgdp") == "true"}
    pop = lambda s: meta.get(s, {}).get("hgdp_tgp_meta.Population", "")
    reg = lambda s: meta.get(s, {}).get("hgdp_tgp_meta.Genetic.region", "")

    # training cohort A: 1000 Genomes EAS-5
    classes = sorted(EAS5); cmap = {c: j for j, c in enumerate(classes)}
    a_idx = [i for i, s in enumerate(sids) if s in tgp and pop(s) in EAS5]
    ya = np.array([cmap[pop(sids[i])] for i in a_idx])
    # within-cohort OOD: 1000 Genomes non-EAS (cap 400, balanced-ish)
    a_ood_idx = [i for i, s in enumerate(sids) if s in tgp and pop(s) not in EAS5][:400]
    # transfer cohort B: HGDP, never seen
    b_in_idx = [i for i, s in enumerate(sids) if s in hgdp and reg(s) == "EAS"]      # 240, 18 novel pops
    b_ood_idx = [i for i, s in enumerate(sids) if s in hgdp and reg(s) not in ("EAS", "")]  # 708 non-EAS

    vocab = FMVocab(rows, k=K); codes = vocab.encode(rows)
    Xa, Xa_ood = codes[a_idx], codes[a_ood_idx]
    Xb_in, Xb_ood = codes[b_in_idx], codes[b_ood_idx]
    print(f"panel={M} | train 1KG-EAS5={len(a_idx)} 1KG-OOD={len(a_ood_idx)} | "
          f"transfer HGDP-EAS={len(b_in_idx)} HGDP-OOD={len(b_ood_idx)} | device={DEV}", flush=True)

    rec = {"acc": [], "within_var": [], "within_msp": [], "cross_var": [], "cross_msp": []}
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.3, stratify=ya, random_state=seed)
        m = fit(Xtr, ytr, M, len(classes), seed)
        pte = softmax_np(m, Xte)
        rec["acc"].append(float((pte.argmax(1) == yte).mean()))
        # within-cohort (1KG test vs 1KG non-EAS)
        rec["within_var"].append(ood_auroc(vscore(m, Xte), vscore(m, Xa_ood)))
        rec["within_msp"].append(ood_auroc(msp_score(pte), msp_score(softmax_np(m, Xa_ood))))
        # cross-cohort zero-shot (HGDP-EAS vs HGDP-nonEAS) — never seen cohort
        rec["cross_var"].append(ood_auroc(vscore(m, Xb_in), vscore(m, Xb_ood)))
        rec["cross_msp"].append(ood_auroc(msp_score(softmax_np(m, Xb_in)), msp_score(softmax_np(m, Xb_ood))))
        print(f"seed {seed}: acc={rec['acc'][-1]:.3f} | within var={rec['within_var'][-1]:.3f} "
              f"msp={rec['within_msp'][-1]:.3f} | CROSS var={rec['cross_var'][-1]:.3f} "
              f"msp={rec['cross_msp'][-1]:.3f}", flush=True)

    agg = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)} for k, v in rec.items()}
    out = {"model": "CREE cross-cohort transfer (1KG->HGDP, variational random-effects)",
           "markers": M, "n": {"train_1KG_EAS5": len(a_idx), "HGDP_EAS_in": len(b_in_idx),
                               "HGDP_OOD": len(b_ood_idx)}, **agg}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/cree_transfer.json").write_text(json.dumps(out, indent=2))
    print(f"\nCREE transfer | within-cohort: var={agg['within_var']['mean']:.3f} vs msp={agg['within_msp']['mean']:.3f}"
          f"  || CROSS-cohort (1KG->HGDP): var={agg['cross_var']['mean']:.3f} vs msp={agg['cross_msp']['mean']:.3f}",
          flush=True)
    print("saved results/baseline/cree_transfer.json", flush=True)
    print("CREE_TRANSFER_DONE", flush=True)


if __name__ == "__main__":
    main()
