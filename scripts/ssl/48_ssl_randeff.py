"""(c) SSL with a random-effects-REGULARIZED encoder (Paper 2 rescue), gnomAD, GPU.

§25 showed naive SSL (54%) does not beat linear. §27 showed the random-effects
Gaussian shrinkage makes embeddings work. Here we ADD that shrinkage to the SSL
transformer's per-(marker,code) embedding during masked + contrastive pretraining
on the clean gnomAD HGDP+1KG pool (4,091, hg38), then finetune EAS-5. Tests
whether a *regularized* embedding lets large-scale SSL finally help.
Compare: SSL-noRE 54.0 / supervised 55.4 / LogReg 78.0 (§25.1).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from forensic_mh.fm.dataset import MHMatrixDataset
from forensic_mh.fm.encoder import MHTransformer
from forensic_mh.fm.heads import AncestryHead
from forensic_mh.fm.objectives import masked_marker_loss, nt_xent
from forensic_mh.fm.vocab import FMVocab
from forensic_mh.pipelines.baseline import collect_genome_wide_strings, discover_chrom_vcfs
from forensic_mh.uq.openset import msp_score, ood_auroc

META = "data/hgdp/gnomad_meta_v1.tsv"
EAS = {"CHB", "CHS", "JPT", "KHV", "CDX"}
K, D, NL, NH = 8, 64, 2, 4
PRE_EP, FT_EP, LR, BATCH, CW, RE_W = 40, 40, 1e-3, 64, 0.5, 0.3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Proj(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.ReLU(), torch.nn.Linear(d, d // 2))

    def forward(self, x):
        return self.net(x)


def re_penalty(enc, log_sig, M):
    slots = enc.slots
    W = enc.value_emb.weight.view(M, slots, enc.d)
    var = (log_sig.exp() ** 2).view(M, 1, 1) + 1e-6
    return ((W ** 2) / (2 * var) + 0.5 * log_sig.view(M, 1, 1)).mean()


def pretrain(rows, vocab, M, epochs, seed=0):
    torch.manual_seed(seed)
    enc = MHTransformer(M, K, D, NL, NH).to(DEV)
    proj = Proj(D).to(DEV)
    log_sig = torch.nn.Parameter(torch.zeros(M, device=DEV))     # per-marker random-effect std
    dl = DataLoader(MHMatrixDataset(rows, vocab, seed=seed), batch_size=BATCH, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()) + [log_sig], lr=LR, weight_decay=1e-2)
    enc.train()
    for _ in range(epochs):
        for b in dl:
            ml = masked_marker_loss(enc.masked_logits(b["input"].to(DEV)), b["target"].to(DEV), b["mask_pos"].to(DEV))
            z1, z2 = proj(enc.embed(b["view1"].to(DEV))), proj(enc.embed(b["view2"].to(DEV)))
            loss = ml + CW * nt_xent(z1, z2) + RE_W * re_penalty(enc, log_sig, M)
            opt.zero_grad(); loss.backward(); opt.step()
    return {k: v.detach().clone() for k, v in enc.state_dict().items()}


def finetune_eval(state, ctr, ytr, cte, cood, M, nc, seed=0):
    torch.manual_seed(seed)
    enc = MHTransformer(M, K, D, NL, NH).to(DEV); enc.load_state_dict(state)
    head = AncestryHead(D, nc).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=LR, weight_decay=1e-2)
    X = torch.as_tensor(ctr, dtype=torch.long).to(DEV); yt = torch.as_tensor(ytr, dtype=torch.long).to(DEV)
    enc.train(); head.train()
    for _ in range(FT_EP):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(head(enc.embed(X[idx])), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); head.eval()
    with torch.no_grad():
        pte = F.softmax(head(enc.embed(torch.as_tensor(cte, dtype=torch.long).to(DEV))), 1).cpu().numpy()
        poo = F.softmax(head(enc.embed(torch.as_tensor(cood, dtype=torch.long).to(DEV))), 1).cpu().numpy()
    return pte, poo


def main() -> None:
    rmap, names = collect_genome_wide_strings(discover_chrom_vcfs("data/hgdp1kg", prefix="HGDP1KG_chr"), build="hg38")
    sids = sorted(rmap)
    rows = [[rmap[s].get(m, "N|N") for m in names] for s in sids]
    M = len(names)
    pop = {r["s"]: r["hgdp_tgp_meta.Population"] for r in csv.DictReader(open(META), delimiter="\t")}
    eas_i = [i for i, s in enumerate(sids) if pop.get(s) in EAS]
    ood_i = [i for i, s in enumerate(sids) if pop.get(s) and pop.get(s) not in EAS][:400]
    classes = sorted(EAS); cmap = {c: j for j, c in enumerate(classes)}
    y = np.array([cmap[pop[sids[i]]] for i in eas_i])
    print(f"gnomAD pool={len(sids)} markers={M} EAS-5={len(eas_i)} device={DEV}", flush=True)

    vocab = FMVocab(rows, k=K); codes = vocab.encode(rows)
    codes_eas, codes_ood = codes[eas_i], codes[ood_i]
    print("pretraining SSL + random-effects @all-4091...", flush=True)
    state = pretrain(rows, vocab, M, PRE_EP)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    accs, aurocs = [], []
    for fi, (tr, te) in enumerate(cv.split(codes_eas, y)):
        pte, poo = finetune_eval(state, codes_eas[tr], y[tr], codes_eas[te], codes_ood, M, len(classes), seed=fi)
        accs.append(float((pte.argmax(1) == y[te]).mean()))
        aurocs.append(ood_auroc(msp_score(pte), msp_score(poo)))
        print(f"fold {fi+1}: acc={accs[-1]:.3f} far-OOD AUROC={aurocs[-1]:.3f}", flush=True)

    out = {"model": "SSL + random-effects encoder @gnomAD-4091", "markers": M,
           "acc_mean": round(float(np.mean(accs)), 4), "acc_std": round(float(np.std(accs)), 4),
           "auroc_mean": round(float(np.mean(aurocs)), 4),
           "reference": {"SSL_noRE": 0.540, "supervised": 0.554, "LogReg": 0.780}}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/ssl_randeff.json").write_text(json.dumps(out, indent=2))
    print(f"\nSSL+RE@4091 acc={out['acc_mean']*100:.1f}±{out['acc_std']*100:.1f}  "
          f"far-OOD AUROC={out['auroc_mean']:.3f}  | ref SSL 54.0, LogReg 78.0", flush=True)
    print("saved results/baseline/ssl_randeff.json", flush=True)
    print("SSL_RANDEFF_DONE", flush=True)


if __name__ == "__main__":
    main()
