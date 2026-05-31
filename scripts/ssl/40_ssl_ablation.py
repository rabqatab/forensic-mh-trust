"""SSL data-scale ablation (RQ-FM / Paper 2) — does MORE pretraining data help?

The raw SSL@2504 (44.6%) vs FM@504 (54.6%) comparison is confounded (different
markers, contrastive added, protocol). This isolates the data-scale effect:
SAME markers (the all1kg pool), SAME finetune, SAME contrastive — vary ONLY the
pretraining pool: {none (supervised) / 504 EAS-only / 2504 all-1KG}.
5-fold finetune+eval on labeled EAS; far-OOD AUROC on non-EAS. GPU (sparkq).
"""
from __future__ import annotations

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

PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
EAS = {"CHB", "CHS", "JPT", "KHV", "CDX"}
K, D, NL, NH = 8, 64, 2, 4
PRE_EP, FT_EP, LR, BATCH, CW = 40, 40, 1e-3, 64, 0.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Proj(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.ReLU(), torch.nn.Linear(d, d // 2))

    def forward(self, x):
        return self.net(x)


def read_panel():
    pop, sup = {}, {}
    for ln in Path(PANEL).read_text().splitlines()[1:]:
        f = ln.split()
        if len(f) >= 3:
            pop[f[0]], sup[f[0]] = f[1], f[2]
    return pop, sup


def pretrain(rows, vocab, M, epochs, seed=0):
    torch.manual_seed(seed)
    enc = MHTransformer(M, K, D, NL, NH).to(DEV)
    proj = Proj(D).to(DEV)
    dl = DataLoader(MHMatrixDataset(rows, vocab, seed=seed), batch_size=BATCH, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=LR, weight_decay=1e-2)
    enc.train()
    for _ in range(epochs):
        for b in dl:
            ml = masked_marker_loss(enc.masked_logits(b["input"].to(DEV)), b["target"].to(DEV), b["mask_pos"].to(DEV))
            z1, z2 = proj(enc.embed(b["view1"].to(DEV))), proj(enc.embed(b["view2"].to(DEV)))
            loss = ml + CW * nt_xent(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
    return {k: v.detach().clone() for k, v in enc.state_dict().items()}


def finetune_eval(state, codes_tr, ytr, codes_te, codes_ood, M, nc, seed=0):
    torch.manual_seed(seed)
    enc = MHTransformer(M, K, D, NL, NH).to(DEV)
    if state is not None:
        enc.load_state_dict(state)
    head = AncestryHead(D, nc).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=LR, weight_decay=1e-2)
    Xtr = torch.as_tensor(codes_tr, dtype=torch.long).to(DEV)
    yt = torch.as_tensor(ytr, dtype=torch.long).to(DEV)
    enc.train(); head.train()
    for _ in range(FT_EP):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(head(enc.embed(Xtr[idx])), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); head.eval()
    with torch.no_grad():
        pte = F.softmax(head(enc.embed(torch.as_tensor(codes_te, dtype=torch.long).to(DEV))), 1).cpu().numpy()
        poo = F.softmax(head(enc.embed(torch.as_tensor(codes_ood, dtype=torch.long).to(DEV))), 1).cpu().numpy()
    return pte, poo


def main() -> None:
    rmap, names = collect_genome_wide_strings(discover_chrom_vcfs("data/all1kg", prefix="ALL_chr"))
    sids = sorted(rmap)
    rows = [[rmap[s].get(m, "N|N") for m in names] for s in sids]
    M = len(names)
    pop, sup = read_panel()
    eas_i = [i for i, s in enumerate(sids) if sup.get(s) == "EAS"]
    ood_i = [i for i, s in enumerate(sids) if sup.get(s) not in ("EAS", None)][:400]
    classes = sorted(EAS); cmap = {c: j for j, c in enumerate(classes)}
    y_eas = np.array([cmap[pop[sids[i]]] for i in eas_i])
    print(f"pool={len(sids)} markers={M} EAS={len(eas_i)} OOD={len(ood_i)} device={DEV}", flush=True)

    vocab = FMVocab(rows, k=K)
    codes_all = vocab.encode(rows)
    codes_eas, codes_ood = codes_all[eas_i], codes_all[ood_i]
    eas_rows = [rows[i] for i in eas_i]

    # pretrain states: none / 504-EAS-only / 2504-all (same markers/vocab)
    print("pretraining 504-EAS-only...", flush=True)
    state_504 = pretrain(eas_rows, vocab, M, PRE_EP)
    print("pretraining 2504-all...", flush=True)
    state_2504 = pretrain(rows, vocab, M, PRE_EP)
    conditions = {"supervised (no pretrain)": None, "SSL @504 EAS": state_504, "SSL @2504 all": state_2504}

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    out = {"pool_n": len(sids), "markers": M, "conditions": {}}
    for name, state in conditions.items():
        accs, aurocs = [], []
        for fi, (tr, te) in enumerate(cv.split(codes_eas, y_eas)):
            pte, poo = finetune_eval(state, codes_eas[tr], y_eas[tr], codes_eas[te], codes_ood, M, len(classes), seed=fi)
            accs.append(float((pte.argmax(1) == y_eas[te]).mean()))
            aurocs.append(ood_auroc(msp_score(pte), msp_score(poo)))
        out["conditions"][name] = {"acc_mean": round(float(np.mean(accs)), 4), "acc_std": round(float(np.std(accs)), 4),
                                   "auroc_mean": round(float(np.mean(aurocs)), 4)}
        r = out["conditions"][name]
        print(f"{name:26s} acc={r['acc_mean']*100:5.1f}±{r['acc_std']*100:4.1f}  far-OOD AUROC={r['auroc_mean']:.3f}", flush=True)

    out["reference"] = {"LogReg_onehot_full3042": 0.796}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/ssl_ablation.json").write_text(json.dumps(out, indent=2))
    print("saved results/baseline/ssl_ablation.json", flush=True)
    print("SSL_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
