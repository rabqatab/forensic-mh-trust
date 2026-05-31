"""SSL foundation-model retraining on the EXPANDED pool (RQ-FM / Paper 2).

The §24 result showed SSL gives a small lift even at n=504 (supervised 51.0 ->
SSL+ft 54.6). The bottleneck was data scale. Here we pretrain the marker-
transformer (masked-marker + NT-Xent contrastive with ADO augmentation) on the
FULL 2,504 1000G pool (label-free), then finetune+evaluate the ancestry head on
the labeled EAS-5 via 5-fold CV. Far-OOD AUROC uses non-EAS 1000G samples.
Compares against the n=504-pretrain FM (54.6%) and LogReg(one-hot) (79.6%).

GPU job — submit via sparkq.
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
POOL_DIR, POOL_PREFIX = "data/all1kg", "ALL_chr"
EAS = {"CHB", "CHS", "JPT", "KHV", "CDX"}
K, D_MODEL, N_LAYERS, N_HEADS = 8, 64, 2, 4
PRETRAIN_EPOCHS, FINETUNE_EPOCHS = 40, 40
LR, BATCH, CONTRA_W = 1e-3, 64, 0.5
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Proj(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.ReLU(),
                                       torch.nn.Linear(d, d // 2))

    def forward(self, x):
        return self.net(x)


def read_panel():
    pop, sup = {}, {}
    for ln in Path(PANEL).read_text().splitlines()[1:]:
        f = ln.split()
        if len(f) >= 3:
            pop[f[0]], sup[f[0]] = f[1], f[2]
    return pop, sup


def pretrain(encoder, proj, rows, vocab, epochs, seed=0):
    torch.manual_seed(seed)
    ds = MHMatrixDataset(rows, vocab, seed=seed)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(proj.parameters()),
                            lr=LR, weight_decay=1e-2)
    encoder.train(); proj.train()
    for ep in range(epochs):
        tot = 0.0
        for b in dl:
            ml = masked_marker_loss(encoder.masked_logits(b["input"].to(DEV)),
                                    b["target"].to(DEV), b["mask_pos"].to(DEV))
            z1 = proj(encoder.embed(b["view1"].to(DEV)))
            z2 = proj(encoder.embed(b["view2"].to(DEV)))
            loss = ml + CONTRA_W * nt_xent(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  pretrain ep{ep}: loss={tot/len(dl):.3f}", flush=True)


def finetune_eval(state, codes_tr, ytr, codes_te, codes_ood, M, n_classes, seed=0):
    torch.manual_seed(seed)
    enc = MHTransformer(M, K, D_MODEL, N_LAYERS, N_HEADS).to(DEV)
    enc.load_state_dict(state)
    head = AncestryHead(D_MODEL, n_classes).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=LR, weight_decay=1e-2)
    Xtr = torch.as_tensor(codes_tr, dtype=torch.long).to(DEV)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long).to(DEV)
    enc.train(); head.train()
    for _ in range(FINETUNE_EPOCHS):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            loss = F.cross_entropy(head(enc.embed(Xtr[idx])), ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); head.eval()
    with torch.no_grad():
        pte = F.softmax(head(enc.embed(torch.as_tensor(codes_te, dtype=torch.long).to(DEV))), 1).cpu().numpy()
        pood = F.softmax(head(enc.embed(torch.as_tensor(codes_ood, dtype=torch.long).to(DEV))), 1).cpu().numpy()
    return pte, pood


def main() -> None:
    rows_map, names = collect_genome_wide_strings(discover_chrom_vcfs(POOL_DIR, prefix=POOL_PREFIX))
    sids = sorted(rows_map)
    rows = [[rows_map[s].get(m, "N|N") for m in names] for s in sids]
    M = len(names)
    pop, sup = read_panel()
    eas_i = [i for i, s in enumerate(sids) if sup.get(s) == "EAS"]
    ood_i = [i for i, s in enumerate(sids) if sup.get(s) not in ("EAS", None)][:400]
    classes = sorted(EAS)
    cmap = {c: j for j, c in enumerate(classes)}
    y_eas = np.array([cmap[pop[sids[i]]] for i in eas_i])
    print(f"pool={len(sids)} markers={M} EAS={len(eas_i)} OOD={len(ood_i)} device={DEV}", flush=True)

    vocab = FMVocab(rows, k=K)
    codes_all = vocab.encode(rows)
    encoder = MHTransformer(M, K, D_MODEL, N_LAYERS, N_HEADS).to(DEV)
    proj = Proj(D_MODEL).to(DEV)
    print("pretraining on full pool (masked + contrastive)...", flush=True)
    pretrain(encoder, proj, rows, vocab, PRETRAIN_EPOCHS)
    state = {k: v.detach().clone() for k, v in encoder.state_dict().items()}

    codes_eas = codes_all[eas_i]
    codes_ood = codes_all[ood_i]
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    accs, aurocs = [], []
    for fi, (tr, te) in enumerate(cv.split(codes_eas, y_eas)):
        pte, pood = finetune_eval(state, codes_eas[tr], y_eas[tr], codes_eas[te],
                                  codes_ood, M, len(classes), seed=fi)
        acc = float((pte.argmax(1) == y_eas[te]).mean())
        au = ood_auroc(msp_score(pte), msp_score(pood))
        accs.append(acc); aurocs.append(au)
        print(f"fold {fi+1}: acc={acc:.3f} far-OOD AUROC={au:.3f}", flush=True)

    out = {"pool_n": len(sids), "markers": M, "n_eas": len(eas_i), "n_ood": len(ood_i),
           "pretrain_epochs": PRETRAIN_EPOCHS, "contrastive": True,
           "acc_mean": round(float(np.mean(accs)), 4), "acc_std": round(float(np.std(accs)), 4),
           "auroc_mean": round(float(np.mean(aurocs)), 4), "auroc_std": round(float(np.std(aurocs)), 4),
           "reference": {"FM_n504_SSL": 0.546, "LogReg_onehot": 0.796}}
    Path("results/baseline").mkdir(parents=True, exist_ok=True)
    Path("results/baseline/ssl_retrain.json").write_text(json.dumps(out, indent=2))
    print(f"\nSSL(pool={len(sids)}) acc={out['acc_mean']*100:.1f}±{out['acc_std']*100:.1f}  "
          f"far-OOD AUROC={out['auroc_mean']:.3f}  | ref: FM@504 54.6, LogReg 79.6", flush=True)
    print("saved results/baseline/ssl_retrain.json", flush=True)
    print("SSL_RETRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
