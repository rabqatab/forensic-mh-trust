import numpy as np
import torch

from forensic_mh.fm.encoder import MHTransformer
from forensic_mh.fm.finetune import multitask_finetune
from forensic_mh.fm.heads import AncestryHead, SexHead


def test_finetune_runs_and_improves_ancestry_accuracy():
    rng = np.random.default_rng(0)
    N, M, K = 64, 4, 4
    X = torch.from_numpy(rng.integers(0, K, size=(N, M)))
    y_anc = torch.from_numpy(rng.integers(0, 3, size=N))
    # make ancestry learnable: marker 0 code correlates with label
    X[:, 0] = y_anc
    y_sex = torch.from_numpy(rng.integers(0, 2, size=N)).float()
    torch.manual_seed(0)   # seed encoder/head init (else accuracy assertion is RNG-order dependent)
    enc = MHTransformer(n_markers=M, k=K, d_model=16, n_layers=1, n_heads=2)
    heads = {"ancestry": AncestryHead(16, 3), "sex": SexHead(16)}
    hist = multitask_finetune(enc, heads, X, y_anc, y_sex,
                              epochs=20, batch_size=16, lr=1e-3, seed=0)
    assert hist[-1]["ancestry_acc"] > hist[0]["ancestry_acc"]
    assert hist[-1]["ancestry_acc"] > 0.5
