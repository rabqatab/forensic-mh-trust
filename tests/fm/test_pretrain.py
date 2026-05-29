import torch

from forensic_mh.fm.dataset import MHMatrixDataset
from forensic_mh.fm.encoder import MHTransformer
from forensic_mh.fm.pretrain import ssl_pretrain
from forensic_mh.fm.vocab import FMVocab


def _setup():
    rows = [["A|T", "G|G", "C|C"], ["A|A", "G|C", "C|T"],
            ["A|T", "C|C", "C|C"], ["T|T", "G|C", "T|T"]] * 4  # 16 samples
    v = FMVocab(rows, k=4)
    ds = MHMatrixDataset(rows, v, seed=0)
    enc = MHTransformer(n_markers=3, k=4, d_model=16, n_layers=1, n_heads=2)
    return enc, ds


def test_ssl_pretrain_runs_and_returns_finite_losses():
    enc, ds = _setup()
    history = ssl_pretrain(enc, ds, epochs=2, batch_size=8, lr=1e-3,
                           lambda_contrastive=0.5, seed=0)
    assert len(history) == 2
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in history)


def test_ssl_pretrain_decreases_loss_on_tiny_data():
    enc, ds = _setup()
    history = ssl_pretrain(enc, ds, epochs=8, batch_size=8, lr=1e-3,
                           lambda_contrastive=0.5, seed=0)
    assert history[-1]["loss"] < history[0]["loss"]   # learns something
