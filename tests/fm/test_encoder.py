import torch

from forensic_mh.fm.encoder import MHTransformer


def test_embed_shape():
    enc = MHTransformer(n_markers=5, k=4, d_model=16, n_layers=2, n_heads=2)
    x = torch.randint(0, 5, (3, 5))   # codes in [0, k+1)
    emb = enc.embed(x)
    assert emb.shape == (3, 16)


def test_masked_logits_shape_over_value_classes_only():
    enc = MHTransformer(n_markers=5, k=4, d_model=16, n_layers=2, n_heads=2)
    x = torch.randint(0, 5, (3, 5))
    logits = enc.masked_logits(x)
    assert logits.shape == (3, 5, 4)   # (N, M, K) — excludes MASK slot


def test_forward_is_differentiable():
    enc = MHTransformer(n_markers=4, k=4, d_model=8, n_layers=1, n_heads=2)
    x = torch.randint(0, 5, (2, 4))
    loss = enc.embed(x).sum()
    loss.backward()
    assert next(enc.parameters()).grad is not None
