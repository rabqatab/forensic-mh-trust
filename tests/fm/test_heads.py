import torch

from forensic_mh.fm.heads import AncestryHead, KinshipHead, SexHead


def test_ancestry_head_outputs_logits_per_class():
    h = AncestryHead(d_model=16, n_classes=5)
    out = h(torch.randn(4, 16))
    assert out.shape == (4, 5)


def test_sex_head_outputs_single_logit():
    h = SexHead(d_model=16)
    out = h(torch.randn(4, 16))
    assert out.shape == (4, 1)


def test_kinship_head_is_symmetric_in_pair_order():
    h = KinshipHead(d_model=16)
    a, b = torch.randn(3, 16), torch.randn(3, 16)
    # |a-b| and a*b are symmetric → swapping pair order gives same logit
    assert torch.allclose(h(a, b), h(b, a), atol=1e-6)
