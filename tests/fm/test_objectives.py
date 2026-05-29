import torch

from forensic_mh.fm.objectives import masked_marker_loss, nt_xent


def test_masked_loss_zero_when_logits_perfect_on_masked_only():
    N, M, K = 2, 3, 4
    targets = torch.tensor([[0, 1, 2], [3, 0, 1]])
    mask_pos = torch.tensor([[True, False, True], [False, True, False]])
    logits = torch.full((N, M, K), -10.0)
    # put a huge value on the true class everywhere
    for n in range(N):
        for m in range(M):
            logits[n, m, targets[n, m]] = 10.0
    loss = masked_marker_loss(logits, targets, mask_pos)
    assert loss.item() < 1e-3          # near zero
    assert loss.requires_grad is False or loss.item() >= 0


def test_masked_loss_ignores_unmasked_positions():
    N, M, K = 1, 2, 3
    targets = torch.tensor([[0, 0]])
    mask_pos = torch.tensor([[True, False]])
    good = torch.tensor([[[10.0, -10, -10], [10.0, -10, -10]]])
    bad = good.clone(); bad[0, 1] = torch.tensor([-10.0, 10, -10])  # wrong at unmasked
    assert torch.allclose(masked_marker_loss(good, targets, mask_pos),
                          masked_marker_loss(bad, targets, mask_pos))


def test_nt_xent_lower_when_positives_aligned():
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    aligned = z1.clone()
    crossed = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert nt_xent(z1, aligned, tau=0.5) < nt_xent(z1, crossed, tau=0.5)
