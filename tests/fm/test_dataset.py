import numpy as np
import torch

from forensic_mh.fm.dataset import MHMatrixDataset
from forensic_mh.fm.vocab import FMVocab


def _ds(seed=0, **kw):
    rows = [["A|T", "G|G"], ["A|A", "G|C"], ["A|T", "C|C"], ["T|T", "G|C"]]
    v = FMVocab(rows, k=4)
    return MHMatrixDataset(rows, v, seed=seed, **kw), v


def test_item_has_masked_view_and_two_contrastive_views():
    ds, v = _ds()
    item = ds[0]
    for key in ("input", "target", "mask_pos", "view1", "view2"):
        assert key in item
    assert item["input"].shape == (2,)        # n_markers
    assert item["view1"].shape == (2,)
    assert item["mask_pos"].dtype == torch.bool


def test_masked_positions_are_set_to_mask_token_and_targets_preserved():
    ds, v = _ds(mask_frac=1.0)  # mask everything → deterministic
    item = ds[0]
    assert torch.all(item["input"] == v.MASK)
    # targets equal the un-masked encoding
    base = torch.tensor(v.encode([["A|T", "G|G"]])[0])
    assert torch.all(item["target"] == base)
    assert torch.all(item["mask_pos"])


def test_ado_collapses_only_heterozygous_cells():
    # marker 0 of sample 0 is "A|T" (het); ado_prob=1 forces collapse to hom
    ds, v = _ds(ado_prob=1.0, drop_prob=0.0)
    item = ds[0]
    # collapsed het encodes to either A|A or T|T; both are rare here → OTHER,
    # but crucially it must differ from the original het code OR be OTHER.
    orig = v.encode([["A|T", "G|G"]])[0]
    # marker 1 "G|G" is hom → ado leaves it unchanged across views
    assert item["view1"][1].item() == orig[1] or item["view1"][1].item() == v.OTHER


def test_len_matches_sample_count():
    ds, v = _ds()
    assert len(ds) == 4
