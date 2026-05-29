import numpy as np

from forensic_mh.eval.lopo import leave_one_population_out


def test_lopo_yields_one_split_per_population():
    y = np.array([0, 0, 1, 1, 2, 2])
    pops = ["CDX", "CHB", "CHS"]
    splits = list(leave_one_population_out(y, pops))
    assert len(splits) == 3
    for held_name, in_idx, out_idx in splits:
        # held population absent from in-distribution indices
        assert held_name in pops
        assert len(set(y[in_idx]) & {pops.index(held_name)}) == 0
        assert all(y[i] == pops.index(held_name) for i in out_idx)
