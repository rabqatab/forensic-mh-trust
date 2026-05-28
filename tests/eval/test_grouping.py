import numpy as np
import pandas as pd

from forensic_mh.eval.grouping import build_groups_from_panel


def test_build_groups_from_panel_returns_one_id_per_sample():
    panel = pd.DataFrame({
        "sample": ["S1", "S2", "S3", "S4", "S5"],
        "pop": ["CHB", "CHB", "JPT", "JPT", "KHV"],
        "super_pop": ["EAS"] * 5,
        "family_id": ["F1", "F1", "F2", "F3", "F4"],  # S1+S2 related
    })
    groups = build_groups_from_panel(panel, sample_ids=["S1", "S2", "S3", "S4", "S5"])
    assert len(groups) == 5
    assert groups[0] == groups[1]  # S1, S2 share family → same group
    assert groups[2] != groups[3]  # different families


def test_build_groups_fallback_to_sample_when_family_id_missing():
    panel = pd.DataFrame({
        "sample": ["S1", "S2"],
        "pop": ["CHB", "JPT"],
        "super_pop": ["EAS", "EAS"],
    })
    groups = build_groups_from_panel(panel, sample_ids=["S1", "S2"])
    assert len(groups) == 2
    assert groups[0] != groups[1]  # treated as unrelated
