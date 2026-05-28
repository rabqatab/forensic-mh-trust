"""Group labels for GroupKFold — prevents related individuals leaking across CV folds.

P0 fix from review #8.

The 1000G panel does not include explicit family_id by default; the file
20130606_g1k.ped (separately downloadable) provides pedigree. Until that
is integrated, fall back to sample_id (treats all as unrelated — overly
optimistic but matches current behavior). A follow-up adds the pedigree
join and full GroupKFold validity.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_groups_from_panel(
    panel: pd.DataFrame, sample_ids: list[str]
) -> np.ndarray:
    """Return integer group IDs (one per sample) for GroupKFold.

    If `family_id` column exists in panel, samples sharing family_id get the
    same group ID. Otherwise each sample is its own group.
    """
    panel_idx = panel.set_index("sample")
    if "family_id" in panel.columns:
        fam_ids = [panel_idx.loc[s, "family_id"] for s in sample_ids]
        unique_fams = sorted(set(fam_ids))
        fam_to_int = {f: i for i, f in enumerate(unique_fams)}
        return np.array([fam_to_int[f] for f in fam_ids], dtype=int)
    return np.arange(len(sample_ids), dtype=int)
