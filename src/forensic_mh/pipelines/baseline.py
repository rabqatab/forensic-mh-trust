"""End-to-end baseline pipeline: VCF → diplotype matrix → label-encoded X → CV."""
from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from forensic_mh.data.markers import (
    filter_by_chromosome,
    filter_with_coords,
    load_mh_markers,
    parse_positions,
)
from forensic_mh.data.vcf_io import extract_diplotypes_for_locus


def build_diplotype_matrix(
    vcf_path: str,
    chrom: str,
    sample_ids: Optional[list[str]] = None,
    build: str = "hg19",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build (n_samples, n_markers) integer diplotype matrix for one chromosome.

    Each cell: integer code of the canonical diplotype tuple at that marker
    (per-marker LabelEncoder).

    Returns:
        X: (n_samples, n_markers) int matrix
        sample_ids_out: ordered list of sample IDs (rows)
        marker_names: ordered list of MH names (columns)
    """
    markers = filter_with_coords(load_mh_markers(), build=build)
    markers = filter_by_chromosome(markers, chrom)
    if len(markers) == 0:
        raise ValueError(f"No MH markers on chromosome {chrom} for build {build}")

    chrom_id = chrom.replace("chr", "")
    rows: dict[str, dict[str, str]] = {}
    marker_names: list[str] = []

    for _, mh in markers.iterrows():
        positions = parse_positions(mh, build=build)
        if not positions:
            continue
        name = mh["Name"]
        marker_names.append(name)
        diplotypes = extract_diplotypes_for_locus(
            vcf_path, chrom_id, positions, sample_ids
        )
        for sid, dipl in diplotypes.items():
            rows.setdefault(sid, {})[name] = f"{dipl[0]}|{dipl[1]}"

    sample_ids_out = sorted(rows.keys())
    X = np.zeros((len(sample_ids_out), len(marker_names)), dtype=np.int32)
    for j, m in enumerate(marker_names):
        col = [rows[s].get(m, "N|N") for s in sample_ids_out]
        le = LabelEncoder()
        X[:, j] = le.fit_transform(col)
    return X, sample_ids_out, marker_names


def load_eas_labels(
    panel_path: str, sample_ids: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Return integer pop labels + ordered pop name list."""
    panel = pd.read_csv(panel_path, sep="\t")
    pop_map = panel.set_index("sample")["pop"].to_dict()
    raw = [pop_map[s] for s in sample_ids]
    pops = sorted(set(raw))
    pop_to_int = {p: i for i, p in enumerate(pops)}
    y = np.array([pop_to_int[p] for p in raw], dtype=int)
    return y, pops
