"""End-to-end baseline pipeline: VCF → diplotype matrix → label-encoded X → CV."""
from __future__ import annotations

import glob
import os
import re
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


def _chrom_sort_key(chrom: str) -> tuple[int, object]:
    """Order chromosomes numerically (1..22) then non-numeric (X, Y, MT)."""
    c = str(chrom).replace("chr", "")
    return (0, int(c)) if c.isdigit() else (1, c)


def _collect_chrom(
    vcf_path: str,
    chrom: str,
    sample_ids: Optional[list[str]],
    build: str,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Collect per-sample diplotype STRINGS ('h0|h1') for one chromosome's MH.

    Returns (rows, marker_names) where rows[sample][marker_name] = 'h0|h1'.
    Pure string collection — encoding is deferred to `_encode` so single- and
    genome-wide builders share one encoding step.
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
    return rows, marker_names


def _encode(
    rows: dict[str, dict[str, str]], marker_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Encode collected diplotype strings into an int matrix (per-marker
    LabelEncoder). Samples missing a marker get the canonical 'N|N' code.

    Returns (X, sample_ids_out) with rows in sorted sample-ID order.
    """
    sample_ids_out = sorted(rows.keys())
    X = np.zeros((len(sample_ids_out), len(marker_names)), dtype=np.int32)
    for j, m in enumerate(marker_names):
        col = [rows[s].get(m, "N|N") for s in sample_ids_out]
        X[:, j] = LabelEncoder().fit_transform(col)
    return X, sample_ids_out


def build_diplotype_matrix(
    vcf_path: str,
    chrom: str,
    sample_ids: Optional[list[str]] = None,
    build: str = "hg19",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build (n_samples, n_markers) integer diplotype matrix for ONE chromosome.

    Each cell: integer code of the canonical diplotype tuple at that marker
    (per-marker LabelEncoder).

    Returns:
        X: (n_samples, n_markers) int matrix
        sample_ids_out: ordered list of sample IDs (rows)
        marker_names: ordered list of MH names (columns)
    """
    rows, marker_names = _collect_chrom(vcf_path, chrom, sample_ids, build)
    X, sample_ids_out = _encode(rows, marker_names)
    return X, sample_ids_out, marker_names


def discover_chrom_vcfs(
    directory: str = "data/eas", prefix: str = "EAS_chr", suffix: str = ".vcf.gz"
) -> dict[str, str]:
    """Discover per-chromosome VCFs as a {chrom: path} dict for genome-wide build.

    Matches files like 'EAS_chr1.vcf.gz' → {"1": ".../EAS_chr1.vcf.gz"}.
    Useful to feed build_genome_wide_matrix once scripts/06 finishes.
    """
    pattern = os.path.join(directory, f"{prefix}*{suffix}")
    out: dict[str, str] = {}
    rx = re.compile(re.escape(prefix) + r"([0-9XYMT]+)" + re.escape(suffix) + r"$")
    for path in glob.glob(pattern):
        m = rx.search(os.path.basename(path))
        if m:
            out[m.group(1)] = path
    return out


def build_genome_wide_matrix(
    vcf_paths: dict[str, str],
    sample_ids: Optional[list[str]] = None,
    build: str = "hg19",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a genome-wide diplotype matrix from per-chromosome VCFs.

    Args:
        vcf_paths: {chrom: vcf_path}, e.g. {"1": "data/eas/EAS_chr1.vcf.gz", ...}.
            Chromosome keys may be "1" or "chr1".
        sample_ids: optional subset (must exist in every VCF); None = all.
        build: coordinate build for marker positions ("hg19" or "hg38").

    Markers are concatenated in sorted chromosome order (1..22, then X/Y/MT).
    Encoding happens once over the merged columns, so codes are consistent
    across the full matrix.

    Returns:
        X, sample_ids_out, marker_names  (same shape contract as
        build_diplotype_matrix, columns spanning all chromosomes).
    """
    merged_rows, all_marker_names = collect_genome_wide_strings(
        vcf_paths, sample_ids, build)
    X, sample_ids_out = _encode(merged_rows, all_marker_names)
    return X, sample_ids_out, all_marker_names


def collect_genome_wide_strings(
    vcf_paths: dict[str, str],
    sample_ids: Optional[list[str]] = None,
    build: str = "hg19",
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Collect genome-wide diplotype STRINGS without encoding.

    Returns (merged_rows, marker_names) where merged_rows[sample][marker]='h0|h1',
    markers concatenated in sorted chromosome order. Used when a caller needs a
    SHARED encoder across groups (e.g. fit a DiplotypeEncoder on EAS strings and
    transform OOD strings for open-set evaluation).
    """
    merged_rows: dict[str, dict[str, str]] = {}
    all_marker_names: list[str] = []
    for chrom in sorted(vcf_paths, key=_chrom_sort_key):
        rows, names = _collect_chrom(vcf_paths[chrom], chrom, sample_ids, build)
        all_marker_names.extend(names)
        for sid, marker_map in rows.items():
            merged_rows.setdefault(sid, {}).update(marker_map)
    return merged_rows, all_marker_names


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
