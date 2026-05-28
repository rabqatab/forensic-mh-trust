"""MicroHapDB wrapper — MH marker coordinates.

Important: 1000 Genomes Phase 3 uses GRCh37 (hg19) coordinates,
while MicroHapDB 0.12 reports both `Positions` (hg38) and `Positions37` (hg19).
For 1000G work, use the `hg19` build (default).
"""
from __future__ import annotations
import pandas as pd
import microhapdb


# Column name lookup per genome build
POSITION_COL = {
    "hg19": "Positions37",
    "grch37": "Positions37",
    "hg38": "Positions",
    "grch38": "Positions",
}


def load_mh_markers() -> pd.DataFrame:
    """Return MicroHapDB markers DataFrame.

    Columns of interest:
      - Name: marker ID (e.g., 'mh22XYZ-001')
      - Chrom: 'chr22' format
      - Positions: hg38 SNP positions, semicolon-delimited
      - Positions37: hg19 SNP positions, semicolon-delimited (NaN if not available)
      - Ae: precomputed effective number of alleles
    """
    return microhapdb.markers.copy()


def filter_by_chromosome(markers: pd.DataFrame, chrom: str) -> pd.DataFrame:
    """Return markers on a given chromosome (e.g., 'chr22' or '22')."""
    target = chrom.replace("chr", "")
    return markers[markers["Chrom"].str.replace("chr", "") == target].copy()


def filter_with_coords(markers: pd.DataFrame, build: str = "hg19") -> pd.DataFrame:
    """Drop markers without coordinates in the requested genome build.

    Markers added to MicroHapDB after the hg19→hg38 transition may have
    NaN in `Positions37`. Filtering avoids extraction errors downstream.
    """
    col = POSITION_COL[build.lower()]
    return markers[markers[col].notna()].copy()


def parse_positions(marker: pd.Series, build: str = "hg19") -> list[int]:
    """Parse the semicolon-delimited position string into a sorted list[int].

    Returns empty list if positions for the requested build are missing.
    """
    col = POSITION_COL[build.lower()]
    val = marker.get(col)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    return sorted(int(p) for p in str(val).split(";") if p.strip())
