import pandas as pd
import pytest

from forensic_mh.data.markers import (
    load_mh_markers,
    filter_by_chromosome,
    filter_with_coords,
    parse_positions,
)


def test_load_mh_markers_returns_dataframe_with_expected_columns():
    df = load_mh_markers()
    assert len(df) > 100, f"expected >100 markers in MicroHapDB, got {len(df)}"
    for col in ("Name", "Chrom", "Positions", "Positions37", "Ae"):
        assert col in df.columns, f"missing column {col}"


def test_filter_by_chromosome_returns_only_target_chromosome():
    df = load_mh_markers()
    chr22 = filter_by_chromosome(df, "chr22")
    assert len(chr22) > 0
    assert all(chr22["Chrom"].str.replace("chr", "") == "22")


def test_filter_with_coords_hg19_keeps_only_markers_with_positions37():
    df = load_mh_markers()
    n_total = len(df)
    df37 = filter_with_coords(df, build="hg19")
    assert len(df37) <= n_total
    assert df37["Positions37"].notna().all()


def test_parse_positions_returns_sorted_int_list_for_hg19():
    df = filter_with_coords(load_mh_markers(), build="hg19")
    chr22 = filter_by_chromosome(df, "chr22")
    first = chr22.iloc[0]
    positions = parse_positions(first, build="hg19")
    assert len(positions) >= 2  # MH requires ≥2 SNPs by definition
    assert positions == sorted(positions)
    assert all(isinstance(p, int) for p in positions)


def test_parse_positions_handles_missing_build_gracefully():
    fake = pd.Series({"Positions37": None, "Positions": "100;200"})
    assert parse_positions(fake, build="hg19") == []
    assert parse_positions(fake, build="hg38") == [100, 200]
