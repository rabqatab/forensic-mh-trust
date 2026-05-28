"""Diplotype extraction tests (P0 fix from review #2).

Integration tests against the actual EAS chr22 VCF + a real MH locus
from MicroHapDB (hg19 coordinates).
"""
from pathlib import Path

import pytest

from forensic_mh.data.markers import (
    filter_by_chromosome,
    filter_with_coords,
    load_mh_markers,
    parse_positions,
)
from forensic_mh.data.vcf_io import extract_diplotypes_for_locus


TEST_VCF = Path("data/eas/EAS_chr22.vcf.gz")
pytestmark = pytest.mark.skipif(
    not TEST_VCF.exists(),
    reason="Run scripts/02_extract_eas_samples.sh first to create EAS_chr22.vcf.gz",
)


@pytest.fixture(scope="module")
def first_chr22_mh_positions() -> list[int]:
    """Real positions from the first chr22 MH locus (hg19)."""
    markers = filter_with_coords(load_mh_markers(), build="hg19")
    chr22 = filter_by_chromosome(markers, "chr22")
    # pick the first marker that lies within the EAS VCF range
    for _, mh in chr22.iterrows():
        positions = parse_positions(mh, build="hg19")
        if positions and positions[0] > 16_050_000:
            return positions
    pytest.skip("No chr22 MH within EAS VCF range")


def test_diplotype_returns_two_haplotypes_per_sample(first_chr22_mh_positions):
    diplotypes = extract_diplotypes_for_locus(
        str(TEST_VCF),
        chrom="22",
        positions=first_chr22_mh_positions,
        sample_ids=None,  # all 504
    )
    assert len(diplotypes) == 504, "expected all 504 EAS samples"
    n_pos = len(first_chr22_mh_positions)
    for sid, dipl in list(diplotypes.items())[:10]:
        assert isinstance(dipl, tuple) and len(dipl) == 2
        # each haplotype is a "-".join of alleles, one per position
        # so n_positions-1 separators
        assert dipl[0].count("-") == n_pos - 1, f"{sid}: hap1 has wrong arity"
        assert dipl[1].count("-") == n_pos - 1, f"{sid}: hap2 has wrong arity"


def test_diplotype_is_in_canonical_sorted_order(first_chr22_mh_positions):
    """diplotype (A-T, G-C) should equal (G-C, A-T) — sorted tuple."""
    diplotypes = extract_diplotypes_for_locus(
        str(TEST_VCF),
        chrom="22",
        positions=first_chr22_mh_positions,
    )
    for sid, dipl in list(diplotypes.items())[:20]:
        assert dipl[0] <= dipl[1], f"{sid}: diplotype not canonical: {dipl}"


def test_diplotype_handles_missing_position_gracefully(first_chr22_mh_positions):
    """Positions not present in VCF should fill with 'N', not crash."""
    fake_positions = first_chr22_mh_positions + [999_999_999]
    diplotypes = extract_diplotypes_for_locus(
        str(TEST_VCF),
        chrom="22",
        positions=fake_positions,
    )
    # Last position is fake → last allele should be 'N' for all samples
    for sid, dipl in list(diplotypes.items())[:5]:
        assert dipl[0].split("-")[-1] == "N"
        assert dipl[1].split("-")[-1] == "N"


def test_diplotype_preserves_heterozygote_information(first_chr22_mh_positions):
    """At least one EAS sample should have a heterozygous diplotype
    (hap1 != hap2). The original proposal's bug would collapse to one haplotype.
    """
    diplotypes = extract_diplotypes_for_locus(
        str(TEST_VCF),
        chrom="22",
        positions=first_chr22_mh_positions,
    )
    n_het = sum(1 for h1, h2 in diplotypes.values() if h1 != h2)
    assert n_het > 0, (
        "No heterozygous diplotypes detected — possible bug in haplotype "
        "extraction (proposal review #2)"
    )
