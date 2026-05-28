import pytest

from forensic_mh.metrics.reliable_ae import (
    compute_ae,
    is_informative_meiosis,
    is_mendelian_consistent_diplotype,
    reliable_ae,
)


def test_compute_ae_uniform_two_haplotypes():
    # two haplotypes at 50/50 → Ae = 1/(0.5²+0.5²) = 2.0
    d = {"s1": ("A-T", "A-T"), "s2": ("G-C", "G-C")}
    assert compute_ae(d) == pytest.approx(2.0)


def test_compute_ae_skips_missing():
    d = {"s1": ("A-T", "N-N"), "s2": ("A-T", "A-T")}
    # only three callable A-T haplotypes → monomorphic → Ae = 1.0
    assert compute_ae(d) == pytest.approx(1.0)


def test_compute_ae_empty_returns_zero():
    assert compute_ae({"s1": ("N-N", "N-N")}) == pytest.approx(0.0)


def test_informative_meiosis_only_when_child_heterozygous():
    assert is_informative_meiosis(("A-T", "G-C")) is True
    assert is_informative_meiosis(("A-T", "A-T")) is False


def test_mendelian_consistent_trivial_homozygote():
    # father A-T/A-T, mother G-C/G-C → child must be A-T/G-C
    father = ("A-T", "A-T")
    mother = ("G-C", "G-C")
    child = ("A-T", "G-C")  # canonical sorted
    assert is_mendelian_consistent_diplotype(father, mother, child)


def test_mendelian_inconsistent_impossible_child():
    father = ("A-T", "A-T")
    mother = ("G-C", "G-C")
    child = ("A-T", "A-T")  # mother contributed nothing — impossible
    assert not is_mendelian_consistent_diplotype(father, mother, child)


def test_mendelian_consistent_heterozygote():
    father = ("A-T", "A-G")  # contributes A-T or A-G
    mother = ("G-C", "T-C")  # contributes G-C or T-C
    child = ("A-G", "T-C")   # A-G from father + T-C from mother → OK
    assert is_mendelian_consistent_diplotype(father, mother, child)


def test_reliable_ae_lowers_score_when_phasing_errors_present():
    ae = 5.0
    assert reliable_ae(ae, p_phase_error=0.1) == pytest.approx(4.5)
    assert reliable_ae(ae, p_phase_error=0.0) == pytest.approx(5.0)
    assert reliable_ae(ae, p_phase_error=1.0) == pytest.approx(0.0)
