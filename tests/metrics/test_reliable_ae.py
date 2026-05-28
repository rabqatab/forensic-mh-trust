import pytest

from forensic_mh.metrics.reliable_ae import (
    is_mendelian_consistent_diplotype,
    reliable_ae,
)


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
