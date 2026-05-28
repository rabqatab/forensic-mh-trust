"""Wei 2025-inspired Reliable Ae metric (P0 #4 response).

    Reliable_Ae(marker) = Ae(marker) × (1 - P_phasing_error(marker))

P_phasing_error is estimated via trio Mendelian-consistency: count
diplotypes that cannot be produced from parental haplotypes.

Reference: Wei, Li, Zhu (2025) FSI:G — phasing error rate rises with Ae,
so high-Ae markers (which a naive selector prefers) are the riskiest.
"""
from __future__ import annotations


def is_mendelian_consistent_diplotype(
    father: tuple[str, str],
    mother: tuple[str, str],
    child: tuple[str, str],
) -> bool:
    """Check if child's diplotype can arise from one parental haplotype each.

    Each diplotype is an unordered tuple of haplotype strings. The child must
    consist of one haplotype contributed by the father and one by the mother.
    """
    f_haps = set(father)
    m_haps = set(mother)
    c0, c1 = child
    for cf, cm in ((c0, c1), (c1, c0)):
        if cf in f_haps and cm in m_haps:
            return True
    return False


def reliable_ae(ae: float, p_phase_error: float) -> float:
    """Penalise Ae by the marker's estimated phasing-error probability."""
    return ae * (1.0 - p_phase_error)
