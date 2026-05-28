"""Wei 2025-inspired Reliable Ae metric (P0 #4 response).

    Reliable_Ae(marker) = Ae(marker) × (1 - P_phasing_error(marker))

P_phasing_error is estimated via trio Mendelian-consistency: count
diplotypes that cannot be produced from parental haplotypes.

Reference: Wei, Li, Zhu (2025) FSI:G — phasing error rate rises with Ae,
so high-Ae markers (which a naive selector prefers) are the riskiest.
"""
from __future__ import annotations

from collections import Counter


def compute_ae(diplotypes: dict[str, tuple[str, str]]) -> float:
    """Effective number of alleles Ae = 1 / Σ p_i² over HAPLOTYPE frequencies.

    Ae is computed on haplotype (allele) frequencies — each sample contributes
    its two haplotypes. Haplotypes containing 'N' (missing) are skipped.
    Returns 0.0 if no callable haplotypes.
    """
    hap_counts: Counter = Counter()
    for h0, h1 in diplotypes.values():
        for h in (h0, h1):
            if "N" not in h:
                hap_counts[h] += 1
    total = sum(hap_counts.values())
    if total == 0:
        return 0.0
    sum_sq = sum((c / total) ** 2 for c in hap_counts.values())
    return 1.0 / sum_sq if sum_sq > 0 else 0.0


def is_informative_meiosis(child: tuple[str, str]) -> bool:
    """A meiosis reveals a within-haplotype switch error only if the child is
    heterozygous at the locus (two distinct haplotypes). Homozygous children
    are uninformative and MUST be excluded from the phasing-error denominator,
    otherwise the error rate is biased toward 0."""
    return child[0] != child[1]


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
