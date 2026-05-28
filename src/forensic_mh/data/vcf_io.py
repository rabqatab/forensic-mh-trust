"""VCF → diplotype extraction.

Critical P0 fix from review #2: the original proposal used `gt[0]` only
(one haplotype), losing all heterozygote information. We extract BOTH
haplotypes per sample and form unordered diplotype tuples.
"""
from __future__ import annotations
from typing import Iterable, Optional

import cyvcf2


def extract_diplotypes_for_locus(
    vcf_path: str,
    chrom: str,
    positions: Iterable[int],
    sample_ids: Optional[Iterable[str]] = None,
) -> dict[str, tuple[str, str]]:
    """Extract per-sample diplotypes at a microhaplotype locus.

    Args:
        vcf_path: phased VCF (1000G shapeit2)
        chrom: chromosome name as in VCF (e.g., '22', not 'chr22' for 1000G)
        positions: 1-based SNP positions defining the MH
        sample_ids: optional subset; if None, all samples in VCF

    Returns:
        {sample_id: (haplotype_A, haplotype_B)} where each haplotype is
        "-".join(alleles) at the given positions, and the tuple is in
        canonical (sorted) order so diplotypes compare equal regardless
        of which haplotype is listed first.
    """
    vcf = cyvcf2.VCF(vcf_path)
    all_samples = list(vcf.samples)
    targets = list(sample_ids) if sample_ids is not None else all_samples
    target_idx = [all_samples.index(s) for s in targets]

    # one list per haplotype slot, per sample
    hap_alleles: dict[str, list[list[str]]] = {s: [[], []] for s in targets}

    positions_list = sorted(int(p) for p in positions)
    for pos in positions_list:
        region = f"{chrom}:{pos}-{pos}"
        found = False
        for rec in vcf(region):
            if rec.POS != pos:
                continue
            found = True
            # cyvcf2 genotypes: list[[a, b, phased_flag]] per sample
            gts = rec.genotypes
            alleles = [rec.REF] + list(rec.ALT)
            for sid, idx in zip(targets, target_idx):
                a, b, _phased = gts[idx]
                hap_alleles[sid][0].append(alleles[a] if a >= 0 else "N")
                hap_alleles[sid][1].append(alleles[b] if b >= 0 else "N")
            break
        if not found:
            for sid in targets:
                hap_alleles[sid][0].append("N")
                hap_alleles[sid][1].append("N")

    diplotypes: dict[str, tuple[str, str]] = {}
    for sid in targets:
        h0 = "-".join(hap_alleles[sid][0])
        h1 = "-".join(hap_alleles[sid][1])
        diplotypes[sid] = tuple(sorted([h0, h1]))
    return diplotypes
