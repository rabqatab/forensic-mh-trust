"""Day 3 (교정본): per-MH EAS Ae 표 + (가능 시) trio phasing error.

상태 (2026-05-29): trio phasing은 **deferred**.
  - 1000G phase3 표준 release(2504)는 unrelated subset → complete trio 0개.
  - related_samples VCF(31명)를 합쳐도 complete trio 6개(전부 non-EAS) → 통계 무의미.
  - Wei 2025 재현은 NYGC 30x high-coverage release(3202명, 602 trios, GRCh38) 필요.
  따라서 이 스크립트는 지금 **EAS Ae 표**를 산출하고, complete trio 수가
  MIN_TRIOS 미만이면 p_phase_error/reliable_ae를 null + status로 남긴다.
  NYGC VCF 확보 시 TRIO_VCF_SOURCES에 경로 추가 + BUILD="hg38"로 그대로 재현.

Reference: Wei, Li, Zhu (2025) FSI:G — phasing error는 Ae가 높을수록 증가 →
naive high-Ae 선택 전략이 가장 위험한 마커를 고르는 셈. Ae 표가 그 위험 노출도.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from forensic_mh.data.markers import (
    filter_by_chromosome,
    load_mh_markers,
    parse_positions,
)
from forensic_mh.data.vcf_io import extract_diplotypes_for_locus
from forensic_mh.metrics.reliable_ae import (
    compute_ae,
    is_informative_meiosis,
    is_mendelian_consistent_diplotype,
    reliable_ae,
)

PED_CANDIDATES = [
    "data/1000g/g1k.ped",
    "data/1000g/integrated_call_samples_v3.20130502.ALL.ped",
]
VCF_EAS = "data/eas/EAS_chr22.vcf.gz"
PANEL = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
# trio genotype sources (members are split across files). NYGC 추가 시 여기에 append.
TRIO_VCF_SOURCES = [
    "data/1000g/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz",
    "data/1000g/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5_related_samples.20130502.genotypes.vcf.gz",
]
BUILD = "hg19"      # phase3 좌표. NYGC면 "hg38".
MIN_TRIOS = 30      # 이 미만이면 phasing 통계 deferred 처리


def _vcf_samples(path: str) -> set[str]:
    return set(subprocess.check_output(["bcftools", "query", "-l", path]).decode().split())


def load_trios(sample_universe: set[str]) -> list[tuple[str, str, str]]:
    ped = next((p for p in PED_CANDIDATES if Path(p).exists()), None)
    if ped is None:
        raise FileNotFoundError("g1k.ped 필요 (data/1000g/)")
    df = pd.read_csv(ped, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    pa, ma, ind = "Paternal ID", "Maternal ID", "Individual ID"
    trios = df[(df[pa].astype(str) != "0") & (df[ma].astype(str) != "0")]
    return [
        (r[pa], r[ma], r[ind])
        for _, r in trios.iterrows()
        if {r[pa], r[ma], r[ind]} <= sample_universe
    ]


def extract_combined(sources, chrom, positions, sample_ids):
    """Query each VCF for the members it contains; merge per-sample diplotypes."""
    out = {}
    for vcf, samples_in_vcf in sources:
        targets = [s for s in sample_ids if s in samples_in_vcf]
        if targets:
            out.update(extract_diplotypes_for_locus(vcf, chrom, positions, targets))
    return out


def main() -> None:
    sources = [(p, _vcf_samples(p)) for p in TRIO_VCF_SOURCES if Path(p).exists()]
    universe = set().union(*[s for _, s in sources]) if sources else set()
    trios = load_trios(universe)
    phasing_enabled = len(trios) >= MIN_TRIOS
    print(f"[1/3] complete trios in trio-VCF union: {len(trios)} "
          f"(MIN_TRIOS={MIN_TRIOS} → phasing {'ENABLED' if phasing_enabled else 'DEFERRED'})")

    markers = filter_by_chromosome(load_mh_markers(), "chr22")
    eas_ids = Path("data/eas/EAS_samples.txt").read_text().split()
    trio_ids = sorted({i for t in trios for i in t})
    print(f"[2/3] {len(markers)} chr22 MH; {len(eas_ids)} EAS; {len(trio_ids)} trio members")

    print("[3/3] per-MH EAS Ae" + (" + phasing" if phasing_enabled else " (phasing deferred)") + "...")
    results = []
    for _, mh in markers.iterrows():
        positions = parse_positions(mh, build=BUILD)
        if not positions:
            continue
        eas_dip = extract_diplotypes_for_locus(VCF_EAS, "22", positions, eas_ids)
        ae = compute_ae(eas_dip)

        if phasing_enabled:
            trio_dip = extract_combined(sources, "22", positions, trio_ids)
            n_inf, n_bad = 0, 0
            for f, m, c in trios:
                if f not in trio_dip or m not in trio_dip or c not in trio_dip:
                    continue
                fd, md, cd = trio_dip[f], trio_dip[m], trio_dip[c]
                if any("N" in h for h in fd + md + cd):
                    continue
                if not is_informative_meiosis(cd):
                    continue
                n_inf += 1
                if not is_mendelian_consistent_diplotype(fd, md, cd):
                    n_bad += 1
            p_err = (n_bad / n_inf) if n_inf > 0 else 0.0
            results.append({
                "marker": mh["Name"], "ae_eas": round(ae, 4),
                "n_informative_meioses": n_inf, "n_mendelian_violations": n_bad,
                "p_phase_error": round(p_err, 5),
                "reliable_ae": round(reliable_ae(ae, p_err), 4),
            })
        else:
            results.append({
                "marker": mh["Name"], "ae_eas": round(ae, 4),
                "n_informative_meioses": 0, "n_mendelian_violations": None,
                "p_phase_error": None, "reliable_ae": None,
                "phasing_status": "deferred_insufficient_trios",
            })

    out = Path("results/baseline/chr22_reliable_ae.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "build": BUILD,
        "n_markers": len(results),
        "phasing_enabled": phasing_enabled,
        "n_complete_trios": len(trios),
        "phasing_note": ("Wei 2025 trio reproduction needs NYGC 30x release "
                         "(3202 samples / 602 trios, GRCh38); phase3 standard "
                         "release has 0 complete trios. Ae is EAS-derived and valid."),
        "markers": results,
    }, indent=2))
    print(f"saved {out}")

    aes = sorted((r["ae_eas"] for r in results), reverse=True)
    if aes:
        print(f"  Ae(EAS): mean={sum(aes)/len(aes):.3f}  max={aes[0]:.3f}  "
              f"top-5={[round(a,2) for a in aes[:5]]}")
        print("  → high-Ae markers are exactly those Wei 2025 flags as phasing-risky;"
              " penalty pending NYGC trios.")


if __name__ == "__main__":
    main()
