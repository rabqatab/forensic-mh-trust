"""SSL pretraining pool (hg38, harmonized) — gnomAD HGDP+1KG at MH loci.

The harmonized, jointly-called HGDP + 1000G resource (~3,942 QC-passing genomes,
GRCh38): a single callset with NO cross-build issues — ideal as the foundation-
model pretraining pool AND for clean external validation (1KG + HGDP together).
Source: gnomAD phased_haplotypes_v2 (SHAPEIT5, phased). Extracts MH positions =
Positions (hg38), ALL samples; contigs renamed chrN->N to match the pipeline.
Output: data/hgdp1kg/HGDP1KG_chr{N}.vcf.gz + sample list.
"""
from __future__ import annotations

import os
import subprocess

from forensic_mh.data.markers import filter_by_chromosome, load_mh_markers, parse_positions

BASE = ("https://storage.googleapis.com/gcp-public-data--gnomad/resources/"
        "hgdp_1kg/phased_haplotypes_v2")
BCF = "hgdp1kgp_chr{c}.filtered.SNV_INDEL.phased.shapeit5.bcf"
OUT = "data/hgdp1kg"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    markers = load_mh_markers()
    # sample list once (from chr22 header)
    slist = f"{OUT}/HGDP1KG_samples.txt"
    if not os.path.exists(slist):
        try:
            s = subprocess.check_output(["bcftools", "query", "-l",
                                         f"{BASE}/{BCF.format(c=22)}"], timeout=600).decode()
            open(slist, "w").write(s)
            print(f"samples: {len(s.split())}", flush=True)
        except Exception as e:
            print(f"sample-list FAILED ({e})", flush=True)

    done = []
    for c in range(1, 23):
        mk = filter_by_chromosome(markers, f"chr{c}")
        pos = sorted({p for _, r in mk.iterrows() for p in parse_positions(r, build="hg38")})
        if not pos:
            continue
        out = f"{OUT}/HGDP1KG_chr{c}.vcf.gz"
        if os.path.exists(out):
            done.append(c); print(f"chr{c}: already present — skip", flush=True); continue
        bed = f"{OUT}/mh_chr{c}.bed"
        with open(bed, "w") as f:
            for p in pos:
                f.write(f"chr{c}\t{p-1}\t{p}\n")        # gnomAD GRCh38 uses 'chr1'..'chr22'
        rn = f"{OUT}/_rn{c}.txt"
        open(rn, "w").write(f"chr{c}\t{c}\n")
        tmp = f"{OUT}/_tmp{c}.vcf.gz"
        url = f"{BASE}/{BCF.format(c=c)}"
        try:
            subprocess.run(["bcftools", "view", "-R", bed, url, "-Oz", "-o", tmp],
                           check=True, timeout=3000)
            subprocess.run(["bcftools", "annotate", "--rename-chrs", rn, tmp, "-Oz", "-o", out],
                           check=True, timeout=300)
            subprocess.run(["bcftools", "index", "-f", out], check=True)
            os.remove(tmp); os.remove(rn)
            n = int(subprocess.check_output(["bcftools", "index", "-n", out]).decode())
            done.append(c)
            print(f"chr{c}: {n} MH-locus records extracted", flush=True)
        except Exception as e:
            print(f"chr{c}: FAILED ({e})", flush=True)
    print(f"DONE chroms={done}", flush=True)
    print("HGDP1KG_DONE", flush=True)


if __name__ == "__main__":
    main()
