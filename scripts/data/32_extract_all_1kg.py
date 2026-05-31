"""SSL pretraining pool (hg19) — extract ALL 2,504 1000G individuals at MH loci.

Label-free pretraining pool for the foundation model (superset of EAS-504 +
OOD-300). Same 1000G Phase 3 v5b VCFs (GRCh37), MH positions = Positions37 (hg19),
NO sample filter → all 2,504. Bare contig names ('1'..'22') match the existing
EAS/OOD subsets, so no rename. Output: data/all1kg/ALL_chr{N}.vcf.gz.
"""
from __future__ import annotations

import os
import subprocess

from forensic_mh.data.markers import filter_by_chromosome, load_mh_markers, parse_positions

BASE = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
VCF = "ALL.chr{c}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
OUT = "data/all1kg"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    markers = load_mh_markers()
    done = []
    for c in range(1, 23):
        mk = filter_by_chromosome(markers, f"chr{c}")
        pos = sorted({p for _, r in mk.iterrows() for p in parse_positions(r, build="hg19")})
        if not pos:
            continue
        out = f"{OUT}/ALL_chr{c}.vcf.gz"
        if os.path.exists(out):
            done.append(c); print(f"chr{c}: already present — skip", flush=True); continue
        bed = f"{OUT}/mh_chr{c}.bed"
        with open(bed, "w") as f:
            for p in pos:
                f.write(f"{c}\t{p-1}\t{p}\n")          # bare contig (1000G uses '1'..'22')
        tmp = f"{OUT}/_tmp{c}.vcf.gz"
        url = f"{BASE}/{VCF.format(c=c)}"
        try:
            subprocess.run(["bcftools", "view", "-R", bed, url, "-Oz", "-o", tmp],
                           check=True, timeout=3000)
            os.replace(tmp, out)
            subprocess.run(["bcftools", "index", "-f", out], check=True)
            n = int(subprocess.check_output(["bcftools", "index", "-n", out]).decode())
            done.append(c)
            print(f"chr{c}: {n} MH-locus records extracted", flush=True)
        except Exception as e:
            print(f"chr{c}: FAILED ({e})", flush=True)
    print(f"DONE chroms={done}", flush=True)
    print("ALL1KG_DONE", flush=True)


if __name__ == "__main__":
    main()
