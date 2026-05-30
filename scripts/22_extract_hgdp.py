"""External cohort: remote-extract HGDP EAS (Han/Japanese/Dai) at MH loci (GRCh38).

Sanger HGDP WGS VCFs are GRCh38, tabix-indexed → fetch ONLY MH positions via
remote bcftools (no full download). Contigs renamed chrN->N to match the
existing pipeline; positions are hg38 (MicroHapDB `Positions`). Output:
data/hgdp/HGDP_chr{N}.vcf.gz + sample/pop lists.
"""
from __future__ import annotations

import csv
import os
import subprocess
import urllib.request

from forensic_mh.data.markers import filter_by_chromosome, load_mh_markers, parse_positions

SANGER = ("https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/"
          "hgdp_wgs.20190516.full.chr{c}.vcf.gz")
META_URL = ("https://storage.googleapis.com/gcp-public-data--gnomad/release/3.1/"
            "secondary_analyses/hgdp_1kg/metadata_and_qc/gnomad_meta_v1.tsv")
CLASSES = {"Han", "Japanese", "Dai"}   # map to 1000G CHB+CHS / JPT / CDX
OUT = "data/hgdp"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    meta = f"{OUT}/gnomad_meta_v1.tsv"
    if not os.path.exists(meta):
        urllib.request.urlretrieve(META_URL, meta)
    rows = list(csv.DictReader(open(meta), delimiter="\t"))
    want = {r["s"]: r["hgdp_tgp_meta.Population"] for r in rows
            if r["s"].startswith("HGDP") and r["hgdp_tgp_meta.Population"] in CLASSES}
    vcf_samples = set(subprocess.check_output(
        ["bcftools", "query", "-l", SANGER.format(c=22)]).decode().split())
    keep = [s for s in want if s in vcf_samples]
    open(f"{OUT}/HGDP_eas_samples.txt", "w").write("\n".join(keep) + "\n")
    with open(f"{OUT}/HGDP_eas_pop.tsv", "w") as f:
        for s in keep:
            f.write(f"{s}\t{want[s]}\n")
    import collections
    print("HGDP EAS kept:", len(keep), dict(collections.Counter(want[s] for s in keep)), flush=True)

    markers = load_mh_markers()
    done = []
    for c in range(1, 23):
        mk = filter_by_chromosome(markers, f"chr{c}")
        pos = sorted({p for _, r in mk.iterrows() for p in parse_positions(r, build="hg38")})
        if not pos:
            continue
        bed = f"{OUT}/mh_chr{c}.bed"
        with open(bed, "w") as f:
            for p in pos:
                f.write(f"chr{c}\t{p-1}\t{p}\n")
        out = f"{OUT}/HGDP_chr{c}.vcf.gz"
        if os.path.exists(out):          # resumable: skip already-extracted chroms
            done.append(c); print(f"chr{c}: already present — skip", flush=True); continue
        rn = f"{OUT}/_rn{c}.txt"
        open(rn, "w").write(f"chr{c}\t{c}\n")
        tmp = f"{OUT}/_tmp{c}.vcf.gz"
        try:
            subprocess.run(["bcftools", "view", "-R", bed, "-S", f"{OUT}/HGDP_eas_samples.txt",
                            "--force-samples", SANGER.format(c=c), "-Oz", "-o", tmp],
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


if __name__ == "__main__":
    main()
