#!/usr/bin/env bash
# scripts/05_download_related_samples.sh
# 1000G phase3 related-samples chr22 VCF (the 698 related individuals).
# The standard 2504-sample release is the UNRELATED subset → 0 complete trios.
# These related samples complete the trios, enabling Wei 2025 phasing reproduction.
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/1000g}"
CHR="${CHR:-22}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

BASE="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/related_samples_vcf"
VCF="ALL.chr${CHR}.phase3_shapeit2_mvncall_integrated_v5_related_samples.20130502.genotypes.vcf.gz"

if [ ! -f "$VCF" ]; then
    echo "[1/1] Downloading related-samples chr${CHR} VCF..."
    wget -q --show-progress "$BASE/$VCF"
    wget -q --show-progress "$BASE/${VCF}.tbi"
fi

echo "Done:"
ls -lh "$VCF"*
echo "samples: $(bcftools query -l "$VCF" | wc -l) (expected ~698 related individuals)"
