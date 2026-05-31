#!/usr/bin/env bash
# scripts/01_download_1000g.sh
# Plan 1 Task 3: 1000G panel + chr22 VCF (pipeline 검증용)
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/1000g}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

BASE="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"

# 1. Panel file (작음, 즉시)
if [ ! -f "integrated_call_samples_v3.20130502.ALL.panel" ]; then
    echo "[1/2] Downloading panel..."
    wget -q --show-progress "$BASE/integrated_call_samples_v3.20130502.ALL.panel"
fi

# 2. chr22 VCF (pipeline test용 — ~200MB)
CHR=${CHR:-22}
VCF="ALL.chr${CHR}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
if [ ! -f "$VCF" ]; then
    echo "[2/2] Downloading chr${CHR} VCF..."
    wget -q --show-progress "$BASE/$VCF"
    wget -q --show-progress "$BASE/${VCF}.tbi"
fi

echo "Done. Files:"
ls -lh
