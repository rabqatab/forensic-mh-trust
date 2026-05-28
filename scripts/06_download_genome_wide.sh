#!/usr/bin/env bash
# scripts/06_download_genome_wide.sh
# Genome-wide EAS extraction (chr1-21; chr22 already done).
# Disk-efficient: download full chrN VCF → subset to EAS 504 → delete full VCF.
# Peak disk ≈ one chromosome (~1.2GB) + accumulating EAS subsets.
set -euo pipefail

DATA="${DATA:-data/1000g}"
OUT="${OUT:-data/eas}"
PANEL="$DATA/integrated_call_samples_v3.20130502.ALL.panel"
BASE="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
mkdir -p "$DATA" "$OUT"

awk '$3=="EAS"{print $1}' "$PANEL" > "$OUT/EAS_samples.txt"
N=$(wc -l < "$OUT/EAS_samples.txt")
[ "$N" -eq 504 ] || { echo "ERROR: expected 504 EAS, got $N"; exit 1; }

for CHR in $(seq 1 21); do
    EAS_OUT="$OUT/EAS_chr${CHR}.vcf.gz"
    if [ -f "$EAS_OUT" ]; then
        echo "[chr$CHR] already present — skip"
        continue
    fi
    VCF="ALL.chr${CHR}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    echo "[chr$CHR] downloading $(date +%H:%M:%S)..."
    wget -q "$BASE/$VCF" -O "$DATA/$VCF"
    wget -q "$BASE/${VCF}.tbi" -O "$DATA/${VCF}.tbi"
    echo "[chr$CHR] subsetting EAS 504..."
    bcftools view -S "$OUT/EAS_samples.txt" "$DATA/$VCF" -Oz -o "$EAS_OUT"
    bcftools index "$EAS_OUT"
    rm -f "$DATA/$VCF" "$DATA/${VCF}.tbi"
    echo "[chr$CHR] done — kept $(ls -lh "$EAS_OUT" | awk '{print $5}'), removed full VCF $(date +%H:%M:%S)"
done

echo "GENOME-WIDE EAS EXTRACTION COMPLETE $(date +%H:%M:%S)"
ls -lh "$OUT"/EAS_chr*.vcf.gz
