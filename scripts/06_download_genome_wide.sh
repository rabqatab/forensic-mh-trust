#!/usr/bin/env bash
# scripts/06_download_genome_wide.sh
# Genome-wide extraction of BOTH the EAS panel (504) and a fixed non-EAS OOD set
# (300 = 75 each from EUR/AFR/SAS/AMR) per chromosome, for Plan 2's open-set eval.
# Disk-efficient: download full chrN VCF → extract both subsets → delete full VCF.
# Skip-aware: a chromosome with both subsets already present is skipped.
set -euo pipefail

DATA="${DATA:-data/1000g}"
EAS_DIR="${EAS_DIR:-data/eas}"
OOD_DIR="${OOD_DIR:-data/ood}"
PANEL="$DATA/integrated_call_samples_v3.20130502.ALL.panel"
BASE="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
mkdir -p "$DATA" "$EAS_DIR" "$OOD_DIR"

# EAS 504
awk '$3=="EAS"{print $1}' "$PANEL" > "$EAS_DIR/EAS_samples.txt"
# OOD 300: deterministic, balanced across the four non-EAS superpopulations
if [ ! -s "$OOD_DIR/OOD_samples.txt" ]; then
    : > "$OOD_DIR/OOD_samples.txt"
    for SP in EUR AFR SAS AMR; do
        awk -v sp="$SP" '$3==sp{print $1}' "$PANEL" | sort | head -75 >> "$OOD_DIR/OOD_samples.txt"
    done
fi
echo "EAS=$(wc -l < "$EAS_DIR/EAS_samples.txt")  OOD=$(wc -l < "$OOD_DIR/OOD_samples.txt")"

subset () {  # $1=full vcf  $2=samples.txt  $3=out.vcf.gz
    bcftools view -S "$2" --force-samples "$1" -Oz -o "$3"
    bcftools index -f "$3"
}

for CHR in $(seq 1 22); do
    EAS_OUT="$EAS_DIR/EAS_chr${CHR}.vcf.gz"
    OOD_OUT="$OOD_DIR/OOD_chr${CHR}.vcf.gz"
    if [ -f "$EAS_OUT" ] && [ -f "$OOD_OUT" ]; then
        echo "[chr$CHR] both subsets present — skip"
        continue
    fi
    VCF="ALL.chr${CHR}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    FULL="$DATA/$VCF"
    if [ ! -f "$FULL" ]; then
        echo "[chr$CHR] downloading $(date +%H:%M:%S)..."
        wget -q "$BASE/$VCF" -O "$FULL"
        wget -q "$BASE/${VCF}.tbi" -O "${FULL}.tbi"
    fi
    [ -f "$EAS_OUT" ] || { echo "[chr$CHR] EAS subset..."; subset "$FULL" "$EAS_DIR/EAS_samples.txt" "$EAS_OUT"; }
    [ -f "$OOD_OUT" ] || { echo "[chr$CHR] OOD subset..."; subset "$FULL" "$OOD_DIR/OOD_samples.txt" "$OOD_OUT"; }
    # keep the chr22 full VCF (used by trio/related-sample work); delete others
    if [ "$CHR" != "22" ]; then
        rm -f "$FULL" "${FULL}.tbi"
    fi
    echo "[chr$CHR] done $(date +%H:%M:%S)"
done

echo "GENOME-WIDE EAS+OOD EXTRACTION COMPLETE $(date +%H:%M:%S)"
echo "EAS: $(ls "$EAS_DIR"/EAS_chr*.vcf.gz 2>/dev/null | wc -l) chroms; OOD: $(ls "$OOD_DIR"/OOD_chr*.vcf.gz 2>/dev/null | wc -l) chroms"
