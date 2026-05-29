#!/usr/bin/env bash
# scripts/06_download_genome_wide.sh
# Genome-wide extraction of BOTH the EAS panel (504) and a fixed non-EAS OOD set
# (300 = 75 each from EUR/AFR/SAS/AMR) per chromosome, for Plan 2's open-set eval.
# Disk-efficient: download full chrN VCF -> extract both subsets -> delete full.
# Robust: per-chromosome retry; the subset operation itself is the integrity
# test (a truncated/corrupt VCF makes bcftools fail -> re-download). Subsets are
# written to .tmp then atomically moved, so a partial subset never counts as done.
set -euo pipefail

DATA="${DATA:-data/1000g}"
EAS_DIR="${EAS_DIR:-data/eas}"
OOD_DIR="${OOD_DIR:-data/ood}"
PANEL="$DATA/integrated_call_samples_v3.20130502.ALL.panel"
BASE="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
mkdir -p "$DATA" "$EAS_DIR" "$OOD_DIR"

awk '$3=="EAS"{print $1}' "$PANEL" > "$EAS_DIR/EAS_samples.txt"
if [ ! -s "$OOD_DIR/OOD_samples.txt" ]; then
    : > "$OOD_DIR/OOD_samples.txt"
    for SP in EUR AFR SAS AMR; do
        awk -v sp="$SP" '$3==sp{print $1}' "$PANEL" | sort | head -75 >> "$OOD_DIR/OOD_samples.txt"
    done
fi
echo "EAS=$(wc -l < "$EAS_DIR/EAS_samples.txt")  OOD=$(wc -l < "$OOD_DIR/OOD_samples.txt")"

process_chrom () {  # $1 = chromosome number
    local chr="$1"
    local vcf="ALL.chr${chr}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    local full="$DATA/$vcf"
    local eas_out="$EAS_DIR/EAS_chr${chr}.vcf.gz"
    local ood_out="$OOD_DIR/OOD_chr${chr}.vcf.gz"
    local attempt=0
    while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
        attempt=$((attempt + 1))
        if [ ! -f "$full" ]; then
            echo "[chr$chr] downloading (attempt $attempt) $(date +%H:%M:%S)..."
            wget -q --tries=5 --timeout=120 -c "$BASE/$vcf" -O "$full" || true
            wget -q --tries=5 --timeout=120 -c "$BASE/${vcf}.tbi" -O "${full}.tbi" || true
        fi
        # The subset reads the whole file; success == integrity verified.
        if bcftools view -S "$EAS_DIR/EAS_samples.txt" --force-samples "$full" -Oz -o "${eas_out}.tmp" 2>/dev/null \
           && bcftools view -S "$OOD_DIR/OOD_samples.txt" --force-samples "$full" -Oz -o "${ood_out}.tmp" 2>/dev/null; then
            mv "${eas_out}.tmp" "$eas_out"; bcftools index -f "$eas_out"
            mv "${ood_out}.tmp" "$ood_out"; bcftools index -f "$ood_out"
            [ "$chr" != "22" ] && rm -f "$full" "${full}.tbi"
            echo "[chr$chr] done (attempt $attempt) $(date +%H:%M:%S)"
            return 0
        fi
        echo "[chr$chr] subset failed (corrupt download?) — re-fetching"
        rm -f "$full" "${full}.tbi" "${eas_out}.tmp" "${ood_out}.tmp"
    done
    echo "[chr$chr] FAILED after $MAX_ATTEMPTS attempts"
    return 1
}

for CHR in $(seq 1 22); do
    if [ -f "$EAS_DIR/EAS_chr${CHR}.vcf.gz" ] && [ -f "$OOD_DIR/OOD_chr${CHR}.vcf.gz" ]; then
        echo "[chr$CHR] both subsets present — skip"
        continue
    fi
    process_chrom "$CHR" || { echo "ABORT at chr$CHR"; exit 1; }
done

echo "GENOME-WIDE EAS+OOD EXTRACTION COMPLETE $(date +%H:%M:%S)"
echo "EAS: $(ls "$EAS_DIR"/EAS_chr*.vcf.gz 2>/dev/null | wc -l) chroms; OOD: $(ls "$OOD_DIR"/OOD_chr*.vcf.gz 2>/dev/null | wc -l) chroms"
