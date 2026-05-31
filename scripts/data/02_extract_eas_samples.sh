#!/usr/bin/env bash
# scripts/02_extract_eas_samples.sh
# Plan 1 Task 4: 1000G ALL → EAS 504 subset (chr22 기준)
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/1000g}"
OUT_DIR="${OUT_DIR:-data/eas}"
PANEL="$DATA_DIR/integrated_call_samples_v3.20130502.ALL.panel"
CHR=${CHR:-22}

mkdir -p "$OUT_DIR"

# 1. EAS 샘플 ID 추출
awk '$3=="EAS" {print $1}' "$PANEL" > "$OUT_DIR/EAS_samples.txt"
N=$(wc -l < "$OUT_DIR/EAS_samples.txt")
echo "EAS samples: $N (expected 504)"
[ "$N" -eq 504 ] || { echo "ERROR: expected 504 EAS samples, got $N"; exit 1; }

# 2. VCF subset (chr22는 v5b)
SRC="$DATA_DIR/ALL.chr${CHR}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
OUT="$OUT_DIR/EAS_chr${CHR}.vcf.gz"
bcftools view -S "$OUT_DIR/EAS_samples.txt" "$SRC" -Oz -o "$OUT"
bcftools index "$OUT"

echo "Output: $OUT"
ls -lh "$OUT"*
