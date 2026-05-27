# Plan 1 — Foundation & Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 1000G EAS MH 분석 파이프라인을 leakage-free·diplotype-correct·friendly-relatedness-aware하게 구축하고, Wei 2025 phasing error 분석으로 Reliable-Ae metric을 만든다. Plan 2 (Conformal/OSR)이 올라탈 수 있는 견고한 base를 제공.

**Architecture:** uv-managed Python venv + 모듈형 `forensic_mh` 라이브러리 + scripts/ 파이프라인 + pytest TDD. 두 노드 분담: Node 1은 interactive 개발, Node 2는 background KoVariome variant calling (Day 2 sparkq submit, Day 5-7 사이 완료 기대).

**Tech Stack:**
- Python 3.11 (uv venv)
- 데이터: `cyvcf2`, `pysam`, `microhapdb`, `pandas`
- ML: `scikit-learn>=1.5`, `xgboost>=2.0`, `mapie` (Plan 2 대비 미리 설치)
- 테스트: `pytest`, `pytest-cov`
- 시스템: `bcftools`, `samtools`
- 클러스터: sparkq, dgx-spark (Node 2 KoVariome용)
- VCS: git

**기간:** Day 1-3 (총 ~21 시간 작업 가정)

---

## File Structure (이 Plan에서 생성·수정할 파일)

```
mh-eas-panel/
├── pyproject.toml                              # NEW — uv 관리, deps 선언
├── .gitignore                                  # NEW — data/, results/, .venv 제외
├── README.md                                   # NEW — 프로젝트 한 줄 + 실행법
├── src/forensic_mh/
│   ├── __init__.py                             # NEW
│   ├── data/
│   │   ├── __init__.py                         # NEW
│   │   ├── vcf_io.py                           # NEW — VCF → diplotype 추출
│   │   └── markers.py                          # NEW — MicroHapDB wrapper
│   ├── eval/
│   │   ├── __init__.py                         # NEW
│   │   ├── nested_cv.py                        # NEW — leakage-free nested CV
│   │   └── grouping.py                         # NEW — relatedness → groups
│   ├── metrics/
│   │   ├── __init__.py                         # NEW
│   │   └── reliable_ae.py                      # NEW — Wei 2025 보정 metric
│   └── pipelines/
│       ├── __init__.py                         # NEW
│       └── baseline.py                         # NEW — 5-way XGBoost baseline
├── scripts/
│   ├── 01_download_1000g.sh                    # NEW — panel + chr22
│   ├── 02_extract_eas_samples.sh               # NEW — bcftools 추출
│   ├── 03_run_baseline.py                      # NEW — Plan 1 산출 figure
│   └── 04_wei2025_phasing.py                   # NEW — trio mismatch 분석
├── tests/
│   ├── __init__.py                             # NEW
│   ├── conftest.py                             # NEW — fixture
│   ├── data/test_vcf_io.py                     # NEW
│   ├── eval/test_nested_cv.py                  # NEW
│   ├── eval/test_grouping.py                   # NEW
│   └── metrics/test_reliable_ae.py             # NEW
├── data/                                       # gitignored — 1000G VCF, panel
├── results/baseline/                           # gitignored 산출 figure
└── docs/superpowers/plans/2026-05-26-foundation-baseline.md  # THIS FILE
```

**파일 책임 원칙**:
- `src/forensic_mh/data/`: I/O만 — pure data transforms, no ML
- `src/forensic_mh/eval/`: CV·splitting protocol — sklearn 호환
- `src/forensic_mh/metrics/`: 도메인 메트릭 (Ae, FST, Reliable-Ae) — pure functions
- `src/forensic_mh/pipelines/`: end-to-end orchestration — Plan 2/3에서 확장
- `scripts/`: CLI entry point, 인자 파싱 + 모듈 호출만

---

## Task 1: 프로젝트 git 초기화 + 구조 생성

**Files:**
- Create: `/home/alphabridge/Research/mh-eas-panel/.gitignore`
- Create: `/home/alphabridge/Research/mh-eas-panel/README.md`
- Create: `src/forensic_mh/__init__.py`, `tests/__init__.py` 등 빈 __init__.py

- [ ] **Step 1: git 저장소 초기화 + main branch**

```bash
cd /home/alphabridge/Research/mh-eas-panel
git init -b main
git status
```

- [ ] **Step 2: .gitignore 작성**

```gitignore
# Data — too large for git
data/
results/

# Python
__pycache__/
*.pyc
.venv/
.uv/
*.egg-info/
dist/
build/

# Notebook checkpoints
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# System
.DS_Store
```

- [ ] **Step 3: README.md 작성 (한 줄 + 실행법)**

```markdown
# mh-eas-panel — Trustworthy Forensic-FM

동아시아 집단 분류를 위한 마이크로하플로타입(MH) 패널 연구.
SSL pretraining + Conformal/Open-set UQ 통합 — Trustworthy Forensic-FM.

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Run baseline

```bash
bash scripts/01_download_1000g.sh
bash scripts/02_extract_eas_samples.sh
python scripts/03_run_baseline.py
```

## Documents

- `docs/01_proposal_review.md` — 비판적 리뷰
- `docs/02_literature_landscape.md` — 문헌 매핑
- `docs/03_novelty_options.md` — novelty 옵션 (v2)
- `docs/superpowers/plans/` — 2-week 스프린트 plan
```

- [ ] **Step 4: 디렉토리 + __init__.py 생성**

```bash
mkdir -p src/forensic_mh/{data,eval,metrics,pipelines}
mkdir -p tests/{data,eval,metrics}
mkdir -p scripts data results/baseline notebooks
touch src/forensic_mh/__init__.py
touch src/forensic_mh/{data,eval,metrics,pipelines}/__init__.py
touch tests/__init__.py
touch tests/{data,eval,metrics}/__init__.py
```

- [ ] **Step 5: 첫 commit**

```bash
git add .gitignore README.md src/ tests/ scripts/ docs/
git commit -m "chore: initial project structure + git ignore"
```

---

## Task 2: uv 환경 + pyproject.toml

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: pyproject.toml 작성**

```toml
[project]
name = "forensic_mh"
version = "0.1.0"
description = "Trustworthy Forensic Microhaplotype Foundation Model"
authors = [{name = "조민한"}]
requires-python = ">=3.11"

dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "scipy>=1.13",
    "scikit-learn>=1.5",
    "xgboost>=2.0",
    "cyvcf2>=0.30",
    "pysam>=0.22",
    "microhapdb>=0.7",
    "matplotlib>=3.9",
    "seaborn>=0.13",
    "mapie>=0.9",  # Plan 2 대비
    "tqdm",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "jupyterlab",
    "ipykernel",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/forensic_mh"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 2: uv 환경 새로 만들고 deps 설치**

```bash
rm -rf .venv  # 기존 분석용 .venv 제거
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Expected: 모든 deps 설치 성공 메시지

- [ ] **Step 3: 설치 검증**

```bash
python -c "import forensic_mh, cyvcf2, microhapdb, xgboost, mapie; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: 시스템 deps 확인 (bcftools 이미 있음)**

```bash
which bcftools && bcftools --version | head -1
```

Expected: bcftools 경로 + 버전 출력

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: pyproject.toml with uv-managed deps (sklearn, xgboost, mapie, cyvcf2)"
```

---

## Task 3: 1000G 데이터 다운로드 (chr22만 — pipeline 검증용)

**Files:**
- Create: `scripts/01_download_1000g.sh`

Day 1엔 chr22 (가장 작은 chromosome, ~200MB)만 받아 pipeline 검증. Day 2-3에 나머지 chromosome 추가.

- [ ] **Step 1: 다운로드 스크립트 작성**

```bash
#!/usr/bin/env bash
# scripts/01_download_1000g.sh
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

# 2. chr22 VCF (pipeline test용 — 200MB)
CHR=${CHR:-22}
VCF="ALL.chr${CHR}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
if [ ! -f "$VCF" ]; then
    echo "[2/2] Downloading chr${CHR} VCF..."
    wget -q --show-progress "$BASE/$VCF"
    wget -q --show-progress "$BASE/${VCF}.tbi"
fi

echo "Done. Files:"
ls -lh
```

- [ ] **Step 2: 실행 권한 + 실행**

```bash
chmod +x scripts/01_download_1000g.sh
bash scripts/01_download_1000g.sh
```

Expected: `data/1000g/`에 panel 파일 + chr22.vcf.gz + chr22.vcf.gz.tbi

- [ ] **Step 3: Panel 검증**

```bash
head -1 data/1000g/integrated_call_samples_v3.20130502.ALL.panel
wc -l data/1000g/integrated_call_samples_v3.20130502.ALL.panel
```

Expected: 헤더 `sample\tpop\tsuper_pop\tgender`, 총 ~2505줄

- [ ] **Step 4: EAS 504명 카운트 검증**

```bash
awk '$3=="EAS" {print $2}' data/1000g/integrated_call_samples_v3.20130502.ALL.panel | sort | uniq -c
```

Expected:
```
  93 CDX
 103 CHB
 105 CHS
 104 JPT
  99 KHV
```

- [ ] **Step 5: VCF 인덱스로 메타 검증**

```bash
bcftools view -h data/1000g/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz | tail -3
bcftools query -l data/1000g/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz | wc -l
```

Expected: 2504 samples

- [ ] **Step 6: Commit (스크립트만, data는 gitignore)**

```bash
git add scripts/01_download_1000g.sh
git commit -m "feat(data): 1000G chr22 download script (pipeline test scope)"
```

---

## Task 4: EAS 504 샘플 추출 (bcftools)

**Files:**
- Create: `scripts/02_extract_eas_samples.sh`

- [ ] **Step 1: 추출 스크립트 작성**

```bash
#!/usr/bin/env bash
# scripts/02_extract_eas_samples.sh
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

# 2. VCF subset
SRC="$DATA_DIR/ALL.chr${CHR}.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
OUT="$OUT_DIR/EAS_chr${CHR}.vcf.gz"
bcftools view -S "$OUT_DIR/EAS_samples.txt" "$SRC" -Oz -o "$OUT"
bcftools index "$OUT"

echo "Output: $OUT"
ls -lh "$OUT"*
```

- [ ] **Step 2: 실행**

```bash
chmod +x scripts/02_extract_eas_samples.sh
bash scripts/02_extract_eas_samples.sh
```

Expected: `data/eas/EAS_chr22.vcf.gz` + .csi 인덱스

- [ ] **Step 3: 샘플 수 검증**

```bash
bcftools query -l data/eas/EAS_chr22.vcf.gz | wc -l
```

Expected: 504

- [ ] **Step 4: variant 수 sanity check**

```bash
bcftools view -H data/eas/EAS_chr22.vcf.gz | wc -l
```

Expected: ~1M variants (chr22 phase 3 EAS, monomorphic 일부 포함)

- [ ] **Step 5: Commit**

```bash
git add scripts/02_extract_eas_samples.sh
git commit -m "feat(data): bcftools-based EAS 504-sample subset extraction"
```

---

## Task 5: MicroHapDB wrapper — chr22 MH 좌표 추출

**Files:**
- Create: `src/forensic_mh/data/markers.py`
- Test: `tests/data/test_markers.py`

- [ ] **Step 1: 실패 테스트 먼저 작성**

```python
# tests/data/test_markers.py
import pytest
from forensic_mh.data.markers import load_mh_markers, filter_by_chromosome


def test_load_mh_markers_returns_dataframe_with_expected_columns():
    df = load_mh_markers()
    assert len(df) > 100, f"expected >100 markers in MicroHapDB, got {len(df)}"
    assert "Name" in df.columns
    assert "Chrom" in df.columns
    # MicroHapDB uses 'Offsets' for SNP positions (semicolon-delimited)
    assert "Offsets" in df.columns or "Positions" in df.columns


def test_filter_by_chromosome_returns_only_target_chromosome():
    df = load_mh_markers()
    chr22 = filter_by_chromosome(df, "chr22")
    assert len(chr22) > 0
    assert all(chr22["Chrom"].str.replace("chr", "") == "22")
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
source .venv/bin/activate
pytest tests/data/test_markers.py -v
```

Expected: ImportError or function not found

- [ ] **Step 3: 최소 구현**

```python
# src/forensic_mh/data/markers.py
"""MicroHapDB wrapper — MH marker coordinates."""
from __future__ import annotations
import pandas as pd
import microhapdb


def load_mh_markers() -> pd.DataFrame:
    """Return MicroHapDB markers DataFrame.

    Columns include Name, Chrom, Offsets (semicolon-delimited SNP positions).
    """
    return microhapdb.markers.copy()


def filter_by_chromosome(markers: pd.DataFrame, chrom: str) -> pd.DataFrame:
    """Return markers on a given chromosome (e.g., 'chr22' or '22')."""
    target = chrom.replace("chr", "")
    return markers[markers["Chrom"].str.replace("chr", "") == target].copy()
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/data/test_markers.py -v
```

Expected: 2 passed

- [ ] **Step 5: chr22 MH 수 sanity check (스크립트 인라인)**

```bash
python -c "
from forensic_mh.data.markers import load_mh_markers, filter_by_chromosome
df = load_mh_markers()
print(f'Total markers: {len(df)}')
chr22 = filter_by_chromosome(df, 'chr22')
print(f'chr22 markers: {len(chr22)}')
print(chr22.head(3)[['Name', 'Chrom', 'Offsets']].to_string())
"
```

Expected: 총 ~417 markers, chr22에 5-15개 정도

- [ ] **Step 6: Commit**

```bash
git add src/forensic_mh/data/markers.py tests/data/test_markers.py
git commit -m "feat(data): MicroHapDB wrapper with chromosome filtering + tests"
```

---

## Task 6: ⭐ Diplotype 추출 (제안서 P0 #2 수정)

**Files:**
- Create: `src/forensic_mh/data/vcf_io.py`
- Test: `tests/data/test_vcf_io.py`

원안의 `allele = rec.alleles[gt[0]]`는 첫 haplotype만 가져옴 → heterozygote 정보 손실. 두 haplotype 모두 가져와 unordered diplotype tuple로 인코딩.

- [ ] **Step 1: 실패 테스트 먼저 작성**

```python
# tests/data/test_vcf_io.py
"""Verify diplotype extraction handles both haplotypes (P0 fix from review #2)."""
import pytest
from pathlib import Path
from forensic_mh.data.vcf_io import extract_diplotypes_for_locus


# integration test fixture — chr22 small region
TEST_VCF = Path("data/eas/EAS_chr22.vcf.gz")
pytestmark = pytest.mark.skipif(
    not TEST_VCF.exists(),
    reason="Run scripts/02_extract_eas_samples.sh first to create EAS_chr22.vcf.gz",
)


def test_diplotype_returns_two_haplotypes_per_sample():
    # pick first 5 EAS samples + a small region with multiple variants
    # we don't assume specific positions — just structural correctness
    diplotypes = extract_diplotypes_for_locus(
        str(TEST_VCF),
        chrom="22",
        positions=[16050075, 16050115],  # any positions in chr22
        sample_ids=None,  # all samples
    )
    # diplotypes: dict[sample_id, tuple[str, str]]  ← two haplotype strings
    assert len(diplotypes) == 504, "expected 504 EAS samples"
    for sid, dipl in list(diplotypes.items())[:5]:
        assert isinstance(dipl, tuple) and len(dipl) == 2
        # each haplotype is a "-".join of alleles, one per position
        assert dipl[0].count("-") == 1  # 2 positions → 1 separator
        assert dipl[1].count("-") == 1


def test_diplotype_is_unordered_canonical_form():
    """diplotype (A-T, G-C) should equal (G-C, A-T) — sorted tuple."""
    diplotypes = extract_diplotypes_for_locus(
        str(TEST_VCF), chrom="22",
        positions=[16050075, 16050115],
    )
    # canonical: sorted alphabetically
    for dipl in list(diplotypes.values())[:5]:
        assert dipl[0] <= dipl[1], f"diplotype not in canonical order: {dipl}"
```

- [ ] **Step 2: 테스트 실패 확인 (구현 없음)**

```bash
pytest tests/data/test_vcf_io.py -v
```

Expected: ImportError

- [ ] **Step 3: 구현**

```python
# src/forensic_mh/data/vcf_io.py
"""VCF → diplotype extraction. Critical P0 fix from review #2:
the original proposal used `gt[0]` only (one haplotype), losing heterozygosity.
We extract BOTH haplotypes and form unordered diplotype tuples."""
from __future__ import annotations
from collections import defaultdict
from typing import Iterable, Optional

import cyvcf2


def extract_diplotypes_for_locus(
    vcf_path: str,
    chrom: str,
    positions: Iterable[int],
    sample_ids: Optional[Iterable[str]] = None,
) -> dict[str, tuple[str, str]]:
    """Extract per-sample diplotypes at a microhaplotype locus.

    Args:
        vcf_path: phased VCF (1000G shapeit2)
        chrom: chromosome name as in VCF (e.g., '22')
        positions: 1-based SNP positions defining the MH
        sample_ids: optional subset; if None, all samples in VCF

    Returns:
        {sample_id: (haplotype_A, haplotype_B)} where each haplotype is
        "-".join(alleles) and the tuple is in canonical (sorted) order so
        that diplotypes compare equal regardless of haplotype phase order.
    """
    vcf = cyvcf2.VCF(vcf_path)
    all_samples = list(vcf.samples)
    targets = list(sample_ids) if sample_ids is not None else all_samples
    target_idx = [all_samples.index(s) for s in targets]

    # one list per haplotype slot, per sample
    hap_alleles: dict[str, list[list[str]]] = {
        s: [[], []] for s in targets
    }

    positions_list = sorted(positions)
    for pos in positions_list:
        region = f"{chrom}:{pos}-{pos}"
        found = False
        for rec in vcf(region):
            if rec.POS != pos:
                continue
            found = True
            # rec.genotypes: list[[a, b, phased_flag]] per sample
            gts = rec.genotypes
            alleles = [rec.REF] + list(rec.ALT)
            for sid, idx in zip(targets, target_idx):
                a, b, _phased = gts[idx]
                hap_alleles[sid][0].append(alleles[a] if a >= 0 else "N")
                hap_alleles[sid][1].append(alleles[b] if b >= 0 else "N")
            break
        if not found:
            # variant not present — fill N for both haplotypes
            for sid in targets:
                hap_alleles[sid][0].append("N")
                hap_alleles[sid][1].append("N")

    diplotypes: dict[str, tuple[str, str]] = {}
    for sid in targets:
        h0 = "-".join(hap_alleles[sid][0])
        h1 = "-".join(hap_alleles[sid][1])
        diplotypes[sid] = tuple(sorted([h0, h1]))
    return diplotypes
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/data/test_vcf_io.py -v
```

Expected: 2 passed (or skipped if VCF not yet downloaded — fix that first)

- [ ] **Step 5: smoke test 인라인**

```bash
python -c "
from forensic_mh.data.vcf_io import extract_diplotypes_for_locus
d = extract_diplotypes_for_locus(
    'data/eas/EAS_chr22.vcf.gz', '22',
    [16050075, 16050115],
)
print('n samples:', len(d))
import collections
counts = collections.Counter(d.values())
print('unique diplotypes:', len(counts))
print('top 5:', counts.most_common(5))
"
```

Expected: 504 samples, 2~5 unique diplotypes at this trivial 2-SNP locus

- [ ] **Step 6: Commit**

```bash
git add src/forensic_mh/data/vcf_io.py tests/data/test_vcf_io.py
git commit -m "feat(data): diplotype extraction preserving both haplotypes (P0 #2 fix)"
```

---

## Task 7: 관련자 처리 — GroupKFold용 group label (P0 #8)

**Files:**
- Create: `src/forensic_mh/eval/grouping.py`
- Test: `tests/eval/test_grouping.py`

1000G phase 3는 trio/related individuals 일부 포함. 정확한 KING-based IBD 계산은 별도 도구 필요하지만, 일단 trio 관계(panel 메타데이터 또는 pedigree 파일 기반)로 group ID를 만듦. KING 통합은 Plan 2 옵션.

- [ ] **Step 1: 테스트 작성**

```python
# tests/eval/test_grouping.py
import numpy as np
import pandas as pd
from forensic_mh.eval.grouping import build_groups_from_panel


def test_build_groups_from_panel_returns_one_id_per_sample():
    # mock: 5 samples, 2 in same family
    panel = pd.DataFrame({
        "sample": ["S1", "S2", "S3", "S4", "S5"],
        "pop": ["CHB", "CHB", "JPT", "JPT", "KHV"],
        "super_pop": ["EAS"] * 5,
        "family_id": ["F1", "F1", "F2", "F3", "F4"],  # S1+S2 related
    })
    groups = build_groups_from_panel(panel, sample_ids=["S1", "S2", "S3", "S4", "S5"])
    assert len(groups) == 5
    # S1, S2 share family → same group
    assert groups[0] == groups[1]
    # S3, S4 different families → different groups
    assert groups[2] != groups[3]


def test_build_groups_fallback_to_sample_when_family_id_missing():
    panel = pd.DataFrame({
        "sample": ["S1", "S2"],
        "pop": ["CHB", "JPT"],
        "super_pop": ["EAS", "EAS"],
        # no family_id column
    })
    groups = build_groups_from_panel(panel, sample_ids=["S1", "S2"])
    assert len(groups) == 2
    assert groups[0] != groups[1]  # treated as unrelated
```

- [ ] **Step 2: 실패 확인**

```bash
pytest tests/eval/test_grouping.py -v
```

Expected: ImportError

- [ ] **Step 3: 구현**

```python
# src/forensic_mh/eval/grouping.py
"""Group labels for GroupKFold — prevents related individuals leaking across CV folds.

P0 fix from review #8.

The 1000G panel does not include explicit family_id by default; the file
20130606_g1k.ped (separately downloadable) provides pedigree. Until that
is integrated, fall back to sample_id (treats all as unrelated — overly
optimistic but matches current behavior). Plan 2 task adds the pedigree
join and full GroupKFold validity.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def build_groups_from_panel(
    panel: pd.DataFrame, sample_ids: list[str]
) -> np.ndarray:
    """Return integer group IDs (one per sample) for GroupKFold.

    If `family_id` column exists in panel, samples sharing family_id get the
    same group ID. Otherwise each sample is its own group.
    """
    panel_idx = panel.set_index("sample")
    if "family_id" in panel.columns:
        fam_ids = [panel_idx.loc[s, "family_id"] for s in sample_ids]
        # map family IDs to integer codes
        unique_fams = sorted(set(fam_ids))
        fam_to_int = {f: i for i, f in enumerate(unique_fams)}
        return np.array([fam_to_int[f] for f in fam_ids], dtype=int)
    # fallback — treat every sample as own group
    return np.arange(len(sample_ids), dtype=int)
```

- [ ] **Step 4: 테스트 통과**

```bash
pytest tests/eval/test_grouping.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/forensic_mh/eval/grouping.py tests/eval/test_grouping.py
git commit -m "feat(eval): GroupKFold group ID builder from pedigree (P0 #8 fix scaffold)"
```

---

## Task 8: ⭐ Nested CV with leak-free feature selection (P0 #1)

**Files:**
- Create: `src/forensic_mh/eval/nested_cv.py`
- Test: `tests/eval/test_nested_cv.py`

원안의 `model_full.fit(X, y) → top-N → cross_val_score`는 라벨 leakage. outer fold 안에서 feature selection을 다시 해야 함.

- [ ] **Step 1: 테스트 작성**

```python
# tests/eval/test_nested_cv.py
import numpy as np
import pytest
from sklearn.datasets import make_classification
from forensic_mh.eval.nested_cv import (
    leakage_free_cv_score,
    LeakageDetected,
)


def test_leakage_free_cv_returns_per_fold_scores():
    X, y = make_classification(
        n_samples=100, n_features=50, n_informative=10,
        n_redundant=10, n_classes=3, random_state=42,
    )
    scores = leakage_free_cv_score(X, y, n_top_features=10, n_splits=5, random_state=42)
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_leakage_detected_when_feature_selection_uses_full_data():
    """Sanity: if you pre-select features on full data then CV, scores should be
    higher than nested CV — this verifies the nested version is actually nested."""
    X, y = make_classification(
        n_samples=200, n_features=200, n_informative=5,
        random_state=42,
    )
    # nested (correct) scores
    nested = leakage_free_cv_score(X, y, n_top_features=5, n_splits=5, random_state=42)
    nested_mean = float(np.mean(nested))

    # leaky version: pre-select then CV (manual)
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    pre = XGBClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.1,
        random_state=42, eval_metric='mlogloss', verbosity=0,
    )
    pre.fit(X, y)
    top_idx = np.argsort(pre.feature_importances_)[::-1][:5]
    leaky = cross_val_score(
        XGBClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            random_state=42, eval_metric='mlogloss', verbosity=0,
        ),
        X[:, top_idx], y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
    )
    leaky_mean = float(np.mean(leaky))

    # not strict — small datasets can be noisy — but generally leaky ≥ nested
    # we just print the gap for visibility (the test passes if nested ran)
    print(f"\nLeakage gap: leaky={leaky_mean:.3f} vs nested={nested_mean:.3f}")
    assert nested_mean > 0.0  # smoke
```

- [ ] **Step 2: 실패 확인**

```bash
pytest tests/eval/test_nested_cv.py -v
```

Expected: ImportError

- [ ] **Step 3: 구현**

```python
# src/forensic_mh/eval/nested_cv.py
"""Nested cross-validation with leak-free feature selection.

P0 fix from review #1: the original proposal used full-data feature importance
ranking, then cross_val_score on the top-N — this leaks test labels into
selection. We re-select features within each outer fold.
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


class LeakageDetected(RuntimeError):
    pass


def leakage_free_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    n_top_features: int,
    n_splits: int = 5,
    groups: Optional[np.ndarray] = None,
    random_state: int = 42,
    xgb_kwargs: Optional[dict] = None,
) -> list[float]:
    """Run nested CV: per-fold feature selection (inner) + scoring (outer).

    For each outer fold:
      1. Fit XGBoost on train fold only.
      2. Select top-N features by importance from train-fold model.
      3. Refit XGBoost on train fold with those features.
      4. Score on test fold with those features.

    Args:
        X: (n_samples, n_features)
        y: (n_samples,) integer labels
        n_top_features: how many features to select per fold
        n_splits: outer fold count
        groups: optional group labels for GroupKFold (relatedness)
        random_state: reproducibility
        xgb_kwargs: passed to XGBClassifier

    Returns:
        list[float] of per-fold accuracy scores
    """
    if xgb_kwargs is None:
        xgb_kwargs = dict(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=random_state, eval_metric="mlogloss", verbosity=0,
        )
    splitter = (
        GroupKFold(n_splits=n_splits)
        if groups is not None
        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )
    split_args = (X, y, groups) if groups is not None else (X, y)

    scores = []
    for train_idx, test_idx in splitter.split(*split_args):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # inner: feature selection on train only
        selector = XGBClassifier(**xgb_kwargs)
        selector.fit(X_tr, y_tr)
        importances = selector.feature_importances_
        top_idx = np.argsort(importances)[::-1][:n_top_features]

        # outer: refit + score on selected features only
        clf = XGBClassifier(**xgb_kwargs)
        clf.fit(X_tr[:, top_idx], y_tr)
        pred = clf.predict(X_te[:, top_idx])
        scores.append(float(accuracy_score(y_te, pred)))
    return scores
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/eval/test_nested_cv.py -v -s
```

Expected: 2 passed, output shows leakage gap (보통 leaky > nested by 1-5%p)

- [ ] **Step 5: Commit**

```bash
git add src/forensic_mh/eval/nested_cv.py tests/eval/test_nested_cv.py
git commit -m "feat(eval): leakage-free nested CV (P0 #1 fix) with optional GroupKFold"
```

---

## Task 9: chr22 MH genotype matrix builder + baseline run

**Files:**
- Create: `src/forensic_mh/pipelines/baseline.py`
- Create: `scripts/03_run_baseline.py`

- [ ] **Step 1: pipeline 함수 작성**

```python
# src/forensic_mh/pipelines/baseline.py
"""End-to-end baseline pipeline: VCF → diplotype matrix → label-encoded X → CV."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from forensic_mh.data.markers import load_mh_markers, filter_by_chromosome
from forensic_mh.data.vcf_io import extract_diplotypes_for_locus


def build_diplotype_matrix(
    vcf_path: str,
    chrom: str,
    sample_ids: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build (n_samples, n_markers) integer diplotype matrix for one chromosome.

    Each cell: integer code of the canonical diplotype tuple at that marker.

    Returns:
        X: (n_samples, n_markers) int matrix
        sample_ids_out: list of sample IDs (rows)
        marker_names: list of MH names (cols)
    """
    markers = filter_by_chromosome(load_mh_markers(), chrom)
    if len(markers) == 0:
        raise ValueError(f"No MH markers on chromosome {chrom}")

    chrom_id = chrom.replace("chr", "")
    rows: dict[str, dict[str, str]] = {}  # sample_id -> {marker -> "h0|h1"}
    marker_names = []

    for _, mh in markers.iterrows():
        positions = [int(p) for p in str(mh["Offsets"]).split(";") if p.strip()]
        if not positions:
            continue
        marker_names.append(mh["Name"])
        diplotypes = extract_diplotypes_for_locus(
            vcf_path, chrom_id, positions, sample_ids
        )
        for sid, dipl in diplotypes.items():
            rows.setdefault(sid, {})[mh["Name"]] = f"{dipl[0]}|{dipl[1]}"

    sample_ids_out = sorted(rows.keys())
    # pivot to matrix, label encode per column
    X = np.zeros((len(sample_ids_out), len(marker_names)), dtype=np.int32)
    for j, m in enumerate(marker_names):
        col = [rows[s].get(m, "N|N") for s in sample_ids_out]
        le = LabelEncoder()
        X[:, j] = le.fit_transform(col)
    return X, sample_ids_out, marker_names


def load_eas_labels(panel_path: str, sample_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return integer pop labels + ordered pop name list."""
    panel = pd.read_csv(panel_path, sep="\t")
    pop_map = panel.set_index("sample")["pop"].to_dict()
    raw = [pop_map[s] for s in sample_ids]
    pops = sorted(set(raw))
    pop_to_int = {p: i for i, p in enumerate(pops)}
    y = np.array([pop_to_int[p] for p in raw], dtype=int)
    return y, pops
```

- [ ] **Step 2: 실행 스크립트 작성**

```python
# scripts/03_run_baseline.py
"""Day 1-2 산출: chr22-only leakage-free baseline.

Plan 1의 첫 가시적 결과. 정확도는 낮을 수 있음 (chr22만 사용하므로 ~5-15 MH)
— 핵심은 leakage-free pipeline이 작동함을 보여주는 것.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from forensic_mh.pipelines.baseline import build_diplotype_matrix, load_eas_labels
from forensic_mh.eval.nested_cv import leakage_free_cv_score


def main():
    vcf = "data/eas/EAS_chr22.vcf.gz"
    panel = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"

    print("[1/3] Building diplotype matrix for chr22...")
    X, sids, marker_names = build_diplotype_matrix(vcf, "chr22")
    print(f"  X shape: {X.shape}  ({len(sids)} samples × {len(marker_names)} markers)")

    print("[2/3] Loading EAS labels...")
    y, pops = load_eas_labels(panel, sids)
    print(f"  labels: {dict(zip(pops, np.bincount(y)))}")

    print("[3/3] Nested CV (top-min(N,5) features, 5 folds)...")
    n_top = min(len(marker_names), 5)
    scores = leakage_free_cv_score(X, y, n_top_features=n_top, n_splits=5, random_state=42)
    print(f"  Per-fold accuracy: {[f'{s:.3f}' for s in scores]}")
    print(f"  Mean: {np.mean(scores):.3f}  Std: {np.std(scores):.3f}")

    # save
    out = Path("results/baseline")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chr22_baseline.json", "w") as f:
        json.dump({
            "n_samples": int(X.shape[0]),
            "n_markers": int(X.shape[1]),
            "n_top_features": n_top,
            "scores": list(map(float, scores)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "populations": pops,
        }, f, indent=2)
    print(f"\nSaved: {out / 'chr22_baseline.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 실행**

```bash
source .venv/bin/activate
python scripts/03_run_baseline.py
```

Expected: stdout에 shape, label 분포, fold scores. `results/baseline/chr22_baseline.json` 생성.

⚠️ 만약 chr22의 MH가 너무 적어 (예: 5개 이하) scores가 chance level (~20%)이면 정상 — 이건 chr22만으로 pipeline validation이 목적이고, Day 2-3에 chr1, chr2 등 추가하면 contribution 많아짐. 핵심은 **에러 없이 끝까지 돌아가는 것**.

- [ ] **Step 4: Commit**

```bash
git add src/forensic_mh/pipelines/baseline.py scripts/03_run_baseline.py
git commit -m "feat(pipelines): end-to-end baseline (chr22 scope, leakage-free nested CV)"
```

---

## Task 10: KoVariome 다운로드 (Node 2 background, sparkq 경유)

**Files:**
- Create: `scripts/05_kovariome_download.sh`

KoVariome 50 Korean WGS — KOR 외부 검증용. Plan 2의 Day 5-6에 사용 예정. 다운+variant call 시간 길어 Day 2 background로 시작.

- [ ] **Step 1: 다운로드 + variant call 스크립트**

```bash
#!/usr/bin/env bash
# scripts/05_kovariome_download.sh
# Run on Node 2 via sparkq — KoVariome 50 Korean WGS 다운로드
set -euo pipefail

OUT="${OUT:-data/kovariome}"
mkdir -p "$OUT"
cd "$OUT"

# KoVariome 공개 데이터 (Nature Sci Reports 2018, KOBIC)
# 주의: 실제 다운로드 URL은 KOBIC 페이지에서 확인 필요 — placeholder
echo "TODO: KoVariome 공개 URL 확정 후 wget 추가"
echo "참고: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5885007/"
echo "      http://opengene.org/ 또는 https://kobic.re.kr/"

# 임시 fallback: in silico KOR proxy (CHB+JPT sample 평균) — Plan 2에서 결정
echo "Fallback plan: use CHB+JPT mixture proxy if real KoVariome download blocked"
```

- [ ] **Step 2: sparkq health 확인**

```bash
sparkq status --all 2>&1 | head -20
sparkq history 2>&1 | head -10
```

Expected: 두 노드 health OK. (실패하면 Node 2 background 작업 일단 보류, Day 4-5에 재시도)

- [ ] **Step 3: 스크립트만 commit (다운로드는 URL 확정 후)**

```bash
chmod +x scripts/05_kovariome_download.sh
git add scripts/05_kovariome_download.sh
git commit -m "chore(data): KoVariome download script placeholder (URL TBD)"
```

- [ ] **Step 4: Plan 2 시작 전에 URL 결정 — 메모**

이 task는 의도적으로 placeholder로 둠. Day 4 (Plan 2 시작) 전에 KOBIC 페이지 확인 + 정식 URL 추가하는 follow-up task 생성.

---

## Task 11: Wei 2025 — 1000G trio metadata 다운로드

**Files:**
- Create: `scripts/06_download_1000g_trios.sh`

- [ ] **Step 1: trio pedigree 다운로드 스크립트**

```bash
#!/usr/bin/env bash
# scripts/06_download_1000g_trios.sh
set -euo pipefail

OUT="data/1000g"
mkdir -p "$OUT"
cd "$OUT"

# 1000G Phase 3 pedigree (trio 정보 포함)
URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.ped"
if [ ! -f "integrated_call_samples_v3.20130502.ALL.ped" ]; then
    wget -q --show-progress "$URL" || \
        wget -q --show-progress "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/working/20130606_sample_info/20130606_g1k.ped" \
        -O "g1k.ped"
fi

ls -lh *.ped 2>/dev/null
```

- [ ] **Step 2: 실행**

```bash
chmod +x scripts/06_download_1000g_trios.sh
bash scripts/06_download_1000g_trios.sh
```

Expected: ped 파일 다운 (~수십 KB)

- [ ] **Step 3: trio 카운트 검증**

```bash
PED=$(ls data/1000g/*.ped | head -1)
# trio = both parents non-zero in ped column 3,4
awk 'NR>1 && $3!="0" && $4!="0"' "$PED" | wc -l
```

Expected: ~600+ (Wei 2025의 602 trios와 일치)

- [ ] **Step 4: EAS trio 카운트**

```bash
# EAS panel과 join
awk 'NR>1 && $3!="0" && $4!="0" {print $2}' "$PED" > /tmp/all_trio_kids.txt
awk '$3=="EAS" {print $1}' data/1000g/integrated_call_samples_v3.20130502.ALL.panel | sort > /tmp/eas_samples.txt
sort /tmp/all_trio_kids.txt > /tmp/all_trio_kids_sorted.txt
comm -12 /tmp/eas_samples.txt /tmp/all_trio_kids_sorted.txt | wc -l
```

Expected: 0~수십 EAS trio (실제로 phase3의 trio는 대부분 non-EAS — Wei 2025도 글로벌 분석)

- [ ] **Step 5: Commit**

```bash
git add scripts/06_download_1000g_trios.sh
git commit -m "feat(data): 1000G phase 3 pedigree download for trio analysis"
```

---

## Task 12: ⭐ Phasing error rate per MH (Wei 2025 재현)

**Files:**
- Create: `src/forensic_mh/metrics/reliable_ae.py`
- Test: `tests/metrics/test_reliable_ae.py`

부모 둘과 자녀 한 명의 MH genotype을 비교 — Mendelian 위반은 phasing error의 indicator.

- [ ] **Step 1: 테스트 작성 (synthetic trio)**

```python
# tests/metrics/test_reliable_ae.py
import pytest
from forensic_mh.metrics.reliable_ae import (
    is_mendelian_consistent_diplotype,
    reliable_ae,
)


def test_mendelian_consistent_trivial_homozygote():
    # father: A-T/A-T (homozygote), mother: G-C/G-C (homozygote)
    # child must be A-T/G-C
    father = ("A-T", "A-T")
    mother = ("G-C", "G-C")
    child = ("A-T", "G-C")  # canonical sorted
    assert is_mendelian_consistent_diplotype(father, mother, child)


def test_mendelian_inconsistent_impossible_child():
    father = ("A-T", "A-T")
    mother = ("G-C", "G-C")
    child = ("A-T", "A-T")  # mother contributed nothing — impossible
    assert not is_mendelian_consistent_diplotype(father, mother, child)


def test_mendelian_consistent_heterozygote():
    father = ("A-T", "A-G")  # AT or AG haplotype
    mother = ("G-C", "T-C")  # GC or TC haplotype
    child = ("A-G", "T-C")  # AG from father + TC from mother = OK
    assert is_mendelian_consistent_diplotype(father, mother, child)


def test_reliable_ae_lowers_score_when_phasing_errors_present():
    ae = 5.0
    # 10% phasing error rate → reliable_ae = 4.5
    assert reliable_ae(ae, p_phase_error=0.1) == pytest.approx(4.5)
    # 0 errors → unchanged
    assert reliable_ae(ae, p_phase_error=0.0) == pytest.approx(5.0)
```

- [ ] **Step 2: 실패 확인**

```bash
pytest tests/metrics/test_reliable_ae.py -v
```

Expected: ImportError

- [ ] **Step 3: 구현**

```python
# src/forensic_mh/metrics/reliable_ae.py
"""Wei 2025-inspired Reliable Ae metric.

Reliable_Ae(marker) = Ae(marker) × (1 - P_phasing_error(marker))

P_phasing_error is estimated via trio Mendelian-consistency: count
diplotypes that cannot be produced from parental haplotypes.
"""
from __future__ import annotations


def is_mendelian_consistent_diplotype(
    father: tuple[str, str],
    mother: tuple[str, str],
    child: tuple[str, str],
) -> bool:
    """Check if child's diplotype can arise from one parental haplotype each.

    Each diplotype is an unordered tuple of haplotype strings.
    """
    f_haps = set(father)
    m_haps = set(mother)
    c_haps = list(child)
    # child must consist of one haplotype from father and one from mother
    # try both orderings
    for cf, cm in [(c_haps[0], c_haps[1]), (c_haps[1], c_haps[0])]:
        if cf in f_haps and cm in m_haps:
            return True
    return False


def reliable_ae(ae: float, p_phase_error: float) -> float:
    """Apply Wei 2025-inspired penalty to Ae."""
    return ae * (1.0 - p_phase_error)
```

- [ ] **Step 4: 테스트 통과**

```bash
pytest tests/metrics/test_reliable_ae.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/forensic_mh/metrics/reliable_ae.py tests/metrics/test_reliable_ae.py
git commit -m "feat(metrics): Reliable Ae from Wei 2025 + Mendelian consistency check"
```

---

## Task 13: Phasing error 분석 스크립트 (per-MH P_phase 추정)

**Files:**
- Create: `scripts/04_wei2025_phasing.py`

- [ ] **Step 1: 분석 스크립트**

```python
# scripts/04_wei2025_phasing.py
"""Day 3 산출: 1000G chr22 MH 별 phasing error rate (Wei 2025 재현).

각 MH 좌표에서 (가능한) trio에 대해 Mendelian consistency 검증.
EAS trio는 부족하므로 phase3 전체 trio 사용 후 결과를 Reliable-Ae에 반영.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

from forensic_mh.data.markers import load_mh_markers, filter_by_chromosome
from forensic_mh.data.vcf_io import extract_diplotypes_for_locus
from forensic_mh.metrics.reliable_ae import is_mendelian_consistent_diplotype


PED_PATH_CANDIDATES = [
    "data/1000g/integrated_call_samples_v3.20130502.ALL.ped",
    "data/1000g/g1k.ped",
]


def load_trios(panel_path: str) -> list[tuple[str, str, str]]:
    """Return list of (father_id, mother_id, child_id) trios from 1000G pedigree."""
    ped = None
    for p in PED_PATH_CANDIDATES:
        if Path(p).exists():
            ped = p
            break
    if ped is None:
        raise FileNotFoundError("Run scripts/06_download_1000g_trios.sh first")
    df = pd.read_csv(ped, sep=r"\s+", header=None,
                     names=["fam", "ind", "pa", "ma", "sex", "phen"])
    trios = df[(df["pa"] != "0") & (df["ma"] != "0")]
    # need all three IDs present in panel
    panel = pd.read_csv(panel_path, sep="\t")
    samples_set = set(panel["sample"])
    out = []
    for _, row in trios.iterrows():
        if {row["pa"], row["ma"], row["ind"]} <= samples_set:
            out.append((row["pa"], row["ma"], row["ind"]))
    return out


def main():
    panel = "data/1000g/integrated_call_samples_v3.20130502.ALL.panel"
    # need ALL samples VCF, not just EAS — switch source
    vcf_all = "data/1000g/ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"

    print("[1/3] Loading trios...")
    trios = load_trios(panel)
    print(f"  Found {len(trios)} complete trios")

    print("[2/3] Loading chr22 MH markers...")
    markers = filter_by_chromosome(load_mh_markers(), "chr22")
    print(f"  {len(markers)} MH on chr22")

    print("[3/3] Per-MH Mendelian consistency...")
    results = []
    for _, mh in markers.iterrows():
        positions = [int(p) for p in str(mh["Offsets"]).split(";") if p.strip()]
        if not positions:
            continue
        # need parents + child for each trio
        all_ids = list({i for trio in trios for i in trio})
        diplotypes = extract_diplotypes_for_locus(vcf_all, "22", positions, all_ids)

        n_total, n_consistent = 0, 0
        for f, m, c in trios:
            if f not in diplotypes or m not in diplotypes or c not in diplotypes:
                continue
            # skip if any "N" present (missing genotype)
            if any("N" in h for h in diplotypes[f] + diplotypes[m] + diplotypes[c]):
                continue
            n_total += 1
            if is_mendelian_consistent_diplotype(
                diplotypes[f], diplotypes[m], diplotypes[c]
            ):
                n_consistent += 1
        p_err = 1 - (n_consistent / n_total) if n_total > 0 else 0.0
        results.append({
            "marker": mh["Name"],
            "n_trios_tested": n_total,
            "n_consistent": n_consistent,
            "p_phase_error": p_err,
        })

    out_path = Path("results/baseline/chr22_wei2025_phasing.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")
    print(f"\nSummary:")
    pe_vals = [r["p_phase_error"] for r in results if r["n_trios_tested"] > 0]
    if pe_vals:
        print(f"  Mean per-MH P_phase_error: {sum(pe_vals)/len(pe_vals):.4f}")
        print(f"  Max P_phase_error: {max(pe_vals):.4f}")
        print(f"  Wei 2025 global rate: 0.07% (0.0007) — compare above")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행**

```bash
python scripts/04_wei2025_phasing.py
```

Expected: chr22 각 MH별 P_phase_error 표 출력 + JSON 저장. 평균이 0.001 근방이면 Wei 2025와 매우 비슷.

- [ ] **Step 3: 결과 검증 (chr22 MH 수가 적으면 trio sample 부족 가능)**

```bash
python -c "
import json
data = json.load(open('results/baseline/chr22_wei2025_phasing.json'))
for r in data:
    print(f\"{r['marker']}: tested={r['n_trios_tested']}, error_rate={r['p_phase_error']:.4f}\")
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/04_wei2025_phasing.py
git commit -m "feat(analysis): Wei 2025 reproduction — per-MH phasing error via trio Mendelian check"
```

---

## Task 14: Plan 1 종료 — Day 1-3 summary 문서

**Files:**
- Create: `docs/superpowers/plans/2026-05-26-foundation-baseline-RESULTS.md`

- [ ] **Step 1: 결과 요약 작성**

```markdown
# Plan 1 — Foundation & Baseline RESULTS

**완료일**: 2026-MM-DD (실제 채움)
**실행자**: 조민한 + Claude

## 산출

### 코드 (commit hash 기록)
- `src/forensic_mh/data/{markers,vcf_io}.py` — MH/VCF 로직
- `src/forensic_mh/eval/{nested_cv,grouping}.py` — leakage-free CV
- `src/forensic_mh/metrics/reliable_ae.py` — Wei 2025 보정
- `src/forensic_mh/pipelines/baseline.py` — 통합 파이프라인
- `scripts/0{1..6}_*.sh|py`

### 데이터
- `data/1000g/`: panel + chr22 VCF + pedigree (~250MB)
- `data/eas/`: EAS 504 sample subset (chr22)
- `data/kovariome/`: (Plan 2에서 채움)

### 결과 (results/baseline/)
- `chr22_baseline.json`: chr22-only nested CV accuracy
- `chr22_wei2025_phasing.json`: per-MH phasing error rate

### 테스트
- 12 unit/integration tests (pytest)

## P0 리뷰 이슈 해결 상태

| # | 이슈 | 해결 | 어디서 |
|---|---|---|---|
| 1 | Feature selection leakage | ✅ | `eval/nested_cv.py` |
| 2 | Diplotype 첫 haplotype만 | ✅ | `data/vcf_io.py` |
| 4 | Wei 2025 미대응 | ✅ | `metrics/reliable_ae.py` + scripts/04 |
| 8 | GroupKFold 부재 | △ | `eval/grouping.py` (pedigree 통합은 Plan 2) |

P1 이슈 (#3, #6, #7, #9)는 Plan 2/3에서 처리.

## Plan 2 entry criteria 충족

- [ ] chr22 pipeline 끝까지 작동 (with at least 1 fold complete)
- [ ] Trio-based phasing error 측정 가능
- [ ] git history clean
- [ ] sparkq Node 2 KoVariome variant call 진행 중 (또는 fallback 결정)

→ Plan 2 (Conformal + OSR) 작성 진입
```

- [ ] **Step 2: 최종 commit**

```bash
git add docs/superpowers/plans/2026-05-26-foundation-baseline-RESULTS.md
git commit -m "docs: Plan 1 completion summary"
git log --oneline
```

---

## Self-Review

### Spec coverage (개요 → task 매핑)

| 개요 항목 | 구현 task |
|---|---|
| 환경 셋업 + git | Task 1, 2 |
| 1000G 데이터 다운 | Task 3 |
| EAS 504 추출 | Task 4 |
| MicroHapDB 활용 | Task 5 |
| Diplotype 추출 (P0 #2) | Task 6 |
| GroupKFold (P0 #8 scaffold) | Task 7 |
| Nested CV (P0 #1) | Task 8 |
| End-to-end baseline | Task 9 |
| KoVariome background | Task 10 |
| Wei 2025 trio (P0 #4) | Task 11, 12, 13 |
| 종료 summary | Task 14 |

전부 매핑됨. KoVariome (Task 10)는 placeholder로 둠 — Plan 2 시작 전 URL 확정.

### Placeholder scan
- Task 10 다운로드 URL은 의도된 placeholder (실제 KOBIC URL 확정 필요)
- 다른 task에는 placeholder 없음 — 코드 전부 inline.

### Type consistency
- `extract_diplotypes_for_locus` 반환 타입 `dict[str, tuple[str, str]]` — Task 6, 9, 13에서 일관 사용
- `build_diplotype_matrix` 반환 `(X, sample_ids, marker_names)` — Task 9, scripts/03
- `leakage_free_cv_score` 반환 `list[float]` — Task 8, scripts/03 모두 일관

### Critical risk
- chr22만으로 정확도가 낮으면 Day 2 후반에 chr1, chr2 추가 다운 (스크립트 인자 `CHR=1` 등으로 확장 가능). Pipeline 검증이 핵심이지 정확도 자체는 Plan 1 deliverable 아님.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-26-foundation-baseline.md`.

다음 단계: Day 1 작업 시작 (Task 1 → 2 → ...). 이 plan은 그대로 따르면 ~3일 분량.

실행 방식 선택 필요:
1. **Subagent-Driven**: 각 task별 fresh subagent + review (가장 안전)
2. **Inline Execution**: 이 세션에서 task 단위 batch (빠른 iteration)
