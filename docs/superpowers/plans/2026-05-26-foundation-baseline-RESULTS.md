# Plan 1 — Foundation & Baseline RESULTS

**완료일**: 2026-05-29
**실행자**: 조민한 + Claude (Opus 4.8)
**상태**: pipeline·metric 완료. genome-wide 추출(Task 10)과 trio phasing(NYGC)은 의도적 deferred.

---

## 산출 코드 (commit)

| 파일 | 책임 | 커밋 |
|---|---|---|
| `src/forensic_mh/data/markers.py` | MicroHapDB wrapper (hg19/hg38 coord-aware) | 70bd806 |
| `src/forensic_mh/data/vcf_io.py` | diplotype 추출 (양 haplotype 보존, P0 #2) | ad8fec6 |
| `src/forensic_mh/eval/{nested_cv,grouping}.py` | leakage-free nested CV + GroupKFold scaffold | 93bab2d |
| `src/forensic_mh/metrics/reliable_ae.py` | Mendelian 일관성 + **EAS Ae** + informative-meiosis | acd60ed, (Ae) Task12 fix |
| `src/forensic_mh/pipelines/baseline.py` | end-to-end diplotype matrix + CV | f1c3404 |
| `scripts/01–05` | 다운로드·추출·baseline·phasing·related-samples | 다수 |

## 데이터
- `data/1000g/`: panel + chr22 **v5b** VCF + g1k.ped + related-samples VCF(31명)
- `data/eas/`: EAS 504 subset (chr22)

## 결과 (results/baseline/ — gitignored)
- `chr22_baseline.json`: chr22 nested-CV 정확도 (panel size 5/10/20/30/53)
- `chr22_reliable_ae.json`: per-marker **EAS Ae** 표 (+ phasing deferred 플래그)

### 핵심 수치
- baseline 정확도: **0.24–0.30** (5-class chance=0.20) — chr22 53 marker만으로는 near-chance. **pipeline 검증 목적 달성**, 정확도는 genome-wide 후 의미.
- EAS Ae: mean **4.91**, max **17.44**, top-5 = [17.4, 13.6, 9.7, 9.7, 8.4]. → 고-Ae marker가 Wei 2025이 phasing-risky로 지목하는 바로 그 marker.
- 테스트: **22 passed**.

---

## P0 리뷰 이슈 해결 상태 (정직본)

| # | 이슈 | 상태 | 근거 |
|---|---|---|---|
| 1 | Feature selection leakage | ✅ 해결 | `eval/nested_cv.py` — outer fold 안에서 재선택 |
| 2 | Diplotype 첫 haplotype만 | ✅ 해결 | `data/vcf_io.py` — 양 haplotype canonical tuple |
| 4 | Wei 2025 / Reliable-Ae | △ **부분** | Ae(EAS) 산출 ✅. phasing penalty는 **deferred** (아래) |
| 8 | GroupKFold 부재 | △ **scaffold만** | `eval/grouping.py` 존재하나 커밋된 baseline은 `groups=` 미전달. 1000G EAS는 사실상 unrelated이라 영향 미미 — 일반화 시 배선 필요 |

P1 이슈(#3, #6, #7, #9)는 Plan 2/3에서 처리.

---

## 의도적 Deferred / Scope 결정

### (1) Trio phasing 재현 — NYGC 필요 (data wall)
- 1000G phase3 **표준 release(2504)는 unrelated subset** → complete trio **0개**.
- related_samples VCF(31명)를 합쳐도 complete trio **6개(전부 non-EAS)** → 통계 무의미.
- Wei 2025(602 trios) 재현은 **NYGC 30x high-coverage release(3202명, 602 trios, GRCh38)** 필요.
- 조치: `scripts/04`는 EAS Ae를 지금 산출하고, complete trio < `MIN_TRIOS(30)`이면 `p_phase_error/reliable_ae = null + status="deferred"`. NYGC VCF 확보 시 `TRIO_VCF_SOURCES`에 경로 추가 + `BUILD="hg38"`로 동일 로직 재현 (hg38-ready).

### (2) KOR 외부 검증 — scope 제외 (2026-05-29)
- Korean WGS(KoVariome/KPGP 등) 확보·variant calling 불가 → KOR 데이터 보강 폐기.
- open-set "unknown"은 Plan 2에서 **1000G 비-EAS superpopulation + LOPO**로 대체. KOR 부재는 논문 limitation으로 기술.

### (3) Genome-wide MH 추출 (redefined Task 10) — pending
- chr22-only가 near-chance인 근본 원인. `scripts/02`는 `CHR` 인자로 chromosome별 실행 가능.
- 남은 작업: 전 autosome VCF 다운로드(~다수 GB) + `build_diplotype_matrix` 다중 chromosome 확장 + 재실행. **다운로드 규모가 커서 별도 greenlight 후 진행** (Plan 2/3는 marker-set agnostic이라 병렬 가능).

---

## Plan 2 entry criteria

- [x] chr22 pipeline end-to-end 작동 (leakage-free, 22 tests green)
- [x] EAS Ae 표 산출 (Reliable-Ae의 base term)
- [x] Plan 2 (Conformal+OSR trust layer) 작성 완료 — `2026-05-29-conformal-osr.md`
- [~] trio phasing: deferred (NYGC) — Plan 2 진행의 blocker 아님
- [~] genome-wide: pending — 결과 강도에 영향, pipeline 정확성엔 무관

→ **Plan 2 진입 가능**. genome-wide는 Plan 2와 병렬로 진행 권장.

## Trustworthy Forensic Model 진척
Plan 1은 representation(향후 SSL FM) + trust layer(Plan 2)가 올라탈 leakage-free base를 제공. 다음: Plan 2 trust layer 구현(`predict_proba` 인터페이스) → Plan 3 SSL FM → Plan 4 통합.
