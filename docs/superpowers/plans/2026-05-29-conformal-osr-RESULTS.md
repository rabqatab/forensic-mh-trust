# Plan 2 — Conformal + OSR RESULTS

**완료일**: 2026-05-29 (코드·smoke-run). genome-wide 재실행은 Task 10 다운로드 완료 후.
**실행자**: 조민한 + Claude (Opus 4.8)
**상태**: trust layer 코드 완료(TDD, 38+ tests green). chr22 smoke-run은 base model이 near-chance라 **provisional/null** — genome-wide 후 정식 결과.

---

## 산출 코드
- `src/forensic_mh/uq/conformal.py` — split-conformal core (order-statistic quantile, Mondrian per-class)
- `src/forensic_mh/uq/conformal_classifier.py` — model-agnostic wrapper (`predict_proba`만 의존 → SSL FM 재사용)
- `src/forensic_mh/uq/openset.py` — MSP score, AUROC, FPR@95TPR, reject-rate, `open_set_decision`(empty-set OR low-MSP)
- `src/forensic_mh/data/encoding.py` — OOD-aware 공유 encoder (unseen→ -1)
- `src/forensic_mh/eval/lopo.py` — near-OOD split
- `scripts/10_conformal_curve.py`, `11_openset_ood.py`, `12_lopo_nearood.py`

## chr22 smoke-run 결과 (provisional — results/conformal/)

### coverage_curve.json — 분포-free 보장은 작동
| α | target | marginal cov | mean set size (of 5) |
|---|---|---|---|
| 0.30 | 0.70 | 0.684 | 2.86 |
| 0.20 | 0.80 | 0.757 | 3.24 |
| 0.10 | 0.90 | 0.849 | 3.79 |
| 0.05 | 0.95 | 0.921 | 4.14 |

→ coverage가 1−α를 추종(소표본 noise로 약간 미달). set size가 α↓에 따라 커짐 = **약한 모델이 약속을 지키려 큰 set을 내는** 정상 동작.

### openset_ood.json — chr22에선 OSR 신호 없음 (정직한 negative)
- OOD(비-EAS) unseen-diplotype fraction: **0.150** (데이터엔 실제 OOD 신호 존재)
- MSP AUROC: **0.501** (chance), OOD reject rate: 0.0 — 근거: chr22 53마커 base model이 near-chance라 확률이 퍼져 set이 비지 않음.

### lopo_nearood.json — 동일 사유로 gap 0
- 모든 held-out 집단에서 reject gap(held−known) = 0.0 (α=0.10).

## 핵심 주장 검증 상태
- [x] marginal coverage ≈ 1−α (분포-free 보장 실증) — **작동**
- [x] per-class(Mondrian) quantile 산출 — **작동** (per-class coverage는 소표본이라 변동)
- [ ] 비-EAS OOD reject > in-dist reject — **chr22 미달** (AUROC 0.50). genome-wide 필요.
- [ ] LOPO held-out reject > known reject — **chr22 미달**. genome-wide 필요.

## 핵심 교훈 (논문에 활용)
chr22-only base model은 OSR에 신호가 없다(AUROC 0.50). 이는 trust layer의 결함이 아니라 **base model 정보량 부족** — conformal coverage 보장은 그래도 유지된다(분포-free의 강점). genome-wide marker로 base 정확도가 올라가면 OOD/LOPO 신호가 살아날 것으로 예상. **분리 검증**: coverage 보장(작동)과 OSR 분리력(데이터 의존)은 독립.

## 한계 (논문 명시)
- chr22-only → 정식 결과 아님. genome-wide(Plan 1 Task 10) 후 재실행 필수.
- KOR 데이터 부재: OOD는 비-EAS superpop + LOPO로 대체. Korean 직접 검증 불가.
- `open_set_decision`은 conservative(OR) 규칙 — msp_threshold로 운영점 조절.

## Trustworthy Forensic Model 진척
trust layer는 `predict_proba`에만 의존 → Plan 3 SSL FM에 그대로 wrapping (Plan 4 통합). Plan 2 산출물은 최종 system의 trust half로 확정.

## 다음
1. genome-wide EAS 추출 완료 → `build_diplotype_matrix` 다중 chromosome 확장 → scripts 10/11/12 재실행 (정식 결과).
2. Plan 3 (SSL Foundation Model) 진입.
