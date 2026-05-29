# Plan 2 — Conformal + OSR RESULTS

**완료일**: 2026-05-29 (genome-wide 정식 결과).
**실행자**: 조민한 + Claude (Opus 4.8)
**상태**: trust layer 완료(TDD). **genome-wide 22-chrom 패널(3042 MH markers, EAS 504 + 비-EAS OOD 300)으로 정식 실행 완료.**

---

## 산출 코드
- `src/forensic_mh/uq/conformal.py` — split-conformal core (order-statistic quantile, Mondrian per-class)
- `src/forensic_mh/uq/conformal_classifier.py` — model-agnostic wrapper (`predict_proba`만 의존 → SSL FM 재사용)
- `src/forensic_mh/uq/openset.py` — MSP score, AUROC, FPR@95TPR, reject-rate, `open_set_decision`(empty-set OR low-MSP)
- `src/forensic_mh/data/encoding.py` — OOD-aware 공유 encoder
- `src/forensic_mh/eval/lopo.py` — near-OOD split
- `scripts/10_conformal_curve.py`, `11_openset_ood.py`, `12_lopo_nearood.py` (genome-wide wired)
- base model: XGBoost (200 trees, depth 4), Mondrian split-conformal, cal=0.3

## genome-wide 결과 (3042 markers, results/conformal/)

### coverage_curve.json — 분포-free 보장 작동 + marker 늘수록 set tighter
| α | target (1−α) | marginal coverage | mean set size (of 5) |
|---|---|---|---|
| 0.30 | 0.70 | 0.770 | 2.09 |
| 0.20 | 0.80 | 0.809 | 2.28 |
| 0.10 | 0.90 | 0.888 | 2.61 |
| 0.05 | 0.95 | **0.954** | 3.16 |

marginal coverage가 모든 α에서 1−α에 근접(α=0.05는 보수적으로 over-cover). **set size는 marker가 늘수록 작아짐** — 정보량↑ → 같은 보장에 더 좁은 prediction set.

### openset_ood.json — far-OOD: MSP 신호 moderate, empty-set reject는 미발화
- OOD(비-EAS) unseen-diplotype fraction: **0.132**
- **MSP AUROC: 0.695**  /  FPR@95TPR: 0.75
- empty-set reject rate: in-dist 0.0, **OOD 0.0** (모든 α) — base model이 비-EAS 샘플도 어떤 EAS 집단으로 **과확신** 분류 → conformal set이 비지 않음.

### lopo_nearood.json — near-OOD: 미검출 (α=0.10)
- 5개 held-out 집단 모두 reject gap(held−known) = **0.0**. 근연 EAS 집단은 학습된 집단으로 과확신 오분류 → empty-set으로 안 잡힘.

### marker 수에 따른 OSR 신호 추이 (MSP AUROC) — paper용
| panel | markers | MSP AUROC (far-OOD) |
|---|---|---|
| chr22 only | 53 | 0.50 (chance) |
| chr1–5 | 1058 | 0.67 |
| **genome-wide** | **3042** | **0.695** |

→ AUROC는 marker↑에 따라 상승하나 **~0.69–0.70에서 포화**. XGBoost MSP만으로는 비-EAS를 강하게 분리 못 함.

## 핵심 주장 검증 상태 (정식)
- [x] **marginal coverage ≈ 1−α (분포-free 보장)** — 실증 성공, 모든 α
- [x] **marker↑ → set tighter** — 53→3042 marker에서 set 3.79→2.61 (@α=0.1) 감소
- [x] per-class(Mondrian) quantile 산출 — 작동
- [~] **far-OOD 분리** — MSP AUROC 0.695 (moderate, 포화). empty-set reject는 0 (미발화)
- [ ] **near-OOD(LOPO) 분리** — 미검출(gap 0). 근연 집단 + 과확신 base model의 한계

## 핵심 교훈 (논문에 활용)
1. **Coverage와 OSR은 독립 성질**: 분포-free coverage 보장은 base model이 약해도 유지되고 marker가 늘수록 set이 좁아진다(성공). 반면 OSR 분리력은 **base model의 calibration에 의존** — XGBoost는 OOD에 과확신이라 (i) conformal **empty-set reject가 발화하지 않고**, (ii) MSP AUROC가 0.69에서 포화한다.
2. **이것이 Plan 3(SSL FM)의 동기**: 더 잘 보정된 표현(contrastive + ADO invariance)이 OOD 과확신을 줄이면 empty-set/MSP OSR이 살아날 가능성. trust layer는 그대로(`predict_proba`) base만 교체.
3. **운영점**: `open_set_decision`의 OR 규칙에서 empty-set이 0이므로 실질적으로 **MSP threshold가 reject를 좌우** — 운영점은 msp_threshold로 조절해야 하며, 현재 FPR@95=0.75는 forensic 사용엔 부족(개선 필요).

## 한계 (논문 명시)
- OSR 분리력 약함(far-OOD AUROC 0.69, near-OOD 미검출). base model calibration 문제 — Plan 3에서 개선 시도.
- KOR 데이터 부재: OOD는 비-EAS superpop + LOPO 대체. Korean 직접 검증 불가.
- phasing penalty(Reliable-Ae)는 NYGC trio 필요로 deferred (Plan 1 참조).

## Trustworthy Forensic Model 진척
trust layer는 `predict_proba`에만 의존 → Plan 3 `ForensicFMClassifier`에 그대로 wrapping(이미 통합 테스트 통과). **Plan 4 = base_estimator 교체 + 본 scripts 재실행**. coverage half는 확정, OSR half는 Plan 3 FM으로 개선 대상.

## 다음
1. Plan 3b: SSL FM을 genome-wide로 pretrain/finetune → `ForensicFMClassifier`로 scripts 10/11/12 재실행 → OSR이 XGBoost 대비 개선되는지 비교(핵심 가설).
2. OSR 운영점: msp_threshold sweep로 FPR@95 개선 곡선 산출.
