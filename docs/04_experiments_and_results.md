# 04 — Experiments & Results (comprehensive log)

**프로젝트**: Trustworthy Forensic Microhaplotype Model — 1000G East-Asian 5집단 분류 + calibrated UQ/open-set + SSL foundation model.
**최종 갱신**: 2026-05-30 (Opus 4.8)
**재현**: 모든 코드 `main` 브랜치, uv 환경(`microhapdb==0.12`, torch 2.12.0+cu130). 결과 JSON은 `results/`(gitignored), 코드/스크립트 커밋됨. 79 pytest green.

> **성격**: 이 문서는 **chronological 실험 로그**다. RQ 확정본과 RQ↔증거 매핑은 [`05_research_questions.md`](05_research_questions.md)가 canonical 진입점이며, 이 문서의 각 실험은 아래 표로 RQ에 매핑된다. **현재 canonical 수치는 §13(정확도)·§20(trust 엄밀성)·§21(최소 패널)**에 있고, 초기 XGBoost/ordinal 섹션(§3–§5, §7, §9)은 *superseded*로 표시(이력 보존용).

> **한 줄 요약 (2026-05-30 — 이전 "57% 천장/XGBoost 과확신" 주장 폐기)**: (1) **인코딩·모델 선택이 결정적**: MH는 명목형이라 **one-hot + Logistic Regression = 79.6% (5-fold CV)** 로, ordinal-tree(XGBoost 52%, RF 57%)·SSL FM(26%)을 모두 압도. "57% 천장"은 FST가 아니라 *ordinal 인코딩 artifact*였음(Simplicity-Test 승리). (2) **LogReg(one-hot)이 정확도·OSR 모두 최고**(acc 79.6%, far-OOD AUROC 0.840±0.016, 10-seed). (3) Conformal coverage 보장 작동(base-agnostic), marker↑로 set 좁아짐; 단 ADO(열화)에서 보장 저하. (4) **Calibration(ECE)은 XGBoost가 최선(0.077)·RF가 최악(0.315)** — OSR AUROC(rank)와 ECE(보정)는 별개; 앙상블 epistemic은 OOD 미분리(≈chance). (5) 고정확도엔 compact 최소 패널 없음 — genome-wide 필요(§21).

### RQ ↔ 섹션 매핑

| RQ | 핵심 주장 | canonical 섹션 | 보조/이력 |
|---|---|---|---|
| **RQ1** ★ | open-set 신뢰도는 base-model이 좌우(정확도 아님) | **§20**(10-seed, 4σ), §11, §24(DL도 OSR 약) | §4·§5·§9 *(superseded)* |
| RQ2 | conformal이 저정확도에서도 목표 커버리지 | **§3**, §21 | — |
| RQ3 | 57% 천장은 인코딩 artifact; one-hot+linear 우위 | **§13**, 부록 A, §24(DL 5계열도 짐) | §7·§8·§9 *(superseded/보조)* |
| RQ4 | ECE ≠ open-set 분리 | **§14** | — |
| RQ5 | 배치 가능한 최소 forensic 패널 존재 (정확도–trust frontier) | **§23**(다변량 선택) | §21 *(univariate sub-result)*, §7 |
| RQ6 | conformal 보장이 ADO에서 graceful 저하 | **§15** | — |
| RQ7 | HGDP 외부 코호트 전이 (in-callset 87.3%) | **§22** *(ANSWERED, clean hg38)* | — |
| (비-RQ) | SSL FM = Paper 2; Reliable-Ae = deferred | §6·§10(FM), §2(Ae) | — |

---

## 0. 데이터셋

### 0.1 개인 genotype = 1000 Genomes Phase 3 VCF (제안서 명시 데이터셋)
- 빌드 **hg19/b37** (v5b VCF, `##source=1000GenomesPhase3Pipeline`, release 20130502, 2,504명).
- in-distribution: EAS 504명 — **CDX 93, CHB 103, CHS 105, JPT 104, KHV 99**.
- 추출: `scripts/01,02,06` (genome-wide, disk-efficient + per-chrom retry).

### 0.2 마커 정의 = MicroHapDB 0.12 (genotype과 별개 DB)
- genome-wide(22 상염색체, hg19 좌표) **3,042개**. chr22 단독 53, chr1–5 1,058.
- 버전 검수: 0.4=290, 0.7=507, 0.10–0.12=3,053. 제안서의 "412"는 동일 DB 구버전 수치(추가 데이터셋 아님). `microhapdb==0.12` 고정.

### 0.3 OOD (open-set "unknown")
- **far-OOD**: 1000G 비-EAS 300명(EUR/AFR/SAS/AMR 각 75, 결정적). KOR 부재 → 비-EAS 대용.
- **near-OOD**: LOPO(leave-one-EAS-population-out).

### 0.4 Trio 데이터 벽
- phase3 표준(2,504)은 unrelated → complete trio 0개; related 합쳐도 6개(비-EAS). Wei 2025 phasing 재현은 NYGC 30x(3,202, 602 trios) 필요 → **deferred**.

---

## 1. Exp 1 — Leakage-free baseline (Plan 1)  `[pipeline 검증]`
**방법**: diplotype 추출(양 haplotype, P0 #2) → per-marker 인코딩 → **leakage-free nested CV**(outer fold 내부 feature selection, P0 #1). GroupKFold scaffold만(P0 #8; EAS는 사실상 unrelated).
**결과 (chr22, 53마커, chance 0.20)**: top-5 0.236, top-20 0.298, all 0.298 → near-chance(정보량 부족). **상태 ✅** pipeline 검증. (genome-wide 정확도는 §7.)

## 2. Exp 2 — Reliable-Ae / Wei 2025 phasing (Plan 1)  `[비-RQ: Reliable-Ae deferred]`
- **Ae(EAS, chr22)**: mean **4.91**, max **17.44**. 고-Ae marker가 Wei 2025의 phasing-위험 marker.
- **P_phase_error/Reliable-Ae**: **deferred**(complete trio 6개, NYGC 필요). 스크립트 hg38-ready. **상태 △**.

## 3. Exp 3 — Conformal coverage (Plan 2, XGBoost base)  `[→ RQ2]`
LAC nonconformity + **Mondrian** per-class quantile(order-statistic). `results/conformal/coverage_curve.json`.

genome-wide (3,042마커):
| α | target | coverage | set size |
|---|---|---|---|
| 0.30 | 0.70 | 0.770 | 2.09 |
| 0.20 | 0.80 | 0.809 | 2.28 |
| 0.10 | 0.90 | 0.888 | 2.61 |
| 0.05 | 0.95 | **0.954** | 3.16 |

→ ✅ coverage가 1−α 추종, **marker↑ → set tighter**(chr22 53마커 α=0.1 set 3.79 → 3,042마커 2.61).

### 3.1 Per-population (class-conditional) coverage — base-model 의존 발견  `[→ RQ2·RQ1 · FSI:G (b)]`
`scripts/trust/53_perpop_coverage.py`(LogReg, 10-seed) + `coverage_curve.json`(XGBoost, 위 표 per_class). **forensic은 per-population 신뢰성 요구** → marginal만으론 부족.

| α (target) | base | marginal | set | CDX | CHB | CHS | JPT | **KHV** |
|---|---|---|---|---|---|---|---|---|
| 0.10 (0.90) | **LogReg**(권장, 10-seed) | 0.916 | 1.72 | 0.95 | 0.90 | 0.93 | 0.90 | **0.90** |
| 0.10 (0.90) | XGBoost(single split) | 0.888 | 2.61 | 1.00 | 0.84 | 0.97 | 0.94 | **0.70** |
| 0.05 (0.95) | **LogReg**(10-seed) | 0.957 | 2.06 | 0.97 | 0.94 | 0.96 | 0.95 | **0.96** |

**발견**: 권장 **LogReg는 α=0.1·0.05에서 모든 집단 target 충족**(KHV 포함). 반면 **XGBoost는 marginal이 비슷(0.888)한데도 KHV 0.70으로 subgroup 붕괴**(CHB 0.84도 미달) — marginal이 숨김. → **per-population coverage 자체가 base-model 의존 = RQ1 재입증**. 정직한 forensic 메시지: "marginal 평균이 가리는 subgroup 실패를 *드러내고*, calibrated 단순 모델에서만 per-pop이 성립". (주의: α=0.2에선 LogReg marginal도 0.786으로 소폭 미달 — 소표본 유한표본 noise; α=0.1·0.05이 운영 권장점.)

## 4. Exp 4 — Open-set far-OOD (Plan 2, XGBoost base)  `[→ RQ1 · superseded by §20]`
- OOD unseen-diplotype fraction **0.132**; **MSP AUROC 0.695**, FPR@95TPR 0.75; empty-set reject **0**(모든 α).
- marker별 AUROC 추이: 53마커 0.50 → 1,058 0.67 → 3,042 **0.695**(포화).
→ ⚠️ XGBoost가 비-EAS도 과확신 분류 → empty-set 미발화. **base-calibration 한계**(§9·§11에서 RF로 개선).

## 5. Exp 5 — Open-set near-OOD / LOPO (Plan 2, XGBoost base)  `[→ RQ1 · superseded by §11/§20]`
5개 hold-out 집단 모두 reject gap **0**(α=0.10) → ❌ 근연 집단 미검출. base model 과확신.

## 6. Exp 6 — SSL FM core (Plan 3a, synthetic)  `[비-RQ: Paper 2]`
구성: FMVocab, MHMatrixDataset(masked + ADO/dropout contrastive views), objectives(masked CE + NT-Xent), MHTransformer(weight-tied head), heads, pretrain/finetune, `ForensicFMClassifier`.
synthetic: SSL loss 11.47→6.52; finetune acc 0.36→0.92; **ConformalClassifier가 FM을 변경 없이 wrapping**(통합 테스트 통과). **상태 ✅ 코어(72 tests)**. 실데이터 성능은 §10.

---

## 7. Exp 7 — Genome-wide 분류 정확도 / 최소 패널 (leakage-free nested 5-fold)  `[→ RQ3·RQ5 · superseded by §13/§21]`
`results/baseline/genome_wide_accuracy.json` (chance 0.20):
| 패널(top-N) | 정확도 | std |
|---|---|---|
| 10 | 28.6% | 3.8 |
| 20 | 39.5% | 1.9 |
| 50 | 43.5% | 1.3 |
| 100 | 44.8% | 3.9 |
| **200** | **56.9%** | 4.9 |
| 3,042 | 54.4% | 3.4 |
→ ordinal-tree 기준 최소 패널 ≈ 200마커(56.9%). **단, 이 ~57%는 FST 천장이 아니라 ordinal 인코딩 한계였음** — one-hot+LogReg는 동일 데이터로 **79.6%**(§13). "어렵다"는 *모델/인코딩* 문제였지 본질적 난이도가 아니었다.

## 8. Exp 8 — PCA-feature ablation (Chen 2025 동기)  `[→ RQ3]`
`results/baseline/pca_ablation.json`, 동일 5-fold:
| 표현 | 분류기 | 정확도 |
|---|---|---|
| **raw ordinal** | XGBoost | **56.1%** |
| one-hot→SVD(10/20/50/100) | XGB | 32.3 / 38.3 / 27.8 / 28.8 |
| one-hot→SVD(10/20/50/100) | LogReg | 37.3 / 28.6 / 25.4 / **45.8** |
→ **PCA류는 도움 안 됨**(최선 45.8% < raw 56.1%, −10.3p). 경쟁작 이득은 PCA가 아니라 ancestry-최적 AISNP 선별 + ADMIXTURE에서 옴. one-hot MH에 SVD를 씌우면 XGBoost가 raw에서 쓰던 marker별 판별 신호가 소실.

## 9. Exp 9 — Model zoo (full 3,042, 동일 5-fold + conformal/OSR)  `[→ RQ1·RQ3 · superseded by §13]`
> ⚠ **인코딩 confounded**(트리=ordinal, linear/distance=one-hot+StandardScaler). 정식 비교는 §13의 전-모델 one-hot(no scaler) 재실행. 아래는 이력 보존용.
`results/baseline/model_zoo.json`. tree=ordinal 인코딩, linear/distance=one-hot Pipeline.
| 모델 | 정확도 | coverage | set size | **MSP AUROC** |
|---|---|---|---|---|
| **RandomForest** | **0.566** | 0.941 | 3.11 | **0.736** |
| XGBoost | 0.520 | 0.928 | 3.20 | 0.595 |
| LogReg | 0.466 | 0.915 | **1.91** | 0.675 |
| kNN | 0.212 | 0.921 | 3.91 | 0.549 |
| SVM-RBF | 0.254 | 0.921 | 4.11 | 0.656 |
→ **RandomForest가 정확도·OSR 모두 최고** (XGBoost 추월). **OSR(AUROC)이 base 모델에 크게 의존**(0.55~0.74) — open-set 약점이 부분적으로 모델 선택 문제. kNN·SVM은 one-hot 고차원에서 붕괴(curse of dimensionality, 정직한 결과). LogReg는 가장 좁은 set(1.91)=가장 결정적.

## 10. Exp 10 — FM vs XGBoost head-to-head (Plan 3b-core)  `[비-RQ: Paper 2]`
`results/baseline/fm_vs_xgboost.json`, top-256 high-Ae 패널, 70/30 split:
| 모델 | 정확도 | coverage | set size | MSP AUROC |
|---|---|---|---|---|
| XGBoost | 0.316 | 0.862 | 3.80 | 0.591 |
| FM | 0.263 | 0.947 | 4.45 | 0.532 |
→ **FM이 XGBoost에 짐**(소데이터 504로 transformer 과적합; 어댑터는 masked-only SSL). 두 모델 절대치가 낮은 건 **high-Ae 패널이 약하기 때문**(Ae=집단 내 다양성 ≠ 집단 간 FST). FM의 활로 = unlabeled 데이터 확장(all-2504/NYGC) + contrastive 활성화(Plan 3b-extended).

## 11. Exp 11 — Plan 2 재산출: base 모델 교체 (XGBoost → RF → LogReg one-hot)  `[→ RQ1]`
`results/baseline/plan2_rf_vs_xgb.json`, `plan2_logreg.json` — 동일 데이터·split, **base_estimator만 교체** (model-agnostic trust layer):

| 지표 (α=0.1 기준) | XGBoost | RandomForest | **LogReg(one-hot)** |
|---|---|---|---|
| set size | 2.61 | 2.45 | **1.79** (가장 결정적) |
| far-OOD MSP AUROC | 0.695 | 0.803 | **0.849** |
| far-OOD FPR@95TPR | 0.75 | 0.572 | **0.461** |
| empty-set OOD reject @α0.3 | **0** | 0.21 | **0.593** |
| LOPO near-OOD (held-out reject) | 0 (전부) | CHS 0.048 | **CHS 0.162, KHV 0.020** |
| coverage @α0.1 | 0.888 | 0.908 | 0.895 |

→ **LogReg(one-hot)이 trustworthy 전 지표 최고**: 가장 좁은 set(1.79), 최고 OSR(AUROC 0.849, FPR@95 0.461), empty-set OOD 거부 **59%**, near-OOD도 CHS 16%로 가장 잘 검출.
**핵심 결론(정정)**: §4·§5의 OSR 약점은 방법 한계가 아니라 **base model 문제**였고 — 단 RF가 아니라 **LogReg(one-hot)이 정확도(79.6%)·신뢰성 모두에서 최고**. model-agnostic layer에서 base만 교체(한 줄)로 달성. (이전 "RF 권장"은 ordinal 비교 기준 — 철회.)
**상태 ✅ — 권장 base = LogReg(one-hot).** (참고: LogReg는 α=0.05에서 cov 0.967로 보수적, RF는 0.928로 약간 under — 운영 α=0.10 권장.)

## 12. 관련 연구 비교 — Chen et al. 2025 (Human Genomics, 95.6%)
경쟁작의 95.6%는 **직접 비교 불가**: (a) **9개 광역·언어계통 클러스터**(중앙아·시베리아·동남아 포함, FST 높음) 분류, (b) **AISNP 2,000개 + PCA/ADMIXTURE** 피처, (c) Human Origins(array) 1,703명/67집단. 그들도 근연 그룹 sensitivity는 **0.66~0.87**. 우리 과제(1000G EAS-5, low FST, MH)와 입도·마커·데이터가 모두 다름. **차별점**: calibrated UQ/OSR + Reliable-Ae + forensic admissibility(경쟁작 부재).

---

## 13. Exp 13 — 인코딩이 천장을 깬다: one-hot LogReg (leakage-free 5-fold CV)  `[→ RQ3 · canonical 정확도]`
`results/baseline/onehot_cv.json` (OneHotEncoder를 fold 내부에서 fit):
| 설정 | 5-fold CV 정확도 |
|---|---|
| **LogReg + one-hot (no scaler)** | **79.6% ± 3.9** (folds 0.86/0.74/0.78/0.78/0.81) |
| LogReg + one-hot + StandardScaler | 46.6% ± 5.4 |

**전 모델 one-hot(no scaler) 5-fold CV** (`results/baseline/model_zoo_onehot.json`) — 모든 모델을 올바른 인코딩으로 재실행:
| 모델 | one-hot CV | (이전 잘못된 값) |
|---|---|---|
| **LogReg** | **79.6%** | — |
| kNN | 64.9% | 21.2% (scaler) |
| RandomForest | 59.7% | 56.6% (ordinal) |
| XGBoost | 56.9% | 52.0% (ordinal) |
| SVM-RBF | 41.1% | 25.4% |

→ **정밀 결론: "one-hot이 다 고친다"가 아니라 "정규화 *선형*+one-hot이 이긴다"**. 트리는 one-hot으로도 57→60%에 그침(48k 희소 binary × 504 샘플 p≫n에서 axis-aligned split 희석). kNN은 scaler 제거로 21→65% 급등(StandardScaler가 거리모델을 죽였던 게 확정). **권장 base = LogReg(one-hot)** — 정확도(79.6%)·OSR(AUROC 0.85, §11) 모두 최고. ("57% 천장"은 ordinal-tree artifact였고, 정확히는 *고차원 one-hot MH에서 정규화 선형모델 우위*가 결론.)

## 14. Exp 14 — Calibration(ECE) + Deep Ensembles / MC-Dropout UQ  `[→ RQ4]`
`results/baseline/calibration_uq.json` (70/30 split):
| 모델 | acc | ECE | MSP AUROC | epistemic AUROC |
|---|---|---|---|---|
| **LogReg (one-hot)** | 0.730 | 0.230 | **0.841** | — |
| RandomForest | 0.566 | 0.315 | 0.725 | — |
| XGBoost | 0.500 | **0.077** | 0.674 | — |
| DeepEnsemble (5×MLP) | 0.724 | 0.080 | 0.716 | 0.362 |
| MC-Dropout (MLP) | 0.717 | 0.177 | 0.809 | 0.521 |
→ **ECE 정정**: XGBoost가 가장 잘 보정(0.077), **RF가 최악(0.315)** — 이전 "XGBoost 과확신" 주장 폐기. RF의 OSR 우위는 보정이 아니라 확률 *순위* 분리(AUROC) 덕(ECE와 AUROC는 별개). **앙상블 epistemic은 OOD 미검출**(0.36/0.52≈chance) → MSP가 더 나음(정직한 negative). 정확도 정식값은 §13의 CV(79.6%).

## 15. Exp 15 — 열화 DNA(ADO) robustness (RF base, clean 학습→ADO test)  `[→ RQ6]`
`results/baseline/ado_robustness.json`:
| ADO rate | acc | coverage | set | OSR AUROC |
|---|---|---|---|---|
| 0.0 | 0.678 | 0.908 | 2.45 | 0.803 |
| 0.2 | 0.645 | 0.882 | 2.47 | 0.800 |
| 0.3 | 0.579 | 0.882 | 2.43 | 0.768 |
| 0.5 | 0.592 | **0.803** | 2.37 | 0.730 |
→ 우아한 저하나 **ADO 50%에서 coverage 0.91→0.80(보장 깨짐)** — conformal exchangeability 위반(train clean vs test 열화). forensic admissibility에 중요(경쟁작 미탐구). base를 LogReg(one-hot)로 한 ADO 재실험은 follow-up.

## 16. 종합 핵심 발견
1. **천장은 FST가 아니라 모델×인코딩이었다(정정)**: 정밀하게는 **고차원 one-hot MH(p≫n)에서 정규화 *선형*모델 우위** — LogReg(one-hot) **79.6%(CV)**, 트리는 one-hot으로도 57→60%, SVM 41%, FM 26%. 70% 목표를 MH-only로 초과(SNP 확장 불요). 가장 단순한 모델이 정확도·신뢰성 모두 석권(Simplicity Test). 메시지가 "low-acc라 UQ로 보완"에서 "**competitive acc + calibrated UQ/OSR, 게다가 단순함이 이긴다**"로 격상.
2. **권장 base = Logistic Regression(one-hot)**: 정확도(79.6%)·far-OOD OSR(AUROC 0.84) 모두 최고. (이전 "RF 권장"은 ordinal 비교 기준이었음.)
3. **Calibration(ECE) ≠ OSR(AUROC)**: XGBoost 최선 보정·RF 최악, 그러나 OSR은 rank 분리가 좌우. epistemic(앙상블/MC-dropout)은 OOD 미검출 — MSP/empty-set이 실효 신호.
4. **Coverage 보장은 작동하나 분포 변화(ADO)엔 취약** — 열화 시료 robustness가 차별적 forensic 기여.

## 17. 한계
- **ADO(분포 변화)에서 conformal coverage 보장 저하**(50% dropout시 0.80) — exchangeability 위반.
- KOR 데이터 부재(비-EAS+LOPO 대용). Trio phasing 재현 불가(NYGC 필요, Reliable-Ae deferred).
- one-hot LogReg 79.6%는 5-fold CV이나 외부(다른 코호트) 검증 미실시. ADO 재실험은 RF base(LogReg-one-hot 미적용). epistemic UQ는 OOD 미검출.
- 직접 경쟁작(Chen 2025) 존재 — 차별점은 trustworthy 축(calibration/OSR/ADO) + "단순 인코딩이 천장을 깬다"는 방법론 메시지.

## 18. 재현 (commands)
```bash
uv pip install -e ".[dev]" && uv run python -m pytest -q     # 79 tests
bash scripts/data/01_download_1000g.sh && bash scripts/data/06_download_genome_wide.sh
uv run python scripts/baseline/03_run_baseline.py        # chr22 baseline
uv run python scripts/baseline/04_wei2025_phasing.py     # EAS Ae (phasing deferred)
uv run python scripts/trust/10_conformal_curve.py     # coverage (Plan 2)
uv run python scripts/trust/11_openset_ood.py         # far-OOD
uv run python scripts/trust/12_lopo_nearood.py        # near-OOD (LOPO)
uv run python scripts/models/13_pca_ablation.py        # PCA ablation (Exp 8)
uv run python scripts/models/14_fm_vs_xgboost.py       # FM head-to-head (Exp 10)
uv run python scripts/models/15_model_zoo.py           # model zoo (Exp 9)
uv run python scripts/trust/16_plan2_rf.py            # RF vs XGB Plan 2 (Exp 11)
uv run python scripts/trust/17_calibration_uq.py      # ECE + Deep Ensembles + MC-Dropout (Exp 14)
uv run python scripts/trust/18_ado_robustness.py      # ADO degraded-DNA (Exp 15)
uv run python scripts/models/19_onehot_cv.py           # one-hot LogReg 5-fold CV (Exp 13)
```
산출: `results/baseline/*.json`, `results/conformal/*.json`.

## 19. 다음 실험
1. **base = LogReg(one-hot) 채택** (Exp 13 — 79.6%, OSR도 최고). Plan 2 전 지표(coverage/far-OOD/LOPO)를 LogReg(one-hot)로 재산출(§11을 RF→LogReg로 갱신).
2. **외부 검증**: 다른 코호트(HGDP/NYGC EAS)로 one-hot LogReg 79.6% 일반화 확인.
3. **ADO 재실험을 LogReg(one-hot) base로** + clean+ADO 혼합 calibration으로 coverage 회복 시도.
4. **Paper 1 메시지 재정립**: "competitive accuracy(80%, one-hot) + calibrated UQ/OSR + ADO robustness; 단순 인코딩이 복잡 모델을 이긴다."
5. (deferred) FM/contrastive + 데이터 확장(Paper 2), NYGC trio(Reliable-Ae), msp_threshold 운영곡선, 도표화(`academic-plotting`).

---

## 20. base-model이 OSR을 좌우 — 10-seed 통계적 엄밀성 (A)  `[→ RQ1 · canonical ★]`

`scripts/trust/24_trust_rigor.py` → `results/baseline/trust_rigor.json`. 동일 OOD 셋, 10개 train/test split(30% test, stratified, seed 0–9) 반복, α=0.10. mean±std.

| base | far-OOD AUROC | coverage | set size | FPR@95 | empty-set reject_OOD |
|---|---|---|---|---|---|
| **LogReg(one-hot)** | **0.840 ± 0.016** | 0.916 ± 0.025 | 1.72 | 0.461 | 0.123 ± 0.101 |
| RandomForest | 0.757 ± 0.048 | 0.920 ± 0.022 | 2.53 | 0.641 | 0.004 ± 0.006 |
| XGBoost | 0.675 ± 0.038 | 0.924 ± 0.029 | 2.93 | 0.747 | 0.000 ± 0.000 |

**해석:**
- **LogReg vs XGBoost AUROC 격차 = 0.165 ≈ 결합 std(0.016+0.038)의 ~4배** → "base-model이 open-set 분리를 좌우한다"(§11)는 핵심 주장이 통계적으로 확고. 단일 split(0.849)이 cherry-pick이 아님(10-seed mean 0.840±0.016로 일관).
- **coverage는 세 모델 모두 ≥0.90** — conformal 보장은 base-model에 무관하게 유지된다(보장은 base-agnostic, 분리는 base가 좌우 — 2단 메시지).
- LogReg가 set size(1.72, 최소)·AUROC·FPR95 모두 최고. XGBoost의 empty-set reject는 **정확히 0.000±0.000**(절대 발화 안 함) → tree류는 OOD에 과확신.
- → Paper 1 §4.3 핵심 표에 error bar 확보.

## 21. 최소 패널 — one-hot LogReg, leakage-free (B)  `[→ RQ5 · sub-result: univariate 선택 부족 — §23이 대체]`

> **갱신(2026-05-30)**: 아래는 *univariate MI/L1* 선택 기준 결과 — RQ5의 답이 아니라 **sub-result**(약한 선택기로는 compact 패널 부족)로 강등. 다변량 선택으로 패널이 살아남(§23). "compact 패널 없음"이라는 단정은 철회.

`scripts/panel/25_min_panel_logreg.py` → `results/baseline/min_panel_logreg.json`. fold 내부에서 mutual_info_classif(ordinal 행렬)로 marker top-N 선별(leakage-free) → 해당 컬럼만 one-hot LogReg, 5-fold.

| markers | accuracy |
|---|---|
| 10 | 28.2% ± 1.4 |
| 50 | 39.1% ± 4.2 |
| 100 | 49.4% ± 2.0 |
| 200 | 54.8% ± 3.8 |
| 500 | 60.3% ± 4.2 |
| 1000 | 68.0% ± 2.8 |
| **3042 (전체)** | **79.6% ± 3.9** |

**해석 — "최소 패널" 전제의 재해석:**
- **plateau 없음** — 정확도가 marker 수에 거의 단조 증가. ordinal-tree가 200마커/57%에서 평탄해진 것과 정반대(그건 인코딩 아티팩트였음 — 부록 A).
- **70%+ 정확도는 전 패널(3042) 필요** — 70/75/78% threshold를 만족하는 최소 N은 전부 3042. 작은 "최소 패널"로는 고정확도 불가.
- → 제안서 원안의 "최소 MH 패널" 메시지가 인코딩 교정 후엔 **"fine-scale EAS는 genome-wide MH가 필요"**로 바뀜. Paper 1의 forensic 함의는 "minimal panel"이 아니라 "genome-wide MH + calibrated UQ".
- **L1-LogReg 축소 패널도 회복 실패** (`scripts/panel/26_l1_panel_cv.py` → `results/baseline/l1_panel_cv.json`, leakage-free 5-fold one-hot L1):

  | L1 C | markers (mean) | accuracy |
  |---|---|---|
  | 0.1 | 71 | 50.8% ± 1.8 |
  | 0.2 | 268 | 55.0% ± 2.6 |
  | 0.5 | 474 | 56.9% ± 3.8 |

  → **L1·MI 모두 univariate/단순 sparsity** — ~70–470 marker에서 ~57% 이하.
  - **결론(철회 — §23 참조)**: 당시 "compact 패널 없음/genome-wide 필수"로 단정했으나, 이는 **약한 선택기**(개별 마커 평가)의 한계였다. **다변량 model-based 선택**은 같은 N에서 +8~19p 개선(25마커 52%, 1000마커 76.8%) — 신호가 소수 마커에 front-load 가능함을 보임(§23). RQ5는 PENDING으로 재조사.

---

## 22. RQ7 — HGDP 외부 코호트 전이 (in-callset, clean)  `[→ RQ7 · ANSWERED]`

**1KG-EAS 학습 → HGDP-EAS 테스트** (독립 코호트). 3-class 매핑(Han=CHB+CHS, Japanese=JPT, Dai=CDX; KHV는 HGDP 매칭 없음 → 제외). LogReg(one-hot).

### 22.1 In-callset 최종 (gnomAD HGDP+1KG harmonized, hg38, 전체 3,042마커) — `scripts/42`, `rq7_incallset.json`
1KG·HGDP가 **같은 callset(GRCh38)** → liftover/build mismatch 없음.

| 항목 | 값 |
|---|---|
| markers | **3,042 (full, hg38)** |
| train 1KG | 461 (Han 266 / Japanese 102 / Dai 93) |
| test HGDP | 71 (Han 40 / Japanese 26 / Dai 5) |
| **transfer accuracy** | **87.3%** |
| per-class recall | **Han 1.00**(40/40) / Japanese 0.731(19/26) / Dai 0.60(n=5) |
| **unseen diplotype fraction** | **0.024** (cross-build였던 0.43에서 소멸) |
| within-HGDP 3-fold CV | 0.593 ± 0.073 |

confusion(rows=true): Dai[3,2,0]/Han[0,40,0]/Japanese[0,7,19] — Han 완벽, Japanese 7명→Han(JPT↔Han 근연, 타당), Dai n=5 비신뢰.

**해석:**
- **모델이 독립 코호트로 깨끗이 전이됨 (87.3%)** — build mismatch 제거(unseen 43%→2.4%) + 전체 마커로 예비 82.4%에서 **상향**(예측대로 "82.4%는 보수적 하한"이 맞음).
- **transfer(87.3%) > within-HGDP CV(59.3%)** — 1KG 461명 대형 학습셋이 HGDP 71명 자체 학습보다 잘 일반화(강한 generalization 신호).
- 틀리는 방식이 생물학적으로 합리적(Japanese↔Han 근연). 한계: Dai n=5 소표본, KHV(베트남) HGDP 매칭 없음.

→ **RQ7 ANSWERED**: trust layer + LogReg(one-hot)이 독립 코호트로 전이.

### 22.2 (참고) Cross-build 예비 — build mismatch 효과 (scripts/22·23, hg19→hg38, 510마커)

> 아래는 **build mismatch가 있던 예비**(1000G hg19 학습 → HGDP hg38, 5/22 chr). 22.1과 비교하면 build 조화의 가치가 보임(unseen 43% vs 2.4%, acc 82.4 vs 87.3).

| 항목 | 값 |
|---|---|
| common markers | **510 / 3042** (5/22 chroms) |
| train (1000G) | 405 (Han 208 / Japanese 104 / Dai 93) |
| test (HGDP) | 68 (Han 39 / Japanese 25 / Dai 4) |
| **transfer accuracy** | **82.4%** |
| per-class recall | Dai 1.00 (n=4) / Han 0.974 / Japanese 0.56 |
| unseen diplotype fraction | 0.43 (hg19↔hg38 build noise) |
| within-HGDP 3-fold CV | 0.603 ± 0.008 |

confusion: Japanese 25명 중 **11명을 Han으로 오분류**(JPT↔Han 근연 — 생물학적으로 타당); Dai·Han은 거의 완벽 전이(4/4, 38/39).

**해석:**
- **모델이 독립 코호트로 전이됨** — 단 **510/3042 마커(1/6)**·**43% unseen diplotype**(build mismatch)에도 3-class **82.4%**. RQ5(정확도는 marker에 단조 증가)에 비추면 전체 3042 마커 시 **상향** 예상 → 82.4%는 **보수적 하한**.
- **build 조화 견고성**: `handle_unknown=ignore`가 43% unseen을 흡수하고도 82% 유지 — hg19↔hg38 교차에서 one-hot 전략의 설계 이점(우연 아님). transfer(82%) > within-HGDP CV(60%)는 1000G 405명 학습이 HGDP 68명보다 강하기 때문.
- **약점**: Japanese recall 0.56(JPT↔Han 근연 + 마커 부족 → 전체 마커로 개선 여지).
- **한계**: Dai n=4(소표본, 1.0 recall 비신뢰), KHV(베트남) HGDP 매칭 없음, 예비(5/22 chroms).

→ (이 cross-build 예비는 §22.1 in-callset 최종으로 대체됨 — build mismatch 효과 비교용으로 보존.)

---

## 23. RQ5 재조사 — 강한 선택기로 최소 패널 살리기  `[→ RQ5]`

`scripts/panel/27_min_panel_strong.py`(정확도) + `scripts/panel/28_min_panel_trust.py`(trust). §21의 음성결과는 **univariate MI**(각 마커를 독립 평가)에 기인 — p≫n 선형 문제에서 약하기로 유명. **다변량 model-based 선택**(fold-train에 전체 one-hot LogReg fit → 마커별 계수 에너지 Σ_class Σ_onehot w² 로 랭킹, fold 내부 = leakage-free)으로 재검.

### 23.1 정확도 vs 패널 크기 (MI vs model-based, 5-fold)

| N 마커 | MI (univariate) | **coef_l2 (다변량)** | gain |
|---|---|---|---|
| 25 | 32.9% | **52.2%** | **+19.2** |
| 50 | 39.1% | 54.6% | +15.5 |
| 75 | 41.9% | 57.3% | +15.5 |
| 100 | 49.4% | 61.1% | +11.7 |
| 150 | 52.6% | 62.9% | +10.3 |
| 200 | 54.8% | 63.9% | +9.1 |
| 300 | 59.9% | 67.5% | +7.6 |
| 500 | 60.3% | 70.0% | +9.8 |
| 1000 | 68.0% | 76.8% | +8.8 |
| 3042 (전체) | — | **79.6%** | — |

→ **model-based가 모든 N에서 +8~19p**. **25마커 52%**(MI는 동일 정확도에 ~200마커 필요 — **8× 효율**); **1000마커 76.8%** = 전체(79.6%)의 **96%**, 3× 축소. §21의 "plateau 없음/패널 없음"은 **약한 선택기 아티팩트** — 판별 신호는 소수 마커에 **front-load 가능**.
**왜 정당한 교정인가(p-hacking 아님)**: MI는 마커를 *독립적으로* 평가해 조합·중복 제거 신호를 놓친다(univariate 필터의 교과서적 약점). model-based 랭킹은 전체 모델이 실제 가중한 마커를 고른다 → 다변량 신호 보존.

### 23.2 forensic trust frontier (leakage-free 3-way split)

패널 크기별 conformal **coverage + set size + far-OOD AUROC + empty-set reject**로 "배치 가능한 최소 패널"을 정의(top-1 정확도가 아니라 *신뢰성* 기준). 5-seed, α=0.10, model-based 선택.

> **⚠ leakage 발견·수정(C11)**: 마커 선택을 conformal calibration과 **같은 데이터**에서 하면 cal 라벨이 score function에 누수 → coverage가 N↑에 따라 붕괴(0.91→**0.60**, 초기 버그). **3-way split(select / fit+cal / test, 서로 disjoint)**으로 수정 → coverage 회복(아래).

5-seed, α=0.10, model-based 선택:

| N | acc† | **coverage** | set size | far-OOD AUROC | empty-set reject |
|---|---|---|---|---|---|
| 25 | 38.4% | **0.932** | 3.86 | 0.588 | 0.00 |
| 50 | 45.7% | **0.941** | 3.66 | 0.608 | 0.00 |
| 100 | 48.1% | **0.936** | 3.42 | 0.659 | 0.00 |
| 200 | 54.1% | **0.954** | 3.15 | 0.702 | 0.00 |
| 300 | 58.4% | **0.941** | 2.97 | 0.734 | 0.00 |
| 500 | 59.7% | **0.941** | 2.70 | 0.758 | 0.00 |

†acc는 3-way split로 추정기 학습 데이터가 작아(≈176) §23.1(5-fold)보다 낮음 — **정확도 canonical은 §23.1**(25→52%, 500→70%); 여기 acc는 trust와 같은 split의 참고치.

**핵심:**
- **conformal coverage가 모든 패널 크기에서 ≥0.93 유지** — 보장은 패널 크기에 robust(C11 수정 후). **25마커 패널도 valid 90%+ coverage** → 작은 배치 패널에서도 신뢰구간 보장.
- **trade-off는 informativeness(set size)·OOD 분리(AUROC)**: 마커↑ → set 3.86→2.70(더 결정적), AUROC 0.59→0.76(OOD 더 잘 분리). (전체 3042는 set 1.72·AUROC 0.84, §20 — 상한.)
- **minimum forensic panel = 운영 스펙에 따른 최소 N**: 단순 valid coverage면 **25–50마커**; set size ≲3.0 + AUROC ≥0.70이면 **~200–300마커**(10–15× 축소); 최고 결정성·OSR은 전체 패널.
- **고정 배치 패널(deliverable)**: top-50/100/200 마커 리스트를 `results/baseline/min_panel_trust.json`(`fixed_panels`)에 산출 — 예: N=100 패널 상위 mh11HYP-28, mh04HYP-11, mh03HYP-09 ….

### 23.3 RFE 교차검증 (independent confirmation)

`scripts/panel/29_min_panel_rfe.py` → `results/baseline/min_panel_rfe.json`. **마커 단위 RFE**(매 단계 one-hot LogReg 재적합 → 마커별 계수 에너지로 약한 마커 제거 → 다음 target까지 반복, fold 내부 leakage-free) — §23.1의 one-shot 랭킹과 **메커니즘이 다른** 강한 wrapper로 교차검증.

| N | **RFE** (recursive) | one-shot coef (§23.1) | MI (univariate) |
|---|---|---|---|
| 25 | 46.6 ± 4.3 | 52.2 | 32.9 |
| 50 | 54.4 ± 3.3 | 54.6 | 39.1 |
| 100 | 60.9 ± 2.5 | 61.1 | 49.4 |
| 200 | **65.9 ± 4.7** | 63.9 | 54.8 |
| 300 | 67.5 ± 2.9 | 67.5 | 59.9 |
| 500 | 70.4 ± 3.0 | 70.0 | 60.3 |
| 1000 | 75.6 ± 3.9 | 76.8 | 68.0 |

→ **RFE ≈ one-shot coef** (N≥50에서 noise 내 일치) → 두 독립 다변량 선택기가 서로 확증, rescue가 선택기 아티팩트 아님. 둘 다 **MI를 모든 N에서 10–20p 압도** → §21 음성은 *univariate 한계*임을 두 번째 독립 증거로 확정. (N=25에선 RFE 46.6 < one-shot 52.2 — greedy backward의 극소 패널 over-prune, 정직한 관찰; 여전히 MI보다 +14p.) **부수 결론**: RFE가 one-shot을 못 이기므로 값싼 one-shot 선택으로 충분.

**RQ5 → ANSWERED (재정의)**: 배치 가능한 **최소 패널이 존재**하며, 정확도(§23.1, **RFE 교차검증 §23.3**)–coverage–set size–OSR(§23.2) frontier로 특성화됨. 원안의 "소수 마커 ≥90% 정확도"는 도달 불가(정직)지만, **trustworthy 최소 패널**(coverage 보장 + 운영점 선택, 10–15× 축소)은 충족. univariate 선택의 음성(§21)은 약한 도구 탓이었음.

---

## 24. 확장 model zoo — DL 아키텍처 + baseline (RQ1·RQ3)  `[→ RQ1·RQ3]`

기존 비교가 불공정했음(고전=5-fold one-hot §13, DL=다른 split/패널 §10·§14). **모든 모델을 동일 프로토콜**(genome-wide 3,042, leakage-free 5-fold, acc + far-OOD MSP AUROC)로 재실행. DL은 popgen-DL 문헌 기반 다양한 계열. `scripts/30`(고전+MLP, CPU) + `scripts/31`(GPU torch, `fm/architectures.py`).

### 24.1 고전 baseline + MLP (one-hot, sklearn) — `model_zoo_dl.json`
| 모델 | accuracy | far-OOD AUROC |
|---|---|---|
| **LogReg(one-hot)** | **79.6 ± 3.9** | **0.863 ± 0.026** |
| BernoulliNB | 66.9 ± 5.7 | 0.678 |
| RandomForest | 60.3 ± 4.1 | 0.688 |
| ExtraTrees | 59.7 ± 4.3 | 0.694 |
| XGBoost | 56.9 ± 1.5 | 0.567 |
| kNN | 56.8 ± 2.5 | 0.679 |
| MLP-1 (256) | 52.0 ± 8.6 | 0.802 |
| MLP-2 (256,128) | 56.0 ± 8.6 | 0.822 |
| MLP-deep (512,256,128) | 50.4 ± 10.4 | 0.804 |

### 24.2 다양한 DL 아키텍처 (popgen-DL 문헌 기반, GPU) — `dl_architectures.json`
| 아키텍처 (계열) | accuracy | far-OOD AUROC |
|---|---|---|
| EmbMLP (Diet-Networks) | 29.7 ± 6.0 | 0.543 |
| CNN1D (popgen-CNN) | 34.5 ± 7.4 | 0.439 |
| SupAE (Neural-ADMIXTURE / popVAE) | 32.5 ± 5.5 | 0.541 |
| Transformer (supervised) | 51.0 ± 4.7 | 0.575 |
| **Transformer (SSL+finetune = our FM)** | **54.6 ± 6.0** | 0.576 |
| ResNet-tabular (Gorishniy 2021) | 33.5 ± 5.5 | 0.475 |
| ResCNN (deep residual popgen-CNN) | 22.8 ± 6.4 | 0.541 |

(추가 `scripts/39`: **ResNet-tabular**(FT-Transformer의 co-SOTA)·**깊은 residual CNN**도 33.5/22.8% — 완패. embedding+pooling 기반 DL(EmbMLP·SupAE·ResNet-tab·ResCNN)이 22–34%로, one-hot sklearn MLP(50–56%)보다도 낮음 → §24.3의 "embedding bottleneck이 손해" 재확인: *표현이 아키텍처 깊이보다 중요*.)

### 24.4 Linear family + 정규화 sweep (RQ3 — 선형 *클래스* 우위) — `extended_zoo.json`
"LogReg 운빨인가, 선형 클래스인가?" `scripts/35`, one-hot 5-fold.

| 모델 | accuracy | far-OOD AUROC |
|---|---|---|
| LogReg L2 (C=1, ref) | 79.6 ± 3.9 | 0.863 |
| LogReg L2 (C=0.1) | 78.6 ± 4.0 | 0.862 |
| LogReg L2 (C=10) | 76.6 ± 4.4 | **0.898** |
| **LinearSVC (calibrated)** | **79.8 ± 3.7** | **0.957** |
| SGD-log | 69.3 ± 5.4 | 0.694 |
| ComplementNB | 75.2 ± 5.0 | 0.647 |
| LogReg L1 (sparse) | 56.9 ± 3.8 | 0.657 |
| LogReg elastic-net | 62.5 ± 5.4 | 0.687 |

→ **선형 클래스 우위 확정(LogReg 운빨 아님)**: hinge-loss **LinearSVC 79.8% ≈ logistic LogReg 79.6%** — 손실함수가 달라도 dense 선형+one-hot이 ~80%. **LinearSVC가 OSR AUROC 0.957로 전 모델 최고**(margin 기반 분리). **L1/elastic-net(sparse)은 급락**(56.9/62.5) — RQ5(sparse 선택 실패)와 일관, dense L2가 핵심. 정규화: C=1이 정확도 sweet spot, C↑는 OSR 소폭↑.

**확장 선형 패밀리** (`scripts/43`, `linear_family.json`) — distinct 결정규칙 6종으로 "선형 클래스" 재확인:

| 선형 모델 (결정규칙) | accuracy | far-OOD AUROC |
|---|---|---|
| **Multinomial NB** (log-linear) | **82.2 ± 2.9** | 0.654 |
| **Passive-Aggressive** (hinge, online) | 80.2 ± 3.2 | **0.963** |
| Ridge (squared loss) | 79.2 ± 3.2 | 0.774 |
| Nearest-Centroid (Euclidean) | 78.6 ± 4.8 | (no proba) |
| Perceptron (mistake-driven) | 73.8 ± 3.4 | 0.843 |
| **LDA (Fisher 판별)** | **INFEASIBLE** (MemoryError) | — |

→ **선형 패밀리 전체가 73–82%** — 6개 distinct 선형 모델이 전부 트리·DL(≤62%)을 압도 → "선형 *클래스*가 이긴다"가 방탄. 추가 발견: (1) **MultinomialNB 82.2%가 최고 정확도**(희소 one-hot이 count-like → log-linear NB에 적합). (2) **margin 기반(Passive-Aggressive 0.963·LinearSVC 0.957)이 OSR 최고** — trust layer엔 이쪽이 LogReg(0.863)보다 우위. (3) **LDA(고전 Fisher)는 적용 불가** — 48k one-hot의 48k×48k 공분산(~9GB) MemoryError → *popgen 전통 선형(LDA)이 p≫n one-hot엔 부적용이고 **정규화 선형(LogReg/LinearSVC/Ridge)이 실용적 대체***. 즉 "선형 우위"는 **정규화** 선형에 한함(비정규화 LDA는 차원의 저주로 탈락).

### 24.5 Native-cat 트리 + tabular-DL SOTA + small-data SOTA (RQ3 capstone)

**Native-categorical 트리** (`scripts/36`→HGBDT capped, 5-fold): 트리가 *네이티브 범주형*(subset split, ordinal-threshold 아님)으로 처리해도 —

| 트리 인코딩 | accuracy |
|---|---|
| ordinal (§13) | 56.9% |
| one-hot (§24.1, RF) | 59.7% |
| **native-categorical (HGBDT)** | **59.5% ± 3.7** (AUROC 0.664) |

→ **세 인코딩 모두 트리 ~60% 천장 → 트리 열세는 *인코딩*이 아니라 *모델 클래스*.** ("트리에 더 좋은 인코딩을 주면 된다" 기각.) CatBoost는 3,042 cat feature에서 비현실적으로 느려 종료 — 그 자체로 트리류가 이 표현에 부적합한 방증.

**Tabular-DL SOTA** (`scripts/37`, GPU, 5-fold): FT-Transformer **30.1 ± 5.7**(AUROC 0.569) · TabNet **23.6 ± 3.2**(AUROC 0.505, chance 20% 근처) → SOTA tabular-DL도 완패 → "generic DL이 DL을 과소평가" 가능성 차단.

**Small-data SOTA: TabPFN** (`scripts/38`, cloud, top-200 패널):

| 모델 | accuracy | far-OOD AUROC |
|---|---|---|
| **TabPFN@200** (소표본 전용 SOTA) | **62.5 ± 1.9** | 0.649 |
| LogReg(one-hot)@200 (ref, §23.1) | 63.9 | — |
| LogReg(one-hot)@full (ref) | **79.6** | 0.863 |

→ **소표본을 위해 설계된 TabPFN조차 LogReg@200을 못 이김**(62.5 vs 63.9); full 3,042 패널은 feature 한도로 접근 불가(선형이 크게 앞서는 영역). encoding = capped ordinal codes(클라이언트가 categorical flag 미지원 — 약한 ordinal handicap).
> **reproducibility 주의**: TabPFN은 **클라우드 추론** — 서버 모델 갱신 시 수치 변동 가능. **사용 버전 = `tabpfn-client 0.3.0`, server_model="auto", n_estimators=8 (2026-05-30)**. 인증은 `.env`의 `TABPFN_TOKEN`(gitignored).

**§24 capstone (RQ3 ironclad)**: classical ML · 트리(ordinal·one-hot·native-categorical 어느 것이든) · generic DL(MLP·CNN·AE·transformer) · tabular-DL SOTA(FT-Transformer·TabNet) · **small-data SOTA(TabPFN)** — **그 무엇도 dense 선형(one-hot LogReg/LinearSVC ~80%)을 못 이김.** 79.6%는 인코딩 운이 아니라 *모델 클래스* 사실(고차원 희소 categorical p≫n에서 dense 선형 우위).

### 24.3 결론
1. **단순함이 이긴다 (RQ3) — DL 전반에서 확정**: LogReg(one-hot) 79.6%가 5개 DL 계열(embedding/CNN/autoencoder/transformer)을 전부 **≥25p** 압도. 최고 DL은 Transformer+SSL 54.6%. n=504·p≫n에서 DL은 과적합(분산 ±5–10, embedding/AE는 chance 20% 근처). "복잡 모델·피처 이전에 인코딩+단순 선형"이 fine-scale MH ancestry의 결론.
2. **SSL pretraining이 (작게) 돕는다**: Transformer supervised 51.0 → **SSL+ft 54.6 (+3.6p)** — n=504에서도 양의 신호(§10의 256-마커 결과와 달리 full-panel·동일 프로토콜에서 처음 확인). **→ 데이터 확장(현재 1000G 2,504·gnomAD HGDP+1KG 4,091 추출 중)으로 lift가 커질 가설을 직접 동기화**(Paper 2).
3. **DL은 OSR도 약하다 (RQ1)**: 모든 DL far-OOD AUROC 0.44–0.58 ≪ LogReg 0.863; CNN은 0.439(<chance). 흥미롭게 **sklearn MLP(one-hot, AUROC 0.80–0.82) > torch embedding-DL(0.54)** — embedding bottleneck이 unseen-diplotype OOD 단서를 버리는 반면 one-hot+`handle_unknown=ignore`는 보존하기 때문. RQ1의 "OSR은 base/표현이 좌우"를 표현 레벨에서 재확인.

4. **선형 *클래스*가 이긴다, LogReg 특정 아님 (RQ3 보강, §24.4)**: LinearSVC 79.8% ≈ LogReg 79.6%(손실함수 무관), 게다가 **LinearSVC OSR AUROC 0.957로 전 모델 최고**. sparse 선형(L1/elastic)은 급락 → dense L2 선형+one-hot이 정확도·신뢰성의 핵심. (트리는 native-categorical도 ~60%, tabular-DL SOTA·TabPFN도 전부 선형에 짐 — §24.5 capstone.)

**문헌 [verify — 제출 전 서지 확인]**: Romero et al. 2017 (Diet Networks, ICLR); Flagel et al. 2019 (CNN popgen inference, MBE); Gower et al. (genomatnn); Mantes et al. 2023 (Neural ADMIXTURE, Nat Comput Sci); Battey et al. 2021 (popVAE); Korfmann et al. 2023 (DL in popgen, review); Gorishniy et al. 2021 (FT-Transformer, NeurIPS); Arik & Pfister 2021 (TabNet, AAAI). 상세 계보는 docs/02 §8.

---

## 25. SSL foundation model — 데이터 확장 (Paper 2)  `[비-RQ · ANSWERED-negative]`

SSL FM 병목은 데이터 규모(n=504, §24.2). 공개 풀 추출(`scripts/32·33`) 후 **통제 ablation 2건**(양 GB10 노드 **병렬**: NFS 스테이징 + sparkq `--node 2`): pretrain 풀만 {none(supervised)/EAS-only/all} 변경, 동일 마커·finetune·contrastive.

### 25.1 깨끗한 gnomAD ablation (4,091, hg38, 전체 3,042마커, EAS 583) — `scripts/41`, `ssl_gnomad.json`
1000G(hg19)과 **동일 마커셋**이라 confound 없음. LogReg baseline 포함:

| 조건 | accuracy | far-OOD AUROC |
|---|---|---|
| **LogReg(one-hot)** | **78.0 ± 1.9** | **0.921** |
| supervised transformer (no pretrain) | 55.4 ± 3.4 | 0.575 |
| SSL @EAS-only (583) | 48.7 ± 3.9 | 0.536 |
| SSL @all-4091 | 54.0 ± 5.2 | 0.574 |

### 25.2 1000G ablation (2,504, hg19, ~2.7k마커, EAS 504) — `scripts/40`, `ssl_ablation.json`
| 조건 | accuracy | far-OOD AUROC |
|---|---|---|
| **LogReg(one-hot)** *(same-data 앵커)* | **79.4 ± 2.6** | 0.825 |
| supervised (no pretrain) | 47.2 ± 6.0 | 0.525 |
| SSL @504 EAS | 42.9 ± 2.3 | 0.537 |
| SSL @2504 all | 43.6 ± 7.0 | 0.387 |

→ §25.1(gnomAD)과 동일 패턴이 hg19에서도: **LogReg 79.4% ≫ transformer/SSL 43–47%**. (LogReg는 같은 hg19 EAS-504, one-hot 5-fold.)

### 25.3 결론 (Paper 2 — 명확한 negative)
1. **LogReg(78.0%)가 모든 transformer/SSL(48–55%)을 압도** — hg38 clean·동일 마커에서도 (gnomAD LogReg 78.0 ≈ 우리 hg19 79.6 → cohort/build 무관, RQ1·RQ3 견고).
2. **SSL pretraining이 supervised를 못 이김** (55.4 ≥ 54.0 ≥ 48.7; 1000G도 47.2 > 43) — 작은 풀에선 오히려 해침.
3. **데이터 scaling은 SSL을 돕지만(monotone: 583→4091, 48.7→54.0, +5.3) "supervised 수준 회복"에 그침** — LogReg와 24p 격차. 외삽 시 78% 도달엔 수십~수백× 데이터 필요(비현실적).
4. **확정**: fine-scale forensic MH에서 **SSL FM은 단순 선형을 못 이긴다 — 병목은 데이터만이 아니라 모델 클래스**. (1차 confounded 44.6%[scripts/34]는 이 통제 ablation으로 정리됨: SSL ≈ supervised ≈ 54%, 둘 다 LogReg 78%에 못 미침.) → Paper 2를 "데이터로 FM 키우기"에서 "왜 이 regime에서 선형이 근본 우월한가"로 재정의.

> **인프라 메모**: Node 2(별도 머신 user nvidia)에서 실행 = 공유 NFS(`/mnt/nfs/ssd1`)에 코드+데이터 스테이징(chmod 777) → `uv run`이 venv 자동생성 → 출력을 NFS에 redirect해 Node 1에서 가시화(sparkq cancel/log는 cross-node 불가).

---

## 26. 왜 선형이 이기는가 — 메커니즘 + 선행연구 (Discussion)

§24–25에서 정규화 선형 패밀리(LogReg·LinearSVC·Ridge·MultinomialNB·Passive-Aggressive, 73–82%)가 트리(≤60%)·DL·tabular-SOTA·TabPFN(≤62%)을 전부 압도했다. **왜인가?** 세 가지 독립 근거가 수렴하며, 각각 선행연구로 뒷받침된다. (서지는 web/Semantic Scholar 검증; Paper 1 Discussion 토대.)

### 26.1 생물학 — ancestry는 대립유전자 빈도의 *가법적* 신호
집단 구조는 본질적으로 **다수 좌위에 걸친 대립유전자 빈도 차이의 합**으로 인코딩된다. 집단유전학의 표준 모형이 전부 이 *선형/빈도-likelihood* 구조다:
- **STRUCTURE** (Pritchard, Stephens & Donnelly 2000, *Genetics* 155:945) · **ADMIXTURE** (Alexander, Novembre & Lange 2009, *Genome Res* 19:1655) — 개인을 집단별 대립유전자 빈도의 혼합으로 모형화하는 likelihood 모델.
- **PCA/eigenanalysis** (Patterson, Price & Reich 2006, *PLoS Genet* 2:e190; Price et al. 2006, *Nat Genet* 38:904) 와 그 해석(Novembre & Stephens 2008, *Nat Genet* 40:646, doi:10.1038/ng.139) — 집단 구조 = 대립유전자 빈도 공분산의 **선형** 고유분해; PC = 빈도 gradient.

→ **함의**: one-hot diplotype 위 log-linear 모델의 `w·x = Σ(마커별 빈도 기여)`는 위 생성 모형의 **지도학습 대응물**이다 — "충분히 좋은" 게 아니라 데이터 생성 구조와 *일치*하는 모델 클래스.
**우리 증거**: RQ5(§23) — 정확도가 마커 수에 단조 증가, compact subset 없음 = 신호가 수천 좌위에 *확산된 가법* 구조. §8 — PCA/SVD 피처 무익(선형 위 추가 비선형 구조 없음).

### 26.2 forensic ancestry의 *native* 모델 = naive-Bayes/log-linear
법과학 ancestry의 표준 도구 **Snipper** (Phillips et al. 2007, *FSI:Genetics* 1(3–4):273–280, doi:10.1016/j.fsigen.2007.06.008)는 AIM-SNP 빈도 위 **naive Bayes 분류기**다. naive Bayes = 좌위 독립 가정의 **log-linear allele-frequency 분류기** — 우리 **MultinomialNB**와 같은 클래스.
→ **우리 증거**: MultinomialNB가 **82.2%로 최고 정확도**(LogReg 79.6·LinearSVC 79.8보다 위, §24.4) — *수십 년 forensic 관행의 빈도-기반 naive-Bayes 모델이 이 과제에서 near-optimal*임을 정량 재확인. (Snipper vs 현대 ML 비교: Heinzel et al. 2025, *Advancing biogeographical ancestry predictions through machine learning*, FSI:G 79:103290, doi:10.1016/j.fsigen.2025.103290 — TabPFN 등과 대조.)

### 26.3 통계 — p≫n + tabular inductive bias가 트리·DL을 탈락시킴
504 샘플 vs ~48k one-hot 피처 = 심한 p≫n.
- **DL은 tabular에서 단순 모델을 못 이김** (Shwartz-Ziv & Armon 2022, *Information Fusion* 81:84–90, arXiv:2106.03253; Gorishniy et al. 2021, *NeurIPS*) — 특히 소데이터 과적합. 우리 DL 분산 ±8–10이 증거.
- **inductive bias 미스매치** (Grinsztajn, Oyallon & Varoquaux 2022, *NeurIPS* D&B, arXiv:2207.08815): 트리는 *불규칙·비평활* 함수에 유리, NN은 평활/회전-불변 bias — **확산된 가법 신호**는 둘 다와 안 맞고 **선형**과 맞음. 트리의 axis-aligned split은 "수천 마커의 미세 가법 기여 합"엔 마커당 split이 필요 → n=504 과적합(§24.5: native-cat도 ~60% 천장).
- **p≫n SNP는 전용 설계 필요** (Romero et al. 2017, Diet Networks, *ICLR*) — DL조차 이 regime용으로 파라미터를 줄여야 함. 우리 결과: 그 전용 DL(EmbMLP)·소표본 SOTA(TabPFN 62.5%)조차 선형에 못 미침.
- **비정규화 LDA 탈락** (§24.4): 48k×48k 공분산 → MemoryError. 고전 Fisher 판별이 공분산을 *명시* 모형화하려다 p≫n에서 죽음 → "선형 우위 = **정규화**(공분산 비명시) 선형".

### 26.4 종합 원리 + 검출 한계
**fine-scale ancestry = (생물학적) 가법 빈도 신호 → 정규화 log-linear(=allele-frequency 분류기)가 native·near-optimal → p≫n에서 그것만 일반화; 트리(axis-aligned)·DL(과적합·평활 bias)·LDA(공분산 폭발)는 각자 다른 방식으로 탈락.**
저-FST의 어려움도 선행과 일치: **phase-change** (Patterson et al. 2006) — 집단 분기가 임계 이하면 검출 불가, 약간 위면 쉬움. 동아시아 5집단은 임계 부근이라 *많은 마커*가 필요(=RQ5 확산), 그 다수 약신호 합산엔 **선형/naive-Bayes가 최적 도구**.

> **메시지(Paper 1 Discussion)**: "단순 선형이 이긴다"는 우연·운빨이 아니라, *집단유전학의 생성 모형(가법 빈도)*과 *p≫n 통계*가 함께 강제하는 결과다 — forensic 관행(Snipper=naive Bayes)과도 정합. 복잡한 모델은 이 regime에서 inductive bias가 어긋나거나 과적합한다.

**참고문헌(전부 검증됨, DOI/arXiv 확보)**: Pritchard et al. 2000 *Genetics* 155:945 · Patterson, Price & Reich 2006 *PLoS Genet* 2:e190 · Price et al. 2006 *Nat Genet* 38:904 · Novembre & Stephens 2008 *Nat Genet* 40:646 (doi:10.1038/ng.139) · Alexander, Novembre & Lange 2009 *Genome Res* 19:1655 · Phillips et al. 2007 (Snipper, *FSI:G* 1:273–280, doi:10.1016/j.fsigen.2007.06.008) · Romero et al. 2017 (Diet Networks, *ICLR*) · Gorishniy et al. 2021 (*NeurIPS*) · Shwartz-Ziv & Armon 2022 *Inf. Fusion* 81:84–90 (arXiv:2106.03253) · Grinsztajn, Oyallon & Varoquaux 2022 (*NeurIPS* D&B, arXiv:2207.08815) · Heinzel et al. 2025 (*Advancing biogeographical ancestry predictions through ML*, FSI:G 79:103290, doi:10.1016/j.fsigen.2025.103290).

---

## 27. Embedding을 *잘 만든* 접근 (Paper 2 후속, §24.2 보강)  `[비-RQ]`

§24.2에서 **naive 학습 embedding(EmbMLP 29.7%)**이 one-hot 선형(79.6%)·심지어 one-hot MLP(50%)보다 낮았다. 문헌(docs/02 §9)이 가리키는 *제대로 설계된* embedding 3종을 동일 프로토콜(genome-wide, 5-fold, acc+far-OOD AUROC)로 검정. **양 GB10 노드 병렬**(Node1 DietNet/NT, Node2 RandEffEmb via NFS).

| embedding 접근 (출처) | accuracy | far-OOD AUROC |
|---|---|---|
| LogReg(one-hot) *기준* | **79.6** | 0.863 |
| **DietNet** (Romero 2017, weight-prediction; `scripts/models/44`) | **73.6 ± 2.3** | 0.678 |
| **Random-effects emb** (Simchoni-Rosset 2021, per-marker shrinkage; `scripts/models/45`) | **72.6 ± 1.7** | 0.683 |
| **NT-transfer** (Dalla-Torre 2024, frozen seq FM; `scripts/models/46`) | **59.9 ± 2.4** | 0.564 |
| naive EmbMLP (§24.2, 참고) | 29.7 | 0.543 |
| Transformer+SSL (§24.2, 참고) | 54.6 | 0.576 |

**핵심 발견:**
- **p≫n-aware embedding은 작동한다 (~73%)**: DietNet(73.6)·random-effects(72.6) 둘 다 naive EmbMLP(29.7)를 **+43p** 압도하고 LogReg(79.6)에 **6–7p까지 근접** — 다른 모든 DL/SOTA(≤62%)를 넘어선다. → §24.2의 "embedding DL 실패"는 *embedding 자체*가 아니라 *naive 학습 방식*(n=504로 임베딩 자유 학습)의 문제였음.
- **메커니즘 일치**: DietNet은 feature를 프로파일로 임베딩+보조망이 가중치 예측(p≫n 파라미터 절약), random-effects는 per-marker Gaussian shrinkage — 둘 다 *embedding에 강한 inductive bias/정규화 부여*해 소표본 과적합 방지. §26의 "정규화가 핵심"과 정합.
- **단, 선형이 여전히 약간 위(79.6 > 73.6)**: 가법 빈도 신호엔 직접 정규화 선형이 최적(§26)이고, embedding은 *거의* 따라잡되 못 넘음.

- **NT-transfer 59.9%**: frozen 서열 FM 임베딩은 naive(29.7)·SSL(54.6)은 넘되 DietNet(73.6)·LogReg(79.6)엔 못 미침 → *서열 내용은 잡으나 ancestry의 빈도 신호는 놓침*(예상대로). far-OOD AUROC 0.564로 약함.

> **함의(Paper 2)**: FM/embedding을 살리려면 (a) **p≫n-aware 설계**(Diet-Net/random-effects, 이미 72–74%)가 (b) **외부 서열 transfer**(NT 59.9%)보다 낫고, *naive end-to-end 임베딩 학습(29.7%)은 막다른 길*. 정규화/inductive bias가 결정 변수.

### 27.1 CREE 검증 (Conformal Random-Effects Embeddings, docs/07)
§27의 random-effects embedding을 trust/transfer로 확장 — CREE method 제안(docs/07)의 검증.
- **(b) embedding + conformal** (`scripts/trust/47`): RandEff를 Mondrian conformal로 wrapping → **coverage 0.920 유지**(trust layer는 embedding-DL에도 model-agnostic), set 1.91; 단 far-OOD MSP-AUROC 0.649(LogReg 0.840보다 약) → embedding의 MSP open-set 약함 확인 → (d)가 *내재 분산*으로 개선.
- **★(d) variance-as-nonconformity** (`scripts/trust/49`): 변분 random-effects embedding의 per-sample posterior 분산을 open-set 점수로 → **open-set AUROC 0.946 ± 0.010 vs 같은 모델 MSP 0.676** (acc 72.8%). **CREE 핵심 주장 성립**: 내재 분산이 MSP를 압도하고 최고 선형(LinearSVC 0.957)에 근접 — *불확실성이 표현에 내재*(RQ4의 "MSP/앙상블은 OOD와 decouple"과 대비되는 새 능력).
- **(e) 극소표본** (`scripts/models/50`): n=50/100/150/200/full에서 **LogReg가 전 구간 +8~10p 우위**(RandEff 45.5→70.9 vs LogReg 55.5→79.3). → **negative**: shrinkage embedding은 소표본에서도 선형을 못 이김. CREE 셀링 포인트를 *정확도/small-n*이 아니라 **내재 open-set 신뢰성(d)**으로 좁힘.
- **(c) SSL + random-effects encoder** (`scripts/ssl/48`, gnomAD 4091): **acc 52.7 ± 3.5%, far-OOD MSP-AUROC 0.632** — naive SSL 54.0 / supervised 55.4 / LogReg 78.0과 사실상 동일 → **negative**: random-effects 수축을 *deep SSL transformer*의 embedding에 붙여도 SSL 한계(§25)를 못 넘는다. 같은 수축이 *얕은* RandEff embedding(§27, 72.6%)에서는 +43p였던 것과 대비 → **정규화의 이득은 그것이 지배적 inductive bias일 때만 발현**(transformer attention 층의 별도 과적합이 묻어버림). CREE의 검증된 형태 = *얕은 변분 random-effects + variance-as-nonconformity(d)*, deep-SSL 경로 아님.

- **★(f) cross-cohort transfer** (`scripts/rq7/52`, 1KG→HGDP zero-shot, 동일 hg38 패널): VRE를 **1000 Genomes EAS-5**(n=583)로 학습 → **HGDP 코호트**(미관측·다른 플랫폼/샘플링, 18개 novel EA 집단 Han/Japanese/Dai/Yakut/Uygur…)로 zero-shot 전이. 학습 cohort acc 75.8%(§22 LogReg 78%에 근접). open-set:
  - within-cohort(1KG-EAS vs 1KG-nonEAS): **var 1.000 vs MSP 0.721**
  - **CROSS-cohort(HGDP-EAS in vs HGDP-OOD): var 0.999 vs MSP 0.633** → 내재 분산은 코호트 경계를 넘어 거의 그대로 전이, **MSP는 0.721→0.633으로 −0.09 붕괴**. *intrinsic uncertainty는 cohort-shift에 강건, post-hoc softmax confidence는 아님*.
  - **메커니즘 검증(반박 선제 차단)**: "그냥 rare/OTHER-slot 세는 것 아니냐"? **무학습 OTHER-fraction AUROC = within 0.002 / cross 0.063** (오히려 EAS in-dist가 OTHER를 더 씀 0.320 vs 0.248 — 전역 vocab이 큰 non-EAS 집단에 지배됨). → 0.999는 빈도 artifact가 **아니라** EAS-5 학습이 빚은 *학습된 친숙도(epistemic)* 신호.
  - **정직한 caveat**: cross-continental **far-OOD** + 3042마커 평균화로 절대값이 ≈1.0까지 부풀려짐 → near-OOD 완벽 분리 주장 아님. 기여 = **var↔MSP 격차 + 코호트 강건성**(절대값 아님).

**CREE 종합 평결**: capability **성립 = 내재 분산 open-set(d, 0.946 vs MSP 0.676)** + model-agnostic conformal coverage(b, 0.920) + **cross-cohort 전이(f, var 0.999 vs MSP 0.633, 코호트 강건)** + **near-OOD(§27.2, 0.782 최고)**. honest negative = **정확도/small-n(e)·대규모 SSL(c) 모두 선형 우위 못 깸**(§26). top-tier 스토리: "embedding이 정확도론 선형을 못 이기나, *신뢰성(open-set)에선 내재 불확실성으로 post-hoc MSP를 압도하고 코호트를 넘어 전이*한다."

### 27.2 near-OOD — 법과학 최대 난제 + CREE 메커니즘 교정  `[→ RQ1 · CREE]`
`scripts/trust/54_near_ood.py`(5-seed)·`scripts/trust/55_cree_mechanism.py`(3-seed). 미수록·근연 집단(예: KOR) 과확신 오분류는 trustworthy 프레이밍의 핵심 위험. KOR 데이터 부재를 **gnomAD HGDP의 15개 미수록 EAS 집단**(Yakut·Uygur·Mongola·Oroqen·Cambodian·Lahu·Yi·Miao·She·Naxi·Tu·Tujia·Hezhen·Daur·Xibo, n=153)으로 *데이터-기반 near-OOD proxy* 전환. in-dist = held-out 1KG-EAS-5; far-OOD = 비-EAS(n=692). 동일 callset/마커.

**near-OOD AUROC (in-dist vs 미수록 HGDP-EAS) | far-OOD = 난이도 gradient:**
| 점수 | near-OOD | far-OOD |
|---|---|---|
| **★ CREE variance** | **0.782 ± 0.005** | 1.000 |
| Mahalanobis(emb) | 0.697 | 0.850 |
| Mahalanobis(logit) | 0.687 | 0.816 |
| kNN(emb) | 0.675 | 0.846 |
| MSP(baseline) | 0.670 | 0.889 |
| set-size | 0.480 | 0.421 |

- **near-OOD가 본질적으로 어려움 실증**: 모든 점수 far(0.82–1.0)→near(0.67–0.78) 대폭 하락. **CREE variance(0.782)가 유일하게 의미 있는 분리** — MSP·거리·set-size를 8–11p 압도, 최안정(±0.005).
- **거리 기반(#2) = 약한 양성**: Mahalanobis-emb 0.697(MSP +0.027), kNN 0.675 ≈ MSP. 포화 MSP보다 소폭, CREE엔 한참 못 미침.
- **set-size soft-reject(#3) = 음성**: AUROC 0.480(chance 이하). Pareto: τ=2 near-reject 0.47인데 **in-dist false-reject 0.49**(무작위보다 나쁨), τ≥3 전부 0(과확신 base라 set 안 넓어짐). set-size는 near-OOD 신호 못 담음.

**메커니즘 (§55, 가설 교정)**: "random-effects 수축(KL)이 variance OOD를 만든다"는 **실측 반증** — KL ablation에서 **KL=0도 near-OOD 0.772**(KL=2.0에서 0.792, +0.02 marginal 증폭에 그침). 진짜 원천 = **변분 학습**: Spearman(학습빈도, per-slot logvar) = **−0.992**(자주 본 코드일수록 분산↓ = 학습된 친숙도), **OTHER-slot 제외해도 0.78–0.80 유지**(단순 rarity 아님). → random-effects는 *정확도* 축 기여(§27)이고 *OOD 신호*의 원인은 변분 분산.

**boost 시도 (`scripts/trust/56`, held-out *population* 평가 = 미관측 근연 집단 일반화, KL=2.0)**: var_single 0.804 baseline에서 —
- **score fusion = modest 양성**: {CREE-variance, relative-Maha, MSP} 단순 **z-평균 0.822**(+0.018), 학습 combiner 0.825(+0.021) — 거의 동일 → *감독 불필요, 단순 z-fusion으로 충분*. **fit-pops로 튜닝 후 미관측 test-pops에서 0.825 유지** → "다른 근연 집단으로 탐지기 튜닝 → KOR-류 미관측 집단 배치" 경로 *부분 검증*.
- **negative**: 5-VRE **앙상블 무효**(+0.001, variance estimator가 이미 ±0.005로 안정); **거리·Relative-Mahalanobis**(Ren 2021) 0.68로 variance 한참 아래 → near-OOD 전용 설계도 CREE variance 못 이김. ⇒ **variance가 핵심 신호 재확정**.

> **종합**: near-OOD는 현 프로젝트 **최대 난제로 확정**(far와 큰 격차). CREE variance가 핵심 레버(0.78–0.80)이고 **단순 z-fusion으로 ~0.82까지**(미관측 집단 전이 확인) 끌어올림 — 단 앙상블·거리류는 무효, *여전히 moderate*(0.95+ 미달)라 near-OOD는 *완화되었으나 미해결*인 한계로 정직하게 frame.

---

## 부록 A — 인코딩 변천사·근거·결함 (post-mortem)

> 왜 ~57% "천장"에 오래 갇혀 있었는가의 근본 원인. 핵심: **명목형(diplotype)을 ordinal 정수로 인코딩**한 것 + **one-hot에 StandardScaler를 씌운 실수**가 진짜 성능을 가렸다. one-hot(no scaler)+LogReg로 79.6%가 드러나며 발견됨(Exp 13).

### A.1 사용한 인코딩 3종 — 어떻게/왜
| # | 인코딩 | 위치 | 어떻게 | 왜 그렇게 했나 |
|---|---|---|---|---|
| 1 | **per-marker LabelEncoder (ordinal)** | `pipelines/baseline.py` `_encode` | marker 열마다 diplotype 문자열 → `LabelEncoder` → 임의 정수 코드. 결측 `N\|N`도 코드화 | Plan 1 baseline. 범주형을 가장 간단히 수치행렬로 → XGBoost(tree)에 투입. tree는 정수 feature에 split 가능 |
| 2 | **DiplotypeEncoder (ordinal + unseen)** | `data/encoding.py` | #1과 동일하나 train에 fit, unseen→예약코드(-1) | Plan 2에서 EAS/OOD 간 **코드 일관성** + OOD의 novel diplotype 처리(unseen→-1 자체가 OOD 신호) 필요 |
| 3 | **FMVocab (capped ordinal, top-K+MASK)** | `fm/vocab.py` | marker별 상위 K-1 빈도 + OTHER + MASK slot | SSL FM의 **per-marker 임베딩 테이블**용(임베딩이 코드를 학습벡터로 변환 → 명목성 무관). 임베딩 크기 cap |

### A.2 왜 결함이었나
- **ordinal 인코딩(#1·#2·#3을 raw feature로 쓸 때)의 근본 문제**: diplotype은 **명목형**(순서 없음)인데 LabelEncoder는 *임의의 순서*를 부여한다(예: `A-T|G-C`=0, `A-A|A-A`=1, `T-T|T-T`=2 …). tree 모델은 `x ≤ t` 임계 split만 하므로, 이 임의 순서에서 **인접한 코드들만 묶을 수 있다**. "코드 0과 5 vs 1·2·3·4"처럼 비인접 범주를 분리하려면 split을 여러 번 낭비 → 명목 신호가 체계적으로 소실. 그래서 XGBoost 52%·RF 57%에 갇혔다. (FM은 임베딩이 명목성을 처리하므로 인코딩이 *FM의* 병목은 아니었고, FM의 문제는 데이터 부족이었다.)
- **StandardScaler(with_mean=False) on one-hot 실수 (model-zoo, `scripts/15`)**: 선형·거리 모델에 "스케일링이 필요하다"는 통념으로 추가했으나, **one-hot 지시컬럼에 적용하면 각 컬럼을 std=√(p(1-p))로 나눠 희소(rare) 범주를 과증폭** → LogReg/SVM이 잡음 많은 rare one-hot에 과적합. 그 결과 LogReg가 46.6%로 보여 "tree가 최고 / 천장 57%"라는 **틀린 결론을 강화**했다. (Exp 13: scaler 제거 시 46.6%→79.6%로 확정.)

### A.3 왜 늦게 발견됐나
ordinal이 Plan 1의 tree-친화 기본값이었고 Plan 1–2 내내 유지. 경쟁작이 SNP+PCA를 써서 "피처가 더 필요하다"에 anchoring. model-zoo의 one-hot arm마저 StandardScaler로 망가져(46.6%) 잘못된 결론을 보강. Thread 1(calibration)에서 **우연히 scaler 없이** one-hot LogReg를 돌리며 73–80%가 드러남.

### A.4 교훈 (Paper 1에도 반영)
- **명목형 유전 마커는 one-hot(+linear)이 기본** — ordinal-LabelEncoder를 tree에 쓰면 성능이 조용히 깎인다.
- **one-hot에 StandardScaler 금지** (지시컬럼은 이미 동일 스케일; 스케일링은 rare 범주를 왜곡).
- 이 자체가 방법론적 기여: "복잡한 모델/피처 이전에 **인코딩**이 fine-scale MH ancestry의 결정 변수."
