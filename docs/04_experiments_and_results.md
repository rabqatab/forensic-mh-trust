# 04 — Experiments & Results (comprehensive log)

**프로젝트**: Trustworthy Forensic Microhaplotype Model — 1000G East-Asian 5집단 분류 + calibrated UQ/open-set + SSL foundation model.
**최종 갱신**: 2026-05-29 (Opus 4.8)
**재현**: 모든 코드 `main` 브랜치, uv 환경(`microhapdb==0.12`, torch 2.12.0+cu130). 결과 JSON은 `results/`(gitignored), 코드/스크립트 커밋됨. 72 pytest green.

> **한 줄 요약 (2026-05-30 갱신 — 이전 "57% 천장/XGBoost 과확신" 주장 폐기)**: (1) **인코딩·모델 선택이 결정적**: MH는 명목형이라 **one-hot + Logistic Regression = 79.6% (5-fold CV)** 로, ordinal-tree(XGBoost 52%, RF 57%)·SSL FM(26%)을 모두 압도. "57% 천장"은 FST가 아니라 *ordinal 인코딩 artifact*였음(Simplicity-Test 승리). (2) **LogReg(one-hot)이 정확도·OSR 모두 최고**(acc 79.6%, far-OOD MSP AUROC 0.84). (3) Conformal coverage 보장 작동, marker↑로 set 좁아짐; 단 ADO(열화)에서 보장 저하. (4) **Calibration(ECE)은 XGBoost가 최선(0.077)·RF가 최악(0.315)** — OSR AUROC(rank)와 ECE(보정)는 별개; 앙상블 epistemic은 OOD 미분리(≈chance). (5) 정확도 70%+를 MH-only로 달성 → SNP/AISNP 확장 불필요.

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

## 1. Exp 1 — Leakage-free baseline (Plan 1)
**방법**: diplotype 추출(양 haplotype, P0 #2) → per-marker 인코딩 → **leakage-free nested CV**(outer fold 내부 feature selection, P0 #1). GroupKFold scaffold만(P0 #8; EAS는 사실상 unrelated).
**결과 (chr22, 53마커, chance 0.20)**: top-5 0.236, top-20 0.298, all 0.298 → near-chance(정보량 부족). **상태 ✅** pipeline 검증. (genome-wide 정확도는 §7.)

## 2. Exp 2 — Reliable-Ae / Wei 2025 phasing (Plan 1)
- **Ae(EAS, chr22)**: mean **4.91**, max **17.44**. 고-Ae marker가 Wei 2025의 phasing-위험 marker.
- **P_phase_error/Reliable-Ae**: **deferred**(complete trio 6개, NYGC 필요). 스크립트 hg38-ready. **상태 △**.

## 3. Exp 3 — Conformal coverage (Plan 2, XGBoost base)
LAC nonconformity + **Mondrian** per-class quantile(order-statistic). `results/conformal/coverage_curve.json`.

genome-wide (3,042마커):
| α | target | coverage | set size |
|---|---|---|---|
| 0.30 | 0.70 | 0.770 | 2.09 |
| 0.20 | 0.80 | 0.809 | 2.28 |
| 0.10 | 0.90 | 0.888 | 2.61 |
| 0.05 | 0.95 | **0.954** | 3.16 |

→ ✅ coverage가 1−α 추종, **marker↑ → set tighter**(chr22 53마커 α=0.1 set 3.79 → 3,042마커 2.61).

## 4. Exp 4 — Open-set far-OOD (Plan 2, XGBoost base)
- OOD unseen-diplotype fraction **0.132**; **MSP AUROC 0.695**, FPR@95TPR 0.75; empty-set reject **0**(모든 α).
- marker별 AUROC 추이: 53마커 0.50 → 1,058 0.67 → 3,042 **0.695**(포화).
→ ⚠️ XGBoost가 비-EAS도 과확신 분류 → empty-set 미발화. **base-calibration 한계**(§9·§11에서 RF로 개선).

## 5. Exp 5 — Open-set near-OOD / LOPO (Plan 2, XGBoost base)
5개 hold-out 집단 모두 reject gap **0**(α=0.10) → ❌ 근연 집단 미검출. base model 과확신.

## 6. Exp 6 — SSL FM core (Plan 3a, synthetic)
구성: FMVocab, MHMatrixDataset(masked + ADO/dropout contrastive views), objectives(masked CE + NT-Xent), MHTransformer(weight-tied head), heads, pretrain/finetune, `ForensicFMClassifier`.
synthetic: SSL loss 11.47→6.52; finetune acc 0.36→0.92; **ConformalClassifier가 FM을 변경 없이 wrapping**(통합 테스트 통과). **상태 ✅ 코어(72 tests)**. 실데이터 성능은 §10.

---

## 7. Exp 7 — Genome-wide 분류 정확도 / 최소 패널 (leakage-free nested 5-fold)
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

## 8. Exp 8 — PCA-feature ablation (Chen 2025 동기)
`results/baseline/pca_ablation.json`, 동일 5-fold:
| 표현 | 분류기 | 정확도 |
|---|---|---|
| **raw ordinal** | XGBoost | **56.1%** |
| one-hot→SVD(10/20/50/100) | XGB | 32.3 / 38.3 / 27.8 / 28.8 |
| one-hot→SVD(10/20/50/100) | LogReg | 37.3 / 28.6 / 25.4 / **45.8** |
→ **PCA류는 도움 안 됨**(최선 45.8% < raw 56.1%, −10.3p). 경쟁작 이득은 PCA가 아니라 ancestry-최적 AISNP 선별 + ADMIXTURE에서 옴. one-hot MH에 SVD를 씌우면 XGBoost가 raw에서 쓰던 marker별 판별 신호가 소실.

## 9. Exp 9 — Model zoo (full 3,042, 동일 5-fold + conformal/OSR)
`results/baseline/model_zoo.json`. tree=ordinal 인코딩, linear/distance=one-hot Pipeline.
| 모델 | 정확도 | coverage | set size | **MSP AUROC** |
|---|---|---|---|---|
| **RandomForest** | **0.566** | 0.941 | 3.11 | **0.736** |
| XGBoost | 0.520 | 0.928 | 3.20 | 0.595 |
| LogReg | 0.466 | 0.915 | **1.91** | 0.675 |
| kNN | 0.212 | 0.921 | 3.91 | 0.549 |
| SVM-RBF | 0.254 | 0.921 | 4.11 | 0.656 |
→ **RandomForest가 정확도·OSR 모두 최고** (XGBoost 추월). **OSR(AUROC)이 base 모델에 크게 의존**(0.55~0.74) — open-set 약점이 부분적으로 모델 선택 문제. kNN·SVM은 one-hot 고차원에서 붕괴(curse of dimensionality, 정직한 결과). LogReg는 가장 좁은 set(1.91)=가장 결정적.

## 10. Exp 10 — FM vs XGBoost head-to-head (Plan 3b-core)
`results/baseline/fm_vs_xgboost.json`, top-256 high-Ae 패널, 70/30 split:
| 모델 | 정확도 | coverage | set size | MSP AUROC |
|---|---|---|---|---|
| XGBoost | 0.316 | 0.862 | 3.80 | 0.591 |
| FM | 0.263 | 0.947 | 4.45 | 0.532 |
→ **FM이 XGBoost에 짐**(소데이터 504로 transformer 과적합; 어댑터는 masked-only SSL). 두 모델 절대치가 낮은 건 **high-Ae 패널이 약하기 때문**(Ae=집단 내 다양성 ≠ 집단 간 FST). FM의 활로 = unlabeled 데이터 확장(all-2504/NYGC) + contrastive 활성화(Plan 3b-extended).

## 11. Exp 11 — RandomForest base로 Plan 2 재산출 (완료)
`results/baseline/plan2_rf_vs_xgb.json` — 동일 데이터·인코딩·split, **base_estimator만 교체**:

| 지표 | XGBoost | **RandomForest** |
|---|---|---|
| coverage @α=0.1 | 0.888 | 0.908 |
| set size @α=0.1 | 2.61 | **2.45** (더 좁음) |
| far-OOD MSP AUROC | 0.695 | **0.803** |
| far-OOD FPR@95TPR | 0.75 | **0.572** |
| empty-set OOD reject | **0** (전 α) | **발화** — α=0.3 →0.21, α=0.2 →0.02 (in-dist ≤0.02) |
| LOPO near-OOD gap | 0 (전 집단) | CHS **0.048** (나머지 0) |

→ **RF가 trustworthy 축을 실질 개선**: far-OOD AUROC **+0.11**, FPR@95 0.75→**0.57**, **empty-set reject 0→발화**(XGBoost는 전혀 안 됨), near-OOD도 CHS에서 미약 검출. set도 더 좁음.
**핵심 결론**: §4·§5의 OSR 약점은 방법의 한계가 아니라 **상당 부분 base model(XGBoost 과확신) 문제**였고, model-agnostic trust layer에서 **base만 RF로 바꾸면(한 줄) 개선**된다.
주의: RF는 α=0.05에서 coverage 0.928(<0.95) — 소표본 per-class calibration으로 약간 under-cover → 운영 α=0.10 권장.
**상태 ✅ — RandomForest를 권장 base로.**

## 12. 관련 연구 비교 — Chen et al. 2025 (Human Genomics, 95.6%)
경쟁작의 95.6%는 **직접 비교 불가**: (a) **9개 광역·언어계통 클러스터**(중앙아·시베리아·동남아 포함, FST 높음) 분류, (b) **AISNP 2,000개 + PCA/ADMIXTURE** 피처, (c) Human Origins(array) 1,703명/67집단. 그들도 근연 그룹 sensitivity는 **0.66~0.87**. 우리 과제(1000G EAS-5, low FST, MH)와 입도·마커·데이터가 모두 다름. **차별점**: calibrated UQ/OSR + Reliable-Ae + forensic admissibility(경쟁작 부재).

---

## 13. Exp 13 — 인코딩이 천장을 깬다: one-hot LogReg (leakage-free 5-fold CV)
`results/baseline/onehot_cv.json` (OneHotEncoder를 fold 내부에서 fit):
| 설정 | 5-fold CV 정확도 |
|---|---|
| **LogReg + one-hot (no scaler)** | **79.6% ± 3.9** (folds 0.86/0.74/0.78/0.78/0.81) |
| LogReg + one-hot + StandardScaler | 46.6% ± 5.4 |
→ **"57% 천장"은 ordinal 인코딩 artifact**(§1·§7의 tree-on-ordinal). MH는 명목형 → one-hot+linear가 ordinal-tree(XGB 52/RF 57)·SSL FM(26)을 **압도(79.6%)**. `StandardScaler(with_mean=False)`가 one-hot 희소컬럼을 과증폭 → §9 model-zoo의 linear/distance 모델을 부당하게 망가뜨림(46.6%가 그 증거). **권장 base = LogReg(one-hot)** (정확도·OSR 모두 최고).

## 14. Exp 14 — Calibration(ECE) + Deep Ensembles / MC-Dropout UQ
`results/baseline/calibration_uq.json` (70/30 split):
| 모델 | acc | ECE | MSP AUROC | epistemic AUROC |
|---|---|---|---|---|
| **LogReg (one-hot)** | 0.730 | 0.230 | **0.841** | — |
| RandomForest | 0.566 | 0.315 | 0.725 | — |
| XGBoost | 0.500 | **0.077** | 0.674 | — |
| DeepEnsemble (5×MLP) | 0.724 | 0.080 | 0.716 | 0.362 |
| MC-Dropout (MLP) | 0.717 | 0.177 | 0.809 | 0.521 |
→ **ECE 정정**: XGBoost가 가장 잘 보정(0.077), **RF가 최악(0.315)** — 이전 "XGBoost 과확신" 주장 폐기. RF의 OSR 우위는 보정이 아니라 확률 *순위* 분리(AUROC) 덕(ECE와 AUROC는 별개). **앙상블 epistemic은 OOD 미검출**(0.36/0.52≈chance) → MSP가 더 나음(정직한 negative). 정확도 정식값은 §13의 CV(79.6%).

## 15. Exp 15 — 열화 DNA(ADO) robustness (RF base, clean 학습→ADO test)
`results/baseline/ado_robustness.json`:
| ADO rate | acc | coverage | set | OSR AUROC |
|---|---|---|---|---|
| 0.0 | 0.678 | 0.908 | 2.45 | 0.803 |
| 0.2 | 0.645 | 0.882 | 2.47 | 0.800 |
| 0.3 | 0.579 | 0.882 | 2.43 | 0.768 |
| 0.5 | 0.592 | **0.803** | 2.37 | 0.730 |
→ 우아한 저하나 **ADO 50%에서 coverage 0.91→0.80(보장 깨짐)** — conformal exchangeability 위반(train clean vs test 열화). forensic admissibility에 중요(경쟁작 미탐구). base를 LogReg(one-hot)로 한 ADO 재실험은 follow-up.

## 16. 종합 핵심 발견
1. **천장은 FST가 아니라 인코딩/모델이었다(정정)**: ordinal-tree로 57%였지만 **one-hot+LogReg로 79.6%(CV)** — 70% 목표를 MH-only로 초과. 가장 단순한 모델이 가장 강함(Simplicity Test). 본 연구 메시지가 "low-acc라 UQ로 보완"에서 "**competitive acc + calibrated UQ/OSR**"로 격상.
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
uv pip install -e ".[dev]" && uv run python -m pytest -q     # 72 tests
bash scripts/01_download_1000g.sh && bash scripts/06_download_genome_wide.sh
uv run python scripts/03_run_baseline.py        # chr22 baseline
uv run python scripts/04_wei2025_phasing.py     # EAS Ae (phasing deferred)
uv run python scripts/10_conformal_curve.py     # coverage (Plan 2)
uv run python scripts/11_openset_ood.py         # far-OOD
uv run python scripts/12_lopo_nearood.py        # near-OOD (LOPO)
uv run python scripts/13_pca_ablation.py        # PCA ablation (Exp 8)
uv run python scripts/14_fm_vs_xgboost.py       # FM head-to-head (Exp 10)
uv run python scripts/15_model_zoo.py           # model zoo (Exp 9)
uv run python scripts/16_plan2_rf.py            # RF vs XGB Plan 2 (Exp 11)
uv run python scripts/17_calibration_uq.py      # ECE + Deep Ensembles + MC-Dropout (Exp 14)
uv run python scripts/18_ado_robustness.py      # ADO degraded-DNA (Exp 15)
uv run python scripts/19_onehot_cv.py           # one-hot LogReg 5-fold CV (Exp 13)
```
산출: `results/baseline/*.json`, `results/conformal/*.json`.

## 19. 다음 실험
1. **base = LogReg(one-hot) 채택** (Exp 13 — 79.6%, OSR도 최고). Plan 2 전 지표(coverage/far-OOD/LOPO)를 LogReg(one-hot)로 재산출(§11을 RF→LogReg로 갱신).
2. **외부 검증**: 다른 코호트(HGDP/NYGC EAS)로 one-hot LogReg 79.6% 일반화 확인.
3. **ADO 재실험을 LogReg(one-hot) base로** + clean+ADO 혼합 calibration으로 coverage 회복 시도.
4. **Paper 1 메시지 재정립**: "competitive accuracy(80%, one-hot) + calibrated UQ/OSR + ADO robustness; 단순 인코딩이 복잡 모델을 이긴다."
5. (deferred) FM/contrastive + 데이터 확장(Paper 2), NYGC trio(Reliable-Ae), msp_threshold 운영곡선, 도표화(`academic-plotting`).
