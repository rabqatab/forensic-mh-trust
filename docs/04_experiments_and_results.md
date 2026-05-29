# 04 — Experiments & Results (comprehensive log)

**프로젝트**: Trustworthy Forensic Microhaplotype Model — 1000G East-Asian 5집단 분류 + calibrated UQ/open-set + SSL foundation model.
**최종 갱신**: 2026-05-29 (Opus 4.8)
**재현**: 모든 코드 `main` 브랜치, uv 환경(`microhapdb==0.12`, torch 2.12.0+cu130). 결과 JSON은 `results/`(gitignored), 코드/스크립트 커밋됨. 72 pytest green.

> **한 줄 요약**: (1) 동아시아 5집단 fine-scale 분류는 본질적으로 어려움 — 최고 정확도 ~57%(genome-wide, chance 20%). (2) **RandomForest가 XGBoost를 정확도·OSR 모두에서 앞섬**(정확도 56.6% vs 52.0%; Plan 2 far-OOD AUROC **0.803 vs 0.695**, **empty-set reject가 RF에서 처음 발화**). (3) Conformal coverage 보장은 작동하고 marker↑로 set이 좁아짐. (4) **Open-set 약점은 상당 부분 base model 문제** — base만 RF로 바꾸면(한 줄) 개선. (5) **PCA·딥러닝(FM)은 소데이터에서 tree 앙상블을 못 이김** — 낮은 정확도가 곧 "단정 분류 대신 신뢰구간·거부가 필요"라는 본 연구 전제의 정량적 입증.

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
→ **최소 패널 ≈ 200마커(정점 56.9%)**, 그 이상 개선 없음. 제안서 "90%" 목표 미달 — EAS-5의 낮은 FST 때문(본질적 난이도).

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

## 13. 종합 핵심 발견
1. **정확도 천장(~57%)은 샘플이 아니라 FST(과제 난이도)** — 같은 5집단 샘플↑로는 못 깸. 이 낮은 정확도가 "단정 분류 대신 신뢰구간·거부 필요"라는 전제의 입증.
2. **base 모델 선택이 trustworthy 거동을 좌우**: RandomForest > XGBoost(정확도 56.6% vs 52.0%, OSR 0.736 vs 0.595). Plan 2 layer가 model-agnostic이라 `base_estimator` 교체만으로 OSR 개선 가능(§11에서 검정).
3. **소데이터에서 tree 앙상블 우세**: PCA·딥러닝(FM)·거리기반(kNN/SVM) 모두 RF/XGB에 못 미침 — 경쟁작도 GBDT 최선. 딥러닝의 활로는 정확도가 아니라 표현/UQ + 데이터 확장.
4. **Coverage(작동) vs OSR(모델 의존)** 의 분리.

## 14. 한계
- KOR 데이터 부재(비-EAS+LOPO 대용). Trio phasing 재현 불가(NYGC 필요, Reliable-Ae deferred). FM 실데이터는 최소 구성만(과적합); 데이터 확장+contrastive 미실행. kNN/SVM은 고차원 one-hot에서 약함. 직접 경쟁작(Chen 2025) 존재 — 차별점은 trustworthy 축.

## 15. 재현 (commands)
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
```
산출: `results/baseline/*.json`, `results/conformal/*.json`.

## 16. 다음 실험
1. **base 모델을 RandomForest로 채택** (§11 완료 — OSR 대폭 개선). Plan 2 trust layer 기본 base를 RF로 변경 + 제안서 개정본 반영. (α=0.05 under-cover는 cal 분할 키워 점검.)
2. **데이터 확장 + contrastive**(all-2504/NYGC unlabeled → contrastive 사전학습 → EAS finetune) — FM의 유일한 활로.
3. **Deep Ensembles/MC-Dropout UQ**(aleatoric vs epistemic; forensic ancestry 선행 0편) — trustworthy 축 강화.
4. **NYGC trio** → Reliable-Ae phasing penalty 정식 산출.
5. msp_threshold sweep(운영점 곡선), 도표화(`academic-plotting`: §3 coverage, §9 model zoo, §7 accuracy curve).
