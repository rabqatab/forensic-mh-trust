# 05 — Research Questions (확정본)

**작성일**: 2026-05-30
**목적**: 제안서(DFS5006) 이후 실제 실험으로 드러난 기여를 바탕으로 **연구 질문(RQ)을 확정**하고, 제안서 원안 RQ → 현재 RQ로의 변천과 각 RQ의 근거·상태를 한 문서로 정리한다.
**근거 문서**: `docs/04_experiments_and_results.md`(실측 로그), `paperwork/Paper1_draft_trustworthy_ancestry.md`(현 draft), `paperwork/DFS5006_final_260529_조민한개정본.md`(제안서 개정본), `docs/03_novelty_options.md`(novelty 앵커).

---

## 0. 한 줄 기여 (one-sentence contribution)

> **저분리(low-separability) fine-scale 집단 추정에서 신뢰 가능한 법과학적 출력은 — 분포-free conformal prediction set + open-set 거부로 — 정확도가 낮은 분류기조차 법정에서 쓸 수 있게 만들며, 그 신뢰도(open-set 거부 능력)는 정확도가 아니라 base-model의 보정(calibration)과 인코딩에 의해 좌우된다.**

**두 문장 pitch** (Explain-It test):
1. (문제) 법과학 집단 추정은 단일 라벨을 신뢰 수준 없이 보고하는데, 유전적으로 가까운 동아시아 집단에서는 단일 라벨 정확도 자체가 본질적으로 낮아 법정 제출에 위험하다.
2. (통찰) 우리는 어떤 확률 분류기에도 씌울 수 있는 model-agnostic trust layer(Mondrian conformal + open-set 거부)를 적용하고, **open-set 신뢰도가 정확도가 아니라 base-model 선택/보정에 의해 결정됨**을 보인다 — 같은 trust layer가 base만 바꾸면 무용지물에서 효과적으로 바뀐다.

---

## 1. 제안서 원안 RQ → 현재 RQ (변천)

제안서는 두 갈래를 동시에 내걸었다: **(A) 최소 MH 패널 도출**, **(B) Trustworthy 모델(CP+OSR)**. 인코딩 교정과 실험 결과로 두 갈래의 위상이 바뀌었다.

| 제안서 원안 RQ (암묵) | 원안 가정 | 실측 결과 | 현재 위상 |
|---|---|---|---|
| "동아시아 5집단을 ≥90% 정확도로 분류하는 **최소 MH 패널**은 몇 개인가?" | 소수 마커로 고정확도 가능 + 90% 달성 가능 | ≥90%는 도달 불가(전체도 79.6%); 그러나 **다변량 선택**으로 배치 가능 **trustworthy 최소 패널 존재**(coverage 보장, 10–15× 축소) | **ANSWERED(재정의)** → RQ5 |
| "CP+OSR로 분류에 신뢰 수준을 부여한다" | base=XGBoost로 충분 | coverage는 충족, 그러나 **OSR은 base에 의해 좌우**(XGBoost 최악) | **확장/심화** → RQ1(핵심)·RQ2·RQ4 |

**핵심 전환**: 부차적이던 "trustworthy 모델"이 — base-model 발견과 함께 — **논문의 핵심 기여**로 올라섰다. 제안서 무게중심이던 "최소 패널"은 *univariate 선택 기준* 음성이었으나, **다변량 선택**으로 살아남아(§23) — **배치 가능한 trustworthy 최소 패널(coverage 보장 + 정확도–set size–OSR frontier, 10–15× 축소)**로 ANSWERED. 단 원안의 "≥90%"는 도달 불가(정직).

---

## 2. 확정 RQ (계층)

> 표기: **[ANSWERED]** = 실측으로 답함 / **[PENDING]** = 진행 중 / **[DEFERRED]** = Paper 2 또는 데이터 확장 후.

### ★ RQ1 (Primary) — Open-set 신뢰도는 정확도가 아니라 base-model이 좌우하는가?

- **질문**: 데이터·인코딩·split을 고정하고 base estimator만 바꿀 때, open-set 거부 성능(far-OOD AUROC, empty-set 거부, near-OOD/LOPO)은 분류 정확도가 아니라 base-model의 보정/선택에 의해 결정되는가?
- **왜 중요**: 법과학 실무·ML 양쪽에 반직관적. "리더보드 1등 모델을 쓰라"는 통념을 뒤집고, 신뢰가 필요한 도메인의 모델 선택 기준을 바꾼다. `docs/03` 기준 forensic ancestry에 CP/OSR 적용 0편(Zhang 2025) — novelty 앵커.
- **가설**: 더 잘 보정된 단순 base(one-hot LogReg)가 정확도와 OSR을 **동시에** 최고로 달성하고, gradient-boosted trees는 OOD에 과확신하여 OSR 최악.
- **근거 [ANSWERED, 강함]**: §20(10-seed). LogReg far-OOD AUROC **0.840±0.016** vs XGBoost **0.675±0.038** → 격차 0.165 ≈ **~4σ**. empty-set 거부: LogReg 발화 / XGBoost 0.000±0.000. coverage는 세 base 모두 ≥0.90(보장은 base-무관). → **2단 메시지: 보장은 base-agnostic, 분리는 base가 좌우.**

### RQ2 — 분포-free conformal이 저정확도에서도 목표 커버리지를 주는가? 패널 크기와 set은?

- **질문**: 점추정 정확도가 낮아도(≈57–80%) Mondrian split-conformal이 1−α 커버리지를 보장하는가? prediction set은 마커 수에 따라 어떻게 좁아지는가?
- **왜 중요**: trust layer의 타당성 근거. "신뢰구간 포함 출력"이라는 법정 가치 제안의 핵심.
- **근거 [ANSWERED]**: §4.3/§21. coverage ≥0.90(α=0.10), α=0.05에서 보수적(~0.95+). set 크기는 마커↑로 축소(chr22 53마커 3.8 → genome-wide 3042 1.8 @α=0.1). → **정확도가 완벽하지 않아도 "5개 중 평균 1.8개 후보로 90% 보장".**

### RQ3 (Enabling) — fine-scale "≈57% 천장"은 생물학적(FST) 한계인가 인코딩 아티팩트인가? 최적 표현은?

- **질문**: 오래 갇혀 있던 ≈57% 정확도 천장은 동아시아 fine-scale의 본질적 한계(낮은 FST)인가, 아니면 명목형 diplotype을 ordinal로 인코딩한 아티팩트인가? 정확도와 신뢰를 동시에 최대화하는 표현은?
- **왜 중요**: 핵심 enabler이자 독립적 "simplicity result". 천장이 아티팩트면 제안서의 비관적 정확도 목표 해석 자체가 바뀐다.
- **근거 [ANSWERED]**: §13/부록 A. one-hot(no scaler)+LogReg **79.6%** vs ordinal-tree ≈57% → **아티팩트**. StandardScaler-on-one-hot은 선형/커널 모델을 붕괴(46.6%). PCA/SVD 피처도 무익(§4.7). **DL 5계열(MLP·CNN·embedding·autoencoder·transformer)도 전부 LogReg에 ≥25p 짐**(§24) — 단순함이 DL 전반에 우위. **선형 *클래스* 우위 확정**(distinct 결정규칙 6종 전부 73–82%: MultinomialNB 82.2 최고정확도·Passive-Aggressive/LinearSVC OSR 0.96 최고·Ridge·Perceptron·Nearest-Centroid; LogReg 79.6 — §24.4). 단 **비정규화 LDA는 p≫n 공분산으로 적용 불가** → "선형 우위 = *정규화* 선형". **트리 열세는 *모델 클래스***(native-categorical HGBDT도 59.5% — 인코딩 무관, §24.5); **tabular-DL SOTA(FT-Transformer/TabNet)·small-data SOTA(TabPFN)도 전부 dense 선형에 짐**(§24.5 capstone). → **권장 recipe: one-hot, no scaler, dense regularized linear.**

### RQ4 — 보정(ECE)과 open-set 분리(AUROC)는 같은 축인가?

- **질문**: confidence calibration(ECE)이 좋은 모델이 곧 OOD를 잘 분리하는가? deep-ensemble의 epistemic UQ가 OOD를 잡는가?
- **왜 중요**: "왜 ensemble epistemic UQ가 아니라 conformal에 trust layer를 앵커하는가"의 근거. RQ1의 메커니즘 해명.
- **근거 [ANSWERED]**: §4.4. 두 축 분리 — XGBoost ECE 최고(0.077)인데 OSR 최악(AUROC 0.695); deep-ensemble은 보정 좋아도 epistemic MI의 OOD AUROC≈0.36(<chance). → **ECE ≠ OSR. conformal coverage는 어느 쪽이든 유지되므로 trust layer를 거기에 앵커.**

### RQ5 (Scope) — forensic 배치 가능한 최소 MH 패널이 존재하는가?

- **질문**: 소수 마커 패널로 (i) 충분한 5집단 정확도, 또는 (ii) 충분한 trust(conformal coverage + 좁은 set + far-OOD 분리)를 **forensic 배치 규모(NGS ~50–200마커)**에서 달성할 수 있는가?
- **왜 중요**: 제안서 중심 deliverable. forensic 메시지를 "minimal panel" ↔ "genome-wide MH"로 가른다.
- **상태 [ANSWERED — 재정의]**: 배치 가능한 최소 패널이 **존재**하며 정확도–trust frontier로 특성화됨(docs/04 §23). 1차 음성(§21)은 RQ5의 답이 아니라 **약한 선택기 sub-result**로 강등.
  - **sub-result (유효)**: *univariate* MI/단순 L1은 compact 패널 부족(MI 200마커 55%) — "패널 없음"이 아니라 "약한 선택기로는 못 찾음"을 의미.
  - **rescue (정확도, §23.1 + RFE 교차검증 §23.3)**: **다변량 선택**(one-hot LogReg 계수 에너지)이 모든 N에서 +8~19p — **25마커 52%**(MI 대비 8× 효율), **1000마커 76.8%**(전체 79.6%의 96%). **메커니즘이 다른 RFE가 ±2p 내로 일치** → rescue가 선택기 아티팩트 아니라 데이터 성질(front-load). 둘 다 MI를 10–20p 압도.
  - **rescue (trust, §23.2)**: leakage-free 3-way split에서 **conformal coverage가 모든 패널 크기에서 ≥0.93 유지** — 25마커도 valid 90%+ coverage. trade-off는 set size(3.9→2.7)·far-OOD AUROC(0.59→0.76). **minimum forensic panel = 운영 스펙별 최소 N**(valid coverage면 25–50; set ≲3·AUROC ≥0.70이면 ~200–300, 10–15× 축소). 고정 배치 패널(top-50/100/200 마커 리스트) 산출.
  - **정직한 한계**: 원안의 "소수 마커 ≥90% 정확도"는 도달 불가(전체 패널도 79.6%); 충족된 것은 **trustworthy 최소 패널**(coverage 보장 + 운영점).

### RQ6 (Robustness) — conformal 보장은 법과학적 열화(allele dropout)에서 살아남는가?

- **질문**: clean 학습 후 degraded DNA(ADO)로 테스트할 때 커버리지 보장이 유지되는가, 얼마나 graceful한가?
- **왜 중요**: 경쟁 연구가 노출하지 않는 forensic-realism 한계. exchangeability 가정의 실제 취약점.
- **근거 [ANSWERED]**: §4.5. graceful하지만 측정 가능한 열화 — ADO 50%에서 coverage 0.91→0.80(exchangeability 위반으로 보장 자체가 약화). → degradation-aware calibration이 future work.

### RQ7 (Generalization) — 모델+trust layer가 독립 코호트(HGDP)로 전이되는가?

- **질문**: 1000G(hg19) 학습 모델이 HGDP WGS(hg38) 중첩 집단(Han↔CHB+CHS, Japanese↔JPT, Dai↔CDX)으로 전이되는가? build 조화(hg19↔hg38) 품질은?
- **왜 중요**: 단일 코호트 일반화의 honest 검증. 외부 타당성.
- **상태 [ANSWERED]**: **in-callset 깨끗한 전이**(gnomAD HGDP+1KG harmonized, hg38, 전체 3,042마커; 1KG·HGDP 같은 callset → build mismatch 없음). 1KG-EAS 학습 → HGDP-EAS 테스트 = **87.3%**(Han 1.00 / Japanese 0.73 / Dai 0.60 n=5). **unseen diplotype 0.43→0.024로 소멸**, transfer(87.3%) > within-HGDP CV(59.3%). cross-build 예비(82.4% @510마커)는 보수적 하한이었음(§22). 상세 docs/04 §22. (한계: Dai n=5 소표본, KHV 매칭 없음, Japanese↔Han 근연 오분류.)

---

## 3. 명시적 비-RQ (scope 경계 — 이번 논문이 다루지 *않는* 것)

- **SSL Foundation Model이 정확도/OSR에서 base를 능가하는가** → **[DEFERRED, Paper 2]**. full-panel·동일 프로토콜에서 SSL+finetune 54.6%(supervised transformer 51.0 대비 **+3.6p 양의 신호**, §24)지만 LogReg 79.6%엔 한참 못 미침. **SSL이 작게라도 돕는다**는 첫 실데이터 증거 → 데이터 확장 검정. **[ANSWERED-negative, §25]**: 통제 ablation(gnomAD 4,091 hg38 clean) — **LogReg 78.0% ≫ supervised 55.4 ≥ SSL@4091 54.0 ≥ SSL@583 48.7**. SSL pretrain이 supervised를 못 이기고, 데이터 scaling(583→4091)은 supervised 수준 *회복*에 그침(LogReg와 24p 격차). **fine-scale MH에서 SSL FM은 단순 선형을 못 이김 — 병목은 데이터만이 아니라 모델 클래스.**
- **한국인(KOR) 미지 집단을 실제로 거부하는가** → 데이터 부재. non-EAS super-pop + LOPO를 honest proxy로만 사용(한계 명시).
- **Reliable-Ae(phasing-error penalty) 정량화** → 표준 1000G에 완전 EAS trio 0개 → NYGC 30× release 필요, **[DEFERRED]**. Ae 분석 자체는 보고(§4.5 docs/04).
- **웹 도구** → 미구현(모델·평가 우선).

---

## 4. RQ ↔ 증거 매핑 (상태 요약)

| RQ | 핵심 주장 | 근거(docs/04) | 상태 |
|---|---|---|---|
| **RQ1** | open-set 신뢰도는 base-model이 좌우(정확도 아님) | §20(4σ), §4.3 | **ANSWERED ★** |
| RQ2 | conformal이 저정확도에서도 목표 커버리지 | §4.3, §21 | ANSWERED |
| RQ3 | 57% 천장은 인코딩 아티팩트; one-hot LogReg 79.6% | §13, 부록 A | ANSWERED |
| RQ4 | ECE ≠ OSR 분리 | §4.4 | ANSWERED |
| RQ5 | 배치 가능 최소 forensic 패널 존재 (정확도–trust frontier; 10–15× 축소) | §23 | ANSWERED(재정의) |
| RQ6 | conformal 보장이 ADO에서 graceful 열화 | §4.5 | ANSWERED |
| RQ7 | HGDP 외부 코호트 전이 (in-callset 87.3%, clean hg38) | §22 | ANSWERED |

---

## 5. 다음 단계 (RQ 관점)

**모든 RQ ANSWERED + Paper 1 마감 항목 완료 (2026-06-04):**
1. ✅ **RQ7 반영**: in-callset 87.3%(clean hg38, §22) → Paper 1 §4.6 작성 완료(placeholder 제거).
2. ✅ **Paper 1 spine 정렬**: RQ1 primary / RQ3 enabling / RQ2·4 method-validity / RQ5·6·7 forensic-realism — 현 draft와 일치.
3. ✅ **citation 검증**: 6건 프로그램적 확인(CrossRef/arXiv/publisher, DOI 확보). **"Olsson 2025"는 실재 불가 → Papangelou et al. 2025로 교정**(Paper 1·docs/02 동시 반영). 잔존 `[verify]` = 0.

**남은 선택지 (RQ 외 산출물):**
- (선택) Paper 1 §6에 CREE(Paper 2) future-work teaser 1줄.
- (미래·별개) Paper 2(CREE) 본 논문 — pilot 완료(docs/07 §7: d/b/f positive, c/e negative), 본실험 DEFERRED.
- 강의 final(DFS5006) 개정 필요 여부 — 사용자 확인 대기.
