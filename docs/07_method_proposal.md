# 07 — Method proposal: Conformal Random-Effects Embeddings (CREE)

**작성일**: 2026-05-31 · **상태**: 제안 + 진행 중 검증 · **근거**: §26(왜 선형이 이김), §27(좋은 embedding), docs/02 §9(문헌).
> Paper 2 후보 method. brainstorming-research-ideas(F3 tension / F9 composition / F4 cross-pollination)로 수렴.

## 1. Two-sentence pitch
- **(문제)** 고위험 p≫n 범주형 분류(forensic 유전체·희귀질환·single-cell)는 *적은 라벨로 작동하는 학습 표현* + *통계적 보장이 있는 abstention/open-set*가 동시에 필요하지만, 학습 embedding은 소표본에서 과적합(EmbMLP 29.7% vs 선형 79.6%)하고 보정된 open-set을 못 준다.
- **(통찰)** embedding을 **계층적 Gaussian random effect**(empirical-Bayes 수축 → 소표본에서 작동, RandEff 72.6%·DietNet 73.6%로 +43p)로 학습하고 대규모 unlabeled 코호트로 SSL 사전학습한다. **per-feature posterior 분산이 곧 conformal nonconformity 점수**가 되어 분포-free coverage + *어느 marker가 낯선지까지 국소화된* open-set 거부를 준다 — 표현·정규화·calibrated 불확실성이 **한 객체**.

## 2. 해소하는 tension (F3)
표현학습 능력 ↔ 소표본 과적합 ↔ calibrated open-set — 보통 셋 중 둘. random-effects 분산 객체가 셋을 동시에:
- 능력: SSL 사전학습된 학습 표현(전이 가능).
- 소표본: Gaussian 수축(empirical-Bayes)이 과적합 억제(우리 증거 §27).
- 신뢰: 분산 = open-set 점수(내재).

## 3. 무엇이 new (3)
1. **p≫n 표현학습의 해법 = random-effects 수축** (James–Stein deep 아날로그) — 과적합 실패를 near-linear 성공으로(§27, *정확도* 축).
2. **Variance-as-nonconformity**: 학습된 per-code posterior 분산이 곧 open-set/conformal 점수 — 불확실성이 표현에 *내재*(MSP/앙상블 epistemic은 OOD와 decouple, RQ4 §14). **메커니즘(실측 교정, §27.2)**: 이 OOD 신호는 *random-effects 수축(KL)이 아니라* **변분 학습 자체**에서 나온다 — reparameterization noise 하 CE loss가 자주·판별적인 코드의 분산을 누르고 드문 코드는 prior 근처에 남겨 *빈도-친숙도 gradient*(Spearman(freq, logvar) = −0.99)를 만든다. KL 수축은 *marginal 증폭*(near-OOD 0.772@KL=0 → 0.792@KL=2)일 뿐 원인이 아니다.
3. **Feature-localized abstention**: 분산이 marker별 → "marker X가 낯설어 불확실"을 *감사 가능*하게. LogReg/MSP 불가.

## 4. 검증 실험
- **(b) embedding + conformal** (`scripts/trust/47`): embedding-DL이 유효 coverage + open-set 받나. *[완료: coverage 0.920 ✓]*
- **(c) SSL + random-effects encoder** (`scripts/ssl/48`, gnomAD 4091): 정규화가 대규모 SSL을 살리나(vs naive SSL 54%). *[완료: 52.7% — NEGATIVE]*
- **(d) Variance-as-nonconformity** (`scripts/trust/49`): 변분 random-effects embedding의 posterior 분산 점수 open-set AUROC vs MSP(목표: LinearSVC 0.957 비교). *[CREE 핵심 — 완료: 0.946 ✓]*
- **(e) 극소표본 robustness** (`scripts/models/50`): n=50/100/200/full에서 CREE vs LogReg 저하 곡선. *[완료: NEGATIVE]*
- **(f) cross-cohort transfer** (`scripts/rq7/52`): 1KG-EAS5 학습 embedding → HGDP zero-shot 전이(동일 hg38 패널). *[완료: cross-cohort var 0.999 vs MSP 0.633, 코호트 강건 — POSITIVE]*

## 5. 정직한 위치 (F7 Simplicity + 강한 반박)
- **raw accuracy로 선형을 이기는 건 우리 데이터상 어렵다(§26)** → 셀링 포인트는 *정확도*가 아니라 **transfer · 내재 분산 open-set · 감사 가능 국소 abstention**.
- **반박 "LogReg+conformal로 충분하지 않나?"** → LogReg+conformal은 (i) 전이 표현 없음 (ii) unlabeled 데이터(SSL) 활용 불가 (iii) feature-국소 불확실성 없음 (iv) multi-task/new-cohort 일반화 안 됨. CREE는 정확도를 *맞추면서* 이를 추가.
- **이론 주의**: conformal coverage는 *어떤 점수든 공짜*. 진짜 기여 = 분산 점수의 *효율/검정력*(경험적) + embedding의 *empirical-Bayes 최적성*. 과대포장 금지.

## 6. 위험 & venue
- 위험: 분산 점수가 MSP를 못 이기면 open-set 주장 약화 → "trustworthy 전이 표현"으로 후퇴. 선형 정확도 초과는 비현실적 — capability로 frame.
- Venue: NeurIPS/ICLR(trustworthy ML·UQ·representation) 또는 AISTATS(EB–conformal 이론); 응용은 forensic/genomics venue.

## 7. 2주 pilot — **결과 (2026-05-31)**
- **★(d) variance-as-nonconformity = POSITIVE**: 변분 분산 점수 far-OOD open-set **AUROC 0.946 ± 0.010** vs 같은 모델 MSP 0.676 → 내재 분산이 MSP를 압도, LinearSVC(0.957) 근접. **CREE 핵심 capability 성립**(§27.1).
- **★(g) near-OOD = POSITIVE (최대 난제에서 유일 작동 레버, §27.2)**: KOR-proxy(15개 미수록 HGDP-EAS 집단)에서 변분 분산 **near-OOD AUROC 0.782 ± 0.005** — MSP(0.670)·거리(Mahalanobis/kNN 0.69)·set-size(0.48)를 8–11p 압도, **모든 점수 중 최고·최안정**. 난이도 gradient 확인(far 1.0 → near 0.78). 메커니즘 검증(§27.2): KL 무관(KL=0도 0.772)·OTHER 무관(제외해도 0.78–0.80) → *변분 학습된 코드별 친숙도*가 원천. **boost**: {variance, rel-Maha, MSP} 단순 z-fusion으로 **~0.82**(미관측 집단 전이), 단 앙상블·거리류 무효. *0.82는 여전히 moderate — near-OOD는 완화 가능 한계지 해결 아님.*
- **(b) coverage = POSITIVE**: conformal coverage 0.920 유지(embedding-DL에서도 model-agnostic).
- **(e) 극소표본 = NEGATIVE**: RandEff가 n=50–full 전 구간 LogReg에 −8~10p. shrinkage embedding은 소표본 정확도 이점 없음.
- **(c) SSL+random-effects = NEGATIVE**: gnomAD 4091 사전학습 후 acc **52.7 ± 3.5%**(far-OOD AUROC 0.632) — naive SSL 54.0·supervised 55.4와 동일. 수축을 *deep* SSL transformer에 붙여도 SSL 한계(§25) 못 넘음. 대비: 같은 수축이 *얕은* RandEff(72.6%)에선 +43p → **정규화 이득은 지배적 inductive bias일 때만** 발현(transformer attention의 별도 과적합이 묻음).

- **★(f) cross-cohort transfer = POSITIVE**: 1KG-EAS5 학습 VRE를 HGDP(미관측 코호트, 18 novel EA 집단)로 zero-shot. cross-cohort open-set **var 0.999 vs MSP 0.633**; MSP는 코호트 경계에서 0.721→0.633 −0.09 붕괴, 분산은 1.000→0.999 유지 → *intrinsic uncertainty는 cohort-shift에 강건*. 메커니즘 검증: 무학습 OTHER-fraction AUROC=0.002/0.063 → 빈도 artifact 아님, *학습된 친숙도*. caveat: cross-continental far-OOD라 절대값 부풀려짐 → 기여는 *var↔MSP 격차+강건성*.

→ **결론(확정)**: CREE의 *정확도·small-n·대규모-SSL* 우위는 없음(선형이 강함, §26; (c)(e) negative). 그러나 **신뢰성 축의 네 capability — far-OOD 내재 분산(d, 0.946 vs MSP 0.676) · model-agnostic conformal coverage(b, 0.920) · cross-cohort 전이(f, var 0.999 vs MSP 0.633) · near-OOD(g, 0.782, 모든 점수 중 최고) — 가 모두 성립**. 검증된 CREE 형태 = **얕은 변분 embedding + variance-as-nonconformity**(메커니즘 = 변분 학습된 코드별 친숙도; random-effects 수축은 정확도 축의 별개 기여이고 OOD 신호의 원인은 아님 — §27.2 교정). honest top-tier 스토리 = "embedding이 정확도론 선형을 못 이기나, *신뢰성(open-set)에선 내재 불확실성으로 far·near·cross-cohort 모두에서 post-hoc MSP를 압도*한다 — near-OOD(법과학 최대 난제)에서 유일하게 작동하는 레버." pilot 완료 — 다음은 paper-2 작성.
