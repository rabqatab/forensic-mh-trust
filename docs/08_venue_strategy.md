# 08 — Venue 전략 & FSI:G 재앵커 plan

**작성일**: 2026-06-04 · **목적**: Paper 1을 forensic 최상위지(FSI:G)에 맞춰 *재앵커*하기 위한 근거·실행계획. venue-fit 검색(C, 검증된 DOI)과 RQ 3-act spine(docs/05)에 종속.
**근거**: docs/05(RQ spine), docs/04(실측), docs/02(문헌), `paperwork/Paper1_draft`(현 draft). venue 검색은 2026-06-04 실측(아래 인용 전부 CrossRef/journal 확인).

---

## 0. 한 줄 결론

> **FSI:G는 현실적이고 오히려 이상적인 top-tier 타깃이다** — 저널이 지금 정확히 우리 이웃 논문들을 싣고 있다(아래 §2). 단 현재 draft의 *ML-우선 프레이밍*(RQ-Ⅰ 헤드라인)을 **forensic 가치명제 중심으로 재앵커**하고, forensic 리뷰어의 3대 반박을 *선제적으로* 흡수해야 한다.

---

## 1. Venue verdict (검색 결과 요약)

- **niche가 실제로 비어 있음**: FSI:G/FSI/IJLM/JFS/Electrophoresis(2019–2026)에서 conformal prediction·prediction set·open-set/OOD·abstention을 forensic DNA에 적용한 논문 **0편**. forensic 전체로 넓혀도 conformal 적용 사례 **0편**(STR·mixture·age·body-fluid·PMI 모두).
- **first-mover claim은 성립 — 단 정밀 scope 필수**: "reliability가 처음"이 아니라 **"forensic ancestry에 conformal prediction set + open-set recognition을 처음 적용"**. *reliability/calibration 대화는 이미 LR-calibration 계보로 존재*하므로 무시하면 안 되고 인용해야 한다(§3 (a)).
- **편집 appetite가 live**: 최근 ~18개월 FSI:G가 ML-on-MH-ancestry + AI-trust 논문을 연속 게재(§2) → 우리 conformal+OSR은 그들이 *밟지 않은 다음 논리적 칸*.

## 2. 경쟁·참조 지형 (검증된 DOI)

| 논문 | 모델/스코프 | 정확도 | UQ/set/open-set | 우리의 활용 |
|---|---|---|---|---|
| **Heinzel, Purucker, Hutter, Pfaffelhuber 2025**, FSI:G 79:103290 (doi:10.1016/j.fsigen.2025.103290) | 분류기 벤치(**TabPFN** 최고), log-loss 보고 | TabPFN best | log-loss(calibration 인접) **있음**, CP/OSR **없음** | **가장 유사 = 직접 reference point**. "그들은 proper scoring까지 갔으나 set-coverage/OSR은 안 함"으로 차별화. (Hutter group=TabPFN 저자) |
| **Chen et al. 2025**, Human Genomics 19 (doi:10.1186/s40246-025-00837-3) | 7 AISNP 패널×6분류기 + Locator; 67 E/SE-Asian | XGBoost 95.6% | **50% posterior threshold = ad-hoc "operational grouping"** | 우리가 **그 ad-hoc threshold를 형식화(OSR)**한다고 positioning |
| **Wang C. et al. 2025**, FSI:G 77:103239 (doi:10.1016/j.fsigen.2025.103239) | PCA+XGBoost, Asian | ~98.96%(7 Asian) | 없음 | 고정확도 광역 대비 → 우리는 fine-scale+trust |
| **Wei et al. 2025**, FSI:G 78:103273 (doi:10.1016/j.fsigen.2025.103273) | 602 trio MH phasing/Ae 신뢰성 감사 | n/a | n/a | **1000G-MH 신뢰성 앵커**(보완, 비경쟁) |
| **Barash et al. 2024**, FSI:G 69:102994 (doi:10.1016/j.fsigen.2023.102994) | ML critical review | n/a | "community ML 인식 부족" 명시 | ML 채택 갭 인용 |
| **Marsico & Amigo 2025**, FSI:G 76:103225 (doi:10.1016/j.fsigen.2025.103225) | AI trust/robustness/adversarial | n/a | trust를 *open problem*으로 제기, 방법 미제시 | "trust를 문제로 제기했으나 *우리가 방법을 준다*" |

> ✅ 검증완료(2026-06-04, CrossRef PII 왕복): de Barros Rodrigues et al. 2025 *Large-scale selection of highly informative MH for ancestry inference*, FSI:G 74:103153 (doi:10.1016/j.fsigen.2024.103153) — §4.7 직결 · Cai et al. 2024 *Systematic AISNPs screening/classification*, FSI 357:111975 (doi:10.1016/j.forsciint.2024.111975) · Podini et al. 2026 *Defining key criteria for MH locus selection (MH Working Group)*, FSI:G 83:103421 (doi:10.1016/j.fsigen.2026.103421).

## 3. ★ 3대 선제대응 (forensic 리뷰어 1순위 반박 — 필수)

리뷰어 핵심 질문: *"coverage 보장이 법정 weight-of-evidence(LR)와 어떻게 연결되고 SWGDAM/ISFG로 validatable한가? marginal coverage는 population-평균이라 per-population 신뢰 요구와 충돌하지 않나?"*

| # | 선제대응 | 현 상태 | 할 일 |
|---|---|---|---|
| **(a)** coverage ↔ **LR calibration 연결** | ✅ **완료** (Paper §2 + §6) | LR-calibration 계보 인용: **Ramos & González-Rodríguez 2013** FSI 230:156 (doi:10.1016/j.forsciint.2013.04.014) · **Hannig & Iyer 2022** JRSS-A 185(1):267 (doi:10.1111/rssa.12747). "scalar LR calibration ↔ set coverage" 보완 명시 |
| **(b)** **per-population(class-conditional) coverage 보고** | ✅ **완료·실측** (Paper §4.2, `scripts/trust/53`, 10-seed) | **반전 발견**: "Mondrian이니 설계상 보장"은 *틀림* — **권장 LogReg는 모든 집단 target 충족**(α=0.1: KHV 0.90; α=0.05: KHV 0.96)이나 **XGBoost는 marginal 0.888인데 KHV 0.70 붕괴**. → per-pop 신뢰성은 *base-model 의존* = **RQ-Ⅰ 재입증**. 정직하게 "marginal이 숨기는 subgroup 실패를 *드러내고*, 단순·보정 모델에서만 per-pop이 성립"으로 프레임 |
| **(c)** **validation-guideline 적합성** | ✅ **완료** (Paper §6) | 신설 "Forensic deployment and validation" §6: (i) LR 관계 (ii) SWGDAM/ISFG developmental·internal validation 적합(thin wrapper·α만 파라미터·per-pop coverage를 필수 readout으로 권고) (iii) reference-DB 전제 → investigative-lead 한정 |

→ 셋을 선제하면 "ML paper wearing forensic clothing" 인상을 차단. 안 하면 desk-level 회의론.

## 4. 재앵커: spine 재가중 (docs/05 3-act → FSI:G 버전)

**원칙: RQ-Ⅲ/Ⅱ(forensic 배치·trust)를 헤드라인, RQ-Ⅰ(base-model이 OSR 좌우)은 *method rationale*로 강등(삭제 아님).**

| 위치 | 현 draft (ML-우선) | FSI:G 재앵커 |
|---|---|---|
| Title/Abstract 1문장 | "open-set 신뢰도는 base-model이 좌우" | "**법정용 calibrated·abstaining fine-scale ancestry** — 분포-free coverage + out-of-reference 거부" |
| 헤드라인 기여 | RQ-Ⅰ (ML 놀라움) | **RQ-Ⅱ trust layer + RQ-Ⅲ 배치(패널·열화·코호트)** |
| RQ-Ⅰ 위치 | 핵심 발견 | "*왜 단순·보정 모델인가*"의 method note (§4.3–4.4, 압축) |
| LR engagement | (B)로 추가됨 | Related Work + Discussion 양쪽 강화 |

**정직-정확도 프레이밍**: 79.6%(저절대정확도)를 *약점*이 아니라 *기능*으로 — 과확신 단일라벨 대신 **통제된 오류율의 기권**(admissibility). Chen 95.6%는 9-클러스터 광역(직접비교 불가, docs/04 §13.1)임을 명시.

## 5. 두 논문 분리 (재확인)

- **Paper 1 → FSI:G**: 본 plan대로 재앵커. forensic 헤드라인 + LR engage + per-pop coverage + validation.
- **Paper 2 → ML/UQ(TMLR/NeurIPS/AISTATS)**: RQ-Ⅰ(ECE≠OSR, base-model이 OSR 좌우) 헤드라인 + CREE(docs/07). forensic은 응용 testbed.
- 같은 결과를 두 청중에 *재프레임*(데이터 확장 아님).

## 6. 제출 전 체크리스트 (FSI:G)

- [x] LR-vs-conformal Related Work 단락 (B, paperwork/Paper1_draft §2)
- [x] §4.2에 **집단별(Mondrian) coverage 표** 추가 — (b) 증거 (10-seed 실측, `scripts/trust/53`)
- [x] **SWGDAM/ISFG validation 적합성** — Paper 신설 §6 (c)
- [x] Abstract forensic 재앵커 (per-pop + 외부코호트 + LR-보완) — Title은 이미 forensic-anchored 유지
- [x] 신규 검증 citation을 docs/02 + Paper1 §References에 반영 (Ramos 2013·Hannig&Iyer 2022·Heinzel 2025·Wang 2025·Marsico 2025·Barash 2024)
- [x] first-mover 문장 정밀 scope("first conformal+OSR for forensic ancestry") — Intro §1 명시 + LR-calibration 계보(Ramos 2013·Hannig&Iyer 2022) 인정
- [x] §4.7 minimal panel 작성 (docs/04 §23 → 본문화: 정확도 곡선 + trust frontier 표, coverage ≥0.93 전 구간, 10–15× 축소) — **RQ-Ⅲ 본문 완결**
- [x] Snipper = **Phillips et al. 2007** FSI:G 1:273–280 (doi:10.1016/j.fsigen.2007.06.008) 확정 — Paper §References 반영
- [x] FSI PII 3건 DOI 확정(de Barros Rodrigues 2025·Cai 2024·Podini 2026) — Paper §References + docs/02 반영

**→ FSI:G 재앵커 체크리스트 전 항목 완료. 잔여 = §4.7 외 본문 미세조정·실제 저널 포맷팅(투고 시).**

## 7. 신규 검증 citation (추가 대상, DOI 확보)

- Ramos & González-Rodríguez 2013, *Reliable support: measuring calibration of likelihood ratios*, FSI 230(1–3):156 — doi:10.1016/j.forsciint.2013.04.014
- Hannig & Iyer 2022, *Testing for calibration discrepancy of reported likelihood ratios in forensic science*, JRSS-A 185(1):267 — doi:10.1111/rssa.12747
- Heinzel et al. 2025, *Advancing biogeographical ancestry predictions through machine learning*, FSI:G 79:103290 — doi:10.1016/j.fsigen.2025.103290
- Wang C. et al. 2025, *A biogeographical ancestry inference pipeline using PCA-XGBoost…Asian populations*, FSI:G 77:103239 — doi:10.1016/j.fsigen.2025.103239
- Marsico & Amigo 2025, *Ethical and security challenges in AI for forensic genetics*, FSI:G 76:103225 — doi:10.1016/j.fsigen.2025.103225
- Barash et al. 2024, FSI:G 69:102994 — doi:10.1016/j.fsigen.2023.102994 (이미 docs/02)
