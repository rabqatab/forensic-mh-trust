# 2-Week Sprint Overview: Trustworthy Forensic-FM

**시작일**: 2026-05-26
**목표일**: 2026-06-09 (14일)
**팀**: 조민한 (AI 박사 과정) + Claude
**자원**: GB-10 × 2 (Node 1 / Node 2), sparkq 큐
**목표 산출**: 학기 final project + arxiv preprint draft

> **REVISION 2026-05-29 (Opus 4.8)**: (1) **목표 유지·강화** — 사용자 결정으로 통합 **Trustworthy Forensic Model** (SSL foundation model + calibrated CP/OSR)을 "whatever the cost" 끝까지 추진. 14일은 hard deadline이 아니라 milestone 가이드로 완화하고, 필요 시 thesis 범위로 연장. SSL은 강등하지 않음 (Plan 3 = 정식 deliverable). (2) **KOR 외부 검증만 폐기** — Korean WGS 확보 불가 (사용자 명시). open-set "unknown"은 1000G 비-EAS superpopulation + LOPO로 대체. 이건 ambition 조정이 아니라 외부 데이터 제약. (3) risk register에 직접 경쟁 논문 + 타임라인 + chr22 chance-level 추가. (4) Plan 1 방법론 수정 (Reliable-Ae·phasing) 반영. (5) **설계 원칙 추가** — CP/OSR trust layer는 model-agnostic interface(`predict_proba` + optional embedding) 위에 구축 → baseline·SSL FM 모두 동일 layer로 wrapping. 따라서 Plan 2 산출물은 폐기물이 아니라 최종 system의 trust half.

---

## 한 줄 message (revised 2026-05-29)

> SSL pretraining (Option 4) + Conformal/Open-set UQ (Option 1)을 통합한 **"Trustworthy Forensic Foundation Model with Calibrated Uncertainty"** — Zhang 2025 (103 refs) 리뷰가 forensic genetics에 SSL·CP·OSR 모두 미적용임을 증명. 두 메서드의 첫 통합을 목표로 함 (사용자 결정: cost 불문 완수).
>
> 단, open-set "unknown"의 실증은 KOR 데이터 없이 1000G 비-EAS superpopulation + LOPO로 수행 (KOR 외부 검증 폐기 — 데이터 확보 불가).

---

## 4개 Plan 구조

| # | Plan | milestone 가이드 | 산출 | 역할 |
|---|---|---|---|---|
| 1 | **Foundation + Baseline** | ~Day 1-3 | Leakage-free baseline + Wei 2025 phasing analysis (Reliable-Ae) | 필수 base — **거의 완료** (점검 2026-05-29) |
| 2 | **Conformal + OSR (Option 1)** | ~Day 4-8 | model-agnostic CP/Mondrian trust layer + empty-set reject OSR + 비-EAS OOD + LOPO eval | ⭐ 독립 제출 가능 + **최종 system의 trust half** (재사용) |
| 3 | **SSL Foundation Model (Option 4)** | ~Day 9-? | Contrastive pretrain on 1000G WGS + multi-task heads (ancestry/sex/kinship) | Trustworthy Forensic Model의 representation half |
| 4 | **Integration + Paper** | 연장 가능 | Plan 2 trust layer로 Plan 3 FM wrapping → 통합 system + 논문 draft | 제출 (thesis 범위 연장 허용) |

빌드 순서 원칙: trust layer(Plan 2)를 **model-agnostic interface** 위에 먼저 만들어 baseline·FM 모두 wrapping. 따라서 Plan 2는 fallback이 아니라 최종 system의 절반. 14일 milestone은 가이드일 뿐 hard deadline 아님 — Trustworthy Forensic Model 완수가 우선.

---

## Plan 1 (Day 1-3) — 작성 완료

`2026-05-26-foundation-baseline.md` 참조.

핵심 목표:
- 환경 / 데이터 / git / sparkq 셋업
- 01_proposal_review.md의 P0 이슈 3개 해결 (leakage, diplotype, GroupKFold)
- Wei 2025 phasing error 분석 + Reliable-Ae metric

---

## Plan 2 — **작성 완료** (2026-05-29)

`2026-05-29-conformal-osr.md` 참조. 핵심 task:
- Split-conformal nonconformity score + **per-class (Mondrian) quantile** — mapie 의존 대신 numpy로 직접 구현 (버전 안정 + 교육적). mapie는 cross-check.
- Prediction-set 구성 + marginal·per-class coverage 측정 (synthetic으로 1−α 보장 검증)
- Coverage vs Set-Size trade-off curve (EAS MH)
- **Open-set = empty prediction set → reject** (CP에 직접 연결) + MSP/entropy baseline
- OOD 데이터 = **1000G 비-EAS superpopulation (EUR/AFR/SAS/AMR)** 을 동일 MH marker에서 추출 (외부 데이터 0개)
- LOPO (leave-one-EAS-pop-out) = near-OOD reject 평가
- 전부 **model-agnostic interface**(`predict_proba`)로 — Plan 3 FM에 그대로 재사용

⭐ 독립 제출 가능 상태 + 최종 Trustworthy Forensic Model의 trust layer 확정.

---

## Plan 3 — SSL Foundation Model (Plan 2 완료 후 작성)

예상 핵심 task:
- SSL contrastive setup (genomic augmentation 디자인이 핵심 risk)
- 1000G WGS pretraining via CONFIG_SLICE on 2 nodes (sparkq)
- Multi-task heads (ancestry + sex + kinship score)
- 학습된 representation을 Plan 2 trust layer interface(`predict_proba`/embedding)에 노출

방침 (2026-05-29 사용자 결정): SSL이 baseline 못 이겨도 폐기 안 함 — Trustworthy Forensic Model의 representation half로 끝까지 추진. cost 불문.

---

## Plan 4 — 최종 통합 (연장 허용)

예상 핵심 task:
- Plan 2 trust layer(CP/OSR)로 Plan 3 SSL FM wrapping = 통합 Trustworthy Forensic Model
- 모든 metric 표 + figure 정리 (coverage, set-size, OOD AUROC, LOPO reject)
- Flask 도구 통합 버전 업데이트
- arxiv-style 논문 draft + GitHub release

---

## 두 노드 분담 원칙

| Node 1 (alphabridge, 192.168.200.12) | Node 2 (nvidia, 192.168.200.13) |
|---|---|
| 메인 개발 환경 (interactive) | Background long-job (genome-wide MH 추출, SSL pretrain sweep) |
| baseline + Plan 2 trust layer 검증 | CONFIG_SLICE SSL hyperparam sweep |
| Notebook / 분석 | Pretraining checkpoint 저장 |

모든 GPU job: `sparkq submit` 경유. (KoVariome KOR 작업은 폐기 — Node 2는 genome-wide 추출 + SSL로 재배정.)

---

## Risk Register (revised 2026-05-29)

| Risk | Impact | Mitigation |
|---|---|---|
| **직접 경쟁 논문** (Human Genomics 2025, XGBoost East Asian incl. Korean; zygosity-aware DNA LM bioRxiv 2025-11) | **High — novelty 침식** | 차별화 vector를 명시적으로: (i) calibrated UQ/OSR layer (경쟁작 부재), (ii) Reliable-Ae phasing-aware marker 선택, (iii) forensic admissibility framing. baseline 정확도 경쟁이 아니라 **trustworthiness**가 contribution임을 논문에서 못박기 |
| **타임라인 — SSL+CP 통합 14일 초과** | **High (확실시)** | 사용자 결정: cost 불문 완수, deadline 연장 허용. milestone은 가이드. Plan 2(trust layer)가 독립 제출 가능하므로 학기 제출 안전망은 유지 |
| **chr22-only baseline = chance level** (0.24~0.30, 53 marker) | High — paper 정확도 story 부재 | genome-wide MH 추출을 명시 task로 승격 (Plan 1 Task 10 재정의). Plan 2/3는 marker-set agnostic하게 설계 |
| KOR 데이터 부재 | (제약, risk 아님) | KOR 외부 검증 폐기. open-set unknown은 비-EAS superpop + LOPO. KOR 부재는 논문 limitation으로 솔직히 기술 |
| SSL이 baseline 못 이김 | Medium | 폐기 안 함 — representation half로 유지. trust layer는 base model 무관하게 가치 |
| genomic augmentation 설계 미해결 (SSL) | Medium | Plan 3 진입 전 brainstorming. masked-SNP modeling을 1차 안전책으로 |
| GB10 SM121 op 미지원 | Low — 학습 op는 안전 | dgx-spark 스킬 fallback 적용 |

---

## 산출 디렉토리 구조

```
mh-eas-panel/
├── docs/                            # 분석·계획 문서
│   ├── superpowers/plans/           # 4개 plan
│   ├── 01_proposal_review.md        # 비판적 리뷰
│   ├── 02_literature_landscape.md   # 문헌 매핑
│   └── 03_novelty_options.md        # novelty 옵션
├── src/forensic_mh/                 # 라이브러리 코드
├── scripts/                         # 파이프라인 단계별
├── tests/                           # pytest
├── data/                            # 1000G, KoVariome (.gitignore)
├── results/                         # figure, metric 표
├── notebooks/                       # 탐색용 jupyter
├── pyproject.toml                   # uv 관리
└── README.md
```
