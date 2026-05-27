# 2-Week Sprint Overview: Trustworthy Forensic-FM

**시작일**: 2026-05-26
**목표일**: 2026-06-09 (14일)
**팀**: 조민한 (AI 박사 과정) + Claude
**자원**: GB-10 × 2 (Node 1 / Node 2), sparkq 큐
**목표 산출**: 학기 final project + arxiv preprint draft

---

## 한 줄 message

> SSL pretraining (Option 4) + Conformal/Open-set UQ (Option 1)을 통합한 **"Trustworthy Forensic Foundation Model with Calibrated Uncertainty"** — Zhang 2025 리뷰가 forensic ancestry에 미적용임을 증명한 두 메서드의 첫 통합.

---

## 4개 Plan 구조

| # | Plan | 기간 | 산출 | 안전망 역할 |
|---|---|---|---|---|
| 1 | **Foundation + Baseline** | Day 1-3 | Leakage-free baseline + Wei 2025 phasing analysis | 필수 base |
| 2 | **Conformal + OSR (Option 1)** | Day 4-7 | CP/Mondrian + Energy OOD + LOPO eval | ⭐ Day 7 checkpoint — 단독 제출 가능 |
| 3 | **SSL + Multi-task (Option 4)** | Day 8-12 | Contrastive pretrain + multi-task heads | Novelty 확장 |
| 4 | **Integration + Paper** | Day 13-14 | 통합 system + 논문 draft | 제출 |

각 Plan은 단독 산출이 의미 있도록 설계 (실패 시 fallback).

---

## Plan 1 (Day 1-3) — 작성 완료

`2026-05-26-foundation-baseline.md` 참조.

핵심 목표:
- 환경 / 데이터 / git / sparkq 셋업
- 01_proposal_review.md의 P0 이슈 3개 해결 (leakage, diplotype, GroupKFold)
- Wei 2025 phasing error 분석 + Reliable-Ae metric

---

## Plan 2 (Day 4-7) — Plan 1 완료 후 작성

예상 핵심 task:
- Mondrian Conformal Prediction wrapping (mapie)
- Energy-based OOD score
- LOPO (leave-one-population-out) 평가 protocol
- KoVariome KOR 외부 검증 (Node 2 background result 사용)
- Coverage vs Set-Size trade-off curve

⭐ **Day 7 checkpoint**: 옵션 1만으로 학기 final 제출 가능 상태 도달.

---

## Plan 3 (Day 8-12) — Day 7 이후 작성

예상 핵심 task:
- SSL contrastive setup (augmentation 디자인이 핵심)
- 1000G WGS pretraining via CONFIG_SLICE on 2 nodes
- Multi-task heads (ancestry + sex + kinship score)
- Option 1과 통합 (SSL FM + CP wrapping)

위험: SSL이 baseline 못 이기면 옵션 1 단독으로 제출.

---

## Plan 4 (Day 13-14) — 최종 통합

예상 핵심 task:
- 모든 metric 표 + 5개 figure 정리
- Flask 도구 SSL+CP 버전 업데이트
- arxiv-style 논문 draft
- GitHub release

---

## 두 노드 분담 원칙

| Node 1 (alphabridge, 192.168.200.12) | Node 2 (nvidia, 192.168.200.13) |
|---|---|
| 메인 개발 환경 (interactive) | Background long-job (KoVariome, SSL sweep) |
| baseline + Option 1 단독 검증 | CONFIG_SLICE hyperparam sweep |
| Notebook / 분석 | Pretraining checkpoint 저장 |

모든 GPU job: `sparkq submit` 경유.

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| KoVariome variant call 시간 초과 | Medium — KOR 외부 검증 약화 | in silico KOR proxy (CHB+JPT mixture) |
| SSL이 baseline 못 이김 | Low — 옵션 1만 제출 | Day 7 checkpoint 도달 후 진행 |
| GB10 SM121 op 미지원 | Low — 학습 op는 안전 | dgx-spark 스킬 fallback 적용 |
| 시간 부족 | Medium | Plan 3 일부 → 후속 thesis로 분리 |

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
