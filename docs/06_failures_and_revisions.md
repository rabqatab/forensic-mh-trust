# 06 — Failure cases & revisions (post-mortem log)

**작성일**: 2026-05-30
**목적**: 이 프로젝트에서 **무엇이 실패했고, 왜 실패했고, 어떻게 바꿨는지**를 한곳에 모은다. 과학적 주장 철회(A), 폐기·보류한 접근(B), 엔지니어링·프로세스 버그(C), 교훈(D). 각 항목은 RQ([`05`](05_research_questions.md))와 증거 섹션([`04`](04_experiments_and_results.md))에 연결된다.
**왜 따로 두나**: 개별 실패는 04/05에 흩어져 있다(부록 A, §11/§14 철회, §21 기각 등). 이 문서는 그것을 **단일 실패 서사**로 묶어, 논문의 Limitations·"lessons" 절과 리뷰어 선제 대응에 쓴다.

---

## A. 폐기·정정된 과학적 주장 (claim retractions)

"믿었던 것 → 실제 → 증거 → 어떻게 정정".

| # | 믿었던 주장 | 실제 | 정정 근거 | RQ |
|---|---|---|---|---|
| A1 | "≈57% 정확도는 **FST 천장**(동아시아 fine-scale의 생물학적 한계)" | 인코딩 아티팩트였음 | 동일 데이터 one-hot LogReg **79.6%** (§13, 부록 A) | RQ3 |
| A2 | "**XGBoost가 과확신**이라 OSR이 약하다(보정 불량)" | XGBoost가 ECE **최선**(0.077), RF가 최악(0.315) | §14. OSR 약점은 *보정*이 아니라 확률 *순위 분리*(AUROC) 문제 | RQ4 |
| A3 | "base는 **RandomForest** 권장"(ordinal 비교 기준) | LogReg(one-hot)이 정확도·OSR **모두** 우위 | §11/§20 (AUROC 0.840 vs RF 0.757) | RQ1 |
| A4 | "소수 마커 **최소 패널**로 고정확도 달성"(제안서 핵심 deliverable) | **이중 정정**: (1차) univariate MI/L1로는 compact 패널 부족(§21) → 한때 "패널 없음"으로 단정. (2차, **자기정정**) 그 단정도 약한 선택기 탓 — **다변량 선택**으로 패널이 살아남(25마커 52%·1000마커 76.8%·coverage 모든 N ≥0.93, §23). **RQ5 ANSWERED(재정의)**: 배치 가능 최소 패널 존재(10–15× 축소). 단 원안 "≥90%"는 도달 불가 | §21→§23 | RQ5 |
| A5 | "open-set 약점은 **방법(conformal) 한계**" | base-model 문제 — base 교체(한 줄)로 해결 | §11/§20 (empty-set 거부 0→59%) | RQ1 |
| A6 | "PCA/구조 피처가 도움"(Chen 2025 동기 가설) | MH one-hot엔 무익(최선 45.8% < raw 56.1%, −10p) | §8 | RQ3 |

> **메타 교훈**: A1·A2·A3·A5는 모두 **하나의 근원**(잘못된 인코딩)에서 파생된 연쇄 오판이었다. ordinal-tree를 기준으로 삼으니 "천장"(A1)·"트리가 최선/RF 권장"(A3)·"방법 한계"(A5)가 줄줄이 따라왔다. 인코딩 교정 하나가 네 주장을 동시에 뒤집었다.

---

## B. 시도했으나 실패·보류한 접근 (abandoned / deferred approaches)

"접근 → 왜 시도 → 왜 실패·보류 → 대체/현 위치".

| # | 접근 | 왜 시도 | 왜 실패·보류 | 대체 / 현 위치 | RQ |
|---|---|---|---|---|---|
| B1 | **Ordinal LabelEncoder**(트리용) | 명목형을 가장 간단히 수치행렬로 → tree 투입 | diplotype은 명목형인데 임의 순서 부여 → tree가 `x≤t` split로 **비인접 범주를 못 묶음**(XGB 52%, RF 57%) | one-hot(+linear) (부록 A) | RQ3 |
| B2 | **StandardScaler on one-hot** | "선형/거리 모델엔 스케일링 필요"는 통념 | one-hot 지시컬럼을 √(p(1−p))로 나눠 **rare 범주 과증폭** → LogReg 46.6%, kNN 21%로 붕괴 → "트리가 최선"이라는 오판 강화 | 권장 파이프라인에서 제거(79.6%/65%); 음성대조 arm으로만 보존(`scripts/15,19`) | RQ3 |
| B3 | **Energy-based OSR**(제안서 명시) | OOD 탐지 표준 기법 | XGBoost(tree)는 energy score를 산출하지 않음 → 부적용 | Conformal empty-set 거부 + MSP로 대체(제안서 정정) | RQ1 |
| B4 | **SSL Foundation Model**(masked + contrastive) | representation 학습 + 데이터 확장으로 선형 능가 기대 | 데이터 확장 후에도 실패 — gnomAD 4,091(clean) ablation에서 **LogReg 78% ≫ supervised 55 ≥ SSL 54**; scaling(583→4091)은 supervised 수준 회복에 그침(§25) | **ANSWERED-negative**: 병목은 데이터만이 아니라 모델 클래스. Paper 2 재정의("데이터로 FM 키우기"→"왜 선형이 근본 우월한가") | 비-RQ |
| B5 | **Reliable-Ae**(Wei 2025 phasing penalty) | 고-Ae 마커일수록 phasing-error 위험 → 신뢰도 보정 | 1000G 표준 release에 **완전 EAS trio 0개** → P_phase 추정 불가 | **deferred** — NYGC 30× (602 trios) 필요. 스크립트 hg38-ready(§2) | 비-RQ |
| B6 | **L1 sparse 축소 패널** | 공동 선별(L1)이 개별 MI 선별을 능가해 ~10× 축소 패널 기대 | 71–474 마커 모두 ≤57% — MI 선별과 동급/이하, 회복 실패 | "compact 패널 없음" **확정**(§21) | RQ5 |
| B7 | **Deep-ensemble / MC-dropout epistemic UQ** | 앙상블 불확실성으로 OOD 탐지 기대 | epistemic(MI) AUROC ≈ **0.36**(<chance) — OOD 미검출 | MSP/empty-set이 실효 신호(§14) | RQ4 |

---

## C. 엔지니어링·프로세스 실패 (bug → 원인 → fix)

| # | 증상 | 근본 원인 | 수정 | commit |
|---|---|---|---|---|
| C1 | conformal coverage 소표본 under-cover | `np.quantile`의 (n−1)-spacing이 유한표본 보장과 불일치 | order-statistic **k=⌈(n+1)(1−α)⌉** 직접 사용 | `1f02fe1` |
| C2 | genome-wide 다운로드 손상(chr8 등) | 부분 다운로드가 "완료"로 카운트 | per-chrom retry + **subset 연산을 무결성 검사로** + atomic `.tmp` 이동 | `36d040f` |
| C3 | OOD open-set 평가 overlap 0 | `discover_chrom_vcfs`가 기본 prefix `EAS_chr`로 OOD 디렉터리를 탐색 | `prefix="OOD_chr"` 전달 | `9a737d7` |
| C4 | FMVocab 비결정적 vocab | 동일 빈도 diplotype tiebreak 불안정 | 결정적 tiebreak + empty guard | `b34c0c3` |
| C5 | pretrain 테스트 flaky | seed 고정 *전*에 RNG 소비(encoder init 순서) | encoder init seed 고정 | `8f9944a` |
| C6 | ADO 테스트가 열화를 미검증 | het→hom collapse 단언 부재 | het collapse 강제 assert + RNG/worker caveat 문서화 | `85696e5` |
| C7 | HGDP 원격 추출 timeout(대형 염색체) | 원격 `bcftools -R`가 1200s 초과 | **resumable**(존재 시 skip) + timeout 3000s | `fb35f44` |
| C8 | HGDP 추출 `Xood` 행렬 오류 | 샘플 ID를 마커로 순회(`for m in sorted(orows)`) | `for m in names`로 수정 | `fb35f44` |
| C9 | 백그라운드 작업 "완료" 오판 | nohup-detached launcher 조기 반환을 종료로 오독 | foreground until-loop로 **완료 마커 폴링** 후 보고 (프로세스 교훈) | — |
| C10 | `python`/`pytest` 직접 실행 거부 | 프로젝트 uv 환경 미사용 | 전 스크립트 **`uv run`** 통일 | — |
| C11 | 패널 크기별 conformal **coverage가 N↑에 따라 붕괴**(0.91→0.60) | **선택-calibration 누수**: 마커 선택을 ConformalClassifier가 내부 calibration에 쓰는 데이터에서 수행 → cal 라벨이 score function에 누수, N 클수록 cal 과적합 → under-cover | **3-way split**(select / fit+cal / test, 서로 disjoint)으로 선택을 calibration과 분리 | `scripts/28` |

---

## D. 교훈 (Paper 1 / 후속에 반영)

1. **명목형 유전 마커 = one-hot(+정규화 선형)** — ordinal-LabelEncoder를 tree에 쓰면 성능이 조용히 깎인다. (RQ3) → [[project_encoding_lesson]]
2. **one-hot에 StandardScaler 금지** — 지시컬럼은 이미 동일 스케일; 스케일링이 rare 범주를 왜곡. (RQ3)
3. **신뢰 도메인의 모델 선택 기준은 리더보드(정확도)가 아니라 보정/분리(OSR)** — 같은 trust layer가 base만 바꿔 무용→유효. (RQ1)
4. **소표본에선 SSL/deep가 단순 baseline에 진다** — 데이터 확장 전엔 baseline-first; FM은 데이터가 준비된 뒤. (비-RQ/Paper 2)
5. **conformal은 order-statistic 구현, quantile 근사 금지.** (RQ2)
6. **원격 대용량 데이터 작업은 resumable·atomic·"연산 자체를 무결성 검사로".** (C2/C7)
7. **백그라운드 완료는 마커 폴링으로 확증** 후 보고 — 조기 반환을 완료로 읽지 말 것. (C9)
8. **방법론적 메시지**: 복잡한 모델·피처 이전에 **인코딩**이 fine-scale MH ancestry의 결정 변수였다 — 실패의 근원이자 기여의 핵심.

---

## 부록 — 실패가 곧 기여인 지점

이 프로젝트의 가장 강한 결과 일부는 **실패의 정직한 기록**에서 나왔다:
- A1(천장=아티팩트)·B1/B2(인코딩 함정) → 방법론적 기여(부록 A, "encoding이 결정 변수").
- A4/B6(최소 패널) → **자기정정의 사례**: 1차 "패널 없음" 단정을 스스로 의심하고 강한 선택기로 재검 → 패널이 살아남(§23). 실패 분석 자체가 잠정적임을 보이는 동시에, "선택기가 결론을 좌우한다"는 방법론 교훈을 남김(RQ5 PENDING).
- C7/C8(HGDP 추출 난항) → 외부 검증(RQ7)의 honest 한계(Dai n≈4, build 조화)로 Limitations에 반영.

→ 상세 인코딩 post-mortem은 [`04` 부록 A](04_experiments_and_results.md), RQ별 상태는 [`05`](05_research_questions.md) 참조.
