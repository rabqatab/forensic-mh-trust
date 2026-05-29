# Plan 3 Design — SSL Forensic Foundation Model (multi-task + trust-layer ready)

**작성일**: 2026-05-29
**작성자**: 조민한 + Claude (Opus 4.8)
**상태**: design (brainstorming 산출). 승인 후 writing-plans로 구현 plan 작성.

---

## Goal

Microhaplotype(MH) profile 위에서 **self-supervised foundation model**을 사전학습하고, ancestry·sex·kinship **multi-task** 로 fine-tune하여, Plan 2의 conformal/open-set **trust layer가 그대로 wrapping**할 수 있는 calibrated forensic 모델을 만든다. 최종 산출 = Trustworthy Forensic Model의 representation half.

## Locked decisions (brainstorming)

| 축 | 결정 |
|---|---|
| Substrate | genome-wide **MH diplotype matrix** (raw WGS DNA-LM 아님 — 경쟁작 회피) |
| SSL objective | **masked-marker + contrastive** (dual) |
| Contrastive augmentation | random marker dropout + **simulated allele-dropout (ADO)** — forensic 노이즈 모델 자체를 augmentation으로 |
| Multi-task heads | **ancestry(5-way) + sex(binary) + kinship(pairwise)** |
| Training | SSL pretrain → multi-task fine-tune |
| Pretrain pool | **2504 − 1개 비-EAS superpop**(far-OOD hold-out용; 아래 "Open-set integrity" 참조) MH 위치 한정 추출, unsupervised → ancestry는 EAS 504로 fine-tune |
| Compute | **single GB10** via sparkq (데이터가 작아 multi-node 불필요) |
| Integration | encoder가 `predict_proba`(ancestry) + `embed()` 노출 → Plan 2 `ConformalClassifier`가 wrapping |

## Open-set integrity under all-2504 pretraining (중요)

all 2504로 unsupervised pretrain하면 비-EAS superpop이 표현공간에 이미 노출되어 **Plan 2의 far-OOD(비-EAS) 검정이 오염**된다 (더 이상 "unknown" 아님). 해결:
- **한 비-EAS superpop을 pretraining에서 완전히 제외**(예: AMR ~347명)하여 **진짜 far-OOD hold-out** 으로 남긴다. 나머지(EUR/AFR/SAS)는 pretrain에 포함 가능.
- pretrain pool = 2504 − (held-out superpop). near-OOD는 **LOPO(EAS)** 가 담당(label 미노출이면 sample이 pretrain에 있어도 "novel class" 검정으로 유효).
- 즉 OOD 검정 2종: ① held-out superpop(표현·label 모두 미노출 = 강한 far-OOD) ② LOPO EAS(label 미노출 = near-OOD).

## Data reality (전 설계를 제약)

genome-wide matrix ≈ **2504 samples × ~4,000–5,000 markers** (chr1–5에서 1,058). features ≫ samples (wide) → deep model 과적합 위험 크고 SSL도 data-starved. 따라서: **작은 모델(2–4 layer) + 강한 정규화**, SSL 이득은 정직하게 검증(베이스라인 못 이길 수 있음 — 사용자 수용). all-2504 pretrain pool이 이 위험의 주 완화책.

---

## Architecture

### Encoder — `MHTransformer` (small)
- 입력: 한 sample의 diplotype code 벡터 (length = n_markers), `DiplotypeEncoder`로 정수 인코딩.
- 토큰화: marker j의 값 → **per-marker value-embedding** + **learned marker-positional embedding**. (marker마다 diplotype vocab이 달라 per-marker embedding table; unseen code는 reserved row.)
- `[CLS]` 토큰 → pooled **sample embedding** (d_model, e.g. 128).
- 2–4 transformer encoder block (작게), dropout 강하게.
- 출력: `embed(X) -> (n_samples, d_model)`; masked head용 per-marker logits.

### SSL objectives — `objectives.py`
- **Masked-marker**: marker 토큰의 ~15%를 `[MASK]`로 치환 → 각 masked marker의 diplotype class를 per-marker classification head로 예측. Loss = mean CE over masked positions.
- **Contrastive (NT-Xent)**: 한 sample에 두 augmentation view (① random marker dropout p≈0.15 ② simulated ADO: het diplotype를 일정 확률로 homozygote로 붕괴) → positive pair, batch 내 나머지는 negative. Loss = NT-Xent(temperature τ).
- Pretrain loss = `L_mask + λ·L_contrastive`.

### Multi-task heads — `heads.py` (공유 trunk 위)
- **AncestryHead**: embedding → 5-way softmax. `predict_proba`의 원천.
- **SexHead**: embedding → binary (panel `gender`).
- **KinshipHead (pairwise/Siamese)**: 두 sample embedding (|e_i − e_j|, e_i⊙e_j) → related/unrelated binary. 학습쌍 = pedigree(parent-child·sibling) positive + random unrelated negative.

### Trust-layer adapter — `sklearn_api.py`
- `ForensicFMClassifier`: `.fit()` (pretrain+finetune 또는 사전학습 체크포인트 로드), `.predict_proba(X)` (ancestry head), `.embed(X)`. sklearn estimator 호환(`clone` 가능)하여 **Plan 2 `ConformalClassifier(base_estimator=ForensicFMClassifier(...))`** 가 변경 없이 동작.

---

## Components & interfaces (isolation)

```
src/forensic_mh/fm/
  dataset.py      # MHMatrixDataset: matrix→tensors; masking; augment(dropout, ADO). 순수 tensor 변환.
  encoder.py      # MHTransformer: forward(x)->per-marker logits; embed(x)->(N,d). 모델만.
  objectives.py   # masked_marker_loss(logits, targets, mask); nt_xent(z1, z2, tau). 순수 함수.
  heads.py        # AncestryHead/SexHead/KinshipHead. nn.Module, 입력=embedding.
  pretrain.py     # ssl_pretrain(encoder, loader, cfg)->checkpoint. 학습 루프.
  finetune.py     # multitask_finetune(encoder, heads, loaders, cfg). 학습 루프.
  sklearn_api.py  # ForensicFMClassifier(predict_proba, embed). trust-layer 어댑터.
scripts/
  20_extract_all_samples_mh.sh  # all-2504 MH-위치 한정 추출 (remote-region tabix, 작음)
  21_pretrain_fm.py             # sparkq SSL pretrain
  22_finetune_multitask.py      # sparkq multi-task fine-tune
  23_fm_conformal_eval.py       # ForensicFMClassifier→ConformalClassifier로 Plan 2 metric 재실행
```

**경계 원칙**: `objectives.py`/`dataset.py`는 순수(테스트 쉬움); `encoder.py`/`heads.py`는 모델만; 학습 루프는 분리; `sklearn_api.py`가 Plan 2와의 유일한 결합점(`predict_proba`).

## Data flow

```
all-2504 MH-restricted VCFs ──(DiplotypeEncoder, build_genome_wide_matrix)──▶ X_all (2504×M)
   │                                                                              │
   └── EAS 504 (labels: pop, sex) + pedigree pairs (kinship)                      ▼
                                                              MHMatrixDataset ──▶ ssl_pretrain (masked+contrastive)
                                                                                     │ encoder ckpt
                                                                                     ▼
                                          multitask_finetune (ancestry/sex/kinship heads)
                                                                                     │
                                                                                     ▼
                                       ForensicFMClassifier.predict_proba / .embed
                                                                                     │
                                                          Plan 2 ConformalClassifier ─▶ coverage / OSR / LOPO
```

## Testing strategy (TDD)

순수/소형 합성 텐서로 GPU·실데이터 없이 빠르게:
- `dataset`: mask fraction ≈ 0.15; ADO가 het만 붕괴; augment shape 보존.
- `objectives`: masked CE가 masked 위치에만; nt_xent에서 positive가 random negative보다 가깝게(합성).
- `heads`: 출력 shape; kinship pair feature 대칭성(|e_i−e_j| symmetric).
- `kinship pair builder`: pedigree에서 positive pair 추출 + negative sampling 균형.
- `sklearn_api`: `predict_proba` shape=(N,5), 합 1; **`ConformalClassifier(ForensicFMClassifier()).fit().predict_set()`가 동작** (통합 계약).
- 학습 루프: 1-step smoke (loss 유한, backward OK) on synthetic.

## Compute / infra

- **single GB10** (sparkq submit). 데이터 작아 tensor/multi-node parallel 불필요.
- dgx-spark 주의: `NVIDIA_DISABLE_REQUIRE=1`, **bf16** (FP8는 SM121 미지원), Docker GPU training 패턴.
- 체크포인트는 Node 간 공유 경로에 저장. sparkq `status`/`history` 선확인.

## Risks & mitigations

| Risk | 완화 |
|---|---|
| **N≪features 과적합 / SSL이 XGBoost 못 이김** | all-2504 pretrain, 작은 모델+dropout, 정직한 비교표. 못 이겨도 trust-layer 통합 + multi-task 자체가 기여 (사용자 수용) |
| kinship label 희소(EAS trio 거의 0) | pedigree 전체(글로벌) pair로 학습, EAS는 평가. kinship은 head 중 가장 약할 수 있음 → 별도 보고 |
| all-2504 추출 비용 | MH 위치 한정(~5000 SNP) remote-region 추출 = 작음. full VCF 재다운 불필요 |
| contrastive augmentation이 ancestry 신호 파괴 | ADO 강도 보수적으로; ancestry head val 정확도로 모니터 |
| GB10 학습 op 이슈 | dgx-spark 스킬 fallback |

## Out of scope (YAGNI)

- raw WGS DNA language model (경쟁작 회피, 다른 substrate)
- multi-node distributed training (데이터 작음)
- 새 forensic task 추가(body-fluid 등) — ancestry/sex/kinship로 한정

## Plan 2와의 관계 (왜 폐기물 아님)

`ForensicFMClassifier`가 `predict_proba`를 노출하는 순간, Plan 2의 `ConformalClassifier`·open-set·LOPO 스크립트가 **base_estimator만 교체**하면 그대로 적용된다. 즉 Plan 4(통합)는 거의 "estimator 교체 + 재실행". 이것이 trust layer를 model-agnostic하게 먼저 만든 이유.
