# 04 — Experiments & Results (comprehensive log)

**프로젝트**: Trustworthy Forensic Microhaplotype Model — 1000G East-Asian 5집단 분류 + calibrated UQ/open-set + SSL foundation model.
**최종 갱신**: 2026-05-29 (Opus 4.8)
**범위**: Plan 1(foundation/baseline) · Plan 2(conformal/OSR) · Plan 3a(SSL FM core). Plan 3b(FM 대규모 학습)는 미실행.
**재현**: 모든 코드 `main` 브랜치, uv 환경, torch 2.12.0+cu130. 결과 JSON은 `results/`(gitignored), 산출 코드/스크립트는 커밋됨. 72 pytest green.

> 한 줄 요약: **conformal coverage 보장은 작동하고 marker 수에 따라 prediction set이 좁아진다(성공). open-set recognition은 base model(XGBoost)의 OOD 과확신 때문에 약하다(far-OOD AUROC 0.695 포화, near-OOD 미검출) — 이것이 Plan 3 SSL FM의 핵심 동기.**

---

## 0. 데이터셋

### 0.1 1000 Genomes Phase 3 — EAS in-distribution
- 빌드: **hg19** (phase3 v5b genotype VCFs). 좌표는 MicroHapDB `Positions37`.
- EAS 504명, 5집단: **CDX 93, CHB 103, CHS 105, JPT 104, KHV 99**.
- 추출: `scripts/01_download_1000g.sh`(다운로드) → `02`(chr22 EAS subset) → `06_download_genome_wide.sh`(genome-wide, disk-efficient: full VCF 받아 EAS+OOD subset 추출 후 삭제, per-chrom retry + .tmp 원자적 이동).

### 0.2 Out-of-distribution (open-set "unknown")
- **far-OOD**: 1000G 비-EAS superpopulation **300명** = EUR/AFR/SAS/AMR 각 75명(결정적, `OOD_samples.txt`). KOR 데이터 부재로 비-EAS를 unknown 대용으로 사용.
- **near-OOD**: LOPO(leave-one-EAS-population-out) — 5집단 중 1개를 학습에서 제외, 그 집단을 unknown으로 검정.

### 0.3 Marker 패널 (MicroHapDB, hg19)
| panel | markers | 용도 |
|---|---|---|
| chr22 only | 53 | pipeline 검증 / smoke |
| chr1–5 | 1058 | 중간 점검 |
| **genome-wide (22 autosomes)** | **3042** | 정식 결과 |

### 0.4 Trio 데이터 벽 (phasing 재현 불가)
- `g1k.ped`에 641 trio. 그러나 phase3 **표준 release(2504)는 unrelated subset** → complete trio **0개**.
- related-samples VCF(31명) 합쳐도 complete trio **6개(전부 non-EAS)** → 통계 무의미.
- Wei 2025(602 trios) 재현은 **NYGC 30x high-coverage release(3202, 602 trios, GRCh38)** 필요 → **deferred**.

---

## 1. Experiment 1 — Leakage-free baseline (Plan 1)

**목표**: 제안서 P0 결함을 수정한 leakage-free 분류 baseline 구축.

**방법**
- VCF → **diplotype 추출**(`data/vcf_io.py`): 두 haplotype 모두 보존, canonical(sorted) tuple. (원안의 `gt[0]`-only는 heterozygote 정보 손실 — P0 #2 수정.)
- per-marker LabelEncoder로 정수 인코딩 → (n_samples × n_markers) 행렬(`pipelines/baseline.py`).
- **leakage-free nested CV**(`eval/nested_cv.py`): outer fold 안에서 XGBoost feature importance로 top-N 재선택 후 재학습·평가. (원안은 전체 데이터로 selection 후 CV → 라벨 leakage, P0 #1 수정.)
- relatedness: GroupKFold scaffold(`eval/grouping.py`) 존재하나 1000G EAS가 사실상 unrelated이라 baseline엔 미적용(P0 #8: scaffold만).

**결과 — chr22 nested-CV 정확도** (`results/baseline/chr22_baseline.json`, 5-class chance=0.20)
| panel size (top-N) | mean acc | std |
|---|---|---|
| 5 | 0.236 | 0.047 |
| 10 | 0.292 | 0.045 |
| 20 | 0.298 | 0.027 |
| 30 | 0.274 | 0.026 |
| 53 (all) | 0.298 | 0.044 |

→ **chr22-only는 near-chance**. accuracy-vs-panel-size가 평탄(0.29 plateau) — chr22 정보량 부족. pipeline 정확성(leakage-free) 검증이 목적이고, 정확도는 genome-wide에서 의미.

**상태**: ✅ pipeline 작동, P0 #1/#2 해결, #8 scaffold. genome-wide plain-accuracy는 별도 미산출(아래 §3 conformal set-size가 간접 지표; 필요 시 `scripts/03`을 genome-wide로 1회 재실행).

---

## 2. Experiment 2 — Wei 2025 phasing / Reliable-Ae (Plan 1)

**목표**: high-Ae marker가 phasing error에 취약하다는 Wei 2025 주장을 재현, **Reliable-Ae = Ae × (1 − P_phase_error)** 산출.

**방법**
- **Ae(EAS)** = 1/Σpᵢ² (haplotype 빈도, inverse Simpson), EAS subset에서 직접 계산(`metrics/reliable_ae.compute_ae`).
- **P_phase_error**: trio Mendelian-consistency 위반율, 단 **informative meiosis(자녀 heterozygous)만 분모**(homozygous 자녀는 switch error 검출 불가 → 0 편향 방지; `is_informative_meiosis`).

**결과 — chr22 Ae** (`results/baseline/chr22_reliable_ae.json`, 53 markers)
- Ae(EAS): **mean 4.91, max 17.44**, top-5 = [17.4, 13.6, 9.7, 9.7, 8.4].
- P_phase_error / Reliable-Ae: **null (deferred)** — complete trio < MIN_TRIOS(30) (실제 6개, 비-EAS).

**해석**: Ae 표 자체가 Wei의 위험 노출도 — 고-Ae marker(naive selector가 선호)가 가장 phasing-risky. 정량 penalty는 NYGC trio 확보 후. 스크립트는 hg38-ready(`TRIO_VCF_SOURCES` append + `BUILD="hg38"`).

**상태**: △ Ae ✅ / phasing penalty deferred(NYGC).

---

## 3. Experiment 3 — Conformal prediction coverage (Plan 2)

**목표**: 분포-free prediction set + per-class(Mondrian) coverage 보장.

**방법** (`uq/conformal.py`, `uq/conformal_classifier.py`)
- base model: **XGBoost**(200 trees, depth 4, lr 0.1), cal split 0.3, seed 42.
- nonconformity(LAC): sᵢ = 1 − p̂(yᵢ|xᵢ). threshold = **k-th 최소값, k=⌈(n+1)(1−α)⌉** (order statistic — np.quantile 아님, 소표본 보장 정확).
- **Mondrian**: 클래스별 quantile q_k → per-class coverage 보장. prediction set C(x)={k: 1−p̂(k|x) ≤ q_k}.

**결과 — coverage vs set-size** (`results/conformal/coverage_curve.json`)

chr22 (53 markers):
| α | target | coverage | set size |
|---|---|---|---|
| 0.30 | 0.70 | 0.684 | 2.86 |
| 0.20 | 0.80 | 0.757 | 3.24 |
| 0.10 | 0.90 | 0.849 | 3.79 |
| 0.05 | 0.95 | 0.921 | 4.14 |

**genome-wide (3042 markers)**:
| α | target | coverage | set size |
|---|---|---|---|
| 0.30 | 0.70 | 0.770 | 2.09 |
| 0.20 | 0.80 | 0.809 | 2.28 |
| 0.10 | 0.90 | 0.888 | 2.61 |
| 0.05 | 0.95 | **0.954** | 3.16 |

**해석**: ✅ marginal coverage가 모든 α에서 1−α 근접(α=0.05는 보수적). **marker↑ → set tighter** (@α=0.1: set size **3.79 → 2.61**, 53→3042 marker). 정보량이 늘면 같은 보장에 더 좁은 set = 분포-free UQ의 바람직한 거동.

**상태**: ✅ 성공.

---

## 4. Experiment 4 — Open-set: far-OOD (비-EAS superpopulation) (Plan 2)

**목표**: 패널 밖 집단(비-EAS)을 reject. 두 신호: (i) **conformal empty-set**(set이 비면 unknown), (ii) **MSP**(1−max softmax prob).

**방법** (`uq/openset.py`, `scripts/11`): EAS·OOD를 동일 chromosome 교집합에서 수집, 공유 `DiplotypeEncoder`(unseen→reserved code)로 인코딩. 지표: empty-set reject rate(in-dist/OOD), MSP AUROC, FPR@95TPR.

**결과** (`results/conformal/openset_ood.json`, genome-wide)
- OOD unseen-diplotype fraction: **0.132** (데이터에 실제 OOD 신호 존재).
- **MSP AUROC: 0.695** / FPR@95TPR: **0.75**.
- empty-set reject rate: in-dist **0.0**, OOD **0.0** (모든 α).

**marker 수에 따른 MSP AUROC 추이** (paper용 핵심 그림)
| panel | markers | MSP AUROC |
|---|---|---|
| chr22 | 53 | 0.50 (chance) |
| chr1–5 | 1058 | 0.67 |
| genome-wide | 3042 | **0.695** |

**해석**: ⚠️ MSP 신호는 marker↑로 상승하나 **~0.70 포화**. empty-set reject는 **전혀 발화 안 함** — XGBoost가 비-EAS도 어떤 EAS 집단으로 **과확신** 분류 → set이 비지 않음. FPR@95=0.75는 forensic 사용 부족.

**상태**: ~ moderate / base-calibration 한계.

---

## 5. Experiment 5 — Open-set: near-OOD (LOPO) (Plan 2)

**목표**: 패널에 없는 **근연** 집단(가장 어려운 unknown)을 reject. KOR-부재의 정직한 대용.

**방법** (`eval/lopo.py`, `scripts/12`): 5집단 중 1개 hold-out, 나머지 4집단으로 학습·conformal calibration, hold-out 집단의 empty-set reject rate를 known과 비교. α=0.10.

**결과** (`results/conformal/lopo_nearood.json`, genome-wide)
| held-out | reject (known) | reject (held-out) | gap |
|---|---|---|---|
| CDX | 0.0 | 0.0 | 0.0 |
| CHB | 0.0 | 0.0 | 0.0 |
| CHS | 0.0 | 0.0 | 0.0 |
| JPT | 0.0 | 0.0 | 0.0 |
| KHV | 0.0 | 0.0 | 0.0 |

**해석**: ❌ near-OOD 미검출. 근연 EAS 집단은 학습된 집단으로 과확신 오분류 → empty-set으로 안 잡힘. (far-OOD보다 본질적으로 어렵고 + base model 과확신.)

**상태**: ❌ 미검출 — Plan 3 동기.

---

## 6. Experiment 6 — SSL FM core 검증 (Plan 3a, synthetic)

**목표**: SSL foundation model의 **코어 모듈을 단위 검증** + trust layer 통합 계약 증명. (실데이터 대규모 학습은 Plan 3b.)

**구성** (`src/forensic_mh/fm/`): FMVocab(per-marker top-K, MASK slot), MHMatrixDataset(masked view + ADO het→hom·dropout contrastive views), objectives(masked-marker CE + NT-Xent), MHTransformer(weight-tied masked head), heads(ancestry/sex/kinship pairwise), ssl_pretrain + multitask_finetune, ForensicFMClassifier(sklearn BaseEstimator).

**결과 (synthetic 단위 실험)**
- `ssl_pretrain` (16 synthetic samples, 8 epochs): loss **11.47 → 6.52** (학습 발생 확인).
- `multitask_finetune` (64 synthetic, marker-0 = label, 20 epochs): ancestry_acc **0.36 → 0.92** (deterministic, seed 고정).
- **trust layer 통합 계약**: `ConformalClassifier(base_estimator=ForensicFMClassifier(...))`가 clone+fit+predict_set 통과 — **Plan 2 layer가 FM을 변경 없이 wrapping** 실증.
- kinship head는 대칭성만 검증, **학습 안 함**(Plan 3b 경계).

**상태**: ✅ 코어 빌드·테스트 완료(72 tests green), main 머지. 실데이터 성능은 미측정.

---

## 7. 종합 결과 & 핵심 발견

1. **Coverage(작동) vs OSR(미흡)의 분리** — 핵심 메시지.
   - conformal coverage 보장은 base model이 약해도 유지되고 marker↑로 set이 좁아진다(Exp 3, ✅).
   - OSR 분리력은 base model **calibration에 의존** — XGBoost는 OOD 과확신 → empty-set reject 미발화(Exp 4·5), MSP AUROC 0.69 포화.
2. **이것이 Plan 3(SSL FM)의 falsifiable 가설**: contrastive+ADO invariance로 OOD 과확신을 줄이면 OSR이 XGBoost 천장(0.69)을 넘는가? 동일 trust layer, `base_estimator`만 교체 + scripts 10/11/12 재실행으로 직접 비교.
3. **운영점**: open_set_decision의 OR 규칙에서 empty-set=0이므로 실질적으로 **MSP threshold가 reject 좌우**. FPR@95=0.75 → msp_threshold sweep로 운영 곡선 산출 필요.
4. **marker 수가 모든 것을 개선**: accuracy plateau 탈출, set tighter, MSP AUROC 상승 — genome-wide 추출이 결정적이었음.

---

## 8. 한계 (논문 명시)

- **KOR 데이터 부재**: open-set unknown을 비-EAS superpop + LOPO로 대용. Korean 직접 검증 불가.
- **Trio phasing 재현 불가(표준 release)**: Reliable-Ae penalty deferred → NYGC 30x 필요.
- **OSR 약함**: far-OOD AUROC 0.695(포화), near-OOD 미검출, FPR@95 0.75. base model 한계.
- **GroupKFold 미적용**: EAS가 사실상 unrelated이라 영향 미미하나 일반화 시 필요.
- **FM 실성능 미측정**: Plan 3a는 synthetic 단위 검증까지. genome-wide 학습은 Plan 3b.
- **직접 경쟁작**: Human Genomics 2025(XGBoost EAS incl. Korean), zygosity-aware DNA LM(bioRxiv 2025-11). 차별점 = calibrated UQ/OSR + Reliable-Ae + forensic framing.

---

## 9. 재현 방법 (commands)

```bash
# 환경
uv pip install -e ".[dev]"
uv run python -m pytest -q          # 72 tests

# 데이터 (genome-wide EAS 504 + 비-EAS OOD 300)
bash scripts/01_download_1000g.sh
bash scripts/06_download_genome_wide.sh        # 22 chrom, EAS+OOD subset

# Plan 1
uv run python scripts/03_run_baseline.py       # chr22 nested-CV 정확도
uv run python scripts/04_wei2025_phasing.py    # EAS Ae 표 (phasing deferred)

# Plan 2 (genome-wide 정식)
uv run python scripts/10_conformal_curve.py    # coverage vs set-size
uv run python scripts/11_openset_ood.py        # far-OOD (비-EAS)
uv run python scripts/12_lopo_nearood.py       # near-OOD (LOPO)
```

산출: `results/baseline/{chr22_baseline,chr22_reliable_ae}.json`, `results/conformal/{coverage_curve,openset_ood,lopo_nearood}.json`.

---

## 10. 다음 실험 (제안)

1. **Plan 3b — FM vs XGBoost OSR 비교 (핵심 가설 검정)**: genome-wide SSL pretrain(all-2504 − 1 superpop hold-out) + multi-task finetune → `ForensicFMClassifier`로 scripts 10/11/12 재실행. 가설: FM이 OOD 과확신을 줄여 MSP AUROC > 0.695, empty-set reject > 0, LOPO gap > 0.
2. **msp_threshold sweep**: open_set_decision 운영점 곡선(FPR vs reject rate).
3. **genome-wide plain accuracy**: `scripts/03`을 genome-wide로 1회 실행해 top-1 정확도 정식 기록.
4. **genome-wide Ae 표**: `scripts/04`를 전 chromosome로 확장(현재 chr22).
5. **NYGC trio**: Reliable-Ae phasing penalty 정식 산출.

> 도표화는 `academic-plotting` 스킬로 §3 coverage curve, §4 AUROC 추이, §1 accuracy-vs-panel-size를 figure로 생성 가능.
