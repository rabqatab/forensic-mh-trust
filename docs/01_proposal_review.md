# 제안서 비판적 리뷰 (v1)

**검토자**: 조민한 (AI 박사 과정)
**대상**: `AIandDigitalForensic_project.docx` (임수연 외, 2026-05-25)
**검토일**: 2026-05-26

---

## 한 줄 요약

> "1000 Genomes Phase3 EAS 504명에서 MicroHapDB의 MH 좌표 ~412개를 추출 → Ae≥3 ∧ FST≥0.05로 1차 필터 → XGBoost Feature Importance로 2차 선별 → 정확도 vs 마커 수 곡선을 그려 '90% 정확도를 내는 최소 N'을 보고하고 Flask 웹 도구로 배포."

연구 질문은 명확하고 데이터/툴체인도 재현 가능하다. 다만 ML/통계 방법론에 그대로 두면 안 되는 결함이 8개 있으며, 그중 4개는 reviewer가 가장 먼저 지적할 **치명적 누수·오용**이다.

---

## 🔴 치명적 문제 (수정 없으면 결과 자체가 의심받음)

### 1. Feature-selection leakage — §4-5 (177–225줄)

```python
model_full.fit(X.values, y_enc)            # 전체 데이터로 importance 계산
importances = model_full.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
# ... 그 다음에 cross_val_score로 평가
```

**문제**: 전체 라벨을 보고 마커를 선별한 뒤 같은 데이터를 CV로 평가 → 보고되는 90% 정확도는 낙관적 편향(optimistic bias).

**수정**: nested CV. outer fold마다 train fold에서만 Feature Importance를 다시 뽑고 top-N을 선택. 또는 `Pipeline`으로 `SelectFromModel(XGB)` → `XGB`를 묶고 `cross_val_score`.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

pipe = Pipeline([
    ('select', SelectFromModel(XGBClassifier(...), max_features=N)),
    ('clf', XGBClassifier(...))
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
```

---

### 2. Diploid 정보의 절반을 버리고 있음 — §4-3 (94–125줄)

```python
gt = rec.samples[sample]['GT']            # (0, 1) — 두 haplotype
allele = rec.alleles[gt[0]] if gt[0] is not None else 'N'  # 하나만 취함
genotypes[sample].append(allele)
```

**문제**: phase 정보가 있는 1000G(shapeit2)인데 **첫 번째 haplotype의 allele만 가져온다**. 즉 모든 heterozygote 정보가 사라지고, 사실상 haploid 분석으로 축소된다. MH의 본질은 "phased haplotype 단위로 보는 것"이므로 이건 MH 분석이 아니라 SNP 단위 분석이 된다.

**수정**:
```python
hap1 = '-'.join(rec.alleles[gt[0]] for ...)   # phase=0
hap2 = '-'.join(rec.alleles[gt[1]] for ...)   # phase=1
diplotype = tuple(sorted([hap1, hap2]))       # unordered diplotype
```

그리고 encoding은 LabelEncoder가 아니라 (a) one-hot per haplotype 또는 (b) haplotype frequency 기반 numeric encoding이 적절. LabelEncoder의 정수 라벨은 XGBoost split에서 잘못된 순서 정보를 준다.

---

### 3. CHB ↔ CHS 혼동이 5-way 정확도를 misleading하게 만듦

CHB(베이징 한족) ↔ CHS(남부 한족) 사이 FST는 보통 0.001~0.005다. 이 두 클래스를 합쳐도 KHV/JPT/CDX 분리는 쉬워지므로, 90% 정확도는 "CHB↔CHS 혼동을 제외하면" 쉽게 나온다.

**반드시 보고할 것**:
- per-class precision/recall/F1
- 5×5 confusion matrix
- **하드 페어(CHB↔CHS)만 떼낸 2-class accuracy** — 이게 진짜 평가지표
- 또는 Han Chinese (CHB+CHS) 합친 4-way도 추가 보고

---

### 4. Wei et al. 2025 — 같은 질문에 이미 답한 논문이 코멘트로 와있음

Comment #0: *"Are microhaplotypes derived from the 1000 Genomes Project reliable for forensic purposes?"* (Wei, Li, Zhu, FSI:Genetics, 2025).

**주요 결론**: 1000G trio 602쌍에서 부모-자녀 mismatch로 phasing error 검출. **전체 0.07%**이지만 **Ae·In이 높을수록 error 확률 증가, 대륙별로 다름**.

**함의**: 제안서의 "Ae≥3로 필터" 전략은 **가장 phasing error가 많은 마커를 선택하게 됨** → 모델이 학습하는 패턴이 진짜 인구 신호인지 phasing 오류 패턴인지 구분 불가.

**필수 대응**: Introduction의 motivation 첫 단락은 Wei et al. 2025에 대한 명시적 응답이어야 한다. *"Wei et al.이 X를 지적했고, 우리는 Y로 그 한계를 우회 또는 해결한다"* 형식 위치(positioning) 필요.

---

## 🟡 심각한 문제 (반드시 보강)

### 5. KOR 부재 — Limitation만으로는 부족

한국에서 진행되는 "AI와 디지털포렌식" 수업 final project인데 정작 한국인이 없는 분류기. 두 가지 보강 옵션:

- **(A) 데이터 추가**: KoVariome (50 KPGP WGS, [PMC5885007](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5885007/)) 또는 GenomeAsia 100K를 KOR proxy로 추가. 단 1000G와 batch 차이 주의 (라이브러리 prep, variant calling pipeline 다름).
- **(B) Counterfactual 분석**: 기존 문헌(KoVariome) FST 기반 위치 추정 + 본 모델의 probabilistic output이 KOR 샘플에 대해 어떻게 나오는지 case study.

(B)만으로도 "한국 학생이 한국 데이터 한계를 어떻게 처리했는가"라는 서사가 살아난다.

---

### 6. Feature Importance의 불안정성

XGBoost의 `feature_importances_`는 (1) 상관된 feature 사이에 importance를 임의 분배하고, (2) gain 기준은 high-cardinality feature를 과대평가한다. MH는 본질적으로 LD(linkage disequilibrium)에 묶여 있어 상관이 높다.

**수정**:
- SHAP value + permutation importance + bootstrap stability
- 50회 bootstrap에서 top-N 마커의 Jaccard overlap 보고
- 이게 forensic panel "신뢰성"의 정량 근거

---

### 7. Baseline 없음

"MH 30개로 92% 달성"이 좋은 결과인지 알려면:
- **Random 30개 마커** 평균 정확도 (lower bound)
- **Ae/FST 단독 상위 30개** (ML 안 쓰고, 순수 통계)
- **STR 표준 패널**(현 forensic 표준) 추정 성능 (논문 인용)

세 baseline 대비 어디까지 좋아졌는지가 contribution.

---

### 8. 친족(relatedness) 처리

1000G에는 일부 친족 관계가 있다. `StratifiedKFold`는 친족이 train/test에 나뉘면 정확도를 부풀린다.

**수정**: `GroupKFold` 또는 KING/PLINK로 IBD 0.0884 이상 제거 후 분석.

---

### 9. (보너스) Kidd & Speed 2015의 Ae 임계값 오용

Kidd & Speed 2015 ([PMC4351693](https://pmc.ncbi.nlm.nih.gov/articles/PMC4351693/))의 **"Ae > 3"은 mixture detection 기준**이며, 같은 논문이 명시적으로:

> "by selecting for high average Ae to maximize mixture detection, we are tending to reduce large regional differences in allele frequencies"

즉 **ancestry inference에는 부적절한 기준**. Ancestry용 informativeness measure는 Rosenberg의 **In (Informativeness for Assignment)** 또는 Shannon-divergence-based metric이 표준. 제안서는 두 기준을 혼동했다.

또한 Podini 2026 ([PubMed 41621241](https://pubmed.ncbi.nlm.nih.gov/41621241/))의 Microhaplotype Working Group 공식 consensus는:
- length ≤250bp
- LINE/SINE/LTR 제외
- forensic STR과 close physical linkage 있는 loci 제외
- low-complexity sequence 제외

이를 적용해 412 → ~수백 개로 정제하는 것 자체가 추가 contribution이 된다.

---

## 🟢 살릴 강점

1. **Two-track filter(유전학 + ML)는 정석**이고 reviewer가 좋아하는 구조. 다만 1차 필터를 outer-fold 안에 넣어도 결과가 안정적인지 보여야 함.
2. **Flask 데모 + GitHub 공개**는 "contribution을 tool로 응축한다"는 형태로 KSFS(한국법과학학회) 발표나 FSI:Genetics 같은 저널에서도 가산점.
3. 데이터·코드 reproducibility(bcftools/pysam/microhapdb)가 명시적.
4. Limitations 섹션을 미리 설계한 점 (부족하지만 방향은 옳음).

---

## 우선순위 수정 작업

| 우선순위 | 작업 | 영향 |
|---|---|---|
| P0 | nested CV로 leakage 제거 (#1) | 보고 정확도 신뢰성 |
| P0 | diplotype encoding으로 변경 (#2) | 분석의 정의 자체가 옳아짐 |
| P0 | Wei 2025 positioning 추가 (#4) | reviewer 1번 질문 차단 |
| P1 | confusion matrix + CHB-CHS hard pair (#3) | 결과 해석 정직성 |
| P1 | Random/Ae-only baseline 추가 (#7) | contribution 정량화 |
| P2 | SHAP + bootstrap stability (#6) | panel 신뢰성 정량 근거 |
| P2 | Kidd&Speed/Podini 기준 정확히 적용 (#9) | 도메인 정확성 |
| P3 | KOR 보강 (#5) | 메시지 완성도 |
| P3 | GroupKFold (#8) | minor accuracy correction |

---

## AI 박사 관점의 추가 권장

1. **Calibrated probability**가 forensic context에서 핵심 — `CalibratedClassifierCV` 또는 conformal prediction 도입 (별도 novelty 옵션 문서 참고)
2. **Reject option** — open-set / OOD detection으로 "이 샘플은 분류 거부" 가능성 (novelty 옵션 문서 참고)
3. **per-pair pairwise classifier**가 5-way보다 forensic LR과 직접 연결됨 — 고려할 가치 있음
