# 제안서 보강 의견 — Option 1: Conformal Prediction + Open-set Reject

**작성**: 조민한 (mh.cho@alphabridge.co.kr)
**작성일**: 2026-05-26
**대상**: 팀 (임수연 외)
**관련 문서**: `docs/01_proposal_review.md`, `docs/02_literature_landscape.md`, `docs/03_novelty_options.md`

---

## 들어가며

먼저, 보내주신 제안서 잘 읽었습니다. 연구 질문이 매우 명확하고 (MH 몇 개로 EAS 5집단 구별 가능한가), 1000G→bcftools→pysam→XGBoost→Flask 이용 FE까지 파이프라인 전체가 reproducible하게 설계되어 있어 리뷰어가 좋아할 구조이지 않을까 싶습니다. 여기에 AI methodology 차별점을 한 가지 정도 더할 수 있어 제안드리고자 합니다.

---

## 왜 보강이 필요한가 — 두 가지 발견

문헌 정독을 하면서 알게 된 사실 두 가지가 있습니다.

**1. 1000G phasing error 문제 (Wei et al., 2025)** — Phasing은 한 개인이 가진 두 haplotype(부모 각각으로부터 물려받은 DNA 한 세트)이 어떤 SNP 조합을 이루는지 통계적으로 재구성하는 작업입니다. 1000G의 phased VCF는 shapeit2 알고리즘으로 추정된 결과인데, 이 과정에서 실제와 다른 조합으로 SNP가 배열되는 *phasing error*가 발생합니다. MH는 정의상 **한 haplotype 위의 SNP 조합**이므로, phasing error는 곧 잘못된 MH allele을 만들어내고 결과적으로 marker의 다양성 지표(Ae) 추정값을 부풀립니다.

Wei et al. (2025)는 1000G phase 3의 602 trio(부모+자녀)에서 mendelian 일치성 검증(자녀의 haplotype이 부모의 haplotype 조합으로 만들어질 수 있는지)으로 marker별 phasing error rate를 직접 측정했고, **전체 평균 0.07%로 낮지만 marker의 Ae·In이 높을수록 error 확률이 비례해 증가**함을 보였습니다. 즉 우리가 Ae≥3로 골라내려는 마커가 정확히 가장 위험한 마커가 됩니다. 단순 인용으로는 부족하고 해당 논문을 극복해야 한다고 판단했습니다.

**2. 직접 경쟁 논문 존재 (Chen et al., 2025)** — Human Genomics에 발표된 해당 논문이 거의 같은 셋업(XGBoost + East Asian)으로 95.6% 정확도를 보고했습니다. 우리가 원안 그대로 가면 이미 한 일의 더 작은 버전으로 보일 수 있어 차별점이 필요하다고 판단했습니다.

---

## 제안: 옵션 1 — Conformal Prediction + Open-set Reject

**한 줄로**: 기존 XGBoost 출력을 *"이 샘플은 JPT"* 같은 단일 라벨이 아니라, **"이 샘플이 속할 가능 집단 set + 신뢰 수준"** 으로 변환하고, 동시에 *"학습 5집단 어디에도 안 속함 (unknown)"* 을 정직하게 출력할 수 있게 만드는 것입니다.

### 출력 변화 예시

| 기존 | 옵션 1 적용 후 |
|---|---|
| `JPT (87.3%)` | `{JPT, CHB} — 90% 신뢰` |
| (강제 분류) | `Unknown — 학습 reference 5집단 외 분포 (e.g., 몽골·위구르·티베트 등 미학습 EAS 하위집단)` |
| `CHS (45%)` | `{CHB, CHS} — CHB·CHS 구별 불가, 두 집단 합쳐 95% 신뢰` |

### 왜 forensic에 중요한가

1. **법정 admissibility**: "이 사람은 JPT입니다"는 court에서 공격받지만 "JPT 또는 CHB 두 가능성, 90% 신뢰"는 통계학적 근거가 명확해 admissible합니다.
2. **Likelihood Ratio(LR) 자연 도출**: forensic 표준 출력인 LR을 신뢰구간과 함께 계산할 수 있습니다.
3. **학습 reference 외 집단 거부**: 1000G EAS 5집단 외 하위집단(몽골·위구르·티베트 등 미학습 EAS, 또는 admixed 개인)을 강제 분류하지 않습니다 — racial profiling 위험 차단.

### 왜 이것이 novel한가

Zhang (2025)의 *"Artificial intelligence in forensic genetics"* 리뷰는 103개 ref를 다루지만 **Conformal Prediction과 Open-set Recognition을 한 번도 언급하지 않습니다**. 이게 본 연구의 novelty claim의 정량적 증거입니다. Forensic ancestry에 적용한 첫 사례가 됩니다.

---

## 무엇이 유지되고 무엇이 추가되는가

| 항목 | 상태 |
|---|---|
| 1000G EAS 데이터 + bcftools 파이프라인 | ✅ 그대로 |
| MicroHapDB 412 마커 | ✅ 그대로 |
| Two-track filter (Ae/FST + ML) | ✅ 그대로 |
| XGBoost 5집단 분류 | ✅ 그대로 |
| Flask 웹 도구 | ✅ 그대로 (출력 형식만 set+신뢰구간으로 업데이트) |
| **+ Mondrian Conformal Prediction wrapping** (Vovk, 2013; Angelopoulos & Bates, 2023) | 🆕 추가 |
| **+ Energy-based Open-set Recognition** (Liu et al., 2020; Scheirer et al., 2013) | 🆕 추가 |
| **+ Leave-one-population-out (LOPO) 평가** | 🆕 추가 |

기존 제안서의 90%는 그대로 살리고 10%만 추가하는 구조입니다.

---

## 일정·자원

- **기간**: 2주 (2026-05-26 ~ 06-09)
- **GPU**: Alphabridge DGX Spark GB-10 × 2 (현재 사용 가능)
- **Day 1-3**: 제안서의 P0 이슈 (leakage, diplotype, GroupKFold) 수정 + baseline 재현
- **Day 4-7**: 옵션 1 본체 구현 + LOPO 평가
  - Day 7 = **체크포인트**: 이 시점에 학기 final로 제출 가능한 상태
- **Day 8-14**: Day 7 체크포인트 통과 시 → 아래 **Further Research(옵션 4)** 진입. 실패·시간 부족 시 옵션 1 단독 제출.

---

## Further Research — 옵션 4: SSL + Multi-task Forensic Foundation Model

옵션 1을 안전망으로 확보한 뒤(Day 7 체크포인트 통과 시), Week 2에 추가로 시도해볼 야심 방향입니다.

### 한 줄로

1000G WGS에서 **self-supervised contrastive learning** (Chen et al., 2020)으로 범용 forensic representation을 pretrain하고, 그 위에 **ancestry · kinship · biological sex**를 동시에 예측하는 **multi-task head** (Caruana, 1997)를 얹어, 옵션 1의 conformal/OSR layer로 wrapping. 한 모델로 forensic의 여러 task를 동시에 푸는 *"Forensic Foundation Model"* 첫 시도.

### 왜 의미 있는가

1. **Forensic-FM은 cross-modality emerging paradigm**: Face security 영역에서는 FS-VFM (Wang et al., 2025)이 이미 등장 — SSL pretraining + downstream forensic tasks(deepfake / diffusion / spoofing detection) 구조. 우리는 **동일 paradigm을 forensic genetics modality에 처음 적용**. 현재 forensic genetics는 ancestry·kinship·sex를 각각 별도 panel·별도 모델로 풀고 있어 (Chen, M. et al., 2023), SSL pretraining으로 통합 표현을 학습하면 *task 간 정보 공유*로 데이터 효율이 올라갑니다.

2. **두 개의 critical review가 독립적으로 SSL/FM 미언급 확인**: Zhang (2025) JTGG 103 refs와 Barash et al. (2024) FSI:G *"Machine learning applications in forensic DNA profiling: A critical review"* 둘 다 SSL · FM · multi-task representation을 0회 언급. Barash et al.은 명시적으로 *"the forensic community is largely unaware of ML capabilities and limitations"* 라는 갭을 지적 → *first-mover advantage* 영역.

3. **옵션 1과 자연스러운 통합**: SSL이 만든 representation 위에 conformal/OSR을 wrapping하면 *"Trustworthy Forensic-FM"* 한 단어로 묶이는 깨끗한 paper 메시지가 됩니다 — 학위논문 chapter seed로도 발전 가능.

### 선행연구 매핑 — 가장 가까운 multi-task forensic genetics work

표 안의 마지막 열 *shared SSL representation* 컬럼이 비어있는 칸이 본 옵션 4의 갭입니다.

| 선행연구 | task 구성 | 메서드 | shared SSL repr |
|---|---|---|---|
| Chen, M. et al. (2023) Front Genet — Yunnan DIP panel | 개인식별 + kinship + ancestry | single panel, **task별 분리 모델** | ❌ |
| Chen, J. et al. (2025) Hum Genomics — AISNP framework | ancestry only | XGBoost + AISNPs | ❌ |
| FSI:G (2025) — DNA methylation framework (PMID 41086638) | age, body fluid (각각 별도) | nanopore + 통계 모델 | ❌ (multi-task 아님) |
| El Rashidy et al. (2025) bioRxiv — Zygosity-aware DNA LM | ancestry + gene expression | DNA FM (HyenaDNA 기반) | △ FM 사용하나 forensic framing 부재 |
| Wang et al. (2025) FS-VFM — face security FM | deepfake + diffusion + spoofing | SSL ViT pretraining + adapters | ✅ — 단 modality는 face image |
| **본 옵션 4 (제안)** | **ancestry + kinship + sex (+ 후속 확장)** | **SSL contrastive pretraining + multi-task heads** | **✅ — forensic genetics 첫 적용** |

### 무엇이 추가되는가

| 항목 | 상태 |
|---|---|
| 옵션 1 전체 구조 | ✅ 그대로 (이미 Day 7까지 완성) |
| **+ SimCLR/Masked-SNP contrastive pretraining** (Chen et al., 2020; 1000G WGS) | 🆕 추가 |
| **+ Multi-task heads** (ancestry + sex + kinship score; Caruana, 1997) | 🆕 추가 |
| **+ Linear-probe vs full fine-tune 비교** | 🆕 추가 |
| **+ SSL representation 위의 conformal/OSR 재평가** | 🆕 추가 |

### 위험과 fallback

- SSL이 XGBoost baseline 정확도를 못 이길 수 있음 → 그 경우 옵션 1만 제출 (Day 7 결과로 충분)
- Genomic augmentation 디자인이 미해결 영역 — random masking, position shuffle 등 시도
- GPU 시간 큼 → Alphabridge DGX Spark GB-10 × 2 (CONFIG_SLICE 분산) 활용

### 직접 경쟁 논문 인지

El Rashidy et al. (2025)이 *"Zygosity-Aware DNA Language Modeling Improves Ancestry and Gene Expression Prediction"*에서 DNA language model (HyenaDNA; Nguyen et al., 2023)을 ancestry 예측에 직접 적용했으나, **forensic-specific framing은 부재**합니다. 우리는 *forensic context*(LR, admissibility) + SSL multi-task 두 vector로 차별화 가능합니다.

---

## 팀에 묻고 싶은 것

1. **방향 동의 여부** — 이 보강이 우리 메시지(*"동아시아 forensic ancestry를 위한 최소 MH 패널"*)와 잘 맞는다고 보시나요? 아니면 원안 그대로가 낫다고 보시나요?

2. **역할 분담** — 제 쪽이 conformal/OSR 구현·평가를 맡고, 기존 제안서의 데이터·도메인 부분은 임수연님 + 팀 그대로 진행하는 게 효율적일 듯합니다. 의견 부탁드립니다.

3. **발표·논문 방향** — 학기 final 단독 제출 vs. 학기 final + 후속 학회/저널 submit(FSI:Genetics, J Forensic Sci 등) 둘 다 가능합니다. 어느 쪽으로 잡을지요?

---

## 참고 문서 (프로젝트 내부)

자세한 분석은 다음 문서들에 정리해두었습니다 (`docs/` 폴더):
- `01_proposal_review.md` — 제안서 비판적 리뷰 (수정 우선순위 P0~P3)
- `02_literature_landscape.md` — 핵심 5편 + 직접 경쟁 논문 매핑
- `03_novelty_options.md` — 10개 novelty 옵션 비교 (옵션 1이 현재 선택)
- `superpowers/plans/` — 2주 스프린트 plan (Day 1-3 detailed)

---

## 방법론 참고 (Methods Glossary)

본 제안서에서 도입하는 AI/ML 방법론을 도메인 전문가 분들이 참고하시도록 한 줄 설명 + 핵심 문헌을 정리했습니다. 전체 bibliographic info는 아래 References 섹션에 있습니다.

| 용어 | 한 줄 설명 | 핵심 문헌 |
|---|---|---|
| **XGBoost** | 그래디언트 부스팅 결정 트리. 표 데이터 분류·회귀의 표준 ML 알고리즘. (제안서에서 이미 사용) | Chen & Guestrin (2016) |
| **Conformal Prediction (CP)** | 분포 가정 없이 *"진짜 라벨이 prediction set 안에 들어갈 확률 ≥ 1−α"* 를 수학적으로 보장하는 신뢰성 보정 frameworks. 출력이 단일 라벨 대신 **라벨 집합 + 신뢰 수준** | Vovk, Gammerman, & Shafer (2005); Angelopoulos & Bates (2023) |
| **Mondrian CP** | CP의 클래스별 변형 — class-conditional coverage 보장. 클래스 불균형/어려운 페어(CHB↔CHS) 동등 보정에 필수. | Vovk (2013) |
| **Open-set Recognition (OSR)** | 학습 시 본 K개 클래스 외 샘플을 "unknown"으로 거부하는 패러다임 (기존 분류기는 강제로 K개 중 하나 출력). | Scheirer et al. (2013); Bendale & Boult (2016, OpenMax) |
| **Energy-based OOD** | 모델 logit의 log-sum-exp를 "energy"로 정의 — in-distribution은 low energy, OOD는 high energy. OSR의 modern variant. | Liu et al. (2020) |
| **Self-Supervised Contrastive Learning (SimCLR)** | 라벨 없이 데이터 augmentation으로 같은 샘플의 두 view를 가깝게, 다른 샘플은 멀게 학습 → 범용 representation. | Chen, Kornblith, Norouzi, & Hinton (2020) |
| **Multi-Task Learning** | 한 모델로 여러 task를 동시에 학습 — task 간 정보 공유로 데이터 효율 ↑. | Caruana (1997) |
| **DNA Foundation Model (HyenaDNA)** | DNA 서열 자체를 입력받아 학습한 대형 모델. SNP/MH panel 설계 없이 raw context에서 representation 추출. | Nguyen et al. (2023) |
| **Leave-one-population-out (LOPO)** | Cross-validation의 변형 — 한 집단을 통째로 빼고 학습, 그 집단이 OOD로 잡히는지 평가. OSR의 표준 검증 방법. | Stone (1974) 의 leave-one-out CV에서 파생 |

---

## 참고문헌 (References)

### 본 연구와 직접 관련된 forensic genetics 논문

Barash, A., McNevin, D., Fedorenko, A., & Giverts, P. (2024). Machine learning applications in forensic DNA profiling: A critical review. *Forensic Science International: Genetics*, *69*, 102994. https://doi.org/10.1016/j.fsigen.2023.102994

Chen, J., Huang, Y., Fan, H., Wang, M., He, G., & Yan, J. (2025). Integrated genetic and geographic ancestry prediction via large-scale genomic data and machine learning. *Human Genomics*, *19*, 126. https://doi.org/10.1186/s40246-025-00837-3

Chen, M., Lan, Q., Nie, S., Mei, S., Liu, Y., Zhu, J., & Zhu, B. (2023). Forensic efficiencies of individual identification, kinship testing and ancestral inference in three Yunnan groups based on a self-developed multiple DIP panel. *Frontiers in Genetics*, *13*, 1057231. https://doi.org/10.3389/fgene.2022.1057231

El Rashidy, H., Saadat, A., & Fellay, J. (2025). *Zygosity-Aware DNA Language Modeling Improves Ancestry and Gene Expression Prediction* [Preprint]. bioRxiv. https://doi.org/10.1101/2025.11.19.689326

Wang, G., Lin, F., Wu, T., Yan, Z., & Ren, K. (2025). *Scalable Face Security Vision Foundation Model for Deepfake, Diffusion, and Spoofing Detection* [Preprint]. arXiv. https://arxiv.org/abs/2510.10663

Wei, Y., Li, X., & Zhu, Q. (2025). Are microhaplotypes derived from the 1000 Genomes Project reliable for forensic purposes? *Forensic Science International: Genetics*, *76*, 103273. https://doi.org/10.1016/j.fsigen.2025.103273

Zhang, H. (2025). Artificial intelligence in forensic genetics: applications and ethical challenges. *Journal of Translational Genetics and Genomics*, *9*. https://www.oaepublish.com/articles/jtgg.2025.76

### 방법론 참고문헌 (AI/ML methods)

Angelopoulos, A. N., & Bates, S. (2023). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *Foundations and Trends in Machine Learning*, *16*(4), 494–591. https://doi.org/10.1561/2200000101

Bendale, A., & Boult, T. E. (2016). Towards open set deep networks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 1563–1572). https://doi.org/10.1109/CVPR.2016.173

Caruana, R. (1997). Multitask learning. *Machine Learning*, *28*(1), 41–75. https://doi.org/10.1023/A:1007379606734

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). https://doi.org/10.1145/2939672.2939785

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. In *Proceedings of the 37th International Conference on Machine Learning (ICML)* (Vol. 119, pp. 1597–1607). https://proceedings.mlr.press/v119/chen20j.html

Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based out-of-distribution detection. In *Advances in Neural Information Processing Systems (NeurIPS)* (Vol. 33, pp. 21464–21475). https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html

Nguyen, E., Poli, M., Faizi, M., Thomas, A., Birch-Sykes, C., Wornow, M., Patel, A., Rabideau, C., Massaroli, S., Bengio, Y., Ermon, S., Baccus, S. A., & Ré, C. (2023). HyenaDNA: Long-range genomic sequence modeling at single nucleotide resolution. In *Advances in Neural Information Processing Systems (NeurIPS)* (Vol. 36). https://arxiv.org/abs/2306.15794

Scheirer, W. J., de Rezende Rocha, A., Sapkota, A., & Boult, T. E. (2013). Toward open set recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *35*(7), 1757–1772. https://doi.org/10.1109/TPAMI.2012.256

Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society: Series B (Methodological)*, *36*(2), 111–133. https://doi.org/10.1111/j.2517-6161.1974.tb00994.x

Vovk, V. (2013). Conditional validity of inductive conformal predictors. *Machine Learning*, *92*(2–3), 349–376. https://doi.org/10.1007/s10994-013-5355-6

Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic learning in a random world*. Springer. https://doi.org/10.1007/b106715

---

의견·질문 편하게 주시면 좋겠습니다. 다음 미팅에서 결정한 방향으로 바로 Day 1 작업 들어가겠습니다.

감사합니다.
조민한 드림
