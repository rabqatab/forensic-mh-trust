# 문헌 정독 + 경쟁 논문 매핑 (v1)

**검토일**: 2026-05-26
**갱신**: 2026-05-30 — Notion §5(Previous Studies) 동기화 → **7부** 추가(댓글 제안 + forward-citation); **8부** 추가(DL-popgen 아키텍처·tabular-DL SOTA·확장 데이터 자원 = §24 실험 근거). 2026-05-31 — **9부**(embedding을 *잘* 만든 접근, Paper 2). 2026-06-01 — **10부** 추가(Notion §5.10·§5.11 댓글 2차: 비-ML X-InDel·친족 DeepKin/LR·BGA Alladio 2022·DL 집단구조 HaploNet/DietNet-bioRxiv).
**검색 범위**: PMC, PubMed, FSI:Genetics, Frontiers, bioRxiv, ResearchGate, Semantic Scholar(forward-citation)
**제한**: ScienceDirect / Springer 일부 paywall — abstract 위주 확보

---

## 1부. 핵심 5편 정독

### ⭐ Wei, Li, Zhu (2025) — 본 제안서에 가장 중요

**제목**: *Are microhaplotypes derived from the 1000 Genomes Project reliable for forensic purposes?*
**저널**: Forensic Science International: Genetics (2025-03-15)
**DOI**: 10.1016/j.fsigen.2025.103273
**URL**: https://www.fsigenetics.com/article/S1872-4973(25)00053-5/abstract
**저자 소속**: Sichuan University

**주요 결과**:
- **602 trios from 1000 Genomes Project** 사용 — 부모-자녀 mismatch를 통해 phasing error 직접 관찰
- **전체 0.07% phasing error rate** across all autosomes (expanded 1000G)
- **MH의 phasing error 확률은 Ae(effective alleles) 및 In(informativeness) 값이 높을수록 증가**
- 대륙별 error rate 다름

**본 제안서에 미치는 영향**: **치명적**.
- 제안서가 골라내려는 high-Ae 마커가 가장 위험한 마커.
- Ae≥3 필터 그대로 적용하면 phasing error가 많은 마커들로 학습 — 진짜 인구 신호 vs phasing artifact 구분 불가.
- Reviewer가 가장 먼저 던질 질문이며, 이를 정면으로 다루지 않으면 contribution이 약함.

**대응 옵션**:
1. Wei 2025의 trio-based filter를 본 분석 전에 적용 (마커 list에서 high-error MH 제외)
2. `Reliable Ae = Ae × (1 − P_phaseerror)` 새 metric 도입 — 이 자체로 contribution
3. Limitation에 명시하고 sensitivity analysis만 제공

---

### Kidd & Speed (2015) — Ae 기준의 원전

**제목**: *Criteria for selecting microhaplotypes: mixture detection and deconvolution*
**저널**: Investigative Genetics, 6:1 (2015-01-28)
**URL (open access)**: https://pmc.ncbi.nlm.nih.gov/articles/PMC4351693/
**저자 소속**: Yale University, Dept of Genetics

**주요 기준** (논문에서 정확히 인용):

| 기준 | 값 | 출처 |
|---|---|---|
| Ae cutoff | **Ae > 3.0** (mixture 목적) | "Microhaplotypes with Ae values of >3 will be exceedingly useful in ordinary forensic practice" |
| Length | **~200 bp 이내** | "two or more SNPs within the span of a single sequence run (arbitrarily set currently at 200 bp)" |
| SNP 수 | 2-SNP: rarely >3.0, 3-SNP: sometimes >3.0, 4-SNP: often >4.0 | |
| Mixture panel | **5 loci with avg Ae≥3 → >95% cumulative detection** | |

**중요한 caveat (제안서가 놓친 부분)**:
> "by selecting for high average Ae to maximize mixture detection, we are tending to reduce large regional differences in allele frequencies"

→ **Ae>3은 mixture detection용 기준이지 ancestry inference용 기준이 아니다.** 이 caveat을 제안서는 인용하지 않았다.

Ancestry용 표준 informativeness measure는:
- Rosenberg의 **In (Informativeness for Assignment)**
- **Shannon divergence** 기반 metric
- **PSMC-style FST 기반 ranking**

---

### Podini et al. (2026) — Microhaplotype Working Group consensus

**제목**: *Defining key criteria for microhaplotype locus selection in forensic genetics: Progress and recommendations by the Microhaplotype Working Group*
**저널**: Forensic Science International: Genetics (online 2026-01-25, print 2026-04)
**URL**: https://pubmed.ncbi.nlm.nih.gov/41621241/
**저자**: Podini, Standage, Phillips, de la Puente, Børsting, Pereira, Davenport, Ballard, Cavanaugh, Young, Lagacé, Kiesler, Oldoni, Rodrigues, Turchi, Bever, Liang, Kidd

**Working Group 합의 기준** (한 줄 요약):

| 기준 | 값 |
|---|---|
| Primary selection | **Ae** |
| Length | <100bp (degraded sample용), <250bp (general) |
| **Exclusion** | LINE / SINE / LTR 내부 |
| **Exclusion** | forensic STR과 close physical linkage |
| **Exclusion** | Indel 포함, low-complexity sequence near allele-defining SNPs |
| SNP content | average informative SNPs 많을수록 informative |
| Theoretically suitable loci | **1,148** identified |

**본 제안서에 미치는 영향**:
- 412개 마커 중 위 exclusion criteria 통과하는 것만 골라내는 작업 자체가 추가 contribution.
- "MicroHapDB의 412개를 그대로 쓰지 말고 Working Group consensus를 적용" 한 줄 더해도 도메인 정확도 ⬆.
- 단, 본 abstract에서는 ancestry vs mixture vs identification 별 thresholds는 분리하지 않음. 모두 동일 기준 적용.

---

### Standage & Mitchell (2020) — MicroHapDB

**제목**: *MicroHapDB: A Portable and Extensible Database of All Published Microhaplotype Marker and Frequency Data*
**저널**: Frontiers in Genetics, 11:781 (2020)
**URL (open access)**: https://pmc.ncbi.nlm.nih.gov/articles/PMC7427474/

**주요 내용**:
- **417 microhaplotype markers** total
- **412 markers**에 대해 1000G Phase 3 (2,504명) 기반 frequency 데이터
- **26 global populations** 커버
- Marker별 **Ae** (per-population, then averaged), **FST** (Weir-Cockerham, averaged across alleles)
- EAS 26개 중 포함

**본 제안서에서의 위치**: 제안서의 기본 marker source. 그대로 사용 가능. 단 Wei 2025의 phasing error 우려는 이 DB의 frequency 값에도 일부 영향.

---

### Oldoni, Kidd, Podini (2019) — MH 리뷰

**제목**: *Microhaplotypes in forensic genetics*
**저널**: Forensic Science International: Genetics, 38:54-69 (2019-01)
**URL**: https://www.sciencedirect.com/science/article/abs/pii/S1872497318303910

**주요 내용** (abstract 기반):
- MH 정의: <300 nucleotides, 2개 이상 closely linked SNP
- 응용: human identification, mixture deconvolution, ancestry inference
- 향후 응용: missing person, relationship testing, medical/non-human DNA
- 본 제안서가 다루는 ancestry inference는 이 review의 한 chapter에 해당

**본 제안서에서의 위치**: standard background citation. ML 기반은 아니므로 제안서가 ML로 차별화 가능.

---

## 2부. 직접 경쟁 논문 매핑

### 🔥 매우 경쟁적 (almost-same setup)

#### Yang et al. (2025) — Frontiers in Ecology and Evolution
**제목**: *Fine-scale biogeographical ancestry inference in Southeast and East Asians via high-efficiency markers and machine learning approaches*
**URL**: https://www.frontiersin.org/journals/ecology-and-evolution/articles/10.3389/fevo.2025.1572596/full

| 항목 | 값 |
|---|---|
| 대상 집단 | 8 sub-pops (Northern/Southern Hui + 6 linguistic families: Sino-Tibetan, Altaic, Hmong-Mien, Tai-Kadai, Austronesian, Austroasiatic). **JPT/KOR/CHB/CHS는 직접 포함되지 않음** |
| 데이터 | **2,191 newly genotyped Hui** + Human Origins reference (총 3,461). **1000G 미사용** |
| 마커 | AISNP 50/100/250/500/1000/2000 (nested) |
| 메서드 | RF + XGBoost + Locator (DNN for geo) |
| 결과 | genetic-only: 84% (RF), +geo: 96% (XGBoost) |
| 본 제안서와의 거리 | 가깝지만 다른 데이터·집단. **본 제안서는 1000G EAS + MH로 차별화 가능** |

#### Human Genomics (2025) — 가장 위협적
**제목**: *Integrated genetic and geographic ancestry prediction via large-scale genomic data and machine learning*
**URL**: https://humgenomics.biomedcentral.com/articles/10.1186/s40246-025-00837-3

| 항목 | 값 |
|---|---|
| 집단 | East/Southeast Asia 67 groups, n=1,703 (KOR 포함) |
| 마커 | AISNP 50~2000 nested |
| 메서드 | logistic, SVM, kNN, RF, CNN, **XGBoost** |
| 결과 | Best XGBoost 95.6% acc, AUC 0.999 (2000 SNPs). **Paired: JPT-CHH 99.73%, JPT-KOR 95%, KOR-CHH 90.11%** |
| 본 제안서와의 거리 | **거의 동일 셋업** — XGBoost로 East Asian fine-scale, Korean 포함. MH 대신 AISNP. 본 제안서가 그대로 가면 "이미 한 일의 더 작은 버전"으로 보일 위험 |

---

### 매우 관련 (관련성 high)

#### Systematic AISNPs Analysis (2024) — Forensic Science International
**제목**: *Systematic analyses of AISNPs screening and classification algorithms based on genome-wide data for forensic biogeographic ancestry inference*
**URL**: https://www.sciencedirect.com/science/article/abs/pii/S0379073824000562

- 58 AISNPs, 6 algorithms (incl. XGBoost, RF)
- intercontinental + intra-East Asian
- ML feature selection 으로 패널 도출

#### Chinese 21-MH for 10 subpops (2022) — bioRxiv
**제목**: *Screening and selection of 21 novel microhaplotype markers for ancestry inference in ten Chinese subpopulations*
**URL**: https://www.biorxiv.org/content/10.1101/2021.11.08.467710v1.full

- 10 Chinese subpopulations 대상
- 21 MH panel
- 본 제안서와 marker type 동일, 다른 셋업

#### Italian 76-MH (2025) — PubMed 41270348
**제목**: *Microhaplotypes in forensic genetics: From exploration to application in degraded DNA specimens*

- 76 microhaps (299 SNPs)
- PCA + STRUCTURE on 1000G 26 pops + Italian
- Degraded DNA (0.05 ng) 작동 확인
- Cumulative matching probability 11.763E-66
- 본 제안서와 marker 다르고, ML 없음 — 차별점 있음

---

### 관련 (참고용)

| 논문 | 메모 |
|---|---|
| Oldoni et al. (2017) *Ancestry inference of 96 population samples using microhaplotypes*, IJLM. https://link.springer.com/article/10.1007/s00414-017-1748-6 | 초기 MH ancestry work. Paywall이라 details 미확보. |
| Standage SNP-SNP MH overview (2022) *An overview of SNP-SNP microhaplotypes in the 26 populations of the 1000 Genomes Project*, IJLM | descriptive only. https://pubmed.ncbi.nlm.nih.gov/35397682/ |
| State of the Art for Microhaplotypes (MDPI 2022) | Open access review. https://www.mdpi.com/2073-4425/13/8/1322 |
| Zhou (2025) *Selection of microhaplotype loci and development of panel for forensic application*, J Forensic Sci | Chinese group, paywall. https://onlinelibrary.wiley.com/doi/abs/10.1111/1556-4029.70139 |
| Brazilian 2025 *Assessment of a microhaplotype panel for human identification and ancestry inference in Brazil*, IJLM | 다른 region |
| Large-scale ancestry MH (2024) *Large-scale selection of highly informative microhaplotypes for ancestry inference and population specific informativeness*, FSI:G | 직접 관련. Paywall로 detail 미확보. https://www.fsigenetics.com/article/S1872-4973(24)00149-2/fulltext |
| Advancing biogeographical ancestry through ML (2025) *Advancing biogeographical ancestry predictions through machine learning*, FSI:G | Paywall. https://www.fsigenetics.com/article/S1872-4973(25)00070-5/fulltext |

---

## 3부. Novelty space 검색 결과

### ⭐ Anchor reference (모든 novelty 옵션 검증의 정량 증거)

**Zhang H (2025)** *"Artificial intelligence in forensic genetics: applications and ethical challenges"* Journal of Translational Genetics and Genomics. https://www.oaepublish.com/articles/jtgg.2025.76

- **103 references**, 2020-2025 weighted, ancestry inference 핵심 review
- **인용된 ancestry ML 메서드**: XGBoost, PLS-DA, soft clustering, decision tree, Bayesian network
- **0회 언급**: Conformal Prediction, Open-set Recognition, DNA Foundation Models (HyenaDNA/Caduceus/Evo/NT), Self-Supervised Learning, Differential Privacy
- **언급된 갭**: Eurocentric bias, model opacity, admissibility, privacy by default
- → `03_novelty_options.md` 옵션 1, 2, 3, 4의 novelty claim의 직접 증거

### 미적용 / 갭이 있는 방향들

| 방향 | 검색 결과 |
|---|---|
| Conformal prediction in **forensic** ancestry | **0편** — genomic medicine 일반에는 있음 (Frontiers Bioinformatics 2025) |
| Open-set / OOD detection for forensic ancestry | **0편** |
| Differential privacy + forensic genomics | very few |
| DNA foundation model (HyenaDNA/Caduceus/Evo) for forensic | **0편** (general genomic은 활발) |
| Federated learning for forensic ancestry | very few |
| Active learning / RL for adaptive sequencing | very few in forensic context |
| In silico mixture + ancestry benchmark EAS-specific | 일부 있으나 systematic EAS-specific 없음 |

→ **이 방향들은 모두 후속 novelty 후보**. 별도 문서 (`03_novelty_options.md`) 참고.

---

## 4부. 한국 데이터 자원

### KoVariome / KPGP
- **PMC5885007**: 50 Korean WGS, 12.7M SNV, 1.7M indel, 4K SV, 3.6K CNV. 19% SNV는 novel.
- **운영**: KOBIC (Korean Bioinformation Center), 2006~
- **접근**: 공개 데이터지만 raw FASTQ/BAM 다운 후 variant call·phasing 직접 재수행 필요 (1~2주 추가)
- **이미 1400 autosomal SNP panel 제안 사례** 존재 (KoVariome 기반)

### GenomeAsia 100K
- 다양한 아시아 집단 포함, 일부 공개
- 1000G와 batch effect 존재

### NFS (국립과학수사연구원)
- 한국 forensic 표준 STR/SNP 마커 운용
- 공개 데이터는 제한적, 협력 필요

---

## 5부. 결론

1. **본 제안서 그대로의 contribution은 좁다** — Yang 2025 (Frontiers), Human Genomics 2025와 거의 같은 공간을 작은 데이터·작은 marker set으로 다시 다루는 형태.
2. **Wei 2025를 정면 대응하는 것은 contribution** — 단순 인용이 아니라 phasing-error-aware MH selection 자체가 차별점.
3. **Podini 2026 working group criteria를 적용해 412 → ~수백으로 정제**하는 것도 도메인 contribution.
4. **AI 메서드 novelty 후보** (forensic 미적용 영역): conformal prediction, open-set/OOD, DNA foundation models, DP generative, federated, active learning.
5. **KOR 보강은 가능** — KoVariome 50 WGS 공개, 단 처리 비용 1~2주.

---

## 6부. 옵션별 새 references (v2 추가)

옵션별 prior art는 `03_novelty_options.md` 각 옵션 섹션 참고. 이 문서에서 새로 확인된 핵심 references:

### Anchor (모든 옵션 검증용)
- **Zhang H (2025) JTGG** — `03` 옵션 1, 2, 3, 4 anchor (위 3부 참고)

### 옵션 1 (Conformal + Open-set)
- Papangelou et al. (2025) Frontiers Bioinf (doi:10.3389/fbinf.2025.1507448) — CP in genomic medicine (no forensic) [이전 "Olsson 2025" 오귀속 교정]
- Yang, Zhou, Li & Liu (2024) IJCV (doi:10.1007/s11263-024-02117-4) — generalized OOD 서베이 [이전 "Liu 2024" = 말미저자 표기 교정]

### 옵션 2 (Bayesian Deep)
- Toneyan & Koo (2024) DEGU — uncertainty in genomic DL
- Depeweg et al. (2018) ICML — aleatoric/epistemic 분해 이론

### 옵션 3 (DNA Foundation Model)
- Nguyen et al. (2023) NeurIPS — HyenaDNA
- Schiff et al. (2024) ICML — Caduceus (Mamba 기반)
- Tang et al. (2025) Nat Commun — 5-FM benchmark
- ⚠️ **bioRxiv (2025-11)** — *Zygosity-Aware DNA Language Modeling for Ancestry* (직접 경쟁)

### 옵션 4 (SSL + multi-task)
- GENEREL (arxiv 2410.10144) — contrastive SNP+medical
- Taleb et al. (2022) CVPR — ContIG multimodal contrastive
- Yunnan DIP panel (2022) Frontiers — multi-task forensic precedent (SSL 없음)

### 옵션 5 (DP generative)
- ⚠️ DP-SNP-TIHMM (arxiv 2510.05777) — 직접 경쟁
- ⚠️ SNPgen (arxiv 2603.10873) — 직접 경쟁 (latent diffusion)

### 옵션 6 (Federated)
- ⚠️ Frontiers Big Data (2024) — 1000G + UKBB federated ancestry (직접 경쟁)
- HE-based federated genomic (eprint 2025/1515)

### 옵션 7 (Mixture deconvolution)
- FSI:G 2024 — MH NOC ML (단 NOC만, ancestry 미포함)
- MixDeR (FSI:G 2025), MH vs STR mixture (Genes 2025)
- Probabilistic genotyping: EuroForMix, STRmix, MPSproto, TrueAllele

### 옵션 8 (Degraded DNA)
- Tvedebrink 2010 FSI:G — ADO 추정 고전
- Italian 76-MH 2025 (PMID 41270348) — degraded 작동 직접 precedent
- Explainable AI forensic DNA EPG (2025) FSI:G

### 옵션 9 (Active learning / RL)
- arxiv 2501.04718 (2025) — RL for gene panel selection (scRNA)
- FSI:G 2024 — Nanopore adaptive sampling forensic (RL 없음)

### 옵션 10 (Substructure discovery)
- ⚠️ UMAP cryptic structure (2019) PLoS Genet — 직접 precedent
- HaploNet (VAE), Neural ADMIXTURE — DL substructure tools

### 그 외 새로 발견된 forensic ML 참고
- Multi-InDel ML ancestry (FSI:G 2022) — multi-marker type ML precedent
- Kinship NGS ML (Expert Syst Appl 2024) — kinship + ML
- Auto-branch multi-task Alzheimer (Frontiers Genet 2025) — multi-task genetic precedent (forensic 아님)

---

## 7부. Notion §5 (Previous Studies) 동기화 — 추가 인용 (2026-05-30)

> Notion 허브의 별도 페이지 **"5. 관련 연구 (Previous Studies)"** 에 정리된 항목 중, 본 문서 1~6부에 없던 선행연구를 역반영. 출처 = (a) Notion 페이지 댓글로 제안된 논문, (b) §5 앵커들의 **forward-citation**(Semantic Scholar `paper/DOI:.../citations` + scope 필터). 기존 항목(Wei 2025·Yang 2025·Chen 2025[=Human Genomics 2025]·Kidd&Speed·Podini·MicroHapDB·Oldoni·Systematic AISNPs·Chinese 21-MH·Italian 76-MH·Large-scale ancestry MH 2024·Zhang 2025)은 위 1~3부·6부 참조.

### 7-A. 댓글 제안 (paper 추천)

#### Zhaoyang Han Y-STR/Y-SNP haplogroup ML (FSI:G 2021)
**제목**: *Improving the regional Y-STR haplotype resolution utilizing haplogroup-determining Y-SNPs and the application of machine learning in Y-SNP haplogroup prediction in a forensic Y-STR database: A pilot study on male Chinese Yunnan Zhaoyang Han population*
**URL**: https://www.sciencedirect.com/science/article/pii/S1872497321001940 (PII S1872497321001940)
- 윈난 자오양 Han 비혈연 남성 **3,473명**, **24 Y-SNP** + Y-STR, 공개 **9종 ML** 비교. major haplogroup **99.71%** / detailed **98.54%**.
- **차별점**: 부계 haploid Y-마커·haplogroup 예측 vs 우리 상염색체 MH diplotype·근연 EAS 집단 분류. 신뢰정량화(CP/OSR) 없음. → ML×forensic-genetics 배경 인용.

#### GeoGenIE — SNP 기반 지리 출처 예측 DL 소프트웨어 (Bioinformatics Advances 2025)
**제목**: *GeoGenIE: a deep learning approach to predict geographic provenance of biodiversity samples from genomic SNPs* (Martin, Zbinden, Douglas, Douglas, Chafin)
**URL**: https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf250/8278073 — DOI 10.1093/bioadv/vbaf250 (오픈, CC BY 4.0)
- MLP(공간 SMOTE·역밀도 가중손실·이상치 탐지)로 **연속 좌표 회귀**. 흰꼬리사슴 436 SNP에서 26.4 km(LOCATOR 59.3 km 대비 2.25×). **더 적은 SNP로 더 높은 정확도**.
- **차별점**: 보전유전학·연속 좌표 회귀·딥 MLP vs 우리 forensic·이산 분류·단순 선형. 불확실성이 bootstrap+이상치 vs conformal/OSR. "적은 마커" 동기는 우리 RQ5와 같은 방향(단 km 회귀라 수치 비교 불가).

#### 생물지리 조상 추정 — SNP 패널 + 지도학습 (Expert Syst. Appl. 2025)
**제목**: *Predictive modeling of biogeographical ancestry using a novel SNP panel and supervised learning approaches*
**URL**: https://www.sciencedirect.com/science/article/pii/S0957417425032774 (PII S0957417425032774)
- 신규 **3,234 SNP** 패널, 지도학습 zoo(CatNB·penalized multinomial LR·linear SVM·RF·GB), nested CV + balanced accuracy, **2단계 vs 단일단계** 분류 전략 비교.
- **차별점**: 대륙·광역 조상 vs 우리 근연 EAS fine-scale. SNP vs MH. CP/OSR 없음. '2단계 vs 단일단계'는 우리 RQ5·계층 분류와 닿음.

#### 친족 분석 + ML (forensic NGS 패널) (Expert Syst. Appl. 2024) — 6부 "Kinship NGS ML" 상세화
**제목**: *Kinship analysis and machine learning algorithms in forensic contexts: A new NGS panel*
**URL**: https://www.sciencedirect.com/science/article/pii/S0957417424030288 (PII S0957417424030288)
- forensic NGS **~4,849 SNP** 패널(성별 무관 고차 혈연도 구별), **15만+ 쌍**, Forrel(R) + ML 병용.
- **차별점**: 과제가 **친족(kinship)** — 우리 집단(ancestry) 분류와 다른 축. CP/OSR 없음. forensic ML 적용 폭(혈연) 사례.

### 7-B. Forward-citation (앵커 인용 후속작, scope 매칭)

#### ★ East Asian-specific AISNP + ML로 5개 중국 집단 substructure (BMC Genomics 2025) — 우리 과제와 최근접
**제목**: *Bioinformatic insights into five Chinese population substructures inferred from the East Asian-specific AISNP panel*
**URL**: https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-025-11947-6 — DOI 10.1186/s12864-025-11947-6 (오픈)
**인용 앵커**: Oldoni 2019 등. 
- **EAS 특화 AISNP**(C5ClusterTag) 6단계 nested(50–2,000) → **5개 중국 집단 substructure**, **2,772명**, ML train/test 검증.
- **차별점**: "EAS fine-scale 5집단 + nested 패널 + ML"로 우리와 **거의 동일 과제 구조**(우리=1000G EAS-5). AISNP vs MH, 점추정 ML vs conformal/OSR. → RQ1·RQ5 최신 대비군.

#### MHappaMundi — AmpliSeq MH ancestry 패널 (FSI:G 2025)
**제목**: *MHappaMundi: A custom AmpliSeq microhaplotype panel for ancestry inference* (Rodrigues et al.)
**URL**: https://www.sciencedirect.com/science/article/pii/S1872497325001693 (PII S1872497325001693)
**인용 앵커**: Chen 2025(Human Genomics)·MicroHapDB.
- **MicroHapDB/1000G FST**로 선별한 **92 MH(466 SNP)** 대륙 ancestry 패널(Sub-Saharan/Europe/South Asia/East Asia/Native American) — Europe·South Asia 분리 개선.
- **차별점**: **우리와 동일한 MicroHapDB·1000G 소스의 MH ancestry 패널**이나 대륙 단위·신뢰층 없음. 직접 비교·차별화에 유용.

#### 구이저우 7 소수민족 MH 집단구조 (JTGG 2025)
**제목**: *Microhaplotype insights into the population structure of seven ethnic minorities in Guizhou, China*
**URL**: Journal of Translational Genetics and Genomics, 2025.
**인용 앵커**: Oldoni·MicroHapDB.
- MH로 중국 구이저우 **7 민족** 집단구조 규명(분류 ML보다 구조분석 중심). EAS MH 집단구조 인접 사례.

**방법 메모**: forward-citation = Semantic Scholar citations API + scope 필터(동아시아·MH/AISNP ancestry·ML·신뢰정량화). **scope 경계로 제외**: "Integrative forensic genomics … trace DNA 마약사건"(FSI:G 2026, 사건 적용)·Afghan/Somali MH(Genes 2025, 비-EAS)·Brazil MH(admixed)·triallelic SNP 친자(degraded, kinship). 앵커가 2025–26 최신이라 색인된 citer가 적어 **시점 스냅샷이며 추후 갱신 권장**.

---

## 8부. DL 아키텍처·tabular-DL·확장 데이터 문헌 (2026-05-30 추가)

> 본 저장소의 **다양한 DL 아키텍처 벤치마크**(docs/04 §24, `scripts/31·37`)와 **SSL 확장 데이터**(`scripts/32·33`)의 문헌적 근거. 각 DL 계열을 popgen/tabular-DL 선행연구에서 차용해 동일 프로토콜(genome-wide one-hot/codes, leakage-free 5-fold)로 적용 → RQ3(단순함이 이김)·RQ1(OSR) 검증. 결과 요약: **5개 DL 계열 전부 LogReg(one-hot) 79.6%에 ≥25p 짐**; SSL+ft가 supervised 대비 +3.6p(데이터 확장 동기).
> **서지 주의**: 일부 venue/연도는 웹검색 기반 — **제출 전 Semantic Scholar/CrossRef로 확정 [verify]**. URL 있는 항목은 확인됨.

### 8-A. Deep learning for population genetics — 아키텍처 계열

| 선행연구 | 무엇 | 우리 적용(§24) |
|---|---|---|
| **"Harnessing deep learning for population genetic inference"** (Nat Rev Genet 2023) https://www.nature.com/articles/s41576-023-00636-3 ; **"Deep Learning in Population Genetics"** (Korfmann/Gaggiotti/Fumagalli, GBE 2023) https://pmc.ncbi.nlm.nih.gov/articles/PMC9897193/ | DL-popgen 개관(구조·인구사·선택). CNN·MLP·AE 주력 | 아키텍처 선정의 우산 레퍼런스 |
| **Flagel, Brandvain, Schrider (2019)** *The unreasonable effectiveness of CNNs in population genetic inference*, MBE [verify] + **genomatnn** (Gower et al.) | 유전형 행렬 CNN; 마커 *순서*가 정확도 좌우 | **CNN1D**(§24.2) — 34.5% |
| **Romero et al. (2017)** *Diet Networks: Thin Parameters for Fat Data*, ICLR | p≫n SNP를 per-SNP 통계로 가중치 제약, 1000G ancestry | **EmbMLP**(entity-embedding, §24.2) — 29.7% |
| **Battey, Ralph, Kern (2020)** *Predicting geographic location from genetic variation with DNNs* (Locator), eLife | MLP 좌표 회귀; 경쟁작(Yang/Chen 2025) geo arm | **MLP**(§24.1) — 50–56% |
| **Mantes et al. (2023)** *Neural ADMIXTURE*, Nat Comput Sci https://www.nature.com/articles/s43588-023-00482-7 ; **popVAE** (Battey et al. 2021, eLife) [verify] | autoencoder로 ancestry/구조 추론 | **SupAE**(supervised AE, §24.2) — 32.5% |

### 8-B. Tabular deep learning SOTA — 벤치마크 추가군 (`scripts/37`)

- **Gorishniy et al. (2021)** *Revisiting Deep Learning Models for Tabular Data*, NeurIPS [verify] — tabular-DL SOTA 정립(**FT-Transformer** = feature tokenizer + transformer; **ResNet-tabular**). → 우리 **FT-Transformer** arm.
- **Arik & Pfister (2021)** *TabNet: Attentive Interpretable Tabular Learning*, AAAI [verify] — 순차적 attentive feature selection. → 우리 **TabNet** arm.
- **의의**: generic transformer/MLP가 DL을 과소평가했을 가능성을 SOTA로 차단. "SOTA tabular-DL조차 LogReg에 지면 RQ3 방탄."

### 8-C. 확장·조화 데이터 자원 — SSL 풀 + 깨끗한 RQ7 (`scripts/32·33`)

- **gnomAD HGDP+1KG harmonized resource** *A harmonized public resource of deeply sequenced diverse human genomes* (Koenig et al.), PMC https://pmc.ncbi.nlm.nih.gov/articles/PMC9900804/ — HGDP(780)+1KG(2,504) → QC 후 **3,942**(raw 4,091), **단일 GRCh38 jointly-called·phased(SHAPEIT5)**. → 우리 **SSL pretrain 풀**(scripts/33, 4,091) + **RQ7 in-callset 외부검증**(hg19↔hg38 mismatch=43% unseen 제거). 메타 `gnomad_meta_v1.tsv`.
- **Mallick et al. (2016)** *The Simons Genome Diversity Project: 300 genomes from 142 diverse populations*, Nature https://www.nature.com/articles/nature18964 — 공개 279 genome(EAS 47). 추가 다양성 풀 후보.
- **Bergström et al. (2020)** *Insights into human genetic variation and population history from 929 diverse genomes* (HGDP), Science [verify] — HGDP WGS 929 (gnomAD harmonized에 포함; 단독 추출 scripts/22).

### 8-D. UQ/conformal·open-set 방법 레퍼런스 (Paper 1 spine — 6부 옵션1 보강)

- Vovk, Gammerman & Shafer *Algorithmic Learning in a Random World*, Springer 1st ed. 2005 (doi:10.1007/b106715)·2nd ed. 2022 (doi:10.1007/978-3-031-06649-8); Angelopoulos & Bates 2023 *Conformal Prediction: A Gentle Introduction*, FnT ML 16(4):494–591 (arXiv:2107.07511) — split/Mondrian CP 근거. **[검증완료]**
- Papangelou et al. (2025) Frontiers Bioinf 5:1507448 (doi:10.3389/fbinf.2025.1507448) — CP-in-genomic-medicine(forensic 미적용, 6부 기재). **[검증완료; 이전 "Olsson et al. 2025" 오귀속 교정]**
- Geng, Huang & Chen 2021 *Recent Advances in Open Set Recognition: A Survey*, IEEE TPAMI 43(10):3614–3631 (doi:10.1109/TPAMI.2020.2981604); Yang, Zhou, Li & Liu 2024 *Generalized OOD Detection: A Survey*, IJCV 132:5635–5662 (doi:10.1007/s11263-024-02117-4) — open-set/OOD 서베이(6부 기재). **[검증완료]**

**Forensic reliability / LR-calibration 계보 (FSI:G 재앵커용, 검증완료 2026-06-04 — docs/08):**
- Ramos & González-Rodríguez 2013 *Reliable support: measuring calibration of likelihood ratios*, FSI 230(1–3):156 (doi:10.1016/j.forsciint.2013.04.014) — LR calibration 측정(우리 coverage의 forensic 보완 대상).
- Hannig & Iyer 2022 *Testing for calibration discrepancy of reported likelihood ratios in forensic science*, JRSS-A 185(1):267 (doi:10.1111/rssa.12747) — 보고된 LR의 calibration 검정.
- Marsico & Amigo 2025 *Ethical and security challenges in AI for forensic genetics*, FSI:G 76:103225 (doi:10.1016/j.fsigen.2025.103225) — AI trust를 *open problem*으로 제기(방법 미제시 → 우리가 방법 제공).

**경쟁/참조 — forensic ancestry ML (검증완료 2026-06-04 — docs/08):**
- Heinzel, Purucker, Hutter, Pfaffelhuber 2025 *Advancing biogeographical ancestry predictions through ML*, FSI:G 79:103290 (doi:10.1016/j.fsigen.2025.103290) — **가장 유사**(TabPFN 벤치, log-loss 보고; CP/OSR 없음).
- Wang C. et al. 2025 *A biogeographical ancestry inference pipeline using PCA-XGBoost…Asian populations*, FSI:G 77:103239 (doi:10.1016/j.fsigen.2025.103239) — 고정확도 광역 Asian(CP/OSR 없음).

**MH 패널 선택 / informativeness (검증완료 2026-06-04 — §4.7 minimal-panel 앵커):**
- de Barros Rodrigues et al. 2025 *Large-scale selection of highly informative microhaplotypes for ancestry inference and population specific informativeness*, FSI:G 74:103153 (doi:10.1016/j.fsigen.2024.103153) — **§4.7 직결**(그들=정확도 informativeness, 우리=trust frontier 추가).
- Cai et al. 2024 *Systematic analyses of AISNPs screening and classification algorithms…forensic biogeographic ancestry*, FSI 357:111975 (doi:10.1016/j.forsciint.2024.111975).
- Podini et al. 2026 *Defining key criteria for microhaplotype locus selection in forensic genetics (MH Working Group)*, FSI:G 83:103421 (doi:10.1016/j.fsigen.2026.103421) — Snipper=Phillips et al. 2007 FSI:G 1:273–280 (doi:10.1016/j.fsigen.2007.06.008).
> niche 확인: forensic DNA 전체에 conformal/OSR 적용 **0편**(검색 2026-06-04) → "first conformal+OSR for forensic ancestry" 성립(정밀 scope). 상세 docs/08.

> Paper 1 §References 인용은 모두 CrossRef/arXiv/publisher로 **프로그램적 검증 완료**(DOI 확보, 2026-06-04).

### 8-E. LexiconArxiv ML-method 선행연구 (arxiv-verified, 2026-05-30)

> arxiv ML 코퍼스(LexiconArxiv) 광범위 검색으로 확인된 **우리 방법론 축**의 원전. forensic 유전학(bio 저널)은 1~7부, 여기는 conformal·open-set·tabular-DL·SSL·calibration. (코퍼스 메타 = 제목·저자·venue·연도 확인; citation 수는 코퍼스 내 값. 최종본은 CrossRef 재확인.)

**Small-data tabular (RQ3 — 추가 테스트 후보)**
- **TabPFN** (Hollmann et al., ICLR 2023 notable top-25%) — 소표본 tabular 분류 SOTA(prior-fitted transformer). **n=504가 정확히 이 regime** → "소표본 tabular SOTA조차 선형에 지는가" 검증에 강력. (단 feature 한도로 **축소 패널(top-N) 필요**.) + TabPFN v2 (Ye et al., NeurIPS 2025), TuneTables (NeurIPS 2024).
- **NODE** (Popov et al., ICLR 2020); **TabReD** (Rubachev et al., ICLR 2025 Spotlight) — tabular-DL 벤치마크 *함정* 경고(우리 fair-comparison 주의와 직결).

**Tabular SSL (Paper 2 직접 근거)**
- **Scarf** (Bahri et al., ICLR 2022 Spotlight) — random feature corruption 대조학습. **우리 ADO-augmentation 대조학습의 명명된 직접 선행**(ADO≈Scarf corruption).
- **T-JEPA** (ICLR 2025, augmentation-free tabular SSL); **XTab** (ICML 2023, FT-Transformer cross-table pretrain); **Tabula** (NeurIPS 2025, single-cell tabular SSL FM — 유전체 인접); **TabDPT** (NeurIPS 2025, tabular FM scaling).

**Conformal under distribution shift (RQ6 ADO의 원리적 fix)**
- **Gibbs & Candès (NeurIPS 2021 Oral)** *Adaptive Conformal Inference Under Distribution Shift* — exchangeability 깨질 때 coverage 회복의 정준. **§4.5(ADO 50%에서 cov 0.91→0.80)의 직접 처방** → RQ6 "한계"를 "weighted/adaptive conformal로 처방 가능한 future work"로 격상.
- **Wasserstein-Regularized CP** (ICLR 2025); **Not all distributional shifts are equal** (Ai & Ren, ICML 2024); **CoDrug** (NeurIPS 2023, covariate-shift conformal in drug); **Conformal Validity for Any Distribution** (Prinster et al., ICML 2024); **Kandinsky CP** (ICML 2025, class/covariate-conditional 넘어 — Mondrian 일반화).

**OOD/calibration (RQ4)**: **Lee et al. (ICLR 2018)** *Training Confidence-calibrated Classifiers for Detecting OOD*; **DUQ** (van Amersfoort et al., ICML 2020 — 단일 결정 신경망 불확실성, 우리 실패한 deep-ensemble 대안); **EGonc** (NeurIPS 2024, energy-based open-set — 제안서 원안의 energy 아이디어).

**DNA LM (novelty 옵션3 landscape)**: NucEL (AAAI 2026), MxDNA (NeurIPS 2024) — DNA 사전학습(우리 마커는 원시서열 아니라 직접 적용은 아님).

> **활용 액션**: (a) **TabPFN을 축소 패널(top-200)로 추가 테스트**; (b) **Scarf를 Paper 2 SSL 근거로 인용**; (c) **Gibbs-Candès를 §4.5 RQ6 future-work 처방으로 인용**.

---

## 9부. Embedding을 *잘 만든* 접근 — Paper 2 후속 (2026-05-31)

> 동기: §24.2–24.3에서 학습 embedding 기반 DL(EmbMLP 29.7%·SupAE 32.5%·ResNet-tab 33.5%)이 one-hot 선형(79.6%)·심지어 one-hot MLP(50–56%)보다도 낮았다. 원인 = (i) n=504로 좋은 embedding 학습 불가, (ii) embedding bottleneck이 unseen-diplotype/명시 신호를 폐기(§24.3). → "embedding을 *실제로* 잘 작동시킨" 선행을 셋으로 정리하고 **모두 실험**(scripts/models/44·45·46, 양 노드). 서지 검증됨.

### 9-A. 유전형 p≫n 전용 embedding
- **Diet Networks** (Romero et al. 2017, *ICLR*) — 각 feature(SNP/one-hot)를 "샘플 전체 프로파일"로 임베딩 + **보조망이 첫 층 가중치를 예측** → p≫n 파라미터 폭발 회피. *우리 EmbMLP는 프록시였고 이 weight-prediction은 미시도* → **scripts/models/44 (DietNet)**.
- autoencoder/VAE 유전형 임베딩: popVAE (Battey 2021)·Neural ADMIXTURE (Mantes 2023, *Nat Comput Sci*)·*Hybrid AE with orthogonal latent space* (2022) — 구조 *표현*엔 좋으나 분류는 선형에 짐(우리 SupAE).

### 9-B. 대규모 사전학습 transfer (small-n 우회)
- **Nucleotide Transformer** (Dalla-Torre et al. 2024, *Nature Methods*)·**Genomics-FM** (bioRxiv 2024)·DNABERT/HyenaDNA — 거대 코퍼스 사전학습 임베딩 transfer. **caveat**: 염기 *서열* 임베딩이라 유전형 행렬 직접 적용 불가 → MH amplicon 서열(<300bp)을 NT로 임베딩하는 우회 → **scripts/models/46 (NT-transfer)**.

### 9-C. 더 나은 범주형-tabular embedding
- **On Embeddings for Numerical Features in Tabular DL** (Gorishniy, Rubachev & Babenko 2022, *NeurIPS*) — *임베딩 방식*이 성능 좌우.
- **Random Effects for High-Cardinality Categorical / LMMNN** (Simchoni & Rosset 2021, *NeurIPS*) — 고-카디널리티 범주형을 *혼합모형 random-effect*로 임베딩(평균으로 shrinkage). 우리 diplotype이 고-카디널리티(마커당 ≤278종) → **scripts/models/45 (random-effects embedding)**.
- Entity Embeddings (Guo & Berkhahn 2016 [verify]) · categorical encoder 벤치마크 (Matteucci et al. 2023, *NeurIPS* D&B) · DHE (Kang et al. 2021, *KDD*) · Feat2Vec (*ICLR* 2018).

### 9-D. 결정적 caveat — 문헌이 우리 negative를 뒷받침
- **"Genomic Foundationless Models: Pretraining Does Not Promise Performance"** (bioRxiv 2024) — 유전체에서 사전학습/임베딩이 *항상* 단순 방법을 이기진 않음.
- 다수 서베이: **"좋은 임베딩은 규모에서 온다"** → n=504/4091에서 임베딩이 진 건 *방법*이 아니라 *규모* 문제. 우회 = (9-B) 외부 대규모 transfer 또는 (9-A) Diet-Net식 파라미터 공유.

→ **실험 계획**: 44(DietNet)·45(RandomEffectEmb)·46(NT-transfer)을 동일 프로토콜(genome-wide, 5-fold, acc+far-OOD AUROC)로, LogReg 79.6%·EmbMLP 29.7% 대비. 결과는 docs/04 §27 + 본 9부에 반영.

---

## 10부. Notion §5 동기화 2차 — 댓글 제안 (2026-06-01)

> Notion "5. 관련 연구" 페이지 댓글(2026-06-01)로 추가 제안된 선행연구. Notion **§5.10**(비-ML forensic 마커) + **§5.11**(친족·대륙조상·딥러닝 집단구조 3묶음)에 대응. 7부(2026-05-30 동기화)의 연장.
> **관찰 메모(댓글)**: "forensic-genetics ML은 해당 분야에서 최근까지도 *간단한 모델*(고전 통계·로지스틱·RF/XGBoost) 위주" → 우리 **RQ3 · docs/04 §24.4–24.5**(규제 선형이 FT-Transformer·TabNet·TabPFN 등 복잡 모델 능가)와 같은 방향의 외부 정황. 우리 기여는 정확도 경쟁이 아니라 **신뢰정량화(conformal/open-set) 축**.
> **서지 주의**: Wiley·Genome Research·bioRxiv 본문 paywall/인증벽 → Crossref/OpenAlex/PMC로 서지·abstract 확보. 제출 전 CrossRef 재확인 권장.

### 10-A. 비-ML forensic 마커 패널 (Notion §5.10)

#### 구이저우 Miao·Bouyei X-염색체 multi-InDel 패널 (BMC Genomics 2024)
**제목**: *Exploring the forensic effectiveness and population genetic differentiation in Guizhou Miao and Bouyei group by the self-constructed panel of X chromosomal multi-insertion/deletions*
**URL**: https://link.springer.com/article/10.1186/s12864-024-11088-2 — DOI 10.1186/s12864-024-11088-2 (오픈)
**저자**: Huang, Gu, Ran, Chen, Tian, Zhong, Ren, Wang, Yang, Ji, Wan, Huang, Zhang, Jin (구이저우 의대 forensic-genetics 그룹)
- 자작 **22 X-multi-InDel + 1 X-STR** 패널, 종합 식별력 **>0.999999999**, 개인식별·친자감정(parentage). 구이저우 3집단 저-FST(저분화). **방법 = 고전 forensic 통계(식별력·HWE·LD·집단거리, ML 미사용)**.
- **차별점**: 같은 EAS(중국 소수민족) forensic 마커 도메인이나 X-InDel·고전통계·kinship/ID — 우리 상염색체 MH diplotype·ML·ancestry 분류·신뢰층과 구분. §7-B 구이저우 7민족 MH(JTGG 2025)와 같은 그룹(Xiaoye Jin). "근연 EAS 저분화"의 독립 사례.

### 10-B. 친족(kinship) 분석 (Notion §5.11.1)

#### DeepKin — CNN으로 저커버리지/고대 DNA 혈연 예측 (Mol. Ecol. Resour. 2025)
**제목**: *DeepKin: Predicting Relatedness From Low-Coverage Genomes and Palaeogenomes With Convolutional Neural Networks* (Güler, Yılmaz, Katırcıoğlu, Kantar, Ünver, Vural, Altınışık, Akbas, Somel)
**URL**: https://onlinelibrary.wiley.com/doi/full/10.1111/1755-0998.70032 — DOI 10.1111/1755-0998.70032
- **CNN을 시뮬레이션으로 생성한 유전 데이터로 학습** → 저커버리지/palaeogenome 3촌까지 혈연 예측, 공유 SNP 10K↑ 정확도 >90%, 기존 READv2 동급↑.
- **차별점**: 친족 축(우리 ancestry 분류와 다름, §5.4 계열). *시뮬레이션 학습데이터*가 데이터 구축 측면 참고가치. CP/OSR 없음.

#### 동적 SNP 선택 우도비 친족 추론 (Frontiers in Genetics 2025)
**제목**: *A likelihood ratio framework for inferring close kinship from dynamically selected SNPs* (Ge, Budowle, Cariaso, Mittelman, Mittelman)
**URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC12325062/
- **gnomAD v4** 큐레이션 222,366 SNP, 동적 SNP 선택(MAF·유전거리 임계), **우도비(LR)·IBD 고전 통계(ML 아님)**. 1000G 1,200 부모-자녀 등 5집단(EAS 포함)으로 2촌까지 검증.
- **차별점**: 친족·고전통계. *gnomAD v4 활용*은 우리 RQ7(gnomAD HGDP+1KG harmonize, §8-C)과 데이터 측면 접점.

### 10-C. 대륙 간·지리 조상 추론 (Notion §5.11.2)

#### ★ 다변량 통계 + ML BGA (Scientific Reports 2022) — RQ3 외부 입증
**제목**: *Multivariate statistical approach and machine learning for the evaluation of biogeographical ancestry inference in the forensic field* (Alladio, Poggiali, Cosenza, Pilli)
**URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC9148302/ — Scientific Reports (2022)
- 3,557명(1000G+SGDP+HGDP), 상용 forensic SNP 패널 4종(EUROFORGEN-128·ForenSeq-55·MAPlex-144·ThermoFisher-165). PCA + **PLS-DA vs XGBoost vs STRUCTURE**.
- **결과**: 대륙 단위 **PLS-DA가 XGBoost보다 견고**(AUC≈1.0), **대륙-내 fine-scale 급락** → 상용 패널 fine-scale 분별 부적합.
- **★ 우리와의 직접 연결**: (i) 단순/선형(PLS-DA) > 부스팅(XGBoost) = 우리 **RQ3·§24.4–24.5**와 같은 결, (ii) fine-scale 급락 = 근연 EAS 5집단 난도 독립 입증, (iii) 점추정 AUC 중심·CP/OSR 없음(우리 차별). 저자(Alladio·Cosenza·Pilli)가 §7-A Kinship NGS(ESA 2024)와 동일 이탈리아 그룹.

> ESA 2025 *Predictive modeling of biogeographical ancestry…* (PII S0957417425032774)도 댓글에 재언급 → **이미 §7-A**에 정리(중복).

### 10-D. 딥러닝 기반 집단구조 추론 (Notion §5.11.3)

#### HaploNet — VAE 기반 haplotype/집단구조 추론 (Genome Research 2022) — §6 옵션10 상세화
**제목**: *Haplotype and population structure inference using neural networks in whole-genome sequencing data* (Meisner & Albrechtsen)
**URL**: https://genome.cshlp.org/content/32/8/1542 — Genome Research 32(8):1542
- 가우시안 혼합 **VAE(변분 오토인코더)**로 phased haplotype 군집 학습 → 집단구조·admixture 추론. **1000G·UK Biobank**(비지도).
- **차별점**: 비지도 VAE 구조추론(라벨 분류·신뢰층 아님) vs 우리 지도 fine-scale + CP/OSR. 단 1000G·DL 계보 인접. (§6 옵션10에 도구로만 기재됐던 것의 정식 서지.)

#### ★ Transparent/Generalizable DL for genomic ancestry (bioRxiv 2025) — §27/RQ7 직접 인접
**제목**: *A Transparent and Generalizable Deep Learning Framework for Genomic Ancestry Prediction*
**URL**: https://www.biorxiv.org/content/10.1101/2025.08.26.672448v1
- **Diet Network(DietNet)** 딥러닝으로 유전체 조상 예측, **일반화(다집단)·설명가능성(투명성)** 강조. CARTaGENE(몬트리올) biobank 등 다집단 일반화 평가.
- **★ 차별점/연결**: 우리 **docs/04 §27 DietNet 임베딩(73.6%) · RQ7(일반화/전이)**과 직접 맞닿음. "일반화+투명성"은 우리 *신뢰성 축*과 같은 문제의식이나, 우리는 거기에 **분포-free conformal 보장 + open-set 거부**를 추가. (Romero 2017 ICLR DietNet[§8-A·§9-A]의 *ancestry 일반화 후속*에 해당 — 원전과 구분.)
