# 문헌 정독 + 경쟁 논문 매핑 (v1)

**검토일**: 2026-05-26
**갱신**: 2026-05-30 — Notion §5(Previous Studies) 동기화 → **7부** 추가(댓글 제안 + forward-citation); **8부** 추가(DL-popgen 아키텍처·tabular-DL SOTA·확장 데이터 자원 = §24 실험 근거).
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
- Olsson H et al. (2025) Frontiers Bioinf — first CP in genomic medicine (no forensic)
- Liu Y et al. (2024) IJCV — open-set/OOD 서베이

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

- Vovk, Gammerman, Shafer *Algorithmic Learning in a Random World* (conformal 이론) [verify]; Angelopoulos & Bates *A Gentle Introduction to Conformal Prediction* [verify] — split/Mondrian CP 근거.
- Olsson et al. (2025) Frontiers Bioinf — CP-in-genomic-medicine 최초(forensic 미적용, 6부 기재).
- Liu et al. (2024) IJCV — open-set/OOD 서베이(6부 기재).
> Paper 1 인용은 모두 `paperwork/Paper1_draft` §References의 [verify] 절차로 확정.
