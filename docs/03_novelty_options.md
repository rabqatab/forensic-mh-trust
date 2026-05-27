# Novelty 옵션 메뉴 (v2 — 선행연구 보강)

**작성일**: 2026-05-26
**작성자**: 조민한 (AI 박사 과정)
**기반**: `01_proposal_review.md`, `02_literature_landscape.md`
**프레임워크**: creative-thinking-for-research (Koestler, Boden, Kauffman, Janusian)

> **v2 변경점**: 각 옵션마다 직접 선행연구 2~6편 인용 + 갭 정확히 명시. Zhang (2025) "AI in forensic genetics" 리뷰 (JTGG, 103 refs)에 **conformal prediction · open-set · foundation model · SSL이 전혀 미언급**됨을 anchor evidence로 활용.

---

## 1부. 설계 공간 — 변경 가능한 6개 축

원안은 6개 축 모두 한 값으로 고정. 한 dimension만 바꿔도 새 contribution이 가능하다.

| 축 | 원안 | 가능한 변경 |
|---|---|---|
| **(1) Sub-task** | 5-pop ancestry classification | kinship, identification, mixture deconvolution, phenotype prediction, age estimation(methylation), biological sex, body-fluid ID, novel-population discovery |
| **(2) Marker** | MH (412) | SNP, STR, Y-STR, mtDNA, methylation, full WGS reads, microbiome |
| **(3) Dataset** | 1000G EAS 504 | + KoVariome 50, + KPGP, + GenomeAsia 100K, + HGDP, + SGDP, + 자체 시뮬레이션, + ancient DNA |
| **(4) Method** | XGBoost | RF, DNA foundation model 파인튜닝(HyenaDNA·Caduceus·NT·Evo), SSL contrastive, diffusion, GNN, RL/active learning, federated, conformal/Bayesian UQ |
| **(5) Output** | discrete label | LR with CI, prediction set + reject, per-contributor posterior, continuous ancestry vector, discovered cluster, generated synthetic profile |
| **(6) Contribution** | 'minimum panel size' | new dataset, new method, new benchmark, new framework, theory, tool |

### Hidden constraint 9개 (F4)

1. "분류할 집단이 reference에 모두 있다" → **open-set classification**
2. "각 샘플은 한 명의 DNA만 포함" → **mixture deconvolution**
3. "panel은 fixed" → **adaptive sequencing (RL/active learning)**
4. "supervised label 필요" → **self-supervised pretraining**
5. "출력은 point estimate" → **calibrated probability + CI**
6. "phasing은 정확" → **phasing-error-aware modeling**
7. "마커는 인간이 정의(MicroHapDB)" → **DNA foundation model이 representation 학습**
8. "데이터는 한 lab에서 모임" → **federated learning**
9. "데이터 공유 자유" → **differential privacy**

---

## 2부. 옵션 메뉴 — 10개 (옵션별 선행연구·갭 보강)

> **공통 anchor citation**: Zhang H (2025) *"Artificial intelligence in forensic genetics: applications and ethical challenges"* JTGG. https://www.oaepublish.com/articles/jtgg.2025.76 — 103 refs 리뷰. **인용된 ancestry 메서드: XGBoost, PLS-DA, soft clustering, decision tree, Bayesian network. Conformal/Open-set/Foundation model/SSL은 0회 언급**. ← 본 옵션 1~4의 novelty 정량 증거.

---

### 🔵 Group A — UQ/안전성 계열 (원안 골격 살림)

#### 옵션 1. Conformal Prediction + Open-set classification

**한 줄**: 분포-free 신뢰구간 + KOR-like unknown reject

- **변경 축**: (5)Output
- **메서드**: Mondrian Conformal Prediction (`mapie` 라이브러리) + Energy-based OOD or OpenMax
- **AI ⭐⭐⭐⭐⭐ / Forensic ⭐⭐⭐⭐ / 차별 ⭐⭐⭐⭐⭐**
- **공수 5-7w, 재활용 90%, 데이터 1000G + 검증용 KOR 1-10명**

**📚 선행연구 (Prior Art)**:

1. **Olsson H et al. (2025)** *"Reliable machine learning models in genomic medicine using conformal prediction"* Frontiers Bioinf. https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2025.1507448/full
   - **첫 CP-in-genomic-medicine 적용**. Drug response (infliximab), B-cell lymphoma subtype, afatinib sensitivity 회귀. 95.2% empirical coverage achieved.
   - **본 옵션과의 갭**: forensic ancestry 미적용; Mondrian CP를 conceptually 언급만 하고 실험에는 안 씀. → **본 옵션은 forensic 첫 적용 + Mondrian 실험 구현 모두 신규**.

2. **Vovk V, Gammerman A, Shafer G (2005-)** *Algorithmic Learning in a Random World* — CP 이론 origin. Mondrian conformal: per-class validity guarantee.

3. **Sun et al. PRS confidence** — Mondrian CCP를 polygenic risk score 신뢰구간에 적용. Disease risk 도메인.

4. **Open-set / OOD 일반 서베이**: Liu Y et al. (2024) IJCV *"Dissecting Out-of-Distribution Detection and Open-Set Recognition"* https://link.springer.com/article/10.1007/s11263-024-02222-4 — 컴비전 중심. **유전체·forensic 0편**.

5. **Zhang (2025) JTGG 리뷰**: forensic genetics에서 CP/OSR 미언급 (verified).

**🎯 갭 (Novelty Gap)**: forensic ancestry inference에 **(i) Conformal Prediction + (ii) Open-set reject** 결합은 0편. Zhang 2025 리뷰가 이 갭의 정량 증거.

---

#### 옵션 2. Bayesian Deep MH classifier (aleatoric vs epistemic 분리)

**한 줄**: aleatoric vs epistemic uncertainty 분리한 forensic LR

- **변경 축**: (4)Method + (5)Output
- **메서드**: MC Dropout / SWAG / Deep Ensemble + uncertainty decomposition
- **AI ⭐⭐⭐⭐⭐ / Forensic ⭐⭐⭐⭐ / 차별 ⭐⭐⭐⭐**
- **공수 7-9w, 위험: BNN 수렴 안되면 ensemble로 fallback**

**📚 선행연구**:

1. **Toneyan S, Koo P (2024)** *"Uncertainty-aware genomic deep learning with knowledge distillation"* (DEGU) bioRxiv. https://www.biorxiv.org/content/10.1101/2024.11.13.623485.full.pdf — PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11601481/
   - Ensemble → 단일 모델로 knowledge distillation, aleatoric+epistemic 분리.
   - **본 옵션과의 갭**: variant effect prediction에 적용; **ancestry/forensic 0편**.

2. **Depeweg S et al. (2018) ICML** *"Decomposition of Uncertainty in Bayesian Deep Learning for the Detection and Discrimination of Aleatoric and Epistemic Uncertainty"* https://proceedings.mlr.press/v80/depeweg18a/depeweg18a.pdf — 이론적 분해 방법론.

3. **Lakshminarayanan B et al. (2017) NeurIPS** — Deep Ensembles 원전, BNN보다 간단·robust.

4. **EpICC (2022) Sci Rep** *"Bayesian neural network model with uncertainty correction for cancer classification"* https://www.nature.com/articles/s41598-022-18874-6 — cancer transcriptomics, NOT forensic ancestry.

5. **Zhang (2025) JTGG**: forensic ancestry uncertainty quantification 부재.

**🎯 갭**: forensic ancestry에서 aleatoric/epistemic 분리는 0편. 또한 epistemic을 small-sample-size signal로 활용해 "어떤 marker가 더 필요한지" 표시하면 옵션 9(active learning)와 자연 연결.

---

### 🟢 Group B — DNA Foundation Model 계열 (2024-2025 enabler)

#### 옵션 3. HyenaDNA / Caduceus / Nucleotide Transformer 파인튜닝

**한 줄**: 패널 설계 없이 raw genomic context window에서 ancestry inference

- **변경 축**: (2)Marker(raw seq) + (4)Method
- **메서드**: HyenaDNA-1k/32k/100k(Apache-2.0), Caduceus-Ph(Apache-2.0), NT-v2 → classification head
- **AI ⭐⭐⭐⭐⭐ / Forensic ⭐⭐⭐ / 차별 ⭐⭐⭐⭐** (⭐⭐⭐⭐⭐에서 ⭐⭐⭐⭐로 하향: bioRxiv 2025 직접 경쟁작 발견)
- **공수 7-10w (GPU 시간 크다), GPU >=4xA100 권장**

**📚 선행연구**:

1. **Nguyen E et al. (2023) NeurIPS** *"HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution"* https://arxiv.org/pdf/2306.15794 — implicit conv, million-token context.

2. **Schiff Y et al. (2024) ICML** *"Caduceus: Bi-Directional Equivariant Long-Range DNA Sequence Modeling"* https://arxiv.org/pdf/2403.03234 — Mamba 기반, reverse-complement equivariant. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12189541/.

3. **Dalla-Torre H et al. (2024) Nat Methods** — Nucleotide Transformer v2, 2.5B params, 6-mer.

4. ⚠️ **DIRECT COMPETITOR** — **(2025-11)** *"Zygosity-Aware DNA Language Modeling Improves Ancestry and Gene Expression Prediction"* bioRxiv. https://www.biorxiv.org/content/10.1101/2025.11.19.689326 — **DNA LM으로 ancestry 예측 직접 다룸**. 6개월 된 매우 신선한 work. (paywall/403로 detail 미확보 — Google Scholar/Twitter alert 권장)
   - **본 옵션과의 차별화 필요**: 이 paper는 ancestry & gene expression 둘 다, **forensic-specific framing은 부재**일 가능성. 본 옵션은 forensic context(LR, 검증, court admissibility) 위주로 차별화 가능. 또는 EAS fine-scale로 좁히기.

5. **Tang Z et al. (2025) Nat Commun** *"Benchmarking DNA foundation models for genomic and genetic tasks"* https://www.nature.com/articles/s41467-025-65823-8 — 5개 FM(DNABERT-2, NTv2, HyenaDNA, Caduceus-Ph, GROVER) zero-shot 비교. **Forensic task 미포함**.

6. **Zhang (2025) JTGG**: forensic genetics에서 DNA foundation model 0회 언급.

**🎯 갭**: forensic ancestry에 DNA FM 적용은 zero, **단 2025-11 bioRxiv가 ancestry에는 적용 시작** → 본 옵션은 **forensic-specific framing**(EAS fine-scale, LR output, admissibility evaluation)으로 차별화해야 함. 6개월 내 follow-up 경쟁 risk.

---

#### 옵션 4. SSL contrastive pretraining → multi-task forensic head

**한 줄**: 1000G WGS로 contrastive pretrain → ancestry + kinship + sex + body fluid 한 모델로

- **변경 축**: (1)Sub-task(multi) + (4)Method
- **메서드**: SimCLR / masked SNP modeling on 1000G WGS → multi-task heads
- **AI ⭐⭐⭐⭐⭐ / Forensic ⭐⭐⭐⭐⭐ / 차별 ⭐⭐⭐⭐⭐**
- **공수 8-10w, 위험: genomic augmentation 디자인 미해결**

**📚 선행연구**:

1. **GENEREL** *"Unified Representation of Genomic and Biomedical Concepts through Multi-Task, Multi-Source Contrastive Learning"* arxiv 2410.10144. https://arxiv.org/html/2410.10144v1 — SNP+biomedical concept contrastive. **Medical, NOT forensic**.

2. **Taleb A et al. (2022) CVPR** *"ContIG: Self-supervised Multimodal Contrastive Learning for Medical Imaging With Genetics"* https://openaccess.thecvf.com/content/CVPR2022/papers/Taleb_ContIG_Self-Supervised_Multimodal_Contrastive_Learning_for_Medical_Imaging_With_Genetics_CVPR_2022_paper.pdf — imaging+genetics, **NOT forensic ancestry**.

3. **(2024) Nat Mach Intell** *"Delineating the effective use of self-supervised learning in single-cell genomics"* https://www.nature.com/articles/s42256-024-00934-3 — scRNA-seq SSL benchmark. **NOT forensic**.

4. **(2023) Commun Biol** *"A self-supervised deep learning method for data-efficient training in genomics"* https://www.nature.com/articles/s42003-023-05310-2 — regulatory genomics.

5. **Multi-task forensic genetic precedents**:
   - **(2022) Frontiers Genet** *"Forensic efficiencies of individual identification, kinship testing and ancestral inference in three Yunnan groups based on a self-developed multiple DIP panel"* https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2022.1057231/full — single panel, multi-task **but separate models per task**. SSL 없음.
   - **Kinship + ML NGS panel (2024) Expert Syst Appl** https://www.sciencedirect.com/science/article/pii/S0957417424030288

6. **Auto-branch multi-task (2025) Frontiers Genet** *"Auto-branch multi-task learning for simultaneous prediction of multiple correlated traits associated with Alzheimer's disease"* https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2025.1538544/full — multi-task genetic, but disease NOT forensic.

7. **Zhang (2025) JTGG**: forensic SSL pretraining 0회 언급.

**🎯 갭**: SSL contrastive pretraining + multi-task forensic head 결합은 0편. "Forensic foundation model" framing이 정착되지 않은 영역 — first-mover advantage 가능.

---

### 🟣 Group C — Privacy / Federation 계열

#### 옵션 5. Differentially-private generative MH profile synthesizer

**한 줄**: 1000G EAS에서 ε-DP guarantee로 합성 forensic profile 생성

- **변경 축**: (4)diffusion/VAE + (5)generated + (6)dataset
- **메서드**: DP-SGD VAE 또는 DP-diffusion on MH; downstream ML utility 평가
- **AI ⭐⭐⭐⭐⭐ / Forensic ⭐⭐⭐⭐⭐ / 차별 ⭐⭐⭐⭐** (⭐⭐⭐⭐⭐에서 ⭐⭐⭐⭐로 하향: 매우 최근 직접 경쟁 발견)

**📚 선행연구**:

1. ⚠️ **DIRECT COMPETITOR** — **(2025-10)** *"DP-SNP-TIHMM: Differentially Private, Time-Inhomogeneous Hidden Markov Models for Synthesizing Genome-Wide Association Datasets"* arxiv 2510.05777. https://arxiv.org/pdf/2510.05777 — **이미 DP synthetic SNP 발표됨**. ε∈[1,10], δ=10⁻⁴.

2. ⚠️ **DIRECT COMPETITOR** — **(2026-03)** *"SNPgen: Phenotype-Supervised Genotype Representation and Synthetic Data Generation via Latent Diffusion"* arxiv 2603.10873. https://arxiv.org/pdf/2603.10873 — VAE + Latent Diffusion으로 genotype 합성. Zero identical matches, near-random membership inference attack 방어 claim. **Disease label conditioning이 main, ancestry/forensic은 부재**.

3. **DP-SGD foundational**: Abadi et al. (2016) — DP-SGD origin. Membership inference attacks → Shokri (2017).

4. **GAN-based DP**: PATE-GAN, DP-CGAN. Mixed-type DP synthetic data: arxiv 1912.03250.

5. **Forensic genomics privacy 측면**: 아직 미성숙. Zhang (2025) JTGG에서 "unauthorized access incidents, privacy by default needed" 언급되지만 generative privacy preserving 0편.

**🎯 갭**: DP-SNP-TIHMM과 SNPgen이 **disease/GWAS context**에서 작업; **forensic MH context는 부재** + ancestry-conditioning DP generation은 아직 unexplored. 본 옵션은:
- (a) MH-specific (vs SNP-only)
- (b) ancestry-conditional generation
- (c) forensic utility evaluation (LR preservation, mixture deconvolution downstream task)

3가지 차별점으로 정의해야 함. 단 SNPgen이 2026-03 발표라 매우 hot 영역.

---

#### 옵션 6. Federated forensic ancestry

**한 줄**: 두 lab이 각자 데이터 보유, FedAvg로 공동 학습 — Korean NFS + 1000G 시나리오

- **변경 축**: (4) + (6)
- **메서드**: FedAvg / FedProx, secure aggregation, optionally + DP
- **AI ⭐⭐⭐⭐ / Forensic ⭐⭐⭐⭐⭐ / 차별 ⭐⭐⭐** (⭐⭐⭐⭐에서 ⭐⭐⭐로 하향: 1000G+UKBB 직접 federated 사례 발견)

**📚 선행연구**:

1. ⚠️ **DIRECT** — **(2024) Frontiers Big Data** *"Efficacy of federated learning on genomic data: a study on the UK Biobank and the 1000 Genomes Project"* https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2024.1266031/full — **1000G + UK Biobank federated phenotype prediction + ancestry-from-genotype 분석**. PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10937521/. **본 옵션의 핵심 setup을 이미 수행함**.

2. **(2025) arxiv 2505.07188** *"Securing Genomic Data Against Inference Attacks in Federated Learning Environments"* https://arxiv.org/html/2505.07188v1 — security against inference attack.

3. **(2025) eprint** *"Privacy-Preserving Federated Inference for Genomic Analysis with Homomorphic Encryption"* https://eprint.iacr.org/2025/1515.pdf — HE-based.

4. **MDPI Genes (2024)** *"Federated Learning: Breaking Down Barriers in Global Genomic Research"* https://www.mdpi.com/2073-4425/15/12/1650.

5. **Zhang (2025) JTGG**: federated forensic 미언급.

**🎯 갭**: Frontiers 2024 work이 ancestry-from-genotype federated 이미 함 → 본 옵션은 차별화 어려움. **차별화 가능 vector**:
- (a) **forensic-specific** (forensic LR with federated training, court admissibility 평가)
- (b) **MH-specific** (Frontiers 2024는 SNP)
- (c) **cross-jurisdictional** (Korean NFS + US Lab + EU 시나리오 + GDPR/PIPA compliance)

순수 federated만으로는 약하니 + DP 또는 + admissibility 다른 vector 추가 권장.

---

### 🟠 Group D — Forensic-realistic 계열

#### 옵션 7. In silico mixture + per-contributor ancestry

**한 줄**: 2~3인 mixture에서 contributor별 ancestry posterior 분리

- **변경 축**: (1)mixture + (4)BSS
- **메서드**: EM/Bayesian mixture, NMF, 또는 deep blind source separation
- **AI ⭐⭐⭐⭐ / Forensic ⭐⭐⭐⭐⭐ / 차별 ⭐⭐⭐⭐**

**📚 선행연구**:

1. ⚠️ **DIRECT** — **(2024) FSI:G** *"Using simulated microhaplotype genotyping data to evaluate the value of machine learning algorithms for inferring DNA mixture contributor numbers"* https://www.sciencedirect.com/science/article/abs/pii/S1872497324000024 — **MH + ML mixture 직접 다룸 — 단 NOC(number of contributors) 추정만, ancestry-of-each-contributor 미수행**.

2. **(2025) FSI:G** *"MixDeR: A SNP mixture deconvolution workflow for forensic genetic genealogy"* https://www.fsigenetics.com/article/S1872-4973(25)00004-3/fulltext — SNP mixture deconvolution workflow, R/Shiny. EuroForMix 기반.

3. **(2025) Genes (MDPI)** *"Mixture Deconvolution with Massively Parallel Sequencing Data: Microhaplotypes Versus Short Tandem Repeats"* https://doi.org/10.3390/genes16091105 — MH vs STR 비교. MPSproto 사용.

4. **(2026) MDPI Genes** *"DNA Mixture Deconvolution: A Four-Strategy Framework"* https://www.mdpi.com/2073-4425/17/4/434 — 4가지 strategy 통합 framework.

5. **Probabilistic genotyping (PG)**: EuroForMix, STRmix, MPSproto, TrueAllele — 모두 MCMC 기반 PG. ML 통합은 부분적.

**🎯 갭**: MH mixture deconvolution은 활발 (2024-2026 4편). NOC 추정 + 일반 deconvolution 위주. **갭**: per-contributor ancestry posterior 분리 + 신뢰구간 출력은 미해결. 본 옵션은 "5집단 prior 위에서 mixture decomposition + per-contributor population assignment with uncertainty"로 framing.

---

#### 옵션 8. Degraded/low-template DNA robustness benchmark

**한 줄**: ADO 10/20/50% 시뮬레이션에서 robust MH 패널 선정

- **변경 축**: (3)+ADO sim + (6)benchmark
- **AI ⭐⭐⭐ / Forensic ⭐⭐⭐⭐⭐ / 차별 ⭐⭐⭐⭐**

**📚 선행연구**:

1. **Tvedebrink T et al. (2010) FSI:G** *"Estimating drop-out probabilities in forensic DNA samples: A simulation approach"* https://www.sciencedirect.com/science/article/abs/pii/S1872497310001924 — ADO probability 추정의 고전.

2. **Italian MH 76-panel (2025)** *"Microhaplotypes in forensic genetics: From exploration to application in degraded DNA specimens"* https://pubmed.ncbi.nlm.nih.gov/41270348/ — **0.05ng까지 작동 확인**. Direct precedent.

3. **(2025) FSI:G** *"Explainable artificial intelligence in forensic DNA analysis: Alleles identification in challenging electropherograms using supervised machine learning methods"* https://www.sciencedirect.com/science/article/abs/pii/S1872497325000699 — ML로 EPG signal classification.

4. **NIJ Report 249157** *"Low-Template DNA Mixture Interpretation"* https://www.ojp.gov/pdffiles1/nij/grants/249157.pdf — 공식 NIJ 가이드.

5. **(2020) FSI:G** *"Estimating the number of contributors to a DNA profile using decision trees"* https://www.sciencedirect.com/science/article/abs/pii/S1872497320301794 — DT for NOC under ADO.

**🎯 갭**: ADO 모델은 있지만 **EAS MH ancestry-specific robustness benchmark**는 부재. 본 옵션은 ADO 시뮬레이션 → ancestry 정확도 degradation curve → robust-marker subset 출력. 또한 Wei 2025의 phasing error와 ADO를 동시에 모델링하면 **double-perturbation analysis**로 차별화.

---

### 🟡 Group E — 새로운 task 계열

#### 옵션 9. Adaptive sequencing via active learning / RL

**한 줄**: Agent가 매 step uncertainty 최대 감소 마커를 다음 시퀀싱 대상으로 선택

- **변경 축**: (1)policy + (4)RL/AL
- **메서드**: BALD acquisition / EIG / PPO with -uncertainty reward
- **AI ⭐⭐⭐⭐⭐ / Forensic ⭐⭐⭐⭐ / 차별 ⭐⭐⭐⭐**

**📚 선행연구**:

1. **(2025) arxiv 2501.04718** *"Knowledge-Guided Gene Panel Selection for Label-Free Single-Cell RNA-Seq Data: A Reinforcement Learning Perspective"* https://arxiv.org/html/2501.04718v2 — **RL for gene panel selection** (scRNA-seq). Reward = expert behavior. **Forensic 미적용**.

2. **(2024) FSI:G** *"Profiling age and body fluid DNA methylation markers using nanopore adaptive sampling"* https://www.sciencedirect.com/science/article/pii/S1872497324000425 — **Nanopore adaptive sampling forensic**. 단 RL 없음, target region 사전 정의.

3. **(2024) FSI:G** *"Exploring nanopore direct sequencing performance of forensic STRs, SNPs, InDels, and DNA methylation markers in a single assay"* https://www.sciencedirect.com/science/article/pii/S1872497324001509 — adaptive sampling for forensic markers.

4. **BALD (Bayesian Active Learning by Disagreement)**: Houlsby et al. (2011) — active learning 핵심.

5. **Active and Adaptive Sequential learning** arxiv 1805.11710 — general framework.

**🎯 갭**: RL/AL for forensic adaptive sequencing은 **부재**. Nanopore adaptive sampling은 forensic에 있지만 RL/AL 없이 pre-defined regions. 본 옵션은 **RL/AL agent + adaptive Nanopore + uncertainty-aware stop criterion** 결합 — 실제 forensic lab 비용 절감.

---

#### 옵션 10. Cryptic population substructure discovery

**한 줄**: SSL + Dirichlet process / VAE로 1000G EAS 내부 숨은 substructure 발견

- **변경 축**: (1)discovery + (4)+(5)cluster
- **메서드**: SSL embedding + DP-GMM 또는 HaploNet/Neural ADMIXTURE 비교
- **AI ⭐⭐⭐⭐ / Forensic ⭐⭐⭐⭐ / 차별 ⭐⭐⭐** (⭐⭐⭐⭐에서 ⭐⭐⭐로 하향: 직접 prior art 다수)

**📚 선행연구**:

1. ⚠️ **DIRECT** — **(2019) PLoS Genet** *"UMAP reveals cryptic population structure and phenotype heterogeneity in large genomic cohorts"* https://pmc.ncbi.nlm.nih.gov/articles/PMC6853336/ — **1000G에서 cryptic structure 발견함**. American Hispanic, UK biobank.

2. **HaploNet** — VAE for population structure & ancestry proportion (1000G, UK Biobank).

3. **Neural ADMIXTURE** — multi-headed autoencoder for admixture, SGDP + HGDP.

4. **t-SNE on 1000G** — populations previously thought singular split into multiple.

5. **(2023) PMC9897193** *"Deep Learning in Population Genetics"* — survey, includes unsupervised methods.

**🎯 갭**: cryptic structure discovery는 활발한 영역 — 본 옵션이 약함. **차별 vector**:
- (a) **forensic-specific framing** (cryptic substructure가 ancestry call에 미치는 영향 정량화 + court admissibility 함의)
- (b) **EAS-specific** (UMAP/HaploNet은 global pop 위주; EAS fine-scale cryptic structure는 덜 연구됨)
- (c) **SSL embedding 비교** (HaploNet은 VAE, Neural ADMIXTURE는 supervised) — SSL+DP-GMM 새 조합

10번은 단독으로는 약함. 다른 옵션과 결합 권장.

---

## 3부. 옵션 비교 매트릭스 (v2 — 선행연구 반영)

| # | 옵션 | AI | Forensic | 차별성 v2 | 공수 | 재활용 | 위험 | 직접 경쟁작 |
|---|---|---|---|---|---|---|---|---|
| 1 | Conformal + Open-set | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 5-7w | 90% | 낮 | 없음 (forensic 0편) |
| 2 | Bayesian Deep | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 7-9w | 70% | 중 | DEGU(variant), 없음(forensic) |
| 3 | DNA FM fine-tune | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 7-10w | 20% | 중상 | **bioRxiv 2025-11 (zygosity-aware)** |
| 4 | SSL + multi-task | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 8-10w | 30% | 중 | 일반 SSL 있음, forensic 0편 |
| 5 | DP generative | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 8-10w | 40% | 중상 | **DP-SNP-TIHMM, SNPgen** |
| 6 | Federated | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 6-8w | 60% | 중 | **Frontiers 2024 (1000G fed)** |
| 7 | Mixture + ancestry | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 7-9w | 60% | 중 | MH-NOC ML, MixDeR, MPSproto |
| 8 | Degraded DNA bench | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 6-8w | 70% | 낮 | 일반 ADO 모델 있음 |
| 9 | Adaptive seq RL | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 9-11w | 50% | 상 | scRNA gene panel RL, nanopore adaptive |
| 10 | Substructure discovery | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 6-8w | 70% | 낮 | **UMAP, HaploNet, Neural ADMIX** |

---

## 4부. 상위 추천 (v2 — paper-grade contribution 기준)

### 🥇 옵션 1 (Conformal + Open-set) — 가장 안전한 강수

**왜**: Zhang (2025) 103-ref 리뷰가 명시적으로 CP/OSR 미언급을 증명 → 학회 reviewer의 "이미 한 사람 없냐" 공격 차단. 공수 작고 재활용 90%, 학기 final로 안전. Conformal 자체는 활발한 분야이므로 "forensic 첫 적용"은 깨끗한 contribution.

**예상 paper title**: *"Distribution-free uncertainty quantification and open-set recognition for forensic biogeographic ancestry inference"*

### 🥈 옵션 4 (SSL + multi-task forensic FM) — 가장 큰 potential

**왜**: "Forensic foundation model"은 미정착 framing. SSL+forensic multi-task가 0편이라 first-mover 가능. PhD thesis chapter seed. 단 공수 큼.

**예상 paper title**: *"Forensic-FM: A self-supervised foundation model for joint ancestry, kinship, and biological sex inference from microhaplotype profiles"*

### 🥉 옵션 7 (Mixture + per-contributor ancestry) — forensic 메시지 가장 강함

**왜**: MH+ML mixture는 활발하지만 per-contributor ancestry까지 가는 work은 부재. Forensic 실무 가치 가장 직접적.

**예상 paper title**: *"Per-contributor biogeographic ancestry inference from microhaplotype DNA mixtures with calibrated uncertainty"*

---

## 5부. 최적 조합 (v2 — 선행연구 누수 최소화)

| 조합 | 차별화 논리 | 강도 |
|---|---|---|
| **#1 + #4** | SSL pretrained forensic FM → conformal UQ wrapping. "Trustworthy forensic foundation model" | ⭐⭐⭐⭐⭐ |
| **#1 + #7** | Mixture deconvolution with **calibrated per-contributor uncertainty + reject option** when contributor outside reference | ⭐⭐⭐⭐⭐ |
| **#1 + #3** | DNA FM + conformal — zygosity-aware paper와 차별: forensic-specific UQ layer | ⭐⭐⭐⭐ |
| **#4 + #8** | SSL forensic FM + ADO robustness 평가 — single submission으로 multi-message | ⭐⭐⭐⭐ |
| **#5 + #8** | DP synthetic forensic benchmark with ADO robustness — community-shareable benchmark | ⭐⭐⭐⭐ |
| **#3 + #7** | DNA FM이 raw read에서 mixture 분리 — 가장 ambitious | ⭐⭐⭐⭐⭐ (high risk) |

---

## 6부. Two-sentence test (v2 — citation 포함)

| # | One-paragraph claim |
|---|---|
| 1 | "Forensic ancestry inference outputs uncalibrated single labels (Zhang 2025 review, 103 refs, no CP/OSR), making courtroom interpretation unreliable. We adapt **Mondrian conformal prediction** (Olsson 2025, first CP in genomic medicine, no forensic) and **energy-based OOD detection** to XGBoost on 1000G EAS MH, providing prediction sets with distribution-free coverage and reject option for unknown populations like KOR." |
| 4 | "Forensic genomic tasks (ancestry, kinship, sex) are solved by task-specific models on hand-designed panels (Zhang 2025; Yunnan DIP 2022). We pretrain a contrastive representation on 1000G WGS (analogous to GENEREL for medical) and fine-tune for multi-task forensic inference, which works because forensic tasks share population-genetic structure SSL captures." |
| 5 | "DP-SNP-TIHMM (2025-10) and SNPgen (2026-03) demonstrate DP synthetic genotypes for GWAS/disease, but not for forensic ancestry. We extend with **MH-specific, ancestry-conditional DP diffusion** and demonstrate that downstream forensic ML utility (LR, mixture) is preserved at ε≤10." |
| 7 | "MH mixture deconvolution exists (FSI:G 2024, 2025; MPSproto) but only solves NOC and per-contributor genotype. We extend to **per-contributor ancestry posterior with calibrated CI**, leveraging conformal prediction for uncertainty in identifiability-limited regime." |

---

## 7부. 의사결정 트리 (v2)

```
GPU >= 4xA100 이용 가능?
├── YES → 옵션 3, 4 검토 (foundation model 계열)
│         │
│         FM 직접 경쟁(zygosity-aware) 받아들이기 OK?
│         ├── YES → #3 (forensic framing으로 차별화)
│         └── NO  → #4 (multi-task framing이 더 안전)
│
└── NO (CPU/single GPU만)
    │
    학기 시간 5-7주만?
    ├── YES → #1 (Conformal + Open-set) — 안전한 강수
    │
    └── NO (8주 이상)
         │
         Forensic 응용 가치 우선?
         ├── YES → #7 (Mixture + ancestry) 또는 #8 (ADO)
         └── NO  → #2 (Bayesian Deep) — AI methodology novelty
         
         Privacy 관심?
         └── #5 (DP) 또는 #6 (Federated) — 둘 다 차별화 vector 필요
```

---

## 8부. 다음 액션 후보

이 v2 문서를 팀과 공유 후:

- **A**: 한 옵션 선정 → 1-page research outline + week-by-week plan 작성
- **B**: 상위 3개 옵션 (1, 4, 7) 각각 1-pager → 비교 후 결정
- **C**: 옵션 1 학기 진행 + 옵션 4 후속 thesis chapter로 분리
- **D**: 직접 경쟁작 정독 (특히 zygosity-aware DNA LM 2025-11)이 우선 — 옵션 3 결정 보류
- **E**: 새 옵션 추가 탐색 (예: methylation age estimation, body fluid ID 등 (1)축 다른 방향)
