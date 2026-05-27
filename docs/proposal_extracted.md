# 본문 (Document body)

- 동아시아 집단 분류를 위한
- 최소 마이크로하플로타입(MH) 패널 연구
- — 1000 Genomes VCF 기반 완성본 연구 개요서 —

## 1. 연구 질문

- 포렌식(법과학) 현장에서는 범죄 현장에 남겨진 DNA로 '이 사람이 어느 집단 출신인가'를 추정합니다. 예를 들어 DNA만 가지고 '이 사람이 중국인인가, 일본인인가, 베트남인인가'를 구별하는 것입니다.
- 현재 이 목적에 사용하는 DNA 마커가 수백 개 있습니다. 그런데 현장에서는 마커가 적을수록 분석이 빠르고 비용이 낮아집니다. 이 연구의 핵심 질문은 딱 하나입니다.
- "몇 개의 마이크로하플로타입(MH) 마커만 있으면 동아시아 5개 집단을 충분한 정확도로 구별할 수 있는가?"
- 이 질문에 머신러닝(XGBoost)으로 답하는 것이 이 연구의 전부입니다.

## 2. 핵심 개념 3가지


### 2-1. 마이크로하플로타입(MH)이란?

- MH를 이해하려면 먼저 SNP(스닙)이라는 개념부터 알아야 합니다.
- SNP(Single Nucleotide Polymorphism): DNA의 특정 위치에서 사람마다 다른 글자(A/T/G/C)를 가지는 것. 예를 들어 어떤 위치에서 어떤 사람은 A, 다른 사람은 G를 가집니다.
- SNP 하나만으로는 정보량이 적습니다. A 또는 G, 두 가지 경우밖에 없기 때문입니다. 그런데 200글자(200bp) 이내의 짧은 구간 안에 SNP 2~5개가 모이면 이야기가 달라집니다.
- 그림 1. SNP 하나 vs 마이크로하플로타입(MH) — 정보량 비교
- 위 그림처럼, SNP 3개가 묶이면 최대 8가지 조합이 생깁니다. 이 짧은 구간의 SNP 조합을 마이크로하플로타입(MH)이라고 합니다.
- MH는 STR(현재 포렌식 표준 마커)보다 짧은 DNA 조각에서도 읽을 수 있어서, 분해된 시료(오래된 뼈, 모발 등)에서도 분석이 가능합니다.
- [List Paragraph] 마이크로하플로타입은 하플로타입과 개념이 완전히 같고, "한 번의 NGS read 안에 들어올 만큼 짧은 하플로타입" 이라는 뜻으로 "마이크로"가 붙은 것입니다. 새로운 개념이 아니라 크기 제한이 추가된 하플로타입입니다.

### 2-2. Ae (유효 대립유전자 수)란?

- Ae는 '이 MH 마커가 실제로 얼마나 다양한 조합을 가지는가'를 숫자 하나로 나타낸 것입니다.

### 2-3. FST (집단 간 차이 지수)란?

- FST는 두 집단이 이 마커에서 얼마나 다른지를 0~1 사이 숫자로 나타냅니다. 0이면 두 집단이 같고, 1이면 완전히 다릅니다.
- 비유: FST는 '한국 음식 메뉴와 일본 음식 메뉴가 얼마나 다른가'와 같습니다. 완전히 다른 메뉴라면 FST = 1, 완전히 같다면 FST = 0.
- 이 연구에서는 Ae와 FST 모두 높은 MH를 선별하여 집단 분류에 사용합니다.

## 3. 데이터: 1000 Genomes Project VCF


### 3-1. VCF 파일이란?

- VCF(Variant Call Format)는 전장유전체 시퀀싱 결과를 저장하는 파일 형식입니다. 각 사람의 DNA 위치별 유전자형(0, 1, 2)이 담겨 있습니다.
- 이 VCF 파일에서 MH 구간의 genotype을 추출하면 개인별 MH 조합을 얻을 수 있습니다.

### 3-2. 사용하는 집단 (EAS 5개)

- 한국인 집단(KOR)은 1000 Genomes에 없습니다. 이것은 이 연구의 명확한 한계이며 논문의 Limitation 섹션에 반드시 명시해야 합니다.

### 3-3. 다운로드 방법

- 1단계: Panel 파일을 먼저 받습니다 (작고 빠름).
- # Panel 파일: 어떤 샘플이 어느 집단인지 알려주는 지도
- wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel
- 2단계: EAS 집단 VCF를 염색체별로 받습니다 (큰 파일들).
- # 염색체 1번 예시 (약 1.5GB)
- wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz
- wget ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz.tbi
- 전략: 처음에는 염색체 1개(chr1)만 받아서 파이프라인 테스트 → 잘 작동하면 나머지 다운로드. 전체 22개 합계 약 20GB.

## 4. 전체 분석 파이프라인

- 그림 2. 전체 분석 파이프라인 (1000G VCF → MH 추출 → ML)

### 4-1. Step 1: EAS 샘플 추출 (bash)

- bcftools를 사용하여 504명의 EAS 샘플만 추출합니다.
- # bcftools 설치 (Ubuntu/Linux)
- sudo apt-get install bcftools
- # Panel 파일에서 EAS 샘플 ID 목록 추출
- grep -E 'CHB|CHS|JPT|KHV|CDX' integrated_call_samples_v3.20130502.ALL.panel \
- | cut -f1 > EAS_samples.txt
- # 확인: 504명인지 체크
- wc -l EAS_samples.txt
- # VCF에서 EAS 샘플만 추출 (chr1 예시)
- bcftools view -S EAS_samples.txt \
- ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz \
- -Oz -o EAS_chr1.vcf.gz
- # 인덱스 생성 (필수)
- bcftools index EAS_chr1.vcf.gz

### 4-2. Step 2: MH 후보 loci 정의 (Python)

- MH는 300bp 이내에 2개 이상의 SNP가 있는 구간입니다. MicroHapDB의 마커 위치 정보를 기준 좌표로 사용합니다.
- import pandas as pd
- import microhapdb  # pip install microhapdb
- # MicroHapDB에서 MH 마커의 게놈 좌표 가져오기
- # (빈도 데이터가 아니라 '위치 정보'를 사용)
- markers = microhapdb.markers  # MH 마커별 SNP 위치 목록
- print(markers.head())
- # 각 MH의 염색체 위치 확인
- # marker_id, chrom, positions(list) 형태
- # 예: mh01KK-117 / chr1 / [100234, 100298, 100341]

### 4-3. Step 3: 개인별 MH genotype 추출 (Python + pysam)

- 각 개인의 MH 위치에서 SNP 값을 읽어 조합합니다.
- import pysam  # pip install pysam
- import numpy as np
- def extract_mh_genotype(vcf_file, chrom, positions, sample_ids):
- """
- VCF에서 특정 MH 위치의 개인별 genotype 추출
- positions: MH를 구성하는 SNP 위치 목록 [pos1, pos2, ...]
- 반환: {sample_id: 'A-C-T'} 형태의 딕셔너리
- """
- vcf = pysam.VariantFile(vcf_file)
- genotypes = {s: [] for s in sample_ids}
- for pos in positions:
- for rec in vcf.fetch(chrom, pos-1, pos):
- for sample in sample_ids:
- gt = rec.samples[sample]['GT']  # (0,1) 형태
- allele = rec.alleles[gt[0]] if gt[0] is not None else 'N'
- genotypes[sample].append(allele)
- # 조합으로 변환: ['A','C','T'] → 'A-C-T'
- return {s: '-'.join(v) for s, v in genotypes.items()}
- # 전체 MH 마커에 대해 적용
- results = []
- for mh_id, mh_info in markers.iterrows():
- chrom = mh_info['chrom']
- positions = mh_info['positions']  # SNP 위치 목록
- gt_dict = extract_mh_genotype(
- f'EAS_{chrom}.vcf.gz', chrom, positions, sample_ids
- )
- for sample, gt in gt_dict.items():
- results.append({'sample': sample, 'marker': mh_id, 'genotype': gt})
- df_raw = pd.DataFrame(results)

### 4-4. Step 4: ML 입력 행렬 만들기

- genotype 문자열을 숫자로 인코딩하여 ML 입력 행렬을 만듭니다.
- from sklearn.preprocessing import LabelEncoder
- # 피벗: 행=샘플(504명), 열=MH마커
- df_pivot = df_raw.pivot(index='sample', columns='marker', values='genotype')
- # genotype 문자열을 숫자로 인코딩
- # 'A-C-T'=0, 'A-G-T'=1, 'T-C-A'=2 ...
- X = pd.DataFrame()
- for col in df_pivot.columns:
- le = LabelEncoder()
- X[col] = le.fit_transform(df_pivot[col].fillna('unknown'))
- # 레이블(집단) 추가
- # panel 파일에서 샘플-집단 매핑 읽기
- panel = pd.read_csv('integrated_call_samples_v3.20130502.ALL.panel', sep='\t')
- sample_pop = panel.set_index('sample')['pop'].to_dict()
- y = [sample_pop.get(s, 'unknown') for s in X.index]
- print(f'행렬 크기: {X.shape}')  # (504, N_markers)
- print(f'집단 분포: {pd.Series(y).value_counts()}')
- 4-5. Step 5: Ae/FST 1차 필터링 — 투 트랙 전략 (중요)
- XGBoost만 바로 돌리는 방식이 아님
- 412개 MH 전부를 필터 없이 XGBoost에 넣으면 Ae가 낮은 쓸모없는 마커도 포함됩니다. 이런 노이즈 마커가 많을수록 모델이 불안정해지고, Feature Importance 결과도 신뢰성이 낮아집니다. Ae/FST로 1차 필터링하면 후보 마커의 질이 높아져 XGBoost 결과가 더 안정적입니다.
- 투 트랙 전략 흐름:
- 1차 필터 (유전학적 기준): Ae >= 3 AND FST >= 0.05 인 마커만 통과
- ↓ (412개 → 수백 개로 감소)
- 2차 선별 (ML 기준): XGBoost Feature Importance 상위 N개 최종 선택
- ↓
- 최종 패널: 고품질 마커 N개 (신뢰성 높음)
- 주의: Ae/FST와 XGBoost Feature Importance는 항상 같은 마커를 고르지 않습니다.
- Ae가 높아도 FST가 낮으면 (5집단 모두에서 균등하게 다양) XGBoost에서 Feature Importance가 낮게 나올 수 있습니다. 반대로 Ae가 보통이어도 특정 집단에서만 특이한 패턴을 보이면 (FST 높음) XGBoost가 이 마커를 핵심으로 선택합니다. 두 기준을 조합해야 더 신뢰성 있는 패널이 만들어집니다.
- 1차 필터링 코드 (XGBoost 실행 전에 먼저 실행):
- import microhapdb
- # MicroHapDB에서 EAS 집단의 Ae와 FST 가져오기
- freq_df = microhapdb.frequencies
- eas_pops = ['CHB', 'CHS', 'JPT', 'KHV', 'CDX']
- # 마커별 Ae 평균 계산 (EAS 5집단 기준)
- ae_df = microhapdb.ae(population=eas_pops)
- ae_mean = ae_df.groupby('Marker')['Ae'].mean()
- # FST 계산 (집단 간 차이)
- fst_df = microhapdb.fst(population=eas_pops)
- # 1차 필터: Ae >= 3 AND FST >= 0.05
- good_markers = ae_mean[(ae_mean >= 3)].index
- good_markers = good_markers.intersection(
- fst_df[fst_df['Fst'] >= 0.05]['Marker'])
- print(f'1차 필터 후 남은 마커 수: {len(good_markers)}개')
- # 이후 X 행렬을 good_markers만 포함하도록 필터링
- X_filtered = X[X.columns.intersection(good_markers)]
- 이후 Step 6의 XGBoost 학습에는 X 대신 X_filtered를 사용합니다. 이렇게 하면 Ae/FST 기준으로 검증된 마커만 ML에 입력되어 결과의 신뢰성이 높아집니다.

### 4-5. Step 5: XGBoost 학습 및 최적 패널 탐색 (핵심)

- from xgboost import XGBClassifier
- from sklearn.preprocessing import LabelEncoder
- from sklearn.model_selection import StratifiedKFold, cross_val_score
- import matplotlib.pyplot as plt
- # 집단 레이블 숫자 변환
- le_y = LabelEncoder()
- y_enc = le_y.fit_transform(y)
- # 교차검증 설정
- cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- # 전체 마커로 먼저 학습 → Feature Importance 추출
- model_full = XGBClassifier(n_estimators=200, max_depth=4,
- learning_rate=0.1, random_state=42,
- eval_metric='mlogloss')
- model_full.fit(X.values, y_enc)
- importances = model_full.feature_importances_
- sorted_idx = np.argsort(importances)[::-1]  # 중요도 내림차순
- # 마커 수를 줄여가며 정확도 측정
- n_list = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, X.shape[1]]
- acc_results = {}
- for n in n_list:
- X_sub = X.values[:, sorted_idx[:n]]
- scores = cross_val_score(
- XGBClassifier(n_estimators=200, max_depth=4,
- learning_rate=0.1, random_state=42,
- eval_metric='mlogloss'),
- X_sub, y_enc, cv=cv, scoring='accuracy'
- )
- acc_results[n] = (scores.mean(), scores.std())
- print(f'MH {n:3d}개: 정확도 {scores.mean():.3f} +/- {scores.std():.3f}')
- # 결과 그래프
- ns = list(acc_results.keys())
- means = [acc_results[n][0] for n in ns]
- stds  = [acc_results[n][1] for n in ns]
- plt.figure(figsize=(10, 6))
- plt.plot(ns, means, 'o-', color='#4472C4', lw=2.5, ms=8)
- plt.fill_between(ns,
- [m-s for m,s in zip(means,stds)],
- [m+s for m,s in zip(means,stds)],
- alpha=0.15, color='#4472C4')
- plt.axhline(y=0.90, color='red', ls='--', alpha=0.7, label='90% 목표')
- plt.xlabel('사용한 MH 마커 개수', fontsize=12)
- plt.ylabel('5집단 분류 정확도', fontsize=12)
- plt.title('MH 마커 수에 따른 분류 정확도 변화', fontsize=14)
- plt.legend(fontsize=11); plt.grid(True, alpha=0.3)
- plt.tight_layout()
- plt.savefig('accuracy_curve.png', dpi=150)
- 그림 3. MH 마커 수에 따른 분류 정확도 변화 (개념적 예시 — 실제 값은 분석 결과로 결정)

### 4-6. Step 6: 모델 비교 분석

- from sklearn.ensemble import RandomForestClassifier
- from sklearn.svm import SVC
- from sklearn.metrics import classification_report
- # 최적 N 찾기 (예: 정확도가 처음으로 90%를 넘는 지점)
- optimal_n = min([n for n,v in acc_results.items() if v[0] >= 0.90])
- X_opt = X.values[:, sorted_idx[:optimal_n]]
- models = {
- 'XGBoost': XGBClassifier(n_estimators=200, max_depth=4,
- learning_rate=0.1, random_state=42,
- eval_metric='mlogloss'),
- 'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
- 'SVM (RBF)': SVC(kernel='rbf', C=10, probability=True, random_state=42),
- }
- print(f'최적 마커 수: {optimal_n}개')
- print('-' * 50)
- for name, clf in models.items():
- sc = cross_val_score(clf, X_opt, y_enc, cv=cv, scoring='accuracy')
- print(f'{name:15s}: {sc.mean():.3f} +/- {sc.std():.3f}')

## 5. 최종 결과물: 웹 기반 집단 분류 도구

- 5-1. 개요 및 목적
- 본 연구의 최종 결과물은 학습된 XGBoost 분류 모델을 웹 인터페이스로 제공하는 도구입니다. 연구자가 MH genotype 데이터를 업로드하면 동아시아 5개 집단(CHB, CHS, JPT, KHV, CDX) 중 어느 집단에 속하는지 확률값으로 즉시 출력합니다. GitHub에 오픈소스로 공개하여 논문의 추가 Contribution으로 활용합니다.
- 5-2. 데이터 흐름
- 사용자는 전장유전체 원본 파일이 아닌, 사전에 선별된 MH 위치의 genotype 값만 추출한 파일(VCF 또는 CSV)을 입력합니다. 도구 내부에서 Ae/FST 1차 필터링을 통과한 최적 마커만 자동으로 선택한 후 저장된 XGBoost 모델에 입력하여 집단 분류 확률을 산출합니다.
- ① 샘플 VCF/CSV 업로드  →  ② MH genotype 추출 (412개 위치)  →  ③ Ae/FST 필터 적용  →  ④ XGBoost 예측  →  ⑤ 집단 분류 결과 출력
- 5-3. 웹 인터페이스 예시
- 아래 그림은 Flask(Python) 기반 웹 애플리케이션의 인터페이스 예시입니다. 왼쪽에서 파일을 업로드하거나 genotype을 직접 입력하고 분석을 실행하면, 오른쪽에 5개 집단별 분류 확률이 막대 그래프와 함께 출력됩니다.
- 그림 5. EAS MH Population Classifier — 웹 인터페이스 분석 결과 화면 예시 (JPT 87.3% 판정)
- 5-4. 구현 방법 (Flask 기반)
- Python Flask 프레임워크로 구현하며, 핵심 코드는 아래와 같습니다. 학습된 모델(joblib)을 서버에 로드하고, 사용자가 업로드한 파일에서 MH genotype을 추출하여 예측합니다.
- from flask import Flask, request, jsonify, render_template
- import joblib, pandas as pd
- app = Flask(__name__)
- model = joblib.load('mh_eas_classifier.pkl')  # 학습된 모델 로드
- @app.route('/predict', methods=['POST'])
- def predict():
- file = request.files['genotype_file']
- X = extract_and_encode_mh(file)   # MH 추출 + 인코딩
- proba = model.predict_proba(X)[0]  # 확률 예측
- pops = ['CHB', 'CHS', 'JPT', 'KHV', 'CDX']
- return jsonify({p: float(prob) for p, prob in zip(pops, proba)})
- 5-5. 한계 및 주의사항
- 분류 가능 집단: CHB, CHS, JPT, KHV, CDX 5개뿐 — 한국인(KOR) 집단 미포함
- 입력 형식: 전장유전체 원본(WGS) 직접 입력 불가 — MH 위치 genotype 사전 추출 필요
- 법적 사용: 학술 연구 목적 전용, 법정 증거로 사용 불가 (별도 공식 검증 절차 필요)
- 배포: GitHub 오픈소스 공개 — 논문 투고 시 “we provide a classification tool at GitHub” 로 기술

## 6. 구성 및 용어 정리


### 6-1. 논문 섹션 구성


## 7. 핵심 용어 정리


### [Table 1]

| Ae 값 | 의미 | 사용 가능성 |
| 1 | 모든 사람이 같은 조합 → 구별 불가 | 쓸모없음 |
| 2 ~ 3 | 2~3가지 조합만 존재 | 최소 수준 |
| 5 ~ 6 | 다양한 조합, 균등 분포 | 좋음 (권장) |
| 8 이상 | 매우 다양하게 분포 | 최상급 |

### [Table 2]

| 값 | 의미 | 예시 |
| 0 | 기준 염기 두 개 (ref/ref) | A-A |
| 1 | 기준+변이 염기 (ref/alt) | A-G |
| 2 | 변이 염기 두 개 (alt/alt) | G-G |

### [Table 3]

| 집단 코드 | 이름 | 샘플 수 | 지역 |
| CHB | Han Chinese in Beijing | 103명 | 중국 베이징 |
| CHS | Southern Han Chinese | 105명 | 중국 남부 |
| JPT | Japanese in Tokyo | 104명 | 일본 도쿄 |
| KHV | Kinh in Ho Chi Minh City | 99명 | 베트남 |
| CDX | Chinese Dai in Xishuangbanna | 93명 | 중국 |
| 합계 |  | 504명 | 동아시아 전체 |

### [Table 4]

| 섹션 | 핵심 내용 |
| Introduction | MH의 포렌식 가치, EAS fine-scale 분류의 필요성, 기존 연구 한계 |
| Materials & Methods | 1000G EAS 데이터, MH 추출 파이프라인, XGBoost + 교차검증 |
| Results | 정확도 vs 마커 수 곡선, 최적 N값, 모델 비교, Feature Importance 상위 마커 |
| Discussion | 포렌식 실용성, 비용 절감 효과, 다른 집단 적용 가능성 |
| Limitations | 한국인 집단 부재, in silico 한계, 실험 검증 미실시 (솔직 명시) |
| Conclusion | EAS 집단 분류에 N개 MH면 충분, 실무 적용 가이드 |

### [Table 5]

| 용어 | 원어 | 한 줄 설명 |
| SNP | Single Nucleotide Polymorphism | DNA 한 위치에서 사람마다 다른 글자. '스닙' |
| MH | Microhaplotype | 200bp 이내 SNP 2개 이상 묶음. 정보량 높음 |
| VCF | Variant Call Format | WGS 결과 파일. 개인별 SNP 유전형 기록 |
| Genotype | 유전형 | 특정 위치에서 개인이 가진 DNA 조합 (0/1/2) |
| Ae | Effective Number of Alleles | 마커 다양성 지수. 높을수록 개인 구별 강함 |
| FST | Fixation Index | 집단 간 차이 지수. 높을수록 집단이 다름 |
| EAS | East Asian | 동아시아. CHB, CHS, JPT, KHV, CDX 5집단 |
| Feature Importance | - | ML에서 각 마커가 예측에 기여하는 중요도 점수 |
| 교차검증 (CV) | Cross Validation | 데이터를 나눠 모델 성능을 안정적으로 평가 |
| in silico | - | 컴퓨터 상의 시뮬레이션. 실제 실험 없이 데이터만 사용 |


# 코멘트 (Comments) — total 7


### Comment #0 — 임수연 (2026-05-25T17:24:00Z)
Wei Y, Li X, Zhu Q et al. (2025). Are microhaplotypes derived from the 1000 Genomes Project reliable for forensic purposes?  Forensic Science International: Genetics . DOI: 10.1016/j.fsigen.2025.103273. https://www.fsigenetics.com/article/S1872-4973(25)00053-5/abstract


### Comment #1 — 임수연 (2026-05-25T17:23:00Z)
Kidd KK & Speed WC (2015). Criteria for selecting microhaplotypes: mixture detection and deconvolution.  Investigative Genetics , 6:1.  https://link.springer.com/article/10.1186/s13323-014-0018-3


### Comment #2 — 임수연 (2026-05-25T17:23:00Z)
Oldoni F, Kidd KK, Podini D (2019). Microhaplotypes in forensic genetics.  Forensic Science International: Genetics , 38:54-69.  https://www.sciencedirect.com/science/article/abs/pii/S1872497318303910


### Comment #3 — 임수연 (2026-05-25T17:24:00Z)
Podini D et al. (2026). Defining Key Criteria for Microhaplotype Locus Selection in Forensic Genetics.  Forensic Science International: Genetics .  https://www.fsigenetics.com/article/S1872-4973(26)00002-5/abstract


### Comment #4 — 임수연 (2026-05-25T17:31:00Z)
클로드가   알려준   데이터   다운로드   방법을   여기다가   적어놓습니다 ... https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/   >  데이터   다운은   이곳에서 !  integrated_call_samples_v3.20130502.ALL.panel   > 패널부터   다운로드 파일명 : ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz  크기 :  약  1.5GB  용도 :  파이프라인   테스트용   같이   받아야   할   인덱스   파일 : ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz.tbi  크기 :  약  1MB  # VCF  파일 wget  https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz #  인덱스   파일  ( 세트로   반드시   함께   받아야   함 ) wget  https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz.tbi Step 1: Panel  파일   받기  ( 지금   당장 , 1 분 )      ↓ Step 2: chr1 VCF  받기  (1.5GB,  수십   분 )      ↓ Step 3: chr1 로   파이프라인   테스트      ↓ Step 4:  잘   되면   나머지  chr2~22  받기     ( 전체   약  15~20GB,  인터넷   속도에   따라   수   시간 )


### Comment #5 — 임수연 (2026-05-25T17:22:00Z)
Standage DS & Mitchell RN (2020). MicroHapDB: A Portable and Extensible Database of All Published Microhaplotype Marker and Frequency Data.  Frontiers in Genetics , 11:781.  doi: 10.3389/fgene.2020.00781. https://pmc.ncbi.nlm.nih.gov/articles/PMC7427474/


### Comment #6 — 임수연 (2026-05-25T17:25:00Z)
MicroHapDB  실제   용도   mh01KK-117  →   염색체 : chr1  →   위치 : 100,234 번  / 100,298 번  / 100,341 번 이   좌표를   가지고  1000G VCF 에서   " 이   위치들의   개인별  genotype 을   꺼내줘 "  라고   하는   것
