# scripts/ — index

Runnable experiment scripts, grouped by **stage** (subdirectory) and named by
**chronological dev order** (the number; matches the `docs/04` narrative). Each
reads the `forensic_mh` package + `data/`, writes JSON to `results/baseline/`.
GPU scripts run via **sparkq** (see project memory); the rest are CPU. Canonical
numbers live in `docs/04` §13/§20/§21/§22.1/§23/§24/§25.

> Run `bash scripts/data/01_…` and `scripts/data/06_…` first to fetch data; full
> reproduce list in `docs/04` §18.

## `data/` — download / extract
| script | purpose |
|---|---|
| `01_download_1000g.sh` | 1000G Phase 3 chr22 (early baseline) |
| `02_extract_eas_samples.sh` | EAS 504 sample list |
| `05_download_related_samples.sh` | related/trio samples (Reliable-Ae) |
| `06_download_genome_wide.sh` | genome-wide EAS + non-EAS OOD (robust, retry/atomic) |
| `22_extract_hgdp.py` | HGDP remote MH extract, hg38 (RQ7 preliminary) |
| `32_extract_all_1kg.py` | all 2,504 1000G, hg19 (SSL pool) |
| `33_extract_hgdp1kg.py` | gnomAD HGDP+1KG 4,091, hg38 harmonized (SSL pool + RQ7) |

## `baseline/` — Plan-1 baseline
| script | purpose | docs |
|---|---|---|
| `03_run_baseline.py` | leakage-free chr22 baseline | §1 |
| `04_wei2025_phasing.py` | EAS Ae / Reliable-Ae (Wei 2025) | §2 |

## `trust/` — conformal · open-set · calibration (RQ1/2/4/6)
| script | purpose | docs |
|---|---|---|
| `10_conformal_curve.py` | coverage vs set-size (Mondrian) | §3 (RQ2) |
| `11_openset_ood.py` | far-OOD (non-EAS) | §4 (RQ1) |
| `12_lopo_nearood.py` | near-OOD (leave-one-population-out) | §5 (RQ1) |
| `16_plan2_rf.py` | base-model swap → RandomForest | §11 (RQ1) |
| `20_plan2_logreg.py` | base-model swap → LogReg | §11 (RQ1) |
| `24_trust_rigor.py` | **10-seed base-model OSR (4σ)** | §20 (RQ1 ★) |
| `53_perpop_coverage.py` | **per-population (Mondrian) coverage, LogReg 10-seed** | §3.1 (RQ1/RQ2) |
| `17_calibration_uq.py` | ECE + deep ensembles / MC-dropout | §14 (RQ4) |
| `18_ado_robustness.py` | degraded-DNA (allele dropout) | §15 (RQ6) |
| `47_embed_conformal.py` | CREE (b): embedding + conformal coverage | §27.1 (CREE) |
| `49_cree_variance.py` | CREE (d): variance-as-nonconformity open-set | §27.1 (CREE ★) |

## `models/` — model zoo: encoding · trees · DL · SOTA (RQ3)
| script | purpose | docs |
|---|---|---|
| `13_pca_ablation.py` | PCA/SVD features (no help) | §8 |
| `14_fm_vs_xgboost.py` | FM vs XGBoost head-to-head | §10 |
| `15_model_zoo.py` | initial zoo (encoding-confounded) | §9 |
| `19_onehot_cv.py` | **one-hot LogReg 79.6%** (ceiling broken) | §13 (canonical) |
| `21_model_zoo_onehot.py` | all models, one-hot | §13 |
| `30_model_zoo_dl.py` | classical + MLP, unified protocol | §24.1 |
| `31_dl_architectures.py` | EmbMLP / CNN1D / SupAE / Transformer | §24.2 |
| `39_dl_resnet_cnn.py` | ResNet-tabular + deep residual CNN | §24.2 |
| `35_extended_zoo.py` | linear family + regularization sweep | §24.4 |
| `43_linear_family.py` | Ridge / PA / Perceptron / MNB / LDA | §24.4 |
| `36_native_cat_trees.py` | native-categorical HGBDT / CatBoost | §24.5 |
| `37_sota_dl.py` | FT-Transformer + TabNet (GPU) | §24.5 |
| `38_tabpfn_panel.py` | TabPFN@200 (cloud, tabpfn-client) | §24.5 |
| `50_smalln_robustness.py` | CREE (e): extreme small-n vs LogReg | §27.1 (CREE) |

## `panel/` — minimum panel (RQ5)
| script | purpose | docs |
|---|---|---|
| `25_min_panel_logreg.py` | MI top-N selection | §21 |
| `26_l1_panel_cv.py` | L1 sparse panel | §21 |
| `27_min_panel_strong.py` | multivariate model-based selection | §23.1 |
| `28_min_panel_trust.py` | forensic trust frontier (3-way split) | §23.2 |
| `29_min_panel_rfe.py` | RFE cross-validation | §23.3 |

## `ssl/` — self-supervised foundation model (Paper 2)
| script | purpose | docs |
|---|---|---|
| `34_ssl_retrain.py` | first SSL@2504 (confounded) | §25 |
| `40_ssl_ablation.py` | data-scale ablation, 1000G hg19 | §25.2 |
| `41_ssl_gnomad.py` | clean ablation, gnomAD hg38 (GPU) | §25.1 |
| `48_ssl_randeff.py` | CREE (c): SSL + random-effects encoder (negative) | §27.1 (CREE) |

## `rq7/` — external-cohort transfer (RQ7)
| script | purpose | docs |
|---|---|---|
| `23_hgdp_transfer.py` | cross-build preliminary (82.4%) | §22.2 |
| `42_rq7_incallset.py` | **in-callset final (87.3%)** | §22.1 |
| `52_cree_transfer.py` | CREE (f): cross-cohort transfer (var 0.999 vs MSP) | §27.1 (CREE ★) |
