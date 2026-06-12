# forensic-mh-trust

> **Distribution-free conformal prediction & open-set recognition for trustworthy forensic ancestry inference from East-Asian microhaplotypes (1000 Genomes).**

**One-sentence contribution.** For fine-scale biogeographic ancestry in a *low-separability* regime — five genetically close East-Asian populations — a model-agnostic **trust layer** (Mondrian split-conformal prediction sets + an open-set reject rule) makes a modest-accuracy classifier courtroom-usable, and **open-set reliability is governed by the base model's calibration and encoding, not by its accuracy.**

`Python ≥3.11` · `microhapdb==0.12` (pinned) · `scikit-learn` / `xgboost` / `torch` · CPU-only for the headline results.

---

## Why this exists

Forensic DNA ancestry inference typically reports a **single most-likely population with no quantified reliability** — dangerous for courtroom admissibility, and especially acute for East-Asian populations (CHB/CHS/JPT/KHV/CDX) whose pairwise F_ST is low, so single-label accuracy is intrinsically limited. A 103-reference review of AI in forensic genetics (Zhang 2025) finds **conformal prediction and open-set recognition entirely unapplied** to forensic ancestry. This repo closes that gap and, in doing so, surfaces a finding with broader ML interest: *which* base model you wrap decides whether the trust layer works at all.

## Key results

Five-class East-Asian classification, genome-wide **3,042 microhaplotypes**, 504 individuals (1000 Genomes Phase 3). Trust metrics use a fixed non-EAS out-of-distribution (OOD) set; AUROC/reject are **10-seed mean ± std**.

| Base model | 5-fold accuracy | far-OOD AUROC (10-seed) | empty-set OOD reject |
|---|---|---|---|
| **Logistic Regression (one-hot)** | **79.6% ± 3.9** | **0.840 ± 0.016** | fires (0.123) |
| RandomForest | 59.7% | 0.757 ± 0.048 | rare (0.004) |
| XGBoost | 56.9% | 0.675 ± 0.038 | **never (0.000)** |
| SSL transformer (FM) | 26.3% | — | — |

- **Open-set reliability is a base-model property** — swapping *only* the estimator moves far-OOD AUROC by **0.165 ≈ 4σ**; the leaderboard model (XGBoost) is the *worst* at abstaining.
- **The conformal guarantee is base-agnostic, but per-population coverage is not** — marginal coverage holds (≥0.90 at α=0.10) for every base; prediction sets average **~1.7 of 5 labels** at 90% with the linear base. Reporting *per-population* (Mondrian) coverage, the linear base holds the target for every population (KHV 0.90), whereas XGBoost — at a comparable marginal 0.888 — silently under-covers KHV at **0.70**: court-relevant subgroup reliability is itself a base-model property.
- **Fine-scale East-Asian separation is not F_ST-limited; a regularized linear model beats the whole complexity ladder** — one-hot diplotypes (no scaler) lift a linear model to **79.6%**, ahead of GBDTs, five DL families, FT-Transformer/TabNet and TabPFN — a *simplicity result* (Grinsztajn et al. 2022). The long-stuck "~57%" was an ordinal-encoding artifact, documented in [`docs/04` Appendix A](docs/04_experiments_and_results.md).
- **A deployable minimum panel exists** — univariate (MI) selection found none, but **multivariate selection front-loads the signal**: 25 markers reach 52% (8× more marker-efficient than MI) and 1,000 reach 76.8% (96% of the full-panel 79.6%) — confirmed by two independent selectors (one-shot coefficient ranking and recursive feature elimination, agreeing within ~2 points). Crucially, **conformal coverage holds (≥0.93) at every panel size** — even a 25-marker panel gives valid prediction sets; markers trade off only set-size and OOD-separation. A 200–300 marker panel (10–15× reduction) meets a practical forensic trust spec; fixed deployable panels (top-50/100/200 markers) are emitted as a deliverable.
- **Degraded-DNA limit** — under simulated allele dropout the conformal guarantee degrades measurably (coverage 0.91 → 0.80 at 50% ADO): exchangeability is violated, a forensic-realism limit competitors do not expose.
- **External-cohort transfer** — trained on 1000 Genomes EAS, the model transfers to an independent cohort (HGDP) at **87.3%** on three overlapping populations, validated *in-callset* on the harmonized gnomAD HGDP+1KG GRCh38 release (full 3,042 markers, build-mismatch eliminated: unseen diplotypes 43% → 2.4%). Transfer accuracy exceeds within-HGDP cross-validation, a strong generalization signal.

## Research questions

The project is organized around the research questions in **[`docs/05_research_questions.md`](docs/05_research_questions.md)** — start there.

The paper is organized as a **3-act spine** (Finding → Method → Deployment); the original seven RQs are preserved as the constituent sub-questions.

| Paper RQ (3-act) | Sub-questions | Status |
|---|---|---|
| **★ RQ-Ⅰ (Finding)** — Is court-defensible reliability (out-of-reference rejection + per-population coverage) obtained by picking the *most accurate* classifier, or governed by base-model calibration? | RQ1 (the accuracy-leader is the *least* court-reliable — base-model governs OSR & per-population coverage) · RQ3 (encoding / simplicity result — *internal enabler*, not a standalone claim) · RQ4 (ECE ≠ open-set separability) | ✅ (4σ) |
| **RQ-Ⅱ (Method)** — Does a distribution-free conformal trust layer give valid coverage at low accuracy? | RQ2 (target coverage + usable set/panel size) | ✅ |
| **RQ-Ⅲ (Deployment)** — Does the pipeline survive forensic constraints? | RQ5 (deployable minimum panel, 10–15× smaller) · RQ6 (degraded-DNA / ADO) · RQ7 (external-cohort HGDP transfer, in-callset 87.3%) | ✅ |

## Repository structure

```
src/forensic_mh/
  data/        diplotype extraction, encoding, marker (MicroHapDB) helpers
  uq/          conformal prediction, open-set, model-agnostic trust wrapper
  metrics/     Ae / Reliable-Ae, Mendelian consistency
  pipelines/   genome-wide matrix builder, leakage-free nested CV
  fm/          self-supervised microhaplotype transformer (Paper 2 / deferred)
  eval/        GroupKFold + nested-CV evaluation
scripts/       runnable experiments, grouped by stage (see scripts/README.md)
  data/        download / extract (1000G, HGDP, gnomAD HGDP+1KG)
  baseline/    Plan-1 baseline + Ae
  trust/       conformal, open-set, calibration, ADO (RQ1/2/4/6)
  models/      model zoo: encoding, trees, DL, tabular-SOTA, linear family (RQ3)
  panel/       minimum-panel selection (RQ5)
  ssl/         self-supervised foundation-model ablations (Paper 2)
  rq7/         external-cohort transfer (RQ7)
docs/          research questions, evidence log, background (see docs/README.md)
tests/         pytest suite
```

## Reproduce

```bash
uv pip install -e ".[dev]" && uv run python -m pytest -q   # test suite
bash scripts/data/01_download_1000g.sh && bash scripts/data/06_download_genome_wide.sh

# Headline experiments (canonical numbers)
uv run python scripts/models/19_onehot_cv.py        # RQ3 — one-hot LogReg 79.6% (simplicity result)
uv run python scripts/trust/24_trust_rigor.py       # RQ1 — base-model governs OSR (10-seed)
uv run python scripts/trust/53_perpop_coverage.py   # RQ1 — per-population coverage (LogReg holds, XGBoost masks KHV)
uv run python scripts/panel/27_min_panel_strong.py  # RQ5 — multivariate selection rescues the minimal panel
uv run python scripts/trust/17_calibration_uq.py    # RQ4 — ECE vs OSR
uv run python scripts/trust/18_ado_robustness.py    # RQ6 — degraded-DNA
uv run python scripts/rq7/42_rq7_incallset.py       # RQ7 — external-cohort transfer (87.3%)
```

Full script→experiment→RQ map is in [`docs/04`](docs/04_experiments_and_results.md) §18.

## Data & reproducibility

- **Genotypes**: 1000 Genomes Project Phase 3 VCF (GRCh37/hg19, release 20130502, 2,504 individuals). EAS 504 in-distribution + 300 non-EAS as OOD.
- **Markers**: MicroHapDB **v0.12** (pinned), genome-wide autosomal, 3,042 markers (hg19). The proposal's "412" is an older version of the same DB.
- Raw data and result JSONs are **gitignored** (large); scripts regenerate them deterministically (seeds fixed).

## Scope

This repository is the **Trustworthy-UQ** contribution. Explicitly out of scope here:
- **Self-supervised foundation model** (`src/forensic_mh/fm/`) — implemented and unit-tested, but underperforms in the small-data regime; deferred to a follow-up with data expansion.
- **Korean / unrepresented-population rejection** — no Korean data; non-EAS + leave-one-population-out are honest proxies.
- **Reliable-Ae phasing penalty** — requires the NYGC 30× release (trios); deferred.

## Citation & license

Manuscript in preparation. Citation and license **TBD** before public release. Method references are tracked in [`docs/05`](docs/05_research_questions.md) §5 and verified per the project's no-fabricated-citations policy.
