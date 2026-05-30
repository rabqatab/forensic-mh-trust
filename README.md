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
- **The conformal guarantee is base-agnostic** — coverage holds (≥0.90 at α=0.10) for every base model; prediction sets average **~1.8 of 5 labels** at 90% coverage with the linear base.
- **The "57% ceiling" was an encoding artifact, not an F_ST limit** — nominal diplotypes one-hot-encoded (no scaler) lift a regularized linear model from ~57% to **79.6%** (see [`docs/04` Appendix A](docs/04_experiments_and_results.md)).
- **No compact "minimum panel"** — accuracy rises monotonically with markers; ≥70% requires the full genome-wide panel. The forensic message is *genome-wide MH + calibrated UQ*, not *minimal panel*.
- **Degraded-DNA limit** — under simulated allele dropout the conformal guarantee degrades measurably (coverage 0.91 → 0.80 at 50% ADO): exchangeability is violated, a forensic-realism limit competitors do not expose.
- **External-cohort transfer (preliminary)** — a 1000G-trained model transfers to an independent cohort (HGDP WGS, hg38) at **82.4%** on 3 overlapping populations with only 510/3042 markers and 43% build-mismatched diplotypes; full-panel result pending genome-wide extraction.

## Research questions

The project is organized around the research questions in **[`docs/05_research_questions.md`](docs/05_research_questions.md)** — start there.

| RQ | Claim | Status |
|---|---|---|
| **RQ1 (primary)** | Open-set reliability is governed by the base model, not accuracy | ✅ (4σ) |
| RQ2 | Conformal delivers target coverage despite modest accuracy | ✅ |
| RQ3 (enabling) | The "ceiling" is an encoding artifact; one-hot + linear wins | ✅ |
| RQ4 | Calibration (ECE) ≠ open-set separability (AUROC) | ✅ |
| RQ5 (scope) | No compact minimum panel; signal is genome-wide | ✅ |
| RQ6 | Conformal coverage degrades under degraded-DNA (ADO) | ✅ |
| RQ7 | The model + trust layer transfers to an external cohort (HGDP) | 🔶 preliminary (82.4%) |

## Repository structure

```
src/forensic_mh/
  data/        diplotype extraction, encoding, marker (MicroHapDB) helpers
  uq/          conformal prediction, open-set, model-agnostic trust wrapper
  metrics/     Ae / Reliable-Ae, Mendelian consistency
  pipelines/   genome-wide matrix builder, leakage-free nested CV
  fm/          self-supervised microhaplotype transformer (Paper 2 / deferred)
  eval/        GroupKFold + nested-CV evaluation
scripts/       numbered, runnable end-to-end (download → experiments)
docs/          research questions, evidence log, background (see docs/README.md)
tests/         pytest suite
```

## Reproduce

```bash
uv pip install -e ".[dev]" && uv run python -m pytest -q   # test suite
bash scripts/01_download_1000g.sh && bash scripts/06_download_genome_wide.sh

# Headline experiments (canonical numbers)
uv run python scripts/19_onehot_cv.py        # RQ3 — one-hot LogReg 79.6%
uv run python scripts/24_trust_rigor.py      # RQ1 — base-model governs OSR (10-seed)
uv run python scripts/25_min_panel_logreg.py # RQ5 — no compact panel
uv run python scripts/26_l1_panel_cv.py      # RQ5 — L1 sparse panel
uv run python scripts/17_calibration_uq.py   # RQ4 — ECE vs OSR
uv run python scripts/18_ado_robustness.py   # RQ6 — degraded-DNA
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
