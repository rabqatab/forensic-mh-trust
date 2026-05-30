# docs/ — reading guide

All documents are anchored by the **research questions**. Read in this order.

## Canonical (current)

| # | Document | Role |
|---|---|---|
| **05** | [`05_research_questions.md`](05_research_questions.md) | **Start here.** The confirmed RQs (RQ1–RQ7), proposal→paper evolution, RQ↔evidence map, explicit non-RQs (scope). |
| **04** | [`04_experiments_and_results.md`](04_experiments_and_results.md) | The evidence log — every experiment, tagged to its RQ. §13/§20/§21 hold the canonical numbers; Appendix A is the encoding post-mortem (RQ3). |

> **Canonical numbers live in `04` §13 (accuracy), §20 (trust rigor, 10-seed), §21 (minimum panel).** Earlier sections (§3–§5, §7, §9) are XGBoost/ordinal-era and are explicitly marked *superseded* — kept as chronological history, not as current results.

## Background (early exploration — context, not current claims)

| # | Document | Role |
|---|---|---|
| 01 | [`01_proposal_review.md`](01_proposal_review.md) | Critical review of the original course proposal. |
| 02 | [`02_literature_landscape.md`](02_literature_landscape.md) | Literature mapping; Zhang 2025 gap-evidence anchor. |
| 03 | [`03_novelty_options.md`](03_novelty_options.md) | Novelty option menu; option 1 (conformal + open-set) became the spine. |
| — | [`proposal_extracted.md`](proposal_extracted.md) | Original proposal text (extracted). |
| — | [`proposal_extension_v1.md`](proposal_extension_v1.md) | Early team extension draft. |

## RQ → where the evidence is

| RQ | Primary evidence (in `04`) |
|---|---|
| **RQ1** base-model governs open-set reliability | §20 (10-seed, 4σ), §11 |
| RQ2 conformal coverage at target | §3, §21 |
| RQ3 encoding artifact, one-hot+linear wins | §13, Appendix A, §8 (PCA ablation) |
| RQ4 ECE ≠ open-set separability | §14 |
| RQ5 no compact minimum panel | §21 (incl. L1) |
| RQ6 degraded-DNA (ADO) robustness | §15 |
| RQ7 external-cohort (HGDP) transfer | §4.6 *(pending)* |

*Not committed:* `paperwork/` (proposal, assignment, paper drafts) and `docs/superpowers/` (internal plans/specs) are intentionally excluded from version control.
