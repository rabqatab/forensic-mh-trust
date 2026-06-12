# docs/ — reading guide

All documents are anchored by the **research questions**. Read in this order.

## Canonical (current)

| # | Document | Role |
|---|---|---|
| **05** | [`05_research_questions.md`](05_research_questions.md) | **Start here.** The 3-act paper spine (RQ-Ⅰ/Ⅱ/Ⅲ) over the seven sub-RQs (RQ1–7), proposal→paper evolution, RQ↔evidence map, explicit non-RQs (scope). |
| **04** | [`04_experiments_and_results.md`](04_experiments_and_results.md) | The evidence log — every experiment, tagged to its RQ. §13/§20/§22/§23 hold the canonical numbers; §3.1 per-population coverage; Appendix A is the encoding post-mortem (RQ3). |
| **06** | [`06_failures_and_revisions.md`](06_failures_and_revisions.md) | **Failure post-mortem** — retracted claims, abandoned/deferred approaches, engineering bugs (what failed, why, how fixed), each anchored to an RQ. |
| **07** | [`07_method_proposal.md`](07_method_proposal.md) | **CREE** (Conformal Random-Effects Embeddings) — Paper-2 method proposal + pilot results (d/b/f positive, c/e negative). |
| **08** | [`08_venue_strategy.md`](08_venue_strategy.md) | FSI:G venue-fit + re-anchor plan: competitor map, the 3 forensic-reviewer pre-empts (LR-calibration, per-population coverage, validation), submission checklist. |

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

| RQ (sub) | 3-act | Primary evidence (in `04`) |
|---|---|---|
| **RQ1** base-model governs open-set reliability & per-population coverage | RQ-Ⅰ | §20 (10-seed, 4σ), §3.1 (per-pop), §11 |
| RQ3 simplicity result (linear > full ladder); not F_ST-limited (*internal enabler*) | RQ-Ⅰ | §13, §24, Appendix A |
| RQ4 ECE ≠ open-set separability | RQ-Ⅰ | §14 |
| RQ2 conformal coverage at target + set/panel size | RQ-Ⅱ | §3, §21 |
| RQ5 **deployable minimum panel exists** (multivariate rescue, 10–15× smaller) | RQ-Ⅲ | §23 |
| RQ6 degraded-DNA (ADO) robustness | RQ-Ⅲ | §15 |
| RQ7 external-cohort (HGDP) transfer — **in-callset 87.3%** | RQ-Ⅲ | §22 |

*Not committed:* `paperwork/` (proposal, assignment, paper drafts) and `docs/superpowers/` (internal plans/specs) are intentionally excluded from version control.
