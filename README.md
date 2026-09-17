# EmbedEd

Research notes and literature review for a proposed undergraduate project on
**contrastive learning for domain-specific embeddings** (fine-tune a small
pretrained encoder with a contrastive objective so its embedding space captures a
domain-specific notion of semantic similarity better than a general-purpose
embedding model).

## Contents

- [`distilled/negative-pair-research/`](distilled/negative-pair-research/) — a literature
  investigation into **how negatives are constructed**, which is the only real independent
  variable in the project. Nine-category taxonomy, 12 candidate strategies with a feasibility
  matrix, five shortlisted experiments, an honest novelty audit, and a recommended direction.
  Read this before you change `negatives.py`.

- [`distilled/adaptive-embedding/`](distilled/adaptive-embedding/) — a **Version 2 plan** for
  making `domembed` domain-flexible: given a new dataset, run cheap pilots with the same three
  negative strategies and let validation pick one, instead of asking the user to choose. ~450
  new lines over existing code. Explicitly future work, with an honest novelty audit —
  pilot-based selection is standard model selection, not a new algorithm.

- [`distilled/embedding-library/`](distilled/embedding-library/) — the **product layer**.
  A specification for `domembed`, a small Python package that wraps the trained domain model
  and exposes `encode` / `similarity` / `search` / `info`, plus a CLI and an optional
  Streamlit demo. Built around the research, not alongside it: ~430 lines, one 40-line export
  script as the only seam between the two halves.

- [`project/distilled/`](project/distilled/) — **start here.** The plan of record: a 4-week
  project with one real independent variable. AskUbuntu duplicate questions (~15K corpus),
  one model (`all-MiniLM-L6-v2`), one loss (`MultipleNegativesRankingLoss`), and **three
  negative-hardness conditions** (random / lexical / model-mined) — because "does
  fine-tuning help?" is a reproduction question, while "does negative hardness decide
  whether it helps?" is an experiment. Three metrics (Recall@10, MRR@10, nDCG@10), paired
  bootstrap on three contrasts, a manual false-negative measurement, 8 essential papers,
  six source files (~650 lines), 12 debugging checkpoints, and a scope document with
  Versions A/B/C and a stop condition.
- [`distilled-project/`](distilled-project/) — the **larger** specification this was distilled
  *from*: CQADupStack, five negative strategies, two encoders, cross-domain evaluation.
  Kept as Version C (8–12 weeks). Also contains the conceptual primer + maths, feasibility
  matrix, five candidate paths, experiment design, error analysis, a 16-paper reading order
  (~4.5 hours), and a "do not do this" list.
- Version A (the smallest complete version: in-batch negatives only, 2 metrics, 4 files) is
  described in [`project/distilled/SCOPE.md`](project/distilled/SCOPE.md) and preserved in git
  history at commit `633db02`.
  The literature distilled into one executable 1-month undergraduate project:
  conceptual primer + maths, feasibility matrix, five candidate paths, the full
  project specification, experiment design and scope, implementation
  architecture, a four-week plan, error analysis, a 16-paper reading order
  (~4.5 hours), and a "do not do this" list.
- [`research/literature-review-contrastive-domain-embeddings.md`](research/literature-review-contrastive-domain-embeddings.md)
  — the full survey behind it. A research-grade literature review covering:
  - executive summary and research landscape
  - a table of the 24 most relevant papers, plus supporting references
  - full per-paper extraction tables for the 12 papers that matter most
  - a 7-paper essential reading list organised by role
  - a dataset shortlist (bug reports, duplicate questions, code)
  - five candidate research questions with literature positions
  - a concrete experimental design, baselines, metrics and ablations
  - compute estimates, reproducibility checks and a risk register
  - critical appraisal of weak/misleading papers
  - novelty assessment and a week-by-week "if I had one month" plan

Every substantive claim in the review is linked to a primary source (ACL
Anthology, PMLR, NeurIPS, OpenReview/ICLR, ACM DL, IEEE, arXiv, or the authors'
code).
