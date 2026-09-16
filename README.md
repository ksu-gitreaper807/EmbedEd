# EmbedEd

Research notes and literature review for a proposed undergraduate project on
**contrastive learning for domain-specific embeddings** (fine-tune a small
pretrained encoder with a contrastive objective so its embedding space captures a
domain-specific notion of semantic similarity better than a general-purpose
embedding model).

## Contents

- [`distilled-project/`](distilled-project/) — **start here if you want to build the project.**
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
