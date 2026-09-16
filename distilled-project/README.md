# Distilled Project — Contrastive Learning for Domain-Specific Embeddings

**Audience:** 2nd-year CSE undergraduate · **Budget:** ~1 month · **Compute:** one consumer/Colab/Kaggle GPU
**Companion document:** [`research/literature-review-contrastive-domain-embeddings.md`](../research/literature-review-contrastive-domain-embeddings.md) — the full survey (24 papers, extraction tables, risk register). Everything here is distilled from that; every factual claim is linked to a primary source.

> **Note on the branch:** this session is fixed to `arena/01a0aae7-embeded`, so the new material lives in this **folder** rather than on a separate branch. If you want it on its own branch later, you can `git checkout -b` from this commit.

---

## Read these in this order

| # | File | What it answers | Your spec §|
|---|---|---|---|
| 01 | [`01-conceptual-primer-and-math.md`](01-conceptual-primer-and-math.md) | What every box in the conceptual chain does, and the only maths you need | §3, §14 |
| 02 | [`02-solved-vs-open-gaps.md`](02-solved-vs-open-gaps.md) | What is already standard, and what is genuinely still open | §4, §5 |
| 03 | [`03-feasibility-matrix-and-paths.md`](03-feasibility-matrix-and-paths.md) | Evidence-based comparison of 5 candidate domains + full paths | §6, §7 |
| 04 | [`04-project-specification.md`](04-project-specification.md) | The single chosen project: RQ, hypothesis, data, model, loss, baselines, metrics | §8, §9, §15 |
| 05 | [`05-clean-experiment-and-scope.md`](05-clean-experiment-and-scope.md) | Leakage, splits, seeds, and MUST / SHOULD / NICE experiments | §10, §11 |
| 06 | [`06-implementation-architecture.md`](06-implementation-architecture.md) | Folder layout, libraries, what each module does | §13 |
| 07 | [`07-four-week-plan.md`](07-four-week-plan.md) | Day-by-day-ish schedule with exit criteria | §12 |
| 08 | [`08-error-analysis.md`](08-error-analysis.md) | How to look at failures and what is worth writing about | §16 |
| 09 | [`09-literature-report-and-reading-order.md`](09-literature-report-and-reading-order.md) | 16 papers, what you need from each, 4-tier reading order (~4 days) | §17, §18 |
| 10 | [`10-do-not-do-list.md`](10-do-not-do-list.md) | 20 ways this project becomes weak | §19 |

---

## The one-paragraph answer

The most technically meaningful project you can finish in one month is **not** "prove that domain-specific contrastive embeddings work" — that is already established ([TSDAE](https://aclanthology.org/2021.findings-emnlp.59/), [GPL](https://aclanthology.org/2022.naacl-main.168/), [SPECTER](https://aclanthology.org/2020.acl-main.207/)), and in some domains it appears *not* to work ([Rosane et al., SBES 2025](https://doi.org/10.5753/sbes.2025.9809)). It is: **run the cleanest possible experiment on the one variable that decides whether it works — which negative examples you train against — in a domain where you can validate your harness against published numbers.** Concretely: duplicate-question retrieval on [CQADupStack](https://eltimster.github.io/www/pubs/adcs2015.pdf) (12 technical StackExchange subforums, human duplicate marks), a 33M-parameter encoder, MultipleNegativesRankingLoss, five negative-sampling strategies, and a measured false-negative rate for each. Full spec in [`04`](04-project-specification.md).

---

# FINAL BLUEPRINT (§20)

```
PROJECT
│
├── Research question
│     In duplicate-question retrieval, does the choice of NEGATIVE examples
│     determine whether contrastive fine-tuning of a small general-purpose
│     encoder beats (a) BM25 and (b) the same encoder used zero-shot?
│
├── Hypothesis (falsifiable)
│     H1: fine-tuning improves nDCG@10 over the zero-shot encoder, and the
│         size of that improvement is ordered by negative strategy, with
│         denoised hard negatives best.
│     H0 (the interesting alternative): all negative strategies are
│         statistically indistinguishable, because the mined "hard negatives"
│         are dominated by unlabelled duplicate questions — whose rate we
│         measure directly.
│     Both outcomes are interpretable. You do not need H1 to be true.
│
├── Dataset
│     CQADupStack (BEIR) — 12 StackExchange subforums, 13,145 queries,
│     457K documents, ~1.4 human-marked duplicates per query.
│     Loader: ir_datasets "beir/cqadupstack/<subforum>" or the BEIR repo.
│     Held-out subforum = the "unseen domain".
│
├── Preprocessing
│     Strip HTML/code blocks; title + body; truncate to 256 tokens;
│     build per-subforum document store; no paraphrase generation.
│
├── Positive pairs
│     (q, d+) where d+ is a question the community marked as a duplicate of q.
│     Natural, human-generated, never synthetic.
│
├── Negative pairs (the independent variable — 5 strategies)
│     N1 random          – uniform sample from the subforum
│     N2 in-batch        – other positives in the mini-batch (free)
│     N3 lexical-hard    – top-k BM25 matches not marked duplicate
│     N4 model-hard      – top-k by current-model embedding similarity
│     N5 denoised-hard   – N4 filtered by cross-encoder score / margin rule
│
├── Base model
│     sentence-transformers/all-MiniLM-L6-v2   (23M params, 384-d, Apache 2.0)
│     Secondary: BAAI/bge-small-en-v1.5        (33M, 384-d, MIT — needs the
│                                               "Represent this sentence..."
│                                               query prefix)
│
├── Contrastive objective
│     MultipleNegativesRankingLoss = InfoNCE with in-batch negatives
│     L = -log  exp(sim(a,p)/τ) / [ exp(sim(a,p)/τ) + Σ_i exp(sim(a,n_i)/τ) ]
│     Explained term-by-term in 01-conceptual-primer-and-math.md
│
├── Baselines (all on the identical test set, pool and code path)
│     B0 BM25 · B1 TF-IDF+cosine · B2 same encoder ZERO-SHOT
│     B3 bge-small-en-v1.5 zero-shot · B4 BM25 ∪ dense hybrid
│
├── Evaluation metrics
│     PRIMARY:   nDCG@10
│     SECONDARY: Recall@1/10/100, MRR@10, MAP
│     DIAGNOSTIC: estimated false-negative rate of each mined negative set;
│                 alignment/uniformity
│     3 seeds → mean ± std; paired bootstrap over queries
│
├── Ablation
│     A1 negative strategy (N1–N5)          [MUST]
│     A2 number of hard negatives 0/1/3/5   [MUST if time]
│     A3 data efficiency 1/5/25/100%        [SHOULD]
│     A4 cross-subforum (held-out domain)   [SHOULD]
│
├── Error analysis
│     False positives (lexical overlap, different question)
│     False negatives (semantically identical, not marked duplicate)
│     Buckets by query length, by lexical overlap with gold, by subforum
│
├── Week 1   Read 5 papers; load CQADupStack; reproduce a published BEIR
│            nDCG@10 number; EDA; build pairs; split
├── Week 2   BM25 + TF-IDF + both zero-shot encoders; complete baseline table
│            BEFORE any training. Pre-register RQ + metrics.
├── Week 3   Train N1–N5 × 3 seeds; measure false-negative rates
├── Week 4   Final eval, ablations, statistics, error analysis, write-up,
│            reproducibility doc
```

---

## What I would actually build

A Python package (`src/`) that: loads CQADupStack subforums through `ir_datasets`; cleans posts into `(id, subforum, title, body, timestamp, duplicate_ids)` records; builds a **duplicate-group graph**; splits queries so that no group straddles train/val/test and, for the domain-generalisation run, so that whole subforums are held out; generates `(anchor, positive, [negatives])` records under five negative strategies; fine-tunes `all-MiniLM-L6-v2` with `MultipleNegativesRankingLoss` from `sentence-transformers`; encodes the fixed document pool with FAISS; and scores every system — BM25, TF-IDF, zero-shot encoders, hybrids, and all five fine-tuned variants — on the **same** query set, the **same** pool, and the **same** metric code. Output: one main results table (mean ± std over 3 seeds), one ablation table, one error-analysis table, and ~4 plots.

## What makes it a research project

Not the model. Not the dataset. The **controlled comparison plus a measurement nobody reports in this setting**: for each negative-sampling strategy, *what fraction of the "negatives" you trained on are actually duplicate questions the community never marked?* That number explains your result either way, and it is cheap to estimate. The rest of the contribution is methodological honesty: matched baselines, identical evaluation, group-aware splits, three seeds, reported uncertainty.

## What is merely engineering

The data loader, the cleaning, the FAISS index, the training loop, the metric computation, the plots. All of it is glue built on `sentence-transformers`, `ir_datasets`, `rank_bm25`/`pyserini`, `faiss-cpu` and `scikit-learn`. It is necessary and it will take most of your time — but it is not the contribution. Write it once, test it, and do not mistake it for research.

## What could go wrong

1. Your harness is silently wrong → mitigated by reproducing a published BEIR number in week 1.
2. You skip BM25 and cannot tell whether you beat lexical matching → build all baselines in week 2.
3. A leaky split gives beautiful meaningless numbers → group-aware splitting, done before pair generation.
4. One seed, so you cannot separate signal from noise → 3 seeds, always.
5. Gains are ~0 → this is an expected and publishable outcome given [Rosane et al.](https://doi.org/10.5753/sbes.2025.9809) and [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607); the false-negative measurement is what makes it a result.
6. You run out of time on data engineering → CQADupStack was chosen precisely because it loads in one line.

## Minimum viable paper

If everything goes wrong and you have only weeks 1–2 complete, you still have a coherent paper:

> *"We reproduce duplicate-question retrieval on CQADupStack across 12 technical domains and report BM25, TF-IDF, and two zero-shot neural retrievers under a unified harness, with per-domain nDCG@10 and an analysis of the ceiling imposed by sparse duplicate labels."*

Add one trained configuration (any negative strategy) with 3 seeds and you have a full experimental paper. Everything else is upside.
