# 05 — Making the experiment scientifically clean, and keeping it small

---

## Part A — Threats to validity, and exactly how you handle each

### 1. Data leakage

**The specific leak in this project:** the supervision signal is a *graph* of duplicate links. If question A is in train and its duplicate B is in test, the model can memorise B's surface form and "retrieve" it. Worse, because duplicate groups are connected components, a chain A→B→C can leak two hops away.

**Fix:**
1. Build the duplicate graph → connected components → each component is a **duplicate group**.
2. Split **by group, not by query.** Assign an entire group to exactly one of train / validation / test.
3. **Verify:** after splitting, assert that the intersection of train-document IDs and test-gold IDs is empty. Put this assertion in your code and print it in your paper's appendix.

**Why it matters:** [Zhang et al. (2023)](https://doi.org/10.1145/3576042) showed that evaluation bias of exactly this kind — data age and tracker choice — changes which method wins, with large effect sizes.

### 2. Negative leakage (false negatives)

**The problem:** a "negative" is any document not in the anchor's group. But with only ~1.4 labelled duplicates per query, many true duplicates are unlabelled, so a substantial fraction of your negatives — especially the *hard* ones — are actually positives.

**Fixes, in increasing order of effort:**
1. **Measure it.** For each strategy, report the fraction of mined negatives that a cross-encoder scores above the anchor–positive pair, plus a hand-checked 100-sample. This is cheap and it is your most original number.
2. **Filter it (strategy N5).** Drop candidates with `sim(a,c) > sim(a,p) − δ`, or use `mine_hard_negatives`' `absolute_margin` / `relative_margin` / `max_score` / `range_min`.
3. **Do not train on the top-1 most similar candidate at all** (`range_min=5`) — the very nearest neighbours are the most likely unlabelled duplicates.

**Theory:** [Robinson et al. (2021)](https://openreview.net/forum?id=S4nZh4WBHxq) prove (empirically) that hard sampling *without* debiasing is worse than with it; [RocketQA (2021)](https://aclanthology.org/2021.naacl-main.466/) shows the same in retrieval.

### 3. Train / validation / test separation

| Split | Role | Rule |
|---|---|---|
| **Train** | Pair generation + gradient updates | Groups assigned to train |
| **Validation** | Early stopping, checkpoint choice, LR selection | Groups assigned to val. **Never** touch test during development |
| **Test** | Reported numbers only | Groups assigned to test. Touches exactly once, at the end |

* Default: 70 / 10 / 20 by query count, after group assignment.
* **For the domain-generalisation run:** hold out an entire **subforum** (e.g. `tex`), train on the other 11, and evaluate on the held-out one. No group-based split needed there because the domains are disjoint by construction.
* **Drop or quarantine unanswerable queries:** a test query with no labelled duplicate in the pool cannot be retrieved by anyone. Report both with and without them; state which is primary.

### 4. Baseline fairness

Checklist — every item is a way of accidentally winning:

* [ ] Same document pool, same pool size, for every system ([Reimers & Gurevych 2021](https://aclanthology.org/2021.acl-short.77/) prove dense retrieval degrades faster with index size, so this is not cosmetic).
* [ ] Same query set and same qrels file.
* [ ] Same metric implementation (use `pytrec_eval` / `ranx` / BEIR's evaluator — do not hand-roll nDCG).
* [ ] Same tokenisation and truncation (256 tokens for everyone).
* [ ] Model-specific requirements honoured — e.g. BGE's query prefix. Forgetting it makes your baseline artificially bad, which is the #1 unfair-comparison pattern in [Musgrave et al. (2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf).
* [ ] No hyperparameter chosen by looking at test results. Tune on validation only.

### 5. Randomness

* **3 fixed seeds: 13, 42, 1337.** Report mean ± std.
* Seed everything: `torch`, `numpy`, `random`, and the data-loader shuffling.
* **Paired bootstrap over queries** (10,000 resamples) for headline comparisons.
* **Why this is not optional:** [Musgrave et al. (2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf) re-ran the metric-learning literature under a fair protocol and found that **most methods tie** once hyperparameters are tuned properly. Without seeds you cannot distinguish your contribution from noise.

### 6. Ablations — one factor at a time

| Change | Hold fixed |
|---|---|
| Negative strategy | model, data size, epochs, LR, batch, pool, split |
| Number of hard negatives | best strategy from A1, everything else |
| Data efficiency | best strategy and everything else |

Never change two things in the same run — then you cannot attribute the difference.

### 7. Index-size and metric-geekery hygiene

* Report the pool size in every table caption.
* Use `flat` (exact) FAISS index — with ≤90K documents per subforum, approximate search is unnecessary and only adds a confound.
* Normalise embeddings consistently; use inner product on normalised vectors (identical to cosine).

---

## Part B — Scope: MUST / SHOULD / NICE

### MUST HAVE — without these there is no valid project

| ID | Item | Effort |
|---|---|---|
| M1 | Load CQADupStack (≥3 subforums), clean, build the duplicate-group graph | 1 day |
| M2 | **Reproduce a published CQADupStack nDCG@10 with BM25** and confirm it is in the published range | 0.5 day |
| M3 | Baselines **B0 BM25**, **B1 TF-IDF**, **B2 MiniLM zero-shot** — all on the identical harness | 1.5 days |
| M4 | Group-constrained train/val/test split + a printed assertion that there is no ID overlap | 0.5 day |
| M5 | Positive-pair generation from duplicate groups (train only) | 0.5 day |
| M6 | **A1: five negative strategies × 3 seeds** | 3 days |
| M7 | Primary metric **nDCG@10** + Recall@1/10/100, MRR@10, MAP | 0.5 day |
| M8 | **Estimated false-negative rate** for each mined strategy | 1 day |
| M9 | Results table with mean ± std; one paragraph of honest interpretation | 0.5 day |

**Total ≈ 9 working days of the 20 available.** Everything below is upside.

### SHOULD HAVE — significantly strengthens the paper

| ID | Item | Effort |
|---|---|---|
| S1 | **A2: number of hard negatives (0/1/3/5)** — tests [DPR](https://arxiv.org/abs/2004.04906)'s "one is enough" finding | 1 day |
| S2 | **A4: held-out subforum** (train on 11, test on 1) — the domain-generalisation result | 1 day |
| S3 | **B3** `bge-small-en-v1.5` zero-shot baseline (with the query prefix!) | 0.5 day |
| S4 | **B4** BM25 ∪ dense hybrid | 0.5 day |
| S5 | Learning-rate sweep {1e-5, 2e-5, 5e-5} on one strategy — [Rosane et al.](https://doi.org/10.5753/sbes.2025.9809) showed LR sensitivity is severe here | 0.5 day |
| S6 | Alignment / uniformity diagnostics | 0.5 day |
| S7 | Ceiling analysis: hand-judge 100 top-1 failures | 0.5 day |
| S8 | Paired bootstrap confidence intervals | 0.5 day |

### NICE TO HAVE — only if everything above is done and stable

| ID | Item |
|---|---|
| N1 | **A3: data efficiency** (1 / 5 / 25 / 100% of pairs) — four short runs, very cheap once the pipeline works |
| N2 | Triplet-loss variant as an alternative objective |
| N3 | 128 vs 512 token truncation |
| N4 | Cross-encoder re-ranker on top-50, as an upper bound |
| N5 | A second dataset (`sentence-transformers/quora-duplicates`) as an external check |
| N6 | t-SNE / UMAP plot of the embedding space before and after fine-tuning |
| N7 | All 12 subforums instead of a subset |

### EXPLICITLY OUT OF SCOPE

* Training any encoder from scratch.
* Any model above ~150M parameters.
* Multi-GPU, distributed, or cross-batch negatives.
* Reproducing [ANCE](https://openreview.net/forum?id=zeFrfgyZln) or [RocketQA](https://aclanthology.org/2021.naacl-main.466/) at MS MARCO scale.
* A new loss function.
* Anything requiring paid API access.

---

## Part C — The minimum viable paper

If you finish only the MUST list, you have this paper:

> **Title:** *Which Negatives? The Effect of Negative-Sampling Strategy on Contrastive Fine-Tuning for Duplicate Question Retrieval*
> **Content:** a validated retrieval harness on CQADupStack; BM25 / TF-IDF / zero-shot-encoder baselines; five contrastive fine-tuning configurations differing only in negative-selection strategy, 3 seeds each; nDCG@10 with mean ± std and paired bootstrap; a measured false-negative rate per strategy; a per-subforum breakdown; and an honest interpretation under the pre-registered H1/H0.
> **Length:** 6–8 pages, 4 tables, 3 figures.

That is a coherent undergraduate experimental paper. Add S1 and S2 and it is a *good* one.
