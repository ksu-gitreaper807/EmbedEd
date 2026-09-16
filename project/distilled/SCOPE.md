# Scope

The most important document in this folder. It exists so that in week 3, when you are tempted
to add a fourth system, you have a written reason not to.

---

## Part 1 — Aggressive critique of the previous specification

The old plan ([`distilled-project/`](../../distilled-project)) was a good research design. That
is exactly the problem: it was designed to produce a finding, not to be finished in four weeks
by one undergraduate. Every row below asks a single question: *does removing this change the
answer to the research question?* Usually it does not.

"Replace" means the component goes away and something simpler takes its job.

| Current component | Keep / Simplify / Replace / Remove | Why |
|---|---|---|
| **Dataset** (CQADupStack) | **Replace → AskUbuntu** | 457K documents, 12 subforums, raw HTML StackExchange posts that need cleaning, no per-split published numbers for a split you build yourself. AskUbuntu is the *same task* (duplicate question retrieval, human duplicate marks) at 3% of the size, loads in one line, arrives pre-tokenised, and ships fixed 200/200 dev/test splits. The 12-subforum cross-domain experiment was the only real casualty, and it was a second research question. |
| **Base model** (MiniLM + bge-small) | **Simplify → MiniLM only** | One model means one independent variable. `bge-small` also requires a query prefix (`"Represent this sentence for searching relevant passages: "`), and forgetting it produces a fake baseline — a pitfall you gain nothing from walking into. |
| **Contrastive loss** (MNRL + triplet comparison) | **Simplify → MNRL only** | `MultipleNegativesRankingLoss` is one line, is the standard for `(anchor, positive)` data, and supplies negatives for free from the batch. Comparing losses is a different, larger project. |
| **Positive pairs** (connected components over the duplicate graph) | **Simplify → use the provided `positive` lists as-is** | A union-find closure over duplicate groups is ~15 lines and buys slightly more complete positives. It is not needed: the dataset already ships 1–3 duplicates per query. Skip it and spend the time on the debugging checkpoints. |
| **Negative pairs** (five strategies) | **Simplify → in-batch negatives only** | Five strategies means five training runs *per seed*, a mining pipeline, and a false-negative measurement loop for each one. The research question — "does fine-tuning help?" — needs exactly one negative strategy. |
| **Hard negatives** (FAISS mining + cross-encoder denoising) | **Remove from core → optional extension** | The single highest-cost, lowest-necessity item in the old plan. It requires a mining script, an index, and a denoising filter, and it only makes sense *after* you know whether plain fine-tuning does anything. |
| **BM25** | **Keep (simplified: default parameters, ~15 lines)** | The one component that survives intact, because it is the literature's own control. [BEIR](https://arxiv.org/abs/2104.08663): "BM25 is a robust baseline"; [Jiang et al.](https://doi.org/10.1016/j.jss.2023.111607): deep models do not reliably beat IR. Without BM25, "my model beat random" is the strongest claim available to you. |
| **TF-IDF** | **Remove** | Strictly weaker than BM25 and equally easy. A second lexical baseline adds a table row and no information. (If `rank_bm25` will not install, `TfidfVectorizer` + cosine is the 5-line fallback — that is the only reason to write it.) |
| **FAISS** | **Remove** | The corpus is ~15,000 × 384 floats ≈ 23 MB. One `numpy` matmul (`corpus @ query`) ranks everything in well under a second. FAISS is an index for corpora you do not have. |
| **Multiple seeds** (3) | **Keep, simplified** | Runs take minutes, so 3 seeds cost ~30 minutes and turn "one number" into "a number with a visible spread". 1 seed is not defensible; 5 seeds is not informative. |
| **Cross-domain evaluation** (12 subforums) | **Remove** | 12 domains × 3 systems × 3 seeds is 100+ evaluation runs and a second research question ("does it generalise?"). You have not yet answered the first one. |
| **Ablations** (several) | **Simplify to exactly one** | One ablation — batch size 16 / 32 / 64, i.e. the number of in-batch negatives — is a one-argument change that directly demonstrates the mechanism the method depends on. Everything else is a second project. |
| **Error analysis** (extensive) | **Keep, simplified** | 20 worst queries, five labels, one table, one paragraph. This is where the report earns its grade and it costs an afternoon, not a week. Do not cut it below 10 queries. |
| **Statistical testing** (paired bootstrap, CIs) | **Remove bootstrap → report mean ± range over 3 seeds** | A bootstrap needs a resampling loop, a choice of resampling unit, and an interpretation section — it turns an ML project into a statistics project. Report the observed spread and state the resolution limit ("with 200 queries, under ~3 points is noise"). |

**Net effect:** 5 negative strategies → 0 extra; 457K documents → 15K; 12 domains → 1;
2 encoders → 1; 5 metrics → 2; FAISS → a matmul; bootstrap → a range. The research question
survives intact.

---

## Part 2 — In scope (MUST)

These nine things are the project. If one of them is missing, the project is not complete.

1. **Dataset:** `sentence-transformers/askubuntu`, loaded in one line.
2. **Corpus and qrels:** ~15,000 documents built from `query` ∪ `positive`; gold = the
   `positive` lists, mapped to corpus ids.
3. **Splits:** train 12,745 / dev 200 / test 200, fixed, never re-derived.
4. **Leakage check:** no train anchor or train positive appears among the 400 eval queries.
5. **Self-exclusion:** no eval query can retrieve itself.
6. **Training pairs:** `(anchor, positive)` from train rows only, ~17,000–20,000 pairs.
7. **Model:** `all-MiniLM-L6-v2`, one encoder.
8. **Loss:** `MultipleNegativesRankingLoss`, in-batch negatives, library implementation.
9. **Three systems × two metrics × two splits:**
   BM25 / zero-shot MiniLM / fine-tuned MiniLM × Recall@10 and MRR@10 × dev and test.
   Plus **3 seeds** and the ten debugging checkpoints.

Plus the deliverables: three tables, two figures, a 5–7 page report, and a `run_all.sh`.

---

## Part 3 — SHOULD HAVE (only after MUST is complete)

In priority order. Stop at any point; the project is still complete.

1. **The batch-size ablation** (16 / 32 / 64) — demonstrates what negatives do.
2. **The error analysis** — 20 failures, five labels, one table.
3. **`BatchSamplers.NO_DUPLICATES`** — prevents two pairs from one duplicate group sharing a batch.
4. **The gold-split breakdown** — Recall@10 separately for gold duplicates in the train split vs
   the eval split. Five lines; tells you whether you measured retrieval or memorisation.
5. **Cross-check with `InformationRetrievalEvaluator`** — an independent implementation of your metric.
6. **The `useb` package's official AskUbuntu eval** as an external sanity check on your harness.

---

## Part 4 — Optional extensions (only after everything above)

Interesting, explicitly **not required**, and none of them may start before the three-system
table exists.

* **Hard-negative mining.** Use the dataset's shipped `negative` lists, or mine with BM25, as
  explicit negatives. This needs a second loss (`TripletLoss`) or a custom batch sampler. That
  is precisely why it is optional: it breaks the one-loss rule.
* **A second encoder** (`bge-small-en-v1.5`) to test whether the effect is model-dependent.
  Remember the query prefix.
* **A second dataset** (CQADupStack, or the `sentence-transformers/quora-duplicates` `pair`
  subset) to test whether the effect is domain-dependent.
* **Symmetric training pairs** — add `(p, a)` alongside `(a, p)`.
* **More seeds** (5 instead of 3), or a paired bootstrap over queries.
* **TSDAE-style domain-adaptive pretraining** before the contrastive stage.
* **A cross-encoder reranker** on top of the top-10.

---

## Part 5 — Out of scope

You must **not** implement any of these. Not "later" — not at all, in this project.

**Data**
* Crawling, scraping, or building your own dataset.
* CQADupStack, or any corpus above ~100K documents.
* Synthetic / LLM-generated positive pairs. Positives must be naturally occurring.
* Any dataset without a pre-defined train/test split.

**Models and training**
* Training any encoder from scratch.
* Any encoder above ~150M parameters.
* A second encoder in the core comparison (that is Part 4).
* Custom Transformer architectures, custom attention, new pooling strategies.
* Multi-GPU, distributed training, gradient accumulation gymnastics.
* Training an LLM, or calling a paid API for anything.
* Hyperparameter search over learning rate, schedulers, warmup, or weight decay.

**Losses and negatives**
* Triplet loss, margin losses, SimCSE, supervised contrastive loss — as *comparisons*. One loss.
* Five negative strategies. Hard-negative mining in the core.
* Cross-encoder denoising of negatives.

**Evaluation and infrastructure**
* FAISS or any approximate-index library.
* nDCG, MAP, Recall@1/100, alignment/uniformity, Spearman correlation.
* Cross-domain or zero-shot transfer experiments.
* Bootstrap confidence intervals, significance tests, p-values.
* Docker, experiment trackers, config frameworks, CI pipelines.

**Claims**
* "Domain-specific embeddings work" as a novelty claim. It is an established technique
  ([TSDAE](https://aclanthology.org/2021.findings-emnlp.59/),
  [GPL](https://aclanthology.org/2022.naacl-main.168/)); you are measuring it, not inventing it.
* Any claim that generalises beyond AskUbuntu.

---

## Part 6 — Three versions

Not ranked by scientific quality — they answer the same question. Ranked by whether you
finish.

### Version A — Minimum viable project

*Same* dataset, model, loss, metrics — the design does not change, only the amount of it.

| | |
|---|---|
| Runs | 1 seed, 2 epochs fixed (no dev-based selection) |
| Systems | BM25, zero-shot MiniLM, fine-tuned MiniLM |
| Eval | test split only |
| Ablation | none |
| Error analysis | 5 queries, one paragraph |
| Report | 3–4 pages |
| **Effort** | ~10–12 days of work |
| **Risk** | Very low. Almost nothing can go wrong that the debugging checkpoints do not catch. |
| **Learning value** | All seven goals: embeddings, similarity, contrastive learning, fine-tuning, pairs, evaluation, methodology. |
| **Research depth** | Low. One number, no error bars. |
| **Use it when** | you lose a week to coursework, or the fine-tuning refuses to work. |

### Version B — Recommended

| | |
|---|---|
| Runs | 3 seeds; epochs ∈ {1,2,3} selected on dev |
| Systems | BM25, zero-shot MiniLM, fine-tuned MiniLM |
| Eval | dev **and** test |
| Ablation | batch size 16 / 32 / 64 |
| Error analysis | 20 queries, five labels, one table |
| Report | 5–7 pages |
| **Effort** | ~3 weeks of work, ~1 week writing, with slack |
| **Risk** | Low–medium. The failure mode is a null result, which is a valid outcome. |
| **Learning value** | All seven goals, plus seed variance, plus model selection on a held-out split, plus reading your own failures. |
| **Research depth** | Right-sized: one claim, one control, one mechanism check, one honest discussion of why the claim might be wrong. |
| **Use it when** | you have a normal four weeks. **This is the target.** |

### Version C — Stretch (the previous specification)

| | |
|---|---|
| Runs | 3 seeds × 5 negative strategies = 15 training runs |
| Systems | BM25, TF-IDF, MiniLM, bge-small, hybrid, × 5 negative strategies |
| Eval | 12 subforums, nDCG@10 + Recall@1/10/100 + MRR + MAP + alignment/uniformity |
| Ablation | several |
| Error analysis | extensive, with false-negative rate per strategy |
| Report | 10–15 pages |
| **Effort** | **8–12 weeks**, not 4 |
| **Risk** | High. Fifteen runs on a 457K-document corpus with a mining pipeline is how four-week projects become unfinished GitHub repos. |
| **Learning value** | Higher ceiling, but only if you finish. Unfinished, it teaches you less than Version A. |
| **Research depth** | Genuinely publishable-adjacent. Also genuinely out of reach. |
| **Use it when** | you have already finished Version B and have a summer. |

### Why Version B is the target

| | A | **B** | C |
|---|---|---|---|
| Implementation effort | Low | **Medium** | Very high |
| Learning value | Good | **Good + seed variance + model selection + failure reading** | Highest, but only if completed |
| Research depth | Low | **Adequate for one defensible claim** | High |
| Risk of not finishing | Very low | **Low–medium** | High |
| Expected completion time | 10–12 days | **3.5 weeks** | 8–12 weeks |

Version A leaves the most valuable lesson on the table — *that a single number from a single run
is not a result*. Three seeds and a held-out split are what teach you that, and they cost half a
day. Version C buys depth you cannot reach in four weeks, and the usual outcome of aiming at it
is a half-built pipeline and no result at all.

Version B is the smallest version that still contains a real control, a real error bar, and a
real acknowledgement of what might be wrong. That is the threshold at which the thing stops
being an exercise and becomes a project.

---

## Part 7 — Stop condition

> **The project is finished when BM25, zero-shot MiniLM, and one contrastively fine-tuned
> MiniLM have each been evaluated on the fixed test split with Recall@10 and MRR@10, and those
> numbers are in one table with the spread across 3 seeds.**

At that moment: stop running experiments and start writing.

**Anti-scope-creep rules, in the order you will be tempted to break them:**

1. If the effect is small, do **not** add a system to chase a bigger one. Small is the finding.
2. If the effect is absent, do **not** change the learning rate to make it appear. Report the
   null result and explain it with the label-incompleteness evidence.
3. If the effect is huge, do **not** celebrate. Go back to debugging checkpoints 4 and 5 —
   you have almost certainly failed to exclude the query's own document from the ranking.
4. If you finish early, do **not** start an optional extension. Improve the report. A clear
   6-page write-up beats a 12-page one with a half-finished ablation.
5. If you are behind, cut from the bottom of the [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
   §9 list. Never cut the leakage check, the self-exclusion check, BM25, or the zero-shot control.
