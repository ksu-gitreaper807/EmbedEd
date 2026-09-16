# Scope — Version B (the middle version)

Read this before writing code, and re-read it in week 3 when you are tempted to add a fourth
condition.

---

## Part 1 — The three versions

| | Version A | **Version B (this)** | Version C |
|---|---|---|---|
| **Where** | described in Part 2 below; full text in git history at commit `633db02` | this folder | [`distilled-project/`](../../distilled-project) |
| **Research question** | Does in-domain contrastive fine-tuning help? | **Does negative hardness decide whether it helps?** | Which negative strategy, out of five, decides it? |
| **Independent variable** | none (a before/after comparison) | **negative hardness, 3 levels** | negative strategy, 5 levels + denoising |
| **Dataset** | AskUbuntu, ~15K corpus | AskUbuntu, ~15K corpus | CQADupStack, 457K docs, 12 subforums |
| **Losses** | 1 (`MNRL`) | 1 (`MNRL`) | 1 + explicit triplets |
| **Encoders** | 1 | 1 (+1 optional, winning condition only) | 2 |
| **Metrics** | 2 | 3 | 6 |
| **Uncertainty** | mean ± range, 3 seeds | **+ paired bootstrap on 3 contrasts** | paired bootstrap, 10k resamples |
| **Mechanism** | none | **measured false-negative rate** | measured FN rate per strategy |
| **Runs** | 3–9 | **14 (+3 optional)** | 15+ |
| **Source files** | 4 (~500 lines) | **6 (~650 lines)** | ~15 modules |
| **Effort** | 10–12 days | **4 weeks** | 8–12 weeks |

---

## Part 2 — Why Version A was too small (and what to put back)

Version A was correct, finishable, and **not quite a research project**. Its question — "does
fine-tuning help?" — is a reproduction: [TSDAE](https://aclanthology.org/2021.findings-emnlp.59/)
and [GPL](https://aclanthology.org/2022.naacl-main.168/) already establish that domain
adaptation helps in general, and
[Rosane et al.](https://doi.org/10.5753/sbes.2025.9809) suggest it barely helps on duplicate
detection. The most likely outcome was a null result with nothing to explain it.

The fix is not more data, more metrics, or more models. **It is one manipulated variable.**

| Version A decision | Version B decision | Why |
|---|---|---|
| No independent variable | **Negative hardness: N1 random / N2 lexical / N3 model-hard** | Turns a reproduction into an experiment. It is also the variable the literature singles out: [AugSBERT](https://aclanthology.org/2021.naacl-main.28/)'s sampling ablation (Random vs BM25 vs Semantic Search vs KDE) is the direct precedent, and [Robinson et al.](https://openreview.net/forum?id=CR1XOQ0UTh-) gives a testable prediction (help, then hurt). |
| 2 metrics (Recall@10, MRR@10) | **+ nDCG@10** | With ~1–2 golds per query, Recall@10 and MRR@10 both reduce to "did we hit one early". nDCG@10 rewards finding *all* duplicates high up, and is the metric the adaptation literature reports. ~10 extra lines. |
| mean ± range over 3 seeds | **+ paired bootstrap over 400 queries, on 3 pre-declared contrasts** | With three conditions you are now making *comparisons*, not just observations, and 200-query differences need an interval. ~20 lines, runs in seconds. This was the right thing to add and the wrong thing to add first. |
| No mechanism measurement | **Manual false-negative rate** (50 queries × top-5 × 2 systems) | One afternoon of human labelling that converts "we found nothing" into "we found nothing, and here is the measured 95%-incompleteness that explains it". The highest value-per-hour item in the project. |
| 4 source files | **6** (+ `negatives.py`, `analyze.py`) | Each has a distinct job. `negatives.py` is the only file with anything new in it. |
| 3–9 runs | **14 runs** | Still ~2 hours of GPU in total, because the corpus is small. This is the dividend from staying on AskUbuntu. |
| AskUbuntu, ~15K corpus | **unchanged** | The dataset was never the problem. Keeping it small is what makes mining a matmul instead of an indexing project. |
| One model | **unchanged** (second encoder is optional, winning condition only) | A second encoder doubles runs without strengthening the causal claim. |
| One loss | **unchanged** | Hardness is manipulated by batch composition, so the objective is byte-identical across conditions — a *cleaner* design than comparing losses. |
| BM25 baseline, zero-shot control | **unchanged** | Non-negotiable in all three versions. |

### What stays removed from Version C

These were cut when going from the big plan to Version A, and they **stay cut**:

* CQADupStack (457K docs, 12 subforums, raw HTML, ~1 week of data engineering).
* Five negative strategies → three. Denoising (cross-encoder filtering) is out.
* FAISS → the corpus is ~15K; mining and retrieval are both single matmuls.
* Cross-domain evaluation over 12 subforums.
* TF-IDF as a second lexical baseline.
* Second encoder in the core comparison.
* MAP, Recall@1/100, alignment/uniformity, Spearman.
* Significance tests, p-values, multiple-comparison corrections.
* Training from scratch, custom architectures, distributed training, paid APIs.
* Synthetic/LLM-generated positive pairs.

---

## Part 3 — In scope (MUST)

1. **Dataset:** `sentence-transformers/askubuntu`, one-line load.
2. **Corpus and qrels:** ~15,000 docs from `query` ∪ `positive`; gold = the `positive` lists.
3. **Splits:** train 12,745 / dev 200 / test 200, fixed, never re-derived.
4. **Leakage check:** no train anchor or positive among the 400 eval queries.
5. **Self-exclusion:** no eval query may retrieve itself.
6. **Duplicate-group filter:** no neighbourhood may contain a known duplicate of the anchor.
7. **Training pairs:** `(anchor, positive)` from train rows only, ~17,000–20,000 pairs.
8. **Model:** `all-MiniLM-L6-v2`.
9. **Loss:** `MultipleNegativesRankingLoss`, identical across conditions.
10. **Three negative conditions:** N1 (`hard_fraction=0.0`), N2 (BM25 neighbourhoods),
    N3 (zero-shot-embedding neighbourhoods), `k=64`, batch size 32 fixed.
11. **Hardness manipulation check:** measured in-batch hardness must satisfy N1 < N2 < N3.
12. **Baselines:** BM25 and zero-shot MiniLM.
13. **Three metrics** × 2 splits × 5 systems.
14. **3 seeds** per fine-tuned condition (9 runs) + epoch selection on dev (3 runs) +
    `hard_fraction` ablation (2 runs).
15. **Paired bootstrap** on three contrasts: N1−B2, N2−N1, N3−N2.
16. **False-negative measurement:** 50 queries × top-5 × 2 systems, manually labelled.
17. **Error analysis:** 30 worst queries, five labels.
18. **Deliverables:** 4 tables, 3 figures, 8–10 page report, `run_all.sh`.

---

## Part 4 — SHOULD HAVE

Only after all of Part 3 is done. In priority order.

1. **`hard_fraction` sweep {0.25, 0.5, 1.0}** on N3 — traces the hardness curve continuously.
2. **A second encoder** (`BAAI/bge-small-en-v1.5`) on the **winning condition only** (3 runs).
   **You must prepend its query prefix** — `"Represent this sentence for searching relevant
   passages: "` — or you will produce a fake baseline.
3. **Refresh N3's neighbourhoods once after epoch 1** (a lite
   [ANCE](https://openreview.net/forum?id=zeFrfgyZln) refresh): re-encode the training anchors
   with the partially-trained model and re-mine. Costs one extra matmul and one reconstruction of
   the sampler; risks the plumbing, so it is a SHOULD, not a MUST.
4. **`NO_DUPLICATES`-style filtering** at the batch level (no two pairs from one duplicate group
   in the same batch).
5. **Gold-split breakdown:** Recall@10 for gold duplicates in the train split vs. the eval split —
   tells you whether you measured retrieval or memorisation.
6. **Cross-check with `InformationRetrievalEvaluator`** — an independent implementation of your
   metrics.
7. **The `useb` package's official AskUbuntu eval** as external validation of your harness.

---

## Part 5 — Optional extensions

Nothing here may start before the main table exists.

* **Denoising** the N3 neighbourhoods (drop candidates whose similarity exceeds
  `sim(a, p) − δ`), [RocketQA](https://aclanthology.org/2021.naacl-main.466/)-style. This is the
  natural next experiment if you observe pattern 2 — but it needs a threshold choice and a
  second mechanism measurement, so it is a week by itself.
* **A second domain** (CQADupStack, or `sentence-transformers/quora-duplicates` `pair`) to test
  whether the hardness curve is domain-specific.
* **Symmetric training pairs** (add `(p, a)`).
* **More seeds** (5), or bootstrapping over seeds as well as queries.
* **A cross-encoder reranker** on the top-10.
* **TSDAE-style domain-adaptive pretraining** before the contrastive stage.
* **Enlarging the corpus** with the 168K unlabelled AskUbuntu questions in
  `sentence-transformers/askubuntu-questions` as distractors — makes retrieval harder and more
  realistic, but every unlabelled distractor is a potential false negative, so it changes what
  your metrics mean.

---

## Part 6 — Out of scope

Not "later". Not at all, in this project.

**Data:** scraping or building your own dataset; CQADupStack or any corpus above ~100K documents;
synthetic or LLM-generated positives; any dataset without pre-defined splits.

**Models and training:** training from scratch; any encoder above ~150M parameters; a second
encoder in the core comparison; custom architectures, attention, or pooling; multi-GPU or
distributed training; hyperparameter search over lr, warmup, schedulers, or weight decay;
training an LLM; paid APIs.

**Losses:** triplet loss, margin losses, SimCSE, supervised contrastive — **as comparisons**. One
loss, always.

**Evaluation:** FAISS or any approximate-index library; MAP, Recall@1/100, alignment/uniformity,
Spearman; cross-domain or zero-shot transfer experiments; significance tests, p-values,
multiple-comparison correction.

**Infrastructure:** Docker, experiment trackers, config frameworks, CI.

**Claims:** "domain-specific embeddings work" as a novelty claim (it is an established
technique — you are measuring it, not inventing it); any claim that generalises beyond AskUbuntu
and MiniLM; any claim that you have isolated hardness from label contamination (you have not —
see threat 2 in [`PROJECT_SPEC.md`](PROJECT_SPEC.md)).

---

## Part 7 — Stop condition

> **The project is finished when BM25, zero-shot MiniLM, and all three fine-tuned conditions
> (N1/N2/N3) have been evaluated on the fixed test split with Recall@10, MRR@10 and nDCG@10
> over 3 seeds, with bootstrap intervals on the three contrasts — and those numbers are in one
> table.**

At that point, stop running experiments and start writing.

**Anti-scope-creep rules, in the order you will be tempted to break them:**

1. If N2 and N3 come out identical, do **not** add a fourth condition to find a difference.
   "Hardness beyond lexical similarity changes nothing here" is a finding.
2. If N3 is worse than N2, do **not** lower the learning rate to rescue it. That is pattern 2 —
   the predicted result. Measure the false-negative rate and write it up.
3. If everything beats BM25 by 20 points, do **not** celebrate. Go to checkpoint 4: you have
   almost certainly failed to exclude the query's own document from the ranking.
4. If checkpoint 11 fails (the three conditions produce equally hard batches), **stop everything
   and fix it.** An experiment whose manipulation did not take effect produces no evidence at
   all, and no amount of extra runs will change that.
5. If you finish early, do **not** start an optional extension. Improve the report. A clear
   9-page write-up beats 14 pages with a half-finished denoising experiment.
6. If you are behind, cut from the top of [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §9.
   Never cut the leakage check, the self-exclusion check, the duplicate-group filter, the
   hardness manipulation check, BM25, or the zero-shot control.

**Version A remains the fallback at any point.** If `negatives.py` is not working by day 14,
drop to N1 only, drop nDCG@10, drop the bootstrap, and ship Version A. A complete project
without an independent variable beats an incomplete project with one.
